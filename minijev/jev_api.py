"""A client for the Jev Decisions API (TypeSafe Jev-1.13 via defapi.org).

Why this exists
---------------
MiniJev's cost model is *declared*: `RewardWeights.cost` and the routing ladder use
hand-chosen numbers -- local 0.001, rag 0.01, cloud 0.10, review 2.0 in the declared world --
because no measured cost was available. The Jev API returns `usage.cost` for every call, so
routing through it replaces an assumption with a measurement. That is the difference between a
reward model that can be argued with and one that can be checked.

The contract, from the published OpenAPI document
-------------------------------------------------
Three question types, and MiniJev currently implements only the second:

* `noul`  -- a boolean question answered with a probability in [0, 1]. The probability *is* the
             answer; there is no separate confidence, because for a Bernoulli there is nothing
             left to summarise.
* `choice` -- one option out of a `criteria` map (up to 255), returned with the full
             `probabilities` distribution and a `confidence`.
* `score` -- an ordered `criteria` list (2-10 levels), returned as a probability-weighted
             `score`, a `legend`, the distribution, and a confidence.

One request carries a `state` (a string, or a JSON object/array for structured context) and a
*map* of questions, so several typed questions are answered about the same state in one call.

What is deliberately not done here
----------------------------------
`confidence` is passed through verbatim and never reconstructed. The published examples do not
pin its semantics -- the `choice` and `score` examples carry placeholder task ids
(`taxxxxxxxx-...`), so their confidence values are illustrative -- and no simple statistic of the
distribution fits both (max probability gives 0.84 against a reported 0.75; normalized entropy
gives 0.60; a rescaled top probability gives 0.76). Guessing a formula and shipping it would
manufacture agreement with a contract we cannot yet read.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .schema import ROUTE_ORDER, RouteScores

DEFAULT_ENDPOINT = "https://api.defapi.org/api/v1/decisions"
"""Verified by probe: an unauthenticated POST to this path returns 401, so it exists and needs a
key. The defapi.org product page advertises `POST /v1/systemone` instead; that path returns 404,
as do `/v1/decisions` and `/api/v1/systemone`. Trust the OpenAPI document, not the marketing copy.
"""

DEFAULT_MODEL = "typesafe/jev-1.13"
API_KEY_ENV = "JEV_API_KEY"

QUESTION_TYPES: tuple[str, ...] = ("noul", "choice", "score")
"""The three primitives the API accepts. MiniJev implements only `choice` today."""

SCORE_LEVEL_BOUNDS = (2, 10)
"""`score` criteria must hold between 2 and 10 levels."""

CHOICE_OPTION_LIMIT = 255


class JevApiError(RuntimeError):
    """A call that could not be completed, carrying the status when there was one."""

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(frozen=True)
class Question:
    """One typed question. `criteria` shape depends on `type`."""

    type: str
    instructions: str
    criteria: Any = None

    def __post_init__(self) -> None:
        if self.type not in QUESTION_TYPES:
            raise ValueError(f"unknown question type {self.type!r}; expected one of {QUESTION_TYPES}")
        if not self.instructions.strip():
            raise ValueError("instructions must not be empty")
        if self.type == "choice":
            if not isinstance(self.criteria, Mapping) or not self.criteria:
                raise ValueError("a choice question needs a non-empty mapping of option -> description")
            if len(self.criteria) > CHOICE_OPTION_LIMIT:
                raise ValueError(f"choice supports at most {CHOICE_OPTION_LIMIT} options")
        if self.type == "score":
            if not isinstance(self.criteria, Sequence) or isinstance(self.criteria, str):
                raise ValueError("a score question needs an ordered sequence of level descriptions")
            low, high = SCORE_LEVEL_BOUNDS
            if not low <= len(self.criteria) <= high:
                raise ValueError(f"a score question needs between {low} and {high} levels, got {len(self.criteria)}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            payload["criteria"] = list(self.criteria) if self.type == "score" else dict(self.criteria)
        return payload


@dataclass
class DecisionResponse:
    """A parsed answer, keeping `usage.cost` because that is the point of calling out at all."""

    answers: dict[str, dict[str, Any]]
    usage: dict[str, Any] = field(default_factory=dict)
    task_id: str = ""
    model: str = ""
    consumed: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def cost(self) -> float | None:
        """The measured cost of this call, or `None` when the API did not report one."""
        value = self.usage.get("cost")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def input_tokens(self) -> int | None:
        value = self.usage.get("input_tokens")
        return int(value) if isinstance(value, (int, float)) else None

    def choice(self, key: str) -> str | None:
        return self.answers.get(key, {}).get("choice")

    def probabilities(self, key: str) -> dict[str, float]:
        raw = self.answers.get(key, {}).get("probabilities") or {}
        return {str(option): float(value) for option, value in raw.items()}

    def noul(self, key: str) -> float | None:
        value = self.answers.get(key, {}).get("noul")
        return float(value) if isinstance(value, (int, float)) else None

    def score(self, key: str) -> float | None:
        value = self.answers.get(key, {}).get("score")
        return float(value) if isinstance(value, (int, float)) else None

    def confidence(self, key: str) -> float | None:
        """Passed through verbatim; see the module docstring for why it is not recomputed."""
        value = self.answers.get(key, {}).get("confidence")
        return float(value) if isinstance(value, (int, float)) else None


class JevApiClient:
    """A minimal, dependency-free client.

    Uses `urllib` rather than a third-party HTTP library so the only requirement is the standard
    library -- the rest of MiniJev has no network dependencies and adding one for four fields of
    a POST would be a poor trade.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def build_payload(
        self,
        state: str | Mapping[str, Any] | Sequence[Any],
        questions: Mapping[str, Question],
        *,
        session_id: str | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        """The request body, assembled and validated before any network call is attempted."""
        if not questions:
            raise ValueError("at least one question is required")
        payload: dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": {key: question.to_dict() for key, question in questions.items()},
        }
        if session_id is not None:
            payload["session_id"] = session_id[:256]
        if user is not None:
            payload["user"] = user[:256]
        return payload

    def decide(
        self,
        state: str | Mapping[str, Any] | Sequence[Any],
        questions: Mapping[str, Question],
        *,
        session_id: str | None = None,
        user: str | None = None,
    ) -> DecisionResponse:
        """Send one decisions request and parse the response."""
        if not self.configured:
            raise JevApiError(
                f"no API key: set {API_KEY_ENV} in the environment, or pass api_key=... "
                "Keys must never be committed; the repository is version-controlled."
            )
        payload = self.build_payload(state, questions, session_id=session_id, user=user)
        request = urllib.request.Request(  # noqa: S310 - endpoint is a fixed https default
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace") if error.fp else ""
            raise JevApiError(f"the API returned HTTP {error.code}", status=error.code, body=detail) from error
        except urllib.error.URLError as error:
            raise JevApiError(f"could not reach {self.endpoint}: {error.reason}") from error
        except TimeoutError as error:
            # A read timeout arrives as a bare `TimeoutError` from `http.client`, *not* wrapped in
            # `URLError`, so the handlers above miss it. A long batch died mid-run on this: frame
            # 51 of 603 raised straight past the client and took the whole experiment with it.
            raise JevApiError(f"the request to {self.endpoint} timed out after {self.timeout}s") from error
        except OSError as error:
            # `URLError` and `TimeoutError` are both `OSError` subclasses, so anything reaching
            # here is a socket-level failure of some other kind. Wrapping it keeps every transport
            # failure behind one exception type for the caller.
            raise JevApiError(f"transport failure talking to {self.endpoint}: {error}") from error
        return parse_response(body)


def parse_response(body: str) -> DecisionResponse:
    """Parse a response body. Kept separate so a recorded fixture can be replayed without network."""
    try:
        data = json.loads(body)
    except json.JSONDecodeError as error:
        raise JevApiError("the API returned a body that is not JSON", body=body[:500]) from error
    if "answers" not in data:
        raise JevApiError("the response has no `answers` field", body=body[:500])
    return DecisionResponse(
        answers={key: dict(value) for key, value in data["answers"].items()},
        usage=dict(data.get("usage") or {}),
        task_id=str(data.get("task_id", "")),
        model=str(data.get("model", "")),
        consumed=str(data.get("consumed", "")),
        raw=data,
    )


def route_question(options: Mapping[str, str] | None = None) -> Question:
    """A `choice` question over MiniJev's four routes.

    This is the adapter that lets the existing ladder sit on top of a remote model: the API
    returns probabilities over the same four routes MiniJev already scores locally, so
    `RouteScores` can be built from them and every policy, gate and metric keeps working.
    """
    if options is None:
        descriptions = {
            "local": "Answerable directly, from what is already in the request",
            "rag": "Needs an external source: a document, the user's own records, or a service",
            "cloud": "Needs multi-step construction or reasoning beyond a single lookup",
            "review": "High stakes or under-specified; a person must decide",
        }
        options = {route.value: descriptions[route.value] for route in ROUTE_ORDER}
    return Question(
        type="choice",
        instructions="Classify this request into the most appropriate handler",
        criteria=dict(options),
    )


def scores_from_response(response: DecisionResponse, key: str = "route") -> RouteScores:
    """Convert a Jev choice answer into MiniJev's `RouteScores`.

    `RouteScores` holds logits and deliberately no confidence -- confidence is derived from the
    distribution by `minijev.confidence`, so nothing is lost by ignoring the API's own field.
    Probabilities are converted back to logits; a zero probability becomes a large negative
    finite value rather than `-inf`, so downstream arithmetic cannot produce NaN.
    """
    import math

    probabilities = response.probabilities(key)
    if not probabilities:
        raise JevApiError(f"no probabilities returned for question {key!r}")
    logits = []
    for route in ROUTE_ORDER:
        value = probabilities.get(route.value, 0.0)
        logits.append(math.log(value) if value > 0 else -60.0)
    return RouteScores(route_logits=tuple(logits))
