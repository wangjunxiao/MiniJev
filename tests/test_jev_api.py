"""The Jev API client, tested against the published contract without touching the network.

Two things these tests are built to catch, both of which happened elsewhere in this project:

1. **A fixture that is mistaken for a measurement.** The published `choice` and `score` examples
   carry placeholder task ids (`taxxxxxxxx-...`), so their `confidence` values are illustrative.
   Only the `noul` example looks captured. The tests record which is which instead of treating
   the whole document as ground truth.

2. **A guessed formula shipped as agreement.** `confidence` is passed through verbatim. No simple
   statistic of the distribution fits both examples, and inventing one would manufacture
   consistency with a contract that has not actually been read.
"""

from __future__ import annotations

import io
import json
import math
import urllib.error
from unittest.mock import patch

import pytest

from minijev.confidence import normalized_entropy, top_probability
from minijev.jev_api import (
    API_KEY_ENV,
    QUESTION_TYPES,
    DecisionResponse,
    JevApiClient,
    JevApiError,
    Question,
    parse_response,
    route_question,
    scores_from_response,
)
from minijev.schema import ROUTE_ORDER

# The `noul` example from the published OpenAPI document: a real-looking task id and a `cost`,
# so this one appears to be a captured response.
NOUL_RESPONSE = {
    "id": "gen-dec-1789916581-nOeiIEyTJa7BsdEw3CvR",
    "model": "typesafe/jev-1.13",
    "provider": "TypeSafe",
    "answers": {"deploy_risk": {"type": "noul", "noul": 0.55}},
    "usage": {"input_tokens": 316, "output_tokens": 22, "cost": 0.000013272},
    "task_id": "tae99430-c22e-403a-9468-02462ddcf9bc",
    "consumed": "0.000013272",
}

# The `choice` example: placeholder task id, illustrative numbers.
CHOICE_RESPONSE = {
    "model": "typesafe/jev-1.13",
    "answers": {
        "team": {
            "type": "choice",
            "choice": "payments",
            "probabilities": {"account": 0, "frontend": 0.16, "payments": 0.84},
            "confidence": 0.75,
        }
    },
    "usage": {"input_tokens": 476, "output_tokens": 70},
    "task_id": "taxxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
}

SCORE_RESPONSE = {
    "model": "typesafe/jev-1.13",
    "answers": {
        "urgency": {
            "type": "score",
            "score": 1.99,
            "legend": {"0": "Can wait", "1": "This week", "2": "Blocking revenue"},
            "probabilities": {"0": 0, "1": 0.01, "2": 0.99},
            "confidence": 0.99,
        }
    },
    "usage": {"input_tokens": 420, "output_tokens": 60},
    "task_id": "taxxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
}


# --- question validation --------------------------------------------------------


def test_the_three_primitives_are_the_published_ones():
    assert QUESTION_TYPES == ("noul", "choice", "score")


def test_a_noul_question_needs_only_instructions():
    question = Question(type="noul", instructions="Is this a threat?")
    assert question.to_dict() == {"type": "noul", "instructions": "Is this a threat?"}


def test_a_choice_question_needs_options_and_serialises_them_as_a_mapping():
    question = Question(type="choice", instructions="Which team?", criteria={"a": "first", "b": "second"})
    payload = question.to_dict()
    assert payload["criteria"] == {"a": "first", "b": "second"}
    assert isinstance(payload["criteria"], dict)


def test_a_score_question_needs_an_ordered_list_not_a_mapping():
    ok = Question(type="score", instructions="How urgent?", criteria=["low", "medium", "high"])
    assert ok.to_dict()["criteria"] == ["low", "medium", "high"]
    with pytest.raises(ValueError):
        Question(type="score", instructions="How urgent?", criteria={"low": "x", "high": "y"})


def test_a_score_question_enforces_the_published_level_bounds():
    with pytest.raises(ValueError):
        Question(type="score", instructions="x", criteria=["only one"])
    with pytest.raises(ValueError):
        Question(type="score", instructions="x", criteria=[str(index) for index in range(11)])
    assert Question(type="score", instructions="x", criteria=["a", "b"]).criteria == ["a", "b"]
    assert Question(type="score", instructions="x", criteria=[str(index) for index in range(10)]).criteria


def test_a_choice_question_refuses_more_options_than_the_api_allows():
    too_many = {f"option{index}": "d" for index in range(256)}
    with pytest.raises(ValueError):
        Question(type="choice", instructions="pick", criteria=too_many)


def test_an_unknown_type_and_empty_instructions_are_refused():
    with pytest.raises(ValueError):
        Question(type="likert", instructions="x", criteria=["a", "b"])
    with pytest.raises(ValueError):
        Question(type="noul", instructions="   ")


# --- payload assembly -----------------------------------------------------------


def test_the_payload_matches_the_documented_request_shape():
    client = JevApiClient(api_key="test-key")
    payload = client.build_payload(
        {"customer_tier": "enterprise", "ticket": "blank screen"},
        {"team": Question(type="choice", instructions="Which team?", criteria={"a": "x", "b": "y"})},
    )
    assert payload["model"] == "typesafe/jev-1.13"
    assert payload["state"] == {"customer_tier": "enterprise", "ticket": "blank screen"}
    assert payload["questions"]["team"] == {"type": "choice", "instructions": "Which team?", "criteria": {"a": "x", "b": "y"}}


def test_a_string_state_is_passed_through_unchanged():
    client = JevApiClient(api_key="k")
    payload = client.build_payload("plain text", {"q": Question(type="noul", instructions="?")})
    assert payload["state"] == "plain text"


def test_a_request_without_questions_is_refused_before_the_network():
    client = JevApiClient(api_key="k")
    with pytest.raises(ValueError):
        client.build_payload("state", {})


def test_session_and_user_are_truncated_to_the_documented_limit():
    client = JevApiClient(api_key="k")
    payload = client.build_payload("s", {"q": Question(type="noul", instructions="?")}, session_id="x" * 400, user="y" * 400)
    assert len(payload["session_id"]) == 256
    assert len(payload["user"]) == 256


# --- credentials ----------------------------------------------------------------


def test_the_key_comes_from_the_environment_and_is_never_hardcoded(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    assert JevApiClient().configured is False
    monkeypatch.setenv(API_KEY_ENV, "from-env")
    assert JevApiClient().api_key == "from-env"


def test_calling_without_a_key_fails_loudly_and_says_where_to_put_one(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    client = JevApiClient(api_key="")
    with pytest.raises(JevApiError) as error:
        client.decide("state", {"q": Question(type="noul", instructions="?")})
    assert API_KEY_ENV in str(error.value)
    assert "committed" in str(error.value), "the message must warn about committing keys"


# --- response parsing -----------------------------------------------------------


def test_a_captured_noul_response_parses_with_its_measured_cost():
    response = parse_response(json.dumps(NOUL_RESPONSE))
    assert response.noul("deploy_risk") == pytest.approx(0.55)
    assert response.cost == pytest.approx(0.000013272)
    assert response.input_tokens == 316
    assert response.task_id == "tae99430-c22e-403a-9468-02462ddcf9bc"


def test_a_noul_answer_has_no_confidence_and_that_is_correct():
    """For a Bernoulli the probability is the answer; there is nothing left to summarise."""
    response = parse_response(json.dumps(NOUL_RESPONSE))
    assert response.confidence("deploy_risk") is None
    assert "confidence" not in response.answers["deploy_risk"]


def test_choice_and_score_answers_parse():
    choice = parse_response(json.dumps(CHOICE_RESPONSE))
    assert choice.choice("team") == "payments"
    assert choice.probabilities("team")["payments"] == pytest.approx(0.84)
    score = parse_response(json.dumps(SCORE_RESPONSE))
    assert score.score("urgency") == pytest.approx(1.99)
    assert score.answers["urgency"]["legend"]["2"] == "Blocking revenue"


def test_confidence_is_passed_through_and_not_recomputed():
    """The published examples do not pin the formula, so nothing is inferred from them.

    For the choice example, `max probability` would give 0.84 and the document says 0.75;
    normalized entropy gives 0.60; neither matches. Since that example carries a placeholder task
    id, the value is illustrative rather than measured -- which is a reason to pass it through,
    not a reason to guess a formula that would appear to agree.
    """
    response = parse_response(json.dumps(CHOICE_RESPONSE))
    assert response.confidence("team") == pytest.approx(0.75)
    probabilities = response.probabilities("team")
    assert top_probability(list(probabilities.values())) == pytest.approx(0.84)
    assert normalized_entropy(list(probabilities.values())) == pytest.approx(0.5998, abs=0.001)
    assert response.confidence("team") != pytest.approx(top_probability(list(probabilities.values())))


def test_a_response_without_a_cost_reports_none_rather_than_zero():
    """Zero would read as 'this call was free'; `None` reads as 'not reported'."""
    response = parse_response(json.dumps(CHOICE_RESPONSE))
    assert response.cost is None


def test_a_malformed_body_is_refused():
    with pytest.raises(JevApiError):
        parse_response("not json at all")
    with pytest.raises(JevApiError):
        parse_response(json.dumps({"model": "typesafe/jev-1.13"}))


def test_a_read_timeout_is_wrapped_not_left_to_escape():
    """A bare `TimeoutError` from `http.client` is not a `URLError`.

    A 603-call batch died on exactly this: frame 51 raised `TimeoutError` straight past the
    client's handlers and took the whole experiment with it, losing every response. Every
    transport failure has to arrive as one exception type the caller can catch.
    """
    client = JevApiClient(api_key="k")
    with patch("urllib.request.urlopen", side_effect=TimeoutError("The read operation timed out")):
        with pytest.raises(JevApiError) as error:
            client.decide("state", {"q": Question(type="noul", instructions="?")})
    assert "timed out" in str(error.value)
    assert error.value.status is None


def test_an_unexpected_socket_error_is_wrapped_too():
    client = JevApiClient(api_key="k")
    with patch("urllib.request.urlopen", side_effect=OSError("connection reset by peer")):
        with pytest.raises(JevApiError) as error:
            client.decide("state", {"q": Question(type="noul", instructions="?")})
    assert "transport failure" in str(error.value)


def test_an_http_error_carries_its_status_and_body():
    client = JevApiClient(api_key="k")

    def raise_http(*_args, **_kwargs):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b'{"message":"invalid api key"}'))

    with patch("urllib.request.urlopen", side_effect=raise_http):
        with pytest.raises(JevApiError) as error:
            client.decide("state", {"q": Question(type="noul", instructions="?")})
    assert error.value.status == 401
    assert "invalid api key" in error.value.body


# --- the adapter onto MiniJev's ladder ------------------------------------------


def test_the_route_question_offers_exactly_minijevs_four_routes():
    question = route_question()
    assert set(question.criteria) == {route.value for route in ROUTE_ORDER}
    assert question.type == "choice"


def test_a_choice_answer_becomes_route_scores_the_ladder_can_use():
    response = DecisionResponse(
        answers={"route": {"type": "choice", "choice": "cloud", "probabilities": {"local": 0.1, "rag": 0.2, "cloud": 0.6, "review": 0.1}}}
    )
    scores = scores_from_response(response)
    assert len(scores.route_logits) == len(ROUTE_ORDER)
    assert scores.route_logits[list(ROUTE_ORDER).index(next(r for r in ROUTE_ORDER if r.value == "cloud"))] == max(scores.route_logits)
    # Probabilities are recovered exactly once exponentiated and normalised.
    exponentials = [math.exp(value) for value in scores.route_logits]
    total = sum(exponentials)
    recovered = [value / total for value in exponentials]
    assert recovered[2] == pytest.approx(0.6, abs=1e-6)


def test_a_zero_probability_becomes_a_finite_logit_not_negative_infinity():
    """`-inf` would turn any downstream average into NaN."""
    response = DecisionResponse(answers={"route": {"probabilities": {"local": 0.0, "rag": 0.0, "cloud": 1.0, "review": 0.0}}})
    scores = scores_from_response(response)
    assert all(math.isfinite(value) for value in scores.route_logits)


def test_a_response_missing_the_distribution_is_refused():
    response = DecisionResponse(answers={"route": {"choice": "cloud"}})
    with pytest.raises(JevApiError):
        scores_from_response(response)
