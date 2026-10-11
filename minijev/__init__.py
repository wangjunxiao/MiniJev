"""MiniJev: the smallest useful Jev decision model, built with tinygrad.

The import surface is deliberately narrow. `Router` and `Decision` are the public
idea; everything else is an implementation detail of how scores are produced and
spent.

Training, inference and evaluation are three entry points over this one set of
modules, not three copies of the code::

    python -m minijev.training   --train data/train.jsonl ...
    python -m minijev.inference  "请总结这份合同的风险"
    python -m minijev.evaluation data/test.jsonl --baseline
"""

from .confidence import CONFIDENCE_MEASURES, DEFAULT_MEASURE, confidence_from, distribution_shape
from .config import JevConfig
from .model import JevOutputs, TinyJev, load_checkpoint, save_checkpoint
from .policies import (
    DEFAULT_LADDER,
    ArgmaxPolicy,
    FlatThresholdPolicy,
    RiskLadderPolicy,
    RouteGate,
    sweep_ladder,
)
from .routers import ROUTERS, DecisionStats, NeuralRouter, PriorRouter, Router, RuleRouter
from .schema import ROUTE_ORDER, Candidate, Decision, JevSample, Route, RouteScores
from .tokenizer import ByteTokenizer

__all__ = [
    "CONFIDENCE_MEASURES",
    "DEFAULT_LADDER",
    "DEFAULT_MEASURE",
    "ROUTE_ORDER",
    "ROUTERS",
    "ArgmaxPolicy",
    "ByteTokenizer",
    "Candidate",
    "Decision",
    "DecisionStats",
    "FlatThresholdPolicy",
    "JevConfig",
    "JevOutputs",
    "JevSample",
    "NeuralRouter",
    "PriorRouter",
    "RiskLadderPolicy",
    "Route",
    "RouteGate",
    "RouteScores",
    "Router",
    "RuleRouter",
    "TinyJev",
    "confidence_from",
    "distribution_shape",
    "load_checkpoint",
    "save_checkpoint",
    "sweep_ladder",
]
