"""Deterministic slot cross-product generation for boundary frames.

The hand-written frame bank is useful for explaining the routing rule, but it is
too small to estimate a collapse rate precisely.  This module expands the same
slots into a controlled product and keeps the route label derived by
``frames.route_from_frame``.  It deliberately lives beside ``frames.py`` rather
than mutating ``FRAME_BANK``: the original bank and its benchmark remain a
stable, human-readable reference set.

The generator selects an equal number of frames for each route from a larger
valid product, then holds out complete semantic groups.  A generated row carries
the complete frame in metadata, so a later audit can recompute the route and
verify that no asserted label slipped into the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
from typing import Any

from .frames import Derivation, Risk, RouteFrame, Source, deciding_slots_differing, route_from_frame
from .schema import ROUTE_ORDER, JevSample, Route, save_jsonl


@dataclass(frozen=True)
class ProductOptions:
    """Controls for the deterministic product.

    ``target_per_route`` is the number of selected frames before the benchmark
    split.  The default produces 1,280 frames (320 per route), with 25% held out
    as a new benchmark and the remaining 960 available for training.
    """

    target_per_route: int = 320
    benchmark_fraction: float = 0.4
    variants: int = 3
    seed: int = 17
    naming: str = "explicit"
    """`explicit` states each deciding slot in the text; `implied` never does.

    Measured difference: a classifier reading only the explicit markers reconstructs the
    route at **1.0000** on the explicit product and 0.2609 (chance) on the hand-written
    benchmark. So `explicit` is a marker lookup and cannot separate encoders -- every arm
    saturates near 0.97. `implied` has no marker to read.
    """

    object_holdout: float = 0.25
    """Fraction of *objects* reserved for the benchmark.

    Holding out frames is not enough. The object fixes `source` and `risk`, which are 3 of
    the 5 deciding slots, so a benchmark frame whose object was seen in training can be
    partly answered from the object table alone. Before this option existed, **every**
    benchmark frame carried a seen object (320/320 explicit, 164/164 implied). Reserving
    whole objects makes the benchmark ask whether the route rule generalises to a topic the
    model has never met.
    """

    def __post_init__(self) -> None:
        if self.target_per_route <= 0:
            raise ValueError("target_per_route must be positive")
        if not 0.0 < self.benchmark_fraction < 1.0:
            raise ValueError("benchmark_fraction must be between 0 and 1")
        if self.variants <= 0:
            raise ValueError("variants must be positive")
        if self.naming not in {"explicit", "implied"}:
            raise ValueError("naming must be 'explicit' or 'implied'")
        if not 0.0 <= self.object_holdout < 1.0:
            raise ValueError("object_holdout must be in [0, 1)")


@dataclass(frozen=True)
class ProductFrameCase:
    """One generated frame and its surface variants."""

    frame_id: str
    frame: RouteFrame
    texts: tuple[str, ...]


@dataclass(frozen=True)
class FrameProduct:
    """Selected product cases and a group-aware train/benchmark split."""

    cases: tuple[ProductFrameCase, ...]
    benchmark_ids: frozenset[str]
    options: ProductOptions

    @property
    def train_ids(self) -> frozenset[str]:
        return frozenset(case.frame_id for case in self.cases if case.frame_id not in self.benchmark_ids)

    @property
    def benchmark_cases(self) -> tuple[ProductFrameCase, ...]:
        return tuple(case for case in self.cases if case.frame_id in self.benchmark_ids)

    @property
    def train_cases(self) -> tuple[ProductFrameCase, ...]:
        return tuple(case for case in self.cases if case.frame_id not in self.benchmark_ids)


def product_contrast_pairs(product_set: FrameProduct, *, split: str = "benchmark") -> tuple[tuple[str, str], ...]:
    """Find a deterministic disjoint set of one-slot contrast pairs.

    The hand-written benchmark stores pairs explicitly.  Product frames are made
    by a Cartesian product, so the same relation can be derived: two frames are a
    contrast when exactly one deciding slot differs and their derived routes are
    different.  Greedy matching keeps pair outcomes independent enough for a
    useful collapse-rate estimate instead of counting every possible cross-pair.
    """

    if split == "benchmark":
        cases = product_set.benchmark_cases
    elif split == "train":
        cases = product_set.train_cases
    elif split == "all":
        cases = product_set.cases
    else:
        raise ValueError("split must be 'all', 'train' or 'benchmark'")
    candidates: list[tuple[str, str, str]] = []
    for left, right in combinations(sorted(cases, key=lambda case: case.frame_id), 2):
        differing = deciding_slots_differing(left.frame, right.frame)
        if len(differing) == 1 and route_from_frame(left.frame) is not route_from_frame(right.frame):
            candidates.append((differing[0], left.frame_id, right.frame_id))
    used: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for _, left_id, right_id in sorted(candidates):
        if left_id in used or right_id in used:
            continue
        pairs.append((left_id, right_id))
        used.update((left_id, right_id))
    return tuple(pairs)


# The source phrase is intentionally visible in the query.  It gives the model
# a learnable cue for the deciding slot while retaining enough surface variation
# in predicates and objects to prevent an id-only lookup.
#
# Measured caveat, and the reason `naming="implied"` exists: making the slot visible is
# not a cue, it is a *decoder*. A classifier that reads only these markers -- the source
# prefix, the derivation clause, the step clauses, the risk bracket and the
# `来源线索：` hint -- reconstructs the route with **accuracy 1.0000** on both product
# splits, against 0.2609 (chance) on the hand-written benchmark. So the explicit product
# measures marker lookup: every model trained on it saturates near 97%, and the corpus
# cannot separate encoders. `naming="implied"` removes every marker and makes the route
# follow from what the words mean instead.
_SOURCE_PHRASES: dict[Source, tuple[str, str]] = {
    Source.GIVEN_TEXT: ("给定文本中的", "输入里的"),
    Source.DEVICE: ("设备里的", "本机中的"),
    Source.USER_DATA: ("我的数据中的", "我的记录里的"),
    Source.DOCUMENT: ("知识库中的", "产品资料里的"),
    Source.SERVICE: ("在线服务返回的", "外部服务里的"),
    Source.NONE: ("", "已知的"),
}

_PREDICATE_DERIVATION: dict[str, Derivation] = {
    # Reading or transforming something that exists is extractive; producing a judgement,
    # a comparison or a plan is constructive. This is the implication a reader makes, and
    # it is what replaces the explicit "，并解释原因或给出方案" marker.
    "查": Derivation.EXTRACT,
    "统计": Derivation.EXTRACT,
    "列出": Derivation.EXTRACT,
    "核对": Derivation.EXTRACT,
    "翻译": Derivation.EXTRACT,
    "转成": Derivation.EXTRACT,
    "读取": Derivation.EXTRACT,
    "提取": Derivation.EXTRACT,
    "分析": Derivation.CONSTRUCT,
    "比较": Derivation.CONSTRUCT,
    "总结": Derivation.CONSTRUCT,
    "规划": Derivation.CONSTRUCT,
    "解释": Derivation.CONSTRUCT,
    "评估": Derivation.CONSTRUCT,
    "设计": Derivation.CONSTRUCT,
    "预测": Derivation.CONSTRUCT,
}

_PREDICATE_GROUPS: dict[str, tuple[str, ...]] = {
    # Verbs are not interchangeable: 统计安装步骤 is not a sentence anyone writes, and a
    # free predicate x object product produced exactly that. Each group is also a
    # *derivation* class -- reading, counting, listing, verifying, converting and
    # translating all return something that already exists, while analysing, judging and
    # composing have to build it.
    "read": ("查", "检索", "定位", "找出"),
    "list": ("列出", "摘出", "标出"),
    "count": ("统计", "计算"),
    "extract": ("提取", "读取", "截取"),
    "verify": ("核对", "比对"),
    "convert": ("换算", "转成"),
    "translate": ("翻译", "复制"),
    "analyse": ("分析", "解释"),
    "judge": ("评估", "比较"),
    "compose": ("总结", "汇总", "规划", "设计", "预测"),
}

_EXTRACTIVE_GROUPS: frozenset[str] = frozenset({"read", "list", "count", "extract", "verify", "convert", "translate"})


def _group_derivation(group: str) -> Derivation:
    return Derivation.EXTRACT if group in _EXTRACTIVE_GROUPS else Derivation.CONSTRUCT


_OBJECT_CONTEXT: dict[str, tuple[Source, Risk, str, tuple[str, ...]]] = {
    # `(source, risk, domain, allowed predicate groups)`. Source and risk are implied by
    # what the object IS; the allowed groups are the verbs a speaker would actually use
    # with it. Both are needed: the object fixes where the answer lives, the verb fixes
    # whether the answer is fetched or built.
    #
    # The table has to spread across all four routes, because the implied product can only
    # take route-balanced quotas from what the table offers: whichever route has fewest
    # legal combinations caps the whole corpus.
    # --- document -> rag ---
    "安装步骤": (Source.DOCUMENT, Risk.LOW, "product_manual", ("read", "list", "translate", "analyse")),
    "产品规格": (Source.DOCUMENT, Risk.LOW, "product_manual", ("read", "list", "extract", "convert", "analyse")),
    "错误码含义": (Source.DOCUMENT, Risk.LOW, "tech_documentation", ("read", "list", "translate", "analyse")),
    "培训材料": (Source.DOCUMENT, Risk.LOW, "enterprise_knowledge", ("read", "list", "translate", "compose")),
    "保修条款": (Source.DOCUMENT, Risk.LOW, "product_manual", ("read", "list", "translate", "judge")),
    "入职指引": (Source.DOCUMENT, Risk.LOW, "enterprise_knowledge", ("read", "list", "extract", "compose")),
    "接口文档": (Source.DOCUMENT, Risk.LOW, "tech_documentation", ("read", "list", "extract", "analyse")),
    "运维手册": (Source.DOCUMENT, Risk.LOW, "product_manual", ("read", "list", "verify", "analyse")),
    # --- user records -> rag ---
    "报销明细": (Source.USER_DATA, Risk.LOW, "enterprise_knowledge", ("read", "list", "count", "verify")),
    "季度数据": (Source.USER_DATA, Risk.LOW, "planning", ("read", "count", "verify", "analyse", "compose")),
    "项目进度": (Source.USER_DATA, Risk.LOW, "enterprise_knowledge", ("read", "list", "analyse", "compose")),
    "库存数量": (Source.USER_DATA, Risk.LOW, "enterprise_knowledge", ("read", "count", "verify", "convert")),
    "排班表": (Source.USER_DATA, Risk.LOW, "enterprise_knowledge", ("read", "list", "verify", "compose")),
    "考勤记录": (Source.USER_DATA, Risk.LOW, "enterprise_knowledge", ("read", "count", "verify", "analyse")),
    "订单明细": (Source.USER_DATA, Risk.LOW, "enterprise_knowledge", ("read", "list", "count", "verify")),
    "预算变化": (Source.USER_DATA, Risk.MEDIUM, "planning", ("read", "count", "verify", "analyse", "compose")),
    "客户名单": (Source.USER_DATA, Risk.MEDIUM, "enterprise_knowledge", ("read", "list", "extract", "judge")),
    "合同条款": (Source.USER_DATA, Risk.MEDIUM, "legal", ("read", "list", "extract", "judge")),
    # --- live services -> rag ---
    "服务用量": (Source.SERVICE, Risk.LOW, "realtime_info", ("read", "count", "verify", "analyse")),
    "天气信息": (Source.SERVICE, Risk.LOW, "realtime_info", ("read", "list", "analyse", "compose")),
    "汇率报价": (Source.SERVICE, Risk.LOW, "realtime_info", ("read", "convert", "verify", "analyse")),
    "物流轨迹": (Source.SERVICE, Risk.LOW, "realtime_info", ("read", "list", "analyse")),
    "余票情况": (Source.SERVICE, Risk.LOW, "realtime_info", ("read", "count", "verify")),
    "航班状态": (Source.SERVICE, Risk.MEDIUM, "realtime_info", ("read", "list", "analyse", "compose")),
    # --- text already in hand, low stakes -> local ---
    "会议记录": (Source.GIVEN_TEXT, Risk.LOW, "format_conversion", ("read", "list", "extract", "translate", "compose")),
    "部署日志": (Source.GIVEN_TEXT, Risk.LOW, "code_analysis", ("read", "list", "count", "extract", "analyse")),
    "接口响应": (Source.GIVEN_TEXT, Risk.LOW, "code_analysis", ("read", "list", "extract", "analyse")),
    "客户来信": (Source.GIVEN_TEXT, Risk.LOW, "format_conversion", ("read", "list", "translate", "analyse")),
    "问卷作答": (Source.GIVEN_TEXT, Risk.LOW, "format_conversion", ("read", "count", "list", "analyse")),
    "访谈文字稿": (Source.GIVEN_TEXT, Risk.LOW, "format_conversion", ("read", "list", "extract", "compose")),
    "变更说明": (Source.GIVEN_TEXT, Risk.LOW, "format_conversion", ("read", "list", "translate", "compose")),
    "巡检笔记": (Source.GIVEN_TEXT, Risk.LOW, "format_conversion", ("read", "list", "extract", "verify")),
    "测试报告": (Source.GIVEN_TEXT, Risk.LOW, "code_analysis", ("read", "count", "verify", "analyse")),
    "需求描述": (Source.GIVEN_TEXT, Risk.LOW, "format_conversion", ("read", "list", "extract", "compose")),
    "报错信息": (Source.GIVEN_TEXT, Risk.LOW, "code_analysis", ("read", "list", "translate", "analyse")),
    "配置片段": (Source.GIVEN_TEXT, Risk.LOW, "code_analysis", ("read", "list", "verify", "analyse")),
    "日程安排": (Source.GIVEN_TEXT, Risk.LOW, "format_conversion", ("read", "list", "verify", "compose")),
    "访谈提纲": (Source.GIVEN_TEXT, Risk.LOW, "format_conversion", ("read", "list", "extract", "compose")),
    "故障原因": (Source.GIVEN_TEXT, Risk.MEDIUM, "code_analysis", ("read", "list", "analyse", "compose")),
    "实验结果": (Source.GIVEN_TEXT, Risk.MEDIUM, "academic", ("read", "count", "verify", "analyse")),
    "现场照片说明": (Source.GIVEN_TEXT, Risk.MEDIUM, "format_conversion", ("read", "list", "extract", "analyse")),
    # --- intrinsically high stakes -> review whatever is asked ---
    "权限记录": (Source.USER_DATA, Risk.HIGH, "security", ("read", "list", "verify", "judge")),
    "用药剂量": (Source.NONE, Risk.HIGH, "medical", ("read", "analyse", "judge")),
    "体检指标": (Source.NONE, Risk.HIGH, "medical", ("read", "analyse", "judge")),
    "投资方案": (Source.NONE, Risk.HIGH, "financial", ("analyse", "judge", "compose")),
    "抵押方案": (Source.NONE, Risk.HIGH, "financial", ("analyse", "judge", "compose")),
    "隐私条款": (Source.NONE, Risk.HIGH, "legal", ("read", "analyse", "judge")),
    "竞业协议": (Source.NONE, Risk.HIGH, "legal", ("read", "analyse", "judge")),
    "保险条款": (Source.NONE, Risk.HIGH, "financial", ("read", "analyse", "judge")),
    # --- expansion for measurement resolution -------------------------------------------
    # The implied benchmark's contrast-pair count was capped by whichever route had the fewest
    # candidate frames, and that was `local` at 163, holding the whole corpus to a quota of 163
    # and the benchmark to 68 pairs. At 68 pairs only a ~19-point collapse difference is
    # resolvable (95% confidence, 80% power); reaching ~197 pairs would resolve 10 points. Since
    # the quota is the minimum across routes, only adding objects to the *binding* routes helps,
    # so these entries were chosen to lift `local`, `rag` and `review` together.
    #
    # Every object is paired only with verb groups a speaker would actually use on it: a free
    # cross-product previously emitted `预测安装步骤` and `统计培训材料`, which is the same defect
    # class as the marker leak one level up.
    #
    # --- given_text: lifts `local` ---
    "会议纪要": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "list", "extract", "compose")),
    "值班记录": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "list", "count", "verify")),
    "客服对话": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "list", "translate", "analyse")),
    "采购清单": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "list", "count", "verify")),
    "巡检报告": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "count", "verify", "analyse")),
    "用户反馈": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "list", "extract", "analyse")),
    "培训签到": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "count", "list", "verify")),
    "设备参数": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "list", "convert", "analyse")),
    "测速结果": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "count", "verify", "analyse")),
    "排产计划": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "list", "analyse", "compose")),
    "库存盘点": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "count", "verify", "extract")),
    "质量抽检": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "count", "verify", "analyse")),
    "服务工单": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "list", "extract", "compose")),
    "系统日志": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "list", "count", "analyse")),
    "报销凭证": (Source.GIVEN_TEXT, Risk.LOW, "work", ("read", "list", "count", "verify")),
    "教学大纲": (Source.GIVEN_TEXT, Risk.LOW, "study", ("read", "list", "extract", "compose")),
    "翻译稿件": (Source.GIVEN_TEXT, Risk.LOW, "study", ("read", "list", "translate", "verify")),
    "调研问卷": (Source.GIVEN_TEXT, Risk.LOW, "study", ("read", "count", "list", "analyse")),
    # --- external: lifts `rag` ---
    "操作指南": (Source.DOCUMENT, Risk.LOW, "work", ("read", "list", "translate", "analyse")),
    "故障手册": (Source.DOCUMENT, Risk.LOW, "work", ("read", "list", "verify", "analyse")),
    "采购制度": (Source.DOCUMENT, Risk.LOW, "work", ("read", "list", "extract", "compose")),
    "安全规范": (Source.DOCUMENT, Risk.MEDIUM, "work", ("read", "list", "verify", "judge")),
    "缴费记录": (Source.USER_DATA, Risk.LOW, "financial", ("read", "count", "verify", "extract")),
    "里程记录": (Source.USER_DATA, Risk.LOW, "work", ("read", "count", "convert", "analyse")),
    "借阅记录": (Source.USER_DATA, Risk.LOW, "study", ("read", "list", "count", "analyse")),
    "售后记录": (Source.USER_DATA, Risk.LOW, "work", ("read", "list", "extract", "analyse")),
    "资产台账": (Source.USER_DATA, Risk.LOW, "work", ("read", "list", "count", "verify")),
    "路况信息": (Source.SERVICE, Risk.LOW, "life", ("read", "list", "analyse", "compose")),
    "排队叫号": (Source.SERVICE, Risk.LOW, "life", ("read", "count", "verify", "analyse")),
    "电价行情": (Source.SERVICE, Risk.LOW, "life", ("read", "convert", "analyse", "compose")),
    # --- high stakes: lifts `review` ---
    "手术方案": (Source.NONE, Risk.HIGH, "medical", ("analyse", "judge")),
    "遗嘱草稿": (Source.NONE, Risk.HIGH, "legal", ("read", "analyse", "judge")),
    "保单条款": (Source.NONE, Risk.HIGH, "financial", ("read", "analyse", "judge")),
    "贷款方案": (Source.NONE, Risk.HIGH, "financial", ("analyse", "judge", "compose")),
    "处方剂量": (Source.NONE, Risk.HIGH, "medical", ("read", "analyse", "judge")),
    "数据出境方案": (Source.NONE, Risk.HIGH, "legal", ("analyse", "judge", "compose")),
}
"""Objects whose meaning fixes the source and the stakes.

`review` has to be reachable without a marker, so several objects are intrinsically
high-stakes: a dose, an investment, a privacy clause. Their route is review for *any*
predicate, which is exactly the boundary the hand-written bank tests by hand.
"""

_IMPLIED_STEP_CLAUSES: tuple[tuple[str, ...], ...] = (
    (),
    ("并给出下一步", "然后说明怎么处理", "接着列出要做的事"),
    ("再排好优先级", "最后给出完整方案", "并说明各自的取舍"),
)
"""Step phrasing by count. Several alternatives per count so that no single clause is a
perfect key -- the count has to be inferred from how many things are asked for."""

_PREDICATES: tuple[str, ...] = (
    "查",
    "统计",
    "列出",
    "核对",
    "翻译",
    "转成",
    "读取",
    "提取",
    "分析",
    "比较",
    "总结",
    "规划",
    "解释",
    "评估",
    "设计",
    "预测",
)

_OBJECTS: tuple[str, ...] = (
    "安装步骤",
    "报销明细",
    "错误码含义",
    "季度数据",
    "合同条款",
    "客户名单",
    "会议记录",
    "部署日志",
    "预算变化",
    "服务用量",
    "航班状态",
    "天气信息",
    "产品规格",
    "故障原因",
    "实验结果",
    "培训材料",
    "项目进度",
    "库存数量",
    "权限记录",
    "接口响应",
)

_SOURCES: tuple[Source, ...] = tuple(Source)
_DERIVATIONS: tuple[Derivation, ...] = tuple(Derivation)
_STEPS: tuple[int, ...] = (1, 2, 3)
_RISKS: tuple[Risk, ...] = tuple(Risk)


def _source_text(source: Source, object_: str, alternate: bool = False) -> str:
    prefix = _SOURCE_PHRASES[source][1 if alternate else 0]
    return f"{prefix}{object_}" if prefix else object_


def _build_text(
    predicate: str,
    object_: str,
    source: Source,
    derivation: Derivation,
    steps: int,
    risk: Risk,
    underspecified: bool,
    *,
    alternate: bool = False,
) -> str:
    if underspecified:
        source_hint = f"来源线索：{source.value}"
        wording = "尚未说明具体对象" if alternate else "具体对象还没有说明"
        return f"请{predicate}一下，关于{object_}的{wording}（{source_hint}；预计{steps}步；{risk.value}风险；{derivation.value}）"

    subject = _source_text(source, object_, alternate)
    text = f"{predicate}{subject}"
    if derivation is Derivation.CONSTRUCT:
        text += "，并解释原因或给出方案"
    else:
        text += "，直接列出结果"
    if steps >= 2:
        text += "，然后核对关键依据"
    if steps >= 3:
        text += "，最后给出下一步行动"
    if risk is Risk.MEDIUM:
        text += "（请注明依据）"
    elif risk is Risk.HIGH:
        text += "（请谨慎核实）"
    return text


def _variants(frame: RouteFrame, *, variants: int, naming: str = "explicit") -> tuple[str, ...]:
    if naming == "implied":
        texts = [
            _build_implied_text(frame.predicate, frame.object, frame.steps, frame.underspecified),
            _build_implied_text(frame.predicate, frame.object, frame.steps, frame.underspecified, alternate=True),
        ]
    else:
        texts = [
            _build_text(frame.predicate, frame.object, frame.source, frame.derivation, frame.steps, frame.risk, frame.underspecified),
            _build_text(frame.predicate, frame.object, frame.source, frame.derivation, frame.steps, frame.risk, frame.underspecified, alternate=True),
        ]
    suffixes = ("请按这些条件回答。", "结果请说清楚。", "需要时说明判断依据。")
    index = 0
    while len(texts) < variants:
        texts.append(texts[0] + suffixes[index % len(suffixes)])
        index += 1
    return tuple(texts[:variants])


def _build_implied_text(predicate: str, object_: str, steps: int, underspecified: bool, *, alternate: bool = False) -> str:
    """Text that never names a deciding slot.

    Every slot must be inferred: the predicate carries the derivation (`统计` is
    extractive, `分析` is constructive), the object carries the source and the stakes
    (`航班状态` must come from a service; `用药剂量` is high-risk whatever is asked), and
    the step count shows up as how many things the sentence asks for. Nothing is labelled,
    so there is no marker to look up -- which is the whole point.
    """
    if underspecified:
        # The topic is named but the operation never is, so the frame is underspecified
        # however the object reads. It must still vary by object: an earlier version
        # returned one constant sentence, so all 38 objects produced identical rows.
        return f"帮我把{object_}那件事处理一下" if not alternate else f"{object_}这事儿你看着办吧"

    # The two realisations must differ even when there is no step clause to vary. They
    # originally did not, so `steps=1` frames produced duplicate variants, the uniqueness
    # check dropped them, and every extractive frame silently vanished -- taking `local`
    # and `rag` with it.
    text = f"帮我{predicate}一下{object_}" if alternate else f"{predicate}{object_}"
    clauses = _IMPLIED_STEP_CLAUSES[min(steps, len(_IMPLIED_STEP_CLAUSES)) - 1]
    if clauses:
        # Deterministic per (predicate, object) so a variant is stable across runs, but
        # not the same clause for every frame at that step count.
        index = (len(predicate) + len(object_)) % len(clauses)
        text += "，" + clauses[(index + (1 if alternate else 0)) % len(clauses)]
    return text


def _implied_candidate_cases() -> Iterable[ProductFrameCase]:
    """The product with every deciding slot implied rather than stated.

    The free cross-product is not available here, and that is the finding: source and risk
    are properties of the *object*, and the derivation is a property of the *predicate*.
    Any other combination is a sentence no speaker would produce, and the only way to make
    it readable is to label it -- which is exactly what turned the explicit product into a
    marker lookup. So this product varies what genuinely varies (predicate, steps,
    underspecified) inside each object's natural context.
    """
    serial = 0
    for object_, (source, risk, domain, groups) in _OBJECT_CONTEXT.items():
        allowed = [(group, predicate) for group in groups for predicate in _PREDICATE_GROUPS[group]]
        for group, predicate in allowed:
            derivation = _group_derivation(group)
            for steps in _STEPS:
                if derivation is Derivation.EXTRACT and steps > 1:
                    # A second clause on an extractive verb reads as a second request,
                    # which would make it constructive; keep the slot combination honest.
                    continue
                frame = RouteFrame(predicate, object_, source, derivation, steps=steps, risk=risk, domain=domain)
                frame_id = f"im{serial:06d}"
                serial += 1
                variants = _variants(frame, variants=3, naming="implied")
                if len(set(variants)) != len(variants):
                    continue
                yield ProductFrameCase(frame_id, frame, variants)
        # The vague request: the object is named but never asked about, so the frame is
        # underspecified and routes to review whatever else is true of it.
        frame = RouteFrame("处理", object_, source, Derivation.EXTRACT, steps=1, risk=risk, underspecified=True, domain=domain)
        frame_id = f"im{serial:06d}"
        serial += 1
        yield ProductFrameCase(frame_id, frame, _variants(frame, variants=3, naming="implied"))


def _candidate_cases(naming: str = "explicit") -> Iterable[ProductFrameCase]:
    """Yield the legal slot product before route-balanced selection.

    An underspecified request names a topic ("关于安装步骤的") but never states what to do
    with it. `derivation` and `steps` are therefore *undefined* for such a frame, not
    merely unstated: there is no operation to classify as extract-vs-construct and no
    step count to count. Pinning them to their canonical values (EXTRACT, 1) keeps the
    annotation honest, and it also makes the surface text an injective function of the
    frame. Left free, the six combinations of derivation and steps would emit six
    identical queries, and a duplicate that straddles the train/benchmark split is
    leakage -- the row would measure recall instead of routing.
    """
    if naming == "implied":
        yield from _implied_candidate_cases()
        return

    serial = 0
    for predicate, object_, source, derivation, steps, risk, underspecified in product(
        _PREDICATES, _OBJECTS, _SOURCES, _DERIVATIONS, _STEPS, _RISKS, (False, True)
    ):
        # A request without a specified object is meaningful only as a review
        # case; keeping it out of other route strata avoids synthetic contradictions.
        if underspecified and (risk is not Risk.MEDIUM or derivation is not Derivation.EXTRACT or steps != 1):
            continue
        frame = RouteFrame(
            predicate,
            object_,
            source,
            derivation,
            steps=steps,
            risk=risk,
            underspecified=underspecified,
            domain="slot_product",
        )
        frame_id = f"xp{serial:06d}"
        serial += 1
        variants = _variants(frame, variants=3)
        if len(set(variants)) != len(variants):
            raise ValueError(f"{frame_id}: variants are not distinct, so the text does not determine the frame")
        yield ProductFrameCase(frame_id, frame, variants)


def _stratum(case: ProductFrameCase) -> tuple[str, ...]:
    frame = case.frame
    return (
        frame.source.value,
        frame.derivation.value,
        str(min(frame.steps, 3)),
        frame.risk.value,
        str(frame.underspecified),
    )


def _select_route(candidates: Sequence[ProductFrameCase], quota: int, seed: int) -> list[ProductFrameCase]:
    if len(candidates) < quota:
        raise ValueError(f"only {len(candidates)} candidates available for a route; need {quota}")
    by_stratum: dict[tuple[str, ...], list[ProductFrameCase]] = defaultdict(list)
    for case in candidates:
        by_stratum[_stratum(case)].append(case)
    rng = random.Random(seed)
    for rows in by_stratum.values():
        rows.sort(key=lambda case: case.frame_id)
        rng.shuffle(rows)

    selected: list[ProductFrameCase] = []
    strata = sorted(by_stratum)
    cursor = 0
    while len(selected) < quota:
        key = strata[cursor % len(strata)]
        rows = by_stratum[key]
        if rows:
            selected.append(rows.pop())
        cursor += 1
        if cursor >= len(strata) and not any(by_stratum[key] for key in strata):
            break
    if len(selected) != quota:
        raise ValueError(f"stratified selection stopped at {len(selected)} of {quota}")
    return selected


@lru_cache(maxsize=8)
def _generate_cached(options: ProductOptions) -> FrameProduct:
    """Generate a balanced, deterministic frame product and group split."""

    candidates_by_route: dict[Route, list[ProductFrameCase]] = defaultdict(list)
    for case in _candidate_cases(options.naming):
        candidates_by_route[route_from_frame(case.frame)].append(case)

    # A semantically constrained product has fewer legal combinations than a free
    # cross-product: `implied` takes source and risk from the object, so the object table
    # caps every route. Clamping every route to the smallest one keeps the balance, which
    # is the property that matters, at the cost of a smaller corpus -- and the reduction is
    # real information about the design rather than something to paper over.
    available = {route: len(candidates_by_route[route]) for route in ROUTE_ORDER}
    missing = [route.value for route, count in available.items() if count < 2]
    if missing:
        raise ValueError(f"no candidates for route(s) {missing} under naming={options.naming!r}")
    quota = min(options.target_per_route, min(available.values()))

    selected: list[ProductFrameCase] = []
    for index, route in enumerate(ROUTE_ORDER):
        selected.extend(_select_route(candidates_by_route[route], quota, options.seed + index))
    selected.sort(key=lambda case: case.frame_id)

    # Hold out whole *objects*, not frames. The object fixes `source` and `risk` -- 3 of the
    # 5 deciding slots -- so a frame whose object was seen in training is partly answerable
    # from the object table. Selecting objects greedily, by the routes they contribute to,
    # keeps all four routes represented in the held-out half.
    held_out = _holdout_objects(selected, options)
    benchmark_ids = {case.frame_id for case in selected if case.frame.object in held_out}
    return FrameProduct(tuple(selected), frozenset(benchmark_ids), options)


def _holdout_objects(selected: Sequence[ProductFrameCase], options: ProductOptions) -> set[str]:
    """Deterministically reserve a route-covering subset of objects for the benchmark."""
    if options.object_holdout <= 0.0:
        return set()

    by_object: dict[str, list[ProductFrameCase]] = defaultdict(list)
    for case in selected:
        by_object[case.frame.object].append(case)
    order = sorted(by_object, key=lambda name: hashlib.sha256(f"{options.seed}:{name}".encode()).hexdigest())

    per_object = {name: Counter(route_from_frame(case.frame) for case in cases) for name, cases in by_object.items()}
    totals = Counter(route_from_frame(case.frame) for case in selected)
    target = {route: max(1, round(totals[route] * options.benchmark_fraction)) for route in ROUTE_ORDER}

    held: set[str] = set()
    got: Counter = Counter()
    for name in order:
        if all(got[route] >= target[route] for route in ROUTE_ORDER):
            break
        held.add(name)
        got.update(per_object[name])
    # A route that no early object covers would leave the benchmark unbalanced; top it up
    # with objects that do contribute there.
    for route in ROUTE_ORDER:
        for name in order:
            if got[route] >= target[route]:
                break
            if name not in held and per_object[name].get(route):
                held.add(name)
                got.update(per_object[name])
    return held


def generate_frame_product(options: ProductOptions | None = None) -> FrameProduct:
    """Generate a balanced, deterministic frame product and group split."""

    return _generate_cached(options or ProductOptions())


def product_rows(product_set: FrameProduct, *, split: str = "all", variants: int | None = None) -> list[JevSample]:
    """Materialise generated cases as schema rows for ``train`` or ``benchmark``."""

    if split not in {"all", "train", "benchmark"}:
        raise ValueError("split must be 'all', 'train' or 'benchmark'")
    count = product_set.options.variants if variants is None else variants
    if count <= 0:
        raise ValueError("variants must be positive")
    rows: list[JevSample] = []
    for case in product_set.cases:
        held_out = case.frame_id in product_set.benchmark_ids
        if split == "train" and held_out or split == "benchmark" and not held_out:
            continue
        route = route_from_frame(case.frame)
        for index, text in enumerate(_variants(case.frame, variants=count, naming=product_set.options.naming)):
            rows.append(
                JevSample(
                    query=text,
                    confidence=0.67 if count > 1 else 1.0,
                    route=route,
                    id=f"{case.frame_id}_{index}",
                    source="slot-product-v1",
                    metadata={
                        "semantic_group_id": f"frame-product:{case.frame_id}",
                        "frame": case.frame.to_dict(),
                        "frame_id": case.frame_id,
                        "route_rule": _route_rule(case.frame),
                        "boundary": True,
                        "label_note": "route derived from the frame by frames.route_from_frame, not asserted",
                        "product_split": "benchmark" if held_out else "train",
                        "generator": "minijev.frame_product",
                    },
                )
            )
    return rows


def _route_rule(frame: RouteFrame) -> str:
    # Reuse the canonical explanation without exposing another route policy.
    from .frames import explain_frame

    return explain_frame(frame)


_SHORTCUT_MARKERS: tuple[str, ...] = ("先", "再", "然后", "最后", "并", "接着")
"""Surface cues a content-free classifier can read. Defined once so the audit and any future
attempt to reduce the shortcut work from the same list."""


def object_of(sample: JevSample) -> str | None:
    """The held-out entity of a generated row, or `None` if it has no frame.

    Rows store the frame, not a flat object field. A leakage check that read `metadata["object"]`
    found nothing, compared two empty sets and reported success -- a check that cannot fail. The
    accessor exists so the entity is read from the one place it actually lives.
    """
    frame = sample.metadata.get("frame")
    return frame.get("object") if isinstance(frame, dict) else None


def entity_overlap(train_rows: Sequence[JevSample], benchmark_rows: Sequence[JevSample]) -> dict[str, Any]:
    """Objects and exact query texts appearing on both sides of the split.

    Both must be zero: the design holds out objects so the benchmark asks about entities the
    model never trained on, and identical text across the split would be a straight duplicate.
    """
    train_objects = {name for name in (object_of(row) for row in train_rows) if name}
    benchmark_objects = {name for name in (object_of(row) for row in benchmark_rows) if name}
    train_text = {row.query for row in train_rows}
    benchmark_text = {row.query for row in benchmark_rows}
    return {
        "train_objects": len(train_objects),
        "benchmark_objects": len(benchmark_objects),
        "shared_objects": sorted(train_objects & benchmark_objects),
        "shared_text": len(train_text & benchmark_text),
    }


def shortcut_audit(train_rows: Sequence[JevSample], benchmark_rows: Sequence[JevSample]) -> dict[str, Any]:
    """How well a classifier does that never reads the *content* of the request.

    Any benchmark is partly solvable by surface statistics, and the defence is to measure the
    ceiling they impose and report it beside every result. An earlier generator was solved
    outright by one such rule: a marker-only classifier reached **1.0000** on the explicit split
    because `naming="explicit"` named the slots in the text. The benchmark looked perfect and
    measured nothing.

    Rules are fitted on the training split and scored on the benchmark, which is how someone
    would actually build them. For the implied split the marker rule is not purely spurious -- a
    clause like `然后核对依据` genuinely signals a multi-step request, which genuinely implies
    `cloud` -- so it is a legitimate feature that happens to be cheap. What matters is the gap
    over the majority baseline, and it is reported as `shortcut_gap_over_majority`.
    """
    if not train_rows or not benchmark_rows:
        return {"rows": len(benchmark_rows)}

    def fit_and_score(extract) -> float:
        table: dict[tuple, Counter] = defaultdict(Counter)
        for row in train_rows:
            table[extract(row.query)][row.route.value] += 1
        correct = 0
        for row in benchmark_rows:
            counts = table.get(extract(row.query))
            guess = counts.most_common(1)[0][0] if counts else Route.LOCAL.value
            correct += guess == row.route.value
        return correct / len(benchmark_rows)

    def marker_features(query: str) -> tuple:
        return tuple(marker in query for marker in _SHORTCUT_MARKERS)

    def coarse_features(query: str) -> tuple:
        return (sum(marker in query for marker in _SHORTCUT_MARKERS), min(4, len(query) // 12))

    prior = Counter(row.route.value for row in benchmark_rows).most_common(1)[0]
    majority = prior[1] / len(benchmark_rows)
    coarse = fit_and_score(coarse_features)
    return {
        "rows": len(benchmark_rows),
        "majority_route": prior[0],
        "majority_accuracy": majority,
        "marker_only_accuracy": fit_and_score(marker_features),
        "marker_plus_length_accuracy": coarse,
        "chance_accuracy": 1.0 / len(ROUTE_ORDER),
        "shortcut_gap_over_majority": coarse - majority,
    }


def product_audit(product_set: FrameProduct) -> dict[str, Any]:
    """Return counts that make product balance and split integrity inspectable."""

    all_cases = product_set.cases
    benchmark_ids = product_set.benchmark_ids
    route_counts = Counter(route_from_frame(case.frame).value for case in all_cases)
    benchmark_counts = Counter(route_from_frame(case.frame).value for case in product_set.benchmark_cases)
    train_counts = Counter(route_from_frame(case.frame).value for case in product_set.train_cases)
    rows = product_rows(product_set)
    train_rows = product_rows(product_set, split="train")
    benchmark_rows = product_rows(product_set, split="benchmark")
    return {
        "frames": len(all_cases),
        "train_frames": len(product_set.train_cases),
        "benchmark_frames": len(product_set.benchmark_cases),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "benchmark_rows": len(benchmark_rows),
        "route_counts": {route.value: route_counts.get(route.value, 0) for route in ROUTE_ORDER},
        "train_route_counts": {route.value: train_counts.get(route.value, 0) for route in ROUTE_ORDER},
        "benchmark_route_counts": {route.value: benchmark_counts.get(route.value, 0) for route in ROUTE_ORDER},
        "source_counts": dict(Counter(case.frame.source.value for case in all_cases)),
        "derivation_counts": dict(Counter(case.frame.derivation.value for case in all_cases)),
        "risk_counts": dict(Counter(case.frame.risk.value for case in all_cases)),
        "steps_counts": dict(Counter(case.frame.steps for case in all_cases)),
        "underspecified_frames": sum(case.frame.underspecified for case in all_cases),
        "unique_queries": len({row.query for row in rows}),
        "minimal_pairs": len(product_contrast_pairs(product_set)),
        "benchmark_ids": sorted(benchmark_ids),
    }


def write_product(output_dir: str | Path, options: ProductOptions | None = None) -> dict[str, Any]:
    """Write train, benchmark and manifest files and return the manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    product_set = generate_frame_product(options)
    train_rows = product_rows(product_set, split="train")
    benchmark_rows = product_rows(product_set, split="benchmark")
    save_jsonl(train_rows, destination / "frame_product_train.jsonl")
    save_jsonl(benchmark_rows, destination / "frame_product_benchmark.jsonl")
    manifest = {
        "version": "slot-product-v1",
        "options": vars(product_set.options),
        "audit": product_audit(product_set),
        # Written into the manifest so a leak or a shortcut shows up in the artifact itself,
        # rather than in an ad-hoc script someone has to remember to run.
        "entity_overlap": entity_overlap(train_rows, benchmark_rows),
        "shortcut": shortcut_audit(train_rows, benchmark_rows),
    }
    (destination / "frame_product_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.frame_product", description="Generate a balanced slot cross-product")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--target-per-route", type=int, default=ProductOptions.target_per_route)
    parser.add_argument("--benchmark-fraction", type=float, default=ProductOptions.benchmark_fraction)
    parser.add_argument("--variants", type=int, default=ProductOptions.variants)
    parser.add_argument("--seed", type=int, default=ProductOptions.seed)
    parser.add_argument(
        "--naming",
        choices=("explicit", "implied"),
        default=ProductOptions.naming,
        help="explicit states each deciding slot in the text (a marker lookup); implied never does",
    )
    parser.add_argument("--audit", action="store_true", help="print the manifest without writing files")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    options = ProductOptions(args.target_per_route, args.benchmark_fraction, args.variants, args.seed, args.naming)
    product_set = generate_frame_product(options)
    manifest = {"version": "slot-product-v1", "options": vars(options), "audit": product_audit(product_set)}
    if args.audit:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    written = write_product(args.output_dir, options)
    print(json.dumps({key: value for key, value in written["audit"].items() if key != "benchmark_ids"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
