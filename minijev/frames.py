"""Structured annotations for route-boundary cases.

A route label is an *assertion*. `(query, "rag")` does not say why the answer is
rag, so it cannot be checked, cannot be varied, and cannot be argued with. This
module replaces the assertion with a **frame**: a small structured description of
the request, whose route is then *derived* by one inspectable rule.

Why not subject-verb-object
---------------------------
The intuition behind asking for 主谓宾 annotation is right -- a query should be
annotated structurally rather than judged holistically -- but 主谓宾 is not the
discriminating structure for routing. Consider the two cases the spec names:

    f01  分析产品手册中的安装步骤         -> rag
    f02  分析这套系统为什么存在并发问题    -> cloud

Both are `[谓] 分析 + [宾] 名词短语`. Identical subject-verb-object structure,
opposite routes. The difference lives in the **adverbial** (where the answer must
come from) and the **complement** (whether the answer already exists or must be
constructed):

    f01  来源=外部手册, 答案=已存在(取回)   -> rag
    f02  来源=给定文本, 答案=需构造(推理)   -> cloud

So the frame carries six slots, two of which are the ones that actually decide the
route:

    predicate        谓语    the operation requested   (分析 / 统计 / 比较 / 判断 ...)
    object           宾语    what it operates on
    source           状语    where the answer must come from
    derivation       补语    does the answer exist in the source, or must it be built
    steps            步数    how many reasoning steps are needed
    risk             情态    how costly a wrong answer is
    underspecified   省略    is the request too vague to act on at all

Deriving the route
------------------
`route_from_frame` is the whole routing policy, stated once::

    risk HIGH            -> review      (cost of being wrong dominates)
    underspecified       -> review      (nothing to act on)
    CONSTRUCT or steps>1 -> cloud       (the answer must be built)
    external source      -> rag         (the answer must be fetched)
    otherwise            -> local       (the answer is already at hand)

Nothing in `FRAME_BANK` stores a route. Routes are computed, so a frame and its
route cannot drift apart, and the rule can be changed in one place and re-derived
over the whole bank.

The benchmark
-------------
Boundary cases are only diagnostic if they are never trained on, and only
measurable if there are enough of them. `BENCHMARK_FRAME_IDS` marks 30 frames as
held out -- 7 to 9 per route, 90 rows -- emitted to a separate file that the
training path never reads. The remaining frames are available to training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .schema import ROUTE_ORDER, JevSample, Route

# ---------------------------------------------------------------------------
# Frame vocabulary
# ---------------------------------------------------------------------------


class Source(str, Enum):
    """Where the answer must come from."""

    GIVEN_TEXT = "given_text"  # the text in the prompt itself
    DEVICE = "device"  # an on-device store or device capability
    USER_DATA = "user_data"  # records belonging to the user, held elsewhere
    DOCUMENT = "document"  # a manual, policy or knowledge base
    SERVICE = "service"  # a live external service (weather, flights, prices)
    NONE = "none"  # no source needed; compute or general knowledge


class Derivation(str, Enum):
    """Whether the answer already exists somewhere, or must be constructed."""

    EXTRACT = "extract"  # read it, compute it, or convert it
    CONSTRUCT = "construct"  # analyse, compare, plan, create or judge


class Risk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


EXTERNAL_SOURCES = frozenset({Source.USER_DATA, Source.DOCUMENT, Source.SERVICE})


@dataclass(frozen=True)
class RouteFrame:
    """The structured annotation of one routing decision."""

    predicate: str
    object: str
    source: Source
    derivation: Derivation
    steps: int = 1
    risk: Risk = Risk.LOW
    underspecified: bool = False
    domain: str = "general"

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must not be negative")
        if not self.predicate.strip() or not self.object.strip():
            raise ValueError("predicate and object must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "object": self.object,
            "source": self.source.value,
            "derivation": self.derivation.value,
            "steps": self.steps,
            "risk": self.risk.value,
            "underspecified": self.underspecified,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "RouteFrame":
        return cls(
            predicate=str(row["predicate"]),
            object=str(row["object"]),
            source=Source(str(row["source"])),
            derivation=Derivation(str(row["derivation"])),
            steps=int(row.get("steps", 1)),
            risk=Risk(str(row.get("risk", "low"))),
            underspecified=bool(row.get("underspecified", False)),
            domain=str(row.get("domain", "general")),
        )


def route_from_frame(frame: RouteFrame) -> Route:
    """The complete routing policy. Every route in the bank comes from here."""
    if frame.risk is Risk.HIGH:
        return Route.REVIEW
    if frame.underspecified:
        return Route.REVIEW
    if frame.derivation is Derivation.CONSTRUCT or frame.steps >= 2:
        return Route.CLOUD
    if frame.source in EXTERNAL_SOURCES:
        return Route.RAG
    return Route.LOCAL


def explain_frame(frame: RouteFrame) -> str:
    """One line saying which slot decided the route. For error analysis output."""
    if frame.risk is Risk.HIGH:
        return "risk=high -> review"
    if frame.underspecified:
        return "underspecified -> review"
    if frame.derivation is Derivation.CONSTRUCT:
        return "derivation=construct -> cloud"
    if frame.steps >= 2:
        return f"steps={frame.steps} -> cloud"
    if frame.source in EXTERNAL_SOURCES:
        return f"source={frame.source.value} -> rag"
    return f"source={frame.source.value}, derivation=extract -> local"


# ---------------------------------------------------------------------------
# The frame bank
# ---------------------------------------------------------------------------

FrameCase = tuple[str, RouteFrame, tuple[str, str]]
"""`(frame_id, frame, (variant_a, variant_b))`."""

VARIANT_SUFFIXES: tuple[str, ...] = ("只回答结论，不要解释。", "简明回答即可。", "不用展开说明。")
"""Extra tail phrasings, used only when more variants than hand-written ones are asked for.

Every frame has two hand-written variants, which carry the real surface variation.
`variants=3` appends `VARIANT_SUFFIXES[0]` to the *first* hand-written variant, so the
extra row is a tail-only variant of an existing phrasing, not an independent one. Tests
pin that down, because counting such rows as independent samples would overstate the
benchmark's effective size.
"""


def _f(
    predicate: str,
    object_: str,
    source: Source,
    derivation: Derivation,
    *,
    steps: int = 1,
    risk: Risk = Risk.LOW,
    underspecified: bool = False,
    domain: str = "general",
) -> RouteFrame:
    return RouteFrame(predicate, object_, source, derivation, steps, risk, underspecified, domain)


FRAME_BANK: tuple[FrameCase, ...] = (
    # --- same 谓宾, different 状语/补语: the pair the whole design exists for ---
    (
        "f01",
        _f("分析", "产品手册中的安装步骤", Source.DOCUMENT, Derivation.EXTRACT, domain="product_manual"),
        ("分析产品手册中的安装步骤", "按产品手册分析一下安装步骤"),
    ),
    (
        "f02",
        _f("分析", "这套系统为什么存在并发问题", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=3, risk=Risk.MEDIUM, domain="code_analysis"),
        ("分析这套系统为什么存在并发问题", "这套系统的并发问题出在哪里"),
    ),
    (
        "f03",
        _f("查", "这个月的报销明细", Source.USER_DATA, Derivation.EXTRACT, domain="enterprise_knowledge"),
        ("查一下这个月的报销明细", "这个月报销了多少"),
    ),
    ("f04", _f("算", "144 的平方根", Source.NONE, Derivation.EXTRACT, domain="calculation"), ("算一下 144 的平方根", "144 的平方根是多少")),
    (
        "f05",
        _f("统计", "这段文字里的汉字数量", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="format_conversion"),
        ("统计这段文字里的汉字数量", "这段文字有多少个汉字"),
    ),
    (
        "f06",
        _f("统计", "公司知识库里关于报销的规定", Source.DOCUMENT, Derivation.EXTRACT, domain="enterprise_knowledge"),
        ("统计公司知识库里关于报销的规定有几条", "知识库里关于报销的规定有多少条"),
    ),
    (
        "f07",
        _f("总结", "这段访谈记录的要点", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="format_conversion"),
        ("总结这段访谈记录的要点", "这段访谈记录的重点是什么"),
    ),
    (
        "f08",
        _f("总结", "这五份周报的共同主题", Source.USER_DATA, Derivation.CONSTRUCT, steps=3, risk=Risk.MEDIUM, domain="planning"),
        ("总结这五份周报的共同主题", "这五份周报讲的是同一件事吗"),
    ),
    (
        "f09",
        _f("比较", "这两个错误码分别代表什么", Source.DOCUMENT, Derivation.EXTRACT, domain="product_manual"),
        ("比较这两个错误码分别代表什么", "这两个错误码的意思一样吗"),
    ),
    (
        "f10",
        _f("比较", "这两套架构方案的优劣", Source.NONE, Derivation.CONSTRUCT, steps=3, risk=Risk.MEDIUM, domain="planning"),
        ("比较这两套架构方案的优劣", "这两套架构方案哪个更合适"),
    ),
    ("f11", _f("翻译", "这句英文", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="translation"), ("翻译这句英文", "这句英文用中文怎么说")),
    (
        "f12",
        _f("翻译", "这份英文合同并标出不利条款", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=2, risk=Risk.MEDIUM, domain="planning"),
        ("翻译这份英文合同并标出对我方不利的条款", "把这份英文合同翻成中文，顺便指出哪些条款不利"),
    ),
    ("f13", _f("改写", "这个标题", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="translation"), ("改写这个标题", "把这个标题换个说法")),
    (
        "f14",
        _f("改写", "这份公告，语气更委婉", Source.GIVEN_TEXT, Derivation.CONSTRUCT, domain="creative_writing"),
        ("改写这份公告，语气更委婉", "这份公告能不能写得客气一点"),
    ),
    (
        "f15",
        _f("核对", "这两个数字是否一致", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="calculation"),
        ("核对这两个数字是否一致", "这两个数字对得上吗"),
    ),
    (
        "f16",
        _f("核对", "这两份合同的条款是否有冲突", Source.USER_DATA, Derivation.CONSTRUCT, steps=3, risk=Risk.MEDIUM, domain="planning"),
        ("核对这两份合同的条款是否有冲突", "这两份合同的条款会不会打架"),
    ),
    # --- writes: on-device store versus an external service ---
    (
        "f17",
        _f("记一笔", "三百元的报销", Source.DEVICE, Derivation.EXTRACT, domain="format_conversion"),
        ("记一笔三百元的报销", "把三百元记到报销里"),
    ),
    (
        "f18",
        _f("提交", "报销申请给财务", Source.SERVICE, Derivation.EXTRACT, domain="enterprise_knowledge"),
        ("帮我把报销申请提交给财务", "把这份报销申请发给财务"),
    ),
    ("f19", _f("设置", "明早七点的提醒", Source.DEVICE, Derivation.EXTRACT, domain="simple_qa"), ("把提醒设到明早七点", "明早七点提醒我一下")),
    (
        "f20",
        _f("群发", "一条变更通知给团队", Source.SERVICE, Derivation.EXTRACT, domain="enterprise_knowledge"),
        ("帮我给团队群发一条变更通知", "把这条变更通知发给团队"),
    ),
    (
        "f21",
        _f("存进", "这条地址到通讯录", Source.DEVICE, Derivation.EXTRACT, domain="simple_qa"),
        ("把这条地址存进通讯录", "这条地址帮我存到通讯录"),
    ),
    (
        "f22",
        _f("更新", "这条地址到客户系统", Source.SERVICE, Derivation.EXTRACT, domain="enterprise_knowledge"),
        ("把这条地址更新到公司客户系统", "客户系统里的这条地址要改一下"),
    ),
    # --- risk boundary: same shape, different stakes ---
    ("f23", _f("调用", "这个函数", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="code_analysis"), ("这个函数怎么调用", "这个函数的调用方式是什么")),
    (
        "f24",
        _f("服用", "这个药一天几次", Source.NONE, Derivation.EXTRACT, risk=Risk.HIGH, domain="medical"),
        ("这个药一天吃几次", "这个药每天吃几次"),
    ),
    ("f25", _f("修", "这条报错", Source.DOCUMENT, Derivation.EXTRACT, domain="tech_documentation"), ("这条报错该怎么修", "这个报错要怎么解决")),
    (
        "f26",
        _f("判断", "我这种情况要不要去医院", Source.NONE, Derivation.EXTRACT, risk=Risk.HIGH, domain="medical"),
        ("我这种情况需要去医院吗", "我这样要不要去医院看看"),
    ),
    (
        "f27",
        _f("检查", "这段代码有没有性能问题", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=2, risk=Risk.MEDIUM, domain="code_analysis"),
        ("这段代码有没有性能问题", "这段代码的性能有问题吗"),
    ),
    (
        "f28",
        _f("判断", "这份体检报告有没有异常", Source.NONE, Derivation.EXTRACT, risk=Risk.HIGH, domain="medical"),
        ("这份体检报告有没有异常", "体检报告上有没有不正常的地方"),
    ),
    (
        "f29",
        _f("比", "这两份报价单哪个更便宜", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="calculation"),
        ("这两份报价单哪个更便宜", "哪份报价单价格低"),
    ),
    (
        "f30",
        _f("决定", "我该不该签这份合同", Source.NONE, Derivation.CONSTRUCT, risk=Risk.HIGH, domain="legal"),
        ("我该不该签这份合同", "这份合同能签吗"),
    ),
    ("f31", _f("打开", "这个 URL", Source.SERVICE, Derivation.EXTRACT, domain="security"), ("这个 URL 能不能打开", "这个网址打得开吗")),
    (
        "f32",
        _f("判断", "点开这个链接会不会泄露信息", Source.NONE, Derivation.CONSTRUCT, risk=Risk.HIGH, domain="security"),
        ("点开这个链接会不会泄露我的信息", "这个链接点了安全吗"),
    ),
    # --- step count: the same task with and without a second step ---
    (
        "f33",
        _f("转成", "这段访谈记录为表格", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="format_conversion"),
        ("把这段访谈记录转成表格", "这段访谈记录整理成表格"),
    ),
    (
        "f34",
        _f("转成", "这段访谈记录为表格并归纳诉求", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=2, risk=Risk.MEDIUM, domain="planning"),
        ("把这段访谈记录转成表格并归纳三类主要诉求", "访谈记录转成表格，再归纳出主要诉求"),
    ),
    (
        "f35",
        _f("列出", "这份文档里的所有章节", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="format_conversion"),
        ("列出这份文档里的所有章节", "这份文档一共有哪些章节"),
    ),
    (
        "f36",
        _f("列出", "这三份文档里互相矛盾的地方", Source.USER_DATA, Derivation.CONSTRUCT, steps=3, risk=Risk.MEDIUM, domain="planning"),
        ("列出这三份文档里互相矛盾的地方", "这三份文档哪里对不上"),
    ),
    (
        "f37",
        _f("加上", "合计行到这张表", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="format_conversion"),
        ("给这张表加上合计行", "这张表加一行合计"),
    ),
    (
        "f38",
        _f("预测", "这张表对应的下季度走势", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=3, risk=Risk.MEDIUM, domain="planning"),
        ("根据这张表预测下季度的走势", "这张表的数据能看出下季度趋势吗"),
    ),
    (
        "f39",
        _f("转成", "会议录音为文字", Source.DEVICE, Derivation.EXTRACT, domain="format_conversion"),
        ("把会议录音转成文字", "这段录音转成文字稿"),
    ),
    (
        "f40",
        _f("整理", "会议决议为任务清单并排优先级", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=3, risk=Risk.MEDIUM, domain="planning"),
        ("把这次会议的决议整理成任务清单并排优先级", "会议决议整理成任务清单，按优先级排好"),
    ),
    # --- underspecified: nothing to act on ---
    (
        "f41",
        _f("处理", "（未指明对象）", Source.NONE, Derivation.EXTRACT, risk=Risk.MEDIUM, underspecified=True, domain="ambiguous"),
        ("帮我把这个办妥", "这事儿你替我张罗一下"),
    ),
    (
        "f42",
        _f("应对", "（未指明对象）", Source.NONE, Derivation.EXTRACT, risk=Risk.MEDIUM, underspecified=True, domain="ambiguous"),
        ("这事我该从哪下手", "这种情况先做哪一步"),
    ),
    (
        "f43",
        _f("建议", "（未指明对象）", Source.NONE, Derivation.EXTRACT, risk=Risk.MEDIUM, underspecified=True, domain="ambiguous"),
        ("有没有什么想法", "你觉得该怎么走下一步"),
    ),
    (
        "f44",
        _f("征询", "（未指明对象）", Source.NONE, Derivation.EXTRACT, risk=Risk.MEDIUM, underspecified=True, domain="ambiguous"),
        ("说说你的看法", "换成你会怎么做"),
    ),
    (
        "f45",
        _f("确认", "（未指明对象）", Source.NONE, Derivation.EXTRACT, risk=Risk.MEDIUM, underspecified=True, domain="ambiguous"),
        ("这么办可以吗", "这条思路走得通吗"),
    ),
    (
        "f46",
        _f("查看", "（未指明对象）", Source.NONE, Derivation.EXTRACT, risk=Risk.MEDIUM, underspecified=True, domain="ambiguous"),
        ("你先瞧一眼", "抽空过一下这个"),
    ),
    (
        "f47",
        _f("确认", "（未指明对象）", Source.NONE, Derivation.EXTRACT, risk=Risk.MEDIUM, underspecified=True, domain="ambiguous"),
        ("那就这么定了吧", "按这个方向推进"),
    ),
    (
        "f48",
        _f("闲聊", "（未指明对象）", Source.NONE, Derivation.EXTRACT, risk=Risk.MEDIUM, underspecified=True, domain="ambiguous"),
        ("聊点别的吧", "想到哪说到哪"),
    ),
    # --- live data versus prediction on the same subject ---
    (
        "f49",
        _f("查", "今天北京的天气", Source.SERVICE, Derivation.EXTRACT, risk=Risk.MEDIUM, domain="realtime_info"),
        ("今天北京天气怎么样", "北京今天什么天气"),
    ),
    (
        "f50",
        _f("查", "明天的航班有没有延误", Source.SERVICE, Derivation.EXTRACT, risk=Risk.MEDIUM, domain="realtime_info"),
        ("帮我看看明天的航班有没有延误", "明天那班飞机会晚点吗"),
    ),
    (
        "f51",
        _f("查", "美元兑人民币汇率", Source.SERVICE, Derivation.EXTRACT, domain="realtime_info"),
        ("现在美元兑人民币是多少", "美元现在换多少人民币"),
    ),
    (
        "f52",
        _f("预测", "下个月的汇率走势", Source.NONE, Derivation.CONSTRUCT, steps=3, risk=Risk.MEDIUM, domain="planning"),
        ("预测一下下个月的汇率走势", "下个月汇率会怎么走"),
    ),
    (
        "f53",
        _f("查", "这家公司的注册地址", Source.SERVICE, Derivation.EXTRACT, domain="realtime_info"),
        ("查一下这家公司的注册地址", "这家公司注册在哪里"),
    ),
    (
        "f54",
        _f("评估", "这家公司值不值得投资", Source.NONE, Derivation.CONSTRUCT, steps=2, risk=Risk.HIGH, domain="financial"),
        ("评估一下这家公司值不值得投资", "这家公司能投吗"),
    ),
    # --- long, multi-clause prompts where the decisive slot is easy to lose ---
    (
        "f55",
        _f("转成", "2024 年 3 月 15 日为中文日期", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="format_conversion"),
        ("把 2024 年 3 月 15 日转成中文日期", "2024 年 3 月 15 日用中文怎么写"),
    ),
    (
        "f56",
        _f("统一", "这份表格的日期列为中文格式", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="format_conversion"),
        ("把这份表格的日期列统一成中文格式", "表格里的日期都改成中文格式"),
    ),
    (
        "f57",
        _f("统一", "日期列格式并统计每月条目数", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=2, risk=Risk.MEDIUM, domain="planning"),
        ("把这份表格的日期列统一成中文格式，并统计每月的条目数", "表格日期改成中文格式，然后统计每个月有多少条"),
    ),
    (
        "f58",
        _f("查找", "产品手册里关于保修的条款", Source.DOCUMENT, Derivation.EXTRACT, domain="product_manual"),
        ("查找产品手册里关于保修的条款", "产品手册上保修是怎么写的"),
    ),
    (
        "f59",
        _f("判断", "这份说明书的免责条款是否合法", Source.NONE, Derivation.CONSTRUCT, risk=Risk.HIGH, domain="legal"),
        ("判断这份说明书的免责条款是否合法", "这个免责条款合法吗"),
    ),
    (
        "f60",
        _f("加上", "类型标注到这段代码", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="code_analysis"),
        ("把这段代码加上类型标注", "这段代码补上类型注解"),
    ),
    (
        "f61",
        _f("重构", "这段代码，消除重复逻辑", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=2, risk=Risk.MEDIUM, domain="code_analysis"),
        ("重构这段代码，消除重复逻辑", "这段代码的重复逻辑可以重构掉吗"),
    ),
    (
        "f62",
        _f("查找", "这段代码里硬编码的密钥", Source.GIVEN_TEXT, Derivation.EXTRACT, risk=Risk.MEDIUM, domain="security"),
        ("这段代码里有没有硬编码的密钥", "代码里是不是写死了密钥"),
    ),
    (
        "f63",
        _f("检查", "这份代码的安全问题", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=2, risk=Risk.MEDIUM, domain="security"),
        ("帮我看看这份代码有没有安全问题", "这份代码安全性怎么样"),
    ),
    (
        "f64",
        _f("列出", "这份合同里的付款节点", Source.USER_DATA, Derivation.EXTRACT, domain="enterprise_knowledge"),
        ("列出这份合同里的付款节点", "这份合同什么时候付款"),
    ),
    # --- minimal pairs isolating a single slot ---
    # Same 谓语 (评估), same 补语 (construct), same 步数 (2). Only 情态 differs,
    # so a model that gets one of these right and the other wrong is failing on
    # exactly the risk slot.
    (
        "f65",
        _f("评估", "这份方案的投入产出比", Source.NONE, Derivation.CONSTRUCT, steps=2, risk=Risk.MEDIUM, domain="planning"),
        ("评估这份方案的投入产出比", "这份方案投入产出比怎么样"),
    ),
    (
        "f66",
        _f("评估", "这份保单值不值得买", Source.NONE, Derivation.CONSTRUCT, steps=2, risk=Risk.HIGH, domain="financial"),
        ("评估这份保单值不值得买", "这份保单该不该买"),
    ),
    # Same 谓语 (抽取), same 宾语 (names in three reports). Only 状语 + 补语 differ,
    # so this pair isolates the fetch-versus-build decision.
    (
        "f67",
        _f("抽取", "这三份周报里的人名", Source.USER_DATA, Derivation.EXTRACT, domain="enterprise_knowledge"),
        ("抽取这三份周报里的人名", "这三份周报提到了哪些人"),
    ),
    (
        "f68",
        _f("抽取", "这三份周报里的人名并推断各自职责", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=2, risk=Risk.MEDIUM, domain="planning"),
        ("抽取这三份周报里的人名并推断各自的职责", "这三份周报里的人各负责什么"),
    ),
    # Kept so the held-out benchmark can be exactly balanced on all four routes.
    (
        "f69",
        _f("查", "供应商合同里的付款条款", Source.DOCUMENT, Derivation.EXTRACT, domain="enterprise_knowledge"),
        ("查一下供应商合同里的付款条款", "供应商合同是怎么约定付款的"),
    ),
    (
        "f70",
        _f("确认", "这条短信是不是诈骗", Source.NONE, Derivation.CONSTRUCT, risk=Risk.HIGH, domain="security"),
        ("帮我确认这条短信是不是诈骗", "这条短信是骗人的吗"),
    ),
    # --- minimal pairs: exactly one deciding slot differs, so a route change is
    # attributable to that slot alone. Without these, a failure could be blamed on any
    # of several slots at once.
    # f71/f72 isolate 步数: same 谓语, same 来源, same 补语, same 情态. Only the
    # second step decides local versus cloud.
    (
        "f71",
        _f("整理", "这段访谈记录为表格", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="format_conversion"),
        ("整理这段访谈记录为表格", "把这段访谈记录整理成一份表格"),
    ),
    (
        "f72",
        _f("整理", "这段访谈记录为表格并归纳诉求", Source.GIVEN_TEXT, Derivation.EXTRACT, steps=2, domain="planning"),
        ("整理这段访谈记录为表格并归纳三类诉求", "把访谈记录整理成表格，再归纳主要诉求"),
    ),
    # f73/f74 isolate 补语: same 谓语, same 来源, same 步数, same 情态. Only
    # "does the answer already exist" decides local versus cloud.
    (
        "f73",
        _f("核对", "这份清单里的两处金额", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="calculation"),
        ("核对这份清单里的两处金额", "清单里这两处金额对得上吗"),
    ),
    (
        "f74",
        _f("核对", "这两份方案的预算差异", Source.GIVEN_TEXT, Derivation.CONSTRUCT, domain="planning"),
        ("核对这两份方案的预算差异", "这两份方案的预算差在哪"),
    ),
    # f75/f76 isolate 省略: identical frame except that the second request never says
    # what to act on. Only the underspecified flag decides local versus review.
    (
        "f75",
        _f("填写", "报销单的日期栏", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="format_conversion"),
        ("填写这份报销单的日期栏", "把报销单的日期栏填上"),
    ),
    (
        "f76",
        _f("填写", "那份单据", Source.GIVEN_TEXT, Derivation.EXTRACT, underspecified=True, domain="ambiguous"),
        ("那份单据你帮我填一下", "那个单子帮我填了"),
    ),
)

FRAME_IDS: tuple[str, ...] = tuple(case[0] for case in FRAME_BANK)

# Held out from training. Chosen as complete pairs, so each benchmark case keeps the
# contrasting partner it exists to be contrasted with. 30 frames x 3 variants = 90
# rows, with 7 to 9 rows per route.
BENCHMARK_FRAME_IDS: frozenset[str] = frozenset(
    {
        "f01",
        "f02",  # 分析 product manual (rag) vs concurrency (cloud)
        "f03",
        "f04",  # 查 user records (rag) vs arithmetic (local)
        "f05",
        "f06",  # 统计 given text (local) vs knowledge base (rag)
        "f07",
        "f08",  # 总结 one recording (local) vs five reports (cloud)
        "f09",
        "f10",  # 比较 error codes (rag) vs architectures (cloud)
        "f11",
        "f12",  # 翻译 a sentence (local) vs contract with clauses (cloud)
        "f15",
        "f16",  # 核对 two numbers (local) vs two contracts (cloud)
        "f17",
        "f18",  # write to device (local) vs submit to finance (rag)
        "f23",
        "f24",  # 这个...怎么 (local) vs 这个药 (review)
        "f25",
        "f26",  # 报错怎么修 (rag) vs 要不要去医院 (review)
        "f27",
        "f28",  # code performance (cloud) vs medical report (review)
        "f29",
        "f30",  # compare quotes (local) vs sign a contract (review)
        "f31",
        "f32",  # open a URL (rag) vs leak data (review)
        "f35",
        "f36",  # list chapters (local) vs find conflicts (cloud)
        "f41",
        "f42",  # underspecified pair (review)
        "f65",
        "f66",  # 评估 plan (cloud) vs 评估 policy (review): the risk slot alone
        "f67",
        "f68",  # 抽取 names from reports (rag) vs infer roles (cloud): source+derivation
        "f69",  # rag, balances the benchmark
        "f70",  # review, balances the benchmark
        "f71",  # minimal pair on 步数
        "f72",
        "f75",  # minimal pair on 省略
        "f76",
        "f49",  # rag singletons, keeping the benchmark balanced at the frame level
        "f58",
        "f73",  # minimal pair on 补语 (derivation)
        "f74",
        "f20",  # rag singleton
        "f59",  # review singleton
    }
)

TRAIN_FRAME_IDS: tuple[str, ...] = tuple(frame_id for frame_id in FRAME_IDS if frame_id not in BENCHMARK_FRAME_IDS)


# Which slots a frame pair differs in. Computed, not hand-labelled, so it cannot drift
# from the bank.
DECIDING_SLOTS: tuple[str, ...] = ("risk", "underspecified", "derivation", "steps", "source_class")


def deciding_slots_differing(left: RouteFrame, right: RouteFrame) -> tuple[str, ...]:
    """Which *rule inputs* two frames disagree on.

    Only these five slots can change a route -- `route_from_frame` reads nothing else.
    `predicate` and `object` are surface realisations, so a pair that differs only in
    wording is not a contrast pair at all. `source` is compared as external-or-not,
    because that is the only way the rule uses it.
    """
    differing: list[str] = []
    if left.risk is not right.risk:
        differing.append("risk")
    if left.underspecified != right.underspecified:
        differing.append("underspecified")
    if left.derivation is not right.derivation:
        differing.append("derivation")
    if left.steps != right.steps:
        differing.append("steps")
    if (left.source in EXTERNAL_SOURCES) != (right.source in EXTERNAL_SOURCES):
        differing.append("source_class")
    return tuple(differing)


def is_minimal_pair(left: RouteFrame, right: RouteFrame) -> bool:
    """Exactly one deciding slot differs, so any route change is attributable to it."""
    return len(deciding_slots_differing(left, right)) == 1


CONTRAST_PAIRS: tuple[tuple[str, str], ...] = (
    ("f01", "f02"),
    ("f03", "f04"),
    ("f05", "f06"),
    ("f07", "f08"),
    ("f09", "f10"),
    ("f11", "f12"),
    ("f15", "f16"),
    ("f17", "f18"),
    ("f23", "f24"),
    ("f25", "f26"),
    ("f27", "f28"),
    ("f29", "f30"),
    ("f31", "f32"),
    ("f35", "f36"),
    ("f65", "f66"),
    ("f67", "f68"),
    ("f71", "f72"),
    ("f73", "f74"),
    ("f75", "f76"),
)
"""Balanced pairs held out together.

Both members must be in the benchmark: a pair only measures separation when the model is
asked both halves. `test_contrast_pairs_are_wholly_held_out` enforces that.
"""


def contrast_pair_table() -> list[dict[str, Any]]:
    """Each pair with the slots that differ and the route on each side."""
    table: list[dict[str, Any]] = []
    for left_id, right_id in CONTRAST_PAIRS:
        left, right = _frame(left_id), _frame(right_id)
        table.append(
            {
                "left": left_id,
                "right": right_id,
                "left_route": route_from_frame(left).value,
                "right_route": route_from_frame(right).value,
                "deciding_slots": deciding_slots_differing(left, right),
                "minimal": is_minimal_pair(left, right),
            }
        )
    return table


def _frame(frame_id: str) -> RouteFrame:
    """The frame for an id. Exposed because tests and analysis both need it."""
    return _case(frame_id)[1]


def _case(frame_id: str) -> FrameCase:
    for case in FRAME_BANK:
        if case[0] == frame_id:
            return case
    raise KeyError(f"unknown frame id {frame_id!r}")


def _texts(frame_id: str, variants: int, variant_suffixes: Sequence[str] | None) -> list[str]:
    _, _, hand_written = _case(frame_id)
    if variants <= 0:
        raise ValueError("variants must be positive")
    texts = list(hand_written[:variants])
    suffixes = tuple(variant_suffixes) if variant_suffixes is not None else VARIANT_SUFFIXES
    extra = 0
    while len(texts) < variants:
        if not suffixes:
            raise ValueError("not enough hand-written variants and no suffixes to extend with")
        texts.append(hand_written[0] + suffixes[extra % len(suffixes)])
        extra += 1
    return texts


def distinct_text_count(variants: int = 3) -> dict[str, int]:
    """Rows, distinct texts and groups -- so a padded benchmark cannot look bigger than it is.

    With `variants=3` the third row of each group is the first phrasing plus a tail
    suffix. It is a real row but not an independent phrasing, so the effective sample
    size sits between `groups` and `rows`.
    """
    rows = generate_benchmark(variants=variants)
    hand_written = sum(min(variants, len(case[2])) for case in FRAME_BANK if case[0] in BENCHMARK_FRAME_IDS)
    return {
        "rows": len(rows),
        "distinct_texts": len({row.query for row in rows}),
        "hand_written_texts": hand_written,
        "groups": len({row.metadata["semantic_group_id"] for row in rows}),
    }


def _rows(frame_ids: Iterable[str], variants: int, source: str, variant_suffixes: Sequence[str] | None = None) -> list[JevSample]:
    rows: list[JevSample] = []
    for frame_id in frame_ids:
        _, frame, _ = _case(frame_id)
        route = route_from_frame(frame)
        for index, text in enumerate(_texts(frame_id, variants, variant_suffixes)):
            rows.append(
                JevSample(
                    query=text,
                    # Boundary cases are contested by construction: the surface form
                    # alone does not determine the route, which is the point of them.
                    confidence=0.67 if variants > 1 else 1.0,
                    route=route,
                    id=f"frame_{frame_id}_{index}",
                    source=source,
                    metadata={
                        "semantic_group_id": f"frame:{frame_id}",
                        "frame": frame.to_dict(),
                        "frame_id": frame_id,
                        "route_rule": explain_frame(frame),
                        "boundary": True,
                        "label_note": "route derived from the frame by frames.route_from_frame, not asserted",
                        # Same top-level keys the template generator emits, so a
                        # consumer that reads `domain` keeps working across both.
                        "domain": frame.domain,
                        "risk_level": frame.risk.value,
                        "requires_retrieval": frame.source in EXTERNAL_SOURCES,
                    },
                )
            )
    return rows


def generate_hard_cases(variants: int = 3) -> list[JevSample]:
    """Training-eligible boundary rows. Never includes held-out benchmark frames."""
    return _rows(TRAIN_FRAME_IDS, variants, source="synthetic-boundary-v2")


def generate_benchmark(variants: int = 3) -> list[JevSample]:
    """The held-out boundary benchmark. The training path must never read this."""
    return _rows(sorted(BENCHMARK_FRAME_IDS), variants, source="hard-benchmark-v1")


def assert_no_text_overlap(rows: Iterable[JevSample], other: Iterable[JevSample]) -> None:
    """Raise if any query text appears on both sides.

    A *surface* collision is as fatal as a group leak: if the benchmark asks a question
    whose exact wording was trained on, the row measures recall, not routing. This
    caught two underspecified frames that had reused ambiguous phrasings verbatim from
    the template bank.
    """
    left = {row.query for row in rows}
    right = {row.query for row in other}
    shared = sorted(left & right)
    if shared:
        raise ValueError(f"{len(shared)} query text(s) appear on both sides, e.g. {shared[:3]}")


def frame_audit() -> dict[str, Any]:
    """Route distribution and slot usage, so an unbalanced bank is visible."""
    routes = Counter(route_from_frame(case[1]).value for case in FRAME_BANK)
    benchmark_routes = Counter(route_from_frame(_case(frame_id)[1]).value for frame_id in BENCHMARK_FRAME_IDS)
    return {
        "frames": len(FRAME_BANK),
        "benchmark_frames": len(BENCHMARK_FRAME_IDS),
        "train_frames": len(TRAIN_FRAME_IDS),
        "benchmark_texts": distinct_text_count(),
        "route_counts_all": {route.value: routes.get(route.value, 0) for route in ROUTE_ORDER},
        "route_counts_benchmark": {route.value: benchmark_routes.get(route.value, 0) for route in ROUTE_ORDER},
        "sources": dict(Counter(case[1].source.value for case in FRAME_BANK)),
        "derivations": dict(Counter(case[1].derivation.value for case in FRAME_BANK)),
        "risks": dict(Counter(case[1].risk.value for case in FRAME_BANK)),
        "underspecified": sum(1 for case in FRAME_BANK if case[1].underspecified),
    }


def load_frame(sample: JevSample) -> RouteFrame | None:
    """Read the frame back out of a sample, if it carries one."""
    payload = sample.metadata.get("frame")
    return RouteFrame.from_dict(payload) if isinstance(payload, Mapping) else None


# ---------------------------------------------------------------------------
# Auxiliary supervision vocabulary
# ---------------------------------------------------------------------------

AUX_SLOTS: tuple[str, ...] = ("source_class", "derivation", "steps", "risk", "underspecified")
"""The deciding slots, as prediction targets.

The route is a deterministic function of these five values. Supervising them directly is
a different bet from supervising the route: instead of asking the encoder to infer "rag"
from surface text, it is asked to say where the answer must come from, whether it already
exists, how many steps it needs and how costly a mistake is. Those questions have dense,
low-cardinality labels.

`predicate` and `object` are deliberately excluded: they are surface realisations, not
decisions, and a head that predicts them would mostly be doing language modelling.
"""


def slot_vocabulary() -> dict[str, tuple[str, ...]]:
    """The value names each auxiliary head can emit, in index order."""
    return {
        "source_class": ("internal", "external"),
        "derivation": ("extract", "construct"),
        "steps": ("1", "2", "3+"),
        "risk": ("low", "medium", "high"),
        "underspecified": ("false", "true"),
    }


def slot_value(frame: RouteFrame, slot: str) -> str:
    """The label of one deciding slot, as a string."""
    if slot == "source_class":
        return "external" if frame.source in EXTERNAL_SOURCES else "internal"
    if slot == "steps":
        return "1" if frame.steps <= 1 else ("2" if frame.steps == 2 else "3+")
    if slot == "underspecified":
        return "true" if frame.underspecified else "false"
    value = getattr(frame, slot)
    return str(getattr(value, "value", value))


def slot_index(frame: RouteFrame, slot: str) -> int:
    """The label of one deciding slot as an index into `slot_vocabulary()`."""
    return slot_vocabulary()[slot].index(slot_value(frame, slot))


def slot_targets(samples: Sequence[JevSample]) -> tuple[dict[str, list[int]], list[float]]:
    """Per-slot label indices plus a mask of which rows carry a frame.

    Template-generated rows have no frame and so contribute no auxiliary signal. The mask
    lets framed and frame-free rows share one corpus instead of forcing a caller to keep
    two datasets.
    """
    columns: dict[str, list[int]] = {slot: [] for slot in AUX_SLOTS}
    mask: list[float] = []
    for sample in samples:
        frame = load_frame(sample)
        if frame is None:
            for slot in AUX_SLOTS:
                columns[slot].append(0)
            mask.append(0.0)
            continue
        for slot in AUX_SLOTS:
            columns[slot].append(slot_index(frame, slot))
        mask.append(1.0)
    return columns, mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.frames", description="Structured boundary frames")
    parser.add_argument("command", choices=("audit", "benchmark", "training"))
    parser.add_argument("--output", type=Path, default=Path("data/hard_benchmark.jsonl"))
    parser.add_argument("--variants", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from .schema import save_jsonl

    args = build_parser().parse_args(argv)
    if args.command == "audit":
        print(json.dumps(frame_audit(), ensure_ascii=False, indent=2))
        return
    rows = generate_benchmark(args.variants) if args.command == "benchmark" else generate_hard_cases(args.variants)
    save_jsonl(rows, args.output)
    counts = Counter(row.route.value for row in rows)
    detail = " ".join(f"{route.value}={counts.get(route.value, 0)}" for route in ROUTE_ORDER)
    print(f"wrote {len(rows)} rows in {len({row.metadata['frame_id'] for row in rows})} frames to {args.output}: {detail}")


if __name__ == "__main__":
    main()
