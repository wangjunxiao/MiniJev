"""Context pairs: the *same* request routing differently because the state differs.

The complaint this answers
--------------------------
Every other benchmark in this project scores single sentences with no state. That is a
real limitation: a router in practice decides inside a conversation, and much of what
determines the route is established earlier -- whether the document is already in the
prompt, whether the money is real, whether a second step was asked for.

The instrument
--------------
A **context pair** is two cases with an *identical query* whose preceding turns differ,
and whose routes therefore differ::

    query:  这两种能不能一起用
    ctx A:  我在整理两种清洁剂的成分表。          -> local   (given text, bounded)
    ctx B:  我母亲在吃降压药，医生又开了安眠药。   -> review  (high stakes)

A model that ignores context **must** answer both halves identically, so it collapses
every pair -- 100% collapse is the signature of an encoder that cannot see state. This is
the sharpest available probe of context use, because the query text is held fixed and
only the state varies.

Which slot the context supplies
-------------------------------
Each half declares a full frame, so the pair records *why* the routes differ and the
`deciding_slots_differing` machinery can name the slot. Coverage is deliberate: the six
unordered route combinations each get two pairs, so every route appears in six pairs and
the benchmark cannot favour one.

Sizing, stated honestly
-----------------------
The held-out probe is 6 pairs / 36 rows. That is sized to detect *total* collapse, which
is the failure mode worth catching first; it is not sized to measure a two-point
difference between encoders. `CONTEXT_PROBE_PAIRS` is the held-out set and
`CONTEXT_TRAIN_PAIRS` the rest.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frames import Derivation, Risk, RouteFrame, Source, explain_frame, route_from_frame
from .schema import ROUTE_ORDER, JevSample, Route, save_jsonl

Realization = tuple[tuple[str, ...], str]
"""One `(preceding_turns, query)` phrasing of a case."""


@dataclass(frozen=True)
class ContextCase:
    """One half of a context pair: a state plus the request made in it."""

    case_id: str
    frame: RouteFrame
    realizations: tuple[Realization, ...]

    @property
    def route(self) -> Route:
        return route_from_frame(self.frame)


@dataclass(frozen=True)
class ContextPair:
    """Two cases with the same request and different state."""

    pair_id: str
    first: ContextCase
    second: ContextCase

    def __post_init__(self) -> None:
        queries_first = {query for _, query in self.first.realizations}
        queries_second = {query for _, query in self.second.realizations}
        if queries_first != queries_second:
            raise ValueError(f"pair {self.pair_id}: both halves must ask the same question")
        if self.first.route is self.second.route:
            raise ValueError(f"pair {self.pair_id}: both halves route to {self.first.route.value}, so it is not a contrast")


def _pair(pair_id: str, first: ContextCase, second: ContextCase) -> ContextPair:
    return ContextPair(pair_id, first, second)


# Each entry: (pair_id, first case, second case). The comment names the two routes and
# the slot the context supplies.
CONTEXT_PAIRS: tuple[ContextPair, ...] = (
    _pair(
        "ctx_source_manual",
        ContextCase(
            "ctx_source_manual_a",
            RouteFrame("整理", "安装步骤", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="product_manual"),
            (
                (("我已经把产品手册贴给你了。",), "帮我把安装步骤整理出来"),
                (("手册内容在下面：第一步断电，第二步拆机。",), "把安装步骤整理一下"),
            ),
        ),
        ContextCase(
            "ctx_source_manual_b",
            RouteFrame("整理", "安装步骤", Source.DOCUMENT, Derivation.EXTRACT, domain="product_manual"),
            (
                ((), "帮我把安装步骤整理出来"),
                (("我想了解这台设备的安装流程。",), "把安装步骤整理一下"),
            ),
        ),
    ),
    _pair(
        "ctx_source_bill",
        ContextCase(
            "ctx_source_bill_a",
            RouteFrame("核对", "账单里的金额", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="calculation"),
            (
                (("账单如下：三月支出 1200 元，四月支出 1350 元。",), "把里面的金额对一遍"),
                (("数字都在这里：一期 800 元，二期 950 元。",), "核对一下里面的金额"),
            ),
        ),
        ContextCase(
            "ctx_source_bill_b",
            RouteFrame("核对", "账单里的金额", Source.USER_DATA, Derivation.EXTRACT, domain="enterprise_knowledge"),
            (
                (("账单在财务系统里，我这边看不到。",), "把里面的金额对一遍"),
                (("这些记录存在公司的报销平台。",), "核对一下里面的金额"),
            ),
        ),
    ),
    _pair(
        "ctx_risk_cleaner",
        ContextCase(
            "ctx_risk_cleaner_a",
            RouteFrame("判断", "清洁剂能否混用", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="format_conversion"),
            (
                (("我在整理两种清洁剂的成分表。",), "这两种能不能一起用"),
                (("手边有两瓶洗涤剂，成分写在标签上。",), "两种能混着用吗"),
            ),
        ),
        ContextCase(
            "ctx_risk_cleaner_b",
            RouteFrame("判断", "两种药物能否同服", Source.NONE, Derivation.CONSTRUCT, risk=Risk.HIGH, domain="medical"),
            (
                (("我母亲在吃降压药，医生又开了安眠药。",), "这两种能不能一起用"),
                (("家里老人同时在服两种处方药。",), "两种能混着用吗"),
            ),
        ),
    ),
    _pair(
        "ctx_risk_investment",
        ContextCase(
            "ctx_risk_investment_a",
            RouteFrame("计算", "课堂案例的回报率", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="calculation"),
            (
                (("这是课堂模拟案例：投入 100 万，三年回款 130 万。",), "帮我算一下这笔投入的回报率"),
                (("练习题数据：本金 50 万，两年后收回 62 万。",), "这笔投入的回报率是多少"),
            ),
        ),
        ContextCase(
            "ctx_risk_investment_b",
            RouteFrame("判断", "是否投入全部积蓄", Source.NONE, Derivation.CONSTRUCT, risk=Risk.HIGH, domain="financial"),
            (
                (("这是我准备拿出的全部积蓄，对方说年化 12%。",), "帮我算一下这笔投入的回报率"),
                (("我打算把养老钱投进去，对方承诺两年翻倍。",), "这笔投入的回报率是多少"),
            ),
        ),
    ),
    _pair(
        "ctx_steps_interview",
        ContextCase(
            "ctx_steps_interview_a",
            RouteFrame("整理", "访谈记录为表格", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="format_conversion"),
            (
                (("这是本次访谈的原始记录。",), "把这段记录整理成表格"),
                (("下面是一份访谈的文字稿。",), "把记录整理成一份表格"),
            ),
        ),
        ContextCase(
            "ctx_steps_interview_b",
            RouteFrame("整理", "访谈记录为表格并归纳诉求", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=2, risk=Risk.MEDIUM, domain="planning"),
            (
                (("这是三份访谈记录，我还想知道受访者有哪些主要诉求。",), "把这段记录整理成表格"),
                (("三份访谈稿都在这里，另外需要总结出共性问题。",), "把记录整理成一份表格"),
            ),
        ),
    ),
    _pair(
        "ctx_steps_cost",
        ContextCase(
            "ctx_steps_cost_a",
            RouteFrame("列出", "A 方案的成本明细", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="calculation"),
            (
                (("这是 A 方案的成本明细。",), "把成本列出来"),
                (("下面是甲方案的各项花费。",), "把里面的成本列一下"),
            ),
        ),
        ContextCase(
            "ctx_steps_cost_b",
            RouteFrame("比较", "两个方案的成本并判断更优", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=2, risk=Risk.MEDIUM, domain="planning"),
            (
                (("这是 A 方案和 B 方案的成本明细，我要知道哪个更划算。",), "把成本列出来"),
                (("甲乙两套方案的花费都在这里，需要判断哪套更省。",), "把里面的成本列一下"),
            ),
        ),
    ),
    _pair(
        "ctx_risk_link",
        ContextCase(
            "ctx_risk_link_a",
            RouteFrame("打开", "公司内部文档链接", Source.SERVICE, Derivation.EXTRACT, domain="product_manual"),
            (
                (("同事发来一个公司内部的文档链接。",), "看看这个链接能不能点"),
                (("团队群里分享了内部 Wiki 的地址。",), "这个链接能打开吗"),
            ),
        ),
        ContextCase(
            "ctx_risk_link_b",
            RouteFrame("判断", "陌生短信链接的风险", Source.NONE, Derivation.CONSTRUCT, risk=Risk.HIGH, domain="security"),
            (
                (("我收到一条陌生短信里的链接，说我的账户异常。",), "看看这个链接能不能点"),
                (("短信说我积分即将过期，附了一个网址。",), "这个链接能打开吗"),
            ),
        ),
    ),
    _pair(
        "ctx_risk_medicine_info",
        ContextCase(
            "ctx_risk_medicine_info_a",
            RouteFrame("查", "药物公开信息", Source.DOCUMENT, Derivation.EXTRACT, domain="medical"),
            (
                (("我在写一篇关于常见药物的科普稿。",), "查一下这个药的信息"),
                (("需要整理一份公众用药科普材料。",), "这个药的资料查一下"),
            ),
        ),
        ContextCase(
            "ctx_risk_medicine_info_b",
            RouteFrame("判断", "是否调整用药剂量", Source.NONE, Derivation.CONSTRUCT, risk=Risk.HIGH, domain="medical"),
            (
                (("我母亲正在吃这个药，我想知道要不要调整剂量。",), "查一下这个药的信息"),
                (("家人服药后反应明显，我考虑增减用量。",), "这个药的资料查一下"),
            ),
        ),
    ),
    _pair(
        "ctx_derive_error",
        ContextCase(
            "ctx_derive_error_a",
            RouteFrame("定位", "报错对应的文档章节", Source.DOCUMENT, Derivation.EXTRACT, domain="tech_documentation"),
            (
                (("报错信息是 500，官方文档里有一节专门讲这个。",), "看看这个问题出在哪"),
                (("异常码已经查到，对应手册的第四章。",), "问题出在哪里"),
            ),
        ),
        ContextCase(
            "ctx_derive_error_b",
            RouteFrame("分析", "跨服务偶发超时的根因", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=3, risk=Risk.MEDIUM, domain="code_analysis"),
            (
                (("系统上线后偶发超时，日志分散在三个服务里。",), "看看这个问题出在哪"),
                (("三个微服务都有零星延迟，没有现成结论。",), "问题出在哪里"),
            ),
        ),
    ),
    _pair(
        "ctx_derive_policy",
        ContextCase(
            "ctx_derive_policy_a",
            RouteFrame("查", "知识库里的差旅标准", Source.DOCUMENT, Derivation.EXTRACT, domain="enterprise_knowledge"),
            (
                (("公司知识库里有一份差旅标准。",), "整理一下相关信息"),
                (("内部制度文档里有明确的差旅规定。",), "把相关信息整理一下"),
            ),
        ),
        ContextCase(
            "ctx_derive_policy_b",
            RouteFrame("综合", "三个部门口径的统一说明", Source.USER_DATA, Derivation.CONSTRUCT, steps=3, risk=Risk.MEDIUM, domain="planning"),
            (
                (("我手上有三个部门的报销口径，需要出一份统一说明。",), "整理一下相关信息"),
                (("各部门规定互相冲突，要合并成一份新制度。",), "把相关信息整理一下"),
            ),
        ),
    ),
    _pair(
        "ctx_risk_feasibility",
        ContextCase(
            "ctx_risk_feasibility_a",
            RouteFrame("分析", "课程假设方案的可行性", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=2, risk=Risk.MEDIUM, domain="planning"),
            (
                (("这是课程作业里的一组假设方案。",), "分析一下这个方案的可行性"),
                (("课堂案例给出了一个虚构项目计划。",), "这个方案可行吗"),
            ),
        ),
        ContextCase(
            "ctx_risk_feasibility_b",
            RouteFrame("判断", "抵押房产项目的可行性", Source.NONE, Derivation.CONSTRUCT, steps=2, risk=Risk.HIGH, domain="financial"),
            (
                (("这是我准备抵押房子去做的项目。",), "分析一下这个方案的可行性"),
                (("我打算用房产抵押来启动这个生意。",), "这个方案可行吗"),
            ),
        ),
    ),
    _pair(
        "ctx_risk_ethics",
        ContextCase(
            "ctx_risk_ethics_a",
            RouteFrame("评估", "缓存方案的性能影响", Source.GIVEN_TEXT, Derivation.CONSTRUCT, steps=2, risk=Risk.MEDIUM, domain="code_analysis"),
            (
                (("我想优化接口的响应时间，打算加一层缓存。",), "评估一下这样做有什么问题"),
                (("为了降低延迟，准备在网关前面加缓存。",), "这样做有没有问题"),
            ),
        ),
        ContextCase(
            "ctx_risk_ethics_b",
            RouteFrame("判断", "默认勾选同意选项的合规性", Source.NONE, Derivation.CONSTRUCT, steps=2, risk=Risk.HIGH, domain="legal"),
            (
                (("为了提升转化率，我打算默认勾选用户的同意选项。",), "评估一下这样做有什么问题"),
                (("准备把授权选项设为默认同意以提高注册率。",), "这样做有没有问题"),
            ),
        ),
    ),
)

CONTEXT_PROBE_PAIRS: frozenset[str] = frozenset(
    {
        "ctx_source_manual",
        "ctx_risk_cleaner",
        "ctx_steps_interview",
        "ctx_risk_link",
        "ctx_derive_error",
        "ctx_risk_feasibility",
    }
)
"""Held out from training. One pair per route combination, so the probe stays balanced."""

CONTEXT_TRAIN_PAIRS: frozenset[str] = frozenset(pair.pair_id for pair in CONTEXT_PAIRS) - CONTEXT_PROBE_PAIRS


def _rows(pairs: Sequence[ContextPair], source: str) -> list[JevSample]:
    rows: list[JevSample] = []
    for pair in pairs:
        for case in (pair.first, pair.second):
            for index, (turns, query) in enumerate(case.realizations):
                rows.append(
                    JevSample(
                        query=query,
                        confidence=1.0,
                        route=case.route,
                        context=turns,
                        id=f"{case.case_id}_{index}",
                        source=source,
                        metadata={
                            "semantic_group_id": f"context:{case.case_id}",
                            "context_pair": pair.pair_id,
                            "context_turns": len(turns),
                            "frame": case.frame.to_dict(),
                            "frame_id": case.case_id,
                            "route_rule": explain_frame(case.frame),
                            "domain": case.frame.domain,
                            "label_note": "the route depends on the preceding turns, not on the query alone",
                        },
                    )
                )
    return rows


def generate_context_training() -> list[JevSample]:
    """Context rows the training path may read."""
    return _rows([pair for pair in CONTEXT_PAIRS if pair.pair_id in CONTEXT_TRAIN_PAIRS], source="context-train-v1")


def generate_context_probe() -> list[JevSample]:
    """The held-out context probe. The training path must never read this."""
    return _rows([pair for pair in CONTEXT_PAIRS if pair.pair_id in CONTEXT_PROBE_PAIRS], source="context-probe-v1")


def context_audit() -> dict[str, Any]:
    """Pairs, realizations and route coverage, so an unbalanced probe is visible."""
    probe = generate_context_probe()
    counts = {route.value: 0 for route in ROUTE_ORDER}
    for row in probe:
        counts[row.route.value] += 1
    combinations: dict[str, int] = {}
    for pair in CONTEXT_PAIRS:
        key = "|".join(sorted((pair.first.route.value, pair.second.route.value)))
        combinations[key] = combinations.get(key, 0) + 1
    return {
        "pairs": len(CONTEXT_PAIRS),
        "probe_pairs": len(CONTEXT_PROBE_PAIRS),
        "train_pairs": len(CONTEXT_TRAIN_PAIRS),
        "probe_rows": len(probe),
        "probe_route_counts": counts,
        "probe_balanced": len(set(counts.values())) == 1,
        "combinations": combinations,
    }


def context_pairs_in(samples: Sequence[JevSample]) -> dict[str, dict[str, JevSample]]:
    """Group probe rows into `{pair_id: {case_id: row}}` for the paired analysis."""
    grouped: dict[str, dict[str, JevSample]] = {}
    for sample in samples:
        pair_id = sample.metadata.get("context_pair")
        if pair_id is None:
            continue
        grouped.setdefault(str(pair_id), {})[str(sample.metadata["frame_id"])] = sample
    return grouped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.contexts", description="Context-dependent routing pairs")
    parser.add_argument("command", choices=("audit", "probe", "training"))
    parser.add_argument("--output", type=Path, default=Path("data/context_probe.jsonl"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        print(json.dumps(context_audit(), ensure_ascii=False, indent=2))
        return
    rows = generate_context_probe() if args.command == "probe" else generate_context_training()
    save_jsonl(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
