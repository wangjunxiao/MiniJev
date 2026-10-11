"""Deterministic synthetic data generation.

Two things here are deliberate and load-bearing:

* **Every template is a semantic group.** The first version assigned a group id to
  only the 36 paraphrase rows, so 1,000 of 1,036 rows had no group, and the split
  scattered the same template across train and test. Every row now carries
  `semantic_group_id`, so a group-aware split cannot leak.
* **Domain, risk and retrieval flags are per template, not sampled.** The first
  version drew them with `random.choice` from a per-route list, which made `domain`
  statistically independent of the query -- a "计算 12 加 34" row could be tagged
  `translation`. They are now properties of the template and mean something.

Output queries are still template-derived. They exist to make the label space
inspectable, not to represent real traffic.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import JevSample, Route, save_jsonl

Template = tuple[str, dict[str, list[Any]], str, str, bool]
"""`(text, slot_values, domain, risk_level, requires_retrieval)`."""


def _slot(template: str, params: dict[str, list[Any]]) -> list[str]:
    """Expand a template over its slot values."""
    if not params:
        return [template]
    keys = list(params)
    if len(keys) == 1:
        return [template.format(**{keys[0]: value}) for value in params[keys[0]]]
    if len(keys) == 2:
        first, second = keys
        return [template.format(**{first: a, second: b}) for a in params[first] for b in params[second]]
    raise ValueError(f"templates support at most two slots, got {len(keys)}")


LOCAL_TEMPLATES: list[Template] = [
    ("计算 {a} 加 {b}", {"a": [12, 45, 123, 789], "b": [34, 67, 456, 234]}, "calculation", "low", False),
    ("把 {a} 乘以 {b}", {"a": [5, 12, 25, 100], "b": [3, 8, 4, 10]}, "calculation", "low", False),
    ("{a} 减去 {b} 等于多少", {"a": [100, 500, 1000], "b": [23, 156, 387]}, "calculation", "low", False),
    ("把 {num} 公斤转换成克", {"num": [2, 5, 10, 25]}, "calculation", "low", False),
    ("{temp} 摄氏度等于多少华氏度", {"temp": [0, 25, 37, 100]}, "calculation", "low", False),
    ("{length} 英寸是多少厘米", {"length": [1, 10, 24, 50]}, "calculation", "low", False),
    ("把 '{text}' 翻译成中文", {"text": ["hello", "thank you", "good morning", "goodbye"]}, "translation", "low", False),
    ("'{text}' 的英文怎么说", {"text": ["你好", "谢谢", "再见", "对不起"]}, "translation", "low", False),
    ("翻译：{text}", {"text": ["bonjour", "merci", "au revoir", "s'il vous plaît"]}, "translation", "low", False),
    ("把这个日期 {date} 转成 YYYY-MM-DD 格式", {"date": ["2024/03/15", "15-03-2024", "Mar 15, 2024"]}, "format_conversion", "low", False),
    ("将 {text} 转换成大写", {"text": ["hello world", "python code", "machine learning"]}, "format_conversion", "low", False),
    ("把这串数字 {nums} 用逗号分隔", {"nums": ["123456", "987654321", "1122334455"]}, "format_conversion", "low", False),
    ("你好", {}, "simple_qa", "low", False),
    ("早上好", {}, "simple_qa", "low", False),
    ("晚安", {}, "simple_qa", "low", False),
    ("在吗", {}, "simple_qa", "low", False),
    ("帮我一个忙", {}, "simple_qa", "low", False),
    ("今天星期几", {}, "simple_qa", "low", False),
    ("现在几点了", {}, "simple_qa", "low", False),
    ("1公里等于多少米", {}, "simple_qa", "low", False),
    ("1小时有多少秒", {}, "simple_qa", "low", False),
    ("圆周率是多少", {}, "simple_qa", "low", False),
]

RAG_TEMPLATES: list[Template] = [
    ("根据公司报销制度判断这笔 {expense} 费用能否报销", {"expense": ["差旅", "餐饮", "办公用品", "培训"]}, "enterprise_knowledge", "low", True),
    ("查一下我们公司的{policy}政策", {"policy": ["请假", "加班", "绩效考核", "晋升"]}, "enterprise_knowledge", "low", True),
    ("从知识库里找{topic}的说明", {"topic": ["产品使用手册", "故障排查流程", "安全规范", "开发规范"]}, "enterprise_knowledge", "low", True),
    ("搜索项目文档里的{keyword}相关内容", {"keyword": ["部署", "配置", "API接口", "数据库"]}, "tech_documentation", "low", True),
    ("查找 {framework} 的安装文档", {"framework": ["Docker", "Kubernetes", "TensorFlow", "Redis"]}, "tech_documentation", "low", True),
    ("{api} 这个API的参数说明在哪", {"api": ["getUserInfo", "sendMessage", "createOrder", "updateStatus"]}, "tech_documentation", "low", True),
    ("项目里 {module} 模块的设计文档", {"module": ["认证", "支付", "消息推送", "日志"]}, "tech_documentation", "low", True),
    ("根据产品手册，{feature} 功能怎么用", {"feature": ["导出报表", "批量导入", "权限设置", "数据备份"]}, "product_manual", "low", True),
    ("查询 {product} 的技术规格", {"product": ["笔记本电脑", "服务器", "路由器", "打印机"]}, "product_manual", "low", True),
    ("这个错误码 {code} 是什么意思", {"code": ["E001", "ERR_NETWORK", "TIMEOUT", "ACCESS_DENIED"]}, "product_manual", "low", True),
    ("搜索最新的 {topic} 新闻", {"topic": ["人工智能", "量子计算", "新能源", "航天"]}, "realtime_info", "medium", True),
    ("查一下 {company} 的股价", {"company": ["特斯拉", "苹果", "微软", "英伟达"]}, "realtime_info", "medium", True),
    ("今天{city}的天气怎么样", {"city": ["北京", "上海", "深圳", "纽约"]}, "realtime_info", "medium", True),
]

CLOUD_TEMPLATES: list[Template] = [
    ("设计一个支持 {feature} 的分布式系统", {"feature": ["容错", "高并发", "弹性伸缩", "异地多活"]}, "system_design", "medium", False),
    (
        "如何架构一个 {scale} 的{system}",
        {"scale": ["百万级", "千万级", "亿级"], "system": ["用户系统", "消息系统", "支付系统"]},
        "system_design",
        "medium",
        False,
    ),
    ("给我一个 {type} 的技术方案", {"type": ["微服务拆分", "缓存优化", "数据库分库分表", "监控告警"]}, "system_design", "medium", False),
    ("分析这段代码的{problem}问题并给出重构方案", {"problem": ["并发", "性能", "内存泄漏", "可维护性"]}, "code_analysis", "medium", False),
    ("解释这个{algorithm}算法的时间复杂度和优化思路", {"algorithm": ["排序", "搜索", "动态规划", "图遍历"]}, "code_analysis", "medium", False),
    ("review这段代码，找出潜在的{issue}", {"issue": ["安全漏洞", "性能瓶颈", "逻辑错误", "代码异味"]}, "code_analysis", "medium", False),
    ("阅读这份代码仓库，梳理整体架构和模块依赖", {}, "code_analysis", "medium", False),
    (
        "制定一个 {duration} 的{plan}",
        {"duration": ["季度", "半年", "年度"], "plan": ["技术规划", "团队发展", "产品路线图"]},
        "planning",
        "medium",
        False,
    ),
    (
        "比较 {optionA} 和 {optionB} 两种方案的优劣",
        {"optionA": ["MySQL", "PostgreSQL", "MongoDB"], "optionB": ["Redis", "Memcached", "DynamoDB"]},
        "planning",
        "medium",
        False,
    ),
    ("分析 {project} 项目的技术债务和改进建议", {"project": ["遗留系统", "老旧代码库", "重构项目"]}, "planning", "medium", False),
    ("对比分析这几个技术方案，给出推荐理由", {}, "planning", "medium", False),
    ("写一篇关于 {topic} 的技术博客", {"topic": ["机器学习入门", "分布式事务", "容器化部署", "前端性能优化"]}, "creative_writing", "low", False),
    ("帮我起草一份 {doc} 的文档大纲", {"doc": ["技术白皮书", "项目提案", "架构设计", "需求说明"]}, "creative_writing", "low", False),
    ("生成一个 {type} 的 PPT 内容框架", {"type": ["技术分享", "产品发布会", "年度总结", "团队建设"]}, "creative_writing", "low", False),
    ("总结这 {num} 篇论文的核心观点和异同", {"num": [3, 5, 10]}, "creative_writing", "low", True),
]

REVIEW_TEMPLATES: list[Template] = [
    ("我 {symptom}，应该服用什么药", {"symptom": ["头痛", "胸痛", "发烧", "咳嗽", "腹痛"]}, "medical", "high", False),
    (
        "{condition} 能不能和 {med} 一起吃",
        {"condition": ["高血压药", "降糖药", "抗生素"], "med": ["感冒药", "止痛药", "维生素"]},
        "medical",
        "high",
        True,
    ),
    ("我的{indicator}是{value}，正常吗", {"indicator": ["血压", "血糖", "心率"], "value": ["150/95", "6.5", "110"]}, "medical", "high", False),
    ("这个症状需要去医院吗：{symptom}", {"symptom": ["持续高烧3天", "胸闷气短", "剧烈头痛", "呕吐不止"]}, "medical", "high", False),
    ("判断这个决定是否会带来法律风险：{action}", {"action": ["解雇员工", "单方面修改合同", "终止合作", "公开客户信息"]}, "legal", "high", True),
    ("这种情况下我能起诉吗：{case}", {"case": ["商家拒绝退货", "房东不退押金", "公司拖欠工资"]}, "legal", "high", True),
    ("{contract} 合同的第{clause}条是否有效", {"contract": ["劳动", "租赁", "买卖"], "clause": ["3", "5", "8"]}, "legal", "high", True),
    ("我应该买{asset}吗", {"asset": ["这只股票", "比特币", "黄金", "房产"]}, "financial", "high", True),
    ("{amount} 万元投资{product}，风险大吗", {"amount": [10, 50, 100], "product": ["股票", "基金", "P2P", "期货"]}, "financial", "high", True),
    ("现在适合{action}吗", {"action": ["抄底", "清仓", "加仓", "止损"]}, "financial", "high", True),
    ("这个链接 {url} 是否安全", {"url": ["http://suspicious-site.com", "https://unknown-domain.xyz"]}, "security", "high", False),
    ("收到一条{msg}的短信，是诈骗吗", {"msg": ["中奖通知", "账户异常", "包裹待领取", "亲友求助"]}, "security", "high", False),
    ("有人让我提供{info}，该给吗", {"info": ["银行卡号", "身份证照片", "验证码", "密码"]}, "security", "high", False),
    ("帮我处理一下", {}, "ambiguous", "medium", False),
    ("这个怎么办", {}, "ambiguous", "medium", False),
    ("给个建议", {}, "ambiguous", "medium", False),
    ("你觉得呢", {}, "ambiguous", "medium", False),
    ("能行吗", {}, "ambiguous", "medium", False),
]

TEMPLATES: dict[Route, list[Template]] = {
    Route.LOCAL: LOCAL_TEMPLATES,
    Route.RAG: RAG_TEMPLATES,
    Route.CLOUD: CLOUD_TEMPLATES,
    Route.REVIEW: REVIEW_TEMPLATES,
}

ParaphraseGroup = tuple[str, str, str, list[str]]
"""`(domain, risk_level, requires_retrieval_flag, variants)` -- the flag is "yes"/"no"."""

PARAPHRASE_GROUPS: dict[Route, list[ParaphraseGroup]] = {
    Route.LOCAL: [
        ("calculation", "low", "no", ["计算 123 加 456", "123 加上 456 等于多少", "123+456=?", "帮我算一下 123 加 456"]),
        ("translation", "low", "no", ["把 hello 翻译成中文", "hello 的中文是什么", "翻译：hello", "hello 用中文怎么说"]),
        ("simple_qa", "low", "no", ["你好", "hi", "您好", "嗨"]),
    ],
    Route.RAG: [
        ("enterprise_knowledge", "low", "yes", ["查公司的报销规定", "帮我找一下报销制度", "报销政策在哪里", "公司报销流程是什么"]),
        ("tech_documentation", "low", "yes", ["搜索项目文档里的部署命令", "部署相关的文档在哪", "找一下部署流程", "项目怎么部署"]),
    ],
    Route.CLOUD: [
        ("system_design", "medium", "no", ["设计一个分布式系统", "如何架构分布式服务", "分布式系统的技术方案", "给我一个分布式架构设计"]),
        ("code_analysis", "medium", "no", ["分析代码的性能问题", "这段代码性能怎么优化", "找出性能瓶颈", "代码性能分析"]),
    ],
    Route.REVIEW: [
        ("medical", "high", "no", ["我胸痛应该吃什么药", "胸痛吃什么药好", "胸痛了怎么办", "胸痛需要吃药吗"]),
        ("legal", "high", "yes", ["这个决定有法律风险吗", "会不会违法", "法律上有问题吗", "是否合法"]),
    ],
}

# Confidence is the share of annotators agreeing with the final route. The spec
# also defines 0.33 for 1-of-3 agreement; producing it requires a real arbitration
# label, so the generator emits only the two values it can honestly claim.
CONFIDENCE_UNANIMOUS = 1.0
CONFIDENCE_MAJORITY = 0.67

# Which neighbouring route plays the dissenting annotator at 2-of-3 agreement.
# These are simulated votes, not collected ones -- no annotators were involved.
DISSENT_ROUTE: dict[Route, Route] = {
    Route.LOCAL: Route.RAG,
    Route.RAG: Route.CLOUD,
    Route.CLOUD: Route.RAG,
    Route.REVIEW: Route.CLOUD,
}


def simulated_votes(route: Route, confidence: float) -> list[str]:
    """Deterministic 3-vote stand-in consistent with `confidence`."""
    if confidence >= CONFIDENCE_UNANIMOUS:
        return [route.value] * 3
    return [route.value, route.value, DISSENT_ROUTE[route].value]


TOY_TEMPLATES: dict[Route, list[tuple[str, float]]] = {
    Route.LOCAL: [("你好", 0.98), ("把 2 加 2", 0.98), ("把 hello 翻译成中文", 0.92), ("今天星期几", 0.90)],
    Route.RAG: [
        ("查一下产品手册中的安装步骤", 0.78),
        ("根据公司知识库找出报销标准", 0.76),
        ("搜索项目文档里的部署命令", 0.74),
        ("查看这份内部政策的适用范围", 0.72),
    ],
    Route.CLOUD: [
        ("比较这两份方案的风险和成本", 0.58),
        ("设计一个新的分布式训练算法", 0.45),
        ("分析这段代码的并发问题", 0.52),
        ("为这个产品制定完整的市场策略", 0.40),
    ],
    Route.REVIEW: [
        ("我不确定这段内容是否涉及安全问题", 0.20),
        ("判断这个医疗建议是否适合我", 0.18),
        ("处理一项可能有法律风险的决定", 0.16),
        ("这件事我完全不知道该怎么判断", 0.12),
    ],
}


@dataclass(frozen=True)
class SyntheticOptions:
    seed: int = 42
    samples_per_route: int = 250
    include_paraphrases: bool = True
    include_hard_cases: bool = True


def _row(
    query: str,
    route: Route,
    confidence: float,
    group: str,
    domain: str,
    risk: str,
    requires_retrieval: bool,
    *,
    notes: str,
) -> JevSample:
    return JevSample(
        query=query,
        confidence=confidence,
        route=route,
        source="synthetic",
        metadata={
            "semantic_group_id": group,
            "domain": domain,
            "risk_level": risk,
            "requires_retrieval": requires_retrieval,
            "notes": notes,
            "annotator_votes": simulated_votes(route, confidence),
        },
    )


def generate_dataset(options: SyntheticOptions | None = None) -> list[JevSample]:
    """Build the full synthetic dataset, one semantic group per template."""
    options = options or SyntheticOptions()
    samples: list[JevSample] = []
    for route, templates in TEMPLATES.items():
        produced: list[JevSample] = []
        index = 0
        while len(produced) < options.samples_per_route:
            template, params, domain, risk, retrieval = templates[index % len(templates)]
            group = f"{route.value}:t{index % len(templates):02d}"
            for query in _slot(template, params):
                if len(produced) >= options.samples_per_route:
                    break
                produced.append(_row(query, route, CONFIDENCE_UNANIMOUS, group, domain, risk, retrieval, notes="Generated from template"))
            index += 1
        samples.extend(produced)

    if options.include_paraphrases:
        for route, groups in PARAPHRASE_GROUPS.items():
            for group_index, (domain, risk, retrieval_flag, variants) in enumerate(groups):
                group = f"{route.value}:p{group_index:02d}"
                for variant in variants:
                    samples.append(
                        _row(variant, route, CONFIDENCE_MAJORITY, group, domain, risk, retrieval_flag == "yes", notes=f"Paraphrase group {group}")
                    )

    if options.include_hard_cases:
        # Boundary cases moved to `minijev.frames`, where each one carries a
        # structured frame and its route is *derived* rather than asserted. Only the
        # training-eligible frames appear here; the held-out benchmark never does.
        from .frames import generate_hard_cases

        samples.extend(generate_hard_cases(variants=3))

    random.Random(options.seed).shuffle(samples)
    return samples


def generate_toy(per_route: int = 25, seed: int = 0) -> list[JevSample]:
    """Toy data with a meaningful confidence spread and no slot cycling.

    Confidence here carries real information (0.98 down to 0.12), unlike the full
    synthetic set, so this is the dataset to use when exercising the confidence head
    or the review override.
    """
    samples: list[JevSample] = []
    for route, templates in TOY_TEMPLATES.items():
        for index in range(per_route):
            query, confidence = templates[index % len(templates)]
            suffix = "" if index < len(templates) else f"（示例 {index}）"
            samples.append(
                JevSample(
                    query=query + suffix,
                    confidence=confidence,
                    route=route,
                    source="toy",
                    metadata={"semantic_group_id": f"toy:{route.value}:{index % len(templates)}", "notes": "Toy template"},
                )
            )
    random.Random(seed).shuffle(samples)
    return samples


def _command_generate(args: argparse.Namespace) -> None:
    if args.toy:
        samples = generate_toy(args.per_route, args.seed)
    else:
        samples = generate_dataset(SyntheticOptions(seed=args.seed, samples_per_route=args.per_route))
    save_jsonl(samples, args.output)
    groups = {sample.metadata.get("semantic_group_id") for sample in samples}
    print(f"wrote {len(samples)} samples in {len(groups)} semantic groups to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.synthetic", description="Generate deterministic MiniJev data")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic.jsonl"))
    parser.add_argument("--per-route", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--toy", action="store_true", help="emit the small confidence-carrying toy set instead")
    parser.set_defaults(func=_command_generate)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
