"""Domain-typed corpora: news, philosophy, literature, academic.

The claim being tested
----------------------
Training on a few broad domains should cover most of the grammar and most of the
situations a router meets, so a decision learned on news and philosophy ought to hold up
on an academic abstract. This module builds those domains.

The constraint that decides whether it works
--------------------------------------------
Merging corpora is only safe when every corpus covers every route. That was measured, not
assumed: MASSIVE's derived rubric contains **zero cloud rows**, and merging it drove cloud
recall from 0.58 to 0.12 for the pooled model and to 0.00 for the transformer, with the
`steps 2+` slot at zero. An absent class is learned as a class that does not exist.

So every domain here covers **all four routes in equal proportion by construction**: each
passage is paired with the same four requests, whose frames differ in the slots that
decide the route.

    local   bounded extraction over the passage itself   source=given_text, extract
    rag     provenance or an external record             source=document, extract
    cloud   multi-step interpretation or comparison      construct, steps>=2
    review  rights, ethics or safety judgement           risk=high

Each passage therefore contributes exactly one row per route, and `domain_audit` reports
the realised proportions so a domain cannot quietly tilt the label prior.

Text provenance
---------------
The literature domain reuses the public-domain classical excerpts in `minijev.literary`.
The other three domains read authored, synthetic passages from
`data/external/domains/passages.json`: no real people, institutions or journals, and no
third-party text. They are training material, not a benchmark.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DATA_DIR
from .frames import Derivation, Risk, RouteFrame, Source, explain_frame, route_from_frame
from .schema import ROUTE_ORDER, JevSample, Route, save_jsonl

PASSAGES_PATH = DATA_DIR / "external" / "domains" / "passages.json"

DOMAINS: tuple[str, ...] = ("news", "philosophy", "literature", "academic")

TaskSpec = tuple[Route, str, str, RouteFrame]
"""`(route, task_id, template with {text}, frame)`."""

VARIANT_SUFFIXES: tuple[str, ...] = ("", "请给出简要结论。", "只回答结论，不要解释。")

TASKS: dict[str, tuple[TaskSpec, ...]] = {
    # News briefs carry numbers, dates and attribution, so extraction and provenance are
    # natural; and a published claim is where a defamation question actually arises.
    "news": (
        (
            Route.LOCAL,
            "news_extract",
            "简讯：{text} 请把其中出现的数字和时间都列出来。",
            RouteFrame("列出", "简讯里的数字和时间", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="news"),
        ),
        (
            Route.RAG,
            "news_provenance",
            "简讯：{text} 请查一下这条简讯的原始发布渠道和发布日期。",
            RouteFrame("查", "这条简讯的原始发布渠道", Source.SERVICE, Derivation.EXTRACT, domain="news"),
        ),
        (
            Route.CLOUD,
            "news_framing",
            "简讯：{text} 请分析这条简讯的叙事框架，并与其他媒体的同类报道比较取舍差异。",
            RouteFrame(
                "分析",
                "叙事框架并与其他报道比较",
                Source.GIVEN_TEXT,
                Derivation.CONSTRUCT,
                steps=2,
                risk=Risk.MEDIUM,
                domain="news",
            ),
        ),
        (
            Route.REVIEW,
            "news_defamation",
            "简讯：{text} 请判断照此表述发布是否构成对当事人的名誉侵害风险。",
            RouteFrame("判断", "是否构成名誉侵害风险", Source.NONE, Derivation.CONSTRUCT, risk=Risk.HIGH, domain="news"),
        ),
    ),
    # Argumentative passages carry a claim, a reason and a concession, so multi-step
    # interpretation bites; and quoting an argument raises a rights question.
    "philosophy": (
        (
            Route.LOCAL,
            "phil_conclusion",
            "论述：{text} 请把这段话的结论原句抄出来。",
            RouteFrame("抄出", "这段话的结论原句", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="philosophy"),
        ),
        (
            Route.RAG,
            "phil_source",
            "论述：{text} 请查一下这段话出自哪本著作以及作者是谁。",
            RouteFrame("查", "这段话的出处在哪本著作", Source.DOCUMENT, Derivation.EXTRACT, domain="philosophy"),
        ),
        (
            Route.CLOUD,
            "phil_premise",
            "论述：{text} 请分析这个论证的隐含前提，并与另一个学派的立场比较。",
            RouteFrame(
                "分析",
                "该论证的隐含前提并比较学派立场",
                Source.GIVEN_TEXT,
                Derivation.CONSTRUCT,
                steps=3,
                risk=Risk.MEDIUM,
                domain="philosophy",
            ),
        ),
        (
            Route.REVIEW,
            "phil_rights",
            "论述：{text} 请判断在教学材料中整段引用这段论述是否存在版权风险。",
            RouteFrame("判断", "整段引用的版权风险", Source.NONE, Derivation.CONSTRUCT, risk=Risk.HIGH, domain="philosophy"),
        ),
    ),
    # Abstracts state a question, a method and a result, so method extraction and
    # comparison are natural; and human-subject research raises an ethics question.
    "academic": (
        (
            Route.LOCAL,
            "acad_keywords",
            "摘要：{text} 请把这段摘要里的研究方法和样本规模摘出来。",
            RouteFrame("摘出", "摘要里的研究方法和样本规模", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="academic"),
        ),
        (
            Route.RAG,
            "acad_publication",
            "摘要：{text} 请查一下这篇论文的收录期刊和发表年份。",
            RouteFrame("查", "这篇论文的收录期刊和发表年份", Source.DOCUMENT, Derivation.EXTRACT, domain="academic"),
        ),
        (
            Route.CLOUD,
            "acad_method",
            "摘要：{text} 请比较这篇研究与同类研究在方法论上的差异，并说明各自适用的条件。",
            RouteFrame(
                "比较",
                "该方法与同类研究的方法论差异",
                Source.GIVEN_TEXT,
                Derivation.CONSTRUCT,
                steps=3,
                risk=Risk.MEDIUM,
                domain="academic",
            ),
        ),
        (
            Route.REVIEW,
            "acad_ethics",
            "摘要：{text} 请判断这份研究设计是否符合受试者知情同意的伦理要求。",
            RouteFrame("判断", "研究设计的伦理合规性", Source.NONE, Derivation.CONSTRUCT, risk=Risk.HIGH, domain="academic"),
        ),
    ),
}


@dataclass(frozen=True)
class Passage:
    """One piece of domain state: the text every task in a group operates on."""

    passage_id: str
    title: str
    text: str
    note: str = ""


def load_passages(path: Path | None = None) -> dict[str, tuple[Passage, ...]]:
    """Read the authored passages, excluding the literature domain."""
    source = path or PASSAGES_PATH
    if not source.exists():
        raise FileNotFoundError(f"no authored passages at {source}; see README for how they are built")
    payload = json.loads(source.read_text(encoding="utf-8"))
    domains: dict[str, tuple[Passage, ...]] = {}
    for domain, entries in payload.items():
        if domain not in DOMAINS:
            raise ValueError(f"unknown domain {domain!r} in {source.name}")
        domains[domain] = tuple(Passage(str(entry["id"]), str(entry["title"]), str(entry["text"]), str(entry.get("note", ""))) for entry in entries)
    return domains


def _literature_rows(variants: int) -> list[JevSample]:
    """The literature domain is the public-domain excerpt generator."""
    from .literary import WORKS, _generate_for_works

    return _generate_for_works(WORKS, variants=variants, source="literary-public-domain-v1")


def _domain_rows(domain: str, passages: Sequence[Passage], variants: int) -> list[JevSample]:
    rows: list[JevSample] = []
    tasks = TASKS[domain]
    for passage in passages:
        for route, task_id, template, frame in tasks:
            derived = route_from_frame(frame)
            if derived is not route:
                raise ValueError(f"{domain}/{task_id} declares {route.value} but its frame derives {derived.value}")
            for index in range(variants):
                suffix = VARIANT_SUFFIXES[index % len(VARIANT_SUFFIXES)]
                rows.append(
                    JevSample(
                        query=template.format(text=passage.text) + suffix,
                        confidence=1.0,
                        route=route,
                        id=f"{domain}_{passage.passage_id}_{task_id}_{index}",
                        source=f"domain-{domain}-v1",
                        metadata={
                            "semantic_group_id": f"domain:{domain}:{passage.passage_id}:{task_id}",
                            "domain": domain,
                            "passage_id": passage.passage_id,
                            "passage_title": passage.title,
                            "task_id": task_id,
                            "frame": frame.to_dict(),
                            "frame_id": f"dom:{domain}:{passage.passage_id}:{task_id}",
                            "route_rule": explain_frame(frame),
                            "label_note": "route derived from the frame; the passage never decides the route",
                        },
                    )
                )
    return rows


def generate_domain(domain: str, variants: int = 3, *, passages: dict[str, tuple[Passage, ...]] | None = None) -> list[JevSample]:
    """Every row of one domain, all four routes covered by construction."""
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")
    if domain == "literature":
        return _literature_rows(variants)
    authored = passages if passages is not None else load_passages()
    if domain not in authored:
        raise KeyError(f"no passages for domain {domain!r}")
    return _domain_rows(domain, authored[domain], variants)


def generate_all_domains(variants: int = 3) -> dict[str, list[JevSample]]:
    """One list per domain, so a caller can weight or hold out a whole domain."""
    passages = load_passages()
    return {domain: generate_domain(domain, variants, passages=passages) for domain in DOMAINS}


def domain_audit(variants: int = 3) -> dict[str, Any]:
    """Per-domain route proportions, so a tilted domain is visible rather than assumed."""
    per_domain: dict[str, Any] = {}
    for domain, rows in generate_all_domains(variants).items():
        counts = {route.value: 0 for route in ROUTE_ORDER}
        for row in rows:
            counts[row.route.value] += 1
        groups = {str(row.metadata["semantic_group_id"]) for row in rows}
        share = {route: count / len(rows) for route, count in counts.items()} if rows else {}
        per_domain[domain] = {
            "rows": len(rows),
            "groups": len(groups),
            "route_counts": counts,
            "route_share": {route: round(value, 4) for route, value in share.items()},
            "balanced": all(abs(value - 0.25) < 1e-9 for value in share.values()),
        }
    return {"variants": variants, "domains": per_domain}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.domains", description="Domain-typed corpora")
    parser.add_argument("command", choices=("audit", "generate"))
    parser.add_argument("--domain", choices=DOMAINS, help="one domain, or all of them when omitted")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "external" / "domains")
    parser.add_argument("--variants", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        print(json.dumps(domain_audit(args.variants), ensure_ascii=False, indent=2))
        return
    passages = load_passages()
    targets = (args.domain,) if args.domain else DOMAINS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for domain in targets:
        rows = generate_domain(domain, args.variants, passages=passages)
        destination = args.output_dir / f"{domain}_v1.jsonl"
        save_jsonl(rows, destination)
        print(f"wrote {len(rows)} rows to {destination}")


if __name__ == "__main__":
    main()
