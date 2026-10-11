"""Deterministic literary-domain data for route-boundary experiments.

The texts are short traditional/public-domain excerpts used as *states*. The
labels describe the operation requested on that state, not the literary work:
local extraction, external provenance lookup, multi-step interpretation, or
high-risk copyright/legal review. The generated set is a controlled supplement,
not a substitute for human-labelled literary traffic.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

from .frames import Derivation, Risk, RouteFrame, Source, explain_frame, route_from_frame
from .schema import JevSample, Route, save_jsonl

WORKS: tuple[tuple[str, str, str], ...] = (
    ("静夜思", "李白", "床前明月光，疑是地上霜。"),
    ("春晓", "孟浩然", "春眠不觉晓，处处闻啼鸟。"),
    ("登鹳雀楼", "王之涣", "白日依山尽，黄河入海流。"),
    ("江雪", "柳宗元", "千山鸟飞绝，万径人踪灭。"),
    ("望庐山瀑布", "李白", "飞流直下三千尺，疑是银河落九天。"),
    ("山居秋暝", "王维", "明月松间照，清泉石上流。"),
    ("游子吟", "孟郊", "谁言寸草心，报得三春晖。"),
    ("天净沙·秋思", "马致远", "枯藤老树昏鸦，小桥流水人家。"),
    ("岳阳楼记", "范仲淹", "先天下之忧而忧，后天下之乐而乐。"),
    ("桃花源记", "陶渊明", "土地平旷，屋舍俨然，有良田美池桑竹之属。"),
    ("陋室铭", "刘禹锡", "山不在高，有仙则名；水不在深，有龙则灵。"),
    ("木兰诗", "北朝民歌", "朔气传金柝，寒光照铁衣。"),
)

TEMPLATES: tuple[tuple[Route, str, str, RouteFrame], ...] = (
    (
        Route.LOCAL,
        "local_extract",
        "片段：{text} 请统计这段文字中的汉字数量。",
        RouteFrame("统计", "这段文字中的汉字数量", Source.GIVEN_TEXT, Derivation.EXTRACT, domain="literature"),
    ),
    (
        Route.RAG,
        "rag_provenance",
        "片段：{text} 请查找这段文字的作者和作品出处。",
        RouteFrame("查找", "这段文字的作者和作品出处", Source.DOCUMENT, Derivation.EXTRACT, domain="literature"),
    ),
    (
        Route.CLOUD,
        "cloud_interpret",
        "片段：{text} 请分析其中的核心意象，并与另一首诗作比较表达差异。",
        RouteFrame(
            "分析",
            "核心意象并与另一首诗比较",
            Source.GIVEN_TEXT,
            Derivation.CONSTRUCT,
            steps=2,
            risk=Risk.MEDIUM,
            domain="literature",
        ),
    ),
    (
        Route.REVIEW,
        "review_rights",
        "片段：{text} 请判断把它用于商业产品是否存在版权或法律风险。",
        RouteFrame("判断", "用于商业产品的版权或法律风险", Source.NONE, Derivation.CONSTRUCT, risk=Risk.HIGH, domain="literature"),
    ),
)


VARIANT_SUFFIXES: tuple[str, ...] = ("", "请给出简短、明确的判断。", "只回答结论，不要展开解释。")


def generate(variants: int = 3) -> list[JevSample]:
    """Generate grouped literary rows with deterministic paraphrase variants."""
    return _generate_for_works(WORKS, variants=variants, source="literary-public-domain-v1")


def _generate_for_works(
    works: Iterable[tuple[str, str, str]],
    *,
    variants: int,
    source: str,
    provenance: str = "traditional/classical excerpt; verify edition rights before publication",
) -> list[JevSample]:
    if variants <= 0:
        raise ValueError("variants must be positive")
    rows: list[JevSample] = []
    for work_id, (title, author, text) in enumerate(works):
        for route, task_id, template, frame in TEMPLATES:
            # The frame must agree with the declared route, or the annotation is a lie.
            derived = route_from_frame(frame)
            if derived is not route:
                raise ValueError(f"task {task_id} declares {route.value} but its frame derives {derived.value}")
            for variant in range(variants):
                # Distinct suffix per variant; every row in a group stays the same
                # task over the same excerpt, only the wording differs.
                suffix = VARIANT_SUFFIXES[variant % len(VARIANT_SUFFIXES)]
                query = template.format(text=text) + suffix
                group = f"literary:{work_id}:{task_id}"
                rows.append(
                    JevSample(
                        query=query,
                        confidence=1.0,
                        route=route,
                        id=f"literary_{work_id}_{task_id}_{variant}",
                        source=source,
                        metadata={
                            "semantic_group_id": group,
                            "literary_work": title,
                            "literary_author": author,
                            "literary_task": task_id,
                            "label_note": "route is assigned from requested operation, not from the work itself",
                            "text_provenance": provenance,
                            "domain": frame.domain,
                            "frame": frame.to_dict(),
                            "frame_id": f"lit:{work_id}:{task_id}",
                            "route_rule": explain_frame(frame),
                        },
                    )
                )
    return rows


def _chunks(text: str, chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    paragraphs = [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [paragraph[index : index + chunk_size] for index in range(0, len(paragraph), chunk_size)]
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > chunk_size:
                chunks.append(current)
                current = ""
            current = piece if not current else f"{current}\n{piece}"
    if current:
        chunks.append(current)
    return chunks


def generate_from_text(
    text: str,
    *,
    title: str = "未命名文本",
    author: str = "未知作者",
    variants: int = 3,
    chunk_size: int = 420,
) -> list[JevSample]:
    """Generate route tasks from user-provided text without storing the text in the repo."""
    chunks = _chunks(text, chunk_size)
    if not chunks:
        raise ValueError("text must contain at least one non-empty line")
    rows = _generate_for_works(
        ((title, author, chunk) for chunk in chunks),
        variants=variants,
        source="literary-local-text-v1",
        provenance="user-provided local text; confirm license and edition rights before publication",
    )
    rows_per_block = len(TEMPLATES) * variants
    return [
        JevSample(
            query=row.query,
            confidence=row.confidence,
            route=row.route,
            context=row.context,
            candidates=row.candidates,
            id=row.id,
            source=row.source,
            metadata={
                **row.metadata,
                "literary_input": "local_text",
                "text_block_index": index // rows_per_block,
                "text_block_count": len(chunks),
            },
        )
        for index, row in enumerate(rows)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.literary", description="Generate literary-domain MiniJev data")
    parser.add_argument("--output", type=Path, default=Path("data/external/literary/literary_v1.jsonl"))
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--text-file", type=Path, help="read a local UTF-8 novel/text file instead of built-in excerpts")
    parser.add_argument("--title", default="未命名文本")
    parser.add_argument("--author", default="未知作者")
    parser.add_argument("--chunk-size", type=int, default=420)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.text_file is not None:
        rows = generate_from_text(
            args.text_file.read_text(encoding="utf-8"),
            title=args.title,
            author=args.author,
            variants=args.variants,
            chunk_size=args.chunk_size,
        )
    else:
        rows = generate(args.variants)
    save_jsonl(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
