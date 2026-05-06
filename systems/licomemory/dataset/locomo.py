#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process LoCoMo into LiCoMemory-compatible folders.

Supported layouts:

1. ``group`` (legacy):
   outdir/group_1/Corpus.json
   outdir/group_1/Question.json

2. ``question`` (runner-compatible):
   outdir/locomo_temporal/lm_g01_q0001/Corpus.json
   outdir/locomo_temporal/lm_g01_q0001/Question.json
   ...

The ``question`` layout is the one to use for q0/q1 batch runs because each
directory contains exactly one question, which matches the current runner/QC
assumptions in LiCoMemory.
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")

MONTH_MAP = {
    "january": "01",
    "jan": "01",
    "february": "02",
    "feb": "02",
    "march": "03",
    "mar": "03",
    "april": "04",
    "apr": "04",
    "may": "05",
    "june": "06",
    "jun": "06",
    "july": "07",
    "jul": "07",
    "august": "08",
    "aug": "08",
    "september": "09",
    "sep": "09",
    "october": "10",
    "oct": "10",
    "november": "11",
    "nov": "11",
    "december": "12",
    "dec": "12",
}

CATEGORY_INFO: Dict[int, Dict[str, str]] = {
    1: {
        "slug": "locomo_multihop",
        "question_type": "locomo-multi-hop",
        "category_name": "multi-hop",
    },
    2: {
        "slug": "locomo_temporal",
        "question_type": "locomo-temporal",
        "category_name": "temporal",
    },
    3: {
        "slug": "locomo_commonsense",
        "question_type": "locomo-commonsense",
        "category_name": "commonsense",
    },
    4: {
        "slug": "locomo_singlehop",
        "question_type": "locomo-single-hop",
        "category_name": "single-hop",
    },
    5: {
        "slug": "locomo_adversarial",
        "question_type": "locomo-adversarial",
        "category_name": "adversarial",
    },
}


def parse_date(date_str: str) -> str:
    """Parse strings like '1:56 pm on 8 May, 2023' to '2023/05/08'."""
    if not date_str:
        return ""

    match = DATE_RE.search(date_str.lower())
    if not match:
        return date_str

    day = match.group(1).zfill(2)
    month_name = match.group(2).lower()
    year = match.group(3)
    month = MONTH_MAP.get(month_name, "00")
    return f"{year}/{month}/{day}"


def get_question_id(group_index: int, question_index: int) -> str:
    return f"locomo_g{group_index:02d}_q{question_index:04d}"


def get_item_dir_name(group_index: int, question_index: int) -> str:
    return f"lm_g{group_index:02d}_q{question_index:04d}"


def get_category_info(category: Any) -> Dict[str, str]:
    try:
        category_int = int(category)
    except (TypeError, ValueError):
        category_int = -1
    return CATEGORY_INFO.get(
        category_int,
        {
            "slug": "locomo_unknown",
            "question_type": "locomo-unknown",
            "category_name": "unknown",
        },
    )


def extract_evidence_prefix(evidence_list: Iterable[Any]) -> List[str]:
    prefixes: List[str] = []
    for ev in evidence_list or []:
        if isinstance(ev, str) and len(ev) >= 2:
            prefix = ev.split(":", 1)[0]
            if prefix and prefix not in prefixes:
                prefixes.append(prefix)
    return prefixes


def normalize_origin(prefixes: List[str]) -> Any:
    if len(prefixes) == 1:
        return prefixes[0]
    return prefixes


def build_context(session_messages: Iterable[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in session_messages:
        speaker = str(msg.get("speaker", ""))
        text = str(msg.get("text", ""))
        blip_caption = str(msg.get("blip_caption", ""))
        if blip_caption:
            text = f"{text} (attached is {blip_caption})"
        parts.append(f"\"{speaker}\": \"{text}\"")
    return "".join(parts)


def build_corpus_records(group_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    conversation = group_data.get("conversation", {})
    corpus_records: List[Dict[str, Any]] = []
    session_num = 1
    while True:
        session_key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        if session_key not in conversation:
            break

        corpus_records.append(
            {
                "session_time": parse_date(str(conversation.get(date_key, ""))),
                "context": build_context(conversation.get(session_key, [])),
                "session_id": f"D{session_num}",
            }
        )
        session_num += 1
    return corpus_records


def select_answer(qa_item: Dict[str, Any]) -> Tuple[str, str]:
    answer = qa_item.get("answer")
    if answer not in (None, ""):
        return str(answer), "answer"

    adversarial_answer = qa_item.get("adversarial_answer")
    if adversarial_answer not in (None, ""):
        return str(adversarial_answer), "adversarial_answer"

    return "Context insufficient to answer", "fallback_context_insufficient"


def build_question_record(
    qa_item: Dict[str, Any],
    group_index: int,
    question_index: int,
    sample_id: str,
    layout: str,
) -> Dict[str, Any]:
    category = qa_item.get("category", "")
    category_info = get_category_info(category)
    answer, answer_source = select_answer(qa_item)
    evidence = qa_item.get("evidence", []) or []
    origin_prefixes = extract_evidence_prefix(evidence)

    question_type = str(category)
    if layout == "question":
        question_type = category_info["question_type"]

    return {
        "question_id": get_question_id(group_index, question_index),
        "sample_id": sample_id,
        "group_id": f"group_{group_index}",
        "group_index": group_index,
        "question_index": question_index,
        "question": qa_item.get("question", ""),
        "answer": answer,
        "label": answer,
        "question_type": question_type,
        "question_type_name": category_info["question_type"],
        "category_id": category,
        "category_name": category_info["category_name"],
        "origin": normalize_origin(origin_prefixes),
        "evidence": evidence,
        "answer_source": answer_source,
        "question_time": "",
    }


def write_ndjson(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_text_lines(path: Path, rows: Iterable[str]) -> None:
    values = [row for row in rows if row]
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def load_selected_question_ids(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    values: Set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value:
            values.add(value)
    return values


def export_group_layout(
    data: List[Dict[str, Any]],
    out_root: Path,
    manifest_rows: List[Dict[str, Any]],
) -> Tuple[int, int, int]:
    total_groups = 0
    total_sessions = 0
    total_questions = 0

    for group_index, group_data in enumerate(data, start=1):
        if not isinstance(group_data, dict):
            continue

        sample_id = str(group_data.get("sample_id", ""))
        corpus_records = build_corpus_records(group_data)
        qa_list = group_data.get("qa", []) or []
        question_records = [
            build_question_record(qa_item, group_index, question_index, sample_id, "group")
            for question_index, qa_item in enumerate(qa_list, start=1)
        ]

        group_dir = out_root / f"group_{group_index}"
        group_dir.mkdir(parents=True, exist_ok=True)
        write_ndjson(group_dir / "Corpus.json", corpus_records)
        write_ndjson(group_dir / "Question.json", question_records)

        total_groups += 1
        total_sessions += len(corpus_records)
        total_questions += len(question_records)

        for question_record in question_records:
            manifest_rows.append(
                {
                    "layout": "group",
                    "rel_path": str(group_dir.relative_to(out_root)),
                    "sample_id": sample_id,
                    "group_id": f"group_{group_index}",
                    "group_index": group_index,
                    "question_id": question_record["question_id"],
                    "question_index": question_record["question_index"],
                    "question_type": question_record["question_type"],
                    "question_type_name": question_record["question_type_name"],
                    "category_id": question_record["category_id"],
                    "category_name": question_record["category_name"],
                    "answer_source": question_record["answer_source"],
                    "session_count": len(corpus_records),
                    "origin_count": len(extract_evidence_prefix(question_record.get("evidence", []))),
                }
            )

        print(
            f"[OK] {group_dir} -> {len(corpus_records)} sessions, "
            f"{len(question_records)} questions"
        )

    return total_groups, total_sessions, total_questions


def export_question_layout(
    data: List[Dict[str, Any]],
    out_root: Path,
    selected_question_ids: Optional[Set[str]],
    manifest_rows: List[Dict[str, Any]],
    dataset_names: List[str],
) -> Tuple[int, int, int]:
    total_items = 0
    total_sessions = 0
    total_questions = 0
    exported_question_ids: Set[str] = set()

    for group_index, group_data in enumerate(data, start=1):
        if not isinstance(group_data, dict):
            continue

        sample_id = str(group_data.get("sample_id", ""))
        corpus_records = build_corpus_records(group_data)
        qa_list = group_data.get("qa", []) or []

        for question_index, qa_item in enumerate(qa_list, start=1):
            question_id = get_question_id(group_index, question_index)
            if selected_question_ids is not None and question_id not in selected_question_ids:
                continue

            question_record = build_question_record(
                qa_item,
                group_index,
                question_index,
                sample_id,
                "question",
            )
            category_info = get_category_info(question_record["category_id"])
            type_dir = out_root / category_info["slug"]
            item_name = get_item_dir_name(group_index, question_index)
            item_dir = type_dir / item_name
            rel_path = str(item_dir.relative_to(out_root))

            item_dir.mkdir(parents=True, exist_ok=True)
            write_ndjson(item_dir / "Corpus.json", corpus_records)
            write_ndjson(item_dir / "Question.json", [question_record])

            manifest_rows.append(
                {
                    "layout": "question",
                    "rel_path": rel_path,
                    "sample_id": sample_id,
                    "group_id": f"group_{group_index}",
                    "group_index": group_index,
                    "question_id": question_record["question_id"],
                    "question_index": question_record["question_index"],
                    "question_type": question_record["question_type"],
                    "question_type_name": question_record["question_type_name"],
                    "category_id": question_record["category_id"],
                    "category_name": question_record["category_name"],
                    "answer_source": question_record["answer_source"],
                    "session_count": len(corpus_records),
                    "origin_count": len(extract_evidence_prefix(question_record.get("evidence", []))),
                    "origin": question_record["origin"],
                    "evidence": question_record["evidence"],
                }
            )
            dataset_names.append(rel_path)
            exported_question_ids.add(question_id)
            total_items += 1
            total_questions += 1
            total_sessions += len(corpus_records)

            print(
                f"[OK] {item_dir} -> {len(corpus_records)} sessions, "
                f"1 question ({question_record['question_type']})"
            )

    if selected_question_ids is not None:
        missing = sorted(selected_question_ids - exported_question_ids)
        if missing:
            raise ValueError(
                "Some selected LoCoMo question IDs were not exported: "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )

    return total_items, total_sessions, total_questions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process LoCoMo dataset into LiCoMemory-compatible folders"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input LoCoMo JSON file")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--layout",
        choices=("group", "question"),
        default="group",
        help="Export layout: legacy per-group or runner-compatible per-question",
    )
    parser.add_argument(
        "--sample-list",
        default="",
        help="Optional file with LoCoMo question_id values to export in question layout",
    )
    parser.add_argument(
        "--manifest-out",
        default="",
        help="Optional manifest JSONL output path",
    )
    parser.add_argument(
        "--dataset-list-out",
        default="",
        help="Optional dataset-name list output path (question layout only)",
    )
    args = parser.parse_args()

    out_root = Path(args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with Path(args.input).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of LoCoMo groups.")

    selected_question_ids = load_selected_question_ids(args.sample_list)
    manifest_rows: List[Dict[str, Any]] = []
    dataset_names: List[str] = []

    if args.layout == "group":
        total_items, total_sessions, total_questions = export_group_layout(
            data,
            out_root,
            manifest_rows,
        )
        print(
            f"\nDone. Processed {total_items} groups with "
            f"{total_sessions} sessions and {total_questions} questions in total."
        )
    else:
        total_items, total_sessions, total_questions = export_question_layout(
            data,
            out_root,
            selected_question_ids,
            manifest_rows,
            dataset_names,
        )
        category_counter = Counter(
            str(row.get("category_name", "")) for row in manifest_rows if row.get("layout") == "question"
        )
        print(
            f"\nDone. Exported {total_items} question directories with "
            f"{total_sessions} duplicated corpus sessions in total."
        )
        print("Questions by category:")
        for category_name, count in sorted(category_counter.items()):
            print(f"  {category_name}: {count}")

    if args.manifest_out:
        manifest_path = Path(args.manifest_out).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_ndjson(manifest_path, manifest_rows)
        print(f"Manifest written to: {manifest_path}")

    if args.dataset_list_out:
        if args.layout != "question":
            raise ValueError("--dataset-list-out is only supported with --layout question")
        dataset_list_path = Path(args.dataset_list_out).resolve()
        dataset_list_path.parent.mkdir(parents=True, exist_ok=True)
        write_text_lines(dataset_list_path, dataset_names)
        print(f"Dataset list written to: {dataset_list_path}")

    print(f"Output directory: {out_root}")


if __name__ == "__main__":
    main()
