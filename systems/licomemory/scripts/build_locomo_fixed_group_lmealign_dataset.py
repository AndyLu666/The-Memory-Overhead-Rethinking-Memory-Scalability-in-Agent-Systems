#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_lines(path: Path, rows: Iterable[str]) -> None:
    values = [str(row).strip() for row in rows if str(row).strip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def hardlink_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def read_first_json_line(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    return json.loads(first) if first else {}


def parse_bins(text: str) -> List[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values:
        raise ValueError("No bins provided")
    if min(values) < 0:
        raise ValueError("Bins must be non-negative")
    return values


def stage_name(bin_s: int, replica: int) -> str:
    return f"s{bin_s:03d}_r{replica:02d}"


def parse_group_index(group_id: str) -> int:
    try:
        return int(str(group_id).split("_")[-1])
    except Exception:
        return 999


def sort_key_session(row: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        str(row.get("session_time", "") or ""),
        1 if bool(row.get("is_injected_noise")) else 0,
        str(row.get("session_id", "") or ""),
    )


def sort_key_question(row: Dict[str, Any]) -> Tuple[int, str]:
    return (int(row.get("question_index", 0) or 0), str(row.get("rel_path", "")))


def corpus_signature(rows: List[Dict[str, Any]]) -> str:
    payload = [
        {
            "session_time": str(row.get("session_time", "") or ""),
            "session_id": str(row.get("session_id", "") or ""),
            "context": str(row.get("context", "") or ""),
        }
        for row in rows
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalize_origin(origin: Any) -> List[str]:
    if isinstance(origin, str):
        return [origin] if origin.strip() else []
    if isinstance(origin, list):
        return [str(item) for item in origin if str(item).strip()]
    if isinstance(origin, tuple):
        return [str(item) for item in origin if str(item).strip()]
    return []


def build_stage_round_robin(rows: List[Dict[str, Any]]) -> List[str]:
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(row)
    group_ids = sorted(by_group.keys(), key=parse_group_index)
    for group_id in group_ids:
        by_group[group_id].sort(key=sort_key_question)

    ordered: List[str] = []
    live_groups = [group_id for group_id in group_ids if by_group[group_id]]
    cursor = 0
    while live_groups:
        group_id = live_groups[cursor % len(live_groups)]
        ordered.append(str(by_group[group_id].pop(0)["rel_path"]))
        if not by_group[group_id]:
            live_groups = [gid for gid in live_groups if by_group[gid]]
            cursor = 0
        else:
            cursor += 1
    return ordered


def split_batches(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def annotate_native_rows(rows: List[Dict[str, Any]], group_id: str) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["session_time"] = str(out.get("session_time", "") or "")
        out["session_id"] = str(out.get("session_id", "") or "")
        out["source_group_id"] = group_id
        out["original_session_id"] = out["session_id"]
        out["memory_role"] = "native"
        out["is_injected_noise"] = False
        annotated.append(out)
    annotated.sort(key=sort_key_session)
    return annotated


def make_external_pool_session_id(row_idx: int, source_task: str, original_session_id: str) -> str:
    safe_task = str(source_task or "unknown").replace("/", "_")
    safe_sid = str(original_session_id or "missing").replace("/", "_")
    return f"noise_lme_{safe_task}_{row_idx:05d}_{safe_sid}"


def external_row_dedupe_key(row: Dict[str, Any]) -> str:
    payload = {
        "session_time": str(row.get("session_time", "") or ""),
        "session_id": str(row.get("session_id", "") or ""),
        "context": str(row.get("context", "") or ""),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_external_longmemeval_pool(external_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    corpus_paths = sorted(external_root.rglob("Corpus.json"))
    if not corpus_paths:
        raise RuntimeError(f"No Corpus.json files found under external root: {external_root}")

    unique_rows: Dict[str, Dict[str, Any]] = {}
    source_counts: Dict[str, int] = defaultdict(int)
    for corpus_path in corpus_paths:
        try:
            source_task = corpus_path.parent.parent.name
        except Exception:
            source_task = "unknown"
        source_rel_path = str(corpus_path.parent.relative_to(external_root)).replace("\\", "/")
        rows = load_jsonl(corpus_path)
        for row in rows:
            key = external_row_dedupe_key(row)
            if key in unique_rows:
                continue
            source_counts[source_task] += 1
            idx = len(unique_rows) + 1
            original_session_id = str(row.get("session_id", "") or "")
            out = dict(row)
            out["session_time"] = str(out.get("session_time", "") or "")
            out["original_session_id"] = original_session_id
            out["session_id"] = make_external_pool_session_id(idx, source_task, original_session_id)
            out["source_group_id"] = "longmemeval_external"
            out["source_task"] = source_task
            out["source_rel_path"] = source_rel_path
            out["memory_role"] = "injected_noise"
            out["is_injected_noise"] = True
            unique_rows[key] = out

    pool = sorted(unique_rows.values(), key=sort_key_session)
    summary = {
        "external_root": str(external_root),
        "corpus_files": len(corpus_paths),
        "unique_external_sessions": len(pool),
        "per_task_unique_sessions": {task: int(count) for task, count in sorted(source_counts.items())},
    }
    return pool, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build LoCoMo fixed-group s-bin datasets aligned to LongMemEval-style bins "
            "using an out-of-domain LongMemEval filler pool."
        )
    )
    parser.add_argument("--src-root", required=True, help="Existing question-layout LoCoMo data root")
    parser.add_argument("--manifest", required=True, help="Question-layout export manifest JSONL")
    parser.add_argument("--external-filler-root", required=True, help="LongMemEval data root used as external filler pool")
    parser.add_argument("--out-data-root", required=True)
    parser.add_argument("--out-list-root", required=True)
    parser.add_argument("--bins", default="0,100,200,300,400")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--smoke-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260312)
    args = parser.parse_args()

    if args.replicas <= 0:
        raise ValueError("--replicas must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.smoke_size <= 0:
        raise ValueError("--smoke-size must be > 0")

    src_root = Path(args.src_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    external_filler_root = Path(args.external_filler_root).resolve()
    out_data_root = Path(args.out_data_root).resolve()
    out_list_root = Path(args.out_list_root).resolve()
    bins = parse_bins(args.bins)

    manifest_rows = [row for row in load_jsonl(manifest_path) if str(row.get("layout", "")) == "question"]
    if not manifest_rows:
        raise RuntimeError(f"No question-layout rows found in manifest: {manifest_path}")

    rows_by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        rows_by_group[str(row["group_id"])].append(row)
    for group_id in list(rows_by_group.keys()):
        rows_by_group[group_id].sort(key=sort_key_question)

    group_native_rows: Dict[str, List[Dict[str, Any]]] = {}
    group_rep_rel_path: Dict[str, str] = {}
    group_validation: Dict[str, Dict[str, Any]] = {}

    for group_id, rows in sorted(rows_by_group.items(), key=lambda item: parse_group_index(item[0])):
        signatures = set()
        canonical_rows: List[Dict[str, Any]] | None = None
        for idx, row in enumerate(rows):
            corpus_path = src_root / str(row["rel_path"]) / "Corpus.json"
            corpus_rows = load_jsonl(corpus_path)
            sig = corpus_signature(corpus_rows)
            signatures.add(sig)
            if idx == 0:
                canonical_rows = corpus_rows
                group_rep_rel_path[group_id] = str(row["rel_path"])
        if canonical_rows is None:
            raise RuntimeError(f"No corpus rows found for group {group_id}")
        if len(signatures) != 1:
            raise RuntimeError(
                f"Group {group_id} has inconsistent per-question corpora; signatures={sorted(signatures)}"
            )
        group_native_rows[group_id] = annotate_native_rows(canonical_rows, group_id)
        group_validation[group_id] = {
            "questions": len(rows),
            "native_session_count": len(canonical_rows),
            "corpus_signature": next(iter(signatures)),
            "representative_rel_path": group_rep_rel_path[group_id],
        }

    external_pool_rows, external_pool_summary = load_external_longmemeval_pool(external_filler_root)
    max_bin = max(bins)
    if len(external_pool_rows) < max_bin:
        raise RuntimeError(
            f"External LongMemEval filler pool cannot satisfy max bin {max_bin}: pool={len(external_pool_rows)}"
        )

    out_data_root.mkdir(parents=True, exist_ok=True)
    out_list_root.mkdir(parents=True, exist_ok=True)
    (out_list_root / "manifest_by_stage").mkdir(parents=True, exist_ok=True)
    (out_list_root / "smoke").mkdir(parents=True, exist_ok=True)

    all_manifest_rows: List[Dict[str, Any]] = []
    per_stage_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_stage_representatives: Dict[str, List[str]] = defaultdict(list)
    chosen_noise: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    corpus_link_stats: Dict[str, int] = defaultdict(int)

    for bin_s in bins:
        for replica in range(1, args.replicas + 1):
            for group_id in sorted(rows_by_group.keys(), key=parse_group_index):
                rng = random.Random(f"{args.seed}:lmealign:{group_id}:s{bin_s}:r{replica}")
                sampled = rng.sample(external_pool_rows, bin_s) if bin_s > 0 else []
                sampled = [copy.deepcopy(row) for row in sampled]
                sampled.sort(key=sort_key_session)
                chosen_noise[(group_id, bin_s, replica)] = sampled

    for bin_s in bins:
        for replica in range(1, args.replicas + 1):
            stage = stage_name(bin_s, replica)
            for group_id in sorted(rows_by_group.keys(), key=parse_group_index):
                native_rows = [copy.deepcopy(row) for row in group_native_rows[group_id]]
                sampled_noise = [copy.deepcopy(row) for row in chosen_noise[(group_id, bin_s, replica)]]
                corpus_rows = native_rows + sampled_noise
                corpus_rows.sort(key=sort_key_session)
                shared_corpus_path = out_data_root / stage / "_shared_group_corpus" / group_id / "Corpus.json"
                shared_corpus_path.parent.mkdir(parents=True, exist_ok=True)
                write_jsonl(shared_corpus_path, corpus_rows)
                corpus_link_stats["shared_corpus_files"] += 1

                for row_idx, manifest_row in enumerate(rows_by_group[group_id]):
                    src_question_path = src_root / str(manifest_row["rel_path"]) / "Question.json"
                    question_record = read_first_json_line(src_question_path)
                    origin_items = normalize_origin(question_record.get("origin", []))
                    native_nonorigin_count = max(0, len(native_rows) - len(origin_items))

                    out_rel_path = str(Path(stage) / str(manifest_row["rel_path"]))
                    out_dir = out_data_root / out_rel_path
                    out_dir.mkdir(parents=True, exist_ok=True)

                    question_record["stage"] = stage
                    question_record["bin_s"] = bin_s
                    question_record["replica"] = replica
                    question_record["memory_unit"] = "group"
                    question_record["noise_mode"] = "outdomain_longmemeval"
                    question_record["alignment_target"] = "longmemeval_fixed2k"
                    question_record["native_group_session_count"] = len(native_rows)
                    question_record["native_nonorigin_count"] = native_nonorigin_count
                    question_record["injected_session_count"] = len(sampled_noise)
                    question_record["total_session_count"] = len(corpus_rows)
                    question_record["group_representative_rel_path"] = group_rep_rel_path[group_id]
                    question_record["external_filler_root"] = str(external_filler_root)
                    question_record["external_filler_unique_session_count"] = len(external_pool_rows)

                    link_mode = hardlink_or_copy(shared_corpus_path, out_dir / "Corpus.json")
                    corpus_link_stats[f"question_corpus_{link_mode}"] += 1
                    write_jsonl(out_dir / "Question.json", [question_record])

                    out_manifest_row = dict(manifest_row)
                    out_manifest_row.update(
                        {
                            "stage": stage,
                            "bin_s": bin_s,
                            "replica": replica,
                            "rel_path": out_rel_path,
                            "native_group_session_count": len(native_rows),
                            "native_nonorigin_count": native_nonorigin_count,
                            "injected_session_count": len(sampled_noise),
                            "total_session_count": len(corpus_rows),
                            "memory_unit": "group",
                            "noise_mode": "outdomain_longmemeval",
                            "alignment_target": "longmemeval_fixed2k",
                            "external_filler_root": str(external_filler_root),
                            "external_filler_unique_session_count": len(external_pool_rows),
                            "representative_rel_path": str(Path(stage) / group_rep_rel_path[group_id]),
                        }
                    )
                    all_manifest_rows.append(out_manifest_row)
                    per_stage_rows[stage].append(out_manifest_row)
                    if row_idx == 0:
                        per_stage_representatives[stage].append(out_rel_path)

    write_jsonl(out_list_root / "manifest.jsonl", all_manifest_rows)

    summary_stages: Dict[str, Dict[str, Any]] = {}
    for stage, rows in sorted(per_stage_rows.items()):
        stage_manifest_path = out_list_root / "manifest_by_stage" / f"{stage}.jsonl"
        write_jsonl(stage_manifest_path, rows)

        ordered = build_stage_round_robin(rows)
        smoke = ordered[: args.smoke_size]
        batches = split_batches(ordered, args.batch_size)

        write_lines(out_list_root / f"{stage}_all.txt", ordered)
        write_lines(out_list_root / f"{stage}_representatives.txt", per_stage_representatives[stage])
        write_lines(out_list_root / f"{stage}_smoke{args.smoke_size}.txt", smoke)
        write_lines(out_list_root / "smoke" / f"{stage}_smoke{args.smoke_size}.txt", smoke)

        batch_summaries: List[Dict[str, Any]] = []
        for batch_idx, batch in enumerate(batches, start=1):
            batch_path = out_list_root / f"{stage}_batch{batch_idx:02d}_{args.batch_size}.txt"
            write_lines(batch_path, batch)
            batch_summaries.append(
                {
                    "batch_index": batch_idx,
                    "path": str(batch_path),
                    "size": len(batch),
                }
            )

        summary_stages[stage] = {
            "items": len(rows),
            "representatives": len(per_stage_representatives[stage]),
            "ordered_list": str(out_list_root / f"{stage}_all.txt"),
            "representative_list": str(out_list_root / f"{stage}_representatives.txt"),
            "smoke_list": str(out_list_root / f"{stage}_smoke{args.smoke_size}.txt"),
            "batch_count": len(batches),
            "batches": batch_summaries,
            "manifest": str(stage_manifest_path),
        }

    summary = {
        "src_root": str(src_root),
        "manifest": str(manifest_path),
        "external_filler_root": str(external_filler_root),
        "out_data_root": str(out_data_root),
        "out_list_root": str(out_list_root),
        "memory_unit": "group",
        "noise_mode": "outdomain_longmemeval",
        "alignment_target": "longmemeval_fixed2k",
        "question_type": "locomo-multi-hop",
        "bins": bins,
        "replicas": args.replicas,
        "batch_size": args.batch_size,
        "smoke_size": args.smoke_size,
        "seed": args.seed,
        "groups": len(rows_by_group),
        "questions": len(manifest_rows),
        "total_stage_items": len(all_manifest_rows),
        "group_validation": group_validation,
        "external_pool_summary": external_pool_summary,
        "corpus_materialization": dict(sorted(corpus_link_stats.items())),
        "stages": summary_stages,
    }
    (out_list_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
