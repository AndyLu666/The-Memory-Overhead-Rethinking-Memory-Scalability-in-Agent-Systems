#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_TYPES = ["longmem_ssp", "longmem_ssu", "longmem_ssa", "longmem_tr"]


def _read_first_json_line(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    return json.loads(first) if first else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _iter_base_items(src_root: Path, include_types: List[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for task in include_types:
        task_dir = src_root / task
        if not task_dir.exists():
            continue

        for lm_dir in sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("lm_")):
            qpath = lm_dir / "Question.json"
            cpath = lm_dir / "Corpus.json"
            if not qpath.exists() or not cpath.exists():
                continue

            question = _read_first_json_line(qpath)
            corpus_rows = _read_jsonl(cpath)

            origin = question.get("origin", [])
            if isinstance(origin, str):
                origins = [origin]
            elif isinstance(origin, list):
                origins = [x for x in origin if isinstance(x, str) and x]
            else:
                origins = []
            if not origins:
                continue

            sid_to_row: Dict[str, Dict[str, Any]] = {}
            for row in corpus_rows:
                sid = row.get("session_id")
                if sid and sid not in sid_to_row:
                    sid_to_row[str(sid)] = row

            if not all(o in sid_to_row for o in origins):
                continue

            origin_rows = [sid_to_row[o] for o in origins]
            origin_set = set(origins)
            irrelevant_rows = [r for r in corpus_rows if str(r.get("session_id", "")) not in origin_set]

            items.append(
                {
                    "task": task,
                    "lm_name": lm_dir.name,
                    "rel_path": f"{task}/{lm_dir.name}",
                    "question_path": qpath,
                    "question": question,
                    "origin_rows": origin_rows,
                    "origins_count": len(origins),
                    "irrelevant_rows": irrelevant_rows,
                    "max_s": len(irrelevant_rows),
                }
            )
    return items


def _split_batches(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _write_txt(path: Path, rows: List[str]) -> None:
    text = "\n".join(rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _sample_anchors(
    include_types: List[str],
    by_task: Dict[str, List[Dict[str, Any]]],
    anchors_per_type: int,
    max_bin: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    rng = random.Random(seed)
    anchors: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, int]] = {}

    for task in include_types:
        candidates = [x for x in by_task.get(task, []) if int(x.get("max_s", 0)) >= max_bin]
        if not candidates:
            raise RuntimeError(f"No candidate base items for task={task} with max_s >= {max_bin}")

        replacement = len(candidates) < anchors_per_type
        selected: List[Dict[str, Any]] = []
        if replacement:
            for _ in range(anchors_per_type):
                selected.append(rng.choice(candidates))
        else:
            selected = rng.sample(candidates, anchors_per_type)

        unique_base = {s["rel_path"] for s in selected}
        stats[task] = {
            "eligible": len(candidates),
            "selected": len(selected),
            "unique_base_selected": len(unique_base),
            "sampling_with_replacement": int(replacement),
        }

        for idx, base in enumerate(selected, start=1):
            anchor_id = f"{task}_a{idx:04d}"
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "anchor_index": idx,
                    "task": task,
                    "base_rel_path": base["rel_path"],
                    "lm_name": base["lm_name"],
                    "question_path": str(base["question_path"]),
                    "origins_count": int(base["origins_count"]),
                    "max_s": int(base["max_s"]),
                    "origin_rows": base["origin_rows"],
                    "irrelevant_rows": base["irrelevant_rows"],
                    "anchor_seed": rng.randrange(1, 2**31 - 1),
                }
            )

    return anchors, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed-anchor LongMemEval s-bin datasets where all bins share the same 2000 anchor identities."
    )
    parser.add_argument(
        "--src-root",
        default=os.getenv("LICOMEMORY_LONGMEMEVAL_SOURCE_ROOT", ""),
    )
    parser.add_argument("--out-data-root", required=True)
    parser.add_argument("--out-list-root", required=True)
    parser.add_argument("--types", default=",".join(DEFAULT_TYPES))
    parser.add_argument("--bins", default="0,100,200,300,400")
    parser.add_argument("--anchors-per-type", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--smoke-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260224)
    args = parser.parse_args()

    include_types = [t.strip() for t in args.types.split(",") if t.strip()]
    bins = [int(x.strip()) for x in args.bins.split(",") if x.strip()]
    if not include_types:
        raise RuntimeError("No task types specified")
    if not bins:
        raise RuntimeError("No bins specified")
    if args.anchors_per_type <= 0:
        raise RuntimeError("anchors-per-type must be > 0")
    if args.batch_size <= 0:
        raise RuntimeError("batch-size must be > 0")
    if args.smoke_size <= 0:
        raise RuntimeError("smoke-size must be > 0")

    src_root = Path(args.src_root).resolve()
    out_data_root = Path(args.out_data_root).resolve()
    out_list_root = Path(args.out_list_root).resolve()
    out_data_root.mkdir(parents=True, exist_ok=True)
    out_list_root.mkdir(parents=True, exist_ok=True)
    (out_list_root / "smoke").mkdir(parents=True, exist_ok=True)

    all_items = _iter_base_items(src_root, include_types)
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in all_items:
        by_task[item["task"]].append(item)
    for task in include_types:
        if not by_task.get(task):
            raise RuntimeError(f"No source items for task={task}")

    max_bin = max(bins)
    anchors, anchor_task_stats = _sample_anchors(
        include_types=include_types,
        by_task=by_task,
        anchors_per_type=args.anchors_per_type,
        max_bin=max_bin,
        seed=args.seed,
    )
    anchors.sort(key=lambda x: (include_types.index(x["task"]), x["anchor_index"]))

    anchor_manifest_rows: List[Dict[str, Any]] = []
    for a in anchors:
        anchor_manifest_rows.append(
            {
                "anchor_id": a["anchor_id"],
                "task": a["task"],
                "anchor_index": a["anchor_index"],
                "base_rel_path": a["base_rel_path"],
                "origins_count": a["origins_count"],
                "max_s": a["max_s"],
                "anchor_seed": a["anchor_seed"],
            }
        )
    _write_jsonl(out_list_root / "anchor_manifest.jsonl", anchor_manifest_rows)
    _write_txt(out_list_root / "anchor_ids.txt", [a["anchor_id"] for a in anchors])

    manifest_rows: List[Dict[str, Any]] = []
    per_bin_names: Dict[int, List[str]] = {b: [] for b in bins}
    per_bin_task_names: Dict[Tuple[int, str], List[str]] = defaultdict(list)
    per_bin_stage_manifest: Dict[int, List[Dict[str, Any]]] = {b: [] for b in bins}

    for b in bins:
        for anchor in anchors:
            task = anchor["task"]
            origin_rows = anchor["origin_rows"]
            need_irrelevant = int(b)
            if need_irrelevant > len(anchor["irrelevant_rows"]):
                raise RuntimeError(
                    f"Anchor {anchor['anchor_id']} cannot satisfy s={b}: "
                    f"need={need_irrelevant}, max={len(anchor['irrelevant_rows'])}"
                )
            rng = random.Random(f"{args.seed}:{anchor['anchor_id']}:s{b}:{anchor['anchor_seed']}")
            chosen_irrelevant = (
                rng.sample(anchor["irrelevant_rows"], need_irrelevant) if need_irrelevant > 0 else []
            )
            corpus_rows = list(origin_rows) + chosen_irrelevant
            rng.shuffle(corpus_rows)

            inst_name = f"{anchor['lm_name']}_a{anchor['anchor_index']:04d}"
            rel_dir = Path(f"s{b}") / task / inst_name
            dataset_name = str(rel_dir).replace("\\", "/")
            dst_dir = out_data_root / rel_dir
            dst_dir.mkdir(parents=True, exist_ok=True)

            _write_jsonl(dst_dir / "Corpus.json", corpus_rows)
            qsrc = Path(anchor["question_path"])
            (dst_dir / "Question.json").write_text(qsrc.read_text(encoding="utf-8"), encoding="utf-8")

            row = {
                "dataset_name": dataset_name,
                "bin_s": int(b),
                "task": task,
                "anchor_id": anchor["anchor_id"],
                "anchor_index": int(anchor["anchor_index"]),
                "base_rel_path": anchor["base_rel_path"],
                "origins_count": int(anchor["origins_count"]),
                "total_sessions": int(anchor["origins_count"]) + int(b),
                "irrelevant_added": int(b),
            }
            manifest_rows.append(row)
            per_bin_stage_manifest[b].append(row)
            per_bin_names[b].append(dataset_name)
            per_bin_task_names[(b, task)].append(dataset_name)

    _write_jsonl(out_list_root / "manifest.jsonl", manifest_rows)
    stage_manifest_dir = out_list_root / "manifest_by_stage"
    stage_manifest_dir.mkdir(parents=True, exist_ok=True)
    for b in bins:
        _write_jsonl(stage_manifest_dir / f"s{b}_manifest.jsonl", per_bin_stage_manifest[b])

    total_anchor = len(anchors)
    expected_per_bin = total_anchor
    smoke_per_task_base = args.smoke_size // len(include_types)
    smoke_remainder = args.smoke_size % len(include_types)

    for b in bins:
        names = per_bin_names[b]
        if len(names) != expected_per_bin:
            raise RuntimeError(
                f"Stage s{b} size mismatch: expected={expected_per_bin} got={len(names)}"
            )

        stage = f"s{b}"
        _write_txt(out_list_root / f"{stage}_all.txt", names)

        for task in include_types:
            task_names = per_bin_task_names[(b, task)]
            if len(task_names) != args.anchors_per_type:
                raise RuntimeError(
                    f"Stage {stage}, task {task} mismatch: expected={args.anchors_per_type}, got={len(task_names)}"
                )
            _write_txt(out_list_root / f"{stage}_{task}.txt", task_names)

        smoke_names: List[str] = []
        for task_idx, task in enumerate(include_types):
            k = smoke_per_task_base + (1 if task_idx < smoke_remainder else 0)
            smoke_names.extend(per_bin_task_names[(b, task)][:k])
        _write_txt(out_list_root / "smoke" / f"{stage}_smoke{args.smoke_size}.txt", smoke_names)
        _write_txt(out_list_root / f"{stage}_smoke{args.smoke_size}.txt", smoke_names)

        batches = _split_batches(names, args.batch_size)
        for idx, batch in enumerate(batches, start=1):
            _write_txt(out_list_root / f"{stage}_batch{idx:02d}_{args.batch_size}.txt", batch)

    summary = {
        "seed": args.seed,
        "src_root": str(src_root),
        "out_data_root": str(out_data_root),
        "out_list_root": str(out_list_root),
        "types": include_types,
        "bins": bins,
        "anchors_per_type": args.anchors_per_type,
        "total_anchors": total_anchor,
        "total_samples_per_bin": expected_per_bin,
        "batch_size": args.batch_size,
        "smoke_size": args.smoke_size,
        "anchor_task_stats": anchor_task_stats,
        "per_bin_total": {f"s{b}": len(per_bin_names[b]) for b in bins},
        "per_bin_task_total": {
            f"s{b}/{task}": len(per_bin_task_names[(b, task)])
            for b in bins
            for task in include_types
        },
        "artifacts": {
            "manifest": str(out_list_root / "manifest.jsonl"),
            "anchor_manifest": str(out_list_root / "anchor_manifest.jsonl"),
            "stage_manifest_dir": str(stage_manifest_dir),
        },
    }
    (out_list_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
