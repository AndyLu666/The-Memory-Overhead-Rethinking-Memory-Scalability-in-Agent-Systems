#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

BENCHMARK_DIR = Path(__file__).resolve().parent
SYSTEM_ROOT = BENCHMARK_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
LICOMEMORY_ROOT = REPO_ROOT / "systems" / "licomemory"
if str(LICOMEMORY_ROOT) not in sys.path:
    sys.path.insert(0, str(LICOMEMORY_ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from run_hipporag_locomo import (  # noqa: E402
    DEFAULT_LIST_ROOT,
    link_or_copy,
    load_dataset_names,
    q0_assets_ready,
)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a destination LoCoMo results root by linking existing q0 assets "
            "from a completed HippoRAG LoCoMo q0 formal root."
        )
    )
    parser.add_argument("--source-results-root", required=True)
    parser.add_argument("--dest-results-root", required=True)
    parser.add_argument(
        "--dataset-lists",
        action="append",
        nargs="+",
        required=True,
        help="One or more LoCoMo list files to prepare.",
    )
    parser.add_argument("--list-root", default=str(DEFAULT_LIST_ROOT))
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any requested q0 asset is missing or incomplete.",
    )
    return parser.parse_args()


def resolve_list_paths(list_root: Path, raw_lists: List[List[str]]) -> List[Path]:
    out: List[Path] = []
    for group in raw_lists:
        for item in group:
            path = Path(item)
            if not path.is_absolute():
                path = list_root / item
            out.append(path.resolve())
    return out


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_results_root).resolve()
    dest_root = Path(args.dest_results_root).resolve()
    list_root = Path(args.list_root).resolve()
    list_paths = resolve_list_paths(list_root, args.dataset_lists)
    dataset_names = load_dataset_names(list_paths)

    ready = 0
    linked = 0
    link_modes: List[str] = []
    missing_or_bad: Dict[str, str] = {}

    for dataset_name in dataset_names:
        src_dir = source_root / dataset_name
        dst_dir = dest_root / dataset_name
        if not src_dir.exists():
            missing_or_bad[dataset_name] = "missing_dataset_dir"
            continue
        if not q0_assets_ready(src_dir, args.llm_model, args.embedding_model):
            missing_or_bad[dataset_name] = "q0_assets_missing_or_incomplete"
            continue
        ready += 1
        dst_dir.mkdir(parents=True, exist_ok=True)
        link_modes.append(link_or_copy(src_dir / "hipporag_q0_cache", dst_dir / "hipporag_q0_cache"))
        link_modes.append(link_or_copy(src_dir / "q0_index_summary.json", dst_dir / "q0_index_summary.json"))
        link_modes.append(link_or_copy(src_dir / "q0_cost_summary.json", dst_dir / "q0_cost_summary.json"))
        linked += 1

    if args.require_complete and missing_or_bad:
        raise RuntimeError(
            "source q0 root is not fully reusable: "
            + json.dumps(missing_or_bad, ensure_ascii=False, indent=2)
        )

    summary = {
        "source_results_root": str(source_root),
        "dest_results_root": str(dest_root),
        "dataset_count": len(dataset_names),
        "ready_count": ready,
        "linked_count": linked,
        "missing_or_bad": missing_or_bad,
        "list_paths": [str(path) for path in list_paths],
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "link_modes": sorted(set(link_modes)),
        "finished_at": time.time(),
    }
    save_json(dest_root / "base_reuse_prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
