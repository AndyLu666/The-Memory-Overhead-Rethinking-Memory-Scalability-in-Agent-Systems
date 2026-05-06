#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
LICOMEMORY_ROOT = REPO_ROOT / "systems" / "licomemory"
if str(LICOMEMORY_ROOT) not in sys.path:
    sys.path.insert(0, str(LICOMEMORY_ROOT))

from run_hipporag_longmemeval import q0_ready, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a destination results root to safely reuse existing HippoRAG "
            "base q0 caches via per-base symlinks."
        )
    )
    parser.add_argument("--source-results-root", required=True)
    parser.add_argument("--dest-results-root", required=True)
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument(
        "--base-rel-paths-file",
        default="",
        help="Optional text file with base_rel_path values to prepare. Defaults to all source bases.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any requested base cache is missing or not q0-ready.",
    )
    return parser.parse_args()


def load_requested_bases(source_root: Path, base_rel_paths_file: str) -> List[str]:
    if base_rel_paths_file:
        return [
            line.strip()
            for line in Path(base_rel_paths_file).resolve().read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    bases: List[str] = []
    base_root = source_root / "base" / "_base_caches"
    for fam_dir in sorted(base_root.iterdir()):
        if not fam_dir.is_dir():
            continue
        for base_dir in sorted(fam_dir.iterdir()):
            if base_dir.is_dir():
                bases.append(f"{fam_dir.name}/{base_dir.name}")
    return bases


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_results_root).resolve()
    dest_root = Path(args.dest_results_root).resolve()
    source_base_root = source_root / "base" / "_base_caches"
    dest_base_root = dest_root / "base" / "_base_caches"
    dest_base_root.mkdir(parents=True, exist_ok=True)

    requested_bases = load_requested_bases(source_root, args.base_rel_paths_file)

    ready: List[str] = []
    missing_or_bad: Dict[str, str] = {}

    for base_rel_path in requested_bases:
        fam, name = base_rel_path.split("/", 1)
        src_dir = source_base_root / fam / name
        src_cache = src_dir / "hipporag_q0_cache"
        src_summary = src_dir / "q0_index_summary.json"
        src_cost = src_dir / "q0_cost_summary.json"
        if not src_dir.exists():
            missing_or_bad[base_rel_path] = "missing_base_dir"
            continue
        if not src_summary.exists():
            missing_or_bad[base_rel_path] = "missing_q0_index_summary"
            continue
        if not src_cost.exists():
            missing_or_bad[base_rel_path] = "missing_q0_cost_summary"
            continue
        if not q0_ready(src_cache, args.llm_model, args.embedding_model):
            missing_or_bad[base_rel_path] = "q0_not_ready"
            continue
        ready.append(base_rel_path)

    if args.require_complete and missing_or_bad:
        raise RuntimeError(
            "source base cache root is not fully reusable: "
            + json.dumps(missing_or_bad, ensure_ascii=False, indent=2)
        )

    linked: List[str] = []
    for base_rel_path in ready:
        fam, name = base_rel_path.split("/", 1)
        src_dir = source_base_root / fam / name
        dst_dir = dest_base_root / fam / name
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        if dst_dir.is_symlink():
            if dst_dir.resolve() == src_dir.resolve():
                linked.append(base_rel_path)
                continue
            dst_dir.unlink()
        elif dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.symlink_to(src_dir, target_is_directory=True)
        linked.append(base_rel_path)

    summary = {
        "source_results_root": str(source_root),
        "dest_results_root": str(dest_root),
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "requested_count": len(requested_bases),
        "ready_count": len(ready),
        "linked_count": len(linked),
        "missing_or_bad": missing_or_bad,
        "linked_bases": linked,
        "finished_at": time.time(),
    }
    save_json(dest_root / "base_reuse_prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
