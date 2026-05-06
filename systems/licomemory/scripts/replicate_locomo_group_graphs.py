#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_key(row: dict, key_fields: List[str]) -> str:
    values = [str(row.get(field, "")) for field in key_fields]
    return "||".join(values)


def link_or_copy(src: Path, dst: Path) -> str:
    """Prefer hard links to avoid duplicating large graph files."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replicate one built LoCoMo graph per group to all same-group question roots."
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--root-prefix", required=True)
    parser.add_argument("--manifest", required=True, help="Question-layout export manifest JSONL")
    parser.add_argument("--group-build-list", required=True, help="Representative dataset list used for q0")
    parser.add_argument("--graph-file", default="dynamic_memory_graph.pkl")
    parser.add_argument(
        "--key-fields",
        default="group_id",
        help="Comma-separated manifest fields that define a shared memory unit (default: group_id)",
    )
    parser.add_argument("--out-summary", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_rows = load_jsonl(Path(args.manifest).resolve())
    group_build_list = [
        line.strip()
        for line in Path(args.group_build_list).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    key_fields = [field.strip() for field in args.key_fields.split(",") if field.strip()]
    if not key_fields:
        raise ValueError("No key fields provided")

    rep_by_group: Dict[str, str] = {}
    for rel_path in group_build_list:
        match = next((row for row in manifest_rows if row.get("rel_path") == rel_path), None)
        if match is None:
            raise ValueError(f"Representative dataset not found in manifest: {rel_path}")
        rep_by_group[build_key(match, key_fields)] = rel_path

    copied = 0
    replication_mode = "none"
    skipped = 0
    rows_by_group: Dict[str, List[dict]] = {}
    for row in manifest_rows:
        rows_by_group.setdefault(build_key(row, key_fields), []).append(row)

    for group_id, rows in rows_by_group.items():
        rep_rel_path = rep_by_group.get(group_id)
        if not rep_rel_path:
            raise ValueError(f"No representative graph source found for group {group_id}")
        src_graph = repo_root / "results" / args.root_prefix / rep_rel_path / args.graph_file
        if not src_graph.exists():
            raise FileNotFoundError(f"Representative graph missing: {src_graph}")

        for row in rows:
            dst_graph = repo_root / "results" / args.root_prefix / row["rel_path"] / args.graph_file
            if dst_graph.resolve() == src_graph.resolve():
                skipped += 1
                continue
            mode = link_or_copy(src_graph, dst_graph)
            if replication_mode == "none":
                replication_mode = mode
            copied += 1

    summary = {
        "repo_root": str(repo_root),
        "root_prefix": args.root_prefix,
        "manifest": str(Path(args.manifest).resolve()),
        "group_build_list": str(Path(args.group_build_list).resolve()),
        "graph_file": args.graph_file,
        "key_fields": key_fields,
        "replication_mode": replication_mode,
        "groups": len(rows_by_group),
        "representatives": len(rep_by_group),
        "copied": copied,
        "skipped_representatives": skipped,
    }
    out_summary = Path(args.out_summary).resolve()
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
