#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PWD}"
RECOVERY_DIR="${RECOVERY_DIR:-$ARTIFACT_ROOT/results/openclaw/qwen235b_forced_retrieve_recovery}"
RESULTS_BASE="${RESULTS_BASE:-$ARTIFACT_ROOT/results/openclaw}"
TARGET_LIST="${TARGET_LIST:-$RECOVERY_DIR/remaining_after_p1_batchfix_20260501_batchfix_p5_all70.txt}"
RUN_SCRIPT="${RUN_SCRIPT:-$ARTIFACT_ROOT/systems/openclaw/scripts/run_qwen235b_forced_openrouter.sh}"
ROLLUP_SCRIPT="${ROLLUP_SCRIPT:-$ARTIFACT_ROOT/systems/openclaw/scripts/rollup_qwen235b_forced_retrieve.py}"
STATUS_JSON="${STATUS_JSON:-$RECOVERY_DIR/supervisor_qwen235b_forced_remaining70_status.json}"
ROUND_FILE="${ROUND_FILE:-$RECOVERY_DIR/supervisor_qwen235b_forced_remaining70_round.txt}"
POLL_SECONDS="${POLL_SECONDS:-600}"
PARALLELISM="${PARALLELISM:-2}"

mkdir -p "$RECOVERY_DIR"

write_status() {
  TARGET_LIST="$TARGET_LIST" RESULTS_BASE="$RESULTS_BASE" RECOVERY_DIR="$RECOVERY_DIR" STATUS_JSON="$STATUS_JSON" python - <<'PY'
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

csv.field_size_limit(sys.maxsize)

target_list = Path(os.environ["TARGET_LIST"])
results_base = Path(os.environ["RESULTS_BASE"])
recovery_dir = Path(os.environ["RECOVERY_DIR"])
status_json = Path(os.environ["STATUS_JSON"])

targets = [line.strip() for line in target_list.read_text(encoding="utf-8").splitlines() if line.strip()]
target_set = set(targets)

root_patterns = [
    "qwen235b_forced_retrieve_openrouter_key*_otherembed_remaining35_stable_p2_20260502_v1",
    "qwen235b_forced_retrieve_openrouter_key*_otherembed_autoresume_p2_20260502_round*",
]
roots = []
for pattern in root_patterns:
    roots.extend(sorted(results_base.glob(pattern)))
roots = sorted(set(roots), key=lambda p: p.name)

accepted: dict[str, dict] = {}
dirty_rows = []
duplicate_rows = 0
success = 0
root_counts = {}
failures = []

for root in roots:
    trace_path = root / "trace_q1_full.csv"
    count = 0
    if trace_path.exists():
        with trace_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                dataset = (row.get("dataset_name") or "").strip()
                if dataset not in target_set:
                    continue
                count += 1
                retrieval = str(row.get("retrieval_calls") or "").strip()
                if retrieval in ("", "0", "0.0"):
                    dirty_rows.append(
                        {
                            "dataset_name": dataset,
                            "root": root.name,
                            "issue": "retrieval_calls_zero_or_missing",
                        }
                    )
                    continue
                if dataset in accepted:
                    duplicate_rows += 1
                accepted[dataset] = {"root": root.name, "row": row}
    for meta_path in root.glob("derived/*/longmem_*/*/bridge_failures/*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {"error_message": "unreadable_meta"}
        failures.append(
            {
                "root": root.name,
                "path": str(meta_path),
                "dataset_name": meta.get("dataset_name", ""),
                "error_message": meta.get("error_message", meta.get("error_type", "")),
            }
        )
    root_counts[root.name] = count

for item in accepted.values():
    if str(item["row"].get("success") or "").strip() == "1":
        success += 1

remaining = [dataset for dataset in targets if dataset not in accepted]
remaining_path = recovery_dir / "supervisor_qwen235b_forced_remaining70_remaining.txt"
key1_path = recovery_dir / "supervisor_qwen235b_forced_remaining70_remaining_key1.txt"
key2_path = recovery_dir / "supervisor_qwen235b_forced_remaining70_remaining_key2.txt"
remaining_path.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
key1 = remaining[0::2]
key2 = remaining[1::2]
key1_path.write_text("\n".join(key1) + ("\n" if key1 else ""), encoding="utf-8")
key2_path.write_text("\n".join(key2) + ("\n" if key2 else ""), encoding="utf-8")

by_s = {}
for dataset, item in accepted.items():
    sbin = dataset.split("/", 1)[0]
    by_s.setdefault(sbin, {"accepted": 0, "success": 0})
    by_s[sbin]["accepted"] += 1
    if str(item["row"].get("success") or "").strip() == "1":
        by_s[sbin]["success"] += 1

status = {
    "checked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "target_count": len(targets),
    "accepted": len(accepted),
    "success": success,
    "success_rate": round(success / len(accepted), 6) if accepted else None,
    "remaining": len(remaining),
    "dirty_rows": len(dirty_rows),
    "duplicate_rows": duplicate_rows,
    "failures": len(failures),
    "recent_failures": failures[-10:],
    "by_s": by_s,
    "root_counts": root_counts,
    "remaining_paths": {
        "all": str(remaining_path),
        "key1": str(key1_path),
        "key2": str(key2_path),
    },
    "state": "complete" if not remaining else "needs_more",
}
status_json.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(status, ensure_ascii=False))
PY
}

active_runner_sessions() {
  tmux ls 2>/dev/null | grep -E 'openclaw235b_(stable|autoresume).*_p2' || true
}

start_autoresume_round() {
  local round="$1"
  local key1_list="$RECOVERY_DIR/supervisor_qwen235b_forced_remaining70_remaining_key1.txt"
  local key2_list="$RECOVERY_DIR/supervisor_qwen235b_forced_remaining70_remaining_key2.txt"
  local key1_count key2_count
  key1_count="$(grep -cve '^[[:space:]]*$' "$key1_list" || true)"
  key2_count="$(grep -cve '^[[:space:]]*$' "$key2_list" || true)"

  if [[ "$key1_count" -gt 0 ]]; then
    local root1="$RESULTS_BASE/qwen235b_forced_retrieve_openrouter_key1_otherembed_autoresume_p2_20260502_round${round}"
    mkdir -p "$root1"
    tmux new-session -d -s "openclaw235b_autoresume_r${round}_key1_p2" \
      "cd '$ARTIFACT_ROOT' && OPENCLAW_MEMORY_EMBEDDING_BATCH_MAX_TOKENS=8000 OPENCLAW_MEMORY_EMBEDDING_BATCH_TIMEOUT_REMOTE_MS=300000 ITEM_TIMEOUT_SECONDS=5400 EMBEDDING_BASE_URL_ENV=OTHER_BASE_URL EMBEDDING_API_KEY_ENV=OTHER_API_KEY EVAL_BASE_URL_ENV=OTHER_BASE_URL EVAL_API_KEY_ENV=OTHER_API_KEY '$RUN_SCRIPT' key1 '$key1_list' '$root1' '$PARALLELISM' > '$root1/launcher.log' 2>&1; echo EXIT:\$? >> '$root1/launcher.log'"
  fi

  if [[ "$key2_count" -gt 0 ]]; then
    local root2="$RESULTS_BASE/qwen235b_forced_retrieve_openrouter_key2_otherembed_autoresume_p2_20260502_round${round}"
    mkdir -p "$root2"
    tmux new-session -d -s "openclaw235b_autoresume_r${round}_key2_p2" \
      "cd '$ARTIFACT_ROOT' && OPENCLAW_MEMORY_EMBEDDING_BATCH_MAX_TOKENS=8000 OPENCLAW_MEMORY_EMBEDDING_BATCH_TIMEOUT_REMOTE_MS=300000 ITEM_TIMEOUT_SECONDS=5400 EMBEDDING_BASE_URL_ENV=OTHER_BASE_URL EMBEDDING_API_KEY_ENV=OTHER_API_KEY EVAL_BASE_URL_ENV=OTHER_BASE_URL EVAL_API_KEY_ENV=OTHER_API_KEY '$RUN_SCRIPT' key2 '$key2_list' '$root2' '$PARALLELISM' > '$root2/launcher.log' 2>&1; echo EXIT:\$? >> '$root2/launcher.log'"
  fi
}

if [[ ! -f "$ROUND_FILE" ]]; then
  echo 0 > "$ROUND_FILE"
fi

while true; do
  status_line="$(write_status)"
  if [[ -x "$ROLLUP_SCRIPT" || -f "$ROLLUP_SCRIPT" ]]; then
    python3 "$ROLLUP_SCRIPT" >/dev/null || true
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $status_line"
  state="$(python - <<PY
import json
print(json.load(open("$STATUS_JSON"))["state"])
PY
)"
  remaining="$(python - <<PY
import json
print(json.load(open("$STATUS_JSON"))["remaining"])
PY
)"

  if [[ "$state" == "complete" ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) complete"
    exit 0
  fi

  if [[ -z "$(active_runner_sessions)" ]]; then
    round="$(cat "$ROUND_FILE")"
    round="$((round + 1))"
    echo "$round" > "$ROUND_FILE"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting autoresume round $round for $remaining remaining"
    start_autoresume_round "$round"
  fi

  sleep "$POLL_SECONDS"
done
