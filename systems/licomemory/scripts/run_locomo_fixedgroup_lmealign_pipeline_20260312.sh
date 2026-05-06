#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
PY="${PYTHON_BIN:-python}"
RUNNER="$SCRIPT_DIR/run_longmemeval_runner.py"
BUILDER="$SCRIPT_DIR/build_locomo_fixed_group_lmealign_dataset.py"
REPLICATOR="$SCRIPT_DIR/replicate_locomo_group_graphs.py"
ENV_FILE="${ENV_FILE:-$REPO/.env}"

SRC_ROOT="${SRC_ROOT:-}"
SRC_MANIFEST="${SRC_MANIFEST:-}"
EXTERNAL_FILLER_ROOT="${EXTERNAL_FILLER_ROOT:-${LICOMEMORY_LONGMEMEVAL_SOURCE_ROOT:-}}"
DATA_ROOT="${DATA_ROOT:-${LICOMEMORY_LOCOMO_DATA_ROOT:-}}"
LIST_ROOT="${LIST_ROOT:-$SCRIPT_DIR/dataset_lists/locomo_multihop282_fixedgroup_sbins_lmealign_20260315}"

BUILD_CFG="$REPO/config/licomemory_locomo_multihop282_fixedgroup_lmealign_gpt5mini_openai_build_q0_20260312.yaml"
QUERY_CFG="$REPO/config/licomemory_locomo_multihop282_fixedgroup_lmealign_qwen3_8b_ifopen_query_q1_llmeval_gpt4omini_memos_20260317.yaml"

RUN_ID="${1:-locomo_multihop282_fixedgroup_lmealign_reactagent_gpt5mini_judge4omini_20260315}"
ROOT_PREFIX="$RUN_ID"
RESULT_ROOT="$REPO/results/$ROOT_PREFIX"
LOG_ROOT="$REPO/logs/$ROOT_PREFIX"

Q0_CSV="$RESULT_ROOT/trace_q0_representatives.csv"
Q0_CKPT="$RESULT_ROOT/checkpoint_q0_representatives.json"
Q1_CSV="$RESULT_ROOT/trace_q1_full.csv"
Q1_CKPT="$RESULT_ROOT/checkpoint_q1_full.json"
REPL_SUMMARY_DIR="$RESULT_ROOT/replication_summaries"
KEEP_STAGE_GRAPHS="${KEEP_STAGE_GRAPHS:-1}"
STAGES="${STAGES:-}"

Q0_WORKERS="${Q0_WORKERS:-1}"
Q1_WORKERS="${Q1_WORKERS:-3}"
TIMEOUT_Q0="${TIMEOUT_Q0:-21600}"
TIMEOUT_Q1="${TIMEOUT_Q1:-7200}"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT" "$REPL_SUMMARY_DIR"

load_env() {
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "[pipeline] missing env file: $ENV_FILE" >&2
    exit 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a

  export OPENAI_API_KEY="${OPENAI_API_KEY:-${GPT_API_KEY:-}}"
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${GPT_BASE_URL:-https://api.openai.com/v1}}"
  export LICOMEMORY_FAIL_FAST_QUERY_ERRORS="${LICOMEMORY_FAIL_FAST_QUERY_ERRORS:-1}"
  export LICOMEMORY_LLM_FAIL_ON_ERROR="${LICOMEMORY_LLM_FAIL_ON_ERROR:-1}"
  export LICOMEMORY_LLM_RETRY_ATTEMPTS="${LICOMEMORY_LLM_RETRY_ATTEMPTS:-20}"
  export LICOMEMORY_LLM_RETRY_BACKOFF="${LICOMEMORY_LLM_RETRY_BACKOFF:-1.5}"
  export LICOMEMORY_LLM_RETRY_BACKOFF_MAX="${LICOMEMORY_LLM_RETRY_BACKOFF_MAX:-120.0}"
  export LICOMEMORY_EMBEDDING_RETRY_ATTEMPTS="${LICOMEMORY_EMBEDDING_RETRY_ATTEMPTS:-6}"
  export LICOMEMORY_EMBEDDING_RETRY_BACKOFF="${LICOMEMORY_EMBEDDING_RETRY_BACKOFF:-1.0}"
  export LICOMEMORY_EMBEDDING_RETRY_BACKOFF_MAX="${LICOMEMORY_EMBEDDING_RETRY_BACKOFF_MAX:-30.0}"

  export QUERY_API_KEY="${QUERY_API_KEY:-${QWEN_API:-${OTHER_API_KEY:-}}}"
  export QUERY_BASE_URL="${QUERY_BASE_URL:-${OTHER_BASE_URL:-}}"
  export JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"
  export JUDGE_BASE_URL="${JUDGE_BASE_URL:-${OPENAI_BASE_URL:-}}"

  if [[ -z "${OPENAI_API_KEY:-}" || -z "${QUERY_API_KEY:-}" || -z "${QUERY_BASE_URL:-}" ]]; then
    echo "[pipeline] required API variables are not set after sourcing $ENV_FILE" >&2
    exit 1
  fi
}

build_dataset() {
  if [[ -z "$SRC_ROOT" || -z "$SRC_MANIFEST" || -z "$EXTERNAL_FILLER_ROOT" || -z "$DATA_ROOT" ]]; then
    echo "[pipeline] dataset build inputs are incomplete. Set SRC_ROOT, SRC_MANIFEST, EXTERNAL_FILLER_ROOT, and DATA_ROOT." >&2
    exit 1
  fi
  if [[ -d "$DATA_ROOT" && -f "$LIST_ROOT/summary.json" ]]; then
    echo "[pipeline] dataset already exists: $DATA_ROOT"
    return
  fi

  echo "[pipeline] building lme-aligned fixed-group sbins dataset"
  "$PY" "$BUILDER" \
    --src-root "$SRC_ROOT" \
    --manifest "$SRC_MANIFEST" \
    --external-filler-root "$EXTERNAL_FILLER_ROOT" \
    --out-data-root "$DATA_ROOT" \
    --out-list-root "$LIST_ROOT" \
    --bins 0,100,200,300,400 \
    --replicas 1 \
    --batch-size 50 \
    --smoke-size 20 \
    --seed 20260312
}

run_q0_for_stage() {
  local stage="$1"
  local list_file="$LIST_ROOT/${stage}_representatives.txt"
  echo "[pipeline] q0 build stage=$stage list=$list_file"
  "$PY" "$RUNNER" \
    --repo-root "$REPO" \
    --data-root "$DATA_ROOT" \
    --config "$BUILD_CFG" \
    --python-bin "$PY" \
    --root-prefix "$ROOT_PREFIX" \
    --log-dir "$LOG_ROOT/q0_${stage}" \
    --csv-out "$Q0_CSV" \
    --checkpoint "$Q0_CKPT" \
    --dataset-list "$list_file" \
    --workers "$Q0_WORKERS" \
    --max-retries 0 \
    --retry-backoff 5 \
    --timeout-sec "$TIMEOUT_Q0" \
    --query 0 \
    --graph-file dynamic_memory_graph.pkl \
    --require-graph-nonempty \
    --stop-on-fail
}

replicate_stage_graphs() {
  local stage="$1"
  echo "[pipeline] replicate graphs stage=$stage"
  "$PY" "$REPLICATOR" \
    --repo-root "$REPO" \
    --root-prefix "$ROOT_PREFIX" \
    --manifest "$LIST_ROOT/manifest_by_stage/${stage}.jsonl" \
    --group-build-list "$LIST_ROOT/${stage}_representatives.txt" \
    --graph-file dynamic_memory_graph.pkl \
    --key-fields group_id,bin_s,replica \
    --out-summary "$REPL_SUMMARY_DIR/${stage}.json"
}

run_q1_list() {
  local stage="$1"
  local list_file="$2"
  local tag="$3"
  echo "[pipeline] q1 query stage=$stage tag=$tag list=$list_file"
  "$PY" "$RUNNER" \
    --repo-root "$REPO" \
    --data-root "$DATA_ROOT" \
    --config "$QUERY_CFG" \
    --python-bin "$PY" \
    --root-prefix "$ROOT_PREFIX" \
    --log-dir "$LOG_ROOT/q1_${tag}_${stage}" \
    --csv-out "$Q1_CSV" \
    --checkpoint "$Q1_CKPT" \
    --dataset-list "$list_file" \
    --workers "$Q1_WORKERS" \
    --max-retries 0 \
    --retry-backoff 5 \
    --timeout-sec "$TIMEOUT_Q1" \
    --query 1 \
    --graph-file dynamic_memory_graph.pkl \
    --require-graph-nonempty \
    --stop-on-fail
}

_stage_trace_check_py() {
  cat <<'PY'
import csv
import json
import sys
from pathlib import Path

_, trace_csv_arg, expected_list_arg, stage = sys.argv
csv.field_size_limit(sys.maxsize)
trace_csv = Path(trace_csv_arg)
expected_list = Path(expected_list_arg)

if not trace_csv.exists() or not expected_list.exists():
    raise SystemExit(2)

expected = [line.strip() for line in expected_list.read_text(encoding="utf-8").splitlines() if line.strip()]
expected_set = set(expected)
present = []
issues = {
    "empty_agent_output": 0,
    "complete_literal_output": 0,
    "top_level_error_output": 0,
    "retrieval_calls_mismatch": 0,
    "react_steps_mismatch": 0,
    "finish_step_mismatch": 0,
    "retrieved_sessions_mismatch": 0,
    "duplicate_top_session_ids": 0,
    "final_answer_output_mismatch": 0,
    "memory_unit_mismatch": 0,
    "noise_mode_mismatch": 0,
    "question_type_mismatch": 0,
    "legacy_trace_fields_present": 0,
    "missing_memos_stats": 0,
    "memos_stats_mismatch": 0,
    "judge_runs_mismatch": 0,
}

for row in csv.DictReader(trace_csv.open("r", encoding="utf-8", newline="")):
    dataset_name = str(row.get("dataset_name", "") or "")
    if not dataset_name.startswith(stage + "/"):
        continue
    present.append(dataset_name)
    trace = json.loads(row.get("trace_json") or "{}")
    react_trace = trace.get("react_trace") or []
    last_step = react_trace[-1] if react_trace else {}
    top_session_ids = trace.get("top_session_ids") or []
    memos_stats = trace.get("memos_stats") or {}
    llm_judgments = trace.get("llm_judgments") or {}
    agent_output = str(row.get("agent_output", "") or "").strip()
    final_answer = str(last_step.get("final_answer") or "").strip()
    if agent_output == "":
        issues["empty_agent_output"] += 1
    if agent_output.lower() == "complete":
        issues["complete_literal_output"] += 1
    if "Error processing query" in agent_output or "Traceback" in agent_output:
        issues["top_level_error_output"] += 1
    if int(float(row.get("retrieval_calls") or 0)) != sum(1 for step in react_trace if step.get("action") == "retrieve"):
        issues["retrieval_calls_mismatch"] += 1
    if int(float(row.get("react_steps") or 0)) != len(react_trace):
        issues["react_steps_mismatch"] += 1
    if sum(1 for step in react_trace if step.get("action") == "finish") != 1:
        issues["finish_step_mismatch"] += 1
    if int(float(row.get("retrieved_sessions") or 0)) != len(top_session_ids):
        issues["retrieved_sessions_mismatch"] += 1
    if len(top_session_ids) != len({str(x) for x in top_session_ids}):
        issues["duplicate_top_session_ids"] += 1
    if final_answer != agent_output:
        issues["final_answer_output_mismatch"] += 1
    if str(trace.get("memory_unit") or "") != "group":
        issues["memory_unit_mismatch"] += 1
    if str(trace.get("noise_mode") or "") != "outdomain_longmemeval":
        issues["noise_mode_mismatch"] += 1
    if str(trace.get("question_type") or "") != "locomo-multi-hop":
        issues["question_type_mismatch"] += 1
    if "query_summary" in trace or "cost_summary" in trace:
        issues["legacy_trace_fields_present"] += 1
    if not memos_stats:
        issues["missing_memos_stats"] += 1
    else:
        try:
            if float(memos_stats.get("context_tokens", -1)) != float(row.get("context_tokens") or -1):
                issues["memos_stats_mismatch"] += 1
            elif float(memos_stats.get("response_duration_ms", -1)) != float(row.get("response_duration_ms") or -1):
                issues["memos_stats_mismatch"] += 1
            elif float(memos_stats.get("search_duration_ms", -1)) != float(row.get("search_duration_ms") or -1):
                issues["memos_stats_mismatch"] += 1
            elif float(memos_stats.get("total_duration_ms", -1)) != float(row.get("total_duration_ms") or -1):
                issues["memos_stats_mismatch"] += 1
        except Exception:
            issues["memos_stats_mismatch"] += 1
    if sorted(llm_judgments.keys()) != ["judgment_1", "judgment_2", "judgment_3"]:
        issues["judge_runs_mismatch"] += 1

present_set = set(present)
missing = sorted(expected_set - present_set)
extra = sorted(present_set - expected_set)
bad = {k: v for k, v in issues.items() if v}

summary = {
    "stage": stage,
    "expected_count": len(expected),
    "present_count": len(present),
    "missing_count": len(missing),
    "extra_count": len(extra),
    "issues": issues,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))

if missing or extra or bad:
    if missing:
        print(f"[pipeline] missing first 5: {missing[:5]}", file=sys.stderr)
    if extra:
        print(f"[pipeline] extra first 5: {extra[:5]}", file=sys.stderr)
    raise SystemExit(1)
PY
}

is_stage_complete() {
  local stage="$1"
  local expected_list="$LIST_ROOT/${stage}_all.txt"
  "$PY" - "$Q1_CSV" "$expected_list" "$stage" >/dev/null 2>&1 <<PY
$(_stage_trace_check_py)
PY
}

audit_stage_trace() {
  local stage="$1"
  local expected_list="$LIST_ROOT/${stage}_all.txt"
  echo "[pipeline] audit trace stage=$stage"
  "$PY" - "$Q1_CSV" "$expected_list" "$stage" <<PY
$(_stage_trace_check_py)
PY
}

cleanup_stage_graphs() {
  local stage="$1"
  if [[ "$KEEP_STAGE_GRAPHS" == "1" ]]; then
    echo "[pipeline] keep graphs stage=$stage"
    return
  fi
  echo "[pipeline] cleanup graphs stage=$stage"
  "$PY" - "$DATA_ROOT" "$RESULT_ROOT" "$stage" <<'PY'
import sys
from pathlib import Path

_, data_root_arg, result_root_arg, stage = sys.argv
data_root = Path(data_root_arg)
result_root = Path(result_root_arg)
removed = 0
freed = 0
for graph_path in (result_root / stage / "locomo_multihop").rglob("dynamic_memory_graph.pkl"):
    try:
        stat = graph_path.stat()
    except FileNotFoundError:
        continue
    if stat.st_nlink <= 1:
        freed += stat.st_size
    graph_path.unlink(missing_ok=True)
    removed += 1
print(f"[pipeline] cleanup removed={removed} approx_freed_bytes={freed}")
PY
}

cleanup_stage_query_outputs() {
  local stage="$1"
  echo "[pipeline] cleanup query outputs stage=$stage"
  "$PY" - "$RESULT_ROOT" "$stage" <<'PY'
import sys
from pathlib import Path

_, result_root_arg, stage = sys.argv
result_root = Path(result_root_arg)
stage_root = result_root / stage
removed_files = 0
removed_bytes = 0
removed_dirs = 0

for path in stage_root.rglob("*"):
    if not path.is_file():
        continue
    if path.name not in {"results.json", "metrics.json"}:
        continue
    try:
        removed_bytes += path.stat().st_size
    except FileNotFoundError:
        pass
    path.unlink(missing_ok=True)
    removed_files += 1

for results_dir in sorted(stage_root.rglob("results"), reverse=True):
    if not results_dir.is_dir():
        continue
    try:
        next(results_dir.iterdir())
    except StopIteration:
        results_dir.rmdir()
        removed_dirs += 1
    except Exception:
        continue

print(
    f"[pipeline] cleanup query outputs removed_files={removed_files} "
    f"removed_dirs={removed_dirs} approx_freed_bytes={removed_bytes}"
)
PY
}

main() {
  load_env
  build_dataset

  mapfile -t stages < <(find "$LIST_ROOT" -maxdepth 1 -type f -name 's*_r*_representatives.txt' | sort)
  if [[ "${#stages[@]}" -eq 0 ]]; then
    echo "[pipeline] no stage representative lists found in $LIST_ROOT" >&2
    exit 1
  fi
  if [[ -n "$STAGES" ]]; then
    IFS=',' read -r -a requested_stages <<< "$STAGES"
    filtered=()
    for rep_file in "${stages[@]}"; do
      stage="$(basename "$rep_file" | sed 's/_representatives\.txt$//')"
      for requested in "${requested_stages[@]}"; do
        if [[ "$stage" == "$requested" ]]; then
          filtered+=("$rep_file")
          break
        fi
      done
    done
    stages=("${filtered[@]}")
    if [[ "${#stages[@]}" -eq 0 ]]; then
      echo "[pipeline] STAGES filter matched no stage lists: $STAGES" >&2
      exit 1
    fi
  fi

  for rep_file in "${stages[@]}"; do
    stage="$(basename "$rep_file" | sed 's/_representatives\.txt$//')"
    echo "[pipeline] prebuild graphs stage=$stage"
    run_q0_for_stage "$stage"
    replicate_stage_graphs "$stage"
  done

  for rep_file in "${stages[@]}"; do
    stage="$(basename "$rep_file" | sed 's/_representatives\.txt$//')"
    if is_stage_complete "$stage"; then
      echo "[pipeline] stage already complete, skipping q1: $stage"
      continue
    fi
    run_q1_list "$stage" "$LIST_ROOT/${stage}_smoke20.txt" "smoke"

    mapfile -t stage_batches < <(find "$LIST_ROOT" -maxdepth 1 -type f -name "${stage}_batch*_50.txt" | sort)
    if [[ "${#stage_batches[@]}" -eq 0 ]]; then
      echo "[pipeline] no batch files found for stage=$stage" >&2
      exit 1
    fi
    for batch_file in "${stage_batches[@]}"; do
      run_q1_list "$stage" "$batch_file" "$(basename "$batch_file" .txt)"
    done
    audit_stage_trace "$stage"
    cleanup_stage_query_outputs "$stage"
    cleanup_stage_graphs "$stage"
  done

  echo "[pipeline] done run_id=$RUN_ID"
}

main "$@"
