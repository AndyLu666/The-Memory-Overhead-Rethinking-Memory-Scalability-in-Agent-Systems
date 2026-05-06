#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PY="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-$ROOT/config/licomemory_longmemeval_fixed2k_main3m_qwen3_8b_ifopen_query_q1_llmeval_gpt4omini_memos_steps6_tmp_20260412.yaml}"
DATA_ROOT="${DATA_ROOT:-${LICOMEMORY_LONGMEMEVAL_DATA_ROOT:-}}"
STAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$ROOT/logs/longmemeval_full_${STAMP}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" | tee -a "$LOG_DIR/run.log"
  exit 1
fi

if [[ -z "$DATA_ROOT" || ! -d "$DATA_ROOT" ]]; then
  echo "Data root not found: $DATA_ROOT" | tee -a "$LOG_DIR/run.log"
  exit 1
fi

mapfile -t ITEMS < <(cd "$DATA_ROOT" && find . -type d -name 'lm_*' | sort)

echo "[$(date)] Found ${#ITEMS[@]} items under $DATA_ROOT" | tee -a "$LOG_DIR/run.log"

for item in "${ITEMS[@]}"; do
  name=${item#./}
  echo "[$(date)] Running $name" | tee -a "$LOG_DIR/run.log"
  log_file="$LOG_DIR/${name//\//_}.log"
  "$PY" "$ROOT/main.py" -opt "$CONFIG" -dataset_name "$name" -root "$name" -query 1 >> "$log_file" 2>&1
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "[$(date)] FAILED $name (exit=$status)" | tee -a "$LOG_DIR/run.log"
  else
    echo "[$(date)] DONE $name" | tee -a "$LOG_DIR/run.log"
  fi
  sleep 1

done

echo "[$(date)] All items completed" | tee -a "$LOG_DIR/run.log"
