#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <key1|key2> <dataset-list> <results-root> <parallelism>" >&2
  exit 2
fi

KEY_SLOT="$1"
DATASET_LIST="$2"
RESULTS_ROOT="$3"
PARALLELISM="$4"

PYTHON_BIN="${PYTHON_BIN:-python}"
OPENCLAW_REPO="${OPENCLAW_REPO:-$PWD/openclaw}"
DATA_ROOT="${DATA_ROOT:-$PWD/data/fixed2k_sbins_fixed2k_main3m_20260224_102211}"
ENV_FILE="${ENV_FILE:-$PWD/.env}"
# Keep OpenRouter for Qwen 235B chat, while allowing retrieval/evaluation
# endpoints to be configured independently.
EMBEDDING_BASE_URL_ENV="${EMBEDDING_BASE_URL_ENV:-OTHER_BASE_URL}"
EMBEDDING_API_KEY_ENV="${EMBEDDING_API_KEY_ENV:-OTHER_API_KEY}"
EVAL_BASE_URL_ENV="${EVAL_BASE_URL_ENV:-OTHER_BASE_URL}"
EVAL_API_KEY_ENV="${EVAL_API_KEY_ENV:-OTHER_API_KEY}"
# The remaining s300/s400 document-memory q0 jobs are large enough that some
# official remote embedding batches can exceed the default 120s wall clock.
# These knobs only control embedding request batching/waiting; they do not
# change memory rendering, chunking, retrieval scoring, the agent loop, or judging.
export OPENCLAW_MEMORY_EMBEDDING_BATCH_MAX_TOKENS="${OPENCLAW_MEMORY_EMBEDDING_BATCH_MAX_TOKENS:-8000}"
export OPENCLAW_MEMORY_EMBEDDING_BATCH_TIMEOUT_REMOTE_MS="${OPENCLAW_MEMORY_EMBEDDING_BATCH_TIMEOUT_REMOTE_MS:-300000}"
ITEM_TIMEOUT_SECONDS="${ITEM_TIMEOUT_SECONDS:-3600}"

set -a
source "$ENV_FILE"
set +a

case "$KEY_SLOT" in
  key1)
    ;;
  key2)
    export OPENROUTER_API_KEY="${OPENROUTER_API_KEY2:?OPENROUTER_API_KEY2 missing in env file}"
    ;;
  *)
    echo "unknown key slot: $KEY_SLOT" >&2
    exit 2
    ;;
esac

mkdir -p "$RESULTS_ROOT"
cd "$OPENCLAW_REPO"

"$PYTHON_BIN" benchmarks/memory_mvp/run_openclaw_benchmark.py \
  --data-root "$DATA_ROOT" \
  --dataset-list "$DATASET_LIST" \
  --results-root "$RESULTS_ROOT" \
  --env-file "$ENV_FILE" \
  --chat-model qwen/qwen3-235b-a22b-2507 \
  --embedding-model text-embedding-3-small \
  --eval-model gpt-4o-mini \
  --eval-prompt-style memos_json \
  --eval-num-runs 3 \
  --memory-backend official \
  --memory-agent-profile openclaw_fidelity \
  --agent-mode memory_tools \
  --max-agent-steps 6 \
  --force-min-memory-searches 1 \
  --sources memory \
  --top-k 6 \
  --chunk-tokens 400 \
  --chunk-overlap 80 \
  --candidate-multiplier 4 \
  --vector-weight 0.7 \
  --text-weight 0.3 \
  --chat-base-url-env OPENROUTER_BASE_URL \
  --chat-api-key-env OPENROUTER_API_KEY \
  --embedding-base-url-env "$EMBEDDING_BASE_URL_ENV" \
  --embedding-api-key-env "$EMBEDDING_API_KEY_ENV" \
  --eval-base-url-env "$EVAL_BASE_URL_ENV" \
  --eval-api-key-env "$EVAL_API_KEY_ENV" \
  --openclaw-repo-root "$OPENCLAW_REPO" \
  --chat-max-tokens 512 \
  --parallelism "$PARALLELISM" \
  --item-max-attempts 4 \
  --retry-delay-seconds 45 \
  --item-timeout-seconds "$ITEM_TIMEOUT_SECONDS" \
  --cleanup-q0-after-item \
  --continue-on-error
