# OpenClaw Document-Memory Benchmark

This directory contains the OpenClaw memory benchmark code used for the
LongMemEval document-memory experiments.

The harness uses OpenClaw's official `memory-core` for indexing, `memory_search`,
`memory_get`, and prompt-section generation. The Python runner only handles
benchmark I/O, answer generation, judging, tracing, retries, and packaging.

## Directory Layout

- `benchmarks/memory_mvp/`: benchmark runner, single-item probe, OpenClaw memory
  backend wrapper, and Node bridge.
- `benchmarks/memory_mvp/dataset_lists/`: final LongMemEval fixed2k balanced
  1000-item list used in the reported runs.
- `scripts/run_qwen235b_forced_openrouter.sh`: Qwen-235B forced-retrieval run
  helper.
- `scripts/supervise_qwen235b_forced_remaining.sh`: resumable two-key
  supervisor for forced-retrieval tail runs.
- `scripts/rollup_qwen235b_forced_retrieve.py`: utility used to roll up
  validated forced-retrieval 235B rows.
- `scripts/reconstruct_qwen235b_forced_costs.py`: offline cost reconstruction
  for the forced-retrieval 235B package.
- `scripts/build_qwen_openclaw_forced235b_package.py`: offline package builder
  that replaces the original 235B slice with the forced-retrieval rollup and
  regenerates combined traces, metrics, manifests, and cost summaries.
- `patches/extensions/memory-core/src/memory/manager-embedding-ops.ts`:
  patched OpenClaw core file that lets embedding batch token limits and remote
  timeouts be controlled by environment variables.
- `setup_openclaw_artifact.sh`: copies this benchmark into an OpenClaw checkout
  and applies the patch.

## Setup

Clone OpenClaw separately, then install the artifact code into that checkout:

```bash
git clone https://github.com/openclaw/openclaw openclaw
OPENCLAW_REPO=$PWD/openclaw bash systems/openclaw/setup_openclaw_artifact.sh
```

Install Python dependencies:

```bash
python -m pip install -r openclaw/benchmarks/memory_mvp/requirements.txt
```

Create a local `.env` from the example:

```bash
cp systems/openclaw/env.example .env
```

Fill in only the provider variables needed for the model being run. Do not
commit `.env`.

## Single-Item Probe

```bash
python openclaw/benchmarks/memory_mvp/probe_dataset_item.py \
  --corpus-json /path/to/Corpus.json \
  --question-json /path/to/Question.json \
  --output-dir results/openclaw_probe \
  --env-file .env \
  --memory-backend official \
  --memory-agent-profile openclaw_fidelity \
  --sources memory \
  --chat-model qwen3-8b \
  --embedding-model text-embedding-3-small \
  --eval-model gpt-4o-mini
```

## LongMemEval Batch Run

```bash
python openclaw/benchmarks/memory_mvp/run_openclaw_benchmark.py \
  --data-root /path/to/fixed2k_sbins_fixed2k_main3m_20260224_102211 \
  --dataset-list openclaw/benchmarks/memory_mvp/dataset_lists/longmemeval_fixed2k_fidelity_balanced50x4x5_seed20260421.txt \
  --results-root results/openclaw_qwen8b \
  --env-file .env \
  --chat-model qwen3-8b \
  --embedding-model text-embedding-3-small \
  --eval-model gpt-4o-mini \
  --eval-prompt-style memos_json \
  --eval-num-runs 3 \
  --memory-backend official \
  --memory-agent-profile openclaw_fidelity \
  --agent-mode memory_tools \
  --max-agent-steps 6 \
  --sources memory \
  --top-k 6 \
  --chunk-tokens 400 \
  --chunk-overlap 80 \
  --candidate-multiplier 4 \
  --vector-weight 0.7 \
  --text-weight 0.3 \
  --cleanup-q0-after-item \
  --continue-on-error
```

For the Qwen-235B forced-retrieval rerun, add:

```bash
--force-min-memory-searches 1
```

or use:

```bash
bash systems/openclaw/scripts/run_qwen235b_forced_openrouter.sh \
  key1 \
  openclaw/benchmarks/memory_mvp/dataset_lists/longmemeval_fixed2k_fidelity_balanced50x4x5_seed20260421.txt \
  results/openclaw_qwen235b_forced_key1 \
  2
```

## Notes

- `openclaw_fidelity` disables benchmark-only search highlights and prompt
  hints; it uses the official OpenClaw document-memory tool payload.
- The forced-retrieval 235B setting changes only the minimum number of
  `memory_search` calls before answer completion; it does not change memory
  rendering, chunking, scoring, or judge behavior.
- The runner records `trace_q1_full.csv`, per-item `results.json`, and
  `search_results.json` under the specified results root.
