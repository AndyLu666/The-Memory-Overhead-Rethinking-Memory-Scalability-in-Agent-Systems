# HippoRAG Benchmark Artifact

This directory contains the HippoRAG code used in the memory scalability
experiments for LongMemEval and LoCoMo.

The artifact is split into two parts:

- `upstream/`: the HippoRAG source tree required to build and query HippoRAG
  caches.
- `benchmarks/`: the benchmark wrappers and cache-reuse utilities used in this
  study.

The wrappers are cleaned versions of the experiment scripts. They keep the
runtime contract and tracing behavior while replacing local machine paths with
repository-relative paths and environment-configured dataset roots.

## Setup

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r systems/hipporag/requirements.txt
python -m pip install -e systems/hipporag/upstream
```

Create a local environment file:

```bash
cp systems/licomemory/env.example .env
```

The HippoRAG wrappers reuse the same provider and dataset environment variables
described in [systems/licomemory/README.md](../licomemory/README.md).

## Included Wrappers

- `benchmarks/run_hipporag_longmemeval.py`: single-pass or ReAct-style q1 on
  LongMemEval.
- `benchmarks/run_hipporag_fixed2k_basefirst_full.py`: base-first fixed2k
  LongMemEval pipeline.
- `benchmarks/run_hipporag_locomo.py`: LoCoMo fixed-group pipeline.
- `benchmarks/probe_hipporag_fixed2k_base_reuse.py`: base-cache reuse probe.
- `benchmarks/prepare_hipporag_base_cache_reuse.py`: safe symlink preparation
  for fixed2k base caches.
- `benchmarks/prepare_hipporag_locomo_q0_reuse.py`: q0 reuse preparation for
  LoCoMo.

## LongMemEval Example

```bash
python systems/hipporag/benchmarks/run_hipporag_longmemeval.py \
  --data-root "$LICOMEMORY_LONGMEMEVAL_DATA_ROOT" \
  --dataset-list systems/licomemory/scripts/dataset_lists/fixed2k_sbins_fixed2k_main3m_20260224_102211/s100_all.txt \
  --results-root runs/hipporag_longmemeval_example \
  --judge-config systems/hipporag/config/hipporag_longmemeval_eval_only_gpt4omini_memos_20260324.yaml \
  --llm-model Qwen/Qwen3-8B \
  --embedding-model text-embedding-3-small
```

## LoCoMo Example

```bash
python systems/hipporag/benchmarks/run_hipporag_locomo.py \
  --data-root "$LICOMEMORY_LOCOMO_DATA_ROOT" \
  --dataset-lists systems/licomemory/scripts/dataset_lists/locomo_multihop282_fixedgroup_sbins_lmealign_20260315/s000_r01_all.txt \
  --results-root runs/hipporag_locomo_example \
  --judge-config systems/hipporag/config/hipporag_locomo_eval_only_gpt4omini_memos_20260403.yaml \
  --llm-model Qwen/Qwen3-32B \
  --embedding-model text-embedding-3-small
```

## Notes

- Benchmark datasets and generated caches are intentionally excluded.
- The wrappers depend on the shared evaluation and tracing utilities under
  `systems/licomemory/`.
- If you want the exact list files used in the experiments, use the copies under
  `systems/licomemory/scripts/dataset_lists/`.

