# LiCoMemory Benchmark Artifact

This directory contains the LiCoMemory code used to run the LoCoMo and
LongMemEval experiments in the memory scalability study.

The artifact keeps the original runtime layout so the main entrypoint and the
supporting packages behave the same way as they did in the experiment codebase,
but local machine paths, caches, and generated outputs have been removed.

## Directory Layout

- `main.py`: LiCoMemory entrypoint.
- `base/`, `chunking/`, `coregraph/`, `dataset/`, `evaluation/`, `init/`,
  `prompt/`, `query/`, `utils/`: runtime packages used by the LiCoMemory
  graph-building, retrieval, and evaluation pipeline.
- `config/`: cleaned benchmark configs derived from the experiment configs.
- `scripts/`: dataset builders, the batch runner, and shell helpers for the
  LoCoMo and LongMemEval experiment pipelines.
- `tests/`: lightweight contract checks for the active benchmark settings.
- `env.example`: local environment template.

## Setup

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r systems/licomemory/requirements.txt
```

Create a local environment file:

```bash
cp systems/licomemory/env.example .env
```

Fill in the provider credentials and dataset paths you plan to use. Do not
commit `.env`.

## Environment Variables

The cleaned configs expect these environment variables:

- `LICOMEMORY_LONGMEMEVAL_DATA_ROOT`
- `LICOMEMORY_LOCOMO_DATA_ROOT`
- `LICOMEMORY_LONGMEMEVAL_SOURCE_ROOT`
- `LOCOMO_SOURCE_ROOT` (needed only when rebuilding the LoCoMo aligned dataset)
- `LOCOMO_SOURCE_MANIFEST` (needed only when rebuilding the LoCoMo aligned dataset)
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `QUERY_API_KEY`
- `QUERY_BASE_URL`
- `JUDGE_API_KEY`
- `JUDGE_BASE_URL`

The runtime also accepts the legacy aliases already handled in the source
(`GPT_API_KEY`, `GPT_BASE_URL`, `QWEN_API`, `OTHER_API_KEY`, and similar), but
the variables above are the recommended artifact defaults.

## Benchmark Configs

The artifact includes four cleaned configs:

- `config/licomemory_longmemeval_fixed2k_main3m_gpt5mini_openai_build_q0_20260410.yaml`
- `config/licomemory_longmemeval_fixed2k_main3m_qwen3_8b_ifopen_query_q1_llmeval_gpt4omini_memos_steps6_tmp_20260412.yaml`
- `config/licomemory_locomo_multihop282_fixedgroup_lmealign_gpt5mini_openai_build_q0_20260312.yaml`
- `config/licomemory_locomo_multihop282_fixedgroup_lmealign_qwen3_8b_ifopen_query_q1_llmeval_gpt4omini_memos_20260317.yaml`

These are artifactized versions of the experiment configs. They keep the
runtime knobs intact but replace local absolute paths with environment-driven
dataset roots.

## LongMemEval

### Build the fixed2k dataset

```bash
python systems/licomemory/scripts/build_fixed_anchor_sbins_dataset.py \
  --src-root "$LICOMEMORY_LONGMEMEVAL_SOURCE_ROOT" \
  --out-data-root "$LICOMEMORY_LONGMEMEVAL_DATA_ROOT" \
  --out-list-root systems/licomemory/scripts/dataset_lists/fixed2k_sbins_fixed2k_main3m_20260224_102211
```

### Run q1 on a list of LongMemEval items

```bash
python systems/licomemory/scripts/run_longmemeval_runner.py \
  --repo-root systems/licomemory \
  --python-bin .venv/bin/python \
  --config systems/licomemory/config/licomemory_longmemeval_fixed2k_main3m_qwen3_8b_ifopen_query_q1_llmeval_gpt4omini_memos_steps6_tmp_20260412.yaml \
  --data-root "$LICOMEMORY_LONGMEMEVAL_DATA_ROOT" \
  --dataset-list systems/licomemory/scripts/dataset_lists/fixed2k_sbins_fixed2k_main3m_20260224_102211/s100_all.txt \
  --root-prefix runs/licomemory_longmemeval_example \
  --log-dir runs/licomemory_longmemeval_example_logs \
  --csv-out runs/licomemory_longmemeval_example/trace_q1_full.csv \
  --checkpoint runs/licomemory_longmemeval_example/checkpoint_q1_full.json \
  --workers 1 \
  --query 1
```

## LoCoMo

### Build the fixed-group aligned dataset

```bash
python systems/licomemory/scripts/build_locomo_fixed_group_lmealign_dataset.py \
  --src-root "$LOCOMO_SOURCE_ROOT" \
  --manifest "$LOCOMO_SOURCE_MANIFEST" \
  --external-filler-root "$LICOMEMORY_LONGMEMEVAL_SOURCE_ROOT" \
  --out-data-root "$LICOMEMORY_LOCOMO_DATA_ROOT" \
  --out-list-root systems/licomemory/scripts/dataset_lists/locomo_multihop282_fixedgroup_sbins_lmealign_20260315
```

### Run the LoCoMo pipeline

```bash
bash systems/licomemory/scripts/run_locomo_fixedgroup_lmealign_pipeline_20260312.sh my_locomo_run_id
```

The shell helper expects `.env` to be present or `ENV_FILE` to be set.

## Tests

Run the artifact contract checks:

```bash
python -m unittest systems/licomemory/tests/test_active_runtime_contracts.py
```

## Notes

- The artifact intentionally excludes benchmark datasets, graph caches, and
  result folders.
- The LongMemEval and LoCoMo dataset lists used during the experiments are
  included under `scripts/dataset_lists/` for reproducibility.
- The batch runner records `trace_q1_full.csv`, checkpoints, and per-item
  `results.json` under the output root you choose.
