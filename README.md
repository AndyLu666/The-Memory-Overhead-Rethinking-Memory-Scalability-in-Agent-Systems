# The Memory Overhead: Reproducibility Code

This repository contains the runnable artifact code used in the memory
scalability study. The repository is organized by memory system and benchmark
pipeline rather than by paper section, so each system directory can be used
independently.

## Repository Layout

- `systems/licomemory/`: LiCoMemory source, benchmark configs, dataset builders,
  and runners for LongMemEval and LoCoMo.
- `systems/hipporag/`: HippoRAG upstream source plus the benchmark wrappers
  used to run LongMemEval and LoCoMo under the shared evaluation contract.
- `systems/openclaw/`: OpenClaw document-memory benchmark artifact code.

No API keys, local caches, generated graphs, SQLite indexes, or experiment
result folders are committed. Configure credentials and dataset paths locally
through system-specific `.env` files or shell environment variables.

## Systems

### LiCoMemory

LiCoMemory is included as a self-contained runnable copy under
`systems/licomemory/`. The directory contains:

- the core LiCoMemory runtime (`main.py` and the supporting Python packages),
- cleaned benchmark configs for LoCoMo and LongMemEval,
- dataset builders for the fixed2k LongMemEval and fixed-group LoCoMo variants,
- the batch runner used for q0/q1 execution and tracing,
- a small contract test suite for the active benchmark settings.

Start with:

- [LiCoMemory README](systems/licomemory/README.md)

### HippoRAG

HippoRAG is organized as:

- `systems/hipporag/upstream/`: the upstream HippoRAG source required to build
  and query caches,
- `systems/hipporag/benchmarks/`: the benchmark wrappers and cache-reuse
  utilities used in this study,
- `systems/hipporag/config/`: judge-contract configs shared with LiCoMemory.

Start with:

- [HippoRAG README](systems/hipporag/README.md)

### OpenClaw

The OpenClaw artifact is already included and left unchanged apart from the
top-level repository integration.

Start with:

- [OpenClaw README](systems/openclaw/README.md)

## What Is Included

The repository includes the code needed to:

- build the benchmark-specific dataset layouts used by the experiments,
- run q0 preprocessing and q1 query/evaluation pipelines,
- reproduce trace files and per-item outputs,
- inspect and validate benchmark contracts.

The repository does not include benchmark data dumps or final result packages.
Reviewers should place benchmark datasets locally and point the configs or CLI
flags at those dataset roots.

## Baseline Setup

Each system directory contains its own setup instructions. In general:

1. create a Python environment,
2. install the requirements for the system you want to run,
3. prepare a local `.env` from that system's example file,
4. place LoCoMo and/or LongMemEval data on disk,
5. run the dataset builder or benchmark runner described in the system README.
