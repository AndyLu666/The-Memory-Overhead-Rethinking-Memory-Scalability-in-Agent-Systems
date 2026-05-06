# The Memory Overhead: OpenClaw Artifact Code

This repository contains reproducibility code for the OpenClaw document-memory
experiments used in the memory scalability study.

The OpenClaw code is under `systems/openclaw/`. It includes:

- the Python benchmark harness used to run LongMemEval items,
- the Node bridge that calls OpenClaw's official `memory-core` tools,
- the Qwen-235B forced-retrieval rerun script,
- the final LongMemEval fixed2k dataset list,
- an OpenClaw core patch for configurable embedding batch limits/timeouts.

No API keys, local caches, SQLite indexes, or generated result folders are
committed here. Configure credentials through a local `.env` file based on
`systems/openclaw/env.example`.
