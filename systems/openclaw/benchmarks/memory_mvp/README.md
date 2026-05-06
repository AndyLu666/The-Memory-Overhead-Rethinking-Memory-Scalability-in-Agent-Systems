# OpenClaw Memory MVP Benchmark Harness

This benchmark harness evaluates OpenClaw's document-memory behavior on
LongMemEval-style `Corpus.json` / `Question.json` items.

The default backend is `official`, which invokes OpenClaw `memory-core` through
`official_memory_bridge.mjs`. The standalone `OpenClawMemoryMVP` backend remains
available only as a fallback/debug implementation.

Important default settings used in the reported OpenClaw LongMemEval runs:

- `--memory-backend official`
- `--memory-agent-profile openclaw_fidelity`
- `--agent-mode memory_tools`
- `--sources memory`
- `--top-k 6`
- `--max-agent-steps 6`
- `--chunk-tokens 400`
- `--chunk-overlap 80`
- `--candidate-multiplier 4`
- `--vector-weight 0.7`
- `--text-weight 0.3`
- `--eval-model gpt-4o-mini`
- `--eval-prompt-style memos_json`
- `--eval-num-runs 3`

For the Qwen-235B forced-retrieval rerun, use
`--force-min-memory-searches 1`.
