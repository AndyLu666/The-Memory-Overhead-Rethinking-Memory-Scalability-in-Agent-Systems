# Reviewer 3 Trajectory Audit Artifact

This directory makes the numerical evidence discussed in the author response
directly auditable. It complements the runnable system code under `systems/`.
The system directories reproduce execution; this directory reproduces the
trajectory-level metrics and checks the specific claims raised during review.

## What This Artifact Resolves

The artifact exposes every trajectory-level outcome used by the added analyses:

- the archived evaluator outcome for the final answer;
- the number of agent-visible memory calls;
- the benchmark, system, model, and memory scale;
- the fields needed to recompute `Pass@B`, `p_wrong`, `p_exh`, and P90R;
- the 22,500 MemOS-text, Mem0, and MemOS-Tree records discussed in the rebuttal;
- the LongMemEval and LoCoMo records used for the decision reversal and LoCoMo diagnosis;
- the OpenClaw records used in the six-system LongMemEval coverage analysis.

The public tables omit duplicated benchmark conversation text, retrieved
passages, provider request payloads, and private filesystem paths. Those fields
are not used by any reported trajectory metric. The task identifiers,
correctness outcomes, call counts, and operational categories needed for
numerical audit are preserved row by row. These are trace-derived outcome
tables, not raw textual trace replay files.

## Exact Coverage

Coverage is intentionally stated per benchmark rather than implying a full
six-system by two-benchmark factorial.

| Benchmark | Memory system | Agents | Scales | Rows | Scope |
| --- | --- | --- | ---: | ---: | --- |
| LongMemEval | HippoRAG | Qwen 8B, 32B, 235B | 5 | 30,000 | 2,000 tasks per cell |
| LongMemEval | LiCoMemory | Qwen 8B, 32B, 235B | 5 | 30,000 | 2,000 tasks per cell |
| LoCoMo | HippoRAG | Qwen 8B, 32B, 235B | 5 | 4,230 | 282 tasks per cell |
| LoCoMo | LiCoMemory | Qwen 8B, 32B, 235B | 5 | 4,230 | 282 tasks per cell |
| LongMemEval | MemOS-text | Qwen 8B, 32B, 235B | 5 | 7,500 | additional interface analysis |
| LongMemEval | Mem0 | Qwen 8B, 32B, 235B | 5 | 7,500 | additional interface analysis |
| LongMemEval | MemOS-Tree | Qwen 8B, 32B, 235B | 5 | 7,500 | additional interface analysis |
| LongMemEval | OpenClaw | Qwen 8B, 32B, 235B | 5 | 3,000 | additional interface analysis |
| LongMemEval | LiCoMemory | Llama 8B, 70B, 405B | 5 | 21,097 | supplementary; 70B/405B are partial |
| LoCoMo | LiCoMemory | GPT-5-mini | 5 | 1,410 | supplementary model check |

The common audit table contains **116,467 trajectory rows**. LongMemEval covers
all six published memory implementations. LoCoMo provides the cross-benchmark
check for HippoRAG and LiCoMemory.

The six-system LongMemEval coverage contains two implementations in each
operational category used by the paper:

| Operational category | Evaluated implementations |
| --- | --- |
| Flat | OpenClaw, MemOS-text |
| Planar | Mem0, MemOS-Tree |
| Hierarchical | HippoRAG, LiCoMemory |

These labels organize coverage. Every published row retains its concrete memory
system and agent model, so the audit can be performed at the
agent-implementation level rather than by assigning one failure regime to an
entire category.

## Directory Layout

- `data/all_trajectory_outcomes.csv`: normalized audit table across every
  included result set.
- `data/hierarchical/trajectory_metrics.csv`: the complete derived Qwen matrix
  plus the explicitly marked supplementary Llama runs.
- `data/hierarchical/gpt5mini_locomo_trajectory_metrics.csv`: supplementary
  GPT-5-mini LoCoMo records.
- `data/additional_interfaces/trajectory_metrics.csv`: all 22,500 joined
  MemOS-text, Mem0, and MemOS-Tree trajectory outcomes.
- `data/openclaw/trajectory_metrics.csv`: all 3,000 OpenClaw trajectory outcomes.
- `aggregates/`: the corresponding source summaries, budget curves, integrity
  checks, and figures.
- `reproduced/`: deterministic outputs generated from the common audit table.
- `scripts/reproduce_metrics.py`: standard-library metric reproduction.
- `scripts/verify_checksums.py`: byte-level artifact verification.
- `scripts/prepare_public_artifact.py`: records how the public, text-free tables
  were produced from the local result exports.
- `DATA_DICTIONARY.md`: field definitions and failure-partition identities.
- `MANIFEST.sha256`: SHA-256 digest for every published artifact file.

## Reproduce the Metrics

From the repository root:

```bash
python3 artifacts/reviewer3_audit/scripts/verify_checksums.py
python3 artifacts/reviewer3_audit/scripts/reproduce_metrics.py
```

The second command reconstructs:

- `reproduced/coverage.csv`;
- `reproduced/metrics_by_cell_and_budget.csv`;
- `reproduced/claim_checks.json`.

It also verifies unique trajectory keys, agreement between `success` and the
separately retained judge field where that field exists, the three-way failure
partition for every budget from 1 through 6, exact benchmark-system coverage,
and the headline decision-reversal, LoCoMo, and Mem0 numbers.

## Headline Checks

The generated `claim_checks.json` directly audits the rebuttal's main examples:

1. On LongMemEval with Qwen 8B at `s400`, unbudgeted success selects
   LiCoMemory over HippoRAG, while `Pass@2` selects HippoRAG. At `B=5`, the
   selection returns to LiCoMemory.
2. On LoCoMo, LiCoMemory with Qwen 8B already has `Pass@2=7.09%`,
   `p_exh=91.84%`, and `P90R=5` at `s0`. This locates the near-floor `Pass@2`
   result as a baseline over-budget interaction pattern rather than a collapse
   caused by adding 400 sessions.
3. On the added Mem0 records, the same low `Pass@2` can correspond to different
   operational regimes across agents. This is computed from the published
   row-level outcomes rather than from a family-level label.
4. Across MemOS-text, Mem0, and MemOS-Tree, the generated checks report how much
   each agent recovers when the call criterion changes from `B=2` to `B=6`.

## Relationship to the Runnable Code

No execution code is duplicated here. The execution programs and configurations
already published in this repository remain in their existing locations. This
directory adds the result-layer counterpart: it lets a reviewer inspect every
trajectory outcome used by the added analyses and recompute the evidence
without rerunning the models.
