# Data Dictionary

## Common Audit Table

`data/all_trajectory_outcomes.csv` contains one row per completed trajectory.

| Field | Meaning |
| --- | --- |
| `record_set` | Source collection within this artifact |
| `source_package` | Local result package identifier, with no filesystem path |
| `benchmark` | `LongMemEval` or `LoCoMo` |
| `memory_system` | Published memory implementation used for the trajectory |
| `scope` | Primary, additional-interface, or supplementary analysis status |
| `model_label` | Human-readable agent model tier |
| `model` | Model identifier recorded by the runner |
| `scale` | Number of added irrelevant sessions: 0, 100, 200, 300, or 400 |
| `scale_label` | Canonical scale label |
| `trajectory_id` | Benchmark item identifier used as the row-level audit key |
| `task_id` | Original task identifier where available |
| `question_type` | Benchmark question category where available |
| `trial` | Rollout index; the reported experiments use one rollout per item |
| `retrieval_calls` | Agent-visible memory-tool invocations, denoted by `R` |
| `correctness_source` | Provenance of the correctness value |
| `success` | Final-answer correctness used in the reported metrics |
| `judge_success` | Separately retained archived judge field, when present in the source export |
| `context_ok` | Runner integrity field where available |

## Metric Definitions

For correctness indicator `S`, retrieval-call count `R`, and call criterion `B`:

```text
Pass@B               = 1[S = 1 and R <= B]
p_wrong              = P(S = 0 and R <= B)
p_exh                = P(R > B)
P90R                 = 90th percentile of R
```

Every trajectory belongs to exactly one operational category:

```text
Pass@B + wrong-within-budget + budget-exhausted = 1
```

`p_exh` is an operational call-compliance event. It includes trajectories that
eventually answer correctly after `B` calls and trajectories that remain wrong.
It is not a component-level causal label.

## Source-Specific Tables

The three source-specific trajectory tables retain additional runtime fields
when those fields were present in the archived exports. These include token
counts, timing, retrieved-session counts, and precomputed budget indicators.
The common table deliberately keeps only fields shared across all systems.

Some copied source summaries retain legacy field names such as
`source_trace_file`, `has_trace_json`, and `trace_integrity`. In this public
artifact, those fields record source-file provenance or source-package integrity
metadata. They do not indicate that raw textual traces are included. The files
published here are trajectory-level outcome and metric tables.

For the 22,500 additional-interface rows, `success` is the archived evaluator
outcome from the joined evaluation export. That export does not retain a second,
independent judge field, so `judge_success` is blank and
`correctness_source=joined_evaluator_result`. The reproduction script does not
claim an independent success-versus-judge check for those rows.

The additional-interface inventory contains filenames rather than private
server paths. The prediction and evaluator exports were joined by their
top-level trajectory key, with duplicate keys resolved before this artifact was
constructed. The public row-level file contains the resulting 22,500 matched
outcomes.
