# AG1 Reproduction Analysis

This note records the current AG1 reproduction evidence and the immediate
QwenGeometry optimization direction derived from it.

## Reference

- AG1 paper target: AlphaGeometry solves 25/30 IMO-AG-30 problems.
- Official README baseline: DDAR alone solves 14/30 IMO-AG-30 problems.
- Local snapshot analyzed: `alphageometry_repro_results/ag_results_snapshot_20260602_231247`
  under the shared workspace.

## Snapshot Evidence

The snapshot contains a partial GPU AG1 reproduction with
`batch_size=32`, `beam_size=512`, `search_depth=16`, and `lm_batch_size=32`.

Root DDAR solved 14 problems:

```text
translated_imo_2000_p1
translated_imo_2002_p2a
translated_imo_2002_p2b
translated_imo_2003_p4
translated_imo_2004_p5
translated_imo_2005_p5
translated_imo_2007_p4
translated_imo_2010_p4
translated_imo_2012_p1
translated_imo_2013_p4
translated_imo_2015_p4
translated_imo_2016_p1
translated_imo_2017_p4
translated_imo_2022_p4
```

The partial AG1 continuation solved two DDAR-unsolved problems:

| Problem | Depth | Winning Rank | Candidates Checked | Translation |
|---|---:|---:|---:|---|
| `translated_imo_2000_p6` | 0 | 31 | 32 | `q = on_line q a h, on_tline q b a h` |
| `translated_imo_2004_p1` | 0 | 18 | 31 | `k = on_line k e f, on_bline k f e` |

The important detail is that the winning AG1 candidate for
`translated_imo_2000_p6` was rank 31 in the first LM batch. A system that only
DDAR-checks the top 16 or top 24 candidates can miss this even if its LM has
already generated the right auxiliary construction.

## Why AG1 Scores Higher

AG1's advantage is not just a stronger LM. The released search keeps a very
wide candidate surface alive:

- It uses a large LM beam and batch (`batch_size=32`, paper-scale
  `beam_size=512`, `depth=16`).
- It verifies candidates with DDAR instead of trusting LM score alone.
- It can win with low-ranked candidates, as seen in the rank-31
  `translated_imo_2000_p6` solve.
- Root DDAR is allowed to solve the easy 14-problem subset first, so LM search
  is focused on the remaining hard cases.

## Direction For QwenGeometry

The next bench-score improvements should bias toward AG1-like candidate
coverage:

- Preserve more low-rank valid candidates for DDAR, especially at depth 0 and
  depth 1, before aggressive depth pruning.
- Keep type/progress diversity, but add coverage policies that do not collapse
  the eval pool to only the highest value-model scores.
- Log candidate rank, source, rerank phase, and whether a candidate was pruned
  before DDAR, so future analysis can distinguish LM miss from verifier-budget
  miss.
- Treat DDAR-progress positives as a way to keep branches alive, but do not
  overfit the reranker to high progress alone; AG1's winning signal is final
  DDAR proof success.
