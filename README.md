# QwenGeometry

Qwen + AlphaGeometry experiment scripts for auxiliary-construction search,
candidate reranking, fact-context SFT, and hard-negative training.

This repository intentionally tracks source code and orchestration scripts only.
Large generated artifacts such as model checkpoints, `outputs/`, JSONL datasets,
logs, and remote AlphaGeometry/Qwen checkouts are ignored.

## Layout

- `scripts/`
  - `qwen_ag_search.py`: Qwen generation, candidate filtering, grammar mask,
    fact-context prompts, value-model reranking, and DDAR helpers.
  - `run_qwen_ag_benchmark.py`: benchmark runner for DDAR/Qwen assisted search.
  - `train_qwen_aux_lora.py`: LoRA SFT trainer with optional target
    unlikelihood loss for generator-side hard negatives.
  - `build_candidate_value_data.py` and `train_candidate_value_model.py`:
    candidate value/reranker data and model training.
  - `build_fact_context_from_aux_sft.py`: reconstruct Stage2 aux prompts,
    run DDAR, and inject selected facts into `{D}` prompt context.
  - `analyze_qwen_ag_events.py` and `report_qwen_ag_analysis.py`: event log
    analysis and compact Markdown reports.
- `data/synth_cpt_1m_pruned_v2/`
  - Shell orchestration scripts for staged training, high-budget reruns,
    fact-context ablations, value-model queues, and candidate-signal SFT.
- `EXPERIMENTS.md`
  - Experiment tag registry and versioning notes.

## Runtime Assumptions

The scripts are designed for the remote workspace:

```bash
cd /root/rivermind-data/qwen_ag_lm
. venv/bin/activate
```

Expected external directories on that host:

- `repos/alphageometry`
- `models/Qwen2.5-7B`
- `outputs/`
- `data/staged_1m_pruned_v2/`

Headless AlphaGeometry runs should use `xvfb-run` because the AG1 code path can
load TkAgg.

## Current Main Tags

- Current GitHub source head:
  `new_solved_baseline_highlight_v1`
- Current running 16-problem bench:
  `unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1`
- Last completed high-budget bench:
  `unsolved_high_budget_value_v3_cost_template_depth16_v1`
- Current fact-context adapter:
  `factctx_promptaug_top8_stage2max2000_v1`
- Current timeout/hard-negative value model:
  `v5_timeout_hardneg_features_v1_plus_v3`
- Next clean candidate-quality code baseline after the running process:
  `semantic_point_mask_v4`, `value_rerank_event_scores_v1`, and
  `template_backfill_seen_canonical_v1`, `generation_canonical_dedup_v1`, and
  `template_backfill_generation_dedup_v1`, `template_initial_backfill_source_v1`,
  and `candidate_value_data_type_inference_v1`
- Latest partial candidate-signal snapshot from the running bench:
  `candidate_signals_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_4events_typed_v2`
- Latest partial value-data snapshot from the running bench:
  `value_data_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_4events_typed_v2`
- Reports can highlight newly solved problems by running
  `scripts/analyze_qwen_ag_events.py --baseline_summary_jsonl <previous-summary.jsonl>`
  before rendering with `scripts/report_qwen_ag_analysis.py`.

See `EXPERIMENTS.md` for detailed tag-to-output mappings.

## Validation

Basic syntax checks:

```bash
python -m py_compile scripts/*.py
bash -n data/synth_cpt_1m_pruned_v2/*.sh
```

Useful smoke checks:

```bash
python scripts/build_aux_hard_negative_from_candidate_signals.py --help
python scripts/build_fact_context_from_aux_sft.py --help
python scripts/analyze_qwen_ag_events.py --help
```

## Versioning Rules

- Git commits and git tags track source-code behavior only.
- Bench tags track generated output directories and runtime configurations.
- A running benchmark keeps the Python code it loaded at process start; later git
  commits apply only to future runs unless the benchmark is restarted.
- Do not overwrite `scripts/` in the remote running workspace while a
  spawn-based benchmark is active; sync new source after that run completes or
  before launching a new tag.
- Use experiment tags as version identifiers for model/data/search changes.
- Do not overwrite completed `outputs/final_eval_*` directories.
- Create a new tag when changing training data, prompt schema, candidate
  filtering, reranker features, or search budget.
- Refresh analysis JSON/Markdown after each completed benchmark.
- Keep code commits separate from generated outputs and model checkpoints.
