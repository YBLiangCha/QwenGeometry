# Qwen + AlphaGeometry Experiment Registry

This file tracks experiment tags, output directories, and the main behavioral
changes. Tags are the practical version identifiers for this workspace.

## Completed Bench

### `unsolved_high_budget_value_v3_cost_template_depth16_v1`

- Output: `outputs/final_eval_imo_ag30_qwen_unsolved_high_budget_value_v3_cost_template_depth16_v1`
- Adapter: `outputs/stage2_aux_after_cpt1m_lora_qwen2_5_7b_v1`
- Value model: `outputs/candidate_value_model_v3_cost_aware_from_qwen_v1_logs/candidate_value_model.json`
- Scope: 16 previously unsolved IMO AG problems.
- Search: `beam_size=64`, `search_depth=4`, `num_return_sequences=32`, `candidate_depth_eval_limit=16`.
- Candidate quality controls: point mask/repair, canonical dedup, DSL filter/token mask, mixed constructive prompt sampling, template backfill, value-model diverse rerank.
- Result: 1/16 solved, `translated_imo_2018_p1`.
- Main observed bottleneck: canonical duplicate collapse remained very high; several problems were DDAR-timeout blocked or made symbolic progress in the wrong direction.

## Value Models

### `v5_timeout_hardneg_features_v1_plus_v3`

- Output: `outputs/candidate_value_model_v5_timeout_hardneg_features_v1_plus_v3`
- Data sources: baseline v1 logs plus completed v3 logs.
- Added signals: timeout negatives, PointTooClose/PointTooFar/point-already-exists hard negatives, candidate source features.
- Purpose: rerank valid candidates before DDAR and reduce wasted DDAR budget.

## Fact-Context Adapters

### `factctx_top8_after_v3_v1`

- Output: `outputs/stage2_fact_context_after_stage2_lora_qwen2_5_7b_factctx_top8_after_v3_v1`
- Data: JGEX/AG mining with DDAR facts in `{D}` prompt context.
- Limitation: only 31 raw fact-context rows were mined, so this is a weak format-adaptation baseline.

### `factctx_promptaug_top8_stage2max2000_v1`

- Output: `outputs/stage2_fact_context_after_stage2_lora_qwen2_5_7b_factctx_promptaug_top8_stage2max2000_v1`
- Data: existing Stage2 aux SFT prompts are reconstructed into AG problems, DDAR is run on each prefix, and selected facts are inserted into `{D}`.
- Build script: `scripts/build_fact_context_from_aux_sft.py`
- Queue script: `data/synth_cpt_1m_pruned_v2/run_fact_context_prompt_aug_sft.sh`
- Actual prompt-aug build from the first 2000 Stage2 rows:
  - prompt-aug rows: 1385
  - train/eval: 1236 / 149
  - mixed train/eval: 4262 / 332
  - skipped no-fact rows: 115
  - reconstruction/value errors: 494
  - row wall-time timeouts: 6
- Training completed: 2026-06-07 20:48 +0800.
- Final adapter size: 155 MB `adapter_model.safetensors`.
- Final eval loss: 0.1574.
- Train runtime: 1521 sec, 267 optimizer steps, 1 epoch.
- Improvements over `factctx_top8_after_v3_v1`:
  - Uses Stage2 aux SFT prompts instead of relying mostly on JGEX clause mining.
  - Carries the original goal back into the reconstructed problem when possible.
  - Adds progress logging during fact-context row construction.
  - Adds per-row wall-time timeout so one pathological prompt cannot block the whole build.
  - Produces O(1000) fact-context rows from the first 2000 Stage2 rows.

## Current Main Ablation Chain

### `unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_v1`

- Adapter: `outputs/stage2_fact_context_after_stage2_lora_qwen2_5_7b_factctx_promptaug_top8_stage2max2000_v1`
- Value model: prefer `outputs/candidate_value_model_v5_timeout_hardneg_features_v1_plus_v3/candidate_value_model.json`
- Output: `outputs/final_eval_imo_ag30_qwen_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_v1`
- Purpose: rerun the same 16 unsolved problems with prompt-augmented fact-context SFT, v5 value model, grammar-prefix token mask, true overgeneration, post-canonical backfill, and hard-negative signal logging.
- Started: 2026-06-07 20:52 +0800.
- Confirmed runtime flags: `beam_size=64`, `search_depth=4`, `num_return_sequences=32`, `candidate_depth_eval_limit=16`, `candidate_ddar_workers=8`, `lm_fact_context_top_k=8`, `candidate_dsl_token_mask`, `candidate_point_mask`, `candidate_point_repair`, `candidate_canonical_dedup`, `candidate_template_backfill`, `candidate_rerank=value_model_diverse`.
- Stopped early at 2026-06-07 21:20 +0800 after partial analysis showed this run started before the `eqangle3` arity and semantic point-mask fixes.
- Partial status before stop: 2 event files, 1 completed problem, 0 solved, 2958 candidates, 2130 valid, 828 invalid, 1534 duplicate-canonical filters, 27 candidate SFT signals, 88 hard-negative signals.

### `unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v1`

- Adapter: `outputs/stage2_fact_context_after_stage2_lora_qwen2_5_7b_factctx_promptaug_top8_stage2max2000_v1`
- Value model: `outputs/candidate_value_model_v5_timeout_hardneg_features_v1_plus_v3/candidate_value_model.json`
- Output: `outputs/final_eval_imo_ag30_qwen_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v1`
- Started: 2026-06-07 21:20 +0800.
- Purpose: rerun the 16 unsolved problems with the same prompt-aug fact-context adapter and search budget, plus `eqangle3_arity_fix_v1`, `unknown_point_hardneg_v1`, and `semantic_point_mask_v1`.
- Queue PIDs at launch: ablation `754460`, candidate signal data `754461`, candidate signal SFT `754462`.
- Partial check at 2026-06-07 21:31 +0800: 2 event files, 1 completed problem, 0 solved, 270 candidates, 203 valid, 67 invalid, 29 duplicate-canonical filters, 16 candidate DDAR done, 16 candidate DDAR errors, 10 candidate SFT signals, 18 hard-negative signals.
- Early readout: compared with the stopped pre-semantic run at a similar early stage, semantic masking sharply reduced unknown-point/duplicate noise; `translated_imo_2000_p6` remains DDAR-timeout blocked.
- Stopped early at 2026-06-07 21:38 +0800 after `semantic_point_mask_v2` and `semantic_point_mask_v3` were added while the run was still on the second problem.
- Later partial status before stop: 2 event files, 1 completed problem, 0 solved, 836 candidates, 640 valid, 196 invalid, 214 duplicate-canonical filters, 10 candidate SFT signals, 50 hard-negative signals.

### `unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1`

- Adapter: `outputs/stage2_fact_context_after_stage2_lora_qwen2_5_7b_factctx_promptaug_top8_stage2max2000_v1`
- Value model: `outputs/candidate_value_model_v5_timeout_hardneg_features_v1_plus_v3/candidate_value_model.json`
- Output: `outputs/final_eval_imo_ag30_qwen_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1`
- Started: 2026-06-07 21:38 +0800.
- Purpose: rerun the 16 unsolved problems with all semantic point-mask fixes through `semantic_point_mask_v3`.
- Queue PIDs at launch: ablation `762611`, candidate signal data `762612`, candidate signal SFT `762613`.
- First-problem check at 2026-06-07 21:49 +0800: `translated_imo_2000_p6` completed unsolved, 49 candidates, 39 valid, 10 invalid, 3 duplicate-canonical filters, 16 candidate DDAR errors/timeouts. The run then advanced to `translated_imo_2004_p1`.
- Early readout: v3 removed most invalid-predicate/unknown-point noise from the first problem, but `translated_imo_2000_p6` is still candidate-DDAR-timeout blocked.

## Next Candidate-Quality Fixes

### `eqangle3_arity_fix_v1`

- Code fix: `eqangle3` constructive RHS uses 5 arguments, matching AG `defs.txt` and problem text, instead of the previous 6-argument form with the new point repeated on the RHS.
- Affected paths: grammar shape filter, mixed constructive prompt prefix, and constrained-predicate-to-construction translation.
- Motivation: the prompt-aug bench surfaced invalid candidates such as `q = eqangle3 q c b i a j`, which AG rejects or treats as referencing the not-yet-created point.
- Hard-negative logging now also includes `unknown_point`, because the same partial run showed many invalid candidates referencing points outside the current graph.
- Semantic point masking now threads current graph point names into DSL prefix validation: candidate arguments must be existing points or the current new point, existing points cannot be reused as the constructed point, and non-`eqangle3` constructions must put the new point only in the first RHS argument.
- Predicate shape masking also rejects constrained predicates with impossible repeated arguments before DDAR validation, e.g. duplicate-point `O/P/C` clauses and `^` clauses that would translate to `on_aline` with the new point twice.
- Construction shape masking rejects obvious degenerate RHS arguments before DDAR validation, e.g. `on_pline q h c c`, `on_tline q h c c`, repeated circumcircle points, and `eqangle3` rows with a degenerate angle triple.
- Status: implemented after `unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_v1` had already started, so it applies to the next rerun/version rather than the currently running process.

### `candidate_signals_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1`

- Output: `data/staged_1m_pruned_v2/candidate_signals_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1`
- Purpose: extract verifier-backed positive candidate SFT rows plus generator-side hard-negative rows after the prompt-aug ablation finishes.

### `candidate_signal_sft_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1`

- Output: `outputs/stage3_candidate_signal_after_factctx_lora_qwen2_5_7b_candidate_signal_sft_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1`
- Purpose: continue training from the prompt-aug fact-context adapter with positive candidate signals and optional hard-negative unlikelihood loss.

## Versioning Rule

- Do not overwrite completed output directories.
- Use a new tag when changing training data, prompt schema, reranker features, search budget, or candidate filtering behavior.
- Keep queue scripts parameterized by tag-specific environment variables.
- Refresh analysis JSON/Markdown after every completed benchmark run.
