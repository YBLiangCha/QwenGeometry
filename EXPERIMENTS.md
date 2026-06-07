# Qwen + AlphaGeometry Experiment Registry

This file tracks experiment tags, output directories, and the main behavioral
changes. Tags are the practical version identifiers for this workspace.

## Source Version State

- Git remote: `git@github.com:YBLiangCha/QwenGeometry.git`
- Current GitHub source head: `duplicate_canonical_negative_signal_v1`
- Current running bench tag:
  `unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1`
- Running bench code behavior: includes semantic point/predicate fixes through
  `semantic_point_mask_v3`; it does not include later `semantic_point_mask_v4`
  degenerate-construction filtering or candidate rerank-score event logging,
  `template_backfill_seen_canonical_v1`, or
  `duplicate_canonical_negative_signal_v1`, because the process was already
  running when those commits were made.
- Next clean code baseline for a rerun: source head
  `duplicate_canonical_negative_signal_v1`, optionally with a new bench tag such
  as
  `unsolved_factctx_promptaug_top8_adapter_value_v7_grammar_semantic_v4_scores_dedup_dupneg_v1`.
- Remote running-workspace scripts are intentionally not overwritten while
  `unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1`
  is active. The benchmark uses spawn-based candidate workers, so overwriting
  `scripts/` mid-run could mix code versions in future workers. Sync the new
  source after this run completes or immediately before launching the next tag.

## Candidate Quality And Training Signals

### `duplicate_canonical_negative_signal_v1`

- Duplicate-canonical filtered candidates now carry `prompt` and `target` in
  both `scripts/run_qwen_ag_benchmark.py` and standalone
  `scripts/qwen_ag_search.py`.
- `scripts/build_aux_hard_negative_from_candidate_signals.py` now reads
  `candidate_filtered` events as well as explicit hard-negative events, and
  includes `duplicate_canonical` in the default hard-negative reasons.
- `scripts/build_candidate_value_data.py` now preserves duplicate-filtered
  candidates in value/reranker rows with `reason=duplicate_canonical`,
  `filtered_reason`, and `canonical_key`.
- This is a future-run improvement: the currently running
  `unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1`
  process was launched before this logging change, so its duplicate-filter
  events lack `prompt/target` and cannot yet generate duplicate hard-negative
  SFT rows. They can still be counted as duplicate value negatives by matching
  candidate/filter events.
- Motivation: the live partial run shows very high duplicate-canonical collapse
  after generation/backfill. Treating those duplicates as learnable negatives
  should help the generator and reranker spend DDAR budget on more diverse,
  directionally useful auxiliary constructions.

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

### `v8_pairwise_preddar_v5_plus_semantic_v3_partial5_typed_v1`

- Output: `outputs/candidate_value_model_v8_pairwise_preddar_v5_plus_semantic_v3_partial5_typed_v1`
- Model file: `outputs/candidate_value_model_v8_pairwise_preddar_v5_plus_semantic_v3_partial5_typed_v1/candidate_value_model.json`
- Objective: `pairwise`; feature policy: `pre_ddar_features`; `train_valid_only=true`.
- Data: v5 base value data plus the current semantic-v3 partial value rows,
  rebuilt at 2026-06-07 23:31 +0800 after 5 completed problems.
- Rows after merge/dedup: 8176 total, 753 positive, 7423 negative.
- Valid-only training rows: 5738 train rows, 71 eval rows.
- Pairwise training details: 37 positive/negative groups by `problem,depth`,
  121420 full pairs per epoch, 136740 sampled pairs total.
- Offline top-k diagnostics on all valid rows from the v8 merged data:
  - v7: AUC 0.6069; first-positive mean rank 1.08; top-4 recall 0.2138; top-8 recall 0.3785; top-16 recall 0.6587.
  - v8: AUC 0.6333; first-positive mean rank 1.12; top-4 recall 0.2231; top-8 recall 0.3891; top-16 recall 0.6507.
- Held-out `split=eval` valid rows remain small (71 rows, 20 positives,
  4 positive groups):
  - v7: AUC 0.7000; first-positive mean rank 1.50; top-4 recall 0.4500; top-8 recall 0.7500; top-16 recall 1.0000.
  - v8: AUC 0.7000; first-positive mean rank 1.00; top-4 recall 0.4000; top-8 recall 0.7500; top-16 recall 1.0000.
- Readout: v8 is a useful evaluated snapshot and slightly improves all-row
  top-8 recall, but it does not beat v7 on the current top-16 budget. Keep v7
  as the next-rerun default; revisit after the full 16-problem benchmark.

### `v7_pairwise_preddar_v5_plus_semantic_v3_partial4_typed_v1`

- Output: `outputs/candidate_value_model_v7_pairwise_preddar_v5_plus_semantic_v3_partial4_typed_v1`
- Model file: `outputs/candidate_value_model_v7_pairwise_preddar_v5_plus_semantic_v3_partial4_typed_v1/candidate_value_model.json`
- Objective: `pairwise`; feature policy: `pre_ddar_features`; `train_valid_only=true`.
- Training command uses `run_value_model_append_partial.sh` with
  `VALUE_TRAIN_EXTRA_ARGS='--objective pairwise --train_valid_only --epochs 20 --lr 0.01 --pairwise_negatives_per_positive 16'`.
- Data: v5 base value data plus the current semantic-v3 partial value rows,
  rebuilt at 2026-06-07 23:24 +0800 while `translated_imo_2008_p1b` was still running.
- Rows after merge/dedup: 8085 total, 742 positive, 7343 negative.
- Valid-only training rows: 5661 train rows, 71 eval rows.
- Pairwise training details: 37 positive/negative groups by `problem,depth`,
  120652 full pairs per epoch, 130020 sampled pairs total.
- Offline top-k diagnostics on all valid rows from the v7 merged data:
  - v6: AUC 0.8596; first-positive mean rank 1.49; top-4 recall 0.1927; top-8 recall 0.3491; top-16 recall 0.6658.
  - v7: AUC 0.6086; first-positive mean rank 1.08; top-4 recall 0.2170; top-8 recall 0.3868; top-16 recall 0.6685.
- Held-out `split=eval` valid rows are still small (71 rows, 20 positives,
  4 positive groups):
  - v6: AUC 0.5196; first-positive mean rank 4.00; top-4 recall 0.2000; top-8 recall 0.5500; top-16 recall 1.0000.
  - v7: AUC 0.7000; first-positive mean rank 1.50; top-4 recall 0.4500; top-8 recall 0.7500; top-16 recall 1.0000.
- Readout: v7 is a better fit for the actual DDAR top-k budget even though its
  row-level AUC is lower on the all-row pool. It becomes the recommended
  default for the next clean rerun, with v6/v5/v4 retained as fallbacks.

### `v6_preddar_v5_plus_semantic_v3_partial4_typed_v1`

- Output: `outputs/candidate_value_model_v6_preddar_v5_plus_semantic_v3_partial4_typed_v1`
- Model file: `outputs/candidate_value_model_v6_preddar_v5_plus_semantic_v3_partial4_typed_v1/candidate_value_model.json`
- Source script: `data/synth_cpt_1m_pruned_v2/run_value_model_append_partial.sh`
- Built with updated `train_candidate_value_model.py` using `feature_policy=pre_ddar_features`.
- Important fix: default value-model features now exclude post-DDAR/verdict fields
  such as `reason=...`, `candidate_ddar_status`, and `candidate_ddar_error`.
  Online reranking happens before DDAR, so those fields would be label leakage
  in training.
- Data: v5 base value data plus current semantic-v3 partial value rows, rebuilt
  from the live four-event-file benchmark snapshot.
- Rows: 7273 total, 715 positive, 6558 negative.
- Source counts after dedup: `v1`: 287, `v3`: 1858,
  `semantic_v3_partial`: 5128.
- Candidate sources: `lm`: 7027, `template_post_canonical_backfill`: 246.
- Main reasons: `valid_but_unsolved`: 2716, `valid_nonwinning`: 1567,
  `point_too_close`: 1223, `ddar_progress_positive`: 712,
  `other_error`: 527, `point_too_far`: 283.
- Metrics without post-hoc leakage:
  - train: accuracy 0.9113, loss 0.2792, AUC 0.9056
  - eval: accuracy 0.7651, loss 0.8100, AUC 0.8101
- Offline top-k reranker diagnostics from
  `scripts/evaluate_candidate_value_model.py`:
  - All valid rows from the v6 merged partial data:
    - v5 online/pre-DDAR AUC: 0.5638; top-16 group hit 0.9796; top-16 positive recall 0.6671.
    - v6 online/pre-DDAR AUC: 0.8567; top-16 group hit 1.0000; top-16 positive recall 0.7063.
  - Held-out `split=eval` valid rows are very small (71 rows, 20 positives,
    4 positive groups):
    - v5 online/pre-DDAR AUC: 0.7667; top-16 group hit 1.0000; top-16 positive recall 1.0000.
    - v6 online/pre-DDAR AUC: 0.5196; top-16 group hit 1.0000; top-16 positive recall 1.0000.
  - Readout: v6 is better on the larger partial pool and is feature-policy
    correct, but the tiny held-out split does not prove it is uniformly better
    than v5. Keep v5 as fallback and rerun this evaluation after the full
    16-problem benchmark finishes.
- Sanity checks: model `feature_policy` is `pre_ddar_features`; no
  `reason=` or `ddar_status=` weights are present.
- Purpose: replace the v5/v6-posthoc reranker with a model trained on features
  that are actually available before candidate DDAR evaluation.

### `v6_v5_plus_semantic_v3_partial4_typed_v1`

- Output: `outputs/candidate_value_model_v6_v5_plus_semantic_v3_partial4_typed_v1`
- Status: scratch model only; do not use as the recommended reranker.
- Issue found after training: the old value-model training tokenization included
  post-DDAR/verdict fields such as `reason=ddar_progress_positive` and
  `ddar_status=saturated`, producing unrealistically perfect metrics. This
  motivated the `pre_ddar_features` fix and the replacement model
  `v6_preddar_v5_plus_semantic_v3_partial4_typed_v1`.

### `value_data_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_4events_typed_v2`

- Output: `data/staged_1m_pruned_v2/value_data_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_4events_typed_v2/candidate_value_rows.jsonl`
- Built from four current v3 event files with local `candidate_value_data_type_inference_v1` scripts staged under `/tmp`, without modifying the active remote benchmark scripts.
- Snapshot time: 2026-06-07 23:01 +0800.
- Benchmark state at snapshot: 4 event files, 3 completed summary rows, 1 solved problem (`translated_imo_2004_p1`); `translated_imo_2008_p1b` was in progress.
- Rows: 6208 total, 136 positive, 6072 negative.
- Positive reasons: 135 `ddar_progress_positive`, 1 `solved_aux`.
- Main negative reasons: 2670 `valid_nonwinning`, 2368 `valid_but_unsolved`, 454 `point_too_close`, 430 `other_error`, 96 `point_too_far`, 32 `invalid_quad_solve`, 20 `candidate_ddar_error`.
- Top construction types in value rows: `on_circum`: 894, `on_pline`: 819, `on_line+on_line`: 814, `on_tline`: 792, `on_line`: 699, `on_circle`: 646, `eqangle3`: 515.
- Sources: `lm`: 5016, `template_post_canonical_backfill`: 1192.
- Purpose: provide the latest typed reranker/value data snapshot from the live run, including partial progress on `translated_imo_2008_p1b`.

### `value_data_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_typed_v1`

- Output: `data/staged_1m_pruned_v2/value_data_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_typed_v1/candidate_value_rows.jsonl`
- Built from the running v3 partial event logs with `candidate_value_data_type_inference_v1`.
- Snapshot time: 2026-06-07 22:50 +0800, while `translated_imo_2008_p1a` was still in progress.
- Rows: 5802 total, 114 positive, 5688 negative.
- Positive reasons: 113 `ddar_progress_positive`, 1 `solved_aux`.
- Main negative reasons: 2670 `valid_nonwinning`, 2079 `valid_but_unsolved`, 409 `point_too_close`, 395 `other_error`, 88 `point_too_far`.
- Invalid candidate construction types are now visible; top invalid types include `on_line+on_line`: 320, `eqangle3`: 152, `on_circum`: 121, `on_pline`: 104, `on_bline+on_line`: 52.
- Purpose: provide a typed value/reranker training snapshot where invalid and hard-negative candidates retain construction-family information.

### `v5_timeout_hardneg_features_v1_plus_v3`

- Output: `outputs/candidate_value_model_v5_timeout_hardneg_features_v1_plus_v3`
- Data sources: baseline v1 logs plus completed v3 logs.
- Added signals: timeout negatives, PointTooClose/PointTooFar/point-already-exists hard negatives, candidate source features.
- Purpose: rerank valid candidates before DDAR and reduce wasted DDAR budget.
- Follow-up instrumentation: candidate rerank scores are now attached to pruned, evaluated, DDAR-done, and DDAR-error events so timeout-heavy value-model decisions can be audited directly in later runs.

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
- Live check at 2026-06-07 22:08 +0800: still running on `translated_imo_2004_p1`, summary rows 1/16. The second problem had reached depth 3 with 1789 candidates, 1489 filtered candidates, 862 canonical duplicates, 627 depth-rank prunes, 49 DDAR runs, 42 verifier-backed candidate SFT signals, and 100 hard-negative signals.
- Live check at 2026-06-07 22:20 +0800: `translated_imo_2004_p1` solved at depth 3 with aux `n = on_line n e f, on_bline n f e`. The solving candidate target was `n : C e f n 00 D n e n f 01 ;`; candidate DDAR took 79.616 sec and added 415 dependencies. At solve time the problem had 3138 candidates, 2668 filtered candidates, 1790 canonical duplicates, 878 depth-rank prunes, 44 candidate SFT signals, and 152 hard-negative signals. The run then advanced to `translated_imo_2008_p1a`.
- Partial analysis snapshot at 2026-06-07 22:24 +0800:
  - Analysis: `outputs/unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_2rows_analysis.json`
  - Report: `outputs/unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_2rows_report.md`
  - Completed 2/16, solved 1/2 completed (`translated_imo_2004_p1`).
  - Event files: 3, because `translated_imo_2008_p1a` was already in progress.
  - Aggregate candidates: 3654; valid/invalid 3146/508, invalid rate 13.9%.
  - Filtered total: 2815; canonical duplicates 1899, 52.0% vs candidates.
  - Candidate DDAR done/errors: 66/16.
  - Candidate SFT signals: 54.
  - Candidate hard negatives: 207 (`point_too_close`: 154, `point_too_far`: 53).
  - Decision: keep the v3 run alive because it solved a genuinely new problem and is producing SFT/hard-negative signal. Do not restart solely for the later template-backfill duplicate fix; apply `template_backfill_seen_canonical_v1` on the next clean rerun.
- Partial analysis snapshot at 2026-06-07 23:01 +0800:
  - Analysis: `outputs/unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_4events_analysis_v2.json`
  - Report: `outputs/unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_4events_report_v2.md`
  - Completed 3/16, solved 1/3 completed (`translated_imo_2004_p1`); event files: 4.
  - Aggregate candidates: 6208; valid/invalid 5194/1014, invalid rate 16.3%.
  - Filtered total: 4831; canonical duplicates 3009, 48.5% vs candidates.
  - Candidate DDAR done/errors: 126/20.
  - Candidate SFT signals: 105.
  - Candidate hard negatives: 456 (`point_too_close`: 371, `point_too_far`: 85).
  - Top generated construction types: `on_tline`: 769, `on_circum`: 765, `on_pline`: 700, `on_line`: 698, `on_circle`: 646, `on_line+on_line`: 461.
  - Top hard-negative construction types: `on_line+on_line`: 240, `on_bline+on_line`: 54, `eqangle3`: 40, `on_circum+on_line`: 36, `on_circum`: 23.
  - `translated_imo_2008_p1a` completed unsolved with diagnosis `duplicate_collapse`, `candidate_ddar_timeouts`, and `symbolic_progress_but_wrong_direction`.
  - `translated_imo_2008_p1b` was in progress with 406 candidates, 314 valid, 92 invalid, 79 canonical duplicates, 16 DDAR runs, and diagnosis `symbolic_progress_but_wrong_direction`.
- Partial analysis snapshot at 2026-06-07 23:31 +0800:
  - Analysis: `outputs/unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_5events_4done_typed_v1_analysis.json`
  - Report: `outputs/unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_5events_4done_typed_v1_report.md`
  - Event files: 6; completed 5/16; solved 1/5 completed (`translated_imo_2004_p1`); `translated_imo_2009_p2` had just started with no candidates.
  - Aggregate candidates: 8490; valid/invalid 7057/1433, invalid rate 16.9%.
  - Filtered total: 6833; canonical duplicates 4040, 47.6% vs candidates.
  - Candidate DDAR done/errors: 172/38.
  - Candidate SFT signals: 142.
  - Candidate hard negatives: 664 (`point_too_close`: 541, `point_too_far`: 123).
  - `translated_imo_2008_p1b` completed unsolved with 2641 candidates, 504 invalid, 1108 canonical duplicates, 62 DDAR runs, and diagnosis `duplicate_collapse`, `candidate_ddar_timeouts`, `symbolic_progress_but_wrong_direction`.
  - `translated_imo_2008_p6` completed unsolved with 47 candidates, 7 invalid, 2 canonical duplicates, 0 DDAR runs, and diagnosis `candidate_ddar_timeout_blocked`, `valid_candidates_not_evaluated`.

### `candidate_signals_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_1row_v1`

- Output: `data/staged_1m_pruned_v2/candidate_signals_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_1row_v1`
- Built with the new partial candidate-signal data mode while the v3 ablation was still running; this reads existing event files only and does not start SFT.
- Snapshot time: 2026-06-07 22:20 +0800, after `translated_imo_2004_p1` had solved and before the full 16-problem run completed.
- Positive candidate SFT rows: 44 total, 40 train, 4 eval; counts include 43 `ddar_progress_positive` rows and 1 `candidate_solved` row.
- Hard-negative rows: 159 total, 144 train, 15 eval; reasons are 111 `point_too_close` and 48 `point_too_far`.
- Purpose: let us mine verifier-backed aux-construction signal from long-running partial benches instead of waiting for all 16 problems before any data can be inspected.

### `candidate_signals_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_typed_v1`

- Output: `data/staged_1m_pruned_v2/candidate_signals_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_typed_v1`
- Built from the same running v3 event directory with `candidate_signal_row_type_inference_v1`, without touching the active benchmark process.
- Snapshot time: 2026-06-07 22:36 +0800, while `translated_imo_2008_p1a` was still in progress.
- Positive candidate SFT rows: 69 total, 63 train, 6 eval; counts include 68 `ddar_progress_positive` rows and 1 `candidate_solved` row.
- Hard-negative rows: 305 total, 275 train, 30 eval; reasons are 237 `point_too_close` and 68 `point_too_far`.
- Positive train construction types top: `on_circum`: 8, `on_circle`: 7, `on_bline+on_line`: 6, `eqangle3`: 6.
- Hard-negative train construction types top: `on_line+on_line`: 126, `on_bline+on_line`: 42, `eqangle3`: 32, `on_circum+on_line`: 25.
- Purpose: produce an immediately usable typed signal snapshot for construction-family balancing and value/reranker auditing.

### `candidate_signals_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_4events_typed_v2`

- Output: `data/staged_1m_pruned_v2/candidate_signals_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_4events_typed_v2`
- Built from the four-event-file v3 partial snapshot with local updated builders staged under `/tmp`; this did not touch active remote benchmark scripts.
- Snapshot time: 2026-06-07 23:01 +0800.
- Positive candidate SFT rows: 105 total, 95 train, 10 eval; counts include 104 `ddar_progress_positive` rows and 1 `candidate_solved` row.
- Companion hard-negative output: `data/staged_1m_pruned_v2/candidate_hardneg_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_4events_typed_v2`
- Hard-negative rows: 456 total, 411 train, 45 eval; reasons are 371 `point_too_close` and 85 `point_too_far`.
- Hard-negative construction types top: `on_line+on_line`: 240, `on_bline+on_line`: 54, `eqangle3`: 40, `on_circum+on_line`: 36, `on_circum`: 23.
- Purpose: current best partial SFT/hard-negative snapshot while waiting for the full 16-problem run.

### `candidate_signals_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_5events_4done_typed_v1`

- Output: `data/staged_1m_pruned_v2/candidate_signals_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_5events_4done_typed_v1`
- Snapshot time: 2026-06-07 23:31 +0800, after 5 completed problems and while `translated_imo_2009_p2` had just started.
- Positive candidate SFT rows: 142 total, 128 train, 14 eval; counts include 141 `ddar_progress_positive` rows and 1 `candidate_solved` row.
- Companion hard-negative output: `data/staged_1m_pruned_v2/candidate_hardneg_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_5events_4done_typed_v1`
- Hard-negative rows: 664 total, 598 train, 66 eval; reasons are 541 `point_too_close` and 123 `point_too_far`.
- Companion value-data output: `data/staged_1m_pruned_v2/value_data_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_5events_4done_typed_v1/candidate_value_rows.jsonl`
- Value rows: 8490 total, 179 positive, 8311 negative.
- Top value-row reasons: `valid_but_unsolved`: 4169, `valid_nonwinning`: 2670, `point_too_close`: 650, `other_error`: 593, `ddar_progress_positive`: 178, `point_too_far`: 137.
- Top value-row construction types: `on_circum`: 1213, `on_pline`: 1126, `on_line+on_line`: 1070, `on_tline`: 1063, `on_line`: 960, `on_circle`: 894, `eqangle3`: 714.
- Purpose: latest partial data snapshot before the full 16-problem run finishes.

## Next Candidate-Quality Fixes

### `pairwise_value_ranker_v7_v1`

- `scripts/train_candidate_value_model.py` now supports
  `--objective pairwise` in addition to the existing logistic objective.
- Pairwise training groups candidates by `--pairwise_group_by`
  (default `problem,depth`) and optimizes positive candidates to rank above
  sampled negatives from the same group.
- Added `--train_valid_only` to train value/reranker models only on candidates
  that are actually available to online reranking after translation/validation.
- `data/synth_cpt_1m_pruned_v2/run_value_model_append_partial.sh` now accepts
  `VALUE_TRAIN_EXTRA_ARGS`, allowing queue scripts to select pairwise training
  without editing the script.
- `data/synth_cpt_1m_pruned_v2/run_fact_context_unsolved_ablation.sh` now
  defaults to the v7 pairwise value model, with v6, v5, and v4 fallbacks.

### `value_model_topk_eval_v1`

- Added `scripts/evaluate_candidate_value_model.py`.
- The evaluator scores candidate value rows with online-compatible pre-DDAR
  features by default, filters to valid translated candidates by default, and
  reports DDAR-budget-oriented metrics grouped by `problem,depth`.
- Reported metrics include AUC, first positive rank, input-order first positive
  rank, top-k group hit-rate, top-k positive recall, precision, and missed
  positive construction types.
- Added optional `--split train|eval` filtering so held-out rows can be checked
  separately from all partial rows.
- Purpose: make reranker selection auditable under the actual top-k DDAR budget
  instead of relying only on whole-row AUC or training loss.

### `next_ablation_uses_preddar_v6_v1`

- `data/synth_cpt_1m_pruned_v2/run_fact_context_unsolved_ablation.sh` now
  defaults `VALUE_MODEL_PREFERRED` to
  `outputs/candidate_value_model_v6_preddar_v5_plus_semantic_v3_partial4_typed_v1/candidate_value_model.json`.
- `VALUE_MODEL_FALLBACK` now points to the prior v5 model, and
  `VALUE_MODEL_LEGACY_FALLBACK` keeps the older v4 fallback.
- Purpose: ensure the next clean fact-context unsolved rerun uses the corrected
  pre-DDAR reranker by default, while preserving an explicit override through
  `QWEN_CANDIDATE_VALUE_MODEL`.
- Superseded by `pairwise_value_ranker_v7_v1`, which defaults the next rerun
  to the pairwise v7 model and keeps v6 as the first fallback.

### `preddar_value_model_v6_partial_v1`

- `scripts/train_candidate_value_model.py` now defaults to pre-DDAR feature
  training. It excludes `reason`, `candidate_ddar_status`, and
  `candidate_ddar_error` unless `--include_posthoc_features` is explicitly set.
- `scripts/qwen_ag_search.py` online value-model scoring now mirrors that
  pre-DDAR feature policy: raw candidate text, translated construction,
  construction type/type combo, translation error class, and candidate source.
- New orchestration script:
  `data/synth_cpt_1m_pruned_v2/run_value_model_append_partial.sh`.
  It builds value rows from a partial benchmark event directory, appends them to
  an existing value dataset, writes a merge summary, and trains a JSON reranker.
- Smoke check verified that default training tokens contain no `reason=`,
  `ddar_status=`, or post-DDAR timeout error tokens, while
  `--include_posthoc_features` still exposes them for controlled ablations.
- Remote training produced the recommended next-rerun model
  `v6_preddar_v5_plus_semantic_v3_partial4_typed_v1`.

### `new_solved_baseline_highlight_v1`

- `scripts/analyze_qwen_ag_events.py` now accepts `--baseline_summary_jsonl`.
- Analysis JSON now records `baseline_summary_jsonl`, `baseline_solved_names`,
  `num_new_solved`, and `new_solved_names`.
- `scripts/report_qwen_ag_analysis.py` renders a top-of-report
  `New Solved Vs Baseline` block before the ordinary failure-analysis summary.
- Verification run against baseline
  `outputs/final_eval_imo_ag30_qwen_unsolved_high_budget_value_v3_cost_template_depth16_v1/summary.jsonl`
  produced:
  - Analysis: `outputs/unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_4events_analysis_v4_baseline.json`
  - Report: `outputs/unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_4events_report_v4_baseline.md`
  - `new_solved_names`: `translated_imo_2004_p1`.
- Purpose: make newly solved problems impossible to miss in future status
  reports, since this is the primary expected signal from the optimization loop.

### `candidate_value_data_type_inference_v1`

- `scripts/build_candidate_value_data.py` now infers `construction_type` for
  invalid candidates from raw DSL `raw` text when `translation` is an `ERROR`.
- `scripts/train_candidate_value_model.py` and online scorer tokens now include
  `type_combo=...` in addition to per-construction `type=...` tokens.
- Purpose: let the value/reranker model learn construction-combination signals
  for PointTooClose/PointTooFar/invalid candidates, rather than collapsing them
  into the generic `error` construction bucket.

### `template_initial_backfill_source_v1`

- Initial template backfill candidates are now marked as
  `source=template_initial_backfill` instead of being logged as `lm`.
- The source is carried into candidate events, duplicate-filter events,
  translated candidate records, and generator-side hard-negative signals.
- Applies in both `scripts/run_qwen_ag_benchmark.py` and standalone
  `scripts/qwen_ag_search.py`.
- Purpose: keep source features honest for value-model/reranker audits and make
  it possible to separate LM distribution errors from template fallback noise.

### `template_backfill_generation_dedup_v1`

- Initial template backfill now compares candidate canonical generation keys
  against the current generate-call candidate set, not only raw text.
- Applies in both `scripts/run_qwen_ag_benchmark.py` and the standalone
  `scripts/qwen_ag_search.py` search entry.
- Purpose: avoid adding template candidates that are raw-text different but
  canonical-equivalent to LM-generated candidates in the same decode step. This
  reduces needless translation/logging and later `duplicate_canonical` filters
  in the next clean run.

### `generation_canonical_dedup_v1`

- Adds `candidate_generation_dedup_key` in `scripts/qwen_ag_search.py`.
- Qwen generation now de-duplicates candidates within a single generate call by
  lightweight canonical auxiliary key, not just raw generated text.
- Examples treated as the same candidate:
  - `n : C e f n 00 D n e n f 01 ;`
  - `n = on_line n e f, on_bline n f e;`
- Purpose: reduce same-decode duplicates before candidates reach AG graph
  validation, canonical filtering, or template backfill. This is a future-run
  candidate-quality fix and does not affect the already-running v3 process.

### `candidate_signal_row_type_inference_v1`

- Candidate signal row builders now infer `candidate_construction_type` when
  older event files do not contain it.
- For valid candidates, the type is inferred from `translation`; for invalid
  PointTooClose/PointTooFar hard negatives, the type is inferred from the raw
  DSL `target` using the lightweight DSL-to-constructive translator.
- Purpose: make partial datasets from currently-running old-format benchmarks
  useful for construction-family sampling and hard-negative analysis without
  waiting for a fresh run with richer event fields.

### `candidate_signal_type_fields_v1`

- Candidate SFT signal events now include `candidate_source`,
  `candidate_construction_type`, and `candidate_rerank_score`.
- Candidate hard-negative signal events now include `candidate_source` and
  `candidate_construction_type`.
- Candidate signal JSONL builders carry these fields into positive and
  hard-negative rows.
- Analysis now resolves construction type from `target` for older
  hard-negative events, so PointTooClose/PointTooFar logs are no longer forced
  into a generic `error` construction bucket when the raw DSL target is present.
- Purpose: make positive and negative auxiliary-construction signals usable for
  construction-family balancing, reranker/value-model audits, and targeted
  hard-negative training.

### `candidate_signal_partial_snapshot_mode_v1`

- Adds `ALLOW_PARTIAL=1` to
  `data/synth_cpt_1m_pruned_v2/run_candidate_signal_data_after_ablation_wait.sh`.
- Default behavior is unchanged: the queue still waits until
  `EXPECTED_PROBLEMS` summary rows are present.
- In partial mode the data builder may proceed once configurable thresholds are
  met: `MIN_PARTIAL_SUMMARY_ROWS`, `MIN_PARTIAL_SFT_SIGNALS`, and
  `MIN_PARTIAL_HARD_NEGATIVE_SIGNALS`.
- Purpose: support training-data inspection and snapshot extraction from
  long-running ablations without forcing candidate-signal SFT to wait for a
  full 16/16 benchmark.

### `template_backfill_seen_canonical_v1`

- Adds a lightweight DSL-to-constructive translation helper for generated
  candidate text, so template candidates can be canonicalized before AG graph
  validation.
- Passes the problem-level `seen_candidate_keys` set into template backfill.
  Template fallback now avoids globally seen canonical auxiliary clauses at
  generation time instead of logging them and filtering them later.
- Purpose: reduce cross-node/cross-depth duplicate template candidates and cut
  translation/log overhead. This does not affect the already-running
  `unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1`
  process; it applies to the next clean rerun.

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
