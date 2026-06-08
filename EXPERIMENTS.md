# Qwen + AlphaGeometry Experiment Registry

This file tracks experiment tags, output directories, and the main behavioral
changes. Tags are the practical version identifiers for this workspace.

## Source Version State

- Git remote: `git@github.com:YBLiangCha/QwenGeometry.git`
- Current GitHub source head: `postv12_duplicate_clause_filter_v1`.
- Current running v12 clean bench tag:
  `unsolved_factctx_promptaug_top8_candidate_signal_postrun_value_v12_default_v1_depth48_t240_w150_nrs48_qm3_sigrep4_blinedia_statededup_nodediv_dsltpl_combotpl_rarecombo_vprior_v1`.
- Current v12 clean source snapshot:
  `/tmp/qwen_ag_scripts_c541dd4` (`value_v12_default_queue_v1`). This active
  process is intentionally not overwritten.
- Current waiting scout queue source:
  `/tmp/qwen_ag_scripts_postv12_duplicate_clause_filter_v1`, tag
  `unsolved_factctx_promptaug_top8_hybrid_v12_front12_v18coverage_after_v12_depth16_decbeam16_t160_w100_nrs48_qm3_timeoutfb4_beamscore_rerank_factmem_binddedup_salvage_v1`.
- Current waiting post-v12 stage4 queue source:
  `/tmp/qwen_ag_scripts_postv12_duplicate_clause_filter_v1`, tag
  `unsolved_factctx_promptaug_top8_stage4_solvedbiased_postv12_hybrid_v12_front12_v18coverage_beamscore_rerank_decbeam16_depth24_t200_w120_nrs48_qm3_timeoutfb4_factmem_binddedup_salvage_v1`.
- Running-workspace scripts are intentionally versioned in `/tmp` snapshots
  rather than overwritten in-place. The benchmark uses spawn-based candidate
  workers, so mixing source versions inside a long process can corrupt
  attribution.

## Candidate Quality And Training Signals

### `postrun_candidate_signal_queue_v1`

- Added
  `data/synth_cpt_1m_pruned_v2/run_postrun_candidate_signal_sft_and_clean_rerun.sh`.
- The script waits for the active semantic-v3 benchmark process to exit before
  using the GPU. It then builds postrun analysis, extracts positive
  candidate-signal rows and hard negatives, mixes them with the prompt-aug
  fact-context rows, and trains a candidate-signal LoRA initialized from
  `factctx_promptaug_top8_stage2max2000_v1`.
- It can optionally launch the next clean 16-problem rerun with the trained
  adapter and the v11 value model by setting `RUN_CLEAN_RERUN=1`.
- Purpose: keep progress moving without overwriting active benchmark scripts or
  stealing GPU memory from the currently running process.

### `postrun_clean_depth24_queue_v1`

- Updates
  `data/synth_cpt_1m_pruned_v2/run_postrun_candidate_signal_sft_and_clean_rerun.sh`
  so the queued clean rerun exposes `CLEAN_CANDIDATE_EVAL_LIMIT`,
  `CLEAN_CANDIDATE_DEPTH_EVAL_LIMIT`, and `CLEAN_CANDIDATE_DDAR_WORKERS`.
- Default clean rerun depth budget is raised from 16 to 24 and the default tag
  now includes `depth24`. This matches
  `run_unsolved_high_budget_qwen.sh` more closely and gives the reranker a wider
  DDAR-verified candidate window when chasing the AG1-style baseline.
- The active semantic-v3 benchmark is still untouched; this only changes the
  postrun queue that waits for it to finish before training and launching the
  clean rerun.

### `postrun_signal_repeat_depth24_queue_v1`

- Adds `SIGNAL_REPEAT` to the postrun queue, defaulting to 4 for the training
  mix only. Eval rows are not repeated.
- Motivation: the postrun candidate-signal set is expected to be hundreds of
  rows, while the fact-context replay set contributes up to 2000 train rows.
  Repeating positive DDAR-progress/candidate-solved rows keeps the SFT focused
  on the new auxiliary-construction signal instead of letting it be diluted by
  fact-context replay.
- This keeps the clean rerun at the depth-24 budget introduced by
  `postrun_clean_depth24_queue_v1`.

### `high_value_template_backfill_queue_v1`

- Expands `template_backfill_candidates` from 7 to 12 construction buckets.
  New fallback families include `angle_bisector`, `angle_mirror`,
  `on_aline`, `on_aline2`, and `eqangle3`.
- Motivation from the active semantic-v3 run: after 7 event files there are
  13 `candidate_backfill_exhausted` events, mostly on `translated_imo_2010_p2`,
  and the finite template set tops out around 96 candidates before AG
  validation. The expanded high-value templates give the clean rerun more
  diverse candidates when LM generations collapse to duplicates.
- The postrun queue still uses `SIGNAL_REPEAT=4` and depth-24 clean rerun
  settings from the previous version.

### `goal_aware_template_backfill_queue_v1`

- Adds an optional `preferred_points` argument to
  `template_backfill_candidates`; benchmark and standalone search now pass the
  current goal predicate's point names.
- Pair/triple template pools are ordered by how many preferred points they
  contain before the existing spread sampler selects diverse candidates.
- Motivation: high-value fallback templates are only useful if their point
  choices are relevant. Goal-aware ordering should make template backfill spend
  more of its budget around the target relation instead of mostly early
  alphabetical point combinations.

### `wide_clean_rerun_queue_v1`

- Makes clean-rerun search strength configurable through
  `CLEAN_BEAM_SIZE`, `CLEAN_SEARCH_DEPTH`, `CLEAN_NUM_RETURN_SEQUENCES`, and
  `CLEAN_CANDIDATE_QUALITY_MULTIPLIER`.
- Default clean rerun budget is widened to `num_return_sequences=48`,
  `candidate_quality_multiplier=3`, and `candidate_depth_eval_limit=32`.
- Motivation: after candidate-signal SFT and goal-aware high-value templates,
  the clean rerun should give the generator and reranker enough candidates to
  surface useful auxiliary constructions. This is more expensive than the
  earlier depth-24 queue, but better aligned with chasing an AG1-style score.

### `semantic_v3_partial_7events_6summary_v1`

- Candidate-signal data:
  `data/staged_1m_pruned_v2/candidate_signals_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_7events_6summary_v1`
- Current status at this snapshot: 7 event files, 6 completed summary rows,
  and still only 1 solved problem:
  `translated_imo_2004_p1`.
- Training signal extraction:
  - positive SFT rows: 212 total, 191 train, 21 eval; 211
    `ddar_progress_positive`, 1 `candidate_solved`.
  - hard-negative rows: 961 total, 865 train, 96 eval; 779
    `point_too_close`, 182 `point_too_far`.
- Value-model check:
  `outputs/candidate_value_model_v12_logistic_preddar_nodup_semantic_v3_partial7events6summary_v1`
  adds the early `translated_imo_2010_p2` signals, but it is not promoted over
  v11. On the updated online-rerank distribution v12 improves all-row top-16
  recall slightly (0.6437 vs v11 0.6360) but regresses top-4, top-8, and the
  small eval split. Keep v11 as the default clean-rerun value model.
- Next safe SFT candidate should use this 7-event data rather than the older
  6-event snapshot if the active benchmark exits before substantially more
  events arrive.

### `semantic_v3_partial_6events_5summary_v1`

- Analysis:
  `outputs/unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_6events_5summary_analysis_v1.json`
- Report:
  `outputs/unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_6events_5summary_report_v1.md`
- Candidate-signal data:
  `data/staged_1m_pruned_v2/candidate_signals_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial_6events_5summary_v1`
- Current status at this snapshot: 6 event files, 5 completed summary rows,
  1 solved problem, and 1 new solved problem vs baseline:
  `translated_imo_2004_p1`.
- Aggregate candidate stats: 11513 candidates, 9579 valid, 1934 invalid,
  9069 filtered, 5663 `duplicate_canonical`, 220 candidate DDAR completions,
  and 38 candidate DDAR errors.
- Training signal extraction:
  - positive SFT rows: 184 total, 166 train, 18 eval; 183
    `ddar_progress_positive`, 1 `candidate_solved`.
  - hard-negative rows: 937 total, 844 train, 93 eval; 756
    `point_too_close`, 181 `point_too_far`.
- Current duplicate-filter rows in this running process still lack
  `prompt/target`, so they cannot yet become generator hard-negative SFT rows.
  That will only happen in a clean rerun using
  `duplicate_canonical_negative_signal_v1` or later source.
- Next safe SFT candidate after the active bench exits:
  `candidate_signal_sft_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_partial6_v1`,
  initialized from
  `outputs/stage2_fact_context_after_stage2_lora_qwen2_5_7b_factctx_promptaug_top8_stage2max2000_v1`
  and mixed with
  `data/staged_1m_pruned_v2/factctx_promptaug_top8_stage2max2000_v1/fact_context_mixed_train.jsonl`.
  Do not start this GPU SFT while the current benchmark process is still using
  the GPU.

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

### `v11_logistic_preddar_nodup_semantic_v3_partial6events5summary_v1`

- Output:
  `outputs/candidate_value_model_v11_logistic_preddar_nodup_semantic_v3_partial6events5summary_v1`
- Model file:
  `outputs/candidate_value_model_v11_logistic_preddar_nodup_semantic_v3_partial6events5summary_v1/candidate_value_model.json`
- Objective: `logistic`; feature policy: `pre_ddar_features`;
  `train_valid_only=true`; `excluded_reasons=["duplicate_canonical"]`.
- Data: v5 base value data plus current semantic-v3 partial value rows after
  6 event files and 5 summary rows, including the in-progress
  `translated_imo_2009_p2` snapshot.
- Rows before training filter: 9389 total, with 2651
  `duplicate_canonical` rows retained in the data file for analysis.
- Rows after online-rerank filter: 4110 valid non-duplicate rows with
  748 positives and 3362 negatives.
- Offline top-k diagnostics on that same online-rerank distribution:
  - v6: AUC 0.8282; first-positive mean rank 1.49; top-8 recall 0.3476;
    top-16 recall 0.6511.
  - v7: AUC 0.6082; first-positive mean rank 1.08; top-8 recall 0.3770;
    top-16 recall 0.6444.
  - v8: AUC 0.6398; first-positive mean rank 1.06; top-8 recall 0.3877;
    top-16 recall 0.6377.
  - v10: AUC 0.6231; first-positive mean rank 1.06; top-8 recall 0.3757;
    top-16 recall 0.6444.
  - v11: AUC 0.8584; first-positive mean rank 1.16; top-8 recall 0.3944;
    top-16 recall 0.6684.
- Held-out eval remains tiny: 71 valid rows, 20 positives, 4 positive groups.
  v11 keeps top-16 recall at 1.0 there but does not dominate v7/v8 on top-8.
- Readout: v11 is the best current model for the actual top-16 DDAR budget on
  the partial online-rerank pool, so it becomes the preferred value model for
  the next clean rerun, with v7, v6, v5, and v4 retained as fallbacks.

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

## Post-v12 Queue Versions

### `timeout_beam_fallback_v1`

- Commit: `c99ca6c`.
- Adds `--candidate_timeout_beam_fallback_limit`, defaulting to 0 for
  compatibility.
- Motivation: `translated_imo_2008_p6` in the v12 clean run had only 2 DDAR
  completions, 95 candidate DDAR timeout errors, and then `beam_empty` at
  depth 1. The fallback carries a small number of top timed-out candidates into
  the next beam without fact context when no verified candidate survives.
- Post-v12 queues use `timeoutfb4`; the active v12 clean process remains on
  the older `c541dd4` source.

### `postv12_refresh_value_model_v1`

- Commit: `4985dd8`.
- The waiting scout can train a refreshed pairwise value model after the full
  v12 clean run completes, then use it as the primary frontfill reranker with
  the v12 logistic model as coverage.
- Motivation: earlier v16 pairwise value training only saw partial current-run
  events. The post-v12 refresh lets the scout absorb the completed v12 hard
  negatives, positives, and solved rows before reranking remaining unsolved
  problems.

### `postv12_solvedonly_value_strict_stage4_v1`

- Commit: `e2fc5ed`.
- Changes the refreshed value model default to
  `v18_pairwise_postv12_solvedonly_timeoutfb4_v1`, with progress positives
  disabled during the value-data append.
- Tightens stage4 candidate-signal SFT defaults:
  `SIGNAL_MIN_PROGRESS_DELTA=80`, `SIGNAL_MAX_ELAPSED_SEC=90`,
  `SIGNAL_MIN_PROGRESS_EFFICIENCY=1.0`,
  `SIGNAL_MAX_PROGRESS_ROWS_PER_PROBLEM=8`,
  `SIGNAL_MAX_PROGRESS_ROWS_PER_TYPE=16`, and
  `SIGNAL_SOLVED_REPEAT=32`.
- Motivation: completed failures such as `translated_imo_2008_p1a` and
  `translated_imo_2008_p1b` produced many DDAR-progress positives but no
  solve, suggesting that "more derived facts" is a noisy proxy for correct
  direction.

### `postv12_rerank_beam_score_v1`

- Commit: `f25fccf`.
- Adds `--candidate_beam_score` with choices `lm_score`, `rerank_score`, and
  `lm_plus_rerank`; default remains `lm_score`.
- Post-v12 scout/stage4 use `candidate_beam_score=rerank_score`.
- Motivation: current event traces show LM scores are often effectively 0.0,
  so the value model was ordering candidate DDAR evaluation but not really
  deciding which states persisted into later search depths.

### `postv12_decode_beam_limit_v1`

- Commit: `a1697a3`.
- Adds `--candidate_decode_beam_limit`, defaulting to 0 for compatibility.
- Post-v12 scout/stage4 use `candidate_decode_beam_limit=16`, with
  `beam_decode_pruned` events logging skipped beam states.
- Motivation: `translated_imo_2009_p2` generated thousands of depth-1/depth-2
  candidates while only a small depth-eval window was DDAR-checked. Once beam
  scores are value-rerank driven, decoding only the top beam states should
  reduce wasted generation and keep later-depth search focused.

### `postv12_v12_primary_v18_coverage_v1`

- Git tag: `postv12_v12_primary_v18_coverage_v1`.
- Adds `SCOUT_REFRESH_VALUE_ROLE={primary,secondary,none}` to the scout queue.
  The refreshed post-v12 pairwise value model can now be trained after the
  reference run and attached as the secondary coverage model instead of
  overwriting the primary value model.
- Changes the scout default to v12 logistic primary with v16 pairwise coverage;
  the deployed queue trains `v18_pairwise_postv12_solvedonly_timeoutfb4_v1`
  with progress positives disabled and uses it as secondary coverage.
- Changes the stage4 clean default to v12 primary plus v18 solved-only
  coverage, with v16 as a secondary fallback if the refreshed v18 model is not
  present.
- Motivation: a preview model trained from the first 5 rows of the active v12
  clean run improved over v16 but still underperformed the v12 logistic model
  on partial-data AUC/top-k recall. Keeping v12 as the primary frontfill
  scorer is therefore safer, while v18 still contributes solved-only coverage.

### `postv12_dynamic_fact_memory_v1`

- Git tag: `postv12_dynamic_fact_memory_v1`.
- The benchmark driver now stores fact context in each beam state and merges a
  small amount of parent DDAR fact context with the newly selected child DDAR
  facts before building the next LM prompt.
- Timeout beam fallback also carries the parent fact context instead of
  rebuilding the prompt with an empty `{D}` fact block.
- Motivation: dynamic DDAR facts were already reinserted into prompts, but each
  depth kept only the current node's top facts. This could discard root or
  parent facts that shaped a promising branch. The fact-memory merge preserves
  continuity while still giving most of the top-k slots to fresh child facts.

### `postv12_constructive_binding_filter_v1`

- Git tag: `postv12_constructive_binding_filter_v1`.
- Tightens raw DSL shape filtering for multi-construction constructive clauses:
  if a candidate has two RHS constructions, at least one construction must
  explicitly bind the newly constructed point as its first argument.
- Single `eqangle3` candidates are still allowed, and mixed candidates such as
  `eqangle3 + on_line` remain allowed. The filtered case is malformed output
  such as `n = eqangle3 ..., eqangle3 ...;`, where neither RHS construction
  binds `n`.
- Motivation: the active v12 clean run on `translated_imo_2009_p2` showed
  repeated double-`eqangle3` LM candidates causing AG clause parsing errors
  like `Cannot find point  in graph`. These should be removed by grammar/shape
  filtering before translation and hard-negative logging.

### `postv12_duplicate_clause_filter_v1`

- Git tag: `postv12_duplicate_clause_filter_v1`.
- Adds exact duplicate-clause checks for both constructive RHS parts and
  constrained predicate parts, in complete-candidate filtering and prefix
  status validation.
- Motivation: `translated_imo_2009_p2` event traces showed redundant candidates
  and prompts such as `n : T n f d c 00 T n f d c ;` and
  `m : O e f h m 17 O e f h m 18`. Exact duplicate clauses add no geometric
  information and can inflate DDAR/prompt complexity.

## Versioning Rule

- Do not overwrite completed output directories.
- Use a new tag when changing training data, prompt schema, reranker features, search budget, or candidate filtering behavior.
- Keep queue scripts parameterized by tag-specific environment variables.
- Refresh analysis JSON/Markdown after every completed benchmark run.
