#!/usr/bin/env bash
set -euo pipefail

WORK=/root/rivermind-data/qwen_ag_lm
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

FACT_TAG=${FACT_TAG:-factctx_top8_after_v3_v1}
FACT_ADAPTER=${FACT_ADAPTER:-outputs/stage2_fact_context_after_stage2_lora_qwen2_5_7b_${FACT_TAG}}
ADAPTER_READY_FILE=${ADAPTER_READY_FILE:-$FACT_ADAPTER/adapter_model.safetensors}
ABLATION_TAG=${ABLATION_TAG:-unsolved_factctx_top8_adapter_value_v5_grammar_v1}
LOG=${LOG:-outputs/${ABLATION_TAG}.wait_then_eval.log}
WAIT_INTERVAL=${WAIT_INTERVAL:-300}
WAIT_TIMEOUT_SEC=${WAIT_TIMEOUT_SEC:-0}
DRY_RUN=${DRY_RUN:-0}

VALUE_MODEL_PREFERRED=${VALUE_MODEL_PREFERRED:-outputs/candidate_value_model_v7_pairwise_preddar_v5_plus_semantic_v3_partial4_typed_v1/candidate_value_model.json}
VALUE_MODEL_FALLBACK=${VALUE_MODEL_FALLBACK:-outputs/candidate_value_model_v6_preddar_v5_plus_semantic_v3_partial4_typed_v1/candidate_value_model.json}
VALUE_MODEL_LEGACY_FALLBACK=${VALUE_MODEL_LEGACY_FALLBACK:-outputs/candidate_value_model_v5_timeout_hardneg_features_v1_plus_v3/candidate_value_model.json}
VALUE_MODEL_OLDEST_FALLBACK=${VALUE_MODEL_OLDEST_FALLBACK:-outputs/candidate_value_model_v4_hardneg_features_from_qwen_v1_logs/candidate_value_model.json}

select_value_model() {
  if [ -n "${QWEN_CANDIDATE_VALUE_MODEL:-}" ]; then
    printf '%s\n' "$QWEN_CANDIDATE_VALUE_MODEL"
  elif [ -s "$VALUE_MODEL_PREFERRED" ]; then
    printf '%s\n' "$VALUE_MODEL_PREFERRED"
  elif [ -s "$VALUE_MODEL_FALLBACK" ]; then
    printf '%s\n' "$VALUE_MODEL_FALLBACK"
  elif [ -s "$VALUE_MODEL_LEGACY_FALLBACK" ]; then
    printf '%s\n' "$VALUE_MODEL_LEGACY_FALLBACK"
  else
    printf '%s\n' "$VALUE_MODEL_OLDEST_FALLBACK"
  fi
}

if [ "$DRY_RUN" = "1" ]; then
  echo "fact adapter: $FACT_ADAPTER"
  echo "adapter ready file: $ADAPTER_READY_FILE"
  echo "ablation tag: $ABLATION_TAG"
  echo "preferred value model: $VALUE_MODEL_PREFERRED"
  echo "fallback value model: $VALUE_MODEL_FALLBACK"
  echo "legacy fallback value model: $VALUE_MODEL_LEGACY_FALLBACK"
  echo "oldest fallback value model: $VALUE_MODEL_OLDEST_FALLBACK"
  SELECTED_VALUE_MODEL=$(select_value_model)
  echo "selected value model: $SELECTED_VALUE_MODEL"
  FINAL_ADAPTER_OVERRIDE="$FACT_ADAPTER" \
  UNSOLVED_TAG="$ABLATION_TAG" \
  QWEN_LM_FACT_CONTEXT_TOP_K="${QWEN_LM_FACT_CONTEXT_TOP_K:-8}" \
  QWEN_CANDIDATE_RERANK="${QWEN_CANDIDATE_RERANK:-value_model_diverse}" \
  QWEN_CANDIDATE_VALUE_MODEL="$SELECTED_VALUE_MODEL" \
  QWEN_CANDIDATE_DEPTH_EVAL_LIMIT="${QWEN_CANDIDATE_DEPTH_EVAL_LIMIT:-16}" \
  QWEN_CANDIDATE_WALL_TIMEOUT="${QWEN_CANDIDATE_WALL_TIMEOUT:-90}" \
  DRY_RUN=1 \
  bash data/synth_cpt_1m_pruned_v2/run_unsolved_high_budget_qwen.sh
  exit 0
fi

echo "waiting for fact-context adapter: $ADAPTER_READY_FILE" | tee -a "$LOG"
start_epoch=$(date +%s)
while [ ! -s "$ADAPTER_READY_FILE" ]; do
  date '+%F %T %z' | tee -a "$LOG"
  if [ "$WAIT_TIMEOUT_SEC" -gt 0 ]; then
    now_epoch=$(date +%s)
    elapsed=$((now_epoch - start_epoch))
    if [ "$elapsed" -ge "$WAIT_TIMEOUT_SEC" ]; then
      echo "timed out waiting for adapter after ${elapsed}s" | tee -a "$LOG"
      exit 1
    fi
  fi
  sleep "$WAIT_INTERVAL"
done

echo "starting fact-context unsolved ablation: $ABLATION_TAG" | tee -a "$LOG"
SELECTED_VALUE_MODEL=$(select_value_model)
echo "selected value model: $SELECTED_VALUE_MODEL" | tee -a "$LOG"
FINAL_ADAPTER_OVERRIDE="$FACT_ADAPTER" \
UNSOLVED_TAG="$ABLATION_TAG" \
QWEN_LM_FACT_CONTEXT_TOP_K="${QWEN_LM_FACT_CONTEXT_TOP_K:-8}" \
QWEN_CANDIDATE_RERANK="${QWEN_CANDIDATE_RERANK:-value_model_diverse}" \
QWEN_CANDIDATE_VALUE_MODEL="$SELECTED_VALUE_MODEL" \
QWEN_CANDIDATE_DEPTH_EVAL_LIMIT="${QWEN_CANDIDATE_DEPTH_EVAL_LIMIT:-16}" \
QWEN_CANDIDATE_WALL_TIMEOUT="${QWEN_CANDIDATE_WALL_TIMEOUT:-90}" \
bash data/synth_cpt_1m_pruned_v2/run_unsolved_high_budget_qwen.sh \
  >> "$LOG" 2>&1

echo "done fact-context unsolved ablation: $ABLATION_TAG" | tee -a "$LOG"
