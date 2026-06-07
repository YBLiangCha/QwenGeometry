#!/usr/bin/env bash
set -euo pipefail

WORK=/root/rivermind-data/qwen_ag_lm
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

WAIT_TAG=${WAIT_TAG:-unsolved_high_budget_value_v3_cost_template_depth16_v1}
FACT_TAG=${FACT_TAG:-factctx_top8_after_v3_v1}
WAIT_INTERVAL=${WAIT_INTERVAL:-300}
LOG=outputs/${FACT_TAG}.wait_then_train.log

echo "waiting for run tag: $WAIT_TAG" | tee -a "$LOG"
while pgrep -f "$WAIT_TAG" >/dev/null; do
  date '+%F %T %z' | tee -a "$LOG"
  pgrep -af "$WAIT_TAG" | tee -a "$LOG" || true
  sleep "$WAIT_INTERVAL"
done

echo "starting fact-context SFT: $FACT_TAG" | tee -a "$LOG"
FACT_TAG="$FACT_TAG" \
FACT_CONTEXT_TOP_K="${FACT_CONTEXT_TOP_K:-8}" \
FACT_CONTEXT_MAX_LEVEL="${FACT_CONTEXT_MAX_LEVEL:-4}" \
FACT_CONTEXT_DDAR_TIMEOUT="${FACT_CONTEXT_DDAR_TIMEOUT:-30}" \
OLD_FORMAT_MAX_ROWS="${OLD_FORMAT_MAX_ROWS:-4000}" \
PROMOTE="${PROMOTE:-0}" \
bash data/synth_cpt_1m_pruned_v2/run_fact_context_stage2_sft.sh \
  >> "$LOG" 2>&1

echo "done fact-context SFT: $FACT_TAG" | tee -a "$LOG"
