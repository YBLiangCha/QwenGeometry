#!/usr/bin/env bash
set -euo pipefail
WORK=/root/rivermind-data/qwen_ag_lm
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"
STAGED=data/staged_1m_pruned_v2
STAGE1=outputs/stage1_cpt_1m_pruned_packed_lora_qwen2_5_7b_v1
STAGE2=outputs/stage2_aux_after_cpt1m_lora_qwen2_5_7b_v1
STAGE3=outputs/stage3_rejection_after_stage2_lora_qwen2_5_7b_v1
STAGE3_DATA="$STAGED/stage3_rejection_sft.jsonl"
STAGE3_SUMMARY="$STAGED/stage3_rejection_summary.json"
while [ ! -s "$STAGE1/adapter_model.safetensors" ] || ! grep -q "train_loss" "$STAGE1/train.log" 2>/dev/null; do
  sleep 180
done
while [ ! -s "$STAGED/stage2_aux_sft_train.jsonl" ] || [ ! -s "$STAGED/stage2_aux_sft_eval.jsonl" ]; do
  sleep 120
done
python - <<'PY' > "$STAGED/stage2_data_health.json"
import json, sys
from pathlib import Path
staged=Path('data/staged_1m_pruned_v2')
def count_rows(path):
    bad=0; rows=0
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.strip():
            continue
        rows += 1
        try:
            obj=json.loads(line)
            if 'prompt' not in obj or 'target' not in obj:
                bad += 1
        except Exception:
            bad += 1
    return rows, bad
train,bad_train=count_rows(staged/'stage2_aux_sft_train.jsonl')
eval_rows,bad_eval=count_rows(staged/'stage2_aux_sft_eval.jsonl')
health={'train_rows':train,'eval_rows':eval_rows,'bad_train':bad_train,'bad_eval':bad_eval,'passed':train>=1000 and eval_rows>=10 and bad_train==0 and bad_eval==0}
print(json.dumps(health, ensure_ascii=False, indent=2))
if not health['passed']:
    sys.exit(4)
PY
rm -rf "$STAGE2"
mkdir -p "$STAGE2"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run_stage2() {
  local batch_size="$1"
  local grad_accum="$2"
  local log_file="$3"
  python -u scripts/train_qwen_aux_lora.py \
    --model_name_or_path models/Qwen2.5-7B \
    --init_adapter_path "$STAGE1" \
    --train_file "$STAGED/stage2_aux_sft_train.jsonl" \
    --eval_file "$STAGED/stage2_aux_sft_eval.jsonl" \
    --output_dir "$STAGE2" \
    --loss_mode target \
    --max_length 1024 \
    --learning_rate 8e-5 \
    --num_train_epochs 2 \
    --per_device_train_batch_size "$batch_size" \
    --gradient_accumulation_steps "$grad_accum" \
    --logging_steps 10 \
    --eval_steps 50 \
    --save_steps 100 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    > "$log_file" 2>&1
}

if ! run_stage2 2 8 "$STAGE2/train.log"; then
  if grep -Eiq 'out of memory|cuda.*memory|CUDA error' "$STAGE2/train.log"; then
    echo "Stage2 batch_size=2 failed with CUDA memory issue; retrying batch_size=1 grad_accum=16." | tee "$STAGE2/retry_reason.txt"
    rm -rf "$STAGE2/checkpoint-"*
    run_stage2 1 16 "$STAGE2/train_retry_b1.log"
    cp "$STAGE2/train_retry_b1.log" "$STAGE2/train.log"
  else
    exit 1
  fi
fi
python - <<'PY' > "$STAGE2/sample_outputs.jsonl"
import json
from itertools import islice
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base='models/Qwen2.5-7B'
adapter='outputs/stage2_aux_after_cpt1m_lora_qwen2_5_7b_v1'
eval_file='data/staged_1m_pruned_v2/stage2_aux_sft_eval.jsonl'
tok=AutoTokenizer.from_pretrained(base, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map={'':0}, trust_remote_code=True, attn_implementation='sdpa')
model=PeftModel.from_pretrained(model, adapter)
model.eval()
with open(eval_file, encoding='utf-8') as f:
    rows=[json.loads(line) for line in islice(f, 12)]
for row in rows:
    prompt=row['prompt'].rstrip()+'\n'
    inp=tok(prompt, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        out=model.generate(**inp, do_sample=False, max_new_tokens=80, pad_token_id=tok.eos_token_id)
    text=tok.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)
    if ';' in text: text=text[:text.index(';')+1]
    print(json.dumps({'id':row.get('id'), 'target':row.get('target'), 'prediction':text.strip()}, ensure_ascii=False))
PY

rm -f "$STAGE3_DATA" "$STAGE3_SUMMARY"
rm -rf "$STAGED/stage3_events"
mkdir -p "$STAGED/stage3_events"
xvfb-run -a -s "-screen 0 1024x768x24" python -u scripts/collect_qwen_rejection_sft.py \
  --script_dir scripts \
  --ag_repo repos/alphageometry \
  --problems_file repos/alphageometry/jgex_ag_231.txt \
  --defs_file repos/alphageometry/defs.txt \
  --rules_file repos/alphageometry/rules.txt \
  --out_file "$STAGE3_DATA" \
  --summary_file "$STAGE3_SUMMARY" \
  --events_dir "$STAGED/stage3_events" \
  --qwen_model models/Qwen2.5-7B \
  --adapter_path "$STAGE2" \
  --limit 100 \
  --max_rows 512 \
  --max_level 300 \
  --ddar_timeout 45 \
  --candidate_max_level 300 \
  --candidate_ddar_timeout 45 \
  --beam_size 4 \
  --search_depth 2 \
  --num_return_sequences 6 \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_p 0.95 \
  --keep_progress \
  --min_cache_gain 60 \
  --min_added_dependencies 1 \
  > "$STAGED/stage3_collect.log" 2>&1

STAGE3_ROWS=$(python - <<'PY'
import json
from pathlib import Path
p=Path('data/staged_1m_pruned_v2/stage3_rejection_summary.json')
if p.exists():
    print(json.loads(p.read_text()).get('rows', 0))
else:
    print(0)
PY
)
if [ "$STAGE3_ROWS" -ge 16 ]; then
  rm -rf "$STAGE3"
  mkdir -p "$STAGE3"
  export CUDA_VISIBLE_DEVICES=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python -u scripts/train_qwen_aux_lora.py \
    --model_name_or_path models/Qwen2.5-7B \
    --init_adapter_path "$STAGE2" \
    --train_file "$STAGE3_DATA" \
    --eval_file "$STAGED/stage2_aux_sft_eval.jsonl" \
    --output_dir "$STAGE3" \
    --loss_mode target \
    --max_length 1024 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 10 \
    --eval_steps 50 \
    --save_steps 100 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    > "$STAGE3/train.log" 2>&1
  echo "$STAGE3" > outputs/final_adapter_path.txt
else
  mkdir -p outputs
  echo "$STAGE2" > outputs/final_adapter_path.txt
fi
