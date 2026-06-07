#!/usr/bin/env python3
"""LoRA training for Qwen on AG proof text and auxiliary constructions.

Auxiliary SFT records are JSONL. Minimal schema:
  {"prompt": "... {F1} x00", "target": "e : C a c e 00 C b d e 01 ;"}

By default the loss is masked on the prompt tokens and applied only to target
tokens. This matches the AG1 LM role: given the current formal state, generate
the next auxiliary construction rather than a natural-language proof.

For CPT-style geometry/proof-distribution learning, pass ``--loss_mode full``
and provide JSONL rows with a ``text`` field. In that mode the loss is applied
to the whole serialized sequence.

For the main AG-compatible path, targets should use the original constrained
LM format (`point : predicate ... ;`). The search script translates that text
to constructive AG clauses before inserting it into Graph. Direct constructive
targets (`point = on_line ...;`) are accepted by the search script for smoke
tests and ablations, but should not be the default training format.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def read_jsonl(
    path: str | Path, loss_mode: str, allow_empty: bool = False
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if loss_mode == "target" and (
                "prompt" not in obj or "target" not in obj
            ):
                raise ValueError(f"{path}:{lineno} needs prompt and target fields")
            if loss_mode == "full" and "text" not in obj:
                raise ValueError(f"{path}:{lineno} needs a text field")
            rows.append(obj)
    if not rows and not allow_empty:
        raise ValueError(f"{path} has no JSONL records")
    return rows


def normalize_target(target: str, eos: str) -> str:
    target = target.strip()
    if not target.endswith(";"):
        target += ";"
    return target + eos


def build_dataset(
    path: str | Path,
    tokenizer,
    max_length: int,
    loss_mode: str,
    negative_file: str | Path | None = None,
) -> Dataset:
    if negative_file and loss_mode != "target":
        raise ValueError("--negative_*_file is only supported with --loss_mode target")
    rows = []
    for row in read_jsonl(path, loss_mode):
        row = dict(row)
        row["_loss_type"] = "positive"
        rows.append(row)
    if negative_file:
        for row in read_jsonl(negative_file, loss_mode, allow_empty=True):
            row = dict(row)
            row["_loss_type"] = "negative"
            rows.append(row)

    def encode(row: dict[str, Any]) -> dict[str, Any]:
        if loss_mode == "full":
            text = row["text"].strip() + (tokenizer.eos_token or "")
            input_ids = tokenizer(text, add_special_tokens=False).input_ids
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": list(input_ids),
                "loss_type": 1,
            }

        prompt = row["prompt"].rstrip() + "\n"
        target = normalize_target(row["target"], tokenizer.eos_token or "")
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        # Keep at least a small target suffix if a sample is too long.
        if len(prompt_ids) + len(target_ids) > max_length:
            keep_prompt = max_length - len(target_ids)
            if keep_prompt < 16:
                target_ids = target_ids[-(max_length - 16):]
                keep_prompt = 16
            prompt_ids = prompt_ids[-keep_prompt:]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "loss_type": 0 if row.get("_loss_type") == "negative" else 1,
        }

    return Dataset.from_list(rows).map(encode, remove_columns=list(rows[0].keys()))


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "loss_type": []}
        for feat in features:
            pad = max_len - len(feat["input_ids"])
            batch["input_ids"].append(feat["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(feat["attention_mask"] + [0] * pad)
            batch["labels"].append(feat["labels"] + [-100] * pad)
            batch["loss_type"].append(int(feat.get("loss_type", 1)))
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


class TargetUnlikelihoodTrainer(Trainer):
    def __init__(self, *args, unlikelihood_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.unlikelihood_weight = float(unlikelihood_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_type = inputs.pop("loss_type", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels.ne(-100)
        if loss_type is None:
            positive_rows = torch.ones(
                labels.shape[0], dtype=torch.bool, device=labels.device
            )
        else:
            positive_rows = loss_type.to(labels.device).eq(1)
        positive_mask = valid_mask & positive_rows[:, None]
        if positive_mask.any():
            ce_loss = F.cross_entropy(
                shift_logits[positive_mask], shift_labels[positive_mask]
            )
        else:
            ce_loss = shift_logits.sum() * 0.0

        negative_mask = valid_mask & (~positive_rows)[:, None]
        if self.unlikelihood_weight > 0 and negative_mask.any():
            neg_logits = shift_logits[negative_mask].float()
            neg_labels = shift_labels[negative_mask]
            neg_probs = F.softmax(neg_logits, dim=-1).gather(
                -1, neg_labels.unsqueeze(-1)
            ).squeeze(-1)
            unlikelihood_loss = -torch.log(
                torch.clamp(1.0 - neg_probs, min=1e-6)
            ).mean()
            loss = ce_loss + self.unlikelihood_weight * unlikelihood_loss
        else:
            loss = ce_loss
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B")
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--loss_mode", choices=["target", "full"], default="target")
    p.add_argument("--negative_train_file")
    p.add_argument("--negative_eval_file")
    p.add_argument("--unlikelihood_weight", type=float, default=0.1)
    p.add_argument(
        "--init_adapter_path",
        help="Optional LoRA adapter to continue training from for staged runs.",
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default=",".join(DEFAULT_TARGET_MODULES))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset(
        args.train_file,
        tokenizer,
        args.max_length,
        args.loss_mode,
        args.negative_train_file,
    )
    eval_ds = (
        build_dataset(
            args.eval_file,
            tokenizer,
            args.max_length,
            args.loss_mode,
            args.negative_eval_file,
        )
        if args.eval_file
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if args.init_adapter_path:
        model = PeftModel.from_pretrained(
            model, args.init_adapter_path, is_trainable=True
        )
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                x.strip() for x in args.target_modules.split(",") if x.strip()
            ],
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_total_limit=3,
        gradient_checkpointing=True,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = TargetUnlikelihoodTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalCollator(tokenizer.pad_token_id),
        unlikelihood_weight=args.unlikelihood_weight,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
