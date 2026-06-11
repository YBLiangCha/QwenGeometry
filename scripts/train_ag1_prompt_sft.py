"""Fine-tune the released AlphaGeometry-1 LM on prompt/target aux rows.

This is a small continued-training driver for the Meliad 150M checkpoint.  It
keeps the AG1 tokenizer and decoder stack, but trains with teacher forcing on
our `{S} ... {D} ... {F1}` prompt format.  The loss mask is target-only: prompt
tokens condition the model, while only auxiliary-construction target tokens
contribute to the loss.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


GIN_FILES = [
    "base_htrans.gin",
    "size/medium_150M.gin",
    "options/positions_t5.gin",
    "options/lr_cosine_decay.gin",
    "options/seq_1024_nocache.gin",
    "geometry_150M_generate.gin",
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_file", action="append", required=True)
  parser.add_argument("--eval_file", action="append", default=[])
  parser.add_argument("--workdir", required=True)
  parser.add_argument("--load_dir", required=True)
  parser.add_argument("--vocab_path", required=True)
  parser.add_argument("--meliad_path", required=True)
  parser.add_argument("--num_steps", type=int, default=1000)
  parser.add_argument("--batch_size", type=int, default=2)
  parser.add_argument("--sequence_length", type=int, default=1024)
  parser.add_argument("--max_train_rows", type=int, default=0)
  parser.add_argument("--max_eval_rows", type=int, default=512)
  parser.add_argument("--learning_rate_multiplier", type=float, default=0.03)
  parser.add_argument("--warmup_steps", type=int, default=50)
  parser.add_argument("--checkpoint_every_steps", type=int, default=250)
  parser.add_argument("--log_every_steps", type=int, default=10)
  parser.add_argument("--test_every_steps", type=int, default=100)
  parser.add_argument("--num_test_steps", type=int, default=10)
  parser.add_argument("--seed", type=int, default=1234)
  return parser.parse_args()


def read_rows(paths: list[str], limit: int = 0) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for path in paths:
    with Path(path).open(encoding="utf-8", errors="replace") as f:
      for line in f:
        if not line.strip():
          continue
        row = json.loads(line)
        if row.get("prompt") and row.get("target"):
          rows.append(row)
        if limit and len(rows) >= limit:
          return rows
  return rows


def normalize_target(target: str) -> str:
  target = str(target or "").strip()
  if target and not target.endswith(";"):
    target += " ;"
  return target


def build_examples(
    rows: list[dict[str, Any]],
    vocab: Any,
    sequence_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  examples: list[dict[str, Any]] = []
  stats = {
      "input_rows": len(rows),
      "kept": 0,
      "dropped_empty": 0,
      "dropped_too_long": 0,
      "max_tokens": 0,
      "max_prompt_tokens": 0,
      "max_target_tokens": 0,
  }
  for row in rows:
    prompt = str(row.get("prompt") or "").strip()
    target = normalize_target(row.get("target") or "")
    if not prompt or not target:
      stats["dropped_empty"] += 1
      continue
    prompt_text = prompt + " "
    full_text = prompt_text + target
    prompt_ids = list(vocab.encode(prompt_text))
    token_ids = list(vocab.encode(full_text))
    if len(token_ids) > sequence_length:
      stats["dropped_too_long"] += 1
      continue
    if len(token_ids) < 2 or len(prompt_ids) >= len(token_ids):
      stats["dropped_empty"] += 1
      continue
    loss_mask = np.zeros((sequence_length,), dtype=bool)
    start = max(0, len(prompt_ids) - 1)
    end = len(token_ids) - 1
    if end <= start:
      stats["dropped_empty"] += 1
      continue
    loss_mask[start:end] = True
    targets = np.zeros((sequence_length,), dtype=np.int32)
    targets[: len(token_ids)] = np.asarray(token_ids, dtype=np.int32)
    examples.append({
        "targets": targets,
        "loss_mask": loss_mask,
        "num_chars": len(target),
        "nonzero_tokens": int(loss_mask.sum()),
        "source_problem": row.get("source_problem") or row.get("problem"),
    })
    stats["kept"] += 1
    stats["max_tokens"] = max(stats["max_tokens"], len(token_ids))
    stats["max_prompt_tokens"] = max(stats["max_prompt_tokens"], len(prompt_ids))
    stats["max_target_tokens"] = max(
        stats["max_target_tokens"], len(token_ids) - len(prompt_ids)
    )
  return examples, stats


def batch_iterator(
    examples: list[dict[str, Any]],
    batch_size: int,
    seed: int,
) -> Iterable[dict[str, np.ndarray]]:
  rng = random.Random(seed)
  epoch = 0
  order = list(range(len(examples)))
  while True:
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
      batch_ids = order[start : start + batch_size]
      if len(batch_ids) < batch_size:
        batch_ids.extend(order[: batch_size - len(batch_ids)])
      batch = [examples[i] for i in batch_ids]
      yield {
          "targets": np.stack([x["targets"] for x in batch]).astype(np.int32),
          "loss_mask": np.stack([x["loss_mask"] for x in batch]).astype(bool),
          "start_of_sequence": np.ones((batch_size,), dtype=bool),
          "epoch": np.full((batch_size,), epoch, dtype=np.int32),
          "num_chars": np.asarray([x["num_chars"] for x in batch], dtype=np.int32),
          "nonzero_tokens": np.asarray(
              [x["nonzero_tokens"] for x in batch], dtype=np.int32
          ),
      }
    epoch += 1


def main() -> int:
  args = parse_args()
  sys.path.insert(0, str(Path(args.meliad_path).resolve()))

  import gin  # pylint: disable=import-error,import-outside-toplevel
  import jax  # pylint: disable=import-error,import-outside-toplevel
  import t5.data  # pylint: disable=import-error,import-outside-toplevel
  import decoder_stack as ag_decoder_stack  # pylint: disable=import-error,import-outside-toplevel
  import lm_inference  # pylint: disable=import-error,import-outside-toplevel
  import optimizer_config  # pylint: disable=import-error,import-outside-toplevel
  import training_loop  # pylint: disable=import-error,import-outside-toplevel
  from transformer import decoder_stack as base_decoder_stack  # pylint: disable=import-error,import-outside-toplevel
  from transformer import models as tx_models  # pylint: disable=import-error,import-outside-toplevel

  gin_paths = [
      str(Path(args.meliad_path).resolve() / "transformer/configs"),
      str(Path.cwd()),
  ]
  gin_params = [
      f"TransformerTaskConfig.batch_size={args.batch_size}",
      f"TransformerTaskConfig.sequence_length={args.sequence_length}",
      "Trainer.restore_state_variables=False",
      "Trainer.generate_every_steps=0",
      "Trainer.use_separate_metric_directories=False",
      "DecoderOnlyLanguageModel.output_token_losses=False",
  ]
  lm_inference.parse_gin_configuration(GIN_FILES, gin_params, gin_paths=gin_paths)

  vocab = t5.data.SentencePieceVocabulary(args.vocab_path)
  train_rows = read_rows(args.train_file, args.max_train_rows)
  eval_rows = read_rows(args.eval_file, args.max_eval_rows) if args.eval_file else []
  train_examples, train_stats = build_examples(
      train_rows, vocab, args.sequence_length
  )
  eval_examples, eval_stats = build_examples(eval_rows, vocab, args.sequence_length)
  if not train_examples:
    raise ValueError("no train examples survived filtering")
  if args.eval_file and not eval_examples:
    raise ValueError("no eval examples survived filtering")

  def model_definition(mode: str):
    return tx_models.DecoderOnlyLanguageModel(
        mode=mode,
        task_config=base_decoder_stack.TransformerTaskConfig(),
        decoder_factory=ag_decoder_stack.DecoderStackGenerate,
    )

  workdir = Path(args.workdir)
  workdir.mkdir(parents=True, exist_ok=True)
  metadata = {
      "train_files": args.train_file,
      "eval_files": args.eval_file,
      "load_dir": args.load_dir,
      "vocab_path": args.vocab_path,
      "num_steps": args.num_steps,
      "batch_size": args.batch_size,
      "sequence_length": args.sequence_length,
      "learning_rate_multiplier": args.learning_rate_multiplier,
      "warmup_steps": args.warmup_steps,
      "train_stats": train_stats,
      "eval_stats": eval_stats,
      "jax_backend": jax.default_backend(),
      "jax_devices": [str(d) for d in jax.devices()],
  }
  (workdir / "prompt_sft_metadata.json").write_text(
      json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  print(json.dumps(metadata, ensure_ascii=False, indent=2))

  trainer = training_loop.Trainer(
      model_definition=model_definition,
      get_training_dataset_iterator=lambda: batch_iterator(
          train_examples, args.batch_size, args.seed
      ),
      get_test_dataset_iterator=(
          (lambda: batch_iterator(eval_examples, args.batch_size, args.seed + 1))
          if eval_examples
          else None
      ),
      pretty_print_input_function=None,
      process_summaries_function=tx_models.process_summaries_function(vocab),
      load_dir=args.load_dir,
      workdir=str(workdir),
      num_steps=args.num_steps,
      status_every_steps=1,
      log_every_steps=args.log_every_steps,
      test_every_steps=args.test_every_steps if eval_examples else 0,
      num_test_steps=args.num_test_steps,
      generate_every_steps=0,
      print_input_every_steps=0,
      save_checkpoints=True,
      checkpoint_every_steps=args.checkpoint_every_steps,
      restore_checkpoints=True,
      restore_state_variables=False,
      optimizer_factory=optimizer_config.FlaxAdafactorConfig(),
      learning_rate_schedule=optimizer_config.lr_cosine_decay,
      max_scheduled_steps=args.num_steps,
      warmup_steps=args.warmup_steps,
      learning_rate_multiplier=args.learning_rate_multiplier,
      rng_key_names=("dropout", "sample"),
      replicate_mode=False,
  )
  trainer.train()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
