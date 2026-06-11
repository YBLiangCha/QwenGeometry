"""Patch AG1 generation helpers so cache dtype follows model dtype."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--ag_dir", required=True)
  return parser.parse_args()


def main() -> int:
  args = parse_args()

  lm_path = Path(args.ag_dir) / "lm_inference.py"
  lm_text = lm_path.read_text(encoding="utf-8")
  desired_dtype_block = (
      "    self.n = imodel.num_heads\n"
      "    self.h = imodel.head_size\n"
      "    self.array_dtype = np.float32\n"
  )
  if desired_dtype_block not in lm_text or "task_config', None), 'dtype'" in lm_text:
    backup = lm_path.with_suffix(lm_path.suffix + ".qwen_dtype_patch.bak")
    if not backup.exists():
      backup.write_text(lm_text, encoding="utf-8")
    lm_text = re.sub(
        r"    self\.n = imodel\.num_heads\n"
        r"    self\.h = imodel\.head_size\n"
        r"(?:    self\.array_dtype = .*\n"
        r"(?:    if isinstance\(self\.array_dtype, str\):\n"
        r"      self\.array_dtype = getattr\(np, self\.array_dtype\)\n)*)*",
        desired_dtype_block,
        lm_text,
        count=1,
    )
    lm_text = lm_text.replace("dtype=np.bfloat16", "dtype=self.array_dtype")
    lm_path.write_text(lm_text, encoding="utf-8")
    print(f"patched: {lm_path}")
  else:
    print(f"already patched: {lm_path}")

  model_path = Path(args.ag_dir) / "models.py"
  model_text = model_path.read_text(encoding="utf-8")
  if (
      "dtype=jnp.bfloat16" in model_text
      or "dtype=self.dtype" in model_text
      or "dtype=self.task_config.dtype" in model_text
  ):
    backup = model_path.with_suffix(model_path.suffix + ".qwen_dtype_patch.bak")
    if not backup.exists():
      backup.write_text(model_text, encoding="utf-8")
    model_text = model_text.replace(
        "dtype=jnp.bfloat16", "dtype=jnp.float32"
    )
    model_text = model_text.replace("dtype=self.dtype", "dtype=jnp.float32")
    model_text = model_text.replace(
        "dtype=self.task_config.dtype", "dtype=jnp.float32"
    )
    model_path.write_text(model_text, encoding="utf-8")
    print(f"patched: {model_path}")
  else:
    print(f"already patched: {model_path}")

  numericals_path = Path(args.ag_dir) / "numericals.py"
  numericals_text = numericals_path.read_text(encoding="utf-8")
  if "(a, b, c), *ps = points" in numericals_text:
    backup = numericals_path.with_suffix(
        numericals_path.suffix + ".qwen_runtime_patch.bak"
    )
    if not backup.exists():
      backup.write_text(numericals_text, encoding="utf-8")
    numericals_text = numericals_text.replace(
        "(a, b, c), *ps = points", "a, b, c, *ps = points"
    )
    numericals_path.write_text(numericals_text, encoding="utf-8")
    print(f"patched: {numericals_path}")
  else:
    print(f"already patched: {numericals_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
