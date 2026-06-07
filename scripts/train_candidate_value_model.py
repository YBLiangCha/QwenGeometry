"""Train a lightweight JSON candidate value model from AG candidate rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
from typing import Any


def normalize_feature(value: Any) -> str:
  text = str(value or '').strip().lower()
  return re.sub(r'[^a-z0-9_^]+', '_', text).strip('_') or 'none'


def classify_error(text: str) -> str:
  lowered = str(text or '').lower()
  if 'already exists' in lowered:
    return 'point_already_exists'
  if 'pointtoocloseerror' in lowered:
    return 'point_too_close'
  if 'pointtoofarerror' in lowered:
    return 'point_too_far'
  if 'invalidquadsolveerror' in lowered:
    return 'invalid_quad_solve'
  if 'invalid predicate' in lowered:
    return 'invalid_predicate'
  if 'does not exist' in lowered:
    return 'unknown_point'
  if 'empty candidate' in lowered:
    return 'empty_candidate'
  if 'timeout' in lowered:
    return 'timeout'
  if lowered.startswith('error:') or 'error:' in lowered:
    return 'other_error'
  return 'not_error'


def add_prefixed_token(tokens: list[str], prefix: str, value: Any) -> None:
  normalized = normalize_feature(value)
  if normalized != 'none':
    tokens.append(f'{prefix}={normalized}')


def tokens_for_row(row: dict[str, Any]) -> list[str]:
  raw = str(row.get('raw') or '')
  translation = str(row.get('translation') or '')
  construction_type = str(row.get('construction_type') or '')
  text = ' '.join(
      str(value or '')
      for value in (
          raw,
          translation,
          construction_type,
          row.get('candidate_ddar_error'),
      )
  )
  tokens = [tok.lower() for tok in re.findall(r'[A-Za-z_^]+|[0-9]+', text)]
  for name in construction_type.split('+'):
    if name and name != 'unknown':
      tokens.append('type=' + name)
  error = classify_error(translation)
  if error == 'not_error':
    error = classify_error(str(row.get('candidate_ddar_error') or ''))
  add_prefixed_token(tokens, 'error', error)
  add_prefixed_token(tokens, 'reason', row.get('reason'))
  add_prefixed_token(tokens, 'source', row.get('source'))
  add_prefixed_token(tokens, 'ddar_status', row.get('candidate_ddar_status'))
  return tokens


def sigmoid(x: float) -> float:
  if x >= 0:
    z = math.exp(-x)
    return 1.0 / (1.0 + z)
  z = math.exp(x)
  return z / (1.0 + z)


def score(weights: dict[str, float], bias: float, tokens: list[str]) -> float:
  return bias + sum(weights.get(token, 0.0) for token in tokens)


def load_rows(path: Path) -> list[dict[str, Any]]:
  rows = []
  for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
    if line.strip():
      rows.append(json.loads(line))
  return rows


def metrics(rows: list[dict[str, Any]], weights: dict[str, float], bias: float):
  if not rows:
    return {'rows': 0}
  preds = []
  correct = 0
  loss = 0.0
  for row in rows:
    y = int(row.get('label', 0))
    p = sigmoid(score(weights, bias, tokens_for_row(row)))
    preds.append((p, y))
    correct += int((p >= 0.5) == bool(y))
    loss += -(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)))
  positives = [p for p, y in preds if y == 1]
  negatives = [p for p, y in preds if y == 0]
  auc = None
  if positives and negatives:
    wins = 0.0
    total = 0
    for pos in positives:
      for neg in negatives:
        wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
        total += 1
    auc = wins / total if total else None
  return {
      'rows': len(rows),
      'positives': len(positives),
      'negatives': len(negatives),
      'accuracy': correct / len(rows),
      'loss': loss / len(rows),
      'auc': auc,
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_file', required=True)
  parser.add_argument('--out_file', required=True)
  parser.add_argument('--epochs', type=int, default=40)
  parser.add_argument('--lr', type=float, default=0.05)
  parser.add_argument('--l2', type=float, default=1e-5)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--min_abs_weight', type=float, default=1e-6)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  rows = load_rows(Path(args.train_file))
  train_rows = [row for row in rows if row.get('split', 'train') == 'train']
  eval_rows = [row for row in rows if row.get('split') == 'eval']
  if not any(int(row.get('label', 0)) for row in train_rows):
    moved = [row for row in eval_rows if int(row.get('label', 0))]
    if moved:
      train_rows.extend(moved)
      eval_rows = [row for row in eval_rows if not int(row.get('label', 0))]
  positives = sum(int(row.get('label', 0)) for row in train_rows)
  negatives = len(train_rows) - positives
  if not train_rows:
    raise ValueError('no training rows')
  pos_weight = negatives / max(positives, 1)
  weights: dict[str, float] = {}
  bias = math.log((positives + 1.0) / (negatives + 1.0))
  rng = random.Random(args.seed)
  for _ in range(args.epochs):
    rng.shuffle(train_rows)
    for row in train_rows:
      toks = tokens_for_row(row)
      y = int(row.get('label', 0))
      weight = pos_weight if y == 1 else 1.0
      p = sigmoid(score(weights, bias, toks))
      grad = (p - y) * weight
      bias -= args.lr * grad
      unique_toks = set(toks)
      for tok in unique_toks:
        old = weights.get(tok, 0.0)
        weights[tok] = old - args.lr * (grad + args.l2 * old)
  weights = {
      token: value
      for token, value in weights.items()
      if abs(value) >= args.min_abs_weight
  }
  model = {
      'format': 'qwen_ag_candidate_value_v1',
      'weights': weights,
      'bias': bias,
      'train_file': args.train_file,
      'epochs': args.epochs,
      'lr': args.lr,
      'l2': args.l2,
      'label_counts': {
          'train_positive': positives,
          'train_negative': negatives,
          'all_positive': sum(int(row.get('label', 0)) for row in rows),
          'all_negative': len(rows) - sum(int(row.get('label', 0)) for row in rows),
      },
      'metrics': {
          'train': metrics(train_rows, weights, bias),
          'eval': metrics(eval_rows, weights, bias),
      },
      'warnings': [],
  }
  if positives < 20:
    model['warnings'].append(
        'fewer than 20 positive candidates; use this as a pipeline smoke model, not a robust reranker'
    )
  out_path = Path(args.out_file)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding='utf-8')
  print(json.dumps({k: model[k] for k in ['format', 'train_file', 'label_counts', 'metrics', 'warnings']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
