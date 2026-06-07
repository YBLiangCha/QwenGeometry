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


def tokens_for_row(
    row: dict[str, Any], include_posthoc_features: bool = False
) -> list[str]:
  raw = str(row.get('raw') or '')
  translation = str(row.get('translation') or '')
  construction_type = str(row.get('construction_type') or '')
  text_parts = [raw, translation, construction_type]
  if include_posthoc_features:
    text_parts.append(str(row.get('candidate_ddar_error') or ''))
  text = ' '.join(
      str(value or '')
      for value in text_parts
  )
  tokens = [tok.lower() for tok in re.findall(r'[A-Za-z_^]+|[0-9]+', text)]
  for name in construction_type.split('+'):
    if name and name != 'unknown':
      tokens.append('type=' + name)
  add_prefixed_token(tokens, 'type_combo', construction_type)
  error = classify_error(translation)
  if include_posthoc_features and error == 'not_error':
    error = classify_error(str(row.get('candidate_ddar_error') or ''))
  add_prefixed_token(tokens, 'error', error)
  add_prefixed_token(tokens, 'source', row.get('source'))
  if include_posthoc_features:
    add_prefixed_token(tokens, 'reason', row.get('reason'))
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


def is_valid_online_candidate(row: dict[str, Any]) -> bool:
  translation = str(row.get('translation') or '')
  return bool(translation.strip()) and not translation.startswith('ERROR:')


def group_key(row: dict[str, Any], fields: list[str]) -> tuple[Any, ...]:
  return tuple(row.get(field) for field in fields)


def metrics(
    rows: list[dict[str, Any]],
    weights: dict[str, float],
    bias: float,
    include_posthoc_features: bool = False,
):
  if not rows:
    return {'rows': 0}
  preds = []
  correct = 0
  loss = 0.0
  for row in rows:
    y = int(row.get('label', 0))
    p = sigmoid(
        score(
            weights,
            bias,
            tokens_for_row(row, include_posthoc_features=include_posthoc_features),
        )
    )
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
  parser.add_argument(
      '--objective',
      choices=('logistic', 'pairwise'),
      default='logistic',
      help='logistic row classification or pairwise positive-vs-negative ranking',
  )
  parser.add_argument(
      '--pairwise_group_by',
      default='problem,depth',
      help='comma-separated row fields used to form pairwise ranking groups',
  )
  parser.add_argument(
      '--pairwise_negatives_per_positive',
      type=int,
      default=16,
      help='maximum sampled negatives per positive per epoch; <=0 uses all negatives',
  )
  parser.add_argument('--epochs', type=int, default=40)
  parser.add_argument('--lr', type=float, default=0.05)
  parser.add_argument('--l2', type=float, default=1e-5)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--min_abs_weight', type=float, default=1e-6)
  parser.add_argument(
      '--train_valid_only',
      action='store_true',
      help='train only candidates whose translation is valid, matching online rerank inputs',
  )
  parser.add_argument(
      '--include_posthoc_features',
      action='store_true',
      help=(
          'include DDAR/verdict features such as reason and candidate_ddar_status; '
          'off by default because online reranking happens before DDAR'
      ),
  )
  return parser.parse_args()


def train_logistic(
    train_rows: list[dict[str, Any]],
    weights: dict[str, float],
    bias: float,
    args: argparse.Namespace,
    rng: random.Random,
    positives: int,
    negatives: int,
) -> float:
  pos_weight = negatives / max(positives, 1)
  for _ in range(args.epochs):
    rng.shuffle(train_rows)
    for row in train_rows:
      toks = tokens_for_row(
          row, include_posthoc_features=args.include_posthoc_features
      )
      y = int(row.get('label', 0))
      weight = pos_weight if y == 1 else 1.0
      p = sigmoid(score(weights, bias, toks))
      grad = (p - y) * weight
      bias -= args.lr * grad
      unique_toks = set(toks)
      for tok in unique_toks:
        old = weights.get(tok, 0.0)
        weights[tok] = old - args.lr * (grad + args.l2 * old)
  return bias


def train_pairwise(
    train_rows: list[dict[str, Any]],
    weights: dict[str, float],
    bias: float,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, Any]:
  fields = [field.strip() for field in args.pairwise_group_by.split(',') if field.strip()]
  grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
  for row in train_rows:
    grouped.setdefault(group_key(row, fields), []).append(row)
  pair_groups = []
  total_pairs_per_full_pass = 0
  for rows in grouped.values():
    positives = [row for row in rows if int(row.get('label', 0))]
    negatives = [row for row in rows if not int(row.get('label', 0))]
    if positives and negatives:
      pair_groups.append((positives, negatives))
      total_pairs_per_full_pass += len(positives) * len(negatives)
  sampled_pairs = 0
  for _ in range(args.epochs):
    rng.shuffle(pair_groups)
    for positives, negatives in pair_groups:
      for pos in positives:
        if args.pairwise_negatives_per_positive <= 0:
          sampled_negatives = list(negatives)
        else:
          count = min(args.pairwise_negatives_per_positive, len(negatives))
          sampled_negatives = rng.sample(negatives, count)
        pos_toks = set(tokens_for_row(
            pos, include_posthoc_features=args.include_posthoc_features
        ))
        for neg in sampled_negatives:
          neg_toks = set(tokens_for_row(
              neg, include_posthoc_features=args.include_posthoc_features
          ))
          pos_score = score(weights, bias, list(pos_toks))
          diff = pos_score - score(weights, bias, list(neg_toks))
          grad = sigmoid(diff) - 1.0
          for tok in pos_toks:
            old = weights.get(tok, 0.0)
            weights[tok] = old - args.lr * (grad + args.l2 * old)
          for tok in neg_toks:
            old = weights.get(tok, 0.0)
            weights[tok] = old - args.lr * (-grad + args.l2 * old)
          sampled_pairs += 1
  return {
      'pairwise_group_by': fields,
      'pairwise_groups': len(pair_groups),
      'pairwise_full_pairs_per_epoch': total_pairs_per_full_pass,
      'pairwise_sampled_pairs': sampled_pairs,
      'pairwise_negatives_per_positive': args.pairwise_negatives_per_positive,
  }


def main() -> None:
  args = parse_args()
  rows = load_rows(Path(args.train_file))
  if args.train_valid_only:
    rows = [row for row in rows if is_valid_online_candidate(row)]
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
  weights: dict[str, float] = {}
  bias = (
      math.log((positives + 1.0) / (negatives + 1.0))
      if args.objective == 'logistic'
      else 0.0
  )
  rng = random.Random(args.seed)
  training_details: dict[str, Any] = {}
  if args.objective == 'logistic':
    bias = train_logistic(
        train_rows, weights, bias, args, rng, positives, negatives
    )
  else:
    training_details = train_pairwise(train_rows, weights, bias, args, rng)
  weights = {
      token: value
      for token, value in weights.items()
      if abs(value) >= args.min_abs_weight
  }
  model = {
      'format': 'qwen_ag_candidate_value_v1',
      'objective': args.objective,
      'feature_policy': (
          'posthoc_features'
          if args.include_posthoc_features
          else 'pre_ddar_features'
      ),
      'train_valid_only': args.train_valid_only,
      'training_details': training_details,
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
          'train': metrics(
              train_rows,
              weights,
              bias,
              include_posthoc_features=args.include_posthoc_features,
          ),
          'eval': metrics(
              eval_rows,
              weights,
              bias,
              include_posthoc_features=args.include_posthoc_features,
          ),
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
  print(json.dumps(
      {
          k: model[k]
          for k in [
              'format',
              'objective',
              'feature_policy',
              'train_valid_only',
              'training_details',
              'train_file',
              'label_counts',
              'metrics',
              'warnings',
          ]
      },
      ensure_ascii=False,
      indent=2,
  ))


if __name__ == '__main__':
  main()
