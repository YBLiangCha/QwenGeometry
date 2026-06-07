"""Evaluate a candidate value model with DDAR-budget-oriented top-k metrics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any

import train_candidate_value_model as train_value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
    if line.strip():
      rows.append(json.loads(line))
  return rows


def sigmoid(x: float) -> float:
  if x >= 0:
    z = math.exp(-x)
    return 1.0 / (1.0 + z)
  z = math.exp(x)
  return z / (1.0 + z)


def score_row(
    row: dict[str, Any],
    weights: dict[str, float],
    bias: float,
    include_posthoc_features: bool,
) -> float:
  tokens = train_value.tokens_for_row(
      row, include_posthoc_features=include_posthoc_features
  )
  return bias + sum(float(weights.get(token, 0.0)) for token in tokens)


def auc(scored_rows: list[dict[str, Any]]) -> float | None:
  positives = [row['score'] for row in scored_rows if int(row.get('label', 0))]
  negatives = [row['score'] for row in scored_rows if not int(row.get('label', 0))]
  if not positives or not negatives:
    return None
  wins = 0.0
  total = 0
  for pos in positives:
    for neg in negatives:
      wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
      total += 1
  return wins / total if total else None


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_file', required=True)
  parser.add_argument('--data_file', required=True)
  parser.add_argument('--out_file')
  parser.add_argument(
      '--group_by',
      default='problem,depth',
      help='comma-separated fields used as independent DDAR budget groups',
  )
  parser.add_argument(
      '--top_k',
      default='1,4,8,16,32',
      help='comma-separated top-k budgets to evaluate per group',
  )
  parser.add_argument(
      '--valid_only',
      action=argparse.BooleanOptionalAction,
      default=True,
      help='evaluate candidates whose translation is not an ERROR',
  )
  parser.add_argument(
      '--split',
      default='',
      help='optional split filter such as train or eval',
  )
  parser.add_argument(
      '--include_posthoc_features',
      action='store_true',
      help='include reason/DDAR-status features; off for online-rerank evaluation',
  )
  return parser.parse_args()


def is_valid_online_candidate(row: dict[str, Any]) -> bool:
  translation = str(row.get('translation') or '')
  return bool(translation.strip()) and not translation.startswith('ERROR:')


def group_key(row: dict[str, Any], fields: list[str]) -> str:
  return '|'.join(f'{field}={row.get(field)}' for field in fields)


def first_positive_rank(rows: list[dict[str, Any]]) -> int | None:
  for index, row in enumerate(rows, 1):
    if int(row.get('label', 0)):
      return index
  return None


def rank_summary(ranks: list[int]) -> dict[str, Any]:
  if not ranks:
    return {'count': 0}
  return {
      'count': len(ranks),
      'min': min(ranks),
      'median': statistics.median(ranks),
      'mean': sum(ranks) / len(ranks),
      'max': max(ranks),
  }


def main() -> None:
  args = parse_args()
  model_path = Path(args.model_file)
  data_path = Path(args.data_file)
  model = json.loads(model_path.read_text(encoding='utf-8'))
  weights = model.get('weights', {})
  bias = float(model.get('bias', 0.0))
  fields = [field.strip() for field in args.group_by.split(',') if field.strip()]
  top_ks = [int(value) for value in args.top_k.split(',') if value.strip()]

  input_rows = load_jsonl(data_path)
  rows = [
      row
      for row in input_rows
      if not args.valid_only or is_valid_online_candidate(row)
  ]
  if args.split:
    rows = [row for row in rows if str(row.get('split') or '') == args.split]
  scored_rows = []
  for input_index, row in enumerate(rows):
    scored = dict(row)
    scored['_input_index'] = input_index
    scored['score'] = score_row(
        row,
        weights,
        bias,
        include_posthoc_features=args.include_posthoc_features,
    )
    scored_rows.append(scored)

  grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for row in scored_rows:
    grouped[group_key(row, fields)].append(row)

  groups_with_positive = 0
  total_positives = 0
  first_ranks = []
  input_first_ranks = []
  top_stats = {
      k: {
          'groups_hit': 0,
          'positives_in_top_k': 0,
          'rows_in_top_k': 0,
      }
      for k in top_ks
  }
  missed_positive_types: dict[int, Counter] = {k: Counter() for k in top_ks}

  for group_rows in grouped.values():
    positives = sum(int(row.get('label', 0)) for row in group_rows)
    if positives:
      groups_with_positive += 1
      total_positives += positives
    ranked = sorted(
        group_rows,
        key=lambda row: (float(row['score']), -int(row['_input_index'])),
        reverse=True,
    )
    rank = first_positive_rank(ranked)
    input_rank = first_positive_rank(
        sorted(group_rows, key=lambda row: int(row['_input_index']))
    )
    if rank is not None:
      first_ranks.append(rank)
    if input_rank is not None:
      input_first_ranks.append(input_rank)
    for k in top_ks:
      top = ranked[:k]
      positives_in_top = sum(int(row.get('label', 0)) for row in top)
      top_stats[k]['positives_in_top_k'] += positives_in_top
      top_stats[k]['rows_in_top_k'] += len(top)
      if positives and positives_in_top:
        top_stats[k]['groups_hit'] += 1
      if positives and not positives_in_top:
        for row in group_rows:
          if int(row.get('label', 0)):
            missed_positive_types[k][row.get('construction_type') or 'unknown'] += 1

  positives = sum(int(row.get('label', 0)) for row in scored_rows)
  negatives = len(scored_rows) - positives
  payload = {
      'model_file': str(model_path),
      'data_file': str(data_path),
      'model_feature_policy': model.get('feature_policy'),
      'evaluation_feature_policy': (
          'posthoc_features' if args.include_posthoc_features else 'pre_ddar_features'
      ),
      'valid_only': args.valid_only,
      'split': args.split or None,
      'group_by': fields,
      'rows': len(scored_rows),
      'positives': positives,
      'negatives': negatives,
      'auc': auc(scored_rows),
      'groups': len(grouped),
      'groups_with_positive': groups_with_positive,
      'total_group_positives': total_positives,
      'first_positive_rank': rank_summary(first_ranks),
      'input_order_first_positive_rank': rank_summary(input_first_ranks),
      'top_k': {},
  }
  for k in top_ks:
    rows_in_top = top_stats[k]['rows_in_top_k']
    positives_in_top = top_stats[k]['positives_in_top_k']
    payload['top_k'][str(k)] = {
        'groups_hit': top_stats[k]['groups_hit'],
        'group_hit_rate': (
            top_stats[k]['groups_hit'] / groups_with_positive
            if groups_with_positive
            else None
        ),
        'positives_in_top_k': positives_in_top,
        'positive_recall': (
            positives_in_top / total_positives if total_positives else None
        ),
        'precision': positives_in_top / rows_in_top if rows_in_top else None,
        'rows_in_top_k': rows_in_top,
        'missed_positive_types_top': dict(missed_positive_types[k].most_common(10)),
    }
  text = json.dumps(payload, ensure_ascii=False, indent=2)
  if args.out_file:
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding='utf-8')
  print(text)


if __name__ == '__main__':
  main()
