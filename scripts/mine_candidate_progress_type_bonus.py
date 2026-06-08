#!/usr/bin/env python3
"""Mine construction-type coverage bonuses from benchmark event logs."""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--events_dir', required=True)
  parser.add_argument('--out', required=True)
  parser.add_argument('--min_delta', type=float, default=50.0)
  parser.add_argument('--min_delta_floor', type=float, default=0.0)
  parser.add_argument('--min_root_ratio', type=float, default=0.0)
  parser.add_argument('--max_elapsed_sec', type=float, default=0.0)
  parser.add_argument('--min_efficiency', type=float, default=0.0)
  parser.add_argument('--topn', type=int, default=64)
  parser.add_argument('--per_problem_topn', type=int, default=12)
  parser.add_argument('--per_problem_bonus_uplift', type=float, default=0.0)
  parser.add_argument('--per_problem_max_bonus', type=float, default=0.0)
  parser.add_argument('--base_bonus', type=float, default=1.6)
  parser.add_argument('--delta_weight', type=float, default=0.9)
  parser.add_argument('--repeat_weight', type=float, default=0.15)
  parser.add_argument('--repeat_bonus_cap', type=float, default=0.4)
  parser.add_argument('--ratio_weight', type=float, default=0.0)
  parser.add_argument('--ratio_bonus_cap', type=float, default=0.0)
  parser.add_argument('--solved_bonus', type=float, default=0.8)
  parser.add_argument('--max_bonus', type=float, default=3.4)
  parser.add_argument(
      '--statuses',
      default='saturated,solved',
      help='comma-separated DDAR statuses to mine',
  )
  return parser.parse_args()


def read_jsonl(path: str):
  with open(path, encoding='utf-8') as f:
    for line in f:
      if not line.strip():
        continue
      try:
        yield json.loads(line)
      except json.JSONDecodeError:
        continue


def update_best(best: dict[str, Any], event: dict[str, Any], delta: float) -> None:
  elapsed = event.get('elapsed_sec')
  try:
    elapsed_value = float(elapsed) if elapsed is not None else None
  except (TypeError, ValueError):
    elapsed_value = None
  if delta > best.get('max_delta', float('-inf')):
    best['max_delta'] = delta
    best['best_added_dependencies'] = event.get('added_dependencies')
    best['best_elapsed_sec'] = elapsed_value
    best['best_status'] = event.get('status')
    best['best_tag'] = event.get('tag')
    best['best_problem'] = event.get('_problem')


def new_progress_item() -> dict[str, Any]:
  return {
      'count': 0,
      'solved_count': 0,
      'problems': set(),
      'max_delta': float('-inf'),
      'max_root_ratio': 0.0,
  }


def score_progress_item(
    args: argparse.Namespace,
    item: dict[str, Any],
    min_delta: float,
) -> tuple[float, float]:
  repeat_bonus = min(
      args.repeat_bonus_cap,
      args.repeat_weight * math.log1p(max(0, item['count'] - 1)),
  )
  ratio_bonus = min(
      args.ratio_bonus_cap,
      args.ratio_weight * math.log1p(max(0.0, item.get('max_root_ratio', 0.0))),
  )
  bonus = (
      args.base_bonus
      + args.delta_weight * math.log1p(max(0.0, item['max_delta']) / min_delta)
      + repeat_bonus
      + ratio_bonus
      + (args.solved_bonus if item['solved_count'] else 0.0)
  )
  if args.max_bonus > 0:
    bonus = min(args.max_bonus, bonus)
  return bonus, ratio_bonus


def mine(args: argparse.Namespace) -> dict[str, Any]:
  statuses = {x.strip() for x in args.statuses.split(',') if x.strip()}
  by_type: dict[str, dict[str, Any]] = {}
  by_problem_type: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
  problem_counts = collections.Counter()
  event_files = sorted(glob.glob(os.path.join(args.events_dir, '*.jsonl')))
  for path in event_files:
    problem = os.path.basename(path).removesuffix('.jsonl')
    root_added = None
    for event in read_jsonl(path):
      event['_problem'] = problem
      if event.get('kind') != 'ddar_done':
        continue
      if event.get('tag') == 'root':
        try:
          root_added = float(event.get('added_dependencies') or 0.0)
        except (TypeError, ValueError):
          root_added = 0.0
        continue
      if root_added is None:
        continue
      status = event.get('status')
      if status not in statuses:
        continue
      typ = event.get('candidate_construction_type')
      if not isinstance(typ, str) or not typ:
        continue
      try:
        added = float(event.get('added_dependencies') or 0.0)
      except (TypeError, ValueError):
        continue
      delta = added - root_added
      if delta <= 0:
        continue
      is_solved = status == 'solved'
      elapsed = event.get('elapsed_sec')
      try:
        elapsed_value = float(elapsed) if elapsed is not None else 0.0
      except (TypeError, ValueError):
        elapsed_value = 0.0
      if (
          args.max_elapsed_sec > 0
          and elapsed_value > args.max_elapsed_sec
          and not is_solved
      ):
        continue
      if (
          args.min_efficiency > 0
          and elapsed_value > 0
          and delta / elapsed_value < args.min_efficiency
          and not is_solved
      ):
        continue
      root_ratio = delta / root_added if root_added > 0 else 0.0
      ratio_pass = (
          args.min_root_ratio > 0
          and delta >= args.min_delta_floor
          and root_ratio >= args.min_root_ratio
      )
      if not is_solved and delta < args.min_delta and not ratio_pass:
        continue
      item = by_type.setdefault(typ, new_progress_item())
      problem_item = by_problem_type[problem].setdefault(typ, new_progress_item())
      for target in (item, problem_item):
        target['count'] += 1
        target['solved_count'] += int(is_solved)
        target['problems'].add(problem)
        target['max_root_ratio'] = max(target['max_root_ratio'], root_ratio)
        update_best(target, event, delta)
      problem_counts[problem] += 1

  rows = []
  min_delta = max(args.min_delta, 1.0)
  for typ, item in by_type.items():
    bonus, ratio_bonus = score_progress_item(args, item, min_delta)
    rows.append({
        'type': typ,
        'bonus': round(bonus, 4),
        'count': item['count'],
        'solved_count': item['solved_count'],
        'problem_count': len(item['problems']),
        'problems': sorted(item['problems']),
        'max_delta': item['max_delta'],
        'max_root_ratio': round(item.get('max_root_ratio', 0.0), 4),
        'ratio_bonus': round(ratio_bonus, 4),
        'best_added_dependencies': item.get('best_added_dependencies'),
        'best_elapsed_sec': item.get('best_elapsed_sec'),
        'best_status': item.get('best_status'),
        'best_tag': item.get('best_tag'),
        'best_problem': item.get('best_problem'),
    })
  rows.sort(
      key=lambda item: (
          -item['bonus'],
          -item['max_delta'],
          -item['solved_count'],
          item['type'],
      )
  )
  if args.topn > 0:
    rows = rows[: args.topn]

  per_problem_types: dict[str, list[dict[str, Any]]] = {}
  for problem, type_items in by_problem_type.items():
    problem_rows = []
    for typ, item in type_items.items():
      bonus, ratio_bonus = score_progress_item(args, item, min_delta)
      if args.per_problem_bonus_uplift > 0:
        bonus += args.per_problem_bonus_uplift
      per_problem_max = args.per_problem_max_bonus or args.max_bonus
      if per_problem_max > 0:
        bonus = min(per_problem_max, bonus)
      problem_rows.append({
          'type': typ,
          'bonus': round(bonus, 4),
          'count': item['count'],
          'solved_count': item['solved_count'],
          'max_delta': item['max_delta'],
          'max_root_ratio': round(item.get('max_root_ratio', 0.0), 4),
          'ratio_bonus': round(ratio_bonus, 4),
          'best_added_dependencies': item.get('best_added_dependencies'),
          'best_elapsed_sec': item.get('best_elapsed_sec'),
          'best_status': item.get('best_status'),
          'best_tag': item.get('best_tag'),
      })
    problem_rows.sort(
        key=lambda item: (
            -item['bonus'],
            -item['max_delta'],
            -item['solved_count'],
            item['type'],
        )
    )
    if args.per_problem_topn > 0:
      problem_rows = problem_rows[: args.per_problem_topn]
    per_problem_types[problem] = problem_rows
  return {
      'type_bonus': {row['type']: row['bonus'] for row in rows},
      'per_problem_type_bonus': {
          problem: {row['type']: row['bonus'] for row in problem_rows}
          for problem, problem_rows in sorted(per_problem_types.items())
      },
      'types': rows,
      'per_problem_types': per_problem_types,
      'metadata': {
          'events_dir': args.events_dir,
          'event_files': len(event_files),
          'min_delta': args.min_delta,
          'min_delta_floor': args.min_delta_floor,
          'min_root_ratio': args.min_root_ratio,
          'max_elapsed_sec': args.max_elapsed_sec,
          'min_efficiency': args.min_efficiency,
          'topn': args.topn,
          'per_problem_topn': args.per_problem_topn,
          'per_problem_bonus_uplift': args.per_problem_bonus_uplift,
          'per_problem_max_bonus': args.per_problem_max_bonus,
          'base_bonus': args.base_bonus,
          'delta_weight': args.delta_weight,
          'repeat_weight': args.repeat_weight,
          'repeat_bonus_cap': args.repeat_bonus_cap,
          'ratio_weight': args.ratio_weight,
          'ratio_bonus_cap': args.ratio_bonus_cap,
          'solved_bonus': args.solved_bonus,
          'max_bonus': args.max_bonus,
          'statuses': sorted(statuses),
          'problem_signal_counts': dict(problem_counts),
      },
  }


def main() -> None:
  args = parse_args()
  result = mine(args)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
  print(json.dumps({
      'out': str(out),
      'type_count': len(result['type_bonus']),
      'top': result['types'][:10],
  }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
  main()
