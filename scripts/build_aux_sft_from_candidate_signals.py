#!/usr/bin/env python3
"""Build auxiliary-construction SFT rows from benchmark candidate signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def safe_name(name: str) -> str:
  return re.sub(r'[^A-Za-z0-9_.-]+', '_', name or 'problem').strip('_')


def inferred_construction_type(event: dict[str, Any]) -> str | None:
  if event.get('candidate_construction_type'):
    return event.get('candidate_construction_type')
  try:
    import qwen_ag_search as qs  # pylint: disable=import-outside-toplevel

    translation = str(event.get('translation') or '').strip()
    if translation and not translation.startswith('ERROR:'):
      return qs.construction_type_key(translation)
    target = str(event.get('target') or '').strip()
    if target:
      return qs.construction_type_key(qs.dsl_to_constructive_candidate(target))
  except Exception:  # pylint: disable=broad-except
    return None
  return None


def split_for_problem(problem: str, eval_mod: int) -> str:
  if eval_mod <= 0:
    return 'train'
  return 'eval' if sum(problem.encode('utf-8')) % eval_mod == 0 else 'train'


def iter_events(events_dir: Path):
  for path in sorted(events_dir.glob('*.jsonl')):
    for line_no, line in enumerate(
        path.read_text(encoding='utf-8', errors='replace').splitlines(), 1
    ):
      line = line.lstrip('\ufeff')
      if not line.strip():
        continue
      try:
        event = json.loads(line)
      except json.JSONDecodeError:
        continue
      yield path, line_no, event


def iter_event_sources(events_dir: Path, extra_events_dirs: list[str]):
  yield None, events_dir
  for index, extra_events_dir in enumerate(extra_events_dirs):
    extra_path = Path(extra_events_dir)
    yield f'extra{index}:{safe_name(extra_path.name)}', extra_path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--events_dir', required=True)
  parser.add_argument(
      '--extra_events_dir',
      action='append',
      default=[],
      help='additional benchmark events directories to mine, e.g. solved scout runs',
  )
  parser.add_argument('--train_file', required=True)
  parser.add_argument('--eval_file', required=True)
  parser.add_argument('--summary_file', required=True)
  parser.add_argument(
      '--eval_mod',
      type=int,
      default=10,
      help='deterministic problem-level eval split modulus; <=0 disables eval',
  )
  parser.add_argument(
      '--include_progress',
      action=argparse.BooleanOptionalAction,
      default=True,
      help='include DDAR-progress positives in addition to solved candidates',
  )
  parser.add_argument(
      '--min_progress_delta',
      type=int,
      default=1,
      help='minimum candidate-root added-dependency delta for progress rows',
  )
  parser.add_argument(
      '--max_elapsed_sec',
      type=float,
      default=120.0,
      help='maximum elapsed seconds for progress rows; <=0 disables',
  )
  parser.add_argument(
      '--min_progress_efficiency',
      type=float,
      default=0.0,
      help='minimum progress_delta_dependencies / elapsed_sec for progress rows',
  )
  parser.add_argument(
      '--max_progress_rows_per_problem',
      type=int,
      default=0,
      help='cap DDAR-progress rows per problem after ranking by efficiency; 0 disables',
  )
  parser.add_argument(
      '--max_progress_rows_per_type',
      type=int,
      default=0,
      help='cap DDAR-progress rows per construction type after ranking; 0 disables',
  )
  parser.add_argument(
      '--solved_repeat',
      type=int,
      default=1,
      help='duplicate solved rows this many times before the train/eval split',
  )
  return parser.parse_args()


def numeric(value: Any, default: float = 0.0) -> float:
  if isinstance(value, (int, float)):
    return float(value)
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def keep_event(event: dict[str, Any], args: argparse.Namespace) -> bool:
  reason = event.get('reason')
  if reason == 'candidate_solved':
    return True
  if reason != 'ddar_progress_positive' or not args.include_progress:
    return False
  delta = numeric(event.get('progress_delta_dependencies'))
  elapsed = event.get('candidate_elapsed_sec')
  if delta < args.min_progress_delta:
    return False
  if args.max_elapsed_sec > 0:
    if not isinstance(elapsed, (int, float)) or elapsed > args.max_elapsed_sec:
      return False
  if args.min_progress_efficiency > 0:
    elapsed_sec = numeric(elapsed, default=0.0)
    if elapsed_sec <= 0 or delta / max(elapsed_sec, 1.0) < args.min_progress_efficiency:
      return False
  return True


def row_from_event(
    event: dict[str, Any], path: Path, line_no: int, source_label: str | None = None
) -> dict[str, Any]:
  problem = event.get('problem') or path.stem
  row_id = f'{safe_name(problem)}::{path.stem}::{line_no}'
  if source_label:
    row_id = f'{safe_name(problem)}::{source_label}::{path.stem}::{line_no}'
  prompt = str(event.get('prompt') or '').rstrip()
  target = str(event.get('target') or '').strip()
  return {
      'id': row_id,
      'source_problem': problem,
      'source_events': source_label or 'primary',
      'prompt': prompt,
      'target': target,
      'candidate_translation': event.get('translation'),
      'candidate_construction_type': inferred_construction_type(event),
      'candidate_source': event.get('candidate_source') or event.get('source') or 'lm',
      'candidate_rerank_score': event.get('candidate_rerank_score'),
      'verdict': event.get('reason'),
      'depth': event.get('depth'),
      'problem_after_aux': event.get('problem_after_aux'),
      'candidate_added_dependencies': event.get('candidate_added_dependencies'),
      'root_added_dependencies': event.get('root_added_dependencies'),
      'progress_delta_dependencies': event.get('progress_delta_dependencies'),
      'candidate_elapsed_sec': event.get('candidate_elapsed_sec'),
      'candidate_ddar_status': event.get('candidate_ddar_status'),
      'candidate_solved': event.get('candidate_solved'),
  }


def progress_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
  delta = numeric(row.get('progress_delta_dependencies'))
  elapsed = numeric(row.get('candidate_elapsed_sec'), default=0.0)
  efficiency = delta / max(elapsed, 1.0) if elapsed >= 0 else 0.0
  rerank_score = numeric(row.get('candidate_rerank_score'))
  # Rank efficient, high-delta progress first; slow fact growth is noisy.
  return (efficiency, delta, rerank_score, -elapsed)


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
  solved_rows = [row for row in rows if row.get('verdict') == 'candidate_solved']
  progress_rows = [row for row in rows if row.get('verdict') != 'candidate_solved']
  selected: list[dict[str, Any]] = []

  solved_repeat = max(1, args.solved_repeat)
  for row in solved_rows:
    for repeat_index in range(solved_repeat):
      repeated = dict(row)
      if repeat_index:
        repeated['id'] = f"{row['id']}::solved_repeat{repeat_index}"
      selected.append(repeated)

  progress_by_problem: dict[str, int] = {}
  progress_by_type: dict[str, int] = {}
  for row in sorted(progress_rows, key=progress_rank_key, reverse=True):
    problem = str(row.get('source_problem') or '')
    construction_type = str(row.get('candidate_construction_type') or 'unknown')
    if (
        args.max_progress_rows_per_problem > 0
        and progress_by_problem.get(problem, 0) >= args.max_progress_rows_per_problem
    ):
      continue
    if (
        args.max_progress_rows_per_type > 0
        and progress_by_type.get(construction_type, 0) >= args.max_progress_rows_per_type
    ):
      continue
    selected.append(row)
    progress_by_problem[problem] = progress_by_problem.get(problem, 0) + 1
    progress_by_type[construction_type] = progress_by_type.get(construction_type, 0) + 1
  return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
      encoding='utf-8',
  )


def main() -> None:
  args = parse_args()
  events_dir = Path(args.events_dir)
  extra_events_dirs = list(args.extra_events_dir or [])
  train_file = Path(args.train_file)
  eval_file = Path(args.eval_file)
  summary_file = Path(args.summary_file)
  rows = []
  seen: set[tuple[str, str]] = set()
  counts: dict[str, int] = {}
  for source_label, source_events_dir in iter_event_sources(events_dir, extra_events_dirs):
    if not source_events_dir.exists():
      counts[f'missing_events_dir:{source_label or "primary"}'] = (
          counts.get(f'missing_events_dir:{source_label or "primary"}', 0) + 1
      )
      continue
    source_key = source_label or 'primary'
    for path, line_no, event in iter_events(source_events_dir):
      if event.get('kind') != 'candidate_sft_signal':
        continue
      counts['signals_seen'] = counts.get('signals_seen', 0) + 1
      counts[f'signals_seen:{source_key}'] = counts.get(f'signals_seen:{source_key}', 0) + 1
      if not keep_event(event, args):
        counts['signals_skipped'] = counts.get('signals_skipped', 0) + 1
        counts[f'signals_skipped:{source_key}'] = (
            counts.get(f'signals_skipped:{source_key}', 0) + 1
        )
        continue
      row = row_from_event(event, path, line_no, source_label=source_label)
      if not row['prompt'] or not row['target']:
        counts['missing_prompt_or_target'] = counts.get('missing_prompt_or_target', 0) + 1
        continue
      key = (row['prompt'], row['target'])
      if key in seen:
        counts['duplicate_prompt_target'] = counts.get('duplicate_prompt_target', 0) + 1
        counts[f'duplicate_prompt_target:{source_key}'] = (
            counts.get(f'duplicate_prompt_target:{source_key}', 0) + 1
        )
        continue
      seen.add(key)
      rows.append(row)
      counts[f"kept_before_cap:{row['verdict']}"] = (
          counts.get(f"kept_before_cap:{row['verdict']}", 0) + 1
      )
      counts[f"kept_before_cap:{source_key}:{row['verdict']}"] = (
          counts.get(f"kept_before_cap:{source_key}:{row['verdict']}", 0) + 1
      )
  rows_before_cap = len(rows)
  rows = select_rows(rows, args)
  for row in rows:
    row['split'] = split_for_problem(row['source_problem'], args.eval_mod)
    counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
    counts[row['split']] = counts.get(row['split'], 0) + 1
  train_rows = [row for row in rows if row['split'] == 'train']
  eval_rows = [row for row in rows if row['split'] == 'eval']
  if not eval_rows and len(train_rows) > 5 and args.eval_mod > 0:
    eval_rows = train_rows[-max(1, len(train_rows) // 10):]
    train_rows = train_rows[:-len(eval_rows)]
  write_jsonl(train_file, train_rows)
  write_jsonl(eval_file, eval_rows)
  summary = {
      'events_dir': str(events_dir),
      'extra_events_dirs': extra_events_dirs,
      'train_file': str(train_file),
      'eval_file': str(eval_file),
      'rows': len(rows),
      'rows_before_cap': rows_before_cap,
      'train_rows': len(train_rows),
      'eval_rows': len(eval_rows),
      'counts': counts,
      'filters': {
          'include_progress': args.include_progress,
          'min_progress_delta': args.min_progress_delta,
          'max_elapsed_sec': args.max_elapsed_sec,
          'min_progress_efficiency': args.min_progress_efficiency,
          'max_progress_rows_per_problem': args.max_progress_rows_per_problem,
          'max_progress_rows_per_type': args.max_progress_rows_per_type,
          'solved_repeat': max(1, args.solved_repeat),
      },
  }
  summary_file.parent.mkdir(parents=True, exist_ok=True)
  summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
