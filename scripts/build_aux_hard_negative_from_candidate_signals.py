#!/usr/bin/env python3
"""Build generator-side hard-negative rows from benchmark candidate signals."""

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


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--events_dir', required=True)
  parser.add_argument('--train_file', required=True)
  parser.add_argument('--eval_file', required=True)
  parser.add_argument('--summary_file', required=True)
  parser.add_argument(
      '--reasons',
      default='point_too_close,point_too_far,point_already_exists,duplicate_canonical',
      help='comma-separated hard-negative reasons to keep',
  )
  parser.add_argument(
      '--eval_mod',
      type=int,
      default=10,
      help='deterministic problem-level eval split modulus; <=0 disables eval',
  )
  return parser.parse_args()


def row_from_event(event: dict[str, Any], path: Path, line_no: int) -> dict[str, Any]:
  problem = event.get('problem') or path.stem
  prompt = str(event.get('prompt') or '').rstrip()
  target = str(event.get('target') or '').strip()
  return {
      'id': f'{safe_name(problem)}::{path.stem}::{line_no}',
      'source_problem': problem,
      'prompt': prompt,
      'target': target,
      'candidate_translation': event.get('translation'),
      'candidate_construction_type': inferred_construction_type(event),
      'candidate_source': event.get('candidate_source') or event.get('source') or 'lm',
      'verdict': event.get('reason'),
      'depth': event.get('depth'),
  }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
      encoding='utf-8',
  )


def main() -> None:
  args = parse_args()
  events_dir = Path(args.events_dir)
  train_file = Path(args.train_file)
  eval_file = Path(args.eval_file)
  summary_file = Path(args.summary_file)
  keep_reasons = {reason.strip() for reason in args.reasons.split(',') if reason.strip()}
  rows = []
  seen: set[tuple[str, str]] = set()
  counts: dict[str, int] = {}
  for path, line_no, event in iter_events(events_dir):
    kind = event.get('kind')
    if kind not in {'candidate_hard_negative_signal', 'candidate_filtered'}:
      continue
    counts['signals_seen'] = counts.get('signals_seen', 0) + 1
    reason = event.get('reason') or 'unknown'
    if keep_reasons and reason not in keep_reasons:
      counts['signals_skipped'] = counts.get('signals_skipped', 0) + 1
      continue
    row = row_from_event(event, path, line_no)
    if kind == 'candidate_filtered':
      row['verdict'] = reason
    if not row['prompt'] or not row['target']:
      counts['missing_prompt_or_target'] = counts.get('missing_prompt_or_target', 0) + 1
      continue
    key = (row['prompt'], row['target'])
    if key in seen:
      counts['duplicate_prompt_target'] = counts.get('duplicate_prompt_target', 0) + 1
      continue
    seen.add(key)
    row['split'] = split_for_problem(row['source_problem'], args.eval_mod)
    rows.append(row)
    counts[reason] = counts.get(reason, 0) + 1
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
      'train_file': str(train_file),
      'eval_file': str(eval_file),
      'rows': len(rows),
      'train_rows': len(train_rows),
      'eval_rows': len(eval_rows),
      'counts': counts,
      'reasons': sorted(keep_reasons),
  }
  summary_file.parent.mkdir(parents=True, exist_ok=True)
  summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
