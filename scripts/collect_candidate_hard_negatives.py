"""Collect invalid Qwen auxiliary candidates as hard-negative training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def classify_error(translation: str) -> str:
  text = translation.lower()
  if 'already exists' in text:
    return 'point_already_exists'
  if 'pointtoocloseerror' in text:
    return 'point_too_close'
  if 'pointtoofarerror' in text:
    return 'point_too_far'
  if 'invalidquadsolveerror' in text:
    return 'invalid_quad_solve'
  if 'invalid predicate' in text:
    return 'invalid_predicate'
  if 'does not exist' in text:
    return 'unknown_point'
  if 'empty candidate' in text:
    return 'empty_candidate'
  if 'error:' in text:
    return 'other_error'
  return 'not_error'


def read_summary(summary_jsonl: Path) -> dict[str, dict[str, Any]]:
  rows = {}
  if not summary_jsonl.exists():
    return rows
  for line in summary_jsonl.read_text(encoding='utf-8', errors='replace').splitlines():
    if not line.strip():
      continue
    row = json.loads(line)
    name = row.get('name') or row.get('problem')
    if name:
      rows[name] = row
  return rows


def iter_events(events_dir: Path):
  for path in sorted(events_dir.glob('*.jsonl')):
    problem = path.stem
    for line_no, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
      if not line.strip():
        continue
      try:
        event = json.loads(line)
      except json.JSONDecodeError:
        continue
      yield problem, path, line_no, event


def split_for_problem(problem: str, eval_mod: int) -> str:
  if eval_mod <= 0:
    return 'train'
  return 'eval' if sum(problem.encode('utf-8')) % eval_mod == 0 else 'train'


def construction_type_for_negative(translation: str) -> str:
  if translation.startswith('ERROR:'):
    return 'error'
  clause = translation.strip().rstrip(';')
  if ' = ' not in clause:
    return 'unknown'
  _, rhs = clause.split(' = ', 1)
  names = []
  for construction in rhs.split(','):
    toks = construction.strip().split()
    if toks:
      names.append(toks[0])
  return '+'.join(sorted(names)) if names else 'unknown'


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--events_dir', required=True)
  parser.add_argument('--summary_jsonl')
  parser.add_argument('--out_file', required=True)
  parser.add_argument('--include_valid_unsolved', action='store_true')
  parser.add_argument(
      '--eval_mod',
      type=int,
      default=10,
      help='deterministic problem-level eval split modulus; <=0 disables eval',
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  events_dir = Path(args.events_dir)
  summary = read_summary(Path(args.summary_jsonl)) if args.summary_jsonl else {}
  out_path = Path(args.out_file)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  counts: dict[str, int] = {}
  written = 0
  with out_path.open('w', encoding='utf-8') as out:
    for problem, path, line_no, event in iter_events(events_dir):
      if event.get('kind') != 'candidate':
        continue
      translation = event.get('translation') or ''
      reason = classify_error(translation)
      solved = bool(summary.get(problem, {}).get('solved'))
      is_negative = reason != 'not_error'
      if args.include_valid_unsolved and not solved and reason == 'not_error':
        reason = 'valid_but_unsolved'
        is_negative = True
      if not is_negative:
        continue
      row = {
          'problem': problem,
          'event_file': str(path),
          'line_no': line_no,
          'depth': event.get('depth'),
          'raw': event.get('raw'),
          'translation': translation,
          'source': event.get('source') or 'lm',
          'construction_type': construction_type_for_negative(translation),
          'label': 0,
          'reason': reason,
          'problem_solved': solved,
          'split': split_for_problem(problem, args.eval_mod),
      }
      out.write(json.dumps(row, ensure_ascii=False) + '\n')
      counts[reason] = counts.get(reason, 0) + 1
      written += 1
  print(json.dumps({'out_file': str(out_path), 'rows': written, 'counts': counts}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
