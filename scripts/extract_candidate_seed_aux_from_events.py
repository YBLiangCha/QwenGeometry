#!/usr/bin/env python3
"""Extract solved auxiliary candidates as replayable benchmark seed rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def safe_name(name: str) -> str:
  return re.sub(r'[^A-Za-z0-9_.-]+', '_', name or 'problem').strip('_') or 'problem'


def iter_events(events_dir: Path):
  for path in sorted(events_dir.glob('*.jsonl')):
    for line_no, line in enumerate(
        path.read_text(encoding='utf-8', errors='replace').splitlines(), 1
    ):
      if not line.strip():
        continue
      try:
        yield path, line_no, json.loads(line)
      except json.JSONDecodeError:
        continue


def parse_problem_names(value: str | None) -> set[str] | None:
  if not value:
    return None
  names = {part.strip() for part in value.split(',') if part.strip()}
  return names or None


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--events_dir', action='append', default=[])
  parser.add_argument(
      '--summary_jsonl',
      action='append',
      default=[],
      help='additional benchmark summaries with solved aux rows to convert to seeds',
  )
  parser.add_argument('--out_file', required=True)
  parser.add_argument('--problem_names')
  parser.add_argument('--max_per_problem', type=int, default=8)
  return parser.parse_args()


def constructive_aux_to_dsl(translation: str) -> str:
  """Best-effort inverse for common constructive aux clauses."""
  clause = str(translation or '').strip().rstrip(';')
  if ' = ' not in clause:
    return clause + (';' if not clause.endswith(';') else '')
  point, body = clause.split(' = ', 1)
  point = point.strip()
  predicates = []
  for part in body.split(','):
    toks = part.strip().split()
    if not toks:
      continue
    name, args = toks[0], toks[1:]
    if name == 'on_line' and len(args) >= 3:
      predicates.append(['C', args[0], args[1], args[2]])
    elif name == 'on_circle' and len(args) >= 3:
      predicates.append(['D', args[1], args[0], args[1], args[2]])
    elif name == 'on_circum' and len(args) >= 4:
      predicates.append(['O', args[0], args[1], args[2], args[3]])
    elif name == 'on_bline' and len(args) >= 3:
      predicates.append(['D', args[0], args[1], args[0], args[2]])
    elif name == 'on_tline' and len(args) >= 4:
      predicates.append(['T', args[0], args[1], args[2], args[3]])
    elif name == 'on_pline' and len(args) >= 4:
      predicates.append(['P', args[0], args[1], args[2], args[3]])
    elif name == 'eqdistance' and len(args) >= 4:
      predicates.append(['D', args[0], args[1], args[2], args[3]])
    else:
      raise ValueError(f'unsupported constructive aux part: {part.strip()}')
  if not predicates:
    raise ValueError(f'empty constructive aux: {translation}')
  chunks = [' '.join(pred) + f' {index:02d}' for index, pred in enumerate(predicates)]
  return f'{point} : ' + ' '.join(chunks) + ' ;'


def row_from_event(
    event: dict[str, Any], path: Path, line_no: int, events_dir: Path
) -> dict[str, Any] | None:
  problem = event.get('problem') or path.stem
  target = str(event.get('target') or '').strip()
  if not problem or not target:
    return None
  translation = (
      event.get('translation')
      or event.get('candidate_translation')
      or event.get('aux')
  )
  return {
      'id': f'{safe_name(problem)}::{safe_name(events_dir.name)}::{path.stem}::{line_no}',
      'problem': problem,
      'target': target,
      'translation': translation,
      'source': str(events_dir),
      'source_events_file': str(path),
      'source_line_no': line_no,
      'candidate_construction_type': event.get('candidate_construction_type'),
      'score': event.get('candidate_rerank_score') or event.get('lm_score') or 0.0,
  }


def row_from_summary(
    row: dict[str, Any], path: Path, line_no: int
) -> dict[str, Any] | None:
  problem = row.get('problem') or row.get('name')
  translation = row.get('aux') or row.get('aux_statement') or row.get('best_aux')
  if not problem or not translation:
    return None
  target = constructive_aux_to_dsl(str(translation))
  return {
      'id': f'{safe_name(problem)}::{safe_name(path.stem)}::{line_no}',
      'problem': problem,
      'target': target,
      'translation': translation,
      'source': str(path),
      'source_summary_jsonl': str(path),
      'source_line_no': line_no,
      'candidate_construction_type': None,
      'score': row.get('candidate_rerank_score') or row.get('score') or 0.0,
  }


def main() -> None:
  args = parse_args()
  selected = parse_problem_names(args.problem_names)
  max_per_problem = max(0, int(args.max_per_problem or 0))
  rows: list[dict[str, Any]] = []
  seen: set[tuple[str, str]] = set()
  per_problem: dict[str, int] = {}
  counts: dict[str, int] = {}
  for events_dir_text in args.events_dir:
    events_dir = Path(events_dir_text)
    if not events_dir.exists():
      raise FileNotFoundError(events_dir)
    for path, line_no, event in iter_events(events_dir):
      if event.get('kind') != 'candidate_sft_signal':
        continue
      if event.get('reason') != 'candidate_solved':
        continue
      row = row_from_event(event, path, line_no, events_dir)
      if not row:
        counts['missing_problem_or_target'] = counts.get('missing_problem_or_target', 0) + 1
        continue
      problem = row['problem']
      if selected is not None and problem not in selected:
        counts['problem_filtered'] = counts.get('problem_filtered', 0) + 1
        continue
      if max_per_problem > 0 and per_problem.get(problem, 0) >= max_per_problem:
        counts['max_per_problem_filtered'] = (
            counts.get('max_per_problem_filtered', 0) + 1
        )
        continue
      key = (problem, row['target'])
      if key in seen:
        counts['duplicate_problem_target'] = counts.get('duplicate_problem_target', 0) + 1
        continue
      seen.add(key)
      rows.append(row)
      per_problem[problem] = per_problem.get(problem, 0) + 1
      counts['candidate_solved'] = counts.get('candidate_solved', 0) + 1
  for summary_text in args.summary_jsonl:
    summary_path = Path(summary_text)
    if not summary_path.exists():
      raise FileNotFoundError(summary_path)
    for line_no, line in enumerate(
        summary_path.read_text(encoding='utf-8', errors='replace').splitlines(), 1
    ):
      if not line.strip():
        continue
      row_json = json.loads(line)
      if not (row_json.get('solved') or row_json.get('root_solved')):
        continue
      try:
        row = row_from_summary(row_json, summary_path, line_no)
      except ValueError:
        counts['unsupported_summary_aux'] = counts.get('unsupported_summary_aux', 0) + 1
        continue
      if not row:
        counts['missing_summary_problem_or_aux'] = (
            counts.get('missing_summary_problem_or_aux', 0) + 1
        )
        continue
      problem = row['problem']
      if selected is not None and problem not in selected:
        counts['problem_filtered'] = counts.get('problem_filtered', 0) + 1
        continue
      if max_per_problem > 0 and per_problem.get(problem, 0) >= max_per_problem:
        counts['max_per_problem_filtered'] = (
            counts.get('max_per_problem_filtered', 0) + 1
        )
        continue
      key = (problem, row['target'])
      if key in seen:
        counts['duplicate_problem_target'] = counts.get('duplicate_problem_target', 0) + 1
        continue
      seen.add(key)
      rows.append(row)
      per_problem[problem] = per_problem.get(problem, 0) + 1
      counts['summary_solved_aux'] = counts.get('summary_solved_aux', 0) + 1
  out_file = Path(args.out_file)
  out_file.parent.mkdir(parents=True, exist_ok=True)
  out_file.write_text(
      ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
      encoding='utf-8',
  )
  print(json.dumps({
      'out_file': str(out_file),
      'rows': len(rows),
      'per_problem': per_problem,
      'counts': counts,
  }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
