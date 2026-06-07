"""Build candidate value/reranker training rows from Qwen AG event logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


def import_search(script_dir: str | None):
  if script_dir:
    sys.path.insert(0, str(Path(script_dir).resolve()))
  import qwen_ag_search as qs  # pylint: disable=import-error,import-outside-toplevel

  return qs


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
  if 'timeout' in text:
    return 'timeout'
  if translation.startswith('ERROR:'):
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


def safe_problem_from_path(path: Path) -> str:
  return re.sub(r'\.jsonl$', '', path.name)


def iter_events(events_dir: Path):
  for path in sorted(events_dir.glob('*.jsonl')):
    problem = safe_problem_from_path(path)
    for line_no, line in enumerate(
        path.read_text(encoding='utf-8', errors='replace').splitlines(), 1
    ):
      if not line.strip():
        continue
      try:
        event = json.loads(line)
      except json.JSONDecodeError:
        continue
      yield problem, path, line_no, event


def candidate_key(depth: Any, raw: str) -> tuple[int | None, str]:
  try:
    parsed_depth = int(depth)
  except (TypeError, ValueError):
    parsed_depth = None
  return parsed_depth, raw


def parse_ddar_candidate_tag(tag: str) -> tuple[int | None, str] | None:
  match = re.match(r'^depth([0-9]+):(.*)$', tag or '')
  if not match:
    return None
  return int(match.group(1)), match.group(2)


def read_event_file(path: Path):
  for line_no, line in enumerate(
      path.read_text(encoding='utf-8', errors='replace').splitlines(), 1
  ):
    if not line.strip():
      continue
    try:
      yield line_no, json.loads(line)
    except json.JSONDecodeError:
      continue


def split_for_problem(problem: str, eval_mod: int) -> str:
  if eval_mod <= 0:
    return 'train'
  return 'eval' if sum(problem.encode('utf-8')) % eval_mod == 0 else 'train'


def inferred_construction_type(qs: Any, raw: str, translation: str) -> str:
  if translation and not translation.startswith('ERROR:'):
    return qs.construction_type_key(translation)
  try:
    if raw:
      return qs.construction_type_key(qs.dsl_to_constructive_candidate(raw))
  except Exception:  # pylint: disable=broad-except
    pass
  return 'error'


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--events_dir', required=True)
  parser.add_argument('--summary_jsonl', required=True)
  parser.add_argument('--out_file', required=True)
  parser.add_argument('--script_dir', default='scripts')
  parser.add_argument(
      '--eval_mod',
      type=int,
      default=10,
      help='deterministic problem-level eval split modulus; <=0 disables eval',
  )
  parser.add_argument(
      '--progress_positive_min_added_dependencies',
      type=int,
      default=10,
      help='label valid candidates with at least this many DDAR additions as weak positives',
  )
  parser.add_argument(
      '--progress_positive_min_root_delta',
      type=int,
      default=1,
      help='require candidate DDAR additions to exceed root additions by this delta for weak positives',
  )
  parser.add_argument(
      '--progress_positive_max_elapsed_sec',
      type=float,
      default=120.0,
      help='do not label slow valid candidates as weak positives; <=0 disables',
  )
  parser.add_argument(
      '--include_unevaluated_valid',
      action='store_true',
      help='keep valid candidates without a DDAR result as negatives',
  )
  parser.add_argument(
      '--disable_progress_positives',
      action='store_true',
      help='only label exact solved auxiliaries as positives',
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  qs = import_search(args.script_dir)
  events_dir = Path(args.events_dir)
  summary = read_summary(Path(args.summary_jsonl))
  out_path = Path(args.out_file)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  counts: dict[str, int] = {}
  written = 0
  with out_path.open('w', encoding='utf-8') as out:
    for path in sorted(events_dir.glob('*.jsonl')):
      problem = safe_problem_from_path(path)
      candidates = []
      ddar_results: dict[tuple[int | None, str], dict[str, Any]] = {}
      ddar_errors: dict[tuple[int | None, str], dict[str, Any]] = {}
      root_ddar: dict[str, Any] = {}
      for line_no, event in read_event_file(path):
        kind = event.get('kind')
        if kind == 'candidate':
          candidates.append((line_no, event))
        elif kind == 'ddar_done':
          tag = event.get('tag') or ''
          if tag == 'root':
            root_ddar = event
          parsed = parse_ddar_candidate_tag(tag)
          if parsed is not None:
            ddar_results[parsed] = event
        elif kind == 'candidate_ddar_error':
          raw = event.get('raw') or ''
          key = candidate_key(event.get('depth'), raw)
          ddar_errors[key] = event
      for line_no, event in candidates:
        raw = event.get('raw') or ''
        translation = event.get('translation') or ''
        reason = classify_error(translation)
        row_summary = summary.get(problem, {})
        solved_aux = row_summary.get('aux') or ''
        solved = bool(row_summary.get('solved'))
        key = candidate_key(event.get('depth'), raw)
        ddar = ddar_results.get(key, {})
        ddar_error = ddar_errors.get(key, {})
        added_dependencies = int(ddar.get('added_dependencies') or 0)
        root_added_dependencies = int(root_ddar.get('added_dependencies') or 0)
        progress_delta = added_dependencies - root_added_dependencies
        candidate_solved = bool(ddar.get('solved'))
        candidate_elapsed = ddar.get('elapsed_sec')
        if (
            reason == 'not_error'
            and not ddar
            and not ddar_error
            and not args.include_unevaluated_valid
        ):
          continue
        label = 0
        if (
            reason == 'not_error'
            and solved
            and solved_aux
            and qs.canonical_aux_key(translation) == qs.canonical_aux_key(solved_aux)
        ):
          label = 1
          reason = 'solved_aux'
        elif reason == 'not_error' and candidate_solved:
          label = 1
          reason = 'candidate_solved'
        elif reason == 'not_error' and ddar_error:
          reason = 'candidate_ddar_error'
        elif (
            reason == 'not_error'
            and not args.disable_progress_positives
            and added_dependencies >= args.progress_positive_min_added_dependencies
            and progress_delta >= args.progress_positive_min_root_delta
            and (
                args.progress_positive_max_elapsed_sec <= 0
                or (
                    isinstance(candidate_elapsed, (int, float))
                    and candidate_elapsed <= args.progress_positive_max_elapsed_sec
                )
            )
        ):
          label = 1
          reason = 'ddar_progress_positive'
        elif reason == 'not_error' and solved:
          reason = 'valid_nonwinning'
        elif reason == 'not_error':
          reason = 'valid_but_unsolved'
        record = {
            'problem': problem,
            'event_file': str(path),
            'line_no': line_no,
            'depth': event.get('depth'),
            'raw': raw,
            'translation': translation,
            'source': event.get('source') or event.get('candidate_source') or 'lm',
            'construction_type': inferred_construction_type(qs, raw, translation),
            'label': label,
            'reason': reason,
            'problem_solved': solved,
            'candidate_solved': candidate_solved,
            'candidate_ddar_status': ddar.get('status'),
            'candidate_ddar_error': ddar_error.get('error'),
            'candidate_added_dependencies': added_dependencies,
            'root_added_dependencies': root_added_dependencies,
            'progress_delta_dependencies': progress_delta,
            'candidate_levels': ddar.get('levels'),
            'candidate_elapsed_sec': candidate_elapsed or ddar_error.get('elapsed_sec'),
            'split': split_for_problem(problem, args.eval_mod),
        }
        out.write(json.dumps(record, ensure_ascii=False) + '\n')
        counts[f'label_{label}'] = counts.get(f'label_{label}', 0) + 1
        counts[reason] = counts.get(reason, 0) + 1
        counts[record['split']] = counts.get(record['split'], 0) + 1
        written += 1
  print(json.dumps({'out_file': str(out_path), 'rows': written, 'counts': counts}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
