#!/usr/bin/env python3
"""Summarize Qwen+DDAR benchmark event logs for failure analysis."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
from typing import Any


def classify_error(text: str) -> str:
  lowered = str(text or '').lower()
  if 'already exists' in lowered:
    return 'point_already_exists'
  if 'pointtoocloseerror' in lowered:
    return 'point_too_close'
  if 'pointtoofarerror' in lowered:
    return 'point_too_far'
  if 'depcheckfailerror' in lowered:
    return 'dep_check_fail'
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


def construction_type(translation: str) -> str:
  if not translation or translation.startswith('ERROR:'):
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


def top_counts(counter: Counter, limit: int = 12) -> dict[str, int]:
  return dict(counter.most_common(limit))


def normalize_candidate_text(text: str) -> str:
  text = str(text or '').strip()
  while text.endswith(';'):
    text = text[:-1].strip()
  return ' '.join(text.split())


def parse_depth_tag(tag: str) -> tuple[int | None, str]:
  match = re.match(r'^depth(\d+):(.*)$', str(tag or ''))
  if not match:
    return None, ''
  return int(match.group(1)), normalize_candidate_text(match.group(2))


def event_candidate_source(event: dict[str, Any]) -> str:
  return (
      event.get('source')
      or event.get('candidate_source')
      or event.get('prompt_source')
      or 'lm'
  )


def raw_candidate_construction_type(raw: str | None) -> str:
  raw = str(raw or '').strip()
  if not raw:
    return 'error'
  try:
    import qwen_ag_search as qs  # pylint: disable=import-outside-toplevel

    return qs.construction_type_key(qs.dsl_to_constructive_candidate(raw))
  except Exception:  # pylint: disable=broad-except
    return 'error'


def candidate_lookup_keys(
    depth: Any, raw: str | None, translation: str | None
) -> list[tuple[int, str]]:
  if not isinstance(depth, int):
    return []
  keys = []
  for text in (raw, translation):
    normalized = normalize_candidate_text(text or '')
    if normalized:
      keys.append((depth, normalized))
  return keys


def candidate_record(
    event: dict[str, Any], lookup: dict[tuple[int, str], dict[str, str]]
) -> dict[str, str]:
  translation = event.get('translation') or ''
  event_ctype = event.get('candidate_construction_type')
  event_source = event_candidate_source(event)
  if translation and classify_error(translation) == 'not_error':
    return {
        'translation': translation,
        'construction_type': event_ctype or construction_type(translation),
        'source': event_source,
    }
  raw = event.get('raw') or event.get('target')
  for key in candidate_lookup_keys(event.get('depth'), raw, translation):
    if key in lookup:
      return lookup[key]
  return {
      'translation': translation,
      'construction_type': (
          event_ctype or raw_candidate_construction_type(raw) or construction_type(translation)
      ),
      'source': event_source,
  }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  if not path.exists():
    return rows
  for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
    line = line.lstrip('\ufeff')
    if not line.strip():
      continue
    try:
      rows.append(json.loads(line))
    except json.JSONDecodeError:
      continue
  return rows


def pct(num: int, den: int) -> float:
  return round(num / den, 4) if den else 0.0


def num_summary(values: list[float]) -> dict[str, Any]:
  if not values:
    return {'count': 0}
  values = [float(v) for v in values]
  return {
      'count': len(values),
      'min': round(min(values), 3),
      'median': round(statistics.median(values), 3),
      'mean': round(sum(values) / len(values), 3),
      'max': round(max(values), 3),
  }


def load_summary(summary_jsonl: Path | None) -> dict[str, dict[str, Any]]:
  if not summary_jsonl:
    return {}
  rows = {}
  for row in read_jsonl(summary_jsonl):
    name = row.get('name') or row.get('problem')
    if name:
      rows[name] = row
  return rows


def problem_from_event_file(path: Path, summary: dict[str, dict[str, Any]]) -> str:
  stem = path.stem
  if stem in summary:
    return stem
  # Event filenames are safe_name(problem).  Most current AG names are unchanged,
  # but keep a loose fallback for future names that need sanitizing.
  for name in summary:
    safe = ''.join(ch if ch.isalnum() or ch in '_.-' else '_' for ch in name).strip('_')
    if safe == stem:
      return name
  return stem


def diagnose(stats: dict[str, Any]) -> list[str]:
  if stats.get('solved'):
    return ['solved']
  labels = []
  candidates = stats.get('candidates', 0)
  valid = stats.get('valid_candidates', 0)
  invalid = stats.get('invalid_candidates', 0)
  duplicate = stats.get('filtered_reasons', {}).get('duplicate_canonical', 0)
  candidate_ddar = stats.get('candidate_ddar_done', 0)
  timeout_errors = sum(
      count
      for key, count in stats.get('candidate_ddar_error_types', {}).items()
      if 'timeout' in str(key).lower()
  )
  if candidates == 0:
    labels.append('no_candidates_generated')
  if candidates and invalid / candidates >= 0.25:
    labels.append('high_invalid_rate')
  if candidates and duplicate / candidates >= 0.35:
    labels.append('duplicate_collapse')
  if candidate_ddar == 0 and timeout_errors:
    labels.append('candidate_ddar_timeout_blocked')
  elif timeout_errors:
    labels.append('candidate_ddar_timeouts')
  if candidate_ddar and stats.get('max_candidate_added_dependencies', 0) >= 100:
    labels.append('symbolic_progress_but_wrong_direction')
  if valid and not candidate_ddar:
    labels.append('valid_candidates_not_evaluated')
  if stats.get('event_counts', {}).get('candidate_backfill_exhausted', 0):
    labels.append('template_backfill_exhausted')
  if not labels:
    labels.append('search_exhausted_no_goal')
  return labels


def analyze_problem(
    path: Path, summary: dict[str, dict[str, Any]]
) -> dict[str, Any]:
  problem = problem_from_event_file(path, summary)
  summary_row = summary.get(problem, {})
  kind_counts = Counter()
  invalid_reasons = Counter()
  filtered_reasons = Counter()
  candidate_ddar_errors = Counter()
  candidate_statuses = Counter()
  construction_types = Counter()
  candidate_sources = Counter()
  valid_candidate_sources = Counter()
  filtered_construction_types = Counter()
  filtered_sources = Counter()
  filtered_types_by_reason: dict[str, Counter] = {}
  evaluated_construction_types = Counter()
  evaluated_sources = Counter()
  candidate_error_construction_types = Counter()
  timeout_construction_types = Counter()
  candidate_elapsed = []
  candidate_added = []
  max_added_record: dict[str, Any] | None = None
  beam_empty_depths = []
  candidate_sft_signals = Counter()
  candidate_sft_signal_types = Counter()
  candidate_hard_negative_signals = Counter()
  candidate_hard_negative_signal_types = Counter()
  candidate_hard_negative_signal_sources = Counter()
  root_ddar: dict[str, Any] = {}
  lookup: dict[tuple[int, str], dict[str, str]] = {}

  for event in read_jsonl(path):
    kind = event.get('kind') or 'unknown'
    kind_counts[kind] += 1
    if kind == 'candidate':
      translation = event.get('translation') or ''
      source = event_candidate_source(event)
      candidate_sources[source] += 1
      reason = classify_error(translation)
      if reason == 'not_error':
        ctype = construction_type(translation)
        construction_types[ctype] += 1
        valid_candidate_sources[source] += 1
        record = {
            'translation': translation,
            'construction_type': ctype,
            'source': source,
        }
        for key in candidate_lookup_keys(
            event.get('depth'), event.get('raw'), translation
        ):
          lookup[key] = record
      else:
        invalid_reasons[reason] += 1
    elif kind == 'candidate_filtered':
      reason = event.get('reason') or 'unknown'
      filtered_reasons[reason] += 1
      record = candidate_record(event, lookup)
      ctype = record['construction_type']
      filtered_construction_types[ctype] += 1
      filtered_sources[record['source']] += 1
      filtered_types_by_reason.setdefault(reason, Counter())[ctype] += 1
    elif kind == 'candidate_ddar_error':
      candidate_ddar_errors[event.get('error') or 'unknown'] += 1
      record = candidate_record(event, lookup)
      ctype = record['construction_type']
      candidate_error_construction_types[ctype] += 1
      error_type = classify_error(event.get('error') or '')
      if error_type == 'timeout':
        timeout_construction_types[ctype] += 1
    elif kind == 'ddar_done':
      tag = event.get('tag') or ''
      if tag == 'root':
        root_ddar = event
      elif tag.startswith('depth'):
        depth_from_tag, tag_candidate = parse_depth_tag(tag)
        record = {}
        if depth_from_tag is not None and tag_candidate:
          record = lookup.get((depth_from_tag, tag_candidate), {})
        if not record and tag_candidate:
          record = {
              'translation': tag_candidate,
              'construction_type': construction_type(tag_candidate),
              'source': 'unknown',
          }
        ctype = record.get('construction_type') or 'unknown'
        evaluated_construction_types[ctype] += 1
        evaluated_sources[record.get('source') or 'unknown'] += 1
        candidate_statuses[event.get('status') or 'unknown'] += 1
        elapsed = event.get('elapsed_sec')
        if isinstance(elapsed, (int, float)):
          candidate_elapsed.append(float(elapsed))
        added = int(event.get('added_dependencies') or 0)
        candidate_added.append(added)
        if max_added_record is None or added > int(
            max_added_record.get('added_dependencies') or 0
        ):
          max_added_record = {
              'tag': tag,
              'status': event.get('status'),
              'solved': event.get('solved'),
              'added_dependencies': added,
              'elapsed_sec': elapsed,
              'levels': event.get('levels'),
              'construction_type': ctype,
              'translation': record.get('translation') or tag_candidate,
              'source': record.get('source') or 'unknown',
          }
    elif kind == 'beam_empty':
      beam_empty_depths.append(event.get('depth'))
    elif kind == 'candidate_sft_signal':
      candidate_sft_signals[event.get('reason') or 'unknown'] += 1
      record = candidate_record(event, lookup)
      candidate_sft_signal_types[record['construction_type']] += 1
    elif kind == 'candidate_hard_negative_signal':
      candidate_hard_negative_signals[event.get('reason') or 'unknown'] += 1
      record = candidate_record(event, lookup)
      candidate_hard_negative_signal_types[record['construction_type']] += 1
      candidate_hard_negative_signal_sources[record['source']] += 1

  candidates = kind_counts.get('candidate', 0)
  invalid = sum(invalid_reasons.values())
  valid = candidates - invalid
  candidate_ddar_done = sum(candidate_statuses.values())
  result = {
      'problem': problem,
      'completed': bool(summary_row),
      'solved': bool(summary_row.get('solved')),
      'root_solved': bool(summary_row.get('root_solved')),
      'solved_depth': summary_row.get('solved_depth'),
      'aux': summary_row.get('aux'),
      'solved_aux_construction_type': (
          construction_type(summary_row.get('aux') or '')
          if summary_row.get('aux')
          else None
      ),
      'event_file': str(path),
      'event_counts': dict(kind_counts),
      'candidates': candidates,
      'valid_candidates': valid,
      'invalid_candidates': invalid,
      'invalid_rate': pct(invalid, candidates),
      'invalid_reasons': dict(invalid_reasons),
      'filtered_total': sum(filtered_reasons.values()),
      'filtered_reasons': dict(filtered_reasons),
      'duplicate_rate_vs_candidates': pct(
          filtered_reasons.get('duplicate_canonical', 0), candidates
      ),
      'candidate_ddar_done': candidate_ddar_done,
      'candidate_ddar_statuses': dict(candidate_statuses),
      'candidate_ddar_errors': sum(candidate_ddar_errors.values()),
      'candidate_ddar_error_types': dict(candidate_ddar_errors),
      'candidate_elapsed_sec': num_summary(candidate_elapsed),
      'candidate_added_dependencies': num_summary(candidate_added),
      'max_candidate_added_dependencies': max(candidate_added) if candidate_added else 0,
      'max_added_candidate': max_added_record,
      'root_added_dependencies': root_ddar.get('added_dependencies'),
      'root_elapsed_sec': root_ddar.get('elapsed_sec'),
      'root_status': root_ddar.get('status'),
      'beam_empty_depths': beam_empty_depths,
      'candidate_sources': dict(candidate_sources),
      'valid_candidate_sources': dict(valid_candidate_sources),
      'construction_types': dict(construction_types),
      'construction_types_top': top_counts(construction_types),
      'filtered_construction_types': dict(filtered_construction_types),
      'filtered_construction_types_top': top_counts(filtered_construction_types),
      'filtered_construction_types_by_reason': {
          reason: dict(counter) for reason, counter in filtered_types_by_reason.items()
      },
      'filtered_construction_types_by_reason_top': {
          reason: top_counts(counter)
          for reason, counter in filtered_types_by_reason.items()
      },
      'filtered_sources': dict(filtered_sources),
      'evaluated_construction_types': dict(evaluated_construction_types),
      'evaluated_construction_types_top': top_counts(evaluated_construction_types),
      'evaluated_sources': dict(evaluated_sources),
      'candidate_error_construction_types': dict(candidate_error_construction_types),
      'candidate_error_construction_types_top': top_counts(
          candidate_error_construction_types
      ),
      'timeout_construction_types': dict(timeout_construction_types),
      'timeout_construction_types_top': top_counts(timeout_construction_types),
      'candidate_sft_signals': dict(candidate_sft_signals),
      'candidate_sft_signal_types': dict(candidate_sft_signal_types),
      'candidate_sft_signal_types_top': top_counts(candidate_sft_signal_types),
      'candidate_hard_negative_signals': dict(candidate_hard_negative_signals),
      'candidate_hard_negative_signal_types': dict(
          candidate_hard_negative_signal_types
      ),
      'candidate_hard_negative_signal_types_top': top_counts(
          candidate_hard_negative_signal_types
      ),
      'candidate_hard_negative_signal_sources': dict(
          candidate_hard_negative_signal_sources
      ),
  }
  result['diagnosis'] = diagnose(result)
  return result


def merge_problem_counter(
    problems: list[dict[str, Any]], key: str
) -> Counter:
  merged = Counter()
  for problem in problems:
    merged.update(problem.get(key) or {})
  return merged


def merge_problem_nested_counter(
    problems: list[dict[str, Any]], key: str
) -> dict[str, Counter]:
  merged: dict[str, Counter] = {}
  for problem in problems:
    nested = problem.get(key) or {}
    for reason, counts in nested.items():
      merged.setdefault(reason, Counter()).update(counts)
  return merged


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--out_dir', help='benchmark output dir containing events/ and summary.jsonl')
  parser.add_argument('--events_dir')
  parser.add_argument('--summary_jsonl')
  parser.add_argument('--out_file')
  parser.add_argument('--top_problems', type=int, default=0)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  out_dir = Path(args.out_dir) if args.out_dir else None
  events_dir = Path(args.events_dir) if args.events_dir else (out_dir / 'events' if out_dir else None)
  if events_dir is None:
    raise ValueError('--out_dir or --events_dir is required')
  summary_jsonl = (
      Path(args.summary_jsonl)
      if args.summary_jsonl
      else (out_dir / 'summary.jsonl' if out_dir else None)
  )
  summary = load_summary(summary_jsonl)
  problems = [
      analyze_problem(path, summary)
      for path in sorted(events_dir.glob('*.jsonl'))
  ]
  aggregate_counts = Counter()
  for problem in problems:
    for label in problem['diagnosis']:
      aggregate_counts[label] += 1
  aggregate_candidate_sources = merge_problem_counter(problems, 'candidate_sources')
  aggregate_valid_candidate_sources = merge_problem_counter(
      problems, 'valid_candidate_sources'
  )
  aggregate_construction_types = merge_problem_counter(problems, 'construction_types')
  aggregate_filtered_construction_types = merge_problem_counter(
      problems, 'filtered_construction_types'
  )
  aggregate_filtered_types_by_reason = merge_problem_nested_counter(
      problems, 'filtered_construction_types_by_reason'
  )
  aggregate_evaluated_construction_types = merge_problem_counter(
      problems, 'evaluated_construction_types'
  )
  aggregate_candidate_error_construction_types = merge_problem_counter(
      problems, 'candidate_error_construction_types'
  )
  aggregate_timeout_construction_types = merge_problem_counter(
      problems, 'timeout_construction_types'
  )
  aggregate_candidate_sft_signal_types = merge_problem_counter(
      problems, 'candidate_sft_signal_types'
  )
  aggregate_candidate_hard_negative_signals = merge_problem_counter(
      problems, 'candidate_hard_negative_signals'
  )
  aggregate_candidate_hard_negative_signal_types = merge_problem_counter(
      problems, 'candidate_hard_negative_signal_types'
  )
  aggregate_candidate_hard_negative_signal_sources = merge_problem_counter(
      problems, 'candidate_hard_negative_signal_sources'
  )
  aggregate_solved_aux_types = Counter(
      p.get('solved_aux_construction_type')
      for p in problems
      if p.get('solved_aux_construction_type')
  )
  solved_names = [p['problem'] for p in problems if p.get('solved')]
  payload = {
      'out_dir': str(out_dir) if out_dir else None,
      'events_dir': str(events_dir),
      'summary_jsonl': str(summary_jsonl) if summary_jsonl else None,
      'num_event_files': len(problems),
      'num_completed': sum(1 for p in problems if p.get('completed')),
      'num_solved': len(solved_names),
      'solved_names': solved_names,
      'aggregate_diagnosis': dict(aggregate_counts),
      'aggregate': {
          'candidates': sum(p['candidates'] for p in problems),
          'valid_candidates': sum(p['valid_candidates'] for p in problems),
          'invalid_candidates': sum(p['invalid_candidates'] for p in problems),
          'candidate_ddar_done': sum(p['candidate_ddar_done'] for p in problems),
          'candidate_ddar_errors': sum(p['candidate_ddar_errors'] for p in problems),
          'filtered_total': sum(p['filtered_total'] for p in problems),
          'duplicate_canonical': sum(
              p['filtered_reasons'].get('duplicate_canonical', 0) for p in problems
          ),
          'candidate_sft_signals': sum(
              sum(p['candidate_sft_signals'].values()) for p in problems
          ),
          'candidate_hard_negative_signals': sum(
              sum(p['candidate_hard_negative_signals'].values()) for p in problems
          ),
          'candidate_hard_negative_signal_reasons': dict(
              aggregate_candidate_hard_negative_signals
          ),
          'candidate_sources': dict(aggregate_candidate_sources),
          'valid_candidate_sources': dict(aggregate_valid_candidate_sources),
          'construction_types_top': top_counts(aggregate_construction_types),
          'filtered_construction_types_top': top_counts(
              aggregate_filtered_construction_types
          ),
          'filtered_construction_types_by_reason_top': {
              reason: top_counts(counter)
              for reason, counter in aggregate_filtered_types_by_reason.items()
          },
          'evaluated_construction_types_top': top_counts(
              aggregate_evaluated_construction_types
          ),
          'candidate_error_construction_types_top': top_counts(
              aggregate_candidate_error_construction_types
          ),
          'timeout_construction_types_top': top_counts(
              aggregate_timeout_construction_types
          ),
          'candidate_sft_signal_types_top': top_counts(
              aggregate_candidate_sft_signal_types
          ),
          'candidate_hard_negative_signal_types_top': top_counts(
              aggregate_candidate_hard_negative_signal_types
          ),
          'candidate_hard_negative_signal_sources': dict(
              aggregate_candidate_hard_negative_signal_sources
          ),
          'solved_aux_construction_types_top': top_counts(
              aggregate_solved_aux_types
          ),
          'candidate_backfill_exhausted': sum(
              p['event_counts'].get('candidate_backfill_exhausted', 0)
              for p in problems
          ),
      },
      'problems': problems,
  }
  if args.top_problems and args.top_problems > 0:
    payload['problems'] = problems[: args.top_problems]
  text = json.dumps(payload, ensure_ascii=False, indent=2)
  if args.out_file:
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding='utf-8')
  print(text)


if __name__ == '__main__':
  main()
