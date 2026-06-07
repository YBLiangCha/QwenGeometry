#!/usr/bin/env python3
"""Add DDAR fact context to existing auxiliary-construction SFT rows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


DEFINITIONS = None
RULES = None

GOAL_NAME_MAP = {
    'C': 'coll',
    'D': 'cong',
    'O': 'cyclic',
    'P': 'para',
    'T': 'perp',
    '^': 'eqangle',
}


def add_paths(ag_repo: str) -> None:
  script_dir = Path(__file__).resolve().parent
  repo = Path(ag_repo).resolve()
  sys.path.insert(0, str(script_dir))
  sys.path.insert(0, str(repo))


def load_modules() -> dict[str, Any]:
  import ddar  # pylint: disable=import-error,import-outside-toplevel
  import graph as gh  # pylint: disable=import-error,import-outside-toplevel
  import pretty as pt  # pylint: disable=import-error,import-outside-toplevel
  import problem as pr  # pylint: disable=import-error,import-outside-toplevel
  import qwen_ag_search as qs  # pylint: disable=import-error,import-outside-toplevel

  return {'ddar': ddar, 'gh': gh, 'pr': pr, 'pt': pt, 'qs': qs}


def iter_jsonl(path: Path):
  for line_no, line in enumerate(
      path.read_text(encoding='utf-8', errors='replace').splitlines(), 1
  ):
    line = line.lstrip('\ufeff')
    if not line.strip():
      continue
    yield line_no, json.loads(line)


def split_prompt(prompt: str) -> tuple[str, str]:
  text = prompt.strip()
  if text.startswith('{S}'):
    text = text[len('{S}'):].strip()
  if '{F1}' in text:
    text = text.split('{F1}', 1)[0].strip()
  if '{D}' in text:
    text = text.split('{D}', 1)[0].strip()
  if ' ? ' in text:
    setup, goal = text.split(' ? ', 1)
  else:
    setup, goal = text, ''
  return setup.strip(), goal.strip()


def with_fact_context(prompt: str, facts: list[str], qs: Any) -> str:
  prefix = prompt.split('{F1}', 1)[0].rstrip()
  if '{D}' in prefix:
    prefix = prefix.split('{D}', 1)[0].rstrip()
  return prefix + qs.fact_context_text(facts) + ' {F1} x00'


def build_problem_from_prompt_setup(
    setup: str,
    goal: str,
    pr: Any,
    gh: Any,
    pt: Any,
    qs: Any,
) -> Any:
  constructive_clauses: list[str] = []
  for raw_clause in setup.split(';'):
    clause = raw_clause.strip()
    if not clause:
      continue
    if ' = ' in clause:
      constructive_clauses.append(clause)
      continue
    if ':' not in clause:
      raise ValueError(f'unsupported setup clause: {clause}')
    points_text, rhs = clause.split(':', 1)
    points = points_text.strip().split()
    rhs = rhs.strip()
    if not points:
      raise ValueError(f'missing point in setup clause: {clause}')
    if not rhs:
      constructive_clauses.append(
          f'{" ".join(points)} = free {" ".join(points)}'
      )
      continue
    if len(points) != 1:
      raise ValueError(f'multi-point constrained clause unsupported: {clause}')
    if not constructive_clauses:
      raise ValueError(f'non-free first clause: {clause}')
    p_cur = pr.Problem.from_txt('; '.join(constructive_clauses), translate=False)
    g_cur, _ = gh.Graph.build_problem(p_cur, DEFINITIONS)
    candidate = f'{points[0]} : {rhs} ;'
    translation = qs.try_translate_candidate(candidate, g_cur, pr, pt)
    if translation.startswith('ERROR:'):
      raise ValueError(translation)
    constructive_clauses.append(translation.strip().rstrip(';'))
  if not constructive_clauses:
    raise ValueError('empty setup')
  setup_text = '; '.join(constructive_clauses)
  goal_text = goal_to_ag(goal)
  if goal_text:
    try:
      return pr.Problem.from_txt(
          setup_text + ' ? ' + goal_text, translate=False
      )
    except Exception:  # pylint: disable=broad-except
      pass
  return pr.Problem.from_txt(setup_text, translate=False)


def goal_to_ag(goal: str) -> str:
  tokens = goal.strip().split()
  if not tokens:
    return ''
  name = GOAL_NAME_MAP.get(tokens[0], tokens[0].lower())
  return ' '.join([name, *tokens[1:]])


def extract_facts(
    row: dict[str, Any],
    pr: Any,
    gh: Any,
    ddar: Any,
    pt: Any,
    qs: Any,
    max_level: int,
    timeout: int,
    top_k: int,
) -> tuple[list[str], int, str]:
  setup, goal = split_prompt(str(row.get('prompt') or ''))
  problem = build_problem_from_prompt_setup(setup, goal, pr, gh, pt, qs)
  g, _ = gh.Graph.build_problem(problem, DEFINITIONS)
  g, _, status, _, added = ddar.solve(
      g, RULES, problem, max_level=max_level, timeout=timeout
  )
  facts = qs.select_ddar_facts(added, problem, top_k)
  return facts, len(added), status


def split_for_row(row: dict[str, Any], eval_mod: int) -> str:
  if eval_mod <= 0:
    return 'train'
  key = row.get('source_problem') or row.get('id') or row.get('prompt') or ''
  return 'eval' if sum(str(key).encode('utf-8')) % eval_mod == 0 else 'train'


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
      encoding='utf-8',
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--ag_repo', required=True)
  parser.add_argument('--defs_file', required=True)
  parser.add_argument('--rules_file', required=True)
  parser.add_argument('--input_file', action='append', required=True)
  parser.add_argument('--train_file', required=True)
  parser.add_argument('--eval_file', required=True)
  parser.add_argument('--summary_file', required=True)
  parser.add_argument('--fact_context_top_k', type=int, default=8)
  parser.add_argument('--fact_context_max_level', type=int, default=4)
  parser.add_argument('--fact_context_ddar_timeout', type=int, default=30)
  parser.add_argument('--max_rows', type=int, default=0)
  parser.add_argument('--eval_mod', type=int, default=10)
  parser.add_argument('--keep_no_fact_rows', action='store_true')
  parser.add_argument('--progress_every', type=int, default=50)
  return parser.parse_args()


def main() -> None:
  global DEFINITIONS, RULES
  args = parse_args()
  add_paths(args.ag_repo)
  modules = load_modules()
  pr, gh, ddar, pt, qs = (
      modules['pr'], modules['gh'], modules['ddar'], modules['pt'], modules['qs']
  )
  DEFINITIONS = pr.Definition.from_txt_file(args.defs_file, to_dict=True)
  RULES = pr.Theorem.from_txt_file(args.rules_file, to_dict=True)
  qs.DEFINITIONS = DEFINITIONS
  qs.RULES = RULES

  counts: Counter[str] = Counter()
  rows: list[dict[str, Any]] = []
  seen: set[tuple[str, str]] = set()
  for input_file in [Path(path) for path in args.input_file]:
    for line_no, row in iter_jsonl(input_file):
      if args.max_rows and counts['seen'] >= args.max_rows:
        break
      counts['seen'] += 1
      if args.progress_every > 0 and counts['seen'] % args.progress_every == 0:
        print(json.dumps({
            'progress': True,
            'seen': counts['seen'],
            'kept': counts['kept'],
            'errors': sum(
                value for key, value in counts.items() if key.startswith('error:')
            ),
            'no_facts': counts['no_facts'],
        }, ensure_ascii=False), flush=True)
      if not row.get('prompt') or not row.get('target'):
        counts['missing_prompt_or_target'] += 1
        continue
      key = (str(row['prompt']), str(row['target']))
      if key in seen:
        counts['duplicate_prompt_target'] += 1
        continue
      seen.add(key)
      try:
        facts, added, status = extract_facts(
            row,
            pr,
            gh,
            ddar,
            pt,
            qs,
            args.fact_context_max_level,
            args.fact_context_ddar_timeout,
            args.fact_context_top_k,
        )
      except Exception as exc:  # pylint: disable=broad-except
        counts[f'error:{type(exc).__name__}'] += 1
        continue
      if not facts and not args.keep_no_fact_rows:
        counts['no_facts'] += 1
        continue
      out_row = dict(row)
      out_row['id'] = f'{row.get("id") or input_file.stem}:{line_no}:factctx'
      out_row['prompt_without_fact_context'] = row['prompt']
      out_row['prompt'] = with_fact_context(str(row['prompt']), facts, qs)
      out_row['fact_context'] = facts
      out_row['fact_context_top_k'] = args.fact_context_top_k
      out_row['fact_context_added_dependencies'] = added
      out_row['fact_context_ddar_status'] = status
      out_row['fact_context_source'] = 'stage2_aux_prompt_ddar'
      out_row['split'] = split_for_row(out_row, args.eval_mod)
      rows.append(out_row)
      counts['kept'] += 1
      counts[out_row['split']] += 1
    if args.max_rows and counts['seen'] >= args.max_rows:
      break

  train_rows = [row for row in rows if row['split'] == 'train']
  eval_rows = [row for row in rows if row['split'] == 'eval']
  if not eval_rows and len(train_rows) > 5 and args.eval_mod > 0:
    eval_rows = train_rows[-max(1, len(train_rows) // 10):]
    train_rows = train_rows[:-len(eval_rows)]
  write_jsonl(Path(args.train_file), train_rows)
  write_jsonl(Path(args.eval_file), eval_rows)
  summary = {
      'input_file': args.input_file,
      'train_file': args.train_file,
      'eval_file': args.eval_file,
      'rows': len(rows),
      'train_rows': len(train_rows),
      'eval_rows': len(eval_rows),
      'fact_context_top_k': args.fact_context_top_k,
      'fact_context_max_level': args.fact_context_max_level,
      'fact_context_ddar_timeout': args.fact_context_ddar_timeout,
      'counts': dict(counts),
  }
  summary_path = Path(args.summary_file)
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
