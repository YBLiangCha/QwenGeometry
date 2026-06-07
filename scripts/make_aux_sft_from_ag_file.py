"""Build Qwen auxiliary-construction SFT rows from AG-format problems.

This is a starter data-mining script for the real training stage. It treats a
problem with construction clauses as a sequence and emits samples of the form:

  prompt = AG prefix state + goal + " {F1} x00"
  target = next auxiliary construction in original AG constrained LM format

The script can optionally verify that the prefix is not solved by DDAR and that
the prefix plus candidate is solved or at least buildable. Verification should
be run under ``xvfb-run`` on headless Linux because AG1 forces TkAgg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections.abc import Callable
from typing import Any


DEFINITIONS = None
RULES = None


def add_ag_repo_to_path(ag_repo: str) -> None:
  repo = Path(ag_repo).resolve()
  if not repo.exists():
    raise FileNotFoundError(repo)
  sys.path.insert(0, str(repo))


def load_ag_modules() -> dict[str, Any]:
  import ddar  # pylint: disable=import-error,import-outside-toplevel
  import graph as gh  # pylint: disable=import-error,import-outside-toplevel
  import problem as pr  # pylint: disable=import-error,import-outside-toplevel

  return {'ddar': ddar, 'gh': gh, 'pr': pr}


def problem_from_clauses(pr: Any, url: str, clauses: list[Any], goal: Any) -> Any:
  txt = '; '.join(c.txt() for c in clauses)
  if goal is not None:
    txt += ' ? ' + goal.txt()
  if url:
    txt = url + '\n' + txt
  return pr.Problem.from_txt(txt, translate=False)


def goal_is_solved(g: Any, p: Any) -> bool:
  if p.goal is None:
    return False
  return g.check(p.goal.name, g.names2nodes(p.goal.args))


def ddar_solves(p: Any, pr: Any, gh: Any, ddar: Any, max_level: int, timeout: int) -> bool:
  g, _ = gh.Graph.build_problem(p, DEFINITIONS)
  ddar.solve(g, RULES, p, max_level=max_level, timeout=timeout)
  return goal_is_solved(g, p)


def ddar_solve_with_facts(
    p: Any,
    gh: Any,
    ddar: Any,
    max_level: int,
    timeout: int,
    fact_context_top_k: int,
) -> tuple[bool, list[str]]:
  import qwen_ag_search as qs  # pylint: disable=import-error,import-outside-toplevel

  qs.DEFINITIONS = DEFINITIONS
  qs.RULES = RULES
  g, _ = gh.Graph.build_problem(p, DEFINITIONS)
  g, _, _, _, added = ddar.solve(
      g, RULES, p, max_level=max_level, timeout=timeout
  )
  facts = qs.select_ddar_facts(added, p, fact_context_top_k)
  return goal_is_solved(g, p), facts


def candidate_is_buildable(
    p_prefix: Any,
    target: str,
    pr: Any,
    gh: Any,
) -> tuple[bool, str]:
  import pretty as pt  # pylint: disable=import-error,import-outside-toplevel
  import qwen_ag_search as qs  # pylint: disable=import-error,import-outside-toplevel

  qs.DEFINITIONS = DEFINITIONS
  g, _ = gh.Graph.build_problem(p_prefix, DEFINITIONS)
  translation = qs.try_translate_candidate(target, g, pr, pt)
  return not translation.startswith('ERROR:'), translation


PredConverter = Callable[[str, list[str]], list[str] | str | None]


def args_with_point_first(
    point: str, args: list[str], num_other_args: int
) -> list[str] | None:
  if len(args) == num_other_args:
    return [point, *args]
  if len(args) == num_other_args + 1 and args[0] == point:
    return args
  return None


def pred_on_line(point: str, args: list[str]) -> str | None:
  if len(args) == 2:
    x, a, b = point, args[0], args[1]
  elif len(args) == 3:
    x, a, b = args
  else:
    return None
  if x != point:
    return None
  return f'C {a} {b} {x}'


def pred_on_circle(point: str, args: list[str]) -> str | None:
  if len(args) == 2:
    x, o, a = point, args[0], args[1]
  elif len(args) == 3:
    x, o, a = args
  else:
    return None
  if x != point:
    return None
  return f'D {o} {x} {o} {a}'


def pred_on_tline(point: str, args: list[str]) -> str | None:
  if len(args) == 3:
    x, a, b, c = point, args[0], args[1], args[2]
  elif len(args) == 4:
    x, a, b, c = args
  else:
    return None
  if x != point:
    return None
  return f'T {x} {a} {b} {c}'


def pred_on_pline(point: str, args: list[str]) -> str | None:
  if len(args) == 3:
    x, a, b, c = point, args[0], args[1], args[2]
  elif len(args) == 4:
    x, a, b, c = args
  else:
    return None
  if x != point:
    return None
  return f'P {x} {a} {b} {c}'


def pred_on_bline(point: str, args: list[str]) -> str | None:
  if len(args) == 2:
    x, a, b = point, args[0], args[1]
  elif len(args) == 3:
    x, a, b = args
  else:
    return None
  if x != point:
    return None
  return f'D {x} {a} {x} {b}'


def pred_on_circum(point: str, args: list[str]) -> str | None:
  if len(args) == 3:
    x, a, b, c = point, args[0], args[1], args[2]
  elif len(args) == 4:
    x, a, b, c = args
  else:
    return None
  if x != point:
    return None
  return f'O {a} {b} {c} {x}'


def pred_eqdistance(point: str, args: list[str]) -> str | None:
  if len(args) == 3:
    x, a, b, c = point, args[0], args[1], args[2]
  elif len(args) == 4:
    x, a, b, c = args
  else:
    return None
  if x != point:
    return None
  return f'D {x} {a} {b} {c}'


def pred_on_dia(point: str, args: list[str]) -> str | None:
  values = args_with_point_first(point, args, 2)
  if values is None:
    return None
  x, a, b = values
  return f'T {x} {a} {x} {b}'


def pred_midpoint(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 2)
  if values is None:
    return None
  x, a, b = values
  return [f'C {a} {b} {x}', f'D {x} {a} {x} {b}']


def pred_foot(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 3)
  if values is None:
    return None
  x, a, b, c = values
  return [f'C {b} {c} {x}', f'T {x} {a} {b} {c}']


def pred_circumcenter(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 3)
  if values is None:
    return None
  x, a, b, c = values
  return [f'D {x} {a} {x} {b}', f'D {x} {b} {x} {c}']


def pred_angle_bisector(point: str, args: list[str]) -> str | None:
  values = args_with_point_first(point, args, 3)
  if values is None:
    return None
  x, a, b, c = values
  return f'^ {b} {a} {b} {x} {b} {x} {b} {c}'


def pred_angle_mirror(point: str, args: list[str]) -> str | None:
  values = args_with_point_first(point, args, 3)
  if values is None:
    return None
  x, a, b, c = values
  return f'^ {b} {a} {b} {c} {b} {c} {b} {x}'


def pred_mirror(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 2)
  if values is None:
    return None
  x, a, b = values
  return [f'C {a} {b} {x}', f'D {b} {a} {b} {x}']


def pred_intersection_ll(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 4)
  if values is None:
    return None
  x, a, b, c, d = values
  return [f'C {a} {b} {x}', f'C {c} {d} {x}']


def pred_intersection_lc(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 3)
  if values is None:
    return None
  x, a, o, b = values
  return [f'C {a} {b} {x}', f'D {o} {b} {o} {x}']


def pred_intersection_cc(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 3)
  if values is None:
    return None
  x, o, w, a = values
  return [f'D {o} {a} {o} {x}', f'D {w} {a} {w} {x}']


def pred_intersection_lp(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 5)
  if values is None:
    return None
  x, a, b, c, m, n = values
  return [f'C {a} {b} {x}', f'P {c} {x} {m} {n}']


def pred_intersection_lt(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 5)
  if values is None:
    return None
  x, a, b, c, d, e = values
  return [f'C {a} {b} {x}', f'T {x} {c} {d} {e}']


def pred_intersection_pp(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 6)
  if values is None:
    return None
  x, a, b, c, d, e, f = values
  return [f'P {x} {a} {b} {c}', f'P {x} {d} {e} {f}']


def pred_intersection_tt(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 6)
  if values is None:
    return None
  x, a, b, c, d, e, f = values
  return [f'T {x} {a} {b} {c}', f'T {x} {d} {e} {f}']


def pred_eq_triangle(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 2)
  if values is None:
    return None
  x, b, c = values
  return [f'D {x} {b} {b} {c}', f'D {b} {c} {c} {x}']


def pred_lc_tangent(point: str, args: list[str]) -> str | None:
  values = args_with_point_first(point, args, 2)
  if values is None:
    return None
  x, a, o = values
  return f'T {a} {x} {a} {o}'


def pred_parallelogram(point: str, args: list[str]) -> list[str] | None:
  if len(args) == 3:
    a, b, c, x = args[0], args[1], args[2], point
  elif len(args) == 4 and args[3] == point:
    a, b, c, x = args
  else:
    return None
  return [f'P {a} {b} {c} {x}', f'P {a} {x} {b} {c}']


def pred_orthocenter(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 3)
  if values is None:
    return None
  x, a, b, c = values
  return [f'T {x} {a} {b} {c}', f'T {x} {b} {c} {a}']


def pred_incenter(point: str, args: list[str]) -> list[str] | None:
  values = args_with_point_first(point, args, 3)
  if values is None:
    return None
  x, a, b, c = values
  return [
      f'^ {a} {b} {a} {x} {a} {x} {a} {c}',
      f'^ {c} {a} {c} {x} {c} {x} {c} {b}',
  ]


def pred_on_aline(point: str, args: list[str]) -> str | None:
  values = args_with_point_first(point, args, 5)
  if values is None:
    return None
  x, a, b, c, d, e = values
  return f'^ {a} {x} {a} {b} {d} {c} {d} {e}'


def pred_on_aline2(point: str, args: list[str]) -> str | None:
  values = args_with_point_first(point, args, 5)
  if values is None:
    return None
  x, a, b, c, d, e = values
  return f'^ {x} {a} {x} {b} {d} {c} {d} {e}'


CONVERTERS: dict[str, PredConverter] = {
    'on_line': pred_on_line,
    'on_circle': pred_on_circle,
    'on_tline': pred_on_tline,
    'on_pline': pred_on_pline,
    'on_bline': pred_on_bline,
    'on_circum': pred_on_circum,
    'eqdistance': pred_eqdistance,
    'on_dia': pred_on_dia,
    'midpoint': pred_midpoint,
    'foot': pred_foot,
    'circle': pred_circumcenter,
    'circumcenter': pred_circumcenter,
    'angle_bisector': pred_angle_bisector,
    'angle_mirror': pred_angle_mirror,
    'mirror': pred_mirror,
    'intersection_ll': pred_intersection_ll,
    'intersection_lc': pred_intersection_lc,
    'intersection_cc': pred_intersection_cc,
    'intersection_lp': pred_intersection_lp,
    'intersection_lt': pred_intersection_lt,
    'intersection_pp': pred_intersection_pp,
    'intersection_tt': pred_intersection_tt,
    'eq_triangle': pred_eq_triangle,
    'lc_tangent': pred_lc_tangent,
    'parallelogram': pred_parallelogram,
    'orthocenter': pred_orthocenter,
    'incenter': pred_incenter,
    'excenter': pred_incenter,
    'on_aline': pred_on_aline,
    'on_aline2': pred_on_aline2,
}


def clause_to_constrained_target(clause: Any) -> str | None:
  if len(clause.points) != 1:
    return None
  point = clause.points[0]
  preds = []
  for idx, cons in enumerate(clause.constructions):
    converter = CONVERTERS.get(cons.name)
    if converter is None:
      return None
    converted = converter(point, cons.args)
    if converted is None:
      return None
    if isinstance(converted, str):
      converted = [converted]
    preds.extend(converted)
    if len(preds) > 2:
      return None
  if not preds:
    return None
  indexed_preds = [f'{pred} {idx:02d}' for idx, pred in enumerate(preds)]
  return f'{point} : ' + ' '.join(indexed_preds) + ' ;'


def make_prompt(p_prefix: Any, facts: list[str] | None = None) -> str:
  if not facts:
    return p_prefix.setup_str_from_problem(DEFINITIONS) + ' {F1} x00'
  import qwen_ag_search as qs  # pylint: disable=import-error,import-outside-toplevel

  return qs.build_lm_prompt(p_prefix, DEFINITIONS, facts)


def goal_points_are_known(clauses: list[Any], goal: Any) -> bool:
  if goal is None:
    return True
  known_points = set()
  for clause in clauses:
    known_points.update(clause.points)
  for arg in goal.args:
    if arg.isdigit():
      continue
    if arg not in known_points:
      return False
  return True


def clause_touches_goal(clause: Any, goal: Any) -> bool:
  if goal is None:
    return False
  return any(point in goal.args for point in clause.points)


def mine_problem(
    p: Any,
    pr: Any,
    gh: Any,
    ddar: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
  rows = []
  clauses = p.clauses
  goal_clause_indices = {
      i for i, clause in enumerate(clauses) if clause_touches_goal(clause, p.goal)
  }
  for idx in range(args.min_prefix_clauses, len(clauses)):
    if args.skip_goal_points and clause_touches_goal(clauses[idx], p.goal):
      continue

    prefix_indices = set(range(idx))
    if args.include_goal_point_clauses_in_prefix:
      prefix_indices.update(goal_clause_indices)
    prefix_indices.discard(idx)
    prefix_clauses = [clauses[i] for i in sorted(prefix_indices)]

    if args.require_goal_points_in_prefix:
      if not goal_points_are_known(prefix_clauses, p.goal):
        continue

    target = clause_to_constrained_target(clauses[idx])
    if target is None:
      continue

    candidate_clauses = prefix_clauses + [clauses[idx]]
    p_prefix = problem_from_clauses(pr, p.url, prefix_clauses, p.goal)
    p_candidate = problem_from_clauses(pr, p.url, candidate_clauses, p.goal)

    candidate_translation = None
    if args.require_buildable:
      try:
        buildable, candidate_translation = candidate_is_buildable(
            p_prefix, target, pr, gh
        )
      except Exception as exc:  # pylint: disable=broad-except
        buildable = False
        candidate_translation = f'ERROR: {type(exc).__name__}: {exc}'
      if not buildable:
        continue

    prefix_solved = None
    candidate_solved = None
    prefix_facts: list[str] = []
    if args.fact_context_top_k > 0:
      try:
        prefix_solved, prefix_facts = ddar_solve_with_facts(
            p_prefix,
            gh,
            ddar,
            args.fact_context_max_level or args.max_level,
            args.fact_context_ddar_timeout or args.ddar_timeout,
            args.fact_context_top_k,
        )
      except Exception:  # pylint: disable=broad-except
        continue
    if args.verify:
      if prefix_solved is None:
        prefix_solved = ddar_solves(
            p_prefix, pr, gh, ddar, args.max_level, args.ddar_timeout
        )
      if args.require_prefix_unsolved and prefix_solved:
        continue
      candidate_solved = ddar_solves(
          p_candidate, pr, gh, ddar, args.max_level, args.ddar_timeout
      )
      if args.require_candidate_solved and not candidate_solved:
        continue

    prompt_without_fact_context = make_prompt(p_prefix)
    rows.append({
        'id': f'{p.url or "problem"}::clause_{idx}',
        'source_problem': p.url,
        'clause_index': idx,
        'prefix_clause_indices': sorted(prefix_indices),
        'prompt': make_prompt(p_prefix, prefix_facts),
        'prompt_without_fact_context': prompt_without_fact_context
        if prefix_facts
        else None,
        'fact_context': prefix_facts,
        'fact_context_top_k': args.fact_context_top_k,
        'target': target,
        'candidate_constructive': clauses[idx].txt(),
        'candidate_translation': candidate_translation,
        'prefix_solved_by_ddar': prefix_solved,
        'candidate_solved_by_ddar': candidate_solved,
    })
  return rows


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--ag_repo', required=True)
  parser.add_argument('--problems_file', required=True)
  parser.add_argument('--defs_file', required=True)
  parser.add_argument('--rules_file', required=True)
  parser.add_argument('--out_file', required=True)
  parser.add_argument('--translate', action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument('--min_prefix_clauses', type=int, default=1)
  parser.add_argument('--max_problems', type=int)
  parser.add_argument('--skip_goal_points', action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
      '--require_goal_points_in_prefix',
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      '--include_goal_point_clauses_in_prefix',
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument('--require_buildable', action='store_true')
  parser.add_argument('--verify', action='store_true')
  parser.add_argument('--require_prefix_unsolved', action='store_true')
  parser.add_argument('--require_candidate_solved', action='store_true')
  parser.add_argument('--max_level', type=int, default=1000)
  parser.add_argument('--ddar_timeout', type=int, default=120)
  parser.add_argument(
      '--fact_context_top_k',
      type=int,
      default=0,
      help='include top-K DDAR-added facts in each SFT prompt; 0 disables',
  )
  parser.add_argument(
      '--fact_context_max_level',
      type=int,
      help='DDAR max level for fact context extraction; defaults to --max_level',
  )
  parser.add_argument(
      '--fact_context_ddar_timeout',
      type=int,
      help='DDAR timeout for fact context extraction; defaults to --ddar_timeout',
  )
  return parser.parse_args()


def main() -> None:
  global DEFINITIONS, RULES
  args = parse_args()
  add_ag_repo_to_path(args.ag_repo)
  ag = load_ag_modules()
  pr, gh, ddar = ag['pr'], ag['gh'], ag['ddar']
  DEFINITIONS = pr.Definition.from_txt_file(args.defs_file, to_dict=True)
  RULES = pr.Theorem.from_txt_file(args.rules_file, to_dict=True)
  problems = pr.Problem.from_txt_file(args.problems_file, translate=args.translate)
  if args.max_problems:
    problems = problems[: args.max_problems]

  total = 0
  with open(args.out_file, 'w', encoding='utf-8') as out:
    for p in problems:
      rows = mine_problem(p, pr, gh, ddar, args)
      for row in rows:
        out.write(json.dumps(row, ensure_ascii=False) + '\n')
      total += len(rows)
      print(json.dumps({
          'problem': p.url,
          'rows': len(rows),
          'total': total,
      }), flush=True)


if __name__ == '__main__':
  main()
