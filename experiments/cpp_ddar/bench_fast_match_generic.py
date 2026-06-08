"""Benchmark the experimental C++ generic DD matcher against AG1 Python.

This script does not replace DDAR.  It compares only generic theorem matching,
which is the pure substitution backtracking portion of ``dd.match_generic``.
The graph mutation / dependency / algebra code remains AG1 Python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import fast_match_generic as fmg


def add_repo_to_path(repo: str | Path) -> None:
  repo = str(Path(repo).resolve())
  if repo not in sys.path:
    sys.path.insert(0, repo)


def import_ag(ag_repo: str | Path) -> dict[str, Any]:
  add_repo_to_path(ag_repo)
  import dd  # pylint: disable=import-error,import-outside-toplevel
  import graph as gh  # pylint: disable=import-error,import-outside-toplevel
  import problem as pr  # pylint: disable=import-error,import-outside-toplevel

  return {'dd': dd, 'gh': gh, 'pr': pr}


def build_graph_for_symbolic_search(gh: Any, p: Any, definitions: Any):
  try:
    return gh.Graph.build_problem(p, definitions)
  except Exception:  # pylint: disable=broad-except
    p_no_goal = p.copy()
    p_no_goal.goal = None
    return gh.Graph.build_problem(p_no_goal, definitions)


def mapping_signature(mapping: dict[Any, Any]) -> tuple[tuple[str, str], ...]:
  pairs = []
  for key, value in mapping.items():
    if isinstance(key, str):
      pairs.append((key, getattr(value, 'name', str(value))))
  return tuple(sorted(pairs))


def eligible_generic_theorems(dd: Any, theorems: dict[str, Any], goal: Any):
  skip_suffixes = {'acompute', 'rcompute', 'fixl', 'fixc', 'fixb', 'fixt', 'fixp'}
  for _, theorem in theorems.items():
    if theorem.name in dd.BUILT_IN_FNS:
      continue
    if theorem.name in dd.SKIP_THEOREMS:
      continue
    suffix = theorem.name.split('_')[-1]
    if suffix in dd.SKIP_THEOREMS:
      continue
    if suffix in skip_suffixes and goal and goal.name != theorem.name:
      continue
    yield theorem


def compare_problem(
    name: str,
    problem: Any,
    definitions: Any,
    theorems: dict[str, Any],
    dd: Any,
    gh: Any,
    lib: Any,
    max_results: int,
) -> dict[str, Any]:
  graph, _ = build_graph_for_symbolic_search(gh, problem, definitions)
  cache = dd.cache_match(graph)
  rows = []
  py_total = 0.0
  cpp_total = 0.0
  checked = 0
  mismatches = []

  for theorem in eligible_generic_theorems(dd, theorems, problem.goal):
    start = time.perf_counter()
    py_mappings = list(dd.match_generic(graph, cache, theorem))
    py_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    cpp_mappings = fmg.match_generic_fast(
        graph, cache, theorem, lib, max_results=max_results
    )
    cpp_elapsed = time.perf_counter() - start

    py_sig = [mapping_signature(m) for m in py_mappings]
    cpp_sig = [mapping_signature(m) for m in cpp_mappings]
    same = py_sig == cpp_sig
    if not same:
      mismatches.append({
          'theorem': theorem.name,
          'python_count': len(py_mappings),
          'cpp_count': len(cpp_mappings),
          'python_head': py_sig[:3],
          'cpp_head': cpp_sig[:3],
      })

    checked += 1
    py_total += py_elapsed
    cpp_total += cpp_elapsed
    rows.append({
        'theorem': theorem.name,
        'python_count': len(py_mappings),
        'cpp_count': len(cpp_mappings),
        'same': same,
        'python_sec': py_elapsed,
        'cpp_sec': cpp_elapsed,
        'speedup': (py_elapsed / cpp_elapsed) if cpp_elapsed > 0 else None,
    })

  return {
      'problem': name,
      'generic_theorems_checked': checked,
      'python_sec': py_total,
      'cpp_sec': cpp_total,
      'speedup': (py_total / cpp_total) if cpp_total > 0 else None,
      'mismatch_count': len(mismatches),
      'mismatches': mismatches[:5],
      'top_rows': sorted(rows, key=lambda row: row['python_sec'], reverse=True)[:10],
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--ag_repo', required=True)
  parser.add_argument('--problems_file', required=True)
  parser.add_argument('--defs_file', required=True)
  parser.add_argument('--rules_file', required=True)
  parser.add_argument('--problem_names', default='')
  parser.add_argument('--translate', action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument('--max_results', type=int, default=50_000)
  parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument('--lib_path')
  parser.add_argument('--json_out')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.compile:
    lib_path = fmg.compile_library(output=args.lib_path)
  else:
    lib_path = Path(args.lib_path) if args.lib_path else None
  lib = fmg.load_library(lib_path)

  ag = import_ag(args.ag_repo)
  dd, gh, pr = ag['dd'], ag['gh'], ag['pr']
  definitions = pr.Definition.from_txt_file(args.defs_file, to_dict=True)
  theorems = pr.Theorem.from_txt_file(args.rules_file, to_dict=True)
  problems = pr.Problem.from_txt_file(
      args.problems_file, to_dict=True, translate=args.translate
  )
  names = [x.strip() for x in args.problem_names.split(',') if x.strip()]
  if not names:
    names = list(problems)[:5]

  results = []
  for name in names:
    result = compare_problem(
        name,
        problems[name],
        definitions,
        theorems,
        dd,
        gh,
        lib,
        args.max_results,
    )
    results.append(result)
    print(
        f"{name}: checked={result['generic_theorems_checked']} "
        f"mismatches={result['mismatch_count']} "
        f"python={result['python_sec']:.6f}s cpp={result['cpp_sec']:.6f}s "
        f"speedup={result['speedup']:.2f}x"
    )

  aggregate_python = sum(r['python_sec'] for r in results)
  aggregate_cpp = sum(r['cpp_sec'] for r in results)
  report = {
      'library': str(lib_path) if lib_path else None,
      'problem_count': len(results),
      'aggregate_python_sec': aggregate_python,
      'aggregate_cpp_sec': aggregate_cpp,
      'aggregate_speedup': (
          aggregate_python / aggregate_cpp if aggregate_cpp > 0 else None
      ),
      'total_mismatches': sum(r['mismatch_count'] for r in results),
      'results': results,
  }
  print(json.dumps(report, indent=2, default=str))
  if args.json_out:
    Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
  main()
