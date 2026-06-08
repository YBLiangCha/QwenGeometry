"""ctypes wrapper for the experimental C++ generic DD matcher."""

from __future__ import annotations

import ctypes
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable, Iterable


_NUMERICAL_CHECKS = {'ncoll', 'npara', 'nperp', 'sameside'}


def shared_library_suffix() -> str:
  system = platform.system().lower()
  if system == 'windows':
    return '.dll'
  if system == 'darwin':
    return '.dylib'
  return '.so'


def compile_library(
    source: str | Path | None = None,
    output: str | Path | None = None,
    cxx: str = 'g++',
) -> Path:
  """Compile the C++ matcher into a shared library."""
  here = Path(__file__).resolve().parent
  source_path = Path(source) if source else here / 'fast_match_generic.cpp'
  output_path = (
      Path(output)
      if output
      else here / ('libfast_match_generic' + shared_library_suffix())
  )
  cmd = [
      cxx,
      '-O3',
      '-std=c++17',
      '-shared',
      '-fPIC',
      str(source_path),
      '-o',
      str(output_path),
  ]
  subprocess.run(cmd, check=True)
  return output_path


def load_library(path: str | Path | None = None) -> ctypes.CDLL:
  here = Path(__file__).resolve().parent
  lib_path = (
      Path(path) if path else here / ('libfast_match_generic' + shared_library_suffix())
  )
  lib = ctypes.CDLL(str(lib_path))
  lib.fast_match_generic.argtypes = [
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_int),
      ctypes.POINTER(ctypes.c_int),
      ctypes.POINTER(ctypes.c_int),
      ctypes.POINTER(ctypes.c_int),
      ctypes.POINTER(ctypes.c_int),
      ctypes.POINTER(ctypes.c_int),
      ctypes.c_int,
      ctypes.c_int,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.fast_match_generic.restype = ctypes.c_int
  return lib


def _int_array(values: Iterable[int]) -> Any:
  vals = list(values)
  if not vals:
    vals = [0]
  return (ctypes.c_int * len(vals))(*vals)


def _positive_clauses(theorem: Any) -> list[Any]:
  clauses = []
  for clause in theorem.premise:
    if clause.name in _NUMERICAL_CHECKS:
      continue
    clauses.append((len(set(clause.args)), clause))
  clauses.sort(key=lambda item: item[0], reverse=True)
  return [clause for _, clause in clauses]


def _numerical_checks(theorem: Any) -> list[Any]:
  return [clause for clause in theorem.premise if clause.name in _NUMERICAL_CHECKS]


def raw_match_generic(
    theorem: Any,
    cache: Callable[[str], list[tuple[Any, ...]]],
    lib: ctypes.CDLL,
    max_results: int = 50_000,
) -> list[dict[Any, Any]]:
  """Return generic theorem mappings before numerical checks.

  The returned dictionaries mirror dd.try_to_map: both variable->point and
  point->variable entries are present.
  """
  clauses = _positive_clauses(theorem)
  if not clauses:
    return []

  enums_by_clause = []
  for clause in clauses:
    enum = list(cache(clause.name))
    if not enum:
      return []
    enums_by_clause.append(enum)

  var_to_id: dict[str, int] = {}
  id_to_var: list[str] = []
  point_to_id: dict[Any, int] = {}
  id_to_point: list[Any] = []

  clause_arities = []
  clause_var_offsets = []
  clause_vars = []
  enum_counts = []
  enum_offsets = []
  enum_points = []

  for clause, enum in zip(clauses, enums_by_clause):
    clause_arities.append(len(clause.args))
    clause_var_offsets.append(len(clause_vars))
    for var in clause.args:
      if var not in var_to_id:
        var_to_id[var] = len(id_to_var)
        id_to_var.append(var)
      clause_vars.append(var_to_id[var])

    enum_counts.append(len(enum))
    enum_offsets.append(len(enum_points))
    for row in enum:
      for point in row:
        if point not in point_to_id:
          point_to_id[point] = len(id_to_point)
          id_to_point.append(point)
        enum_points.append(point_to_id[point])

  num_vars = len(id_to_var)
  num_points = len(id_to_point)
  out = (ctypes.c_int * (max_results * max(1, num_vars)))()
  result_count = lib.fast_match_generic(
      len(clauses),
      _int_array(clause_arities),
      _int_array(clause_var_offsets),
      _int_array(clause_vars),
      _int_array(enum_counts),
      _int_array(enum_offsets),
      _int_array(enum_points),
      num_vars,
      num_points,
      max_results,
      out,
  )
  if result_count < 0:
    raise RuntimeError(f'fast_match_generic returned {result_count}')
  if result_count == max_results:
    raise RuntimeError(
        f'fast_match_generic reached max_results={max_results}; '
        'raise the cap before using the timing/equivalence result'
    )

  mappings = []
  for row in range(result_count):
    mapping: dict[Any, Any] = {}
    for var_id, point_id in enumerate(out[row * num_vars : (row + 1) * num_vars]):
      if point_id < 0:
        continue
      var = id_to_var[var_id]
      point = id_to_point[point_id]
      mapping[var] = point
      mapping[point] = var
    mappings.append(mapping)
  return mappings


def match_generic_fast(
    g: Any,
    cache: Callable[[str], list[tuple[Any, ...]]],
    theorem: Any,
    lib: ctypes.CDLL,
    max_results: int = 50_000,
) -> list[dict[Any, Any]]:
  """Equivalent of dd.match_generic with Python numerical checks preserved."""
  checks = _numerical_checks(theorem)
  out = []
  for mapping in raw_match_generic(theorem, cache, lib, max_results=max_results):
    if not mapping:
      continue
    checks_ok = True
    for check in checks:
      args = [mapping[a] for a in check.args]
      if check.name == 'ncoll':
        checks_ok = g.check_ncoll(args)
      elif check.name == 'npara':
        checks_ok = g.check_npara(args)
      elif check.name == 'nperp':
        checks_ok = g.check_nperp(args)
      elif check.name == 'sameside':
        checks_ok = g.check_sameside(args)
      if not checks_ok:
        break
    if checks_ok:
      out.append(mapping)
  return out
