"""Run AG1 DDAR, optionally with a Qwen auxiliary-construction generator.

This script keeps the AG1 symbolic stack unchanged: problems are parsed by
AG1's ``problem.py``, proof states are AG1 ``Graph`` objects, and every
candidate construction is inserted only after AG1 can parse/build it. Qwen is
used only at the same boundary as the original AlphaGeometry LM: when DDAR
fails, propose the next auxiliary construction.
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any


DEFINITIONS = None
RULES = None

_POINT_RE = re.compile(r'^[A-Za-z]$')
_DSL_WORD_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$|^\^$')
_CONSTRAINED_PREDICATE_ARITY: dict[str, int | tuple[int, ...]] = {
    'T': 4,
    'perp': 4,
    'P': 4,
    'para': 4,
    'D': 4,
    'cong': 4,
    'C': 3,
    'coll': 3,
    '^': (6, 8),
    'eqangle': (6, 8),
    'O': 4,
    'cyclic': 4,
}
_CONSTRUCTIVE_ARG_ARITY: dict[str, int | tuple[int, ...]] = {
    'angle_bisector': 4,
    'angle_mirror': 4,
    'eqangle3': 5,
    'eqdistance': 4,
    'on_aline': 6,
    'on_aline2': 6,
    'on_bline': 3,
    'on_circle': 3,
    'on_circum': 4,
    'on_dia': 3,
    'on_line': 3,
    'on_pline': 4,
    'on_tline': 4,
}
_DSL_KEYWORDS = set(_CONSTRAINED_PREDICATE_ARITY) | set(_CONSTRUCTIVE_ARG_ARITY)
_DSL_KEYWORD_FRAGMENTS = {
    word[i:j]
    for word in _DSL_KEYWORDS
    for i in range(len(word))
    for j in range(i + 2, len(word) + 1)
}
_HIGH_VALUE_CONSTRUCTION_TYPES = {
    'angle_bisector',
    'angle_mirror',
    'eqangle3',
    'on_bline',
    'on_aline',
    'on_aline2',
    'on_circum',
    'on_dia',
}
_VERIFIER_PRIOR_TYPE_BONUS = {
    # Repeated online SFT positives in the unsolved benchmark, under-ranked by
    # the pre-run value model. Keep these modest; diversity still gates coverage.
    'on_bline+on_line': 2.5,
    'eqdistance+on_line': 2.0,
    'on_bline+on_circum': 1.8,
    'on_circle+on_line': 1.7,
    'on_tline+on_tline': 1.5,
    'on_bline+on_circle': 1.5,
    'on_bline+on_bline': 1.4,
    'eqdistance+on_circle': 1.4,
    'on_circle+on_circum': 1.3,
    'on_dia+on_line': 1.2,
    'on_circum+on_circum': 1.2,
    'on_circle+on_pline': 1.1,
    'on_bline+on_pline': 1.1,
    'on_dia+on_tline': 1.1,
    'angle_bisector': 1.0,
    'on_aline+on_circle': 1.0,
    'on_circle+on_dia': 1.0,
    'on_circle+on_tline': 1.0,
    'on_circum+on_tline': 1.0,
    'on_circum+on_line': 0.9,
    'on_line+on_tline': 0.9,
    'on_line+on_line': 0.9,
    'on_circle+on_circle': 0.9,
    'on_circum+on_dia': 0.8,
    'on_bline+on_dia': 0.8,
    'on_bline': 1.0,
    'on_dia': 0.8,
    'eqangle3': 0.8,
    'on_aline': 0.8,
    'eqdistance+on_tline': 0.5,
}
_PROGRESS_SIGNAL_TYPE_BONUS = {
    # Mined from online DDAR-progress positives in the post-v12 clean traces.
    # These are not proof labels; they only reserve coverage slots for
    # construction families that the value model tends to under-rank before
    # DDAR can observe their effect.
    'on_tline+on_tline': 4.0,
    'on_bline+on_dia': 3.5,
    'on_bline': 3.2,
    'on_bline+on_bline': 3.0,
    'on_bline+on_line': 3.0,
    'on_line+on_tline': 3.0,
    'on_line+on_line': 2.8,
    'eqdistance+on_circle': 2.5,
    'on_bline+on_circum': 2.5,
    # IMO 2018 P1 current trace: strongest DDAR-progress candidate uses
    # on_circle+on_tline (delta 193), so reserve coverage before it is solved.
    'on_circle+on_tline': 2.4,
    # Same trace shows these families adding large DDAR fact deltas but they
    # were not protected by the previous anchor set.
    'on_bline+on_tline': 2.3,
    'eqdistance+on_circum': 2.2,
    'on_bline+on_circle': 2.3,
    'on_circle+on_line': 2.3,
    'on_circle+on_dia': 2.2,
    'on_circle+on_circum': 2.1,
    'on_circle+on_pline': 2.0,
    'eqdistance+on_bline': 2.0,
    'on_circum+on_tline': 2.0,
    # Newly solved IMO 2015 P3 used on_line+on_circum; keep this family near
    # other solved/progress-positive two-construction combos.
    'on_circum+on_line': 2.2,
    'on_bline+on_pline': 1.9,
    'on_circum+on_dia': 1.8,
    'eqdistance+on_line': 1.8,
    'on_dia': 1.8,
    'on_dia+on_tline': 1.7,
    'on_dia+on_line': 1.7,
    'on_circum+on_circum': 1.5,
    'on_aline+on_circle': 1.5,
    'angle_bisector': 1.4,
    # Single-construction families are noisy, so keep these below the
    # progress-positive combos and let the adaptive value score decide.
    'on_line': 1.2,
    'on_circum': 1.1,
    'on_circle': 1.1,
    'on_tline': 1.0,
    'on_aline': 0.9,
}
_SIGNAL_ANCHOR_TYPE_BONUS = {
    # Stronger anchor slots for construction families that have already solved
    # or repeatedly produced DDAR-positive signals in the current online loop.
    # They run before the broader progress coverage phase so narrow depth eval
    # budgets keep at least a few proven families alive.
    'on_line+on_tline': 6.0,
    'on_bline+on_line': 6.0,
    'on_line+on_line': 5.5,
    'on_line+on_circle': 4.5,
    'on_circum': 4.2,
    'on_tline': 4.0,
    'on_circle+on_line': 4.0,
    'on_circle': 3.8,
    'on_dia+on_line': 3.6,
    'on_circum+on_tline': 3.6,
    'on_circle+on_circum': 3.5,
    'eqangle3': 3.4,
    'on_circle+on_tline': 3.4,
    'on_bline+on_tline': 3.3,
    'on_circum+on_circum': 3.4,
    'on_circum+on_line': 5.6,
    'on_bline': 3.2,
    'on_bline+on_circle': 3.2,
    'eqdistance+on_circum': 3.1,
    'on_circle+on_pline': 3.1,
    'on_tline+on_tline': 3.1,
    'eqdistance+on_line': 3.0,
    'on_bline+on_bline': 3.0,
    'eqdistance+on_circle': 2.9,
    'eqdistance+on_bline': 2.8,
    'on_dia': 2.8,
    'eqdistance': 2.7,
    'on_line+on_pline': 2.7,
    'on_aline': 2.6,
    'on_line': 2.5,
    'on_pline': 2.3,
    'angle_bisector': 2.0,
}
_CONSTRUCTIVE_REQUIRES_OUTPUT_FIRST_ARG = (
    set(_CONSTRUCTIVE_ARG_ARITY) - {'eqangle3'}
)
_CONSTRAINED_PROMPT_PREFIXES = ('', 'C', 'O', 'P', 'T', 'D', '^')
_CONSTRAINED_PREDICATE_ALIASES = {
    'T': 'perp',
    'P': 'para',
    'D': 'cong',
    'C': 'coll',
    'O': 'cyclic',
    '^': 'eqangle',
}
_DSL_TOKEN_CHARS = set(
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789'
    ' _^:=,;'
)


def add_ag_repo_to_path(ag_repo: str) -> None:
  repo = Path(ag_repo).resolve()
  if not repo.exists():
    raise FileNotFoundError(f'AG repo not found: {repo}')
  sys.path.insert(0, str(repo))


def load_ag_modules() -> dict[str, Any]:
  import ddar  # pylint: disable=import-error,import-outside-toplevel
  import graph as gh  # pylint: disable=import-error,import-outside-toplevel
  import pretty as pt  # pylint: disable=import-error,import-outside-toplevel
  import problem as pr  # pylint: disable=import-error,import-outside-toplevel

  return {'ddar': ddar, 'gh': gh, 'pt': pt, 'pr': pr}


def event(path: str | None, **payload: Any) -> None:
  payload = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **payload}
  print(json.dumps(payload, ensure_ascii=False), flush=True)
  if path:
    with open(path, 'a', encoding='utf-8') as f:
      f.write(json.dumps(payload, ensure_ascii=False) + '\n')


def graph_stats(g: Any, gh: Any) -> dict[str, int]:
  """Small, stable progress counters for AG1 Graph."""
  cache_items = 0
  for value in getattr(g, 'cache', {}).values():
    try:
      cache_items += len(value)
    except TypeError:
      cache_items += 1
  return {
      'points': len(g.type2nodes.get(gh.Point, [])),
      'lines': len(g.type2nodes.get(gh.Line, [])),
      'circles': len(g.type2nodes.get(gh.Circle, [])),
      'segments': len(g.type2nodes.get(gh.Segment, [])),
      'cache_keys': len(getattr(g, 'cache', {})),
      'cache_items': cache_items,
  }


_FACT_PREDICATE_WEIGHTS = {
    'coll': 10.0,
    'cyclic': 10.0,
    'perp': 9.0,
    'para': 8.0,
    'cong': 7.0,
    'midp': 7.0,
    'circle': 6.0,
    'eqangle': 4.0,
    'eqangle6': 4.0,
    'eqratio': 3.5,
    'eqratio6': 3.5,
}


def _node_name(node: Any) -> str:
  return getattr(node, 'name', str(node))


def _goal_point_names(p: Any) -> set[str]:
  goal = getattr(p, 'goal', None)
  if goal is None:
    return set()
  return {str(arg) for arg in getattr(goal, 'args', []) if _POINT_RE.match(str(arg))}


def goal_point_names(p: Any) -> set[str]:
  return _goal_point_names(p)


def dependency_fact_text(dep: Any) -> str | None:
  name = getattr(dep, 'name', None)
  if name not in _FACT_PREDICATE_WEIGHTS:
    return None
  args = [_node_name(arg) for arg in getattr(dep, 'args', [])]
  if not args or any(not _POINT_RE.match(arg) for arg in args):
    return None
  return ' '.join([name] + args)


def select_ddar_facts(
    added: list[Any],
    p: Any,
    max_facts: int,
    recent_points: set[str] | None = None,
) -> list[str]:
  """Select a compact, goal-biased text summary of DDAR-added facts."""
  if max_facts <= 0:
    return []
  goal_points = _goal_point_names(p)
  recent_points = recent_points or set()
  ranked: list[tuple[float, int, str]] = []
  seen = set()
  for index, dep in enumerate(added):
    text = dependency_fact_text(dep)
    if not text or text in seen:
      continue
    seen.add(text)
    name = text.split(' ', 1)[0]
    args = text.split()[1:]
    score = _FACT_PREDICATE_WEIGHTS.get(name, 0.0)
    score += 4.0 * len(goal_points.intersection(args))
    score += 2.5 * len(recent_points.intersection(args))
    score -= 0.05 * len(args)
    level = getattr(dep, 'level', None)
    if isinstance(level, int):
      score -= 0.01 * level
    ranked.append((score, -index, text))
  ranked.sort(reverse=True)
  facts = []
  predicate_counts: dict[str, int] = {}
  for _, _, text in ranked:
    name = text.split(' ', 1)[0]
    if name in {'eqangle', 'eqangle6'}:
      limit = max(2, min(6, max_facts // 2))
    elif name in {'eqratio', 'eqratio6'}:
      limit = max(1, min(4, max_facts // 3))
    else:
      limit = max_facts
    if predicate_counts.get(name, 0) >= limit:
      continue
    facts.append(text)
    predicate_counts[name] = predicate_counts.get(name, 0) + 1
    if len(facts) >= max_facts:
      break
  return facts


def fact_context_text(facts: list[str] | None) -> str:
  if not facts:
    return ''
  return ' {D} ' + ' ; '.join(facts) + ' ;'


def build_lm_prompt(p: Any, definitions: Any, facts: list[str] | None = None) -> str:
  return p.setup_str_from_problem(definitions) + fact_context_text(facts) + ' {F1} x00'


def build_lm_prompt_from_problem_text(
    pstring: str,
    pr: Any,
    definitions: Any,
    facts: list[str] | None = None,
) -> str:
  p = pr.Problem.from_txt(pstring, translate=False)
  return build_lm_prompt(p, definitions, facts)


def existing_point_names(g: Any) -> set[str]:
  return {p.name for p in g.all_points()}


def candidate_new_point(text: str) -> str | None:
  """Return the constructed point name from raw LM or constructive text."""
  text = text.strip()
  if ' = ' in text:
    return text.split(' = ', 1)[0].strip()
  if ' : ' in text:
    return text.split(' : ', 1)[0].strip()
  return None


def candidate_passes_point_mask(
    text: str, forbidden_points: set[str] | None
) -> bool:
  if not forbidden_points:
    return True
  point = candidate_new_point(text)
  return point is None or point not in forbidden_points


def next_free_point_name(forbidden_points: set[str]) -> str | None:
  for point in 'abcdefghijklmnopqrstuvwxyz':
    if point not in forbidden_points:
      return point
  for point in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    if point not in forbidden_points:
      return point
  return None


def repair_candidate_point_name(
    text: str, forbidden_points: set[str] | None
) -> str:
  """Rename the candidate's newly constructed point if it already exists."""
  if not forbidden_points:
    return text
  old_point = candidate_new_point(text)
  if old_point not in forbidden_points:
    return text
  new_point = next_free_point_name(forbidden_points)
  if not new_point:
    return text
  return re.sub(rf'\b{re.escape(old_point)}\b', new_point, text)


def candidate_prompt_prefixes(
    strategy: str, forbidden_points: set[str] | None
) -> list[str]:
  if strategy == 'none':
    return ['']
  if strategy not in {
      'balanced_constrained',
      'mixed_constructive',
      'mixed_progress_constructive',
  }:
    raise ValueError(f'unknown candidate prompt sampling strategy: {strategy}')
  new_point = next_free_point_name(forbidden_points or set())
  if not new_point:
    return ['']
  prefixes = []
  for predicate in _CONSTRAINED_PROMPT_PREFIXES:
    if predicate == '':
      prefixes.append('')
    elif predicate == 'C':
      prefixes.append(f'{new_point} : C ')
    else:
      prefixes.append(f'{new_point} : {predicate} {new_point} ')
  if strategy in {'mixed_constructive', 'mixed_progress_constructive'}:
    prefixes.extend([
        f'{new_point} = on_line {new_point} ',
        f'{new_point} = on_circum {new_point} ',
        f'{new_point} = on_pline {new_point} ',
        f'{new_point} = on_tline {new_point} ',
        f'{new_point} = on_circle {new_point} ',
        f'{new_point} = eqangle3 ',
    ])
  if strategy == 'mixed_progress_constructive':
    prefixes.extend([
        f'{new_point} = on_bline {new_point} ',
        f'{new_point} = on_dia {new_point} ',
        f'{new_point} = eqdistance {new_point} ',
    ])
  return prefixes


def normalize_generated_candidate(text: str) -> str:
  text = text.strip()
  if ';' in text:
    text = text[: text.index(';') + 1]
  return ' '.join(text.split())


def _arity_ok(count: int, arity: int | tuple[int, ...]) -> bool:
  if isinstance(arity, tuple):
    return count in arity
  return count == arity


def candidate_dsl_shape_error(text: str) -> str | None:
  """Cheap structural DSL gate before AG graph validation.

  This is intentionally a syntax/shape filter, not a geometry checker.  AG's
  parser and numeric construction still decide whether a candidate is legal.
  """
  text = text.strip()
  if not text:
    return 'empty_candidate'
  if ' = ' in text:
    point, rhs = text.rstrip(';').split(' = ', 1)
    point = point.strip()
    if not _POINT_RE.match(point):
      return 'invalid_constructed_point'
    parts = [part.strip() for part in rhs.split(',') if part.strip()]
    if not parts or len(parts) > 2:
      return 'invalid_construction_count'
    binds_output = False
    seen_parts: set[tuple[str, ...]] = set()
    for part in parts:
      toks = part.split()
      if len(toks) < 2:
        return 'empty_construction'
      name, args = toks[0], toks[1:]
      signature = tuple([name] + args)
      if signature in seen_parts:
        return 'duplicate_construction'
      seen_parts.add(signature)
      if not _DSL_WORD_RE.match(name):
        return 'invalid_construction_name'
      arity = _CONSTRUCTIVE_ARG_ARITY.get(name)
      if arity is not None and not _arity_ok(len(args), arity):
        return 'invalid_construction_arity'
      if any(not _POINT_RE.match(arg) for arg in args):
        return 'invalid_construction_arg'
      if not _construction_args_shape_ok(point, name, args):
        return 'invalid_construction_args'
      binds_output = binds_output or _construction_binds_output(point, name, args)
    if len(parts) > 1 and not binds_output:
      return 'multi_construction_without_output_binding'
    return None

  if not text.endswith(';'):
    return 'missing_semicolon'
  if ' : ' not in text:
    return 'missing_constrained_separator'
  point, prem_str = text[:-1].split(' : ', 1)
  point = point.strip()
  if not _POINT_RE.match(point):
    return 'invalid_constructed_point'
  prem_toks = prem_str.split()
  if not prem_toks:
    return 'empty_predicate'
  prems = [[]]
  for i, tok in enumerate(prem_toks):
    if tok.isdigit():
      if i < len(prem_toks) - 1:
        prems.append([])
    else:
      prems[-1].append(tok)
  if not prems or len(prems) > 2:
    return 'invalid_predicate_count'
  seen_prems: set[tuple[str, ...]] = set()
  for prem in prems:
    if not prem:
      return 'empty_predicate'
    name, args = prem[0], prem[1:]
    signature = tuple([name] + args)
    if signature in seen_prems:
      return 'duplicate_predicate'
    seen_prems.add(signature)
    if name not in _CONSTRAINED_PREDICATE_ARITY:
      return 'unknown_predicate'
    if not _arity_ok(len(args), _CONSTRAINED_PREDICATE_ARITY[name]):
      return 'invalid_predicate_arity'
    if point not in args:
      return 'constructed_point_not_in_predicate'
    if any(not _POINT_RE.match(arg) for arg in args):
      return 'invalid_predicate_arg'
    if not _predicate_args_shape_ok(point, name, args):
      return 'invalid_predicate_args'
  return None


def candidate_passes_dsl_filter(text: str) -> bool:
  return candidate_dsl_shape_error(text) is None


def _arity_min_max(arity: int | tuple[int, ...]) -> tuple[int, int]:
  if isinstance(arity, tuple):
    return min(arity), max(arity)
  return arity, arity


def _arity_accepts(count: int, arity: int | tuple[int, ...]) -> bool:
  return count in arity if isinstance(arity, tuple) else count == arity


def _predicate_args_shape_ok(point: str, name: str, args: list[str]) -> bool:
  mapped_name = _CONSTRAINED_PREDICATE_ALIASES.get(name, name)
  if not check_valid_args(mapped_name, args):
    return False
  if name not in {'^', 'eqangle'}:
    return True
  try:
    construction_name, construction_args = translate_constrained_to_constructive(
        point, name, args
    )
  except Exception:  # pylint: disable=broad-except
    return False
  return construction_name != 'on_aline' or construction_args.count(point) <= 1


def _construction_args_shape_ok(point: str, name: str, args: list[str]) -> bool:
  if name in _CONSTRUCTIVE_REQUIRES_OUTPUT_FIRST_ARG:
    if not args or args[0] != point or point in args[1:]:
      return False
  if name == 'eqangle3':
    if point in args or len(args) != 5:
      return False
    a, b, d, e, f = args
    return a != b and len({d, e, f}) == 3
  other = args[1:] if name in _CONSTRUCTIVE_REQUIRES_OUTPUT_FIRST_ARG else args
  if name in {'on_line', 'on_circle', 'on_bline', 'on_dia'}:
    return len(other) == 2 and other[0] != other[1]
  if name in {'on_pline', 'on_tline'}:
    return len(other) == 3 and other[1] != other[2]
  if name == 'on_circum':
    return len(other) == 3 and len(set(other)) == 3
  if name == 'eqdistance':
    return len(other) == 3 and other[0] != other[1]
  return True


def _construction_binds_output(point: str, name: str, args: list[str]) -> bool:
  return name in _CONSTRUCTIVE_REQUIRES_OUTPUT_FIRST_ARG and bool(args) and args[0] == point


def _dsl_tokenize_prefix(text: str) -> list[str]:
  return re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+|\^|[:=,;]|\S', text)


def _word_matches_expected(token: str, expected: set[str], allow_prefix: bool) -> bool:
  if token in expected:
    return True
  return allow_prefix and any(word.startswith(token) for word in expected)


def _parse_constrained_prefix(
    point: str,
    tokens: list[str],
    has_semicolon: bool,
    known_points: set[str] | None = None,
) -> str:
  if not tokens:
    return 'possible'
  prem_count = 1
  seen_prems: set[tuple[str, ...]] = set()
  index = 0
  while index < len(tokens):
    if prem_count > 2:
      return 'invalid'
    pred = tokens[index]
    is_last = index == len(tokens) - 1
    if not _word_matches_expected(
        pred, set(_CONSTRAINED_PREDICATE_ARITY), allow_prefix=not has_semicolon and is_last
    ):
      return 'invalid'
    if pred not in _CONSTRAINED_PREDICATE_ARITY:
      return 'possible'
    arity = _CONSTRAINED_PREDICATE_ARITY[pred]
    min_arity, max_arity = _arity_min_max(arity)
    args = []
    index += 1
    while index < len(tokens):
      tok = tokens[index]
      if tok == ';':
        break
      if tok.isdigit():
        if len(args) < min_arity or not _arity_accepts(len(args), arity):
          return 'invalid'
        if point not in args:
          return 'invalid'
        if not _predicate_args_shape_ok(point, pred, args):
          return 'invalid'
        signature = tuple([pred] + args)
        if signature in seen_prems:
          return 'invalid'
        seen_prems.add(signature)
        index += 1
        if index >= len(tokens):
          return 'possible'
        if tokens[index] == ';':
          return 'complete' if has_semicolon else 'invalid'
        prem_count += 1
        break
      if not _POINT_RE.match(tok):
        return 'invalid'
      if known_points is not None and tok != point and tok not in known_points:
        return 'invalid'
      args.append(tok)
      if len(args) > max_arity:
        return 'invalid'
      index += 1
    else:
      if has_semicolon:
        if not _arity_accepts(len(args), arity) or point not in args:
          return 'invalid'
        if not _predicate_args_shape_ok(point, pred, args):
          return 'invalid'
        signature = tuple([pred] + args)
        if signature in seen_prems:
          return 'invalid'
        return 'complete'
      return 'possible'

    if index < len(tokens) and tokens[index] == ';':
      if not _arity_accepts(len(args), arity) or point not in args:
        return 'invalid'
      if not _predicate_args_shape_ok(point, pred, args):
        return 'invalid'
      signature = tuple([pred] + args)
      if signature in seen_prems:
        return 'invalid'
      return 'complete'
  return 'possible'


def _parse_constructive_prefix(
    point: str,
    tokens: list[str],
    has_semicolon: bool,
    known_points: set[str] | None = None,
) -> str:
  if not tokens:
    return 'possible'
  construction_count = 1
  binds_output = False
  seen_constructions: set[tuple[str, ...]] = set()
  index = 0
  while index < len(tokens):
    if construction_count > 2:
      return 'invalid'
    name = tokens[index]
    is_last = index == len(tokens) - 1
    if not _word_matches_expected(
        name, set(_CONSTRUCTIVE_ARG_ARITY), allow_prefix=not has_semicolon and is_last
    ):
      return 'invalid'
    if name not in _CONSTRUCTIVE_ARG_ARITY:
      return 'possible'
    arity = _CONSTRUCTIVE_ARG_ARITY[name]
    min_arity, max_arity = _arity_min_max(arity)
    args = []
    index += 1
    while index < len(tokens):
      tok = tokens[index]
      if tok in {',', ';'}:
        break
      if not _POINT_RE.match(tok):
        return 'invalid'
      if known_points is not None:
        if tok != point and tok not in known_points:
          return 'invalid'
        if name == 'eqangle3' and tok == point:
          return 'invalid'
        if name in _CONSTRUCTIVE_REQUIRES_OUTPUT_FIRST_ARG:
          if not args and tok != point:
            return 'invalid'
          if args and tok == point:
            return 'invalid'
      args.append(tok)
      if len(args) > max_arity:
        return 'invalid'
      index += 1
    if index >= len(tokens):
      if has_semicolon:
        if not _arity_accepts(len(args), arity):
          return 'invalid'
        current_binds_output = _construction_binds_output(point, name, args)
        if construction_count > 1 and not (binds_output or current_binds_output):
          return 'invalid'
        signature = tuple([name] + args)
        if signature in seen_constructions:
          return 'invalid'
        return (
            'complete'
            if _construction_args_shape_ok(point, name, args)
            else 'invalid'
        )
      return 'possible'
    sep = tokens[index]
    if sep == ',':
      if len(args) < min_arity or not _arity_accepts(len(args), arity):
        return 'invalid'
      if not _construction_args_shape_ok(point, name, args):
        return 'invalid'
      signature = tuple([name] + args)
      if signature in seen_constructions:
        return 'invalid'
      seen_constructions.add(signature)
      binds_output = binds_output or _construction_binds_output(point, name, args)
      construction_count += 1
      index += 1
      if index >= len(tokens):
        return 'possible'
      continue
    if sep == ';':
      if not _arity_accepts(len(args), arity):
        return 'invalid'
      current_binds_output = _construction_binds_output(point, name, args)
      if construction_count > 1 and not (binds_output or current_binds_output):
        return 'invalid'
      signature = tuple([name] + args)
      if signature in seen_constructions:
        return 'invalid'
      return (
          'complete' if _construction_args_shape_ok(point, name, args) else 'invalid'
      )
    return 'invalid'
  return 'possible'


def candidate_dsl_prefix_status(
    text: str, known_points: set[str] | None = None
) -> str:
  """Return possible/complete/invalid for a raw candidate generation prefix."""
  text = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
  if any(ch not in _DSL_TOKEN_CHARS and not ch.isspace() for ch in text):
    return 'invalid'
  if ';' in text:
    head, tail = text.split(';', 1)
    if tail.strip():
      return 'invalid'
    text = head + ';'
  tokens = _dsl_tokenize_prefix(text)
  if not tokens:
    return 'possible'
  if tokens.count(';') > 1:
    return 'invalid'
  has_semicolon = bool(tokens and tokens[-1] == ';')
  if ';' in tokens and not has_semicolon:
    return 'invalid'
  point = tokens[0]
  if not _POINT_RE.match(point):
    return 'invalid'
  if known_points is not None and point in known_points:
    return 'invalid'
  if len(tokens) == 1:
    return 'possible'
  sep = tokens[1]
  if sep not in {':', '='}:
    if len(tokens) == 2 and not has_semicolon:
      return 'possible' if any(s.startswith(sep) for s in {':', '='}) else 'invalid'
    return 'invalid'
  body = tokens[2:]
  if has_semicolon:
    body = body[:-1] + [';']
  if sep == ':':
    return _parse_constrained_prefix(point, body, has_semicolon, known_points)
  return _parse_constructive_prefix(point, body, has_semicolon, known_points)


def dsl_to_constructive_candidate(text: str) -> str:
  """Translate DSL candidate text to a constructive clause without AG validation."""
  text = normalize_generated_candidate(text)
  clause = text.rstrip(';').strip()
  if ' = ' in clause:
    return clause
  if ' : ' not in text or not text.endswith(';'):
    raise ValueError('candidate is not a complete constrained DSL clause')
  point, prem_str = text.split(' : ', 1)
  point = point.strip()
  if not _POINT_RE.match(point):
    raise ValueError(f'invalid point name {point}')
  prem_toks = prem_str.split()[:-1]
  prems: list[list[str]] = [[]]
  for i, tok in enumerate(prem_toks):
    if tok.isdigit():
      if i < len(prem_toks) - 1:
        prems.append([])
    else:
      prems[-1].append(tok)
  if len(prems) > 2:
    raise ValueError('there cannot be more than two predicates')
  constructions = []
  for prem in prems:
    if not prem:
      continue
    name, args = prem[0], prem[1:]
    construction_name, construction_args = translate_constrained_to_constructive(
        point, name, args
    )
    constructions.append(construction_name + ' ' + ' '.join(construction_args))
  if not constructions:
    raise ValueError('empty constrained DSL clause')
  return point + ' = ' + ', '.join(constructions)


def candidate_generation_dedup_key(text: str) -> str:
  """Cheap canonical key for de-duplicating candidates before AG validation."""
  text = normalize_generated_candidate(text)
  try:
    return canonical_aux_key(dsl_to_constructive_candidate(text))
  except Exception:  # pylint: disable=broad-except
    return text


def template_backfill_candidates(
    point_names: set[str] | None,
    max_candidates: int,
    excluded_canonical_keys: set[str] | None = None,
    preferred_points: set[str] | None = None,
) -> list[str]:
  """Generate type-diverse DSL candidates from simple construction templates."""
  if not point_names or max_candidates <= 0:
    return []
  new_point = next_free_point_name(point_names)
  preferred_points = {
      point for point in (preferred_points or set()) if point in point_names
  }
  pts = sorted(point_names, key=lambda point: (point not in preferred_points, point))
  if not new_point or len(pts) < 3:
    return []
  buckets: list[list[str]] = [[] for _ in range(30)]
  seen_canonical: set[str] = set()

  def prefer_key(item: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
    return (-sum(point in preferred_points for point in item), item)

  def add(bucket: int, text: str) -> None:
    key = text
    try:
      translated = dsl_to_constructive_candidate(text)
      key = canonical_aux_key(translated)
    except Exception:  # pylint: disable=broad-except
      pass
    if excluded_canonical_keys and key in excluded_canonical_keys:
      return
    if key not in seen_canonical and text not in buckets[bucket]:
      seen_canonical.add(key)
      buckets[bucket].append(text)

  def spread(items: list[tuple[str, ...]], limit: int) -> list[tuple[str, ...]]:
    if len(items) <= limit:
      return items
    step = max(1, len(items) // limit)
    out = []
    seen = set()
    for offset in range(step):
      for item in items[offset::step]:
        if item in seen:
          continue
        out.append(item)
        seen.add(item)
        if len(out) >= limit:
          return out
    return out

  pairs = [
      (pts[i], pts[j])
      for i in range(len(pts))
      for j in range(i + 1, len(pts))
  ]
  pairs.sort(key=prefer_key)
  selected_pairs = spread(pairs, 24)
  for a, b in selected_pairs[:12]:
    add(0, f'{new_point} = on_line {new_point} {a} {b};')
  disjoint_pair_sets = []
  for i, (a, b) in enumerate(selected_pairs):
    for c, d in selected_pairs[i + 1 :]:
      if {a, b}.isdisjoint({c, d}):
        disjoint_pair_sets.append((a, b, c, d))
  for a, b, c, d in spread(disjoint_pair_sets, 24):
    add(1, f'{new_point} : C {a} {b} {new_point} 00 C {c} {d} {new_point} 01 ;')
  triples = [
      (pts[i], pts[j], pts[k])
      for i in range(len(pts))
      for j in range(i + 1, len(pts))
      for k in range(j + 1, len(pts))
  ]
  triples.sort(key=prefer_key)
  selected_triples = spread(triples, 24)
  for a, b, c in selected_triples[:12]:
    add(2, f'{new_point} : O {new_point} {a} {b} {c} 00 ;')
  for a, b, c in selected_triples[:12]:
    add(3, f'{new_point} : P {new_point} {a} {b} {c} 00 ;')
    add(4, f'{new_point} : T {new_point} {a} {b} {c} 00 ;')
    add(5, f'{new_point} : D {new_point} {a} {b} {c} 00 ;')
  for a, b in selected_pairs[:12]:
    add(6, f'{new_point} : D {new_point} {a} {b} {a} 00 ;')
    add(7, f'{new_point} : D {new_point} {a} {new_point} {b} 00 ;')
    add(8, f'{new_point} : D {new_point} {a} {new_point} {b} 00 C {new_point} {a} {b} 01 ;')
    add(9, f'{new_point} : T {new_point} {a} {new_point} {b} 00 ;')
  for a, b, c in selected_triples[:12]:
    add(10, f'{new_point} = angle_bisector {new_point} {a} {b} {c};')
    add(11, f'{new_point} = angle_mirror {new_point} {a} {b} {c};')
  for a, b, c, d in spread(disjoint_pair_sets, 12):
    add(12, f'{new_point} : T {new_point} {a} {new_point} {b} 00 C {new_point} {c} {d} 01 ;')
  pair_triples = []
  seen_pair_triples = set()
  for a, b in selected_pairs:
    for c, d, e in selected_triples:
      item = (a, b, c, d, e)
      if item in seen_pair_triples:
        continue
      if len({a, b, c, d, e}) < 4:
        continue
      pair_triples.append(item)
      seen_pair_triples.add(item)
      if len(pair_triples) >= 24:
        break
    if len(pair_triples) >= 24:
      break
  for a, b, c, d, e in pair_triples[:12]:
    add(16, f'{new_point} : D {new_point} {a} {b} {a} 00 C {new_point} {c} {d} 01 ;')
    add(17, f'{new_point} : O {new_point} {c} {d} {e} 00 C {new_point} {a} {b} 01 ;')
    add(20, f'{new_point} : D {new_point} {a} {b} {a} 00 T {new_point} {c} {d} {e} 01 ;')
    add(21, f'{new_point} : D {new_point} {c} {d} {e} 00 C {new_point} {a} {b} 01 ;')
  disjoint_triples = []
  for i, (a, b, c) in enumerate(selected_triples):
    for d, e, f in selected_triples[i + 1 :]:
      if {a, b, c}.isdisjoint({d, e, f}):
        disjoint_triples.append((a, b, c, d, e, f))
  for a, b, c, d, e, f in spread(disjoint_triples, 12):
    add(18, f'{new_point} : O {new_point} {a} {b} {c} 00 O {new_point} {d} {e} {f} 01 ;')
    add(19, f'{new_point} : T {new_point} {a} {b} {c} 00 T {new_point} {d} {e} {f} 01 ;')
    add(22, f'{new_point} : O {new_point} {a} {b} {c} 00 T {new_point} {d} {e} {f} 01 ;')
    add(23, f'{new_point} : D {new_point} {a} {b} {c} 00 T {new_point} {d} {e} {f} 01 ;')
  selected_quintuples = []
  seen_quintuples = set()
  for a, b in selected_pairs:
    for c, d, e in selected_triples:
      if len({a, b, c, d, e}) != 5:
        continue
      item = (a, b, c, d, e)
      if item in seen_quintuples:
        continue
      selected_quintuples.append(item)
      seen_quintuples.add(item)
      if len(selected_quintuples) >= 24:
        break
    if len(selected_quintuples) >= 24:
      break
  for a, b, c, d, e in selected_quintuples[:12]:
    add(13, f'{new_point} = on_aline {new_point} {a} {b} {c} {d} {e};')
    add(14, f'{new_point} = on_aline2 {new_point} {a} {b} {c} {d} {e};')
    add(15, f'{new_point} = eqangle3 {a} {b} {c} {d} {e};')
    add(24, f'{new_point} : D {new_point} {a} {new_point} {b} 00 D {new_point} {c} {d} {c} 01 ;')
    add(25, f'{new_point} : D {new_point} {a} {new_point} {b} 00 D {new_point} {c} {new_point} {d} 01 ;')
    add(26, f'{new_point} : D {new_point} {a} {b} {c} 00 D {new_point} {d} {e} {d} 01 ;')
    add(27, f'{new_point} : T {new_point} {a} {new_point} {b} 00 T {new_point} {c} {d} {e} 01 ;')
    add(28, f'{new_point} : D {new_point} {a} {new_point} {b} 00 T {new_point} {c} {d} {e} 01 ;')

  candidates: list[str] = []
  positions = [0 for _ in buckets]
  while len(candidates) < max_candidates:
    progressed = False
    for bucket_idx, bucket in enumerate(buckets):
      pos = positions[bucket_idx]
      if pos < len(bucket):
        candidates.append(bucket[pos])
        positions[bucket_idx] += 1
        progressed = True
        if len(candidates) >= max_candidates:
          break
    if not progressed:
      break
  return candidates


def canonical_aux_key(auxstring: str) -> str:
  """Canonical key for de-duplicating equivalent auxiliary clauses."""
  clause = auxstring.strip()
  if clause.endswith(';'):
    clause = clause[:-1].strip()
  if ' = ' not in clause:
    return clause
  point, rhs = clause.split(' = ', 1)
  point = point.strip()

  def norm_construction(construction: str) -> str:
    toks = construction.strip().split()
    if not toks:
      return ''
    name, args = toks[0], toks[1:]
    args = ['@' if arg == point else arg for arg in args]
    if name in {'on_line', 'on_bline'} and len(args) >= 3 and args[0] == '@':
      args = [args[0], *sorted(args[1:])]
    elif name == 'on_circum' and len(args) > 2 and args[0] == '@':
      args = [args[0], *sorted(args[1:])]
    elif name in {'on_pline', 'on_tline'} and len(args) == 4 and args[0] == '@':
      args = [
          args[0],
          args[1],
          *sorted(args[2:4]),
      ]
    return name + ' ' + ' '.join(args)

  parts = [norm_construction(part) for part in rhs.split(',') if part.strip()]
  return '@ = ' + ' | '.join(sorted(parts))


def construction_type_key(auxstring: str) -> str:
  clause = auxstring.strip().rstrip(';')
  if ' = ' not in clause:
    return 'unknown'
  _, rhs = clause.split(' = ', 1)
  names = []
  for construction in rhs.split(','):
    toks = construction.strip().split()
    if toks:
      names.append(toks[0])
  if not names:
    return 'unknown'
  return '+'.join(sorted(names))


def candidate_rerank_score(auxstring: str) -> int:
  type_key = construction_type_key(auxstring)
  names = [name for name in type_key.split('+') if name and name != 'unknown']
  score = 0
  if len(names) > 1:
    score += 2
  if any(name in _HIGH_VALUE_CONSTRUCTION_TYPES for name in names):
    score += 1
  if names and set(names) <= {'on_line'}:
    score -= 1
  return score


def interleave_ranked_records_by_node(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Keep depth-level candidate evaluation from collapsing onto one beam node."""
  if len(records) <= 1 or not any('node_index' in record for record in records):
    return records
  buckets: dict[int, list[dict[str, Any]]] = {}
  node_scores: dict[int, float] = {}
  missing_node: list[dict[str, Any]] = []
  for rank, record in enumerate(records):
    node_index = record.get('node_index')
    if not isinstance(node_index, int):
      missing_node.append(record)
      continue
    buckets.setdefault(node_index, []).append(record)
    score = float(record.get('_candidate_rerank_score', 0.0))
    # Tiny rank tie-break keeps the previous global rerank order stable.
    score -= 1e-6 * rank
    node_scores[node_index] = max(node_scores.get(node_index, -10.0), score)
  if len(buckets) <= 1:
    return records
  ordered_nodes = sorted(
      buckets,
      key=lambda node: (-node_scores[node], node),
  )
  interleaved: list[dict[str, Any]] = []
  while len(interleaved) < len(records) - len(missing_node):
    progressed = False
    for node in ordered_nodes:
      if buckets[node]:
        interleaved.append(buckets[node].pop(0))
        progressed = True
    if not progressed:
      break
  return interleaved + missing_node


def type_signal_coverage_records(
    records: list[dict[str, Any]],
    type_bonus: dict[str, float],
    bonus_field: str,
    bucket_score_field: str,
) -> list[dict[str, Any]]:
  """Order candidates from construction families with external signal."""
  def frontfill_score(record: dict[str, Any]) -> float:
    try:
      return float(record.get('_candidate_frontfill_score', 0.0))
    except (TypeError, ValueError):
      return 0.0

  buckets: dict[str, list[dict[str, Any]]] = {}
  bucket_scores: dict[str, float] = {}
  for record in records:
    key = construction_type_key(record['translation'])
    bonus = type_bonus.get(key)
    if bonus is None:
      continue
    buckets.setdefault(key, []).append(record)
    bucket_scores[key] = max(
        bucket_scores.get(key, float('-inf')),
        frontfill_score(record) + bonus,
    )
  for bucket in buckets.values():
    bucket.sort(
        key=frontfill_score,
        reverse=True,
    )
  ordered_keys = sorted(
      buckets,
      key=lambda key: (
          -bucket_scores[key],
          -type_bonus[key],
          key,
      ),
  )
  reranked: list[dict[str, Any]] = []
  total_records = sum(len(bucket) for bucket in buckets.values())
  while len(reranked) < total_records:
    progressed = False
    for key in ordered_keys:
      if buckets[key]:
        record = buckets[key].pop(0)
        record[bonus_field] = type_bonus[key]
        record[bucket_score_field] = bucket_scores[key]
        reranked.append(record)
        progressed = True
    if not progressed:
      break
  return interleave_ranked_records_by_node(reranked)


def progress_type_coverage_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Order candidates from progress-positive construction families."""
  return type_signal_coverage_records(
      records,
      _PROGRESS_SIGNAL_TYPE_BONUS,
      '_candidate_progress_type_bonus',
      '_candidate_progress_type_bucket_score',
  )


def signal_anchor_coverage_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Order candidates from solved/SFT-positive construction families."""
  return type_signal_coverage_records(
      records,
      _SIGNAL_ANCHOR_TYPE_BONUS,
      '_candidate_signal_anchor_type_bonus',
      '_candidate_signal_anchor_type_bucket_score',
  )


def rerank_candidate_records(
    records: list[dict[str, Any]],
    strategy: str,
    value_model: dict[str, Any] | None = None,
    secondary_value_model: dict[str, Any] | None = None,
    frontfill_limit: int = 8,
) -> list[dict[str, Any]]:
  """Optionally interleave translated candidates by construction type."""
  if strategy == 'none' or len(records) <= 1:
    return records
  if strategy == 'value_model':
    if value_model is None:
      raise ValueError('--candidate_value_model is required for value_model rerank')
    for record in records:
      record['_candidate_rerank_score'] = score_candidate_value_model(
          value_model, record
      )
    return sorted(
        records,
        key=lambda record: record['_candidate_rerank_score'],
        reverse=True,
    )
  if strategy in {
      'value_model_frontfill_diverse',
      'value_model_frontfill_progress_diverse',
  }:
    if value_model is None or secondary_value_model is None:
      raise ValueError(
          '--candidate_value_model and --candidate_secondary_value_model are '
          f'required for {strategy} rerank'
      )
    front_records = rerank_candidate_records(
        records,
        'value_model_diverse',
        value_model,
    )
    for record in records:
      record['_candidate_frontfill_score'] = record.get('_candidate_rerank_score')
    coverage_records = rerank_candidate_records(
        records,
        'value_model_diverse',
        secondary_value_model,
    )
    for record in records:
      record['_candidate_coverage_score'] = record.get('_candidate_rerank_score')
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    limit = max(0, min(frontfill_limit, len(records)))
    for record in front_records[:limit]:
      selected.append(record)
      selected_ids.add(id(record))
      record['_candidate_rerank_score'] = record.get('_candidate_frontfill_score')
      record['_candidate_rerank_phase'] = 'frontfill'
    if strategy == 'value_model_frontfill_progress_diverse':
      signal_anchor_limit = max(0, min(frontfill_limit, len(records)))
      signal_anchor_count = 0
      for record in signal_anchor_coverage_records(records):
        if id(record) in selected_ids:
          continue
        selected.append(record)
        selected_ids.add(id(record))
        record['_candidate_rerank_score'] = record.get('_candidate_frontfill_score')
        record['_candidate_rerank_phase'] = 'signal_anchor_coverage'
        signal_anchor_count += 1
        if signal_anchor_count >= signal_anchor_limit:
          break
      for record in progress_type_coverage_records(records):
        if id(record) in selected_ids:
          continue
        selected.append(record)
        selected_ids.add(id(record))
        record['_candidate_rerank_score'] = record.get('_candidate_frontfill_score')
        record['_candidate_rerank_phase'] = 'progress_type_coverage'
    for record in coverage_records:
      if id(record) in selected_ids:
        continue
      selected.append(record)
      selected_ids.add(id(record))
      record['_candidate_rerank_score'] = record.get('_candidate_coverage_score')
      record['_candidate_rerank_phase'] = 'coverage'
    return selected
  if strategy == 'value_model_diverse':
    if value_model is None:
      raise ValueError(
          '--candidate_value_model is required for value_model_diverse rerank'
      )
  elif strategy != 'heuristic_diverse':
    raise ValueError(f'unknown candidate rerank strategy: {strategy}')
  buckets: dict[str, list[dict[str, Any]]] = {}
  bucket_scores: dict[str, float] = {}
  for record in records:
    key = construction_type_key(record['translation'])
    buckets.setdefault(key, []).append(record)
    if strategy == 'value_model_diverse':
      score = score_candidate_value_model(value_model, record)
    else:
      score = candidate_rerank_score(record['translation'])
    record['_candidate_rerank_score'] = score
    bucket_scores[key] = max(bucket_scores.get(key, -10.0), score)
  if strategy == 'value_model_diverse':
    for bucket in buckets.values():
      bucket.sort(key=lambda record: record['_candidate_rerank_score'], reverse=True)
  ordered_keys = sorted(bucket_scores, key=lambda key: (-bucket_scores[key], key))
  reranked = []
  while len(reranked) < len(records):
    progressed = False
    for key in ordered_keys:
      if buckets[key]:
        reranked.append(buckets[key].pop(0))
        progressed = True
    if not progressed:
      break
  return interleave_ranked_records_by_node(reranked)


def normalize_candidate_value_feature(value: Any) -> str:
  text = str(value or '').strip().lower()
  return re.sub(r'[^a-z0-9_^]+', '_', text).strip('_') or 'none'


def candidate_value_error_key(text: str) -> str:
  lowered = str(text or '').lower()
  if 'already exists' in lowered:
    return 'point_already_exists'
  if 'pointtoocloseerror' in lowered:
    return 'point_too_close'
  if 'pointtoofarerror' in lowered:
    return 'point_too_far'
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


def add_candidate_value_feature(
    tokens: list[str], prefix: str, value: Any
) -> None:
  normalized = normalize_candidate_value_feature(value)
  if normalized != 'none':
    tokens.append(f'{prefix}={normalized}')


def add_candidate_problem_type_features(
    tokens: list[str], problem: Any, type_key: str
) -> None:
  problem_key = normalize_candidate_value_feature(problem)
  combo_key = normalize_candidate_value_feature(type_key)
  if problem_key == 'none' or combo_key == 'none':
    return
  tokens.append(f'problem_type_combo={problem_key}__{combo_key}')
  for name in type_key.split('+'):
    name_key = normalize_candidate_value_feature(name)
    if name_key != 'none' and name_key != 'unknown':
      tokens.append(f'problem_type={problem_key}__{name_key}')


def _candidate_value_tokens(record: dict[str, Any]) -> list[str]:
  raw = str(record.get('raw') or '')
  translation = str(record.get('translation') or '')
  type_key = construction_type_key(translation)
  text = ' '.join(
      str(value or '')
      for value in (
          raw,
          translation,
          type_key,
      )
  )
  tokens = []
  for token in re.findall(r'[A-Za-z_^]+|[0-9]+', text):
    tokens.append(token.lower())
  for name in type_key.split('+'):
    if name and name != 'unknown':
      tokens.append('type=' + name)
  add_candidate_value_feature(tokens, 'type_combo', type_key)
  add_candidate_problem_type_features(
      tokens, record.get('problem') or record.get('problem_name'), type_key
  )
  error = candidate_value_error_key(translation)
  add_candidate_value_feature(tokens, 'error', error)
  add_candidate_value_feature(tokens, 'source', record.get('source'))
  return tokens


def load_candidate_value_model(path: str | None) -> dict[str, Any] | None:
  if not path:
    return None
  with open(path, 'r', encoding='utf-8') as f:
    return json.load(f)


def score_candidate_value_model(
    model: dict[str, Any] | None, record: dict[str, Any]
) -> float:
  if model is None:
    return 0.0
  weights = model.get('weights', {})
  bias = float(model.get('bias', 0.0))
  score = bias
  for token in _candidate_value_tokens(record):
    score += float(weights.get(token, 0.0))
  score += _VERIFIER_PRIOR_TYPE_BONUS.get(
      construction_type_key(str(record.get('translation') or '')), 0.0
  )
  return score


def load_problem(pr: Any, problems_file: str, problem_name: str, translate: bool):
  problems = pr.Problem.from_txt_file(
      problems_file, to_dict=True, translate=translate
  )
  if problem_name not in problems:
    raise KeyError(f'{problem_name} not found in {problems_file}')
  return problems[problem_name]


def build_graph_for_symbolic_search(gh: Any, p: Any, definitions: Any):
  """Build an AG graph while tolerating numeric goal-check bugs.

  AG1's Graph.build_problem runs a numerical sanity check for the goal after
  building the construction graph. Some released-code predicates can raise in
  that numeric check even though symbolic DDAR can still reason about the goal.
  Search and benchmark evaluation should only require a valid construction
  graph; goal checking is performed symbolically by DDAR afterwards.
  """
  try:
    return gh.Graph.build_problem(p, definitions)
  except Exception:  # pylint: disable=broad-except
    p_no_goal = p.copy()
    p_no_goal.goal = None
    return gh.Graph.build_problem(p_no_goal, definitions)


def goal_is_solved(g: Any, p: Any) -> bool:
  if p.goal is None:
    return False
  goal_args = g.names2nodes(p.goal.args)
  return g.check(p.goal.name, goal_args)


def run_ddar_once(
    g: Any,
    p: Any,
    ddar: Any,
    gh: Any,
    max_level: int,
    timeout: int,
    events_file: str | None,
    tag: str,
    fact_context_top_k: int = 0,
    fact_context_recent_points: set[str] | None = None,
) -> dict[str, Any]:
  started = time.time()
  g, level_times, status, branches, added = ddar.solve(
      g, RULES, p, max_level=max_level, timeout=timeout
  )
  solved = goal_is_solved(g, p)
  result = {
      'tag': tag,
      'status': status,
      'solved': solved,
      'levels': len(level_times),
      'level_times': [round(x, 3) for x in level_times],
      'branches': branches,
      'added_dependencies': len(added),
      'elapsed_sec': round(time.time() - started, 3),
      **graph_stats(g, gh),
  }
  facts = select_ddar_facts(
      added, p, fact_context_top_k, fact_context_recent_points
  )
  if facts:
    result['fact_context'] = facts
    result['fact_context_count'] = len(facts)
  event(events_file, kind='ddar_done', **result)
  return result


def translate_constrained_to_constructive(
    point: str, name: str, args: list[str]
) -> tuple[str, list[str]]:
  """Copy of AG1's constrained-predicate to construction translator."""
  if name in ['T', 'perp']:
    a, b, c, d = args
    if point in [c, d]:
      a, b, c, d = c, d, a, b
    if point == b:
      a, b = b, a
    if point == d:
      c, d = d, c
    if a == c and a == point:
      return 'on_dia', [a, b, d]
    return 'on_tline', [a, b, c, d]

  if name in ['P', 'para']:
    a, b, c, d = args
    if point in [c, d]:
      a, b, c, d = c, d, a, b
    if point == b:
      a, b = b, a
    return 'on_pline', [a, b, c, d]

  if name in ['D', 'cong']:
    a, b, c, d = args
    if point in [c, d]:
      a, b, c, d = c, d, a, b
    if point == b:
      a, b = b, a
    if point == d:
      c, d = d, c
    if a == c and a == point:
      return 'on_bline', [a, b, d]
    if b in [c, d]:
      if b == d:
        c, d = d, c
      return 'on_circle', [a, b, d]
    return 'eqdistance', [a, b, c, d]

  if name in ['C', 'coll']:
    a, b, c = args
    if point == b:
      a, b = b, a
    if point == c:
      a, b, c = c, a, b
    return 'on_line', [a, b, c]

  if name in ['^', 'eqangle']:
    if len(args) == 8:
      a, b, c, d, e, f, g, h = args
      if a == c == e == g and d == point and f == point:
        return 'angle_bisector', [point, b, a, h]
      if a == c == e == g and h == point:
        return 'angle_mirror', [point, b, a, d]
      if a == point and c == point and e == g:
        return 'eqangle3', [b, d, e, f, h]
      if b == point and a == c and e == g:
        return 'on_aline', [point, a, d, f, e, h]
      if a == point and c == point and e == g:
        return 'on_aline2', [point, b, d, f, e, h]
      raise ValueError('unsupported 8-argument eqangle candidate')

    a, b, c, d, e, f = args
    if point in [d, e, f]:
      a, b, c, d, e, f = d, e, f, a, b, c
    x, b, y, c, d = b, c, e, d, f
    if point == b:
      a, b, c, d = b, a, d, c
    if point == d and x == y:
      return 'angle_bisector', [point, b, x, c]
    if point == x:
      return 'eqangle3', [a, b, y, c, d]
    return 'on_aline', [a, x, b, c, y, d]

  if name in ['cyclic', 'O']:
    a, b, c = [x for x in args if x != point]
    return 'on_circum', [point, a, b, c]

  return name, args


def check_valid_args(name: str, args: list[str]) -> bool:
  if name == 'perp':
    if len(args) != 4:
      return False
    a, b, c, d = args
    return len({a, b}) >= 2 and len({c, d}) >= 2
  if name == 'para':
    return len(args) == 4 and len(set(args)) >= 4
  if name == 'cong':
    if len(args) != 4:
      return False
    a, b, c, d = args
    return len({a, b}) >= 2 and len({c, d}) >= 2
  if name == 'coll':
    return len(args) == 3 and len(set(args)) >= 3
  if name == 'cyclic':
    return len(args) == 4 and len(set(args)) >= 4
  if name == 'eqangle':
    return len(args) in (6, 8)
  return True


def validate_constructive_clause(text: str, g: Any, pr: Any) -> str:
  clause_text = text.strip()
  if clause_text.endswith(';'):
    clause_text = clause_text[:-1].strip()
  if ' = ' not in clause_text:
    return 'ERROR: constructive clause must contain " = "'
  point = clause_text.split(' = ', 1)[0].strip()
  if not _POINT_RE.match(point):
    return f'ERROR: invalid point name {point}'
  if point in existing_point_names(g):
    return f'ERROR: point {point} already exists.'
  try:
    clause = pr.Clause.from_txt(clause_text)
    g.copy().add_clause(clause, 0, DEFINITIONS)
  except Exception:  # pylint: disable=broad-except
    return 'ERROR: ' + traceback.format_exc()
  return clause_text


def try_translate_candidate(text: str, g: Any, pr: Any, pt: Any) -> str:
  """Translate/validate either original LM format or direct construction."""
  text = text.strip()
  if not text:
    return 'ERROR: empty candidate'
  if ' = ' in text:
    return validate_constructive_clause(text, g, pr)
  if not text.endswith(';'):
    return 'ERROR: constrained candidate must end with ;'
  if ' : ' not in text:
    return 'ERROR: constrained candidate must contain " : "'

  head, prem_str = text.split(' : ', 1)
  point = head.strip()
  if not _POINT_RE.match(point):
    return f'ERROR: invalid point name {point}'

  existing_points = existing_point_names(g)
  if point in existing_points:
    return f'ERROR: point {point} already exists.'

  prem_toks = prem_str.split()[:-1]
  prems = [[]]
  for i, tok in enumerate(prem_toks):
    if tok.isdigit():
      if i < len(prem_toks) - 1:
        prems.append([])
    else:
      prems[-1].append(tok)
  if len(prems) > 2:
    return 'ERROR: there cannot be more than two predicates.'

  constructions = []
  for prem in prems:
    if not prem:
      return 'ERROR: empty predicate'
    name, *args = prem
    if point not in args:
      return f'ERROR: {point} not found in predicate args.'
    mapped_name = pt.map_symbol(name) if name in getattr(pt, 'MAP_SYMBOL', {}) else name
    if not check_valid_args(mapped_name, args):
      return 'ERROR: invalid predicate ' + name + ' ' + ' '.join(args)
    for a in args:
      if a != point and a not in existing_points:
        return f'ERROR: point {a} does not exist.'
    try:
      cname, cargs = translate_constrained_to_constructive(point, name, args)
    except Exception:  # pylint: disable=broad-except
      return 'ERROR: invalid predicate ' + name + ' ' + ' '.join(args)
    if cname == 'on_aline' and cargs.count(point) > 1:
      return f'ERROR: on_aline involves twice {point}'
    constructions.append(cname + ' ' + ' '.join(cargs))

  return validate_constructive_clause(point + ' = ' + ', '.join(constructions), g, pr)


def insert_aux_to_premise(pstring: str, auxstring: str) -> str:
  setup, goal = pstring.split(' ? ')
  return setup + '; ' + auxstring + ' ? ' + goal


class BeamQueue:
  def __init__(self, max_size: int):
    self.max_size = max_size
    self._items: list[tuple[float, int, Any]] = []
    self._seq = 0

  def add(self, node: Any, score: float) -> None:
    item = (score, self._seq, node)
    self._seq += 1
    if len(self._items) < self.max_size:
      heapq.heappush(self._items, item)
    elif score > self._items[0][0]:
      heapq.heapreplace(self._items, item)

  def ordered(self) -> list[tuple[float, Any]]:
    return [(s, n) for s, _, n in sorted(self._items, reverse=True)]

  def __len__(self) -> int:
    return len(self._items)


class QwenGenerator:
  def __init__(
      self,
      model_name_or_path: str,
      adapter_path: str | None,
      dtype: str,
      device_map: str,
  ):
    import torch  # pylint: disable=import-outside-toplevel
    from transformers import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=import-outside-toplevel

    torch_dtype = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32,
    }[dtype]
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
    mapped_device = self._parse_device_map(device_map)
    self.model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=mapped_device,
        trust_remote_code=True,
        attn_implementation='sdpa',
    )
    if adapter_path:
      from peft import PeftModel  # pylint: disable=import-outside-toplevel

      self.model = PeftModel.from_pretrained(self.model, adapter_path)
    self.model.eval()
    self.device = next(self.model.parameters()).device
    self._dsl_allowed_token_ids: list[int] | None = None
    self._dsl_allowed_token_text: dict[int, str] | None = None
    self._dsl_grammar_token_ids: list[int] | None = None

  @staticmethod
  def _parse_device_map(device_map: str) -> str | dict[str, int]:
    if device_map in {'auto', 'balanced', 'balanced_low_0', 'sequential'}:
      return device_map
    if device_map.startswith('cuda'):
      if ':' in device_map:
        return {'': int(device_map.split(':', 1)[1])}
      return {'': 0}
    if device_map.isdigit():
      return {'': int(device_map)}
    return device_map

  def _dsl_token_mask_processor_legacy(self):
    import torch  # pylint: disable=import-outside-toplevel
    from transformers import LogitsProcessor, LogitsProcessorList  # pylint: disable=import-outside-toplevel

    if self._dsl_allowed_token_ids is None:
      allowed = []
      vocab_size = len(self.tokenizer)
      for token_id in range(vocab_size):
        text = self.tokenizer.decode([token_id], skip_special_tokens=False)
        if token_id == self.tokenizer.eos_token_id:
          allowed.append(token_id)
          continue
        if text in getattr(self.tokenizer, 'all_special_tokens', []):
          continue
        text = text.replace('Ġ', ' ').replace('▁', ' ')
        if text and all(ch in _DSL_TOKEN_CHARS or ch.isspace() for ch in text):
          allowed.append(token_id)
      self._dsl_allowed_token_ids = allowed

    allowed_ids = torch.tensor(self._dsl_allowed_token_ids, device=self.device)

    class DslTokenMask(LogitsProcessor):
      def __call__(self, input_ids, scores):  # pylint: disable=unused-argument
        masked = scores.new_full(scores.shape, float('-inf'))
        masked.index_copy_(1, allowed_ids, scores.index_select(1, allowed_ids))
        return masked

    return LogitsProcessorList([DslTokenMask()])

  @staticmethod
  def _clean_decoded_token_text(text: str) -> str:
    return (
        text.replace('\u0120', ' ')
        .replace('\u2581', ' ')
        .replace('\u010a', '\n')
    )

  @staticmethod
  def _is_dsl_grammar_token_piece(text: str) -> bool:
    compact = ''.join(text.split())
    if not compact:
      return True
    if compact == '_':
      return True
    if any(ch not in _DSL_TOKEN_CHARS for ch in compact):
      return False
    if all(ch in ':=,;' for ch in compact):
      return len(compact) <= 3
    if compact.isdigit():
      return len(compact) <= 3
    if _POINT_RE.match(compact):
      return True
    lowered = compact.lower()
    return lowered in _DSL_KEYWORD_FRAGMENTS or lowered in _DSL_KEYWORDS

  def _ensure_dsl_allowed_tokens(self) -> None:
    if self._dsl_allowed_token_ids is not None:
      return
    allowed = []
    grammar_allowed = []
    token_text = {}
    vocab_size = len(self.tokenizer)
    for token_id in range(vocab_size):
      text = self.tokenizer.decode([token_id], skip_special_tokens=False)
      if token_id == self.tokenizer.eos_token_id:
        allowed.append(token_id)
        token_text[token_id] = ''
        continue
      if text in getattr(self.tokenizer, 'all_special_tokens', []):
        continue
      text = self._clean_decoded_token_text(text)
      if text and all(ch in _DSL_TOKEN_CHARS or ch.isspace() for ch in text):
        allowed.append(token_id)
        token_text[token_id] = text
        if self._is_dsl_grammar_token_piece(text):
          grammar_allowed.append(token_id)
    self._dsl_allowed_token_ids = allowed
    self._dsl_grammar_token_ids = grammar_allowed
    self._dsl_allowed_token_text = token_text

  def _dsl_token_mask_processor(
      self,
      grammar_constrained: bool = False,
      prompt_len: int | None = None,
      prefix: str = '',
      known_points: set[str] | None = None,
  ):
    import torch  # pylint: disable=import-outside-toplevel
    from transformers import LogitsProcessor, LogitsProcessorList  # pylint: disable=import-outside-toplevel

    self._ensure_dsl_allowed_tokens()
    active_token_ids = (
        self._dsl_grammar_token_ids
        if grammar_constrained
        else self._dsl_allowed_token_ids
    )
    allowed_ids = torch.tensor(active_token_ids, device=self.device)
    allowed_token_text = self._dsl_allowed_token_text or {}
    eos_id = self.tokenizer.eos_token_id

    class DslTokenMask(LogitsProcessor):
      def __call__(inner_self, input_ids, scores):  # pylint: disable=unused-argument
        masked = scores.new_full(scores.shape, float('-inf'))
        if not grammar_constrained or prompt_len is None:
          masked.index_copy_(1, allowed_ids, scores.index_select(1, allowed_ids))
          return masked
        for row_idx in range(scores.shape[0]):
          gen_ids = input_ids[row_idx, prompt_len:].tolist()
          suffix = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
          current = prefix + suffix
          status = candidate_dsl_prefix_status(current, known_points)
          if status == 'complete':
            masked[row_idx, eos_id] = scores[row_idx, eos_id]
            continue
          row_allowed = []
          for token_id in active_token_ids or []:
            if token_id == eos_id:
              continue
            token_piece = allowed_token_text.get(token_id, '')
            if (
                candidate_dsl_prefix_status(
                    current + token_piece, known_points
                )
                != 'invalid'
            ):
              row_allowed.append(token_id)
          if row_allowed:
            ids = torch.tensor(row_allowed, device=scores.device)
            masked[row_idx, ids] = scores[row_idx, ids]
          else:
            masked[row_idx, eos_id] = scores[row_idx, eos_id]
        return masked

    return LogitsProcessorList([DslTokenMask()])

  def generate(
      self,
      prompt: str,
      num_return_sequences: int,
      max_new_tokens: int,
      temperature: float,
      top_p: float,
      forbidden_point_names: set[str] | None = None,
      candidate_multiplier: int = 1,
      dsl_filter: bool = False,
      dsl_token_mask: bool = False,
      point_repair: bool = False,
      prompt_sampling: str = 'none',
  ) -> list[tuple[str, float]]:
    import torch  # pylint: disable=import-outside-toplevel

    # Keep inference formatting aligned with train_qwen_aux_lora.py, which
    # separates the AG state prompt from the target with a single newline.
    do_sample = temperature > 0
    requested_sequences = max(num_return_sequences, 1) * max(candidate_multiplier, 1)
    prefixes = candidate_prompt_prefixes(prompt_sampling, forbidden_point_names)
    sequences_per_prefix = max(
        1, (requested_sequences + len(prefixes) - 1) // len(prefixes)
    )
    candidates = []
    seen = set()
    for prefix in prefixes:
      model_prompt = prompt.rstrip() + '\n' + prefix
      inputs = self.tokenizer(model_prompt, return_tensors='pt').to(self.device)
      prompt_len = inputs['input_ids'].shape[1]
      logits_processor = (
          self._dsl_token_mask_processor(
              grammar_constrained=dsl_filter,
              prompt_len=prompt_len,
              prefix=prefix,
              known_points=forbidden_point_names if forbidden_point_names else None,
          )
          if dsl_token_mask
          else None
      )
      with torch.inference_mode():
        outputs = self.model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5) if do_sample else None,
            top_p=top_p if do_sample else None,
            num_return_sequences=sequences_per_prefix,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            logits_processor=logits_processor,
        )
      for seq in outputs.sequences:
        gen_ids = seq[prompt_len:]
        suffix = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        text = normalize_generated_candidate(prefix + suffix)
        if point_repair:
          text = repair_candidate_point_name(text, forbidden_point_names)
        if dsl_filter and not candidate_passes_dsl_filter(text):
          continue
        if (
            dsl_filter
            and forbidden_point_names
            and candidate_dsl_prefix_status(text, forbidden_point_names) == 'invalid'
        ):
          continue
        if not candidate_passes_point_mask(text, forbidden_point_names):
          continue
        dedup_key = candidate_generation_dedup_key(text)
        if text and dedup_key not in seen:
          candidates.append((text, 0.0))
          seen.add(dedup_key)
        if len(candidates) >= requested_sequences:
          return candidates
    return candidates


def run_qwen_search(args: argparse.Namespace) -> bool:
  ag = load_ag_modules()
  pr, gh, ddar, pt = ag['pr'], ag['gh'], ag['ddar'], ag['pt']

  p = load_problem(pr, args.problems_file, args.problem_name, args.translate)
  g, _ = build_graph_for_symbolic_search(gh, p, DEFINITIONS)
  event(args.events_file, kind='problem_loaded', name=args.problem_name, text=p.txt())

  first = run_ddar_once(
      g,
      p,
      ddar,
      gh,
      args.max_level,
      args.ddar_timeout,
      args.events_file,
      'root',
      args.lm_fact_context_top_k,
  )
  if first['solved']:
    return True
  if args.mode == 'ddar':
    return False

  generator = QwenGenerator(
      args.qwen_model, args.adapter_path, args.dtype, args.device_map
  )
  prompt0 = build_lm_prompt(
      p, DEFINITIONS, first.get('fact_context') if args.lm_fact_context_top_k else None
  )
  beam = BeamQueue(args.beam_size)
  beam.add((g, prompt0, p.txt()), 0.0)
  value_model = load_candidate_value_model(args.candidate_value_model)
  secondary_value_model = load_candidate_value_model(
      args.candidate_secondary_value_model
  )

  for depth in range(args.search_depth):
    event(args.events_file, kind='depth_start', depth=depth, nodes=len(beam))
    next_beam = BeamQueue(args.beam_size)
    if args.candidate_depth_eval_limit and args.candidate_depth_eval_limit > 0:
      depth_candidates = []
      for node_index, (prev_score, (g_cur, prompt, pstring)) in enumerate(
          beam.ordered()
      ):
        p_cur = pr.Problem.from_txt(pstring, translate=False)
        event(
            args.events_file,
            kind='decode_start',
            depth=depth,
            score=prev_score,
            prompt_tail=prompt[-240:],
        )
        forbidden_points = (
            existing_point_names(g_cur)
            if (
                args.candidate_point_mask
                or args.candidate_point_repair
                or args.candidate_prompt_sampling != 'none'
            )
            else None
        )
        candidates = generator.generate(
            prompt,
            args.num_return_sequences,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            forbidden_points,
            args.candidate_quality_multiplier,
            args.candidate_dsl_filter,
            args.candidate_dsl_token_mask,
            args.candidate_point_repair,
            args.candidate_prompt_sampling,
        )
        if args.candidate_template_backfill and len(candidates) < args.num_return_sequences:
          seen_raw = {raw for raw, _ in candidates}
          candidate_sources = {raw: 'lm' for raw, _ in candidates}
          seen_generation_keys = {
              candidate_generation_dedup_key(raw) for raw, _ in candidates
          }
          needed = args.num_return_sequences - len(candidates)
          for raw in template_backfill_candidates(
              forbidden_points,
              needed * 4,
              seen_generation_keys if args.candidate_canonical_dedup else None,
              goal_point_names(p_cur),
          ):
            generation_key = candidate_generation_dedup_key(raw)
            if raw not in seen_raw and generation_key not in seen_generation_keys:
              candidates.append((raw, 0.0))
              seen_raw.add(raw)
              candidate_sources[raw] = 'template_initial_backfill'
              seen_generation_keys.add(generation_key)
            if len(candidates) >= args.num_return_sequences:
              break
        else:
          candidate_sources = {raw: 'lm' for raw, _ in candidates}
        # Dedup within this proof state; pruned auxes may still help other branches.
        seen_candidate_keys: set[str] = set()
        translated_candidates = []
        for raw, lm_score in candidates:
          source = candidate_sources.get(raw, 'lm')
          translation = try_translate_candidate(raw, g_cur, pr, pt)
          event(
              args.events_file,
              kind='candidate',
              depth=depth,
              raw=raw,
              translation=translation,
              lm_score=lm_score,
              source=source,
          )
          if translation.startswith('ERROR:'):
            continue
          canonical_key = canonical_aux_key(translation)
          if args.candidate_canonical_dedup and canonical_key in seen_candidate_keys:
            event(
                args.events_file,
                kind='candidate_filtered',
                depth=depth,
                raw=raw,
                translation=translation,
                reason='duplicate_canonical',
                canonical_key=canonical_key,
                prompt=prompt,
                target=raw,
                source=source,
            )
            continue
          seen_candidate_keys.add(canonical_key)
          translated_candidates.append({
              'raw': raw,
              'lm_score': lm_score,
              'translation': translation,
              'problem': args.problem_name,
              'source': source,
          })
        ranked_node_candidates = rerank_candidate_records(
            translated_candidates,
            args.candidate_rerank,
            value_model,
            secondary_value_model,
            args.candidate_frontfill_limit,
        )
        if args.candidate_eval_limit and args.candidate_eval_limit > 0:
          for record in ranked_node_candidates[args.candidate_eval_limit:]:
            event(
                args.events_file,
                kind='candidate_filtered',
                depth=depth,
                raw=record['raw'],
                translation=record['translation'],
                reason='rank_pruned',
                candidate_rerank=args.candidate_rerank,
                candidate_eval_limit=args.candidate_eval_limit,
            )
          ranked_node_candidates = ranked_node_candidates[:args.candidate_eval_limit]
        for record in ranked_node_candidates:
          record.update({
              'node_index': node_index,
              'prev_score': prev_score,
              'prompt': prompt,
              'pstring': pstring,
          })
          depth_candidates.append(record)
      ranked_depth_candidates = rerank_candidate_records(
          depth_candidates,
          args.candidate_rerank,
          value_model,
          secondary_value_model,
          args.candidate_frontfill_limit,
      )
      for record in ranked_depth_candidates[args.candidate_depth_eval_limit:]:
        event(
            args.events_file,
            kind='candidate_filtered',
            depth=depth,
            raw=record['raw'],
            translation=record['translation'],
            reason='depth_rank_pruned',
            candidate_rerank=args.candidate_rerank,
            candidate_depth_eval_limit=args.candidate_depth_eval_limit,
        )
      for record in ranked_depth_candidates[:args.candidate_depth_eval_limit]:
        raw = record['raw']
        lm_score = record['lm_score']
        translation = record['translation']
        p_new_txt = insert_aux_to_premise(record['pstring'], translation)
        p_new = pr.Problem.from_txt(p_new_txt, translate=False)
        g_new, _ = build_graph_for_symbolic_search(gh, p_new, DEFINITIONS)
        tag = f'depth{depth}:{raw}'
        result = run_ddar_once(
            g_new,
            p_new,
            ddar,
            gh,
            args.max_level,
            args.ddar_timeout,
            args.events_file,
            tag,
            args.lm_fact_context_top_k,
            {candidate_new_point(raw)} if candidate_new_point(raw) else None,
        )
        if result['solved']:
          event(args.events_file, kind='solved', depth=depth, aux=translation)
          if args.out_file:
            with open(args.out_file, 'w', encoding='utf-8') as f:
              f.write(p_new_txt + '\n')
          return True
        next_beam.add(
            (
                g_new,
                build_lm_prompt(
                    p_new,
                    DEFINITIONS,
                    result.get('fact_context')
                    if args.lm_fact_context_top_k
                    else None,
                )
                if args.lm_fact_context_top_k
                else record['prompt'] + ' ' + raw + ' x00',
                p_new_txt,
            ),
            record['prev_score'] + lm_score,
        )
      beam = next_beam
      if len(beam) == 0:
        event(args.events_file, kind='beam_empty', depth=depth)
        break
      continue
    for prev_score, (g_cur, prompt, pstring) in beam.ordered():
      p_cur = pr.Problem.from_txt(pstring, translate=False)
      event(
          args.events_file,
          kind='decode_start',
          depth=depth,
          score=prev_score,
          prompt_tail=prompt[-240:],
      )
      forbidden_points = (
          existing_point_names(g_cur)
          if (
              args.candidate_point_mask
              or args.candidate_point_repair
              or args.candidate_prompt_sampling != 'none'
          )
          else None
      )
      candidates = generator.generate(
          prompt,
          args.num_return_sequences,
          args.max_new_tokens,
          args.temperature,
          args.top_p,
          forbidden_points,
          args.candidate_quality_multiplier,
          args.candidate_dsl_filter,
          args.candidate_dsl_token_mask,
          args.candidate_point_repair,
          args.candidate_prompt_sampling,
      )
      if args.candidate_template_backfill and len(candidates) < args.num_return_sequences:
        seen_raw = {raw for raw, _ in candidates}
        candidate_sources = {raw: 'lm' for raw, _ in candidates}
        seen_generation_keys = {
            candidate_generation_dedup_key(raw) for raw, _ in candidates
        }
        needed = args.num_return_sequences - len(candidates)
        for raw in template_backfill_candidates(
            forbidden_points,
            needed * 4,
            seen_generation_keys if args.candidate_canonical_dedup else None,
            goal_point_names(p_cur),
        ):
          generation_key = candidate_generation_dedup_key(raw)
          if raw not in seen_raw and generation_key not in seen_generation_keys:
            candidates.append((raw, 0.0))
            seen_raw.add(raw)
            candidate_sources[raw] = 'template_initial_backfill'
            seen_generation_keys.add(generation_key)
          if len(candidates) >= args.num_return_sequences:
            break
      else:
        candidate_sources = {raw: 'lm' for raw, _ in candidates}
      # Dedup within this proof state; pruned auxes may still help other branches.
      seen_candidate_keys: set[str] = set()
      translated_candidates = []
      for raw, lm_score in candidates:
        source = candidate_sources.get(raw, 'lm')
        translation = try_translate_candidate(raw, g_cur, pr, pt)
        event(
            args.events_file,
            kind='candidate',
            depth=depth,
            raw=raw,
            translation=translation,
            lm_score=lm_score,
            source=source,
        )
        if translation.startswith('ERROR:'):
          continue
        canonical_key = canonical_aux_key(translation)
        if args.candidate_canonical_dedup and canonical_key in seen_candidate_keys:
          event(
              args.events_file,
              kind='candidate_filtered',
              depth=depth,
              raw=raw,
              translation=translation,
              reason='duplicate_canonical',
              canonical_key=canonical_key,
              prompt=prompt,
              target=raw,
              source=source,
          )
          continue
        seen_candidate_keys.add(canonical_key)
        translated_candidates.append({
            'raw': raw,
            'lm_score': lm_score,
            'translation': translation,
            'problem': args.problem_name,
            'source': source,
        })
      ranked_candidates = rerank_candidate_records(
          translated_candidates,
          args.candidate_rerank,
          value_model,
          secondary_value_model,
          args.candidate_frontfill_limit,
      )
      if args.candidate_eval_limit and args.candidate_eval_limit > 0:
        for record in ranked_candidates[args.candidate_eval_limit:]:
          event(
              args.events_file,
              kind='candidate_filtered',
              depth=depth,
              raw=record['raw'],
              translation=record['translation'],
              reason='rank_pruned',
              candidate_rerank=args.candidate_rerank,
              candidate_eval_limit=args.candidate_eval_limit,
          )
        ranked_candidates = ranked_candidates[:args.candidate_eval_limit]
      for record in ranked_candidates:
        raw = record['raw']
        lm_score = record['lm_score']
        translation = record['translation']
        p_new_txt = insert_aux_to_premise(pstring, translation)
        p_new = pr.Problem.from_txt(p_new_txt, translate=False)
        g_new, _ = build_graph_for_symbolic_search(gh, p_new, DEFINITIONS)
        tag = f'depth{depth}:{raw}'
        result = run_ddar_once(
            g_new,
            p_new,
            ddar,
            gh,
            args.max_level,
            args.ddar_timeout,
            args.events_file,
            tag,
            args.lm_fact_context_top_k,
            {candidate_new_point(raw)} if candidate_new_point(raw) else None,
        )
        if result['solved']:
          event(args.events_file, kind='solved', depth=depth, aux=translation)
          if args.out_file:
            with open(args.out_file, 'w', encoding='utf-8') as f:
              f.write(p_new_txt + '\n')
          return True
        next_beam.add(
            (
                g_new,
                build_lm_prompt(
                    p_new,
                    DEFINITIONS,
                    result.get('fact_context')
                    if args.lm_fact_context_top_k
                    else None,
                )
                if args.lm_fact_context_top_k
                else prompt + ' ' + raw + ' x00',
                p_new_txt,
            ),
            prev_score + lm_score,
        )
    beam = next_beam
    if len(beam) == 0:
      event(args.events_file, kind='beam_empty', depth=depth)
      break
  return False


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--ag_repo', required=True)
  parser.add_argument('--problems_file', required=True)
  parser.add_argument('--problem_name', required=True)
  parser.add_argument('--defs_file', required=True)
  parser.add_argument('--rules_file', required=True)
  parser.add_argument('--mode', choices=['ddar', 'qwen'], default='ddar')
  parser.add_argument('--translate', action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument('--max_level', type=int, default=1000)
  parser.add_argument('--ddar_timeout', type=int, default=600)
  parser.add_argument('--events_file')
  parser.add_argument('--out_file')
  parser.add_argument('--qwen_model')
  parser.add_argument('--adapter_path')
  parser.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
  parser.add_argument('--device_map', default='cuda:0')
  parser.add_argument('--beam_size', type=int, default=4)
  parser.add_argument('--search_depth', type=int, default=2)
  parser.add_argument('--num_return_sequences', type=int, default=4)
  parser.add_argument('--max_new_tokens', type=int, default=64)
  parser.add_argument('--temperature', type=float, default=0.7)
  parser.add_argument('--top_p', type=float, default=0.95)
  parser.add_argument(
      '--candidate_point_mask',
      action=argparse.BooleanOptionalAction,
      default=True,
      help='filter generated candidates whose new point name already exists',
  )
  parser.add_argument(
      '--candidate_canonical_dedup',
      action=argparse.BooleanOptionalAction,
      default=True,
      help='skip equivalent auxiliary clauses after translation',
  )
  parser.add_argument(
      '--candidate_quality_multiplier',
      type=int,
      default=1,
      help='sample extra raw candidates before mask/dedup to preserve diversity',
  )
  parser.add_argument(
      '--candidate_dsl_filter',
      action=argparse.BooleanOptionalAction,
      default=False,
      help='filter raw generations that do not match the auxiliary DSL shape',
  )
  parser.add_argument(
      '--candidate_dsl_token_mask',
      action=argparse.BooleanOptionalAction,
      default=False,
      help='mask tokenizer tokens that cannot extend a valid auxiliary DSL prefix',
  )
  parser.add_argument(
      '--candidate_point_repair',
      action=argparse.BooleanOptionalAction,
      default=False,
      help='rename repeated constructed point names to the next free point',
  )
  parser.add_argument(
      '--candidate_prompt_sampling',
      choices=[
          'none',
          'balanced_constrained',
          'mixed_constructive',
          'mixed_progress_constructive',
      ],
      default='none',
      help='sample candidates from construction-family prefixes',
  )
  parser.add_argument(
      '--candidate_template_backfill',
      action=argparse.BooleanOptionalAction,
      default=False,
      help='fill missing model candidates with type-diverse DSL templates',
  )
  parser.add_argument(
      '--candidate_rerank',
      choices=[
          'none',
          'heuristic_diverse',
          'value_model',
          'value_model_diverse',
          'value_model_frontfill_diverse',
          'value_model_frontfill_progress_diverse',
      ],
      default='none',
      help='optional translated-candidate reranker before DDAR evaluation',
  )
  parser.add_argument(
      '--candidate_value_model',
      help='JSON value model for value_model candidate reranking',
  )
  parser.add_argument(
      '--candidate_secondary_value_model',
      help='secondary JSON value model for value_model_frontfill_diverse coverage',
  )
  parser.add_argument(
      '--candidate_frontfill_limit',
      type=int,
      default=8,
      help='front slots filled by --candidate_value_model in frontfill rerank',
  )
  parser.add_argument(
      '--candidate_eval_limit',
      type=int,
      default=0,
      help='evaluate only the top-N reranked valid candidates per beam node; 0 disables',
  )
  parser.add_argument(
      '--candidate_depth_eval_limit',
      type=int,
      default=0,
      help='evaluate only the top-N reranked valid candidates per search depth; 0 disables',
  )
  parser.add_argument(
      '--lm_fact_context_top_k',
      type=int,
      default=0,
      help='prepend top-K DDAR-added facts to LM prompts; 0 disables',
  )
  return parser.parse_args()


def main() -> None:
  global DEFINITIONS, RULES
  args = parse_args()
  if args.mode == 'qwen' and not args.qwen_model:
    raise ValueError('--qwen_model is required in qwen mode')
  add_ag_repo_to_path(args.ag_repo)
  ag = load_ag_modules()
  pr = ag['pr']
  DEFINITIONS = pr.Definition.from_txt_file(args.defs_file, to_dict=True)
  RULES = pr.Theorem.from_txt_file(args.rules_file, to_dict=True)
  solved = run_qwen_search(args)
  event(args.events_file, kind='finished', solved=solved)
  raise SystemExit(0 if solved else 2)


if __name__ == '__main__':
  main()
