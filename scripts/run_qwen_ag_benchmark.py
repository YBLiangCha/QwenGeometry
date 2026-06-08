"""Run DDAR or Qwen+DDAR over an AG problem file and write a summary.

This benchmark driver reuses ``qwen_ag_search.py`` and loads Qwen only once.
Detailed per-problem events are written under ``out_dir/events``; one JSONL
summary row is written to ``out_dir/summary.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import multiprocessing
import re
import signal
import sys
import time
import traceback
from typing import Any


def import_search(script_dir: str):
  sys.path.insert(0, str(Path(script_dir).resolve()))
  import qwen_ag_search as qs  # pylint: disable=import-error,import-outside-toplevel

  return qs


def safe_name(name: str) -> str:
  name = name or 'problem'
  return re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('_') or 'problem'


def build_graph_for_symbolic_search(gh: Any, p: Any, definitions: Any):
  """Build an AG graph while tolerating numeric goal-check bugs."""
  try:
    return gh.Graph.build_problem(p, definitions)
  except Exception:  # pylint: disable=broad-except
    p_no_goal = p.copy()
    p_no_goal.goal = None
    return gh.Graph.build_problem(p_no_goal, definitions)


_WORKER: dict[str, Any] = {}


class CandidateWallTimeoutError(TimeoutError):
  pass


def raise_candidate_wall_timeout(signum, frame):  # pylint: disable=unused-argument
  raise CandidateWallTimeoutError('candidate DDAR wall-clock timeout')


def effective_candidate_ddar_timeout(
    timeout: int | None,
    wall_timeout: int | None,
    margin_sec: int | float = 5,
) -> int | None:
  """Keep DDAR's cooperative timeout ahead of the hard wall-clock kill.

  When the wall timer fires first, the worker raises and loses DDAR's partial
  dependencies.  A slightly shorter cooperative timeout lets DDAR return a
  normal saturated/timeout result, which can still provide fact context and a
  progress signal for the next beam.
  """
  if timeout is None or wall_timeout is None:
    return timeout
  try:
    timeout_int = int(timeout)
    wall_timeout_int = int(wall_timeout)
    margin = int(float(margin_sec))
  except (TypeError, ValueError):
    return timeout
  if timeout_int <= 0 or wall_timeout_int <= 0 or margin <= 0:
    return timeout_int
  if timeout_int < wall_timeout_int:
    return timeout_int
  dynamic_margin = min(margin, max(1, wall_timeout_int // 20))
  return max(1, wall_timeout_int - dynamic_margin)


def init_candidate_ddar_worker(
    script_dir: str,
    ag_repo: str,
    defs_file: str,
    rules_file: str,
) -> None:
  """Initialize a CPU-only DDAR worker process.

  The benchmark process may already hold a CUDA model, so workers are spawned
  and import only the AG symbolic stack.  They do not print events; the parent
  writes events in candidate order to keep logs deterministic.
  """
  qs = import_search(script_dir)
  qs.add_ag_repo_to_path(ag_repo)
  ag = qs.load_ag_modules()
  pr = ag['pr']
  qs.DEFINITIONS = pr.Definition.from_txt_file(defs_file, to_dict=True)
  qs.RULES = pr.Theorem.from_txt_file(rules_file, to_dict=True)
  _WORKER.clear()
  _WORKER.update({'qs': qs, **ag})


def run_candidate_ddar_worker(task: dict[str, Any]) -> dict[str, Any]:
  """Run one candidate DDAR check in a spawned worker."""
  qs = _WORKER['qs']
  pr, gh, ddar = _WORKER['pr'], _WORKER['gh'], _WORKER['ddar']
  started = time.time()
  wall_timeout = task.get('wall_timeout')
  old_handler = None
  try:
    if wall_timeout:
      old_handler = signal.signal(signal.SIGALRM, raise_candidate_wall_timeout)
      signal.setitimer(signal.ITIMER_REAL, float(wall_timeout))
    p_new = pr.Problem.from_txt(task['p_new_txt'], translate=False)
    g_new, _ = build_graph_for_symbolic_search(gh, p_new, qs.DEFINITIONS)
    g_new, level_times, status, branches, added = ddar.solve(
        g_new,
        qs.RULES,
        p_new,
        max_level=task['max_level'],
        timeout=task['timeout'],
    )
    solved = qs.goal_is_solved(g_new, p_new)
    result = {
        'tag': task['tag'],
        'status': status,
        'solved': solved,
        'requested_timeout': task.get('requested_timeout'),
        'effective_timeout': task.get('timeout'),
        'wall_timeout': task.get('wall_timeout'),
        'candidate_rerank_score': task.get('candidate_rerank_score'),
        'candidate_rerank_phase': task.get('candidate_rerank_phase'),
        'candidate_source': task.get('candidate_source'),
        'candidate_construction_type': task.get('candidate_construction_type'),
        'levels': len(level_times),
        'level_times': [round(x, 3) for x in level_times],
        'branches': branches,
        'added_dependencies': len(added),
        'elapsed_sec': round(time.time() - started, 3),
        **qs.graph_stats(g_new, gh),
    }
    facts = qs.select_ddar_facts(
        added,
        p_new,
        int(task.get('fact_context_top_k') or 0),
        set(task.get('fact_context_recent_points') or []),
    )
    if facts:
      result['fact_context'] = facts
      result['fact_context_count'] = len(facts)
    return {**task, 'result': result, 'error': None}
  except Exception as exc:  # pylint: disable=broad-except
    return {
        **task,
        'result': None,
        'error': type(exc).__name__,
        'traceback': traceback.format_exc(),
        'elapsed_sec': round(time.time() - started, 3),
        'requested_timeout': task.get('requested_timeout'),
        'effective_timeout': task.get('timeout'),
        'wall_timeout': task.get('wall_timeout'),
    }
  finally:
    if wall_timeout:
      signal.setitimer(signal.ITIMER_REAL, 0)
      if old_handler is not None:
        signal.signal(signal.SIGALRM, old_handler)


def make_candidate_task(
    order: int,
    depth: int,
    raw: str,
    lm_score: float,
    translation: str,
    p_new_txt: str,
    prompt_next: str,
    max_level: int,
    timeout: int,
    wall_timeout: int | None,
    fact_context_top_k: int = 0,
    requested_timeout: int | None = None,
    soft_timeout_margin_sec: int | float = 5,
    prompt: str | None = None,
    candidate_rerank_score: float | None = None,
    candidate_rerank_phase: str | None = None,
    candidate_source: str | None = None,
    candidate_construction_type: str | None = None,
    candidate_depth_rank: int | None = None,
    candidate_depth_eval_phase: str | None = None,
    parent_fact_context: list[str] | None = None,
) -> dict[str, Any]:
  raw_text = raw.strip()
  new_point = None
  if ' = ' in raw_text:
    new_point = raw_text.split(' = ', 1)[0].strip()
  elif ' : ' in raw_text:
    new_point = raw_text.split(' : ', 1)[0].strip()
  return {
      'order': order,
      'depth': depth,
      'raw': raw,
      'lm_score': lm_score,
      'translation': translation,
      'p_new_txt': p_new_txt,
      'prompt_next': prompt_next,
      'prompt': prompt,
      'max_level': max_level,
      'timeout': timeout,
      'requested_timeout': requested_timeout if requested_timeout is not None else timeout,
      'wall_timeout': wall_timeout,
      'soft_timeout_margin_sec': soft_timeout_margin_sec,
      'fact_context_top_k': fact_context_top_k,
      'fact_context_recent_points': [new_point] if new_point else [],
      'candidate_rerank_score': candidate_rerank_score,
      'candidate_rerank_phase': candidate_rerank_phase,
      'candidate_source': candidate_source,
      'candidate_construction_type': candidate_construction_type,
      'candidate_depth_rank': candidate_depth_rank,
      'candidate_depth_eval_phase': candidate_depth_eval_phase,
      'parent_fact_context': list(parent_fact_context or []),
      'tag': f'depth{depth}:{raw}',
  }


def make_beam_state(
    g: Any,
    prompt: str,
    pstring: str,
    fact_context: list[str] | None = None,
) -> tuple[Any, str, str, list[str]]:
  return (g, prompt, pstring, list(fact_context or []))


def unpack_beam_state(state: Any) -> tuple[Any, str, str, list[str]]:
  """Return a normalized beam state while accepting older 3-tuples."""
  if isinstance(state, tuple) and len(state) == 4:
    g, prompt, pstring, fact_context = state
    return g, prompt, pstring, list(fact_context or [])
  if isinstance(state, tuple) and len(state) == 3:
    g, prompt, pstring = state
    return g, prompt, pstring, []
  raise ValueError(f'unexpected beam state shape: {state!r}')


def merge_fact_context(
    parent_facts: list[str] | None,
    child_facts: list[str] | None,
    max_facts: int,
) -> list[str]:
  """Keep current DDAR facts while preserving a little parent context."""
  if max_facts <= 0:
    return []
  parent = [fact for fact in parent_facts or [] if fact]
  child = [fact for fact in child_facts or [] if fact]
  if not parent:
    return child[:max_facts]
  if not child:
    return parent[:max_facts]
  parent_keep = max(1, min(len(parent), max_facts // 3))
  child_keep = max(1, max_facts - parent_keep)
  merged: list[str] = []
  seen: set[str] = set()

  def add_many(facts: list[str], limit: int | None = None) -> None:
    added = 0
    for fact in facts:
      if fact in seen:
        continue
      merged.append(fact)
      seen.add(fact)
      added += 1
      if len(merged) >= max_facts or (limit is not None and added >= limit):
        break

  add_many(child, child_keep)
  add_many(parent, parent_keep)
  add_many(child)
  add_many(parent)
  return merged[:max_facts]


def build_next_prompt(
    qs: Any,
    pr: Any,
    p_new: Any | None,
    p_new_txt: str,
    prompt_next: str,
    fact_context: list[str],
    args: argparse.Namespace,
) -> str:
  if not args.lm_fact_context_top_k:
    return prompt_next
  if p_new is not None:
    return qs.build_lm_prompt(p_new, qs.DEFINITIONS, fact_context)
  return qs.build_lm_prompt_from_problem_text(
      p_new_txt, pr, qs.DEFINITIONS, fact_context
  )


def maybe_log_candidate_sft_signal(
    qs: Any,
    events_file: str,
    problem_name: str,
    depth: int,
    raw: str,
    translation: str,
    prompt: str | None,
    p_new_txt: str,
    root: dict[str, Any],
    result: dict[str, Any],
    args: argparse.Namespace,
    candidate_source: str = 'lm',
    candidate_construction_type: str | None = None,
    candidate_rerank_score: float | None = None,
    candidate_rerank_phase: str | None = None,
) -> None:
  if not args.log_candidate_sft_signals or not prompt:
    return
  added = int(result.get('added_dependencies') or 0)
  root_added = int(root.get('added_dependencies') or 0)
  delta = added - root_added
  elapsed = result.get('elapsed_sec')
  reason = None
  if result.get('solved'):
    reason = 'candidate_solved'
  elif (
      not args.disable_candidate_sft_progress_signals
      and added >= args.candidate_sft_signal_min_added_dependencies
      and delta >= args.candidate_sft_signal_min_root_delta
      and (
          args.candidate_sft_signal_max_elapsed_sec <= 0
          or (
              isinstance(elapsed, (int, float))
              and elapsed <= args.candidate_sft_signal_max_elapsed_sec
          )
      )
  ):
    reason = 'ddar_progress_positive'
  if reason is None:
    return
  qs.event(
      events_file,
      kind='candidate_sft_signal',
      problem=problem_name,
      depth=depth,
      prompt=prompt,
      target=raw,
      translation=translation,
      reason=reason,
      problem_after_aux=p_new_txt,
      candidate_solved=bool(result.get('solved')),
      candidate_added_dependencies=added,
      root_added_dependencies=root_added,
      progress_delta_dependencies=delta,
      candidate_elapsed_sec=elapsed,
      candidate_levels=result.get('levels'),
      candidate_ddar_status=result.get('status'),
      candidate_source=candidate_source,
      candidate_construction_type=(
          candidate_construction_type or qs.construction_type_key(translation)
      ),
      candidate_rerank_score=candidate_rerank_score,
      candidate_rerank_phase=candidate_rerank_phase,
  )


def candidate_construction_type_for_event(qs: Any, raw: str, translation: str) -> str:
  if translation and not translation.startswith('ERROR:'):
    return qs.construction_type_key(translation)
  try:
    return qs.construction_type_key(qs.dsl_to_constructive_candidate(raw))
  except Exception:  # pylint: disable=broad-except
    return 'error'


def maybe_log_candidate_hard_negative_signal(
    qs: Any,
    events_file: str,
    problem_name: str,
    depth: int,
    raw: str,
    translation: str,
    prompt: str | None,
    args: argparse.Namespace,
    source: str = 'lm',
) -> None:
  if not args.log_candidate_hard_negative_signals or not prompt:
    return
  reasons = {
      reason.strip()
      for reason in args.candidate_hard_negative_signal_reasons.split(',')
      if reason.strip()
  }
  reason = qs.candidate_value_error_key(translation)
  if reason not in reasons:
    return
  qs.event(
      events_file,
      kind='candidate_hard_negative_signal',
      problem=problem_name,
      depth=depth,
      prompt=prompt,
      target=raw,
      translation=translation,
      reason=reason,
      candidate_source=source,
      candidate_construction_type=candidate_construction_type_for_event(
          qs, raw, translation
      ),
  )


def candidate_unique_backfill_target(args: argparse.Namespace) -> int:
  if args.candidate_eval_limit and args.candidate_eval_limit > 0:
    return max(1, min(args.num_return_sequences, args.candidate_eval_limit))
  if args.candidate_depth_eval_limit and args.candidate_depth_eval_limit > 0:
    return max(1, min(args.num_return_sequences, args.candidate_depth_eval_limit))
  return max(1, args.num_return_sequences)


def select_depth_candidates_for_eval(
    qs: Any,
    records: list[dict[str, Any]],
    eval_limit: int,
    type_cap: int,
    tail_slots: int = 0,
    tail_strategy: str = 'even',
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Pick a depth-level eval pool while delaying over-represented types.

  A few optional tail slots preserve AG1-like coverage for low-ranked valid
  candidates.  This matters because AG1 reproduction logs show solved
  candidates at rank 18/31 in the first LM batch; a strict top-N depth cutoff
  can miss those even when the LM already generated them.
  """
  if eval_limit <= 0:
    return records, []
  if len(records) <= eval_limit:
    for rank, record in enumerate(records):
      record['_candidate_depth_rank'] = rank
      record['_candidate_depth_eval_phase'] = 'all'
    return records, []

  for rank, record in enumerate(records):
    record['_candidate_depth_rank'] = rank

  tail_slots = max(0, min(int(tail_slots or 0), eval_limit))
  if tail_slots > 0:
    tail_slots = min(tail_slots, max(0, len(records) - eval_limit))
  tail_selected: list[dict[str, Any]] = []
  tail_ids: set[int] = set()
  if tail_slots > 0:
    tail_pool = records[eval_limit:]
    if len(tail_pool) <= tail_slots:
      tail_selected = list(tail_pool)
    elif tail_strategy == 'near_spread':
      # Keep candidates just beyond the cutoff, then spread remaining tail
      # slots to the far end.  AG1 repros solved cases at ranks such as 18/31,
      # which an evenly sampled 16+4 tail can skip.
      near_count = min(tail_slots, len(tail_pool))
      if tail_slots > 3:
        near_count = min(near_count, max(3, tail_slots // 2))
      selected_indexes = list(range(near_count))
      remaining_slots = tail_slots - len(selected_indexes)
      far_start = near_count
      far_len = len(tail_pool) - far_start
      if remaining_slots > 0 and far_len > 0:
        if remaining_slots == 1:
          selected_indexes.append(len(tail_pool) - 1)
        else:
          for i in range(remaining_slots):
            index = far_start + round(
                i * (far_len - 1) / max(1, remaining_slots - 1)
            )
            if index not in selected_indexes:
              selected_indexes.append(index)
      tail_selected = [tail_pool[index] for index in sorted(selected_indexes)]
    else:
      # Evenly sample the ranked tail, including the far end.  For example,
      # a 32-candidate AG1-style batch with eval_limit=24 and tail_slots=4
      # selects roughly ranks 24, 26, 28, and 31.
      for i in range(tail_slots):
        index = round(i * (len(tail_pool) - 1) / max(1, tail_slots - 1))
        tail_selected.append(tail_pool[index])
    for record in tail_selected:
      tail_ids.add(id(record))
      record['_candidate_depth_eval_phase'] = 'tail_rank_coverage'

  primary_limit = max(0, eval_limit - len(tail_selected))
  if type_cap <= 0:
    selected = [record for record in records if id(record) not in tail_ids][
        :primary_limit
    ] + tail_selected
    selected_ids = {id(record) for record in selected}
    for record in selected:
      record.setdefault('_candidate_depth_eval_phase', 'primary')
    return (
        [record for record in records if id(record) in selected_ids],
        [record for record in records if id(record) not in selected_ids],
    )

  selected: list[dict[str, Any]] = []
  delayed: list[dict[str, Any]] = []
  selected_ids: set[int] = set()
  type_counts: dict[str, int] = {}
  type_delayed_ids: set[int] = set()
  for record in records:
    if id(record) in tail_ids:
      continue
    construction_type = qs.construction_type_key(record['translation'])
    count = type_counts.get(construction_type, 0)
    if len(selected) < primary_limit and count < type_cap:
      selected.append(record)
      selected_ids.add(id(record))
      record['_candidate_depth_eval_phase'] = 'primary'
      type_counts[construction_type] = count + 1
    else:
      delayed.append(record)
      if count >= type_cap:
        type_delayed_ids.add(id(record))

  for record in delayed:
    if len(selected) >= primary_limit:
      break
    selected.append(record)
    selected_ids.add(id(record))
    record['_candidate_depth_eval_phase'] = 'primary_backfill'

  for record in tail_selected:
    selected.append(record)
    selected_ids.add(id(record))

  pruned = []
  for record in records:
    if id(record) in selected_ids:
      continue
    if id(record) in type_delayed_ids:
      record['_candidate_depth_prune_reason'] = 'depth_type_cap_pruned'
    pruned.append(record)
  return selected, pruned


def post_canonical_template_backfill(
    qs: Any,
    events_file: str,
    problem_name: str,
    depth: int,
    prompt: str,
    translated_candidates: list[dict[str, Any]],
    forbidden_points: set[str] | None,
    seen_candidate_keys: set[str],
    seen_raw: set[str],
    target_count: int,
    g_cur: Any,
    pr: Any,
    pt: Any,
    preferred_points: set[str] | None = None,
) -> None:
  if len(translated_candidates) >= target_count:
    return
  start_len = len(translated_candidates)
  attempted = 0
  needed = target_count - len(translated_candidates)
  template_budget = max(target_count * 32, needed * 16, 128)
  for raw in qs.template_backfill_candidates(
      forbidden_points, template_budget, seen_candidate_keys, preferred_points
  ):
    attempted += 1
    if raw in seen_raw:
      continue
    seen_raw.add(raw)
    translation = qs.try_translate_candidate(raw, g_cur, pr, pt)
    qs.event(
        events_file,
        kind='candidate',
        depth=depth,
        raw=raw,
        translation=translation,
        lm_score=0.0,
        source='template_post_canonical_backfill',
    )
    if translation.startswith('ERROR:'):
      continue
    canonical_key = qs.canonical_aux_key(translation)
    if canonical_key in seen_candidate_keys:
      qs.event(
          events_file,
          kind='candidate_filtered',
          depth=depth,
          raw=raw,
          translation=translation,
          reason='duplicate_canonical',
          canonical_key=canonical_key,
          prompt=prompt,
          target=raw,
          source='template_post_canonical_backfill',
      )
      continue
    seen_candidate_keys.add(canonical_key)
    translated_candidates.append({
        'raw': raw,
        'lm_score': 0.0,
        'translation': translation,
        'problem': problem_name,
        'source': 'template_post_canonical_backfill',
    })
    if len(translated_candidates) >= target_count:
      return
  if len(translated_candidates) < target_count:
    qs.event(
        events_file,
        kind='candidate_backfill_exhausted',
        depth=depth,
        target_count=target_count,
        before_count=start_len,
        after_count=len(translated_candidates),
        attempted_templates=attempted,
        template_budget=template_budget,
    )


def run_candidate_process_entry(
    task: dict[str, Any],
    script_dir: str,
    ag_repo: str,
    defs_file: str,
    rules_file: str,
    child_conn: Any,
) -> None:
  try:
    init_candidate_ddar_worker(script_dir, ag_repo, defs_file, rules_file)
    child_conn.send(run_candidate_ddar_worker(task))
  except Exception as exc:  # pylint: disable=broad-except
    child_conn.send({
        **task,
        'result': None,
        'error': type(exc).__name__,
        'traceback': traceback.format_exc(),
        'elapsed_sec': None,
    })
  finally:
    try:
      child_conn.close()
    except Exception:  # pylint: disable=broad-except
      pass


def run_candidate_tasks_parallel(
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
  if not tasks:
    return []
  worker_count = min(max(args.candidate_ddar_workers, 1), len(tasks))
  main_file = getattr(sys.modules.get('__main__'), '__file__', '')
  start_method = 'spawn'
  if not main_file or main_file == '<stdin>' or not Path(main_file).exists():
    # This branch is for smoke tests/imported use.  The real benchmark entry
    # has a file-backed __main__ and uses spawn so CUDA state is never forked.
    start_method = 'fork'
  context = multiprocessing.get_context(start_method)
  pending = list(tasks)
  active: list[dict[str, Any]] = []
  results: list[dict[str, Any]] = []

  def start_task(task: dict[str, Any]) -> dict[str, Any]:
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=run_candidate_process_entry,
        args=(
            task,
            args.script_dir,
            args.ag_repo,
            args.defs_file,
            args.rules_file,
            child_conn,
        ),
    )
    process.start()
    child_conn.close()
    return {
        'task': task,
        'process': process,
        'conn': parent_conn,
        'started': time.monotonic(),
    }

  def close_item(item: dict[str, Any]) -> None:
    process = item['process']
    process.join(timeout=1)
    try:
      item['conn'].close()
    except Exception:  # pylint: disable=broad-except
      pass

  while pending or active:
    while pending and len(active) < worker_count:
      active.append(start_task(pending.pop(0)))

    remaining = []
    progressed = False
    for item in active:
      task = item['task']
      process = item['process']
      conn = item['conn']
      if conn.poll():
        result = conn.recv()
        results.append(result)
        close_item(item)
        progressed = True
      else:
        hard_timeout = task.get('wall_timeout') or task.get('timeout') or 0
        if hard_timeout and time.monotonic() - item['started'] > hard_timeout + 5:
          process.terminate()
          process.join(timeout=5)
          if process.is_alive() and hasattr(process, 'kill'):
            process.kill()
            process.join(timeout=5)
          results.append({
              **task,
              'result': None,
              'error': 'CandidateProcessTimeout',
              'traceback': None,
              'elapsed_sec': round(time.monotonic() - item['started'], 3),
          })
          close_item(item)
          progressed = True
        elif not process.is_alive() and process.exitcode is not None:
          results.append({
              **task,
              'result': None,
              'error': 'CandidateProcessExit',
              'traceback': None,
              'elapsed_sec': round(time.monotonic() - item['started'], 3),
              'exitcode': process.exitcode,
          })
          close_item(item)
          progressed = True
        else:
          remaining.append(item)
    active = remaining
    if active and not progressed:
      time.sleep(0.2)
  return sorted(results, key=lambda item: item['order'])


def is_candidate_timeout_error(item: dict[str, Any]) -> bool:
  error = str(item.get('error') or '')
  return 'Timeout' in error


def maybe_add_timeout_beam_fallback(
    qs: Any,
    pr: Any,
    events_file: str,
    next_beam: Any,
    timeout_items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
  limit = int(getattr(args, 'candidate_timeout_beam_fallback_limit', 0) or 0)
  mode = getattr(args, 'candidate_timeout_beam_fallback_mode', 'empty')
  if limit <= 0 or not timeout_items:
    return
  if mode == 'empty' and len(next_beam) > 0:
    return
  ranked = sorted(
      timeout_items,
      key=lambda item: (
          item.get('candidate_rerank_score')
          if item.get('candidate_rerank_score') is not None
          else float('-inf'),
          item.get('prev_score', 0.0) + item.get('lm_score', 0.0),
          -item.get('order', 0),
      ),
      reverse=True,
  )
  for fallback_rank, item in enumerate(ranked[:limit], start=1):
    fallback_fact_context = (
        list(item.get('parent_fact_context') or [])
        if args.lm_fact_context_top_k
        else []
    )
    prompt = (
        qs.build_lm_prompt_from_problem_text(
            item['p_new_txt'],
            pr,
            qs.DEFINITIONS,
            fallback_fact_context,
        )
        if args.lm_fact_context_top_k
        else item['prompt_next']
    )
    next_beam.add(
        make_beam_state(
            None,
            prompt,
            item['p_new_txt'],
            fallback_fact_context,
        ),
        candidate_beam_score(
            item.get('prev_score', 0.0),
            item.get('lm_score', 0.0),
            item.get('candidate_rerank_score'),
            args,
        ),
    )
    qs.event(
        events_file,
        kind='candidate_timeout_beam_fallback',
        depth=item.get('depth'),
        raw=item.get('raw'),
        translation=item.get('translation'),
        error=item.get('error'),
        candidate_rerank_score=item.get('candidate_rerank_score'),
        candidate_rerank_phase=item.get('candidate_rerank_phase'),
        candidate_source=item.get('candidate_source'),
        candidate_construction_type=item.get('candidate_construction_type'),
        candidate_depth_rank=item.get('candidate_depth_rank'),
        candidate_depth_eval_phase=item.get('candidate_depth_eval_phase'),
        fallback_rank=fallback_rank,
        fallback_limit=limit,
        fallback_mode=mode,
        elapsed_sec=item.get('elapsed_sec'),
        parent_fact_context_count=len(fallback_fact_context),
    )


def candidate_beam_score(
    prev_score: float,
    lm_score: float,
    candidate_rerank_score: float | None,
    args: argparse.Namespace,
    progress_delta_dependencies: int | float = 0,
) -> float:
  strategy = getattr(args, 'candidate_beam_score', 'lm_score')
  rerank_score = (
      float(candidate_rerank_score)
      if isinstance(candidate_rerank_score, (int, float))
      else 0.0
  )
  progress_delta = max(0.0, float(progress_delta_dependencies or 0.0))
  progress_weight = max(0.0, float(getattr(args, 'candidate_beam_progress_weight', 0.0)))
  progress_bonus = progress_weight * math.log1p(progress_delta)
  progress_cap = float(getattr(args, 'candidate_beam_progress_cap', 0.0) or 0.0)
  if progress_cap > 0:
    progress_bonus = min(progress_bonus, progress_cap)
  if strategy == 'rerank_plus_progress':
    return prev_score + rerank_score + progress_bonus
  if strategy == 'rerank_score':
    return prev_score + rerank_score
  if strategy == 'lm_plus_rerank':
    return prev_score + lm_score + rerank_score
  return prev_score + lm_score


def candidate_progress_delta(root: dict[str, Any], result: dict[str, Any]) -> int:
  return max(
      0,
      int(result.get('added_dependencies') or 0)
      - int(root.get('added_dependencies') or 0),
  )


def candidate_beam_progress_bonus(
    args: argparse.Namespace,
    progress_delta_dependencies: int | float,
) -> float:
  progress_delta = max(0.0, float(progress_delta_dependencies or 0.0))
  progress_weight = max(0.0, float(getattr(args, 'candidate_beam_progress_weight', 0.0)))
  progress_bonus = progress_weight * math.log1p(progress_delta)
  progress_cap = float(getattr(args, 'candidate_beam_progress_cap', 0.0) or 0.0)
  if progress_cap > 0:
    progress_bonus = min(progress_bonus, progress_cap)
  return progress_bonus


def limited_beam_nodes(
    qs: Any,
    events_file: str,
    beam: Any,
    args: argparse.Namespace,
    depth: int,
) -> list[tuple[float, Any]]:
  ordered = beam.ordered()
  limit = int(getattr(args, 'candidate_decode_beam_limit', 0) or 0)
  if limit <= 0 or len(ordered) <= limit:
    return ordered
  for score, state in ordered[limit:]:
    _, prompt, _, _ = unpack_beam_state(state)
    qs.event(
        events_file,
        kind='beam_decode_pruned',
        depth=depth,
        score=score,
        candidate_decode_beam_limit=limit,
        prompt_tail=prompt[-240:],
    )
  return ordered[:limit]


def solve_one(
    p: Any,
    qs: Any,
    pr: Any,
    gh: Any,
    ddar: Any,
    pt: Any,
    generator: Any,
    args: argparse.Namespace,
    events_file: str,
) -> dict[str, Any]:
  g, _ = build_graph_for_symbolic_search(gh, p, qs.DEFINITIONS)
  root_max_level = args.root_max_level or args.max_level
  root_timeout = args.root_ddar_timeout or args.ddar_timeout
  root = qs.run_ddar_once(
      g,
      p,
      ddar,
      gh,
      root_max_level,
      root_timeout,
      events_file,
      'root',
      args.lm_fact_context_top_k,
  )
  row = {
      'problem': p.url,
      'mode': args.mode,
      'root_solved': root['solved'],
      'solved': root['solved'],
      'solved_depth': None,
      'aux': None,
      'events_file': events_file,
      'root_levels': root['levels'],
      'root_cache_items': root['cache_items'],
  }
  if root['solved'] or args.mode == 'ddar':
    return row

  prompt0 = qs.build_lm_prompt(
      p,
      qs.DEFINITIONS,
      root.get('fact_context') if args.lm_fact_context_top_k else None,
  )
  root_fact_context = (
      list(root.get('fact_context') or []) if args.lm_fact_context_top_k else []
  )
  beam = qs.BeamQueue(args.beam_size)
  beam.add(make_beam_state(g, prompt0, p.txt(), root_fact_context), 0.0)
  candidate_max_level = args.candidate_max_level or args.max_level
  candidate_timeout = args.candidate_ddar_timeout or args.ddar_timeout
  candidate_wall_timeout = args.candidate_wall_timeout or candidate_timeout
  candidate_solve_timeout = effective_candidate_ddar_timeout(
      candidate_timeout,
      candidate_wall_timeout,
      args.candidate_soft_timeout_margin_sec,
  )
  qs.event(
      events_file,
      kind='candidate_timeout_config',
      requested_timeout=candidate_timeout,
      effective_timeout=candidate_solve_timeout,
      wall_timeout=candidate_wall_timeout,
      soft_timeout_margin_sec=args.candidate_soft_timeout_margin_sec,
  )
  value_model = getattr(args, '_candidate_value_model', None)
  secondary_value_model = getattr(args, '_candidate_secondary_value_model', None)

  for depth in range(args.search_depth):
    qs.event(events_file, kind='depth_start', depth=depth, nodes=len(beam))
    next_beam = qs.BeamQueue(args.beam_size)
    if args.candidate_depth_eval_limit and args.candidate_depth_eval_limit > 0:
      depth_candidates = []
      for node_index, (prev_score, beam_state) in enumerate(
          limited_beam_nodes(qs, events_file, beam, args, depth)
      ):
        g_cur, prompt, pstring, parent_fact_context = unpack_beam_state(beam_state)
        p_cur = pr.Problem.from_txt(pstring, translate=False)
        if g_cur is None:
          g_cur, _ = build_graph_for_symbolic_search(gh, p_cur, qs.DEFINITIONS)
        qs.event(
            events_file,
            kind='decode_start',
            depth=depth,
            score=prev_score,
            prompt_tail=prompt[-240:],
        )
        forbidden_points = (
            qs.existing_point_names(g_cur)
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
        seen_raw = {raw for raw, _ in candidates}
        candidate_sources = {raw: 'lm' for raw, _ in candidates}
        seen_generation_keys = {
            qs.candidate_generation_dedup_key(raw) for raw, _ in candidates
        }
        # Dedup within this proof state; pruned auxes may still help other branches.
        seen_candidate_keys: set[str] = set()
        if args.candidate_template_backfill and len(candidates) < args.num_return_sequences:
          needed = args.num_return_sequences - len(candidates)
          for raw in qs.template_backfill_candidates(
              forbidden_points,
              needed * 4,
              seen_generation_keys if args.candidate_canonical_dedup else None,
              qs.goal_point_names(p_cur),
          ):
            generation_key = qs.candidate_generation_dedup_key(raw)
            if raw not in seen_raw and generation_key not in seen_generation_keys:
              candidates.append((raw, 0.0))
              seen_raw.add(raw)
              candidate_sources[raw] = 'template_initial_backfill'
              seen_generation_keys.add(generation_key)
            if len(candidates) >= args.num_return_sequences:
              break
        translated_candidates = []
        for raw, lm_score in candidates:
          source = candidate_sources.get(raw, 'lm')
          translation = qs.try_translate_candidate(raw, g_cur, pr, pt)
          qs.event(
              events_file,
              kind='candidate',
              depth=depth,
              raw=raw,
              translation=translation,
              lm_score=lm_score,
              source=source,
          )
          maybe_log_candidate_hard_negative_signal(
              qs,
              events_file,
              p.url,
              depth,
              raw,
              translation,
              prompt,
              args,
              source=source,
          )
          if translation.startswith('ERROR:'):
            continue
          canonical_key = qs.canonical_aux_key(translation)
          if args.candidate_canonical_dedup and canonical_key in seen_candidate_keys:
            qs.event(
                events_file,
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
              'problem': p.url,
              'source': source,
          })
        if args.candidate_template_backfill:
          post_canonical_template_backfill(
              qs,
              events_file,
              p.url,
              depth,
              prompt,
              translated_candidates,
              forbidden_points,
              seen_candidate_keys,
              seen_raw,
              candidate_unique_backfill_target(args),
              g_cur,
              pr,
              pt,
              qs.goal_point_names(p_cur),
          )
        ranked_node_candidates = qs.rerank_candidate_records(
            translated_candidates,
            args.candidate_rerank,
            value_model,
            secondary_value_model,
            args.candidate_frontfill_limit,
        )
        if args.candidate_eval_limit and args.candidate_eval_limit > 0:
          for record in ranked_node_candidates[args.candidate_eval_limit:]:
            qs.event(
                events_file,
                kind='candidate_filtered',
                depth=depth,
                raw=record['raw'],
                translation=record['translation'],
                reason='rank_pruned',
                candidate_rerank=args.candidate_rerank,
                candidate_rerank_score=record.get('_candidate_rerank_score'),
                candidate_rerank_phase=record.get('_candidate_rerank_phase'),
                candidate_construction_type=qs.construction_type_key(
                    record['translation']
                ),
                candidate_eval_limit=args.candidate_eval_limit,
                source=record.get('source', 'lm'),
            )
          ranked_node_candidates = ranked_node_candidates[:args.candidate_eval_limit]
        for record in ranked_node_candidates:
          record.update({
              'node_index': node_index,
              'prev_score': prev_score,
              'prompt': prompt,
              'pstring': pstring,
              'parent_fact_context': parent_fact_context,
          })
          depth_candidates.append(record)
      ranked_depth_candidates = qs.rerank_candidate_records(
          depth_candidates,
          args.candidate_rerank,
          value_model,
          secondary_value_model,
          args.candidate_frontfill_limit,
      )
      eval_depth_candidates, pruned_depth_candidates = select_depth_candidates_for_eval(
          qs,
          ranked_depth_candidates,
          args.candidate_depth_eval_limit,
          args.candidate_depth_type_eval_cap,
          args.candidate_depth_tail_eval_slots,
          args.candidate_depth_tail_eval_strategy,
      )
      for record in pruned_depth_candidates:
        qs.event(
            events_file,
            kind='candidate_filtered',
            depth=depth,
            raw=record['raw'],
            translation=record['translation'],
            reason=record.get('_candidate_depth_prune_reason', 'depth_rank_pruned'),
            candidate_rerank=args.candidate_rerank,
            candidate_rerank_score=record.get('_candidate_rerank_score'),
            candidate_rerank_phase=record.get('_candidate_rerank_phase'),
            candidate_construction_type=qs.construction_type_key(
                record['translation']
            ),
            candidate_depth_eval_limit=args.candidate_depth_eval_limit,
            candidate_depth_type_eval_cap=args.candidate_depth_type_eval_cap,
            candidate_depth_tail_eval_slots=args.candidate_depth_tail_eval_slots,
            candidate_depth_tail_eval_strategy=args.candidate_depth_tail_eval_strategy,
            candidate_depth_rank=record.get('_candidate_depth_rank'),
            candidate_depth_eval_phase=record.get('_candidate_depth_eval_phase'),
            source=record.get('source', 'lm'),
        )
      parallel_tasks = []
      for record in eval_depth_candidates:
        raw = record['raw']
        lm_score = record['lm_score']
        translation = record['translation']
        qs.event(
            events_file,
            kind='candidate_depth_eval_selected',
            depth=depth,
            raw=raw,
            translation=translation,
            candidate_rerank=args.candidate_rerank,
            candidate_rerank_score=record.get('_candidate_rerank_score'),
            candidate_rerank_phase=record.get('_candidate_rerank_phase'),
            candidate_construction_type=qs.construction_type_key(translation),
            candidate_depth_eval_limit=args.candidate_depth_eval_limit,
            candidate_depth_type_eval_cap=args.candidate_depth_type_eval_cap,
            candidate_depth_tail_eval_slots=args.candidate_depth_tail_eval_slots,
            candidate_depth_tail_eval_strategy=args.candidate_depth_tail_eval_strategy,
            candidate_depth_rank=record.get('_candidate_depth_rank'),
            candidate_depth_eval_phase=record.get('_candidate_depth_eval_phase'),
            source=record.get('source', 'lm'),
        )
        p_new_txt = qs.insert_aux_to_premise(record['pstring'], translation)
        prompt_next = record['prompt'] + ' ' + raw + ' x00'
        if args.candidate_ddar_workers > 1:
          parallel_tasks.append(make_candidate_task(
              len(parallel_tasks),
              depth,
              raw,
              lm_score,
              translation,
              p_new_txt,
              prompt_next,
              candidate_max_level,
              candidate_solve_timeout,
              candidate_wall_timeout,
              args.lm_fact_context_top_k,
              requested_timeout=candidate_timeout,
              soft_timeout_margin_sec=args.candidate_soft_timeout_margin_sec,
              prompt=record['prompt'],
              candidate_rerank_score=record.get('_candidate_rerank_score'),
              candidate_rerank_phase=record.get('_candidate_rerank_phase'),
              candidate_source=record.get('source', 'lm'),
              candidate_construction_type=qs.construction_type_key(translation),
              candidate_depth_rank=record.get('_candidate_depth_rank'),
              candidate_depth_eval_phase=record.get('_candidate_depth_eval_phase'),
              parent_fact_context=record.get('parent_fact_context'),
          ))
          parallel_tasks[-1]['prev_score'] = record['prev_score']
          continue
        p_new = pr.Problem.from_txt(p_new_txt, translate=False)
        g_new, _ = build_graph_for_symbolic_search(gh, p_new, qs.DEFINITIONS)
        result = qs.run_ddar_once(
            g_new,
            p_new,
            ddar,
            gh,
            candidate_max_level,
            candidate_solve_timeout,
            events_file,
            f'depth{depth}:{raw}',
            args.lm_fact_context_top_k,
            {qs.candidate_new_point(raw)} if qs.candidate_new_point(raw) else None,
        )
        result['requested_timeout'] = candidate_timeout
        result['effective_timeout'] = candidate_solve_timeout
        result['wall_timeout'] = candidate_wall_timeout
        maybe_log_candidate_sft_signal(
            qs,
            events_file,
            p.url,
            depth,
            raw,
            translation,
            record['prompt'],
            p_new_txt,
            root,
            result,
            args,
            candidate_source=record.get('source', 'lm'),
            candidate_construction_type=qs.construction_type_key(translation),
            candidate_rerank_score=record.get('_candidate_rerank_score'),
            candidate_rerank_phase=record.get('_candidate_rerank_phase'),
        )
        if result['solved']:
          qs.event(
              events_file,
              kind='solved',
              depth=depth,
              aux=translation,
              candidate_depth_rank=record.get('_candidate_depth_rank'),
              candidate_depth_eval_phase=record.get('_candidate_depth_eval_phase'),
              candidate_rerank_score=record.get('_candidate_rerank_score'),
              candidate_rerank_phase=record.get('_candidate_rerank_phase'),
              candidate_source=record.get('source', 'lm'),
              candidate_construction_type=qs.construction_type_key(translation),
          )
          row.update({
              'solved': True,
              'solved_depth': depth,
              'aux': translation,
          })
          return row
        next_fact_context = merge_fact_context(
            record.get('parent_fact_context'),
            result.get('fact_context'),
            args.lm_fact_context_top_k,
        )
        progress_delta = candidate_progress_delta(root, result)
        beam_score = candidate_beam_score(
            record['prev_score'],
            lm_score,
            record.get('_candidate_rerank_score'),
            args,
            progress_delta,
        )
        next_beam.add(
            make_beam_state(
                g_new,
                build_next_prompt(
                    qs,
                    pr,
                    p_new,
                    p_new_txt,
                    prompt_next,
                    next_fact_context,
                    args,
                ),
                p_new_txt,
                next_fact_context,
            ),
            beam_score,
        )
        qs.event(
            events_file,
            kind='candidate_beam_add',
            depth=depth,
            raw=raw,
            translation=translation,
            candidate_beam_score=beam_score,
            candidate_beam_score_strategy=args.candidate_beam_score,
            candidate_beam_progress_delta=progress_delta,
            candidate_beam_progress_bonus=candidate_beam_progress_bonus(
                args, progress_delta
            ),
            candidate_rerank_score=record.get('_candidate_rerank_score'),
            candidate_rerank_phase=record.get('_candidate_rerank_phase'),
            candidate_source=record.get('source', 'lm'),
            candidate_construction_type=qs.construction_type_key(translation),
            candidate_depth_rank=record.get('_candidate_depth_rank'),
            candidate_depth_eval_phase=record.get('_candidate_depth_eval_phase'),
        )
      timeout_items = []
      for item in run_candidate_tasks_parallel(parallel_tasks, args):
        if item.get('error'):
          qs.event(
              events_file,
              kind='candidate_ddar_error',
              depth=depth,
              raw=item['raw'],
              translation=item['translation'],
              error=item['error'],
              candidate_rerank_score=item.get('candidate_rerank_score'),
              candidate_rerank_phase=item.get('candidate_rerank_phase'),
              candidate_source=item.get('candidate_source'),
              candidate_construction_type=item.get('candidate_construction_type'),
              candidate_depth_rank=item.get('candidate_depth_rank'),
              candidate_depth_eval_phase=item.get('candidate_depth_eval_phase'),
              traceback=item.get('traceback'),
              elapsed_sec=item.get('elapsed_sec'),
              requested_timeout=item.get('requested_timeout'),
              effective_timeout=item.get('effective_timeout'),
              wall_timeout=item.get('wall_timeout'),
          )
          if is_candidate_timeout_error(item):
            timeout_items.append(item)
          continue
        result = item['result']
        qs.event(events_file, kind='ddar_done', **result)
        maybe_log_candidate_sft_signal(
            qs,
            events_file,
            p.url,
            depth,
            item['raw'],
            item['translation'],
            item.get('prompt'),
            item['p_new_txt'],
            root,
            result,
            args,
            candidate_source=item.get('candidate_source') or 'lm',
            candidate_construction_type=item.get('candidate_construction_type'),
            candidate_rerank_score=item.get('candidate_rerank_score'),
            candidate_rerank_phase=item.get('candidate_rerank_phase'),
        )
        if result['solved']:
          qs.event(
              events_file,
              kind='solved',
              depth=depth,
              aux=item['translation'],
              candidate_depth_rank=item.get('candidate_depth_rank'),
              candidate_depth_eval_phase=item.get('candidate_depth_eval_phase'),
              candidate_rerank_score=item.get('candidate_rerank_score'),
              candidate_rerank_phase=item.get('candidate_rerank_phase'),
              candidate_source=item.get('candidate_source') or 'lm',
              candidate_construction_type=item.get('candidate_construction_type'),
          )
          row.update({
              'solved': True,
              'solved_depth': depth,
              'aux': item['translation'],
          })
          return row
        next_fact_context = merge_fact_context(
            item.get('parent_fact_context'),
            result.get('fact_context'),
            args.lm_fact_context_top_k,
        )
        progress_delta = candidate_progress_delta(root, result)
        beam_score = candidate_beam_score(
            item['prev_score'],
            item['lm_score'],
            item.get('candidate_rerank_score'),
            args,
            progress_delta,
        )
        next_beam.add(
            make_beam_state(
                None,
                build_next_prompt(
                    qs,
                    pr,
                    None,
                    item['p_new_txt'],
                    item['prompt_next'],
                    next_fact_context,
                    args,
                ),
                item['p_new_txt'],
                next_fact_context,
            ),
            beam_score,
        )
        qs.event(
            events_file,
            kind='candidate_beam_add',
            depth=depth,
            raw=item['raw'],
            translation=item['translation'],
            candidate_beam_score=beam_score,
            candidate_beam_score_strategy=args.candidate_beam_score,
            candidate_beam_progress_delta=progress_delta,
            candidate_beam_progress_bonus=candidate_beam_progress_bonus(
                args, progress_delta
            ),
            candidate_rerank_score=item.get('candidate_rerank_score'),
            candidate_rerank_phase=item.get('candidate_rerank_phase'),
            candidate_source=item.get('candidate_source') or 'lm',
            candidate_construction_type=item.get('candidate_construction_type'),
            candidate_depth_rank=item.get('candidate_depth_rank'),
            candidate_depth_eval_phase=item.get('candidate_depth_eval_phase'),
        )
      maybe_add_timeout_beam_fallback(
          qs,
          pr,
          events_file,
          next_beam,
          timeout_items,
          args,
      )
      beam = next_beam
      if len(beam) == 0:
        qs.event(events_file, kind='beam_empty', depth=depth)
        break
      continue
    for prev_score, beam_state in limited_beam_nodes(
        qs,
        events_file,
        beam,
        args,
        depth,
    ):
      g_cur, prompt, pstring, parent_fact_context = unpack_beam_state(beam_state)
      p_cur = pr.Problem.from_txt(pstring, translate=False)
      if g_cur is None:
        g_cur, _ = build_graph_for_symbolic_search(gh, p_cur, qs.DEFINITIONS)
      qs.event(
          events_file,
          kind='decode_start',
          depth=depth,
          score=prev_score,
          prompt_tail=prompt[-240:],
      )
      forbidden_points = (
          qs.existing_point_names(g_cur)
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
      seen_raw = {raw for raw, _ in candidates}
      candidate_sources = {raw: 'lm' for raw, _ in candidates}
      seen_generation_keys = {
          qs.candidate_generation_dedup_key(raw) for raw, _ in candidates
      }
      # Dedup within this proof state; pruned auxes may still help other branches.
      seen_candidate_keys: set[str] = set()
      if args.candidate_template_backfill and len(candidates) < args.num_return_sequences:
        needed = args.num_return_sequences - len(candidates)
        for raw in qs.template_backfill_candidates(
            forbidden_points,
            needed * 4,
            seen_generation_keys if args.candidate_canonical_dedup else None,
            qs.goal_point_names(p_cur),
        ):
          generation_key = qs.candidate_generation_dedup_key(raw)
          if raw not in seen_raw and generation_key not in seen_generation_keys:
            candidates.append((raw, 0.0))
            seen_raw.add(raw)
            candidate_sources[raw] = 'template_initial_backfill'
            seen_generation_keys.add(generation_key)
          if len(candidates) >= args.num_return_sequences:
            break
      parallel_tasks = []
      translated_candidates = []
      for raw, lm_score in candidates:
        source = candidate_sources.get(raw, 'lm')
        translation = qs.try_translate_candidate(raw, g_cur, pr, pt)
        qs.event(
            events_file,
            kind='candidate',
            depth=depth,
            raw=raw,
            translation=translation,
            lm_score=lm_score,
            source=source,
        )
        maybe_log_candidate_hard_negative_signal(
            qs,
            events_file,
            p.url,
            depth,
            raw,
            translation,
            prompt,
            args,
            source=source,
        )
        if translation.startswith('ERROR:'):
          continue
        canonical_key = qs.canonical_aux_key(translation)
        if args.candidate_canonical_dedup and canonical_key in seen_candidate_keys:
          qs.event(
              events_file,
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
            'problem': p.url,
            'source': source,
        })
      if args.candidate_template_backfill:
        post_canonical_template_backfill(
            qs,
            events_file,
            p.url,
            depth,
            prompt,
            translated_candidates,
            forbidden_points,
            seen_candidate_keys,
            seen_raw,
            candidate_unique_backfill_target(args),
            g_cur,
            pr,
            pt,
            qs.goal_point_names(p_cur),
        )
      ranked_candidates = qs.rerank_candidate_records(
          translated_candidates,
          args.candidate_rerank,
          value_model,
          secondary_value_model,
          args.candidate_frontfill_limit,
      )
      if args.candidate_eval_limit and args.candidate_eval_limit > 0:
        for record in ranked_candidates[args.candidate_eval_limit:]:
          qs.event(
              events_file,
              kind='candidate_filtered',
              depth=depth,
              raw=record['raw'],
              translation=record['translation'],
              reason='rank_pruned',
              candidate_rerank=args.candidate_rerank,
              candidate_rerank_score=record.get('_candidate_rerank_score'),
              candidate_rerank_phase=record.get('_candidate_rerank_phase'),
              candidate_construction_type=qs.construction_type_key(
                  record['translation']
              ),
              candidate_eval_limit=args.candidate_eval_limit,
              source=record.get('source', 'lm'),
          )
        ranked_candidates = ranked_candidates[:args.candidate_eval_limit]
      for record in ranked_candidates:
        raw = record['raw']
        lm_score = record['lm_score']
        translation = record['translation']
        p_new_txt = qs.insert_aux_to_premise(pstring, translation)
        prompt_next = prompt + ' ' + raw + ' x00'
        if args.candidate_ddar_workers > 1:
          parallel_tasks.append(make_candidate_task(
              len(parallel_tasks),
              depth,
              raw,
              lm_score,
              translation,
              p_new_txt,
              prompt_next,
              candidate_max_level,
              candidate_solve_timeout,
              candidate_wall_timeout,
              args.lm_fact_context_top_k,
              requested_timeout=candidate_timeout,
              soft_timeout_margin_sec=args.candidate_soft_timeout_margin_sec,
              prompt=prompt,
              candidate_rerank_score=record.get('_candidate_rerank_score'),
              candidate_rerank_phase=record.get('_candidate_rerank_phase'),
              candidate_source=record.get('source', 'lm'),
              candidate_construction_type=qs.construction_type_key(translation),
              candidate_depth_rank=record.get('_candidate_depth_rank'),
              candidate_depth_eval_phase=record.get('_candidate_depth_eval_phase'),
              parent_fact_context=parent_fact_context,
          ))
          parallel_tasks[-1]['prev_score'] = prev_score
          continue
        p_new = pr.Problem.from_txt(p_new_txt, translate=False)
        g_new, _ = build_graph_for_symbolic_search(gh, p_new, qs.DEFINITIONS)
        result = qs.run_ddar_once(
            g_new,
            p_new,
            ddar,
            gh,
            candidate_max_level,
            candidate_solve_timeout,
            events_file,
            f'depth{depth}:{raw}',
            args.lm_fact_context_top_k,
            {qs.candidate_new_point(raw)} if qs.candidate_new_point(raw) else None,
        )
        result['requested_timeout'] = candidate_timeout
        result['effective_timeout'] = candidate_solve_timeout
        result['wall_timeout'] = candidate_wall_timeout
        maybe_log_candidate_sft_signal(
            qs,
            events_file,
            p.url,
            depth,
            raw,
            translation,
            prompt,
            p_new_txt,
            root,
            result,
            args,
            candidate_source=record.get('source', 'lm'),
            candidate_construction_type=qs.construction_type_key(translation),
            candidate_rerank_score=record.get('_candidate_rerank_score'),
            candidate_rerank_phase=record.get('_candidate_rerank_phase'),
        )
        if result['solved']:
          qs.event(
              events_file,
              kind='solved',
              depth=depth,
              aux=translation,
              candidate_depth_rank=record.get('_candidate_depth_rank'),
              candidate_depth_eval_phase=record.get('_candidate_depth_eval_phase'),
              candidate_rerank_score=record.get('_candidate_rerank_score'),
              candidate_rerank_phase=record.get('_candidate_rerank_phase'),
              candidate_source=record.get('source', 'lm'),
              candidate_construction_type=qs.construction_type_key(translation),
          )
          row.update({
              'solved': True,
              'solved_depth': depth,
              'aux': translation,
          })
          return row
        next_fact_context = merge_fact_context(
            parent_fact_context,
            result.get('fact_context'),
            args.lm_fact_context_top_k,
        )
        progress_delta = candidate_progress_delta(root, result)
        beam_score = candidate_beam_score(
            prev_score,
            lm_score,
            record.get('_candidate_rerank_score'),
            args,
            progress_delta,
        )
        next_beam.add(
            make_beam_state(
                g_new,
                build_next_prompt(
                    qs,
                    pr,
                    p_new,
                    p_new_txt,
                    prompt_next,
                    next_fact_context,
                    args,
                ),
                p_new_txt,
                next_fact_context,
            ),
            beam_score,
        )
        qs.event(
            events_file,
            kind='candidate_beam_add',
            depth=depth,
            raw=raw,
            translation=translation,
            candidate_beam_score=beam_score,
            candidate_beam_score_strategy=args.candidate_beam_score,
            candidate_beam_progress_delta=progress_delta,
            candidate_beam_progress_bonus=candidate_beam_progress_bonus(
                args, progress_delta
            ),
            candidate_rerank_score=record.get('_candidate_rerank_score'),
            candidate_rerank_phase=record.get('_candidate_rerank_phase'),
            candidate_source=record.get('source', 'lm'),
            candidate_construction_type=qs.construction_type_key(translation),
            candidate_depth_rank=record.get('_candidate_depth_rank'),
            candidate_depth_eval_phase=record.get('_candidate_depth_eval_phase'),
        )
      timeout_items = []
      for item in run_candidate_tasks_parallel(parallel_tasks, args):
        if item.get('error'):
          qs.event(
              events_file,
              kind='candidate_ddar_error',
              depth=depth,
              raw=item['raw'],
              translation=item['translation'],
              error=item['error'],
              candidate_rerank_score=item.get('candidate_rerank_score'),
              candidate_rerank_phase=item.get('candidate_rerank_phase'),
              candidate_source=item.get('candidate_source'),
              candidate_construction_type=item.get('candidate_construction_type'),
              candidate_depth_rank=item.get('candidate_depth_rank'),
              candidate_depth_eval_phase=item.get('candidate_depth_eval_phase'),
              traceback=item.get('traceback'),
              elapsed_sec=item.get('elapsed_sec'),
              requested_timeout=item.get('requested_timeout'),
              effective_timeout=item.get('effective_timeout'),
              wall_timeout=item.get('wall_timeout'),
          )
          if is_candidate_timeout_error(item):
            timeout_items.append(item)
          continue
        result = item['result']
        qs.event(events_file, kind='ddar_done', **result)
        maybe_log_candidate_sft_signal(
            qs,
            events_file,
            p.url,
            depth,
            item['raw'],
            item['translation'],
            item.get('prompt'),
            item['p_new_txt'],
            root,
            result,
            args,
            candidate_source=item.get('candidate_source') or 'lm',
            candidate_construction_type=item.get('candidate_construction_type'),
            candidate_rerank_score=item.get('candidate_rerank_score'),
            candidate_rerank_phase=item.get('candidate_rerank_phase'),
        )
        if result['solved']:
          qs.event(
              events_file,
              kind='solved',
              depth=depth,
              aux=item['translation'],
              candidate_depth_rank=item.get('candidate_depth_rank'),
              candidate_depth_eval_phase=item.get('candidate_depth_eval_phase'),
              candidate_rerank_score=item.get('candidate_rerank_score'),
              candidate_rerank_phase=item.get('candidate_rerank_phase'),
              candidate_source=item.get('candidate_source') or 'lm',
              candidate_construction_type=item.get('candidate_construction_type'),
          )
          row.update({
              'solved': True,
              'solved_depth': depth,
              'aux': item['translation'],
          })
          return row
        next_fact_context = merge_fact_context(
            item.get('parent_fact_context'),
            result.get('fact_context'),
            args.lm_fact_context_top_k,
        )
        progress_delta = candidate_progress_delta(root, result)
        beam_score = candidate_beam_score(
            item['prev_score'],
            item['lm_score'],
            item.get('candidate_rerank_score'),
            args,
            progress_delta,
        )
        next_beam.add(
            make_beam_state(
                None,
                build_next_prompt(
                    qs,
                    pr,
                    None,
                    item['p_new_txt'],
                    item['prompt_next'],
                    next_fact_context,
                    args,
                ),
                item['p_new_txt'],
                next_fact_context,
            ),
            beam_score,
        )
        qs.event(
            events_file,
            kind='candidate_beam_add',
            depth=depth,
            raw=item['raw'],
            translation=item['translation'],
            candidate_beam_score=beam_score,
            candidate_beam_score_strategy=args.candidate_beam_score,
            candidate_beam_progress_delta=progress_delta,
            candidate_beam_progress_bonus=candidate_beam_progress_bonus(
                args, progress_delta
            ),
            candidate_rerank_score=item.get('candidate_rerank_score'),
            candidate_rerank_phase=item.get('candidate_rerank_phase'),
            candidate_source=item.get('candidate_source') or 'lm',
            candidate_construction_type=item.get('candidate_construction_type'),
            candidate_depth_rank=item.get('candidate_depth_rank'),
            candidate_depth_eval_phase=item.get('candidate_depth_eval_phase'),
        )
      maybe_add_timeout_beam_fallback(
          qs,
          pr,
          events_file,
          next_beam,
          timeout_items,
          args,
      )
    beam = next_beam
    if len(beam) == 0:
      qs.event(events_file, kind='beam_empty', depth=depth)
      break
  return row


def parse_problem_names(value: str | None) -> set[str] | None:
  if not value:
    return None
  return {x.strip() for x in value.split(',') if x.strip()}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--script_dir', required=True)
  parser.add_argument('--ag_repo', required=True)
  parser.add_argument('--problems_file', required=True)
  parser.add_argument('--defs_file', required=True)
  parser.add_argument('--rules_file', required=True)
  parser.add_argument('--out_dir', required=True)
  parser.add_argument('--mode', choices=['ddar', 'qwen'], default='ddar')
  parser.add_argument('--problem_names')
  parser.add_argument('--limit', type=int)
  parser.add_argument('--translate', action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument('--max_level', type=int, default=1000)
  parser.add_argument('--ddar_timeout', type=int, default=600)
  parser.add_argument('--root_max_level', type=int)
  parser.add_argument('--root_ddar_timeout', type=int)
  parser.add_argument('--candidate_max_level', type=int)
  parser.add_argument('--candidate_ddar_timeout', type=int)
  parser.add_argument(
      '--candidate_wall_timeout',
      type=int,
      help='hard wall-clock timeout for each candidate DDAR worker',
  )
  parser.add_argument(
      '--candidate_soft_timeout_margin_sec',
      type=int,
      default=5,
      help=(
          'when a candidate wall timeout is active, reduce DDAR timeout by up '
          'to this many seconds so DDAR can return partial facts before SIGALRM; '
          '0 disables'
      ),
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
      '--candidate_depth_type_eval_cap',
      type=int,
      default=0,
      help=(
          'delay candidates from a construction type after this many selected '
          'depth-level eval slots; 0 disables'
      ),
  )
  parser.add_argument(
      '--candidate_depth_tail_eval_slots',
      type=int,
      default=0,
      help=(
          'reserve this many depth-level DDAR eval slots for evenly sampled '
          'low-rank candidates beyond --candidate_depth_eval_limit; 0 disables'
      ),
  )
  parser.add_argument(
      '--candidate_depth_tail_eval_strategy',
      choices=['even', 'near_spread'],
      default='even',
      help=(
          'tail slot selection strategy: even samples the ranked tail; '
          'near_spread keeps candidates just after the cutoff and spreads the rest'
      ),
  )
  parser.add_argument(
      '--candidate_timeout_beam_fallback_limit',
      type=int,
      default=0,
      help=(
          'when all parallel candidate DDAR checks at a depth time out, carry '
          'top-N timed-out candidates into the next beam without fact context; '
          '0 disables'
      ),
  )
  parser.add_argument(
      '--candidate_timeout_beam_fallback_mode',
      choices=['empty', 'append'],
      default='empty',
      help=(
          'empty keeps the old behavior and only uses timeout fallback when the '
          'next beam is empty; append also lets timed-out candidates compete '
          'with completed candidates in the next beam'
      ),
  )
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
      '--candidate_beam_score',
      choices=[
          'lm_score',
          'rerank_score',
          'lm_plus_rerank',
          'rerank_plus_progress',
      ],
      default='lm_score',
      help='score used to keep/order candidate states in the next search beam',
  )
  parser.add_argument(
      '--candidate_beam_progress_weight',
      type=float,
      default=0.0,
      help='log-scaled DDAR progress weight for rerank_plus_progress beam score',
  )
  parser.add_argument(
      '--candidate_beam_progress_cap',
      type=float,
      default=4.0,
      help='maximum DDAR progress bonus added by rerank_plus_progress; <=0 disables',
  )
  parser.add_argument(
      '--candidate_decode_beam_limit',
      type=int,
      default=0,
      help='decode only the top-N current beam states per depth; 0 disables',
  )
  parser.add_argument(
      '--candidate_ddar_workers',
      type=int,
      default=1,
      help='parallel CPU workers for candidate DDAR checks; 1 keeps serial order',
  )
  parser.add_argument(
      '--lm_fact_context_top_k',
      type=int,
      default=0,
      help='prepend top-K DDAR-added facts to LM prompts; 0 disables',
  )
  parser.add_argument(
      '--log_candidate_sft_signals',
      action=argparse.BooleanOptionalAction,
      default=True,
      help='log verifier-backed prompt/target rows for later auxiliary SFT mining',
  )
  parser.add_argument(
      '--disable_candidate_sft_progress_signals',
      action='store_true',
      help='log only solved candidates as SFT signals, not DDAR-progress positives',
  )
  parser.add_argument(
      '--candidate_sft_signal_min_added_dependencies',
      type=int,
      default=10,
      help='minimum candidate DDAR additions for progress-positive SFT signals',
  )
  parser.add_argument(
      '--candidate_sft_signal_min_root_delta',
      type=int,
      default=1,
      help='minimum candidate-root added-dependency delta for progress-positive SFT signals',
  )
  parser.add_argument(
      '--candidate_sft_signal_max_elapsed_sec',
      type=float,
      default=120.0,
      help='maximum candidate DDAR elapsed time for progress-positive SFT signals; <=0 disables',
  )
  parser.add_argument(
      '--log_candidate_hard_negative_signals',
      action=argparse.BooleanOptionalAction,
      default=True,
      help='log prompt/raw rows for generator-side hard negatives from invalid constructions',
  )
  parser.add_argument(
      '--candidate_hard_negative_signal_reasons',
      default='point_too_close,point_too_far,point_already_exists,unknown_point',
      help='comma-separated invalid-construction reasons to log as hard-negative signals',
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.mode == 'qwen' and not args.qwen_model:
    raise ValueError('--qwen_model is required in qwen mode')

  qs = import_search(args.script_dir)
  qs.add_ag_repo_to_path(args.ag_repo)
  ag = qs.load_ag_modules()
  pr, gh, ddar, pt = ag['pr'], ag['gh'], ag['ddar'], ag['pt']
  qs.DEFINITIONS = pr.Definition.from_txt_file(args.defs_file, to_dict=True)
  qs.RULES = pr.Theorem.from_txt_file(args.rules_file, to_dict=True)
  args._candidate_value_model = qs.load_candidate_value_model(
      args.candidate_value_model
  )
  args._candidate_secondary_value_model = qs.load_candidate_value_model(
      args.candidate_secondary_value_model
  )

  problems = pr.Problem.from_txt_file(
      args.problems_file, to_dict=True, translate=args.translate
  )
  selected = parse_problem_names(args.problem_names)
  names = [n for n in problems if selected is None or n in selected]
  if args.limit:
    names = names[: args.limit]

  out_dir = Path(args.out_dir)
  events_dir = out_dir / 'events'
  events_dir.mkdir(parents=True, exist_ok=True)
  summary_path = out_dir / 'summary.jsonl'

  generator = None
  if args.mode == 'qwen':
    generator = qs.QwenGenerator(
        args.qwen_model, args.adapter_path, args.dtype, args.device_map
    )

  solved = 0
  with open(summary_path, 'w', encoding='utf-8') as summary:
    for i, name in enumerate(names, 1):
      events_file = str(events_dir / f'{safe_name(name)}.jsonl')
      p = problems[name]
      try:
        row = solve_one(p, qs, pr, gh, ddar, pt, generator, args, events_file)
      except Exception as exc:  # pylint: disable=broad-except
        row = {
            'problem': p.url,
            'mode': args.mode,
            'root_solved': False,
            'solved': False,
            'solved_depth': None,
            'aux': None,
            'events_file': events_file,
            'root_levels': None,
            'root_cache_items': None,
            'error': type(exc).__name__,
            'traceback': traceback.format_exc(),
        }
      row.update({'index': i, 'name': name})
      solved += int(row['solved'])
      summary.write(json.dumps(row, ensure_ascii=False) + '\n')
      summary.flush()
      print(json.dumps({
          'index': i,
          'name': name,
          'solved': row['solved'],
          'solved_total': solved,
          'total_seen': i,
      }), flush=True)

  with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
    json.dump(
        {
            'mode': args.mode,
            'num_problems': len(names),
            'solved': solved,
            'summary_jsonl': str(summary_path),
        },
        f,
        ensure_ascii=False,
        indent=2,
    )


if __name__ == '__main__':
  main()
