"""Run DDAR or Qwen+DDAR over an AG problem file and write a summary.

This benchmark driver reuses ``qwen_ag_search.py`` and loads Qwen only once.
Detailed per-problem events are written under ``out_dir/events``; one JSONL
summary row is written to ``out_dir/summary.jsonl``.
"""

from __future__ import annotations

import argparse
import json
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
    prompt: str | None = None,
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
      'wall_timeout': wall_timeout,
      'fact_context_top_k': fact_context_top_k,
      'fact_context_recent_points': [new_point] if new_point else [],
      'tag': f'depth{depth}:{raw}',
  }


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
  )


def maybe_log_candidate_hard_negative_signal(
    qs: Any,
    events_file: str,
    problem_name: str,
    depth: int,
    raw: str,
    translation: str,
    prompt: str | None,
    args: argparse.Namespace,
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
  )


def candidate_unique_backfill_target(args: argparse.Namespace) -> int:
  if args.candidate_eval_limit and args.candidate_eval_limit > 0:
    return max(1, min(args.num_return_sequences, args.candidate_eval_limit))
  if args.candidate_depth_eval_limit and args.candidate_depth_eval_limit > 0:
    return max(1, min(args.num_return_sequences, args.candidate_depth_eval_limit))
  return max(1, args.num_return_sequences)


def post_canonical_template_backfill(
    qs: Any,
    events_file: str,
    depth: int,
    translated_candidates: list[dict[str, Any]],
    forbidden_points: set[str] | None,
    seen_candidate_keys: set[str],
    seen_raw: set[str],
    target_count: int,
    g_cur: Any,
    pr: Any,
    pt: Any,
) -> None:
  if len(translated_candidates) >= target_count:
    return
  start_len = len(translated_candidates)
  attempted = 0
  needed = target_count - len(translated_candidates)
  template_budget = max(target_count * 32, needed * 16, 128)
  for raw in qs.template_backfill_candidates(forbidden_points, template_budget):
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
          source='template_post_canonical_backfill',
      )
      continue
    seen_candidate_keys.add(canonical_key)
    translated_candidates.append({
        'raw': raw,
        'lm_score': 0.0,
        'translation': translation,
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
  beam = qs.BeamQueue(args.beam_size)
  beam.add((g, prompt0, p.txt()), 0.0)
  candidate_max_level = args.candidate_max_level or args.max_level
  candidate_timeout = args.candidate_ddar_timeout or args.ddar_timeout
  candidate_wall_timeout = args.candidate_wall_timeout or candidate_timeout
  seen_candidate_keys: set[str] = set()
  value_model = getattr(args, '_candidate_value_model', None)

  for depth in range(args.search_depth):
    qs.event(events_file, kind='depth_start', depth=depth, nodes=len(beam))
    next_beam = qs.BeamQueue(args.beam_size)
    if args.candidate_depth_eval_limit and args.candidate_depth_eval_limit > 0:
      depth_candidates = []
      for node_index, (prev_score, (g_cur, prompt, pstring)) in enumerate(
          beam.ordered()
      ):
        if g_cur is None:
          p_cur = pr.Problem.from_txt(pstring, translate=False)
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
        if args.candidate_template_backfill and len(candidates) < args.num_return_sequences:
          needed = args.num_return_sequences - len(candidates)
          for raw in qs.template_backfill_candidates(forbidden_points, needed * 4):
            if raw not in seen_raw:
              candidates.append((raw, 0.0))
              seen_raw.add(raw)
            if len(candidates) >= args.num_return_sequences:
              break
        translated_candidates = []
        for raw, lm_score in candidates:
          translation = qs.try_translate_candidate(raw, g_cur, pr, pt)
          qs.event(
              events_file,
              kind='candidate',
              depth=depth,
              raw=raw,
              translation=translation,
              lm_score=lm_score,
              source='lm',
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
                source='lm',
            )
            continue
          seen_candidate_keys.add(canonical_key)
          translated_candidates.append({
              'raw': raw,
              'lm_score': lm_score,
              'translation': translation,
              'source': 'lm',
          })
        if args.candidate_template_backfill:
          post_canonical_template_backfill(
              qs,
              events_file,
              depth,
              translated_candidates,
              forbidden_points,
              seen_candidate_keys,
              seen_raw,
              candidate_unique_backfill_target(args),
              g_cur,
              pr,
              pt,
          )
        ranked_node_candidates = qs.rerank_candidate_records(
            translated_candidates, args.candidate_rerank, value_model
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
          })
          depth_candidates.append(record)
      ranked_depth_candidates = qs.rerank_candidate_records(
          depth_candidates, args.candidate_rerank, value_model
      )
      for record in ranked_depth_candidates[args.candidate_depth_eval_limit:]:
        qs.event(
            events_file,
            kind='candidate_filtered',
            depth=depth,
            raw=record['raw'],
            translation=record['translation'],
            reason='depth_rank_pruned',
            candidate_rerank=args.candidate_rerank,
            candidate_depth_eval_limit=args.candidate_depth_eval_limit,
            source=record.get('source', 'lm'),
        )
      parallel_tasks = []
      for record in ranked_depth_candidates[:args.candidate_depth_eval_limit]:
        raw = record['raw']
        lm_score = record['lm_score']
        translation = record['translation']
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
              candidate_timeout,
              candidate_wall_timeout,
              args.lm_fact_context_top_k,
              prompt=record['prompt'],
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
            candidate_timeout,
            events_file,
            f'depth{depth}:{raw}',
            args.lm_fact_context_top_k,
            {qs.candidate_new_point(raw)} if qs.candidate_new_point(raw) else None,
        )
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
        )
        if result['solved']:
          qs.event(events_file, kind='solved', depth=depth, aux=translation)
          row.update({
              'solved': True,
              'solved_depth': depth,
              'aux': translation,
          })
          return row
        next_beam.add(
            (
                g_new,
                qs.build_lm_prompt(
                    p_new,
                    qs.DEFINITIONS,
                    result.get('fact_context')
                    if args.lm_fact_context_top_k
                    else None,
                )
                if args.lm_fact_context_top_k
                else prompt_next,
                p_new_txt,
            ),
            record['prev_score'] + lm_score,
        )
      for item in run_candidate_tasks_parallel(parallel_tasks, args):
        if item.get('error'):
          qs.event(
              events_file,
              kind='candidate_ddar_error',
              depth=depth,
              raw=item['raw'],
              translation=item['translation'],
              error=item['error'],
              traceback=item.get('traceback'),
              elapsed_sec=item.get('elapsed_sec'),
          )
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
        )
        if result['solved']:
          qs.event(
              events_file,
              kind='solved',
              depth=depth,
              aux=item['translation'],
          )
          row.update({
              'solved': True,
              'solved_depth': depth,
              'aux': item['translation'],
          })
          return row
        next_beam.add(
            (
                None,
                qs.build_lm_prompt_from_problem_text(
                    item['p_new_txt'],
                    pr,
                    qs.DEFINITIONS,
                    result.get('fact_context')
                    if args.lm_fact_context_top_k
                    else None,
                )
                if args.lm_fact_context_top_k
                else item['prompt_next'],
                item['p_new_txt'],
            ),
            item['prev_score'] + item['lm_score'],
        )
      beam = next_beam
      if len(beam) == 0:
        qs.event(events_file, kind='beam_empty', depth=depth)
        break
      continue
    for prev_score, (g_cur, prompt, pstring) in beam.ordered():
      if g_cur is None:
        p_cur = pr.Problem.from_txt(pstring, translate=False)
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
      if args.candidate_template_backfill and len(candidates) < args.num_return_sequences:
        needed = args.num_return_sequences - len(candidates)
        for raw in qs.template_backfill_candidates(forbidden_points, needed * 4):
          if raw not in seen_raw:
            candidates.append((raw, 0.0))
            seen_raw.add(raw)
          if len(candidates) >= args.num_return_sequences:
            break
      parallel_tasks = []
      translated_candidates = []
      for raw, lm_score in candidates:
        translation = qs.try_translate_candidate(raw, g_cur, pr, pt)
        qs.event(
            events_file,
            kind='candidate',
            depth=depth,
            raw=raw,
            translation=translation,
            lm_score=lm_score,
            source='lm',
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
              source='lm',
          )
          continue
        seen_candidate_keys.add(canonical_key)
        translated_candidates.append({
            'raw': raw,
            'lm_score': lm_score,
            'translation': translation,
            'source': 'lm',
        })
      if args.candidate_template_backfill:
        post_canonical_template_backfill(
            qs,
            events_file,
            depth,
            translated_candidates,
            forbidden_points,
            seen_candidate_keys,
            seen_raw,
            candidate_unique_backfill_target(args),
            g_cur,
            pr,
            pt,
        )
      ranked_candidates = qs.rerank_candidate_records(
          translated_candidates, args.candidate_rerank, value_model
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
              candidate_timeout,
              candidate_wall_timeout,
              args.lm_fact_context_top_k,
              prompt=prompt,
          ))
          continue
        p_new = pr.Problem.from_txt(p_new_txt, translate=False)
        g_new, _ = build_graph_for_symbolic_search(gh, p_new, qs.DEFINITIONS)
        result = qs.run_ddar_once(
            g_new,
            p_new,
            ddar,
            gh,
            candidate_max_level,
            candidate_timeout,
            events_file,
            f'depth{depth}:{raw}',
            args.lm_fact_context_top_k,
            {qs.candidate_new_point(raw)} if qs.candidate_new_point(raw) else None,
        )
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
        )
        if result['solved']:
          qs.event(events_file, kind='solved', depth=depth, aux=translation)
          row.update({
              'solved': True,
              'solved_depth': depth,
              'aux': translation,
          })
          return row
        next_beam.add(
            (
                g_new,
                qs.build_lm_prompt(
                    p_new,
                    qs.DEFINITIONS,
                    result.get('fact_context')
                    if args.lm_fact_context_top_k
                    else None,
                )
                if args.lm_fact_context_top_k
                else prompt_next,
                p_new_txt,
            ),
            prev_score + lm_score,
        )
      for item in run_candidate_tasks_parallel(parallel_tasks, args):
        if item.get('error'):
          qs.event(
              events_file,
              kind='candidate_ddar_error',
              depth=depth,
              raw=item['raw'],
              translation=item['translation'],
              error=item['error'],
              traceback=item.get('traceback'),
              elapsed_sec=item.get('elapsed_sec'),
          )
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
        )
        if result['solved']:
          qs.event(
              events_file,
              kind='solved',
              depth=depth,
              aux=item['translation'],
          )
          row.update({
              'solved': True,
              'solved_depth': depth,
              'aux': item['translation'],
          })
          return row
      next_beam.add(
          (
              None,
              qs.build_lm_prompt_from_problem_text(
                  item['p_new_txt'],
                  pr,
                  qs.DEFINITIONS,
                  result.get('fact_context')
                  if args.lm_fact_context_top_k
                  else None,
              )
              if args.lm_fact_context_top_k
              else item['prompt_next'],
              item['p_new_txt'],
          ),
          prev_score + item['lm_score'],
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
      choices=['none', 'balanced_constrained', 'mixed_constructive'],
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
      choices=['none', 'heuristic_diverse', 'value_model', 'value_model_diverse'],
      default='none',
      help='optional translated-candidate reranker before DDAR evaluation',
  )
  parser.add_argument(
      '--candidate_value_model',
      help='JSON value model for value_model candidate reranking',
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
