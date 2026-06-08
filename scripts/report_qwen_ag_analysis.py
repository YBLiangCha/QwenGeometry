#!/usr/bin/env python3
"""Render a compact Markdown report from Qwen+AG event analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DIAGNOSIS_TEXT = {
    'solved': 'solved',
    'no_candidates_generated': 'no candidates generated',
    'high_invalid_rate': 'high invalid candidate rate',
    'duplicate_collapse': 'canonical duplicate collapse',
    'candidate_ddar_timeout_blocked': 'candidate DDAR timeout blocked',
    'candidate_ddar_timeouts': 'candidate DDAR timeouts',
    'symbolic_progress_but_wrong_direction': (
        'DDAR makes facts but misses the goal'
    ),
    'valid_candidates_not_evaluated': 'valid candidates not evaluated',
    'template_backfill_exhausted': 'template backfill exhausted',
    'search_exhausted_no_goal': 'search exhausted without goal',
}


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding='utf-8'))


def fmt_count_map(values: dict[str, Any] | None, limit: int = 8) -> str:
  if not values:
    return '-'
  items = list(values.items())[:limit]
  return ', '.join(f'{key}:{value}' for key, value in items) or '-'


def fmt_rate(num: int, den: int) -> str:
  return f'{num / den:.1%}' if den else '0.0%'


def fmt_num_summary(values: dict[str, Any] | None) -> str:
  if not values or not values.get('count'):
    return '-'
  parts = [f"n={values.get('count')}"]
  for key in ('median', 'mean', 'max'):
    if key in values:
      parts.append(f"{key}={values.get(key)}")
  return ', '.join(parts)


def diagnosis_text(labels: list[str]) -> str:
  return ', '.join(DIAGNOSIS_TEXT.get(label, label) for label in labels) or '-'


def phase_count(problem: dict[str, Any], phase: str) -> int:
  return int((problem.get('depth_eval_selected_phases') or {}).get(phase, 0) or 0)


def problem_line(problem: dict[str, Any]) -> str:
  filtered = problem.get('filtered_reasons') or {}
  candidate_errors = problem.get('candidate_ddar_error_types') or {}
  max_added = problem.get('max_added_candidate') or {}
  timeout_errors = sum(
      count
      for key, count in candidate_errors.items()
      if 'timeout' in str(key).lower()
  )
  return (
      f"| {problem.get('problem')} "
      f"| {'Y' if problem.get('solved') else 'N'} "
      f"| {problem.get('candidates', 0)} "
      f"| {problem.get('valid_candidates', 0)} "
      f"| {problem.get('depth_eval_selected', 0)} "
      f"| {phase_count(problem, 'tail_rank_coverage')} "
      f"| {problem.get('adaptive_penalized_records', 0)} "
      f"| {filtered.get('depth_rank_pruned', 0)} "
      f"| {problem.get('candidate_ddar_done', 0)} "
      f"| {timeout_errors} "
      f"| {problem.get('candidate_timeout_beam_fallbacks', 0)} "
      f"| {max_added.get('added_dependencies', 0) or 0}"
      f"/{max_added.get('construction_type') or '-'} "
      f"| {diagnosis_text(problem.get('diagnosis') or [])} |"
  )


def solved_detail(problem: dict[str, Any]) -> str:
  event = problem.get('solved_event') or {}
  phase = event.get('candidate_depth_eval_phase') or '-'
  rerank_phase = event.get('candidate_rerank_phase') or '-'
  rank = event.get('candidate_depth_rank')
  rank_text = '-' if rank is None else str(rank)
  penalty = event.get('candidate_adaptive_type_penalty')
  penalty_text = '-' if penalty in (None, 0) else str(penalty)
  aux_type = (
      event.get('candidate_construction_type')
      or problem.get('solved_aux_construction_type')
      or '-'
  )
  return (
      f"depth={problem.get('solved_depth')} rank={rank_text} "
      f"phase={phase} rerank={rerank_phase} adapt_penalty={penalty_text} "
      f"type={aux_type} "
      f"aux=`{problem.get('aux')}`"
  )


def render_report(payload: dict[str, Any]) -> str:
  aggregate = payload.get('aggregate') or {}
  problems = payload.get('problems') or []
  completed = [problem for problem in problems if problem.get('completed')]
  unsolved = [problem for problem in completed if not problem.get('solved')]
  solved = [problem for problem in completed if problem.get('solved')]
  new_solved_names = set(payload.get('new_solved_names') or [])
  new_solved = [
      problem for problem in solved if problem.get('problem') in new_solved_names
  ]
  candidates = int(aggregate.get('candidates') or 0)
  invalid = int(aggregate.get('invalid_candidates') or 0)
  filtered = int(aggregate.get('filtered_total') or 0)
  duplicate = int(aggregate.get('duplicate_canonical') or 0)

  lines = [
      '# Qwen+AG Failure Analysis',
      '',
  ]
  if new_solved:
    lines.extend([
        '## New Solved Vs Baseline',
        '',
        f"- NEW SOLVED COUNT: {len(new_solved)}",
    ])
    for problem in new_solved:
      lines.append(
          f"- NEW SOLVED: `{problem.get('problem')}` {solved_detail(problem)}"
      )
    lines.extend([
        f"- Baseline summary: `{payload.get('baseline_summary_jsonl')}`",
        '',
    ])

  lines.extend([
      '## Summary',
      '',
      f"- Output dir: `{payload.get('out_dir')}`",
      f"- Event files: {payload.get('num_event_files', 0)}",
      f"- Completed problems: {payload.get('num_completed', 0)}",
      f"- Solved problems: {payload.get('num_solved', 0)}",
      f"- Solved names: {', '.join(payload.get('solved_names') or []) or '-'}",
      f"- New solved vs baseline: "
      f"{', '.join(payload.get('new_solved_names') or []) or '-'}",
      f"- Candidates: {candidates}",
      f"- Valid / invalid: {aggregate.get('valid_candidates', 0)} / {invalid}"
      f" ({fmt_rate(invalid, candidates)} invalid)",
      f"- Filtered total: {filtered}",
      f"- Duplicate canonical: {duplicate}"
      f" ({fmt_rate(duplicate, candidates)} vs candidates)",
      f"- Candidate DDAR done / errors: "
      f"{aggregate.get('candidate_ddar_done', 0)} / "
      f"{aggregate.get('candidate_ddar_errors', 0)}",
      f"- Candidate SFT signals: {aggregate.get('candidate_sft_signals', 0)}",
      f"- Candidate hard-negative signals: "
      f"{aggregate.get('candidate_hard_negative_signals', 0)} "
      f"({fmt_count_map(aggregate.get('candidate_hard_negative_signal_reasons'))})",
      f"- Adaptive type failures: {aggregate.get('adaptive_type_failures', 0)} "
      f"({fmt_count_map(aggregate.get('adaptive_type_failure_reasons'))})",
      f"- Adaptive penalty applied / records: "
      f"{aggregate.get('adaptive_penalty_applied_events', 0)} / "
      f"{aggregate.get('adaptive_penalized_records', 0)}",
      '',
      '## Evaluation Coverage',
      '',
      f"- Depth-eval selected: {aggregate.get('depth_eval_selected', 0)}",
      f"- Selected phases: "
      f"{fmt_count_map(aggregate.get('depth_eval_selected_phases'))}",
      f"- Selected rerank phases: "
      f"{fmt_count_map(aggregate.get('depth_eval_selected_rerank_phases'))}",
      f"- Selected tail strategies: "
      f"{fmt_count_map(aggregate.get('depth_eval_selected_strategies'))}",
      f"- Selected rank bins: "
      f"{fmt_count_map(aggregate.get('depth_eval_selected_rank_bins'))}",
      f"- Selected construction top: "
      f"{fmt_count_map(aggregate.get('depth_eval_selected_construction_types_top'))}",
      f"- Filtered phases: {fmt_count_map(aggregate.get('filtered_eval_phases'))}",
      f"- Filtered rerank phases: "
      f"{fmt_count_map(aggregate.get('filtered_rerank_phases'))}",
      f"- Filtered rank bins: {fmt_count_map(aggregate.get('filtered_rank_bins'))}",
      f"- Beam-add phases: {fmt_count_map(aggregate.get('candidate_beam_add_phases'))}",
      f"- Beam-add rerank phases: "
      f"{fmt_count_map(aggregate.get('candidate_beam_add_rerank_phases'))}",
      f"- Beam-add rank bins: "
      f"{fmt_count_map(aggregate.get('candidate_beam_add_rank_bins'))}",
      '',
      '## Timeout Readout',
      '',
      f"- Timeout phases: {fmt_count_map(aggregate.get('timeout_eval_phases'))}",
      f"- Timeout rerank phases: "
      f"{fmt_count_map(aggregate.get('timeout_rerank_phases'))}",
      f"- Timeout rank bins: {fmt_count_map(aggregate.get('timeout_rank_bins'))}",
      f"- Timeout fallback count: "
      f"{aggregate.get('candidate_timeout_beam_fallbacks', 0)}",
      f"- Timeout fallback modes: "
      f"{fmt_count_map(aggregate.get('candidate_timeout_beam_fallback_modes'))}",
      f"- Timeout fallback phases: "
      f"{fmt_count_map(aggregate.get('candidate_timeout_beam_fallback_phases'))}",
      f"- Timeout fallback rerank phases: "
      f"{fmt_count_map(aggregate.get('candidate_timeout_beam_fallback_rerank_phases'))}",
      f"- Timeout fallback rank bins: "
      f"{fmt_count_map(aggregate.get('candidate_timeout_beam_fallback_rank_bins'))}",
      f"- Timeout fallback types: "
      f"{fmt_count_map(aggregate.get('candidate_timeout_beam_fallback_types_top'))}",
      '',
      '## Aggregate Diagnosis',
      '',
      fmt_count_map(payload.get('aggregate_diagnosis'), 20),
      '',
      '## Construction-Type Signals',
      '',
      f"- Generated valid top: {fmt_count_map(aggregate.get('construction_types_top'))}",
      f"- Evaluated top: "
      f"{fmt_count_map(aggregate.get('evaluated_construction_types_top'))}",
      f"- Filtered top: "
      f"{fmt_count_map(aggregate.get('filtered_construction_types_top'))}",
      f"- Timeout top: {fmt_count_map(aggregate.get('timeout_construction_types_top'))}",
      f"- Hard-negative top: "
      f"{fmt_count_map(aggregate.get('candidate_hard_negative_signal_types_top'))}",
      f"- Adaptive failure top: "
      f"{fmt_count_map(aggregate.get('adaptive_type_failure_types_top'))}",
      f"- Adaptive penalty top: "
      f"{fmt_count_map(aggregate.get('adaptive_penalty_top_types_top'))}",
      f"- Adaptive penalty stages: "
      f"{fmt_count_map(aggregate.get('adaptive_penalty_stages'))}",
      f"- Adaptive penalized selected top: "
      f"{fmt_count_map(aggregate.get('adaptive_penalized_selected_types_top'))}",
      f"- Adaptive penalized beam-add top: "
      f"{fmt_count_map(aggregate.get('adaptive_penalized_beam_add_types_top'))}",
      f"- Adaptive penalized SFT top: "
      f"{fmt_count_map(aggregate.get('adaptive_penalized_sft_signal_types_top'))}",
      f"- SFT signal rerank phases: "
      f"{fmt_count_map(aggregate.get('candidate_sft_signal_rerank_phases'))}",
      f"- DDAR error rerank phases: "
      f"{fmt_count_map(aggregate.get('candidate_error_rerank_phases'))}",
      f"- Solved aux top: "
      f"{fmt_count_map(aggregate.get('solved_aux_construction_types_top'))}",
      '',
      '## Optimization Readout',
      '',
      (
          '- If tail_rank_coverage appears in selected phases but never in '
          'beam-add/solved phases, tail candidates are being sampled but DDAR '
          'or reranking is not extracting useful branches.'
      ),
      (
          '- If selected rank bins are concentrated in 000-047 while filtered '
          'rank bins contain most candidates, increase tail slots or loosen the '
          'depth eval limit only on problems with low timeout pressure.'
      ),
      (
          '- If timeout rank bins are mostly tail or fallback candidates, reduce '
          'timeout fallback or require a stronger value-model prior before '
          'keeping timed-out branches alive.'
      ),
      (
          '- If a rerank phase dominates selected candidates but contributes no '
          'beam-add, SFT signal, or solved event, down-weight that phase or move '
          'its slots behind phases with proven DDAR progress.'
      ),
      (
          '- signal_anchor_coverage is expected to be a small but high-value '
          'bridge: it should preserve solved/SFT-positive construction families '
          'without drowning out generic progress_type_coverage.'
      ),
      (
          '- adaptive type penalty should reduce repeated invalid families while '
          'still allowing penalized types to appear in beam-add/SFT if DDAR gives '
          'real progress; a zero SFT/beam-add count for heavily penalized types '
          'means the penalty may be too aggressive.'
      ),
      (
          '- Candidate SFT data should continue to emphasize solved candidates '
          'and fast DDAR-progress positives; slow timeout branches should remain '
          'hard negatives unless fallback later solves a problem.'
      ),
      '',
      '## Per-Problem Table',
      '',
      '| Problem | Solved | Cand | Valid | Selected | TailSel | AdaptPen | RankPruned | DDAR | Timeout | Fallback | Max Added/Type | Diagnosis |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|',
  ])
  lines.extend(problem_line(problem) for problem in problems)
  lines.extend([
      '',
      '## Unsolved Focus',
      '',
  ])
  if unsolved:
    for problem in unsolved:
      lines.append(
          f"- `{problem.get('problem')}`: "
          f"{diagnosis_text(problem.get('diagnosis') or [])}; "
          f"selected={fmt_count_map(problem.get('depth_eval_selected_phases'), 5)}; "
          f"selected_ranks={fmt_count_map(problem.get('depth_eval_selected_rank_bins'), 5)}; "
          f"filtered_ranks={fmt_count_map(problem.get('filtered_rank_bins'), 5)}; "
          f"adapt_penalty={problem.get('adaptive_penalized_records', 0)} "
          f"{fmt_count_map(problem.get('adaptive_penalty_top_types_top'), 3)}; "
          f"timeout={fmt_count_map(problem.get('timeout_construction_types_top'), 5)}; "
          f"fallback={problem.get('candidate_timeout_beam_fallbacks', 0)}"
      )
  else:
    lines.append('- No unsolved completed problems in this analysis payload.')
  if solved:
    lines.extend(['', '## Solved Cases', ''])
    for problem in solved:
      lines.append(f"- `{problem.get('problem')}` {solved_detail(problem)}")
  return '\n'.join(lines) + '\n'


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--analysis_json', required=True)
  parser.add_argument('--out_file')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  report = render_report(load_json(Path(args.analysis_json)))
  if args.out_file:
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding='utf-8')
  print(report)


if __name__ == '__main__':
  main()
