#!/usr/bin/env python3
"""Render a compact Markdown report from Qwen+AG event analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DIAGNOSIS_TEXT = {
    'solved': '已解决',
    'no_candidates_generated': '没有生成候选',
    'high_invalid_rate': '无效候选比例高',
    'duplicate_collapse': 'canonical 重复坍缩',
    'candidate_ddar_timeout_blocked': '候选 DDAR 超时阻塞',
    'candidate_ddar_timeouts': '存在候选 DDAR 超时',
    'symbolic_progress_but_wrong_direction': 'DDAR 有事实增长但方向不对',
    'valid_candidates_not_evaluated': '有合法候选但未完成评估',
    'template_backfill_exhausted': '模板回填仍补不满',
    'search_exhausted_no_goal': '搜索耗尽但未达目标',
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


def diagnosis_text(labels: list[str]) -> str:
  return ', '.join(DIAGNOSIS_TEXT.get(label, label) for label in labels) or '-'


def problem_line(problem: dict[str, Any]) -> str:
  filtered = problem.get('filtered_reasons') or {}
  candidate_errors = problem.get('candidate_ddar_error_types') or {}
  max_added = problem.get('max_added_candidate') or {}
  return (
      f"| {problem.get('problem')} "
      f"| {'Y' if problem.get('solved') else 'N'} "
      f"| {problem.get('candidates', 0)} "
      f"| {problem.get('invalid_candidates', 0)} "
      f"| {filtered.get('duplicate_canonical', 0)} "
      f"| {problem.get('candidate_ddar_done', 0)} "
      f"| {fmt_count_map(candidate_errors, 3)} "
      f"| {max_added.get('added_dependencies', 0) or 0}"
      f"/{max_added.get('construction_type') or '-'} "
      f"| {diagnosis_text(problem.get('diagnosis') or [])} |"
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
          f"- NEW SOLVED: `{problem.get('problem')}` "
          f"depth={problem.get('solved_depth')} "
          f"aux=`{problem.get('aux')}` "
          f"type={problem.get('solved_aux_construction_type') or '-'}"
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
  ])
  lines.extend([
      '',
      '## Aggregate Diagnosis',
      '',
      fmt_count_map(payload.get('aggregate_diagnosis'), 20),
      '',
      '## Construction-Type Signals',
      '',
      f"- Generated valid top: {fmt_count_map(aggregate.get('construction_types_top'))}",
      f"- Evaluated top: {fmt_count_map(aggregate.get('evaluated_construction_types_top'))}",
      f"- Filtered top: {fmt_count_map(aggregate.get('filtered_construction_types_top'))}",
      f"- Timeout top: {fmt_count_map(aggregate.get('timeout_construction_types_top'))}",
      f"- Hard-negative top: "
      f"{fmt_count_map(aggregate.get('candidate_hard_negative_signal_types_top'))}",
      f"- Solved aux top: {fmt_count_map(aggregate.get('solved_aux_construction_types_top'))}",
      '',
      '## Optimization Readout',
      '',
      (
          '- Candidate quality remains the first lever: duplicate_canonical and '
          'invalid_candidates dominate the loss before DDAR even has a useful branch.'
      ),
      (
          '- Search diversity is partly working if evaluated construction types are '
          'more balanced than generated types; remaining depth_rank_pruned and timeout '
          'types show where value-model reranking should improve.'
      ),
      (
          '- Auxiliary SFT data should prioritize solved candidates and fast '
          'DDAR-progress positives; slow timeout candidates and PointTooClose/TooFar '
          'should remain hard negatives.'
      ),
      (
          '- Generator-side hard-negative signals are useful only when they keep '
          'the original LM prompt and raw target; value-model-only negatives do not '
          'teach the decoder to avoid invalid point constructions.'
      ),
      (
          '- Fact-context ablations should be judged by whether duplicate collapse, '
          'wrong-direction DDAR progress, and timeout-heavy evaluated types decrease.'
      ),
      '',
      '## Per-Problem Table',
      '',
      '| Problem | Solved | Cand | Invalid | Dup | DDAR | DDAR Errors | Max Added/Type | Diagnosis |',
      '|---|---:|---:|---:|---:|---:|---|---|---|',
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
          f"generated={fmt_count_map(problem.get('construction_types_top'), 5)}; "
          f"evaluated={fmt_count_map(problem.get('evaluated_construction_types_top'), 5)}; "
          f"timeout={fmt_count_map(problem.get('timeout_construction_types_top'), 5)}"
      )
  else:
    lines.append('- No unsolved completed problems in this analysis payload.')
  if solved:
    lines.extend(['', '## Solved Cases', ''])
    for problem in solved:
      lines.append(
          f"- `{problem.get('problem')}` depth={problem.get('solved_depth')} "
          f"aux=`{problem.get('aux')}` "
          f"type={problem.get('solved_aux_construction_type') or '-'}"
      )
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
