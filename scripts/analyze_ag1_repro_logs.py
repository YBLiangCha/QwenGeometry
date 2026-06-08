"""Analyze AlphaGeometry/AG1 reproduction logs.

The AG1 baseline can be expensive to reproduce fully because it needs the
released 150M model and a GPU/JAX setup.  This analyzer works on partial or
complete reproduction snapshots and extracts the evidence that matters for
QwenGeometry tuning: DDAR-only solves, LM-assisted solve depth/rank, candidate
translations, and the amount of search width needed before DDAR succeeds.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  if not path.exists():
    return rows
  for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
    if not line.strip():
      continue
    rows.append(json.loads(line))
  return rows


def load_summaries(root: Path) -> list[dict[str, Any]]:
  out = []
  for path in root.rglob('summary.json'):
    try:
      data = json.loads(path.read_text(encoding='utf-8', errors='replace'))
    except Exception:  # pylint: disable=broad-except
      continue
    out.append({'path': str(path), 'data': data})
  return out


def load_events(root: Path) -> list[tuple[Path, list[dict[str, Any]]]]:
  out = []
  for path in root.rglob('events.jsonl'):
    rows = read_jsonl(path)
    if rows:
      out.append((path, rows))
  return out


def event_counts(events: list[dict[str, Any]]) -> Counter[str]:
  return Counter(str(row.get('event') or row.get('kind') or 'unknown') for row in events)


def problem_order_from_events(events: list[dict[str, Any]]) -> list[str]:
  seen = []
  found = set()
  for row in events:
    name = row.get('problem')
    if isinstance(name, str) and name and name not in found:
      found.add(name)
      seen.append(name)
  return seen


def summarize_event_file(path: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
  counts = event_counts(events)
  ddl_solved = []
  ag_solved = []
  lm_decode_by_problem: dict[str, int] = defaultdict(int)
  candidate_checks_by_problem: dict[str, int] = defaultdict(int)
  max_depth_by_problem: dict[str, int] = defaultdict(lambda: -1)
  root_ddar_results = {}

  for row in events:
    problem = row.get('problem')
    event = row.get('event') or row.get('kind')
    if isinstance(row.get('depth'), int) and isinstance(problem, str):
      max_depth_by_problem[problem] = max(max_depth_by_problem[problem], row['depth'])
    if event == 'ddar_result' and isinstance(problem, str):
      root_ddar_results[problem] = row
      if row.get('solved'):
        ddl_solved.append(problem)
    elif event == 'lm_decode' and isinstance(problem, str):
      lm_decode_by_problem[problem] += 1
    elif event == 'ag_solved':
      ag_solved.append({
          'problem': problem,
          'depth': row.get('depth'),
          'node_index': row.get('node_index'),
          'rank': row.get('rank'),
          'score': row.get('score'),
          'translation': row.get('translation'),
          'candidates_checked': row.get('candidates_checked'),
          'lm_calls': row.get('lm_calls'),
          'elapsed_sec': row.get('elapsed_sec'),
          'candidate_proof_file': row.get('candidate_proof_file'),
      })
    if event in {'ag_candidate_done', 'ag_candidate_error'} and isinstance(problem, str):
      candidate_checks_by_problem[problem] += 1

  return {
      'path': str(path),
      'event_count': len(events),
      'event_counts': dict(counts),
      'problems_seen': problem_order_from_events(events),
      'root_ddar_solved': sorted(set(ddl_solved)),
      'ag_solved': ag_solved,
      'lm_decode_by_problem': dict(lm_decode_by_problem),
      'candidate_checks_by_problem': dict(candidate_checks_by_problem),
      'max_depth_by_problem': {
          k: v for k, v in max_depth_by_problem.items() if v >= 0
      },
      'root_ddar_results': root_ddar_results,
  }


def summarize(root: Path) -> dict[str, Any]:
  summaries = load_summaries(root)
  event_files = [
      summarize_event_file(path, events)
      for path, events in load_events(root)
  ]
  summary_rows = []
  for item in summaries:
    data = item['data']
    for row in data.get('rows') or []:
      summary_rows.append({**row, '_summary_path': item['path']})
  return {
      'root': str(root),
      'summary_files': summaries,
      'summary_rows': summary_rows,
      'event_files': event_files,
  }


def render_markdown(report: dict[str, Any]) -> str:
  lines = []
  lines.append('# AG1 Reproduction Log Analysis')
  lines.append('')
  lines.append(f"Root: `{report['root']}`")
  lines.append('')

  rows = report['summary_rows']
  if rows:
    solved = [row for row in rows if row.get('solved')]
    lines.append('## Summary Rows')
    lines.append('')
    lines.append(
        f"- Summary rows: {len(rows)}; solved: {len(solved)}; "
        f"unsolved/unfinished: {len(rows) - len(solved)}"
    )
    lines.append('')
    lines.append('| Problem | Solved | Method | Depth/Rank | Candidates | LM Calls | Translation |')
    lines.append('|---|---:|---|---:|---:|---:|---|')
    for row in rows:
      depth_rank = ''
      if row.get('depth') is not None or row.get('rank') is not None:
        depth_rank = f"{row.get('depth', '')}/{row.get('rank', '')}"
      lines.append(
          '| {problem} | {solved} | {method} | {depth_rank} | {candidates} | '
          '{lm_calls} | {translation} |'.format(
              problem=row.get('problem', ''),
              solved='yes' if row.get('solved') else 'no',
              method=row.get('method', ''),
              depth_rank=depth_rank,
              candidates=row.get('candidates_checked', ''),
              lm_calls=row.get('lm_calls', ''),
              translation=str(row.get('translation') or row.get('aux') or ''),
          )
      )
    lines.append('')

  for event_file in report['event_files']:
    lines.append('## Event File')
    lines.append('')
    lines.append(f"- Path: `{event_file['path']}`")
    lines.append(f"- Events: {event_file['event_count']}")
    lines.append(f"- Counts: `{event_file['event_counts']}`")
    if event_file['root_ddar_solved']:
      lines.append(
          '- Root DDAR solved: ' + ', '.join(event_file['root_ddar_solved'])
      )
    if event_file['ag_solved']:
      lines.append('')
      lines.append('| Problem | Depth | Node | Rank | Candidates Checked | Score | Translation |')
      lines.append('|---|---:|---:|---:|---:|---:|---|')
      for row in event_file['ag_solved']:
        lines.append(
            '| {problem} | {depth} | {node} | {rank} | {checked} | {score} | {translation} |'.format(
                problem=row.get('problem', ''),
                depth=row.get('depth', ''),
                node=row.get('node_index', ''),
                rank=row.get('rank', ''),
                checked=row.get('candidates_checked', ''),
                score=(
                    f"{row.get('score'):.4f}"
                    if isinstance(row.get('score'), (int, float))
                    else ''
                ),
                translation=row.get('translation') or '',
            )
        )
    lines.append('')

  lines.append('## Immediate Tuning Implications')
  lines.append('')
  lines.append(
      '- AG1 can solve cases where the winning auxiliary construction is not '
      'near the top of the LM beam. Preserve depth-level coverage for low-rank '
      'but valid candidates before aggressive pruning.'
  )
  lines.append(
      '- Root DDAR already solves a substantial subset; LM effort should be '
      'focused on DDAR-unsolved problems and should log candidate rank/depth '
      'so budget misses are visible.'
  )
  lines.append(
      '- Candidate proof checks, not LM score alone, are the decisive signal. '
      'Use DDAR-progress and solved-candidate traces to bias future SFT/rerank.'
  )
  return '\n'.join(lines) + '\n'


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', required=True, help='AG1 reproduction snapshot root')
  parser.add_argument('--json_out')
  parser.add_argument('--markdown_out')
  parser.add_argument('--quiet', action='store_true')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  report = summarize(Path(args.root))
  if not args.quiet:
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
  if args.json_out:
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )
  if args.markdown_out:
    markdown_out = Path(args.markdown_out)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(
        render_markdown(report),
        encoding='utf-8',
    )


if __name__ == '__main__':
  main()
