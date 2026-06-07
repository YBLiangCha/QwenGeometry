#!/usr/bin/env python3
"""Collect verifier-accepted auxiliary-construction samples from Qwen+DDAR.

This is the lightweight Stage 3 alignment loop for the AG1/Qwen reproduction:

1. Load non-IMO AG problems.
2. Let DDAR try the root state.
3. When DDAR fails, ask the current Qwen adapter for auxiliary constructions.
4. Translate each candidate through the same AG1 LM boundary as benchmark search.
5. Re-run DDAR and keep only candidates that solve the goal, or optionally
   candidates that create a large symbolic-closure gain.

The output JSONL is compatible with train_qwen_aux_lora.py in target-loss mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import traceback
from typing import Any


def import_search(script_dir: str):
    sys.path.insert(0, str(Path(script_dir).resolve()))
    import qwen_ag_search as qs  # pylint: disable=import-error,import-outside-toplevel

    return qs


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "problem").strip("_")


def parse_problem_names(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def candidate_verdict(
    root: dict[str, Any],
    result: dict[str, Any],
    keep_progress: bool,
    min_cache_gain: int,
    min_added_dependencies: int,
) -> str | None:
    if result.get("solved"):
        return "solved"
    if not keep_progress:
        return None
    cache_gain = int(result.get("cache_items", 0)) - int(root.get("cache_items", 0))
    added = int(result.get("added_dependencies", 0))
    if cache_gain >= min_cache_gain and added >= min_added_dependencies:
        return "progress"
    return None


def build_graph_for_symbolic_search(gh: Any, p: Any, definitions: Any):
    """Build an AG graph while tolerating numeric goal-check bugs.

    AG1's Graph.build_problem runs a numerical sanity check for the goal after
    building the construction graph. Some predicates, notably cyclic in the
    released code, can raise in that numeric check even though symbolic DDAR can
    still reason about the goal. For verifier-guided data collection we only
    need the construction graph; goal checking is done symbolically by DDAR.
    """
    try:
        return gh.Graph.build_problem(p, definitions)
    except Exception:  # pylint: disable=broad-except
        p_no_goal = p.copy()
        p_no_goal.goal = None
        return gh.Graph.build_problem(p_no_goal, definitions)


def collect_for_problem(
    name: str,
    p: Any,
    qs: Any,
    pr: Any,
    gh: Any,
    ddar: Any,
    pt: Any,
    generator: Any,
    args: argparse.Namespace,
    events_file: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    g, _ = build_graph_for_symbolic_search(gh, p, qs.DEFINITIONS)
    root = qs.run_ddar_once(
        g,
        p,
        ddar,
        gh,
        args.root_max_level or args.max_level,
        args.root_ddar_timeout or args.ddar_timeout,
        events_file,
        "root",
    )
    stats = {
        "name": name,
        "root_solved": root["solved"],
        "valid_candidates": 0,
        "accepted": 0,
        "accepted_solved": 0,
        "accepted_progress": 0,
        "errors": 0,
    }
    if root["solved"]:
        return [], stats

    prompt0 = p.setup_str_from_problem(qs.DEFINITIONS) + " {F1} x00"
    beam = qs.BeamQueue(args.beam_size)
    beam.add((g, prompt0, p.txt(), []), 0.0)
    accepted: list[dict[str, Any]] = []
    seen_targets: set[str] = set()

    for depth in range(args.search_depth):
        qs.event(events_file, kind="stage3_depth_start", depth=depth, nodes=len(beam))
        next_beam = qs.BeamQueue(args.beam_size)
        for prev_score, (g_cur, prompt, pstring, prev_aux) in beam.ordered():
            try:
                candidates = generator.generate(
                    prompt,
                    args.num_return_sequences,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_p,
                )
            except Exception:  # pylint: disable=broad-except
                stats["errors"] += 1
                qs.event(
                    events_file,
                    kind="stage3_generate_error",
                    depth=depth,
                    traceback=traceback.format_exc(),
                )
                continue

            for raw, lm_score in candidates:
                translation = qs.try_translate_candidate(raw, g_cur, pr, pt)
                qs.event(
                    events_file,
                    kind="stage3_candidate",
                    depth=depth,
                    raw=raw,
                    translation=translation,
                    lm_score=lm_score,
                )
                if translation.startswith("ERROR:"):
                    continue
                stats["valid_candidates"] += 1
                try:
                    p_new_txt = qs.insert_aux_to_premise(pstring, translation)
                    p_new = pr.Problem.from_txt(p_new_txt, translate=False)
                    g_new, _ = build_graph_for_symbolic_search(
                        gh, p_new, qs.DEFINITIONS
                    )
                    result = qs.run_ddar_once(
                        g_new,
                        p_new,
                        ddar,
                        gh,
                        args.candidate_max_level or args.max_level,
                        args.candidate_ddar_timeout or args.ddar_timeout,
                        events_file,
                        f"stage3_depth{depth}:{raw}",
                    )
                except Exception:  # pylint: disable=broad-except
                    stats["errors"] += 1
                    qs.event(
                        events_file,
                        kind="stage3_verify_error",
                        depth=depth,
                        raw=raw,
                        translation=translation,
                        traceback=traceback.format_exc(),
                    )
                    continue

                verdict = candidate_verdict(
                    root,
                    result,
                    args.keep_progress,
                    args.min_cache_gain,
                    args.min_added_dependencies,
                )
                if verdict:
                    dedup_key = prompt + "\n" + raw
                    if dedup_key not in seen_targets:
                        seen_targets.add(dedup_key)
                        stats["accepted"] += 1
                        stats[f"accepted_{verdict}"] += 1
                        row = {
                            "id": f"{safe_name(name)}::depth{depth}::accepted{stats['accepted']:04d}",
                            "source_problem": name,
                            "prompt": prompt,
                            "target": raw,
                            "candidate_translation": translation,
                            "verdict": verdict,
                            "depth": depth,
                            "prev_aux": prev_aux,
                            "root_cache_items": root.get("cache_items"),
                            "candidate_cache_items": result.get("cache_items"),
                            "cache_gain": int(result.get("cache_items", 0))
                            - int(root.get("cache_items", 0)),
                            "added_dependencies": result.get("added_dependencies"),
                            "solved": result.get("solved"),
                            "problem_after_aux": p_new_txt,
                        }
                        accepted.append(row)
                        qs.event(
                            events_file,
                            kind="stage3_accepted",
                            depth=depth,
                            verdict=verdict,
                            raw=raw,
                            translation=translation,
                        )
                if result["solved"] and args.stop_after_first_solution:
                    return accepted, stats
                next_beam.add(
                    (
                        g_new,
                        prompt + " " + raw + " x00",
                        p_new_txt,
                        prev_aux + [raw],
                    ),
                    prev_score + lm_score,
                )
        beam = next_beam
        if len(beam) == 0:
            break

    return accepted, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--script_dir", required=True)
    p.add_argument("--ag_repo", required=True)
    p.add_argument("--problems_file", required=True)
    p.add_argument("--defs_file", required=True)
    p.add_argument("--rules_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--summary_file", required=True)
    p.add_argument("--events_dir", required=True)
    p.add_argument("--problem_names")
    p.add_argument("--limit", type=int)
    p.add_argument("--max_rows", type=int, default=512)
    p.add_argument("--translate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_level", type=int, default=300)
    p.add_argument("--ddar_timeout", type=int, default=60)
    p.add_argument("--root_max_level", type=int)
    p.add_argument("--root_ddar_timeout", type=int)
    p.add_argument("--candidate_max_level", type=int)
    p.add_argument("--candidate_ddar_timeout", type=int)
    p.add_argument("--qwen_model", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device_map", default="cuda:0")
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--search_depth", type=int, default=2)
    p.add_argument("--num_return_sequences", type=int, default=6)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--keep_progress", action="store_true")
    p.add_argument("--min_cache_gain", type=int, default=40)
    p.add_argument("--min_added_dependencies", type=int, default=1)
    p.add_argument("--stop_after_first_solution", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    qs = import_search(args.script_dir)
    qs.add_ag_repo_to_path(args.ag_repo)
    ag = qs.load_ag_modules()
    pr, gh, ddar, pt = ag["pr"], ag["gh"], ag["ddar"], ag["pt"]
    qs.DEFINITIONS = pr.Definition.from_txt_file(args.defs_file, to_dict=True)
    qs.RULES = pr.Theorem.from_txt_file(args.rules_file, to_dict=True)

    problems = pr.Problem.from_txt_file(
        args.problems_file, to_dict=True, translate=args.translate
    )
    selected = parse_problem_names(args.problem_names)
    names = [n for n in problems if selected is None or n in selected]
    if args.limit:
        names = names[: args.limit]

    out_file = Path(args.out_file)
    summary_file = Path(args.summary_file)
    events_dir = Path(args.events_dir)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    generator = qs.QwenGenerator(
        args.qwen_model, args.adapter_path, args.dtype, args.device_map
    )

    summary = {
        "problems_file": args.problems_file,
        "num_problems": len(names),
        "rows": 0,
        "accepted_solved": 0,
        "accepted_progress": 0,
        "problem_stats": [],
    }
    with out_file.open("w", encoding="utf-8") as out:
        for i, name in enumerate(names, 1):
            events_file = str(events_dir / f"{i:04d}_{safe_name(name)}.jsonl")
            try:
                rows, stats = collect_for_problem(
                    name,
                    problems[name],
                    qs,
                    pr,
                    gh,
                    ddar,
                    pt,
                    generator,
                    args,
                    events_file,
                )
            except Exception:  # pylint: disable=broad-except
                rows = []
                stats = {
                    "name": name,
                    "fatal_error": traceback.format_exc(),
                    "accepted": 0,
                    "accepted_solved": 0,
                    "accepted_progress": 0,
                }
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary["rows"] += 1
                summary["accepted_solved"] += int(row["verdict"] == "solved")
                summary["accepted_progress"] += int(row["verdict"] == "progress")
                if args.max_rows and summary["rows"] >= args.max_rows:
                    break
            out.flush()
            summary["problem_stats"].append(stats)
            print(
                json.dumps(
                    {
                        "index": i,
                        "name": name,
                        "accepted": stats.get("accepted", 0),
                        "rows_total": summary["rows"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            summary_file.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if args.max_rows and summary["rows"] >= args.max_rows:
                break

    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
