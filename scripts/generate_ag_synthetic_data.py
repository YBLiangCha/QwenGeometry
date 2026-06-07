#!/usr/bin/env python3
"""Generate AG1-compatible synthetic CPT/SFT data.

This follows the AlphaGeometry synthetic-data idea at a smaller local scale:

1. sample random constructive diagrams in the AG1 language;
2. run DDAR to a deduction closure;
3. sample proven facts from Graph.cache;
4. use trace_back.get_logs to extract proof traces and dependency differences;
5. emit CPT rows with full theorem/proof text and SFT rows for next aux.

The script intentionally stays inside AG1 syntax. It does not generate AG2-only
features such as locus statements, double points, or non-constructive points.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random
import sys
import time
from typing import Any


DEFINITIONS = None
RULES = None

GOAL_FACTS = {
    "coll",
    "cong",
    "cyclic",
    "eqangle",
    "eqratio",
    "midp",
    "para",
    "perp",
}

CONSTRAINED_SYMBOLS = {
    "coll": "C",
    "cong": "D",
    "cyclic": "O",
    "eqangle": "^",
    "para": "P",
    "perp": "T",
}


def add_paths(ag_repo: str) -> None:
    repo = Path(ag_repo).resolve()
    if not repo.exists():
        raise FileNotFoundError(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_ag_modules() -> dict[str, Any]:
    import ddar  # pylint: disable=import-error,import-outside-toplevel
    import graph as gh  # pylint: disable=import-error,import-outside-toplevel
    import make_aux_sft_from_ag_file as ma  # pylint: disable=import-error,import-outside-toplevel
    import problem as pr  # pylint: disable=import-error,import-outside-toplevel
    import qwen_ag_search as qs  # pylint: disable=import-error,import-outside-toplevel
    import trace_back as tb  # pylint: disable=import-error,import-outside-toplevel

    return {"ddar": ddar, "gh": gh, "ma": ma, "pr": pr, "qs": qs, "tb": tb}


def point_name(index: int) -> str:
    if index < 26:
        return chr(ord("a") + index)
    return chr(ord("a") + index % 26) + str(index // 26)


def clause_text(name: str, point: str, args: list[str]) -> str:
    if name == "parallelogram":
        return f"{point} = parallelogram {' '.join(args)} {point}"
    return f"{point} = {name} {point} {' '.join(args)}"


def sample_clause(rng: random.Random, point: str, points: list[str]) -> str:
    """Sample one conservative AG1 construction clause."""
    n = len(points)
    choices: list[tuple[str, int, float]] = [
        ("on_line", 2, 1.0),
        ("on_circle", 2, 0.9),
        ("midpoint", 2, 0.9),
        ("mirror", 2, 0.45),
        ("on_tline", 3, 0.9),
        ("on_pline", 3, 0.75),
        ("foot", 3, 0.8),
        ("circumcenter", 3, 0.45),
        ("orthocenter", 3, 0.35),
        ("angle_bisector", 3, 0.3),
        ("intersection_ll", 4, 0.8),
        ("parallelogram", 3, 0.55),
        ("eq_triangle", 2, 0.35),
    ]
    available = [(name, arity, w) for name, arity, w in choices if n >= arity]
    names = [x[0] for x in available]
    weights = [x[2] for x in available]
    name = rng.choices(names, weights=weights, k=1)[0]
    arity = dict((x[0], x[1]) for x in available)[name]
    args = rng.sample(points, arity)
    return clause_text(name, point, args)


def problem_from_clauses(pr: Any, url: str, clauses: list[Any], goal: Any | None):
    txt = "; ".join(c.txt() for c in clauses)
    if goal is not None:
        txt += " ? " + goal.txt()
    if url:
        txt = url + "\n" + txt
    return pr.Problem.from_txt(txt, translate=False)


def try_add_clause(g: Any, p: Any, clause: Any, pr: Any) -> tuple[Any, Any] | None:
    """Add a clause without Graph.build_problem's unbounded retry loop."""
    try:
        g2 = g.copy()
        adds, plevel = g2.add_clause(clause, getattr(g, "plevel", 0), DEFINITIONS)
        for add in adds:
            g2.add_algebra(add, level=0)
        clauses = list(p.clauses) + [clause]
        p2 = pr.Problem(p.url, clauses, p.goal)
        g2.plevel = plevel
        g2.url = p2.url
        g2.build_def = (p2, DEFINITIONS)
        return p2, g2
    except Exception:  # pylint: disable=broad-except
        return None


def build_random_diagram(
    rng: random.Random,
    pr: Any,
    gh: Any,
    diagram_id: int,
    min_points: int,
    max_points: int,
    max_clause_tries: int,
) -> tuple[Any, Any] | None:
    base = f"synthetic_{diagram_id:06d}\na b c = triangle a b c"
    p = pr.Problem.from_txt(base, translate=False)
    try:
        g, _ = gh.Graph.build_problem(p, DEFINITIONS, verbose=False)
    except Exception:  # pylint: disable=broad-except
        return None

    target_points = rng.randint(min_points, max_points)
    while len(p.clauses) < target_points - 2:
        x = point_name(len(g.all_points()))
        added = False
        for _ in range(max_clause_tries):
            text = sample_clause(rng, x, [pt.name for pt in g.all_points()])
            try:
                clause = pr.Clause.from_txt(text)
            except Exception:  # pylint: disable=broad-except
                continue
            result = try_add_clause(g, p, clause, pr)
            if result is None:
                continue
            p, g = result
            added = True
            break
        if not added:
            break

    if len(g.all_points()) < min_points:
        return None
    return p, g


def names(args: list[Any]) -> list[str]:
    return [getattr(a, "name", str(a)) for a in args]


def dep_text(dep: Any) -> str:
    return " ".join([dep.name] + names(dep.args))


def dep_to_constrained_piece(dep: Any, point: str, idx: int) -> str | None:
    symbol = CONSTRAINED_SYMBOLS.get(dep.name)
    if symbol is None:
        return None
    arg_names = names(dep.args)
    if point not in arg_names:
        return None
    return f"{symbol} {' '.join(arg_names)} {idx:02d}"


def make_aux_target(point: str, deps: list[Any]) -> str | None:
    pieces = []
    for dep in deps:
        piece = dep_to_constrained_piece(dep, point, len(pieces))
        if piece is not None:
            pieces.append(piece)
        if len(pieces) == 2:
            break
    if not pieces:
        return None
    return f"{point} : " + " ".join(pieces) + " ;"


def goal_from_key(pr: Any, key: tuple[str, ...]) -> Any | None:
    name = key[0]
    args = list(key[1:])
    if name not in GOAL_FACTS:
        return None
    if not is_nondegenerate_goal(name, args):
        return None
    if any("pi/" in a for a in args):
        return None
    try:
        return pr.Construction(name, args)
    except Exception:  # pylint: disable=broad-except
        return None


def good_pair(a: str, b: str) -> bool:
    return a != b


def same_unordered_pair(a: str, b: str, c: str, d: str) -> bool:
    return {a, b} == {c, d}


def is_nondegenerate_goal(name: str, args: list[str]) -> bool:
    """Reject tautological or badly repeated facts as theorem targets."""
    if name == "coll":
        return len(args) == 3 and len(set(args)) == 3
    if name == "cyclic":
        return len(args) == 4 and len(set(args)) == 4
    if name == "midp":
        return len(args) == 3 and len(set(args)) == 3
    if name in {"para", "perp", "cong"}:
        if len(args) != 4:
            return False
        a, b, c, d = args
        if not (good_pair(a, b) and good_pair(c, d)):
            return False
        if same_unordered_pair(a, b, c, d):
            return False
        if name == "para":
            return len(set(args)) == 4
        return True
    if name in {"eqangle", "eqratio"}:
        if len(args) not in {6, 8}:
            return False
        pairs = list(zip(args[::2], args[1::2]))
        if any(a == b for a, b in pairs):
            return False
        return len(set(args)) >= 4
    return len(set(args)) >= 2


def candidate_keys(g: Any, rng: random.Random) -> list[tuple[str, ...]]:
    keys = []
    for key, dep in g.cache.items():
        name = key[0]
        if name not in GOAL_FACTS:
            continue
        if not is_nondegenerate_goal(name, list(key[1:])):
            continue
        # Prefer nontrivial DD/AR products over raw construction facts.
        if getattr(dep, "rule_name", "") == "c0":
            continue
        if len(set(key[1:])) < 2:
            continue
        keys.append(key)
    rng.shuffle(keys)
    return keys


def validate_aux_target(
    target: str,
    p_prefix: Any,
    goal: Any,
    pr: Any,
    gh: Any,
    qs: Any,
) -> tuple[bool, str]:
    qs.DEFINITIONS = DEFINITIONS
    try:
        g_prefix, _ = gh.Graph.build_problem(p_prefix, DEFINITIONS, verbose=False)
        translation = qs.try_translate_candidate(target, g_prefix, pr, None)
        if translation.startswith("ERROR:"):
            return False, translation
        p_candidate = problem_from_clauses(
            pr,
            p_prefix.url,
            p_prefix.clauses + [pr.Clause.from_txt(translation)],
            goal,
        )
        gh.Graph.build_problem(p_candidate, DEFINITIONS, verbose=False)
        return True, translation
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"ERROR: {type(exc).__name__}: {exc}"


def goal_solved(p: Any, pr: Any, gh: Any, ddar: Any, max_level: int, timeout: int) -> bool:
    try:
        g, _ = gh.Graph.build_problem(p, DEFINITIONS, verbose=False)
        ddar.solve(g, RULES, p, max_level=max_level, timeout=timeout)
        if p.goal is None:
            return False
        return g.check(p.goal.name, g.names2nodes(p.goal.args))
    except Exception:  # pylint: disable=broad-except
        return False


def solve_to_closure(
    p: Any,
    gh: Any,
    ddar: Any,
    max_level: int,
    timeout: int,
) -> Any | None:
    try:
        g, _ = gh.Graph.build_problem(p, DEFINITIONS, verbose=False)
        ddar.solve(g, RULES, p, max_level=max_level, timeout=timeout)
        return g
    except Exception:  # pylint: disable=broad-except
        return None


def make_withheld_clause_sft_rows(
    p: Any,
    pr: Any,
    gh: Any,
    ddar: Any,
    ma: Any,
    max_level: int,
    timeout: int,
    max_rows: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Create verifier-backed state -> aux samples by withholding one clause.

    A row is kept only if DDAR cannot prove the selected goal from the prefix,
    but can prove it after adding the withheld construction.
    """
    rows = []
    for idx in range(1, len(p.clauses)):
        clause = p.clauses[idx]
        if len(clause.points) != 1:
            continue
        aux_point = clause.points[0]
        target = ma.clause_to_constrained_target(clause)
        if target is None:
            continue

        prefix_clauses = p.clauses[:idx]
        known = {point for c in prefix_clauses for point in c.points}
        p_prefix_no_goal = problem_from_clauses(pr, p.url, prefix_clauses, None)
        p_candidate_no_goal = problem_from_clauses(
            pr, p.url, prefix_clauses + [clause], None
        )
        g_prefix = solve_to_closure(
            p_prefix_no_goal, gh, ddar, max_level=max_level, timeout=timeout
        )
        g_candidate = solve_to_closure(
            p_candidate_no_goal, gh, ddar, max_level=max_level, timeout=timeout
        )
        if g_prefix is None or g_candidate is None:
            continue

        prefix_keys = set(g_prefix.cache.keys())
        keys = candidate_keys(g_candidate, rng)
        for key in keys:
            if key in prefix_keys:
                continue
            if aux_point in key[1:]:
                continue
            if any((not arg.isdigit()) and arg not in known for arg in key[1:]):
                continue
            goal = goal_from_key(pr, key)
            if goal is None:
                continue
            p_prefix = problem_from_clauses(pr, p.url, prefix_clauses, goal)
            p_candidate = problem_from_clauses(
                pr, p.url, prefix_clauses + [clause], goal
            )
            prefix_solved = goal_solved(
                p_prefix, pr, gh, ddar, max_level=max_level, timeout=timeout
            )
            if prefix_solved:
                continue
            candidate_solved = goal_solved(
                p_candidate, pr, gh, ddar, max_level=max_level, timeout=timeout
            )
            if not candidate_solved:
                continue
            rows.append({
                "id": f"{p.url}::withheld_{idx}::{key}",
                "source": "synthetic_withheld_clause",
                "problem": p_candidate.txt(),
                "prompt": p_prefix.setup_str_from_problem(DEFINITIONS) + " {F1} x00",
                "target": target,
                "candidate_constructive": clause.txt(),
                "goal": goal.txt(),
                "prefix_solved_by_ddar": False,
                "candidate_solved_by_ddar": True,
            })
            break
        if len(rows) >= max_rows:
            break
    return rows


def point_dependencies(clauses: list[Any]) -> dict[str, set[str]]:
    deps: dict[str, set[str]] = {}
    for clause in clauses:
        new_points = set(clause.points)
        used = set()
        for cons in clause.constructions:
            for arg in cons.args:
                if not arg.isdigit():
                    used.add(arg)
        for point in new_points:
            deps[point] = set(used) - new_points
    return deps


def depends_on(point: str, target: str, deps: dict[str, set[str]]) -> bool:
    stack = list(deps.get(point, set()))
    seen = set()
    while stack:
        cur = stack.pop()
        if cur == target:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(deps.get(cur, set()))
    return False


def prefix_without_aux_point(pr: Any, p_goal: Any, aux_point: str) -> Any | None:
    deps = point_dependencies(p_goal.clauses)
    keep = []
    for clause in p_goal.clauses:
        clause_points = set(clause.points)
        if aux_point in clause_points:
            continue
        if any(depends_on(point, aux_point, deps) for point in clause_points):
            continue
        keep.append(clause)
    known = {point for clause in keep for point in clause.points}
    if p_goal.goal is not None:
        for arg in p_goal.goal.args:
            if not arg.isdigit() and arg not in known:
                return None
    return problem_from_clauses(pr, p_goal.url, keep, p_goal.goal)


def make_sft_rows(
    p_goal: Any,
    aux_setup: list[Any],
    setup_points: set[Any],
    pr: Any,
    gh: Any,
    ddar: Any,
    qs: Any,
    max_level: int,
    timeout: int,
) -> list[dict[str, Any]]:
    setup_names = {p.name for p in setup_points}
    goal_names = set(p_goal.goal.args if p_goal.goal else [])
    grouped: dict[str, list[Any]] = {}
    for dep in aux_setup:
        if dep.name not in CONSTRAINED_SYMBOLS:
            continue
        for arg in names(dep.args):
            if arg not in setup_names and arg not in goal_names:
                grouped.setdefault(arg, []).append(dep)

    rows = []
    for point, deps in grouped.items():
        # Try short one- or two-predicate definitions that AG1 can translate.
        for size in (1, 2):
            for combo in itertools.combinations(deps, min(size, len(deps))):
                target = make_aux_target(point, list(combo))
                if target is None:
                    continue
                p_prefix = prefix_without_aux_point(pr, p_goal, point)
                if p_prefix is None:
                    continue
                ok, translation = validate_aux_target(
                    target, p_prefix, p_goal.goal, pr, gh, qs
                )
                if not ok:
                    continue
                prefix_solved = goal_solved(
                    p_prefix, pr, gh, ddar, max_level=max_level, timeout=timeout
                )
                p_candidate = problem_from_clauses(
                    pr,
                    p_prefix.url,
                    p_prefix.clauses + [pr.Clause.from_txt(translation)],
                    p_goal.goal,
                )
                candidate_solved = goal_solved(
                    p_candidate, pr, gh, ddar, max_level=max_level, timeout=timeout
                )
                if prefix_solved or not candidate_solved:
                    continue
                rows.append({
                    "id": f"{p_goal.url}::{point}::{target}",
                    "source": "synthetic_traceback_aux",
                    "problem": p_goal.txt(),
                    "prompt": p_prefix.setup_str_from_problem(DEFINITIONS) + " {F1} x00",
                    "target": target,
                    "candidate_constructive": translation,
                    "prefix_solved_by_ddar": prefix_solved,
                    "candidate_solved_by_ddar": candidate_solved,
                })
                return rows
    return rows


def proof_text(
    p_goal: Any,
    setup: list[Any],
    aux_setup: list[Any],
    log: list[tuple[list[Any], list[Any]]],
) -> str:
    setup_part = "; ".join(sorted(dep_text(d) for d in setup)) or "none"
    aux_part = "; ".join(sorted(dep_text(d) for d in aux_setup)) or "none"
    proof_steps = []
    for prems, cons in log:
        lhs = ", ".join(sorted(dep_text(d) for d in prems)) or "given"
        rhs = ", ".join(sorted(dep_text(d) for d in cons))
        proof_steps.append(lhs + " -> " + rhs)
    proof_part = "; ".join(proof_steps) or "direct"
    return (
        "{S} "
        + setup_part
        + " ? "
        + p_goal.goal.txt()
        + "\n{AUX} "
        + aux_part
        + "\n{PROOF} "
        + proof_part
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(out_dir: Path, cpt_rows: list[dict[str, Any]], sft_rows: list[dict[str, Any]]) -> None:
    write_jsonl(out_dir / "synthetic_cpt.jsonl", cpt_rows)
    write_jsonl(out_dir / "synthetic_aux_sft.jsonl", sft_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ag_repo", required=True)
    parser.add_argument("--defs_file", required=True)
    parser.add_argument("--rules_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--num_diagrams", type=int, default=200)
    parser.add_argument("--min_points", type=int, default=6)
    parser.add_argument("--max_points", type=int, default=11)
    parser.add_argument("--sample_facts_per_diagram", type=int, default=8)
    parser.add_argument("--max_clause_tries", type=int, default=80)
    parser.add_argument("--max_level", type=int, default=50)
    parser.add_argument("--ddar_timeout", type=int, default=30)
    parser.add_argument("--max_cpt_rows", type=int, default=2000)
    parser.add_argument("--max_sft_rows", type=int, default=1000)
    parser.add_argument("--withheld_sft_per_diagram", type=int, default=3)
    parser.add_argument("--flush_every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    global DEFINITIONS, RULES
    args = parse_args()
    add_paths(args.ag_repo)
    ag = load_ag_modules()
    pr = ag["pr"]
    gh = ag["gh"]
    ma = ag["ma"]
    ddar = ag["ddar"]
    qs = ag["qs"]
    tb = ag["tb"]
    DEFINITIONS = pr.Definition.from_txt_file(args.defs_file, to_dict=True)
    RULES = pr.Theorem.from_txt_file(args.rules_file, to_dict=True)

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cpt_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    seen_cpt = set()
    seen_sft = set()
    started = time.time()

    for diagram_id in range(args.num_diagrams):
        built = build_random_diagram(
            rng,
            pr,
            gh,
            diagram_id,
            args.min_points,
            args.max_points,
            args.max_clause_tries,
        )
        if built is None:
            print(json.dumps({"diagram": diagram_id, "status": "build_failed"}), flush=True)
            continue
        p, g = built
        before_cache = len(g.cache)
        try:
            g, level_times, status, branches, added = ddar.solve(
                g, RULES, p, max_level=args.max_level, timeout=args.ddar_timeout
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(json.dumps({
                "diagram": diagram_id,
                "status": "ddar_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "points": len(g.all_points()),
                "clauses": len(p.clauses),
                "cache_before": before_cache,
                "cpt_total": len(cpt_rows),
                "sft_total": len(sft_rows),
                "elapsed_sec": round(time.time() - started, 1),
            }), flush=True)
            if args.flush_every and diagram_id % args.flush_every == 0:
                write_outputs(out_dir, cpt_rows, sft_rows)
            continue
        fact_keys = candidate_keys(g, rng)[: args.sample_facts_per_diagram]
        made_cpt = 0
        made_sft = 0

        if len(sft_rows) < args.max_sft_rows and args.withheld_sft_per_diagram > 0:
            rows = make_withheld_clause_sft_rows(
                p,
                pr,
                gh,
                ddar,
                ma,
                max_level=args.max_level,
                timeout=args.ddar_timeout,
                max_rows=args.withheld_sft_per_diagram,
                rng=rng,
            )
            for row in rows:
                key_sft = (row["prompt"], row["target"])
                if key_sft in seen_sft:
                    continue
                sft_rows.append(row)
                seen_sft.add(key_sft)
                made_sft += 1
                if len(sft_rows) >= args.max_sft_rows:
                    break

        for key in fact_keys:
            goal = goal_from_key(pr, key)
            if goal is None:
                continue
            p_goal = problem_from_clauses(pr, p.url, p.clauses, goal)
            try:
                query = pr.Dependency(
                    key[0], g.names2nodes(list(key[1:])), None, None
                )
                setup, aux_setup, log, setup_points = tb.get_logs(
                    query, g, merge_trivials=False
                )
            except Exception:  # pylint: disable=broad-except
                continue
            if not log:
                continue

            cpt_key = (p_goal.txt(), tuple(dep_text(d) for _, cons in log for d in cons))
            if cpt_key not in seen_cpt and len(cpt_rows) < args.max_cpt_rows:
                cpt_rows.append({
                    "id": f"{p.url}::{key}",
                    "source": "synthetic_traceback",
                    "problem": p_goal.txt(),
                    "goal": p_goal.goal.txt() if p_goal.goal else None,
                    "num_points": len(g.all_points()),
                    "num_clauses": len(p.clauses),
                    "proof_steps": len(log),
                    "has_aux": bool(aux_setup),
                    "setup_deps": [dep_text(d) for d in setup],
                    "aux_deps": [dep_text(d) for d in aux_setup],
                    "text": proof_text(p_goal, setup, aux_setup, log),
                })
                seen_cpt.add(cpt_key)
                made_cpt += 1

            if aux_setup and len(sft_rows) < args.max_sft_rows:
                rows = make_sft_rows(
                    p_goal,
                    aux_setup,
                    setup_points,
                    pr,
                    gh,
                    ddar,
                    qs,
                    max_level=args.max_level,
                    timeout=args.ddar_timeout,
                )
                for row in rows:
                    key_sft = (row["prompt"], row["target"])
                    if key_sft in seen_sft:
                        continue
                    sft_rows.append(row)
                    seen_sft.add(key_sft)
                    made_sft += 1
                    if len(sft_rows) >= args.max_sft_rows:
                        break

        print(json.dumps({
            "diagram": diagram_id,
            "status": status,
            "points": len(g.all_points()),
            "clauses": len(p.clauses),
            "cache_before": before_cache,
            "cache_after": len(g.cache),
            "facts_sampled": len(fact_keys),
            "cpt_new": made_cpt,
            "sft_new": made_sft,
            "cpt_total": len(cpt_rows),
            "sft_total": len(sft_rows),
            "elapsed_sec": round(time.time() - started, 1),
            "levels": len(level_times),
            "branches": sum(branches) if branches else 0,
            "added": len(added),
        }), flush=True)

        if args.flush_every and (diagram_id + 1) % args.flush_every == 0:
            write_outputs(out_dir, cpt_rows, sft_rows)

        if len(cpt_rows) >= args.max_cpt_rows and len(sft_rows) >= args.max_sft_rows:
            break

    write_outputs(out_dir, cpt_rows, sft_rows)
    summary = {
        "cpt_rows": len(cpt_rows),
        "sft_rows": len(sft_rows),
        "seed": args.seed,
        "num_diagrams_requested": args.num_diagrams,
        "elapsed_sec": round(time.time() - started, 3),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"kind": "finished", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
