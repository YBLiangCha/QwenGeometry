# Copyright 2026
#
# Utility script for reproducing the IMO-AG-30 benchmark with a single LM load
# and parallel DDAR checks. This keeps the original AlphaGeometry code path but
# avoids restarting the 150M language model once per problem.

from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import shutil
import sys
import time
import traceback
import types
from concurrent import futures
from typing import Any

from absl import flags
from absl import logging as absl_logging

import alphageometry as ag
import graph as gh
import pretty as pt
import problem as pr


class StableBeamQueue:
  """Top-k queue that never compares graph/node payloads on score ties."""

  def __init__(self, max_size: int = 512):
    self.queue: list[tuple[float, object]] = []
    self.max_size = max_size

  def add(self, node: object, val: float) -> None:
    if len(self.queue) < self.max_size:
      self.queue.append((val, node))
      return
    min_idx, (min_val, _) = min(
        enumerate(self.queue), key=lambda item: item[1][0]
    )
    if val > min_val:
      self.queue[min_idx] = (val, node)

  def __iter__(self):
    yield from self.queue

  def __len__(self) -> int:
    return len(self.queue)


GIN_FILES = [
    "base_htrans.gin",
    "size/medium_150M.gin",
    "options/positions_t5.gin",
    "options/lr_cosine_decay.gin",
    "options/seq_1024_nocache.gin",
    "geometry_150M_generate.gin",
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--problems_file", default="imo_ag_30.txt")
  parser.add_argument("--defs_file", default="defs.txt")
  parser.add_argument("--rules_file", default="rules.txt")
  parser.add_argument("--ckpt_path", default="ag_ckpt_vocab")
  parser.add_argument("--vocab_path", default="ag_ckpt_vocab/geometry.757.model")
  parser.add_argument("--meliad_path", default="meliad_lib/meliad")
  parser.add_argument("--results_dir", default="benchmark_runs/imo_ag30")
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--beam_size", type=int, default=512)
  parser.add_argument("--search_depth", type=int, default=16)
  parser.add_argument("--sequence_length", type=int, default=128)
  parser.add_argument("--model_dtype", choices=["float32", "bfloat16"], default="float32")
  parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
  parser.add_argument("--problem", action="append", default=None)
  parser.add_argument("--skip_ddar_prefilter", action="store_true")
  parser.add_argument("--skip_initial_ddar", action="store_true")
  parser.add_argument("--problem_time_limit_sec", type=float, default=0.0)
  parser.add_argument("--keep_failed_candidate_logs", action="store_true")
  parser.add_argument(
      "--qwen_search_path",
      default="",
      help="Optional path containing qwen_ag_search.py for value reranking.",
  )
  parser.add_argument(
      "--candidate_rerank",
      choices=[
          "none",
          "heuristic_diverse",
          "value_model",
          "value_model_diverse",
          "value_model_frontfill_diverse",
          "value_model_frontfill_progress_diverse",
      ],
      default="none",
  )
  parser.add_argument("--candidate_value_model", default="")
  parser.add_argument("--candidate_secondary_value_model", default="")
  parser.add_argument("--candidate_frontfill_limit", type=int, default=8)
  parser.add_argument("--candidate_static_progress_type_bonus", default="")
  parser.add_argument(
      "--candidate_point_mask",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Filter candidates whose new point name already exists.",
  )
  parser.add_argument(
      "--candidate_point_repair",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Rename a generated existing new-point to the next free point.",
  )
  parser.add_argument(
      "--candidate_canonical_dedup",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Skip equivalent auxiliary clauses within a beam node.",
  )
  parser.add_argument(
      "--candidate_depth_canonical_dedup",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Skip equivalent auxiliary clauses already seen at the same depth.",
  )
  parser.add_argument(
      "--candidate_template_backfill",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Add type-diverse template candidates before reranking.",
  )
  parser.add_argument("--candidate_template_backfill_extra_slots", type=int, default=0)
  parser.add_argument(
      "--candidate_node_eval_limit",
      type=int,
      default=0,
      help="Maximum reranked candidates evaluated per beam node; 0 disables.",
  )
  parser.add_argument(
      "--candidate_node_type_eval_cap",
      type=int,
      default=0,
      help="Maximum reranked candidates per construction type per beam node; 0 disables.",
  )
  parser.add_argument(
      "--candidate_ddar_timeout_sec",
      type=float,
      default=0.0,
      help="Per-candidate DDAR wall timeout inside worker; 0 disables.",
  )
  parser.add_argument(
      "--candidate_adaptive_type_penalty",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Downrank construction types that repeatedly fail validation/DDAR.",
  )
  parser.add_argument("--candidate_adaptive_type_penalty_threshold", type=int, default=4)
  parser.add_argument("--candidate_adaptive_type_penalty_weight", type=float, default=0.55)
  parser.add_argument("--candidate_adaptive_type_penalty_max", type=float, default=3.0)
  parser.add_argument(
      "--candidate_adaptive_type_penalty_reasons",
      default=(
          "point_too_close,point_too_far,point_already_exists,unknown_point,"
          "invalid_quad_solve,dep_check_fail,invalid_line_intersect,"
          "value_error,invalid_predicate"
      ),
  )
  parser.add_argument(
      "--candidate_adaptive_type_penalty_ddar_errors",
      action=argparse.BooleanOptionalAction,
      default=False,
  )
  parser.add_argument(
      "--candidate_adaptive_type_penalty_ddar_error_reasons",
      default="timeout,point_too_close,point_too_far,invalid_quad_solve,dep_check_fail",
  )
  parser.add_argument("--lm_fact_context_top_k", type=int, default=0)
  parser.add_argument("--root_fact_max_level", type=int, default=1000)
  parser.add_argument("--root_fact_timeout", type=int, default=600)
  return parser.parse_args()


def install_jax_compat_shims(jax_module: Any) -> None:
  if not hasattr(jax_module.core, "Shape"):
    jax_module.core.Shape = tuple[int, ...]
  if not hasattr(jax_module, "xla"):
    jax_module.xla = types.SimpleNamespace()
  if not hasattr(jax_module.xla, "DeviceArray") and hasattr(jax_module, "Array"):
    jax_module.xla.DeviceArray = jax_module.Array


def ensure_absl_flags_parsed() -> None:
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(["run_imo_ag30_benchmark"])
  absl_logging.set_verbosity(absl_logging.ERROR)


def write_jsonl(path: Path, event: dict[str, Any]) -> None:
  event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **event}
  with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")


def raw_problem_texts(problems_file: str) -> dict[str, str]:
  lines = [line for line in Path(problems_file).read_text(encoding="utf-8").splitlines() if line]
  return {name: problem for name, problem in pr.reshape(lines, 2)}


def worker_init(defs_file: str, rules_file: str) -> None:
  ensure_absl_flags_parsed()
  ag.DEFINITIONS = pr.Definition.from_txt_file(defs_file, to_dict=True)
  ag.RULES = pr.Theorem.from_txt_file(rules_file, to_dict=True)
  absl_logging.set_verbosity(absl_logging.ERROR)


def run_ddar_worker(
    problem_name: str,
    pstring: str,
    out_file: str,
    log_file: str,
    keep_failed_log: bool,
    timeout_sec: float = 0.0,
) -> dict[str, Any]:
  start = time.time()
  log_path = Path(log_file)
  log_path.parent.mkdir(parents=True, exist_ok=True)
  Path(out_file).parent.mkdir(parents=True, exist_ok=True)
  try:
    p = pr.Problem.from_txt(pstring, translate=False)
    g, _ = gh.Graph.build_problem(p, ag.DEFINITIONS)
    with log_path.open("w", encoding="utf-8") as log:
      with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        old_handler = None
        if timeout_sec > 0 and hasattr(signal, "SIGALRM"):
          def _timeout_handler(signum, frame):  # pylint: disable=unused-argument
            raise TimeoutError(f"DDAR timed out after {timeout_sec:.1f}s")

          old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
          signal.setitimer(signal.ITIMER_REAL, timeout_sec)
        try:
          solved = ag.run_ddar(g, p, out_file)
        finally:
          if timeout_sec > 0 and hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
              signal.signal(signal.SIGALRM, old_handler)
    if not solved and not keep_failed_log:
      with contextlib.suppress(FileNotFoundError):
        log_path.unlink()
    return {
        "problem": problem_name,
        "solved": bool(solved),
        "elapsed_sec": time.time() - start,
        "out_file": out_file if solved else "",
        "log_file": str(log_path) if solved or keep_failed_log else "",
        "error": "",
    }
  except Exception as exc:  # pylint: disable=broad-except
    with log_path.open("a", encoding="utf-8") as log:
      log.write("\nERROR\n")
      log.write("".join(traceback.format_exception(exc)))
    return {
        "problem": problem_name,
        "solved": False,
        "elapsed_sec": time.time() - start,
        "out_file": "",
        "log_file": str(log_path),
        "error": repr(exc),
    }


def submit_ddar(
    pool: futures.ProcessPoolExecutor,
    problem_name: str,
    pstring: str,
    out_file: Path,
    log_file: Path,
    keep_failed_log: bool,
    timeout_sec: float = 0.0,
) -> futures.Future[dict[str, Any]]:
  return pool.submit(
      run_ddar_worker,
      problem_name,
      pstring,
      str(out_file),
      str(log_file),
      keep_failed_log,
      timeout_sec,
  )


def load_lm(args: argparse.Namespace):
  import jax
  install_jax_compat_shims(jax)
  import lm_inference

  gin_paths = [
      str(Path(args.meliad_path).resolve() / "transformer/configs"),
      str(Path.cwd()),
  ]
  gin_params = [
      f'DTYPE="{args.model_dtype}"',
      "DecoderOnlyLanguageModelGenerate.output_token_losses=True",
      f"TransformerTaskConfig.batch_size={args.batch_size}",
      f"TransformerTaskConfig.sequence_length={args.sequence_length}",
      "Trainer.restore_state_variables=False",
  ]
  lm_inference.parse_gin_configuration(
      GIN_FILES,
      gin_params,
      gin_paths=gin_paths,
  )
  model = lm_inference.LanguageModelInference(
      args.vocab_path, args.ckpt_path, mode="beam_search"
  )
  return model, {
      "jax_backend": jax.default_backend(),
      "jax_devices": [str(d) for d in jax.devices()],
      "lm_batch_size": model.batch_size,
  }


def load_qwen_search(args: argparse.Namespace):
  if args.candidate_rerank == "none" and args.lm_fact_context_top_k <= 0:
    return None
  if args.qwen_search_path:
    sys.path.insert(0, str(Path(args.qwen_search_path).resolve()))
  import qwen_ag_search as qs  # pylint: disable=import-error,import-outside-toplevel

  return qs


def load_static_type_bonus(path: str) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
  if not path:
    return {}, {}
  data = json.loads(Path(path).read_text(encoding="utf-8"))
  if not isinstance(data, dict):
    return {}, {}
  type_bonus = data.get("type_bonus") if isinstance(data.get("type_bonus"), dict) else {}
  per_problem = (
      data.get("per_problem_type_bonus")
      if isinstance(data.get("per_problem_type_bonus"), dict)
      else {}
  )
  return (
      {str(k): float(v) for k, v in type_bonus.items()},
      {
          str(problem): {str(k): float(v) for k, v in values.items()}
          for problem, values in per_problem.items()
          if isinstance(values, dict)
      },
  )


def problem_type_bonus(
    global_bonus: dict[str, float],
    per_problem_bonus: dict[str, dict[str, float]],
    problem_name: str,
) -> dict[str, float]:
  merged = dict(global_bonus)
  merged.update(per_problem_bonus.get(problem_name, {}))
  return merged


def csv_arg_set(value: str | None) -> set[str]:
  return {item.strip() for item in str(value or "").split(",") if item.strip()}


def candidate_ddar_error_key(error: str | None) -> str:
  lowered = str(error or "").lower()
  if "timeout" in lowered:
    return "timeout"
  if "pointtoocloseerror" in lowered:
    return "point_too_close"
  if "pointtoofarerror" in lowered:
    return "point_too_far"
  if "depcheckfailerror" in lowered:
    return "dep_check_fail"
  if "invalidquadsolveerror" in lowered:
    return "invalid_quad_solve"
  if "invalid predicate" in lowered:
    return "invalid_predicate"
  if "does not exist" in lowered:
    return "unknown_point"
  return "other_error"


def adaptive_type_penalty(
    failures: int,
    threshold: int,
    weight: float,
    max_penalty: float,
) -> float:
  if failures < threshold:
    return 0.0
  penalty = weight * math.log1p(failures - threshold + 1)
  if max_penalty > 0:
    penalty = min(max_penalty, penalty)
  return max(0.0, penalty)


def translate_candidate(raw: str, graph: gh.Graph, qs: Any | None) -> str:
  if qs is not None:
    return qs.try_translate_candidate(raw, graph, pr, pt)
  try:
    return ag.try_translate_constrained_to_construct(raw, graph)
  except Exception as exc:  # pylint: disable=broad-except
    return f"ERROR: translate_exception: {exc!r}"


def candidate_type_key(record: dict[str, Any], qs: Any | None) -> str:
  translation = str(record.get("translation") or "")
  if qs is not None and not translation.startswith("ERROR:"):
    return qs.construction_type_key(translation)
  raw = str(record.get("raw") or record.get("lm_out") or "")
  if qs is not None:
    if hasattr(qs, "raw_candidate_construction_type_hint"):
      return qs.raw_candidate_construction_type_hint(raw)
    if raw and " = " in raw and hasattr(qs, "construction_type_key"):
      return qs.construction_type_key(raw)
  return "unknown"


def note_adaptive_validation_failure(
    *,
    qs: Any | None,
    events_path: Path,
    adaptive_type_failures: dict[str, int],
    args: argparse.Namespace,
    problem: str,
    depth: int,
    node_index: int,
    record: dict[str, Any],
) -> None:
  if qs is None or not args.candidate_adaptive_type_penalty:
    return
  reason = qs.candidate_value_error_key(str(record.get("translation") or ""))
  if reason not in csv_arg_set(args.candidate_adaptive_type_penalty_reasons):
    return
  construction_type = candidate_type_key(record, qs)
  if not construction_type or construction_type == "unknown":
    return
  failures = adaptive_type_failures.get(construction_type, 0) + 1
  adaptive_type_failures[construction_type] = failures
  threshold = max(1, int(args.candidate_adaptive_type_penalty_threshold))
  if failures == threshold or (
      failures > threshold and failures % max(threshold, 16) == 0
  ):
    write_jsonl(
        events_path,
        {
            "event": "candidate_adaptive_type_failure",
            "problem": problem,
            "depth": depth,
            "node_index": node_index,
            "raw": record.get("raw") or record.get("lm_out"),
            "translation": record.get("translation"),
            "reason": reason,
            "source": record.get("source"),
            "candidate_construction_type": construction_type,
            "candidate_adaptive_type_failures": failures,
            "candidate_adaptive_type_penalty_threshold": threshold,
        },
    )


def note_adaptive_ddar_failure(
    *,
    qs: Any | None,
    events_path: Path,
    adaptive_type_failures: dict[str, int],
    args: argparse.Namespace,
    problem: str,
    depth: int,
    node_index: int,
    record: dict[str, Any],
    error: str | None,
) -> None:
  if (
      qs is None
      or not args.candidate_adaptive_type_penalty
      or not args.candidate_adaptive_type_penalty_ddar_errors
  ):
    return
  reason = candidate_ddar_error_key(error)
  if reason not in csv_arg_set(args.candidate_adaptive_type_penalty_ddar_error_reasons):
    return
  construction_type = candidate_type_key(record, qs)
  if not construction_type or construction_type == "unknown":
    return
  failures = adaptive_type_failures.get(construction_type, 0) + 1
  adaptive_type_failures[construction_type] = failures
  threshold = max(1, int(args.candidate_adaptive_type_penalty_threshold))
  if failures == threshold or (
      failures > threshold and failures % max(threshold, 16) == 0
  ):
    write_jsonl(
        events_path,
        {
            "event": "candidate_adaptive_type_ddar_failure",
            "problem": problem,
            "depth": depth,
            "node_index": node_index,
            "raw": record.get("raw") or record.get("lm_out"),
            "translation": record.get("translation"),
            "error": error,
            "reason": reason,
            "source": record.get("source"),
            "candidate_construction_type": construction_type,
            "candidate_adaptive_type_failures": failures,
            "candidate_adaptive_type_penalty_threshold": threshold,
        },
    )


def apply_adaptive_type_penalties(
    *,
    qs: Any | None,
    events_path: Path,
    records: list[dict[str, Any]],
    adaptive_type_failures: dict[str, int],
    args: argparse.Namespace,
    problem: str,
    depth: int,
    node_index: int,
) -> list[dict[str, Any]]:
  if (
      qs is None
      or not args.candidate_adaptive_type_penalty
      or not adaptive_type_failures
      or not records
  ):
    return records
  threshold = max(1, int(args.candidate_adaptive_type_penalty_threshold))
  weight = max(0.0, float(args.candidate_adaptive_type_penalty_weight))
  max_penalty = max(0.0, float(args.candidate_adaptive_type_penalty_max))
  applied = []
  for record in records:
    construction_type = candidate_type_key(record, qs)
    failures = adaptive_type_failures.get(construction_type, 0)
    penalty = adaptive_type_penalty(failures, threshold, weight, max_penalty)
    if penalty <= 0:
      continue
    base_score = float(record.get("_candidate_rerank_score", record.get("score", 0.0)) or 0.0)
    record["_candidate_base_rerank_score"] = base_score
    record["_candidate_rerank_score"] = base_score - penalty
    record["_candidate_adaptive_type_failures"] = failures
    record["_candidate_adaptive_type_penalty"] = penalty
    applied.append((construction_type, failures, penalty))
  if not applied:
    return records
  ordered = sorted(
      records,
      key=lambda record: float(
          record.get("_candidate_rerank_score", record.get("score", 0.0)) or 0.0
      ),
      reverse=True,
  )
  if hasattr(qs, "interleave_ranked_records_by_node"):
    ordered = qs.interleave_ranked_records_by_node(ordered)
  write_jsonl(
      events_path,
      {
          "event": "candidate_adaptive_type_penalty_applied",
          "problem": problem,
          "depth": depth,
          "node_index": node_index,
          "top": [
              {
                  "construction_type": typ,
                  "failures": failures,
                  "penalty": round(penalty, 4),
              }
              for typ, failures, penalty in sorted(
                  applied, key=lambda item: (-item[2], -item[1], item[0])
              )[:8]
          ],
      },
  )
  return ordered


def candidate_records_from_raws(
    raw_scores: list[tuple[str, float]],
    graph: gh.Graph,
    args: argparse.Namespace,
    qs: Any | None,
    source: str,
    start_rank: int = 0,
) -> list[dict[str, Any]]:
  forbidden_points = qs.existing_point_names(graph) if qs is not None else set()
  records = []
  for offset, (raw_in, score) in enumerate(raw_scores):
    raw = raw_in.strip()
    original_raw = raw
    repaired = False
    if qs is not None and args.candidate_point_repair:
      repaired_raw = qs.repair_candidate_point_name(raw, forbidden_points)
      repaired = repaired_raw != raw
      raw = repaired_raw
    if (
        qs is not None
        and args.candidate_point_mask
        and not args.candidate_point_repair
        and not qs.candidate_passes_point_mask(raw, forbidden_points)
    ):
      translation = "ERROR: point already exists"
    else:
      translation = translate_candidate(raw, graph, qs)
    records.append({
        "rank": start_rank + offset,
        "lm_out": raw,
        "raw": raw,
        "original_lm_out": original_raw,
        "translation": translation,
        "score": float(score),
        "lm_score": float(score),
        "source": source,
        "point_repaired": repaired,
    })
  return records


def candidate_records(
    outputs: dict[str, Any],
    graph: gh.Graph,
    args: argparse.Namespace,
    qs: Any | None,
) -> list[dict[str, Any]]:
  raw_scores = [
      (lm_out, float(score))
      for lm_out, score in reversed(list(zip(outputs["seqs_str"], outputs["scores"])))
  ]
  return candidate_records_from_raws(raw_scores, graph, args, qs, "ag1_lm")


def add_template_backfill_records(
    *,
    records: list[dict[str, Any]],
    graph: gh.Graph,
    p_cur: pr.Problem,
    args: argparse.Namespace,
    qs: Any | None,
) -> list[dict[str, Any]]:
  if (
      qs is None
      or not hasattr(qs, "template_backfill_candidates")
      or not args.candidate_template_backfill
      or args.candidate_template_backfill_extra_slots <= 0
  ):
    return records
  point_names = qs.existing_point_names(graph)
  excluded = set()
  if args.candidate_canonical_dedup:
    for record in records:
      translation = str(record.get("translation") or "")
      if translation and not translation.startswith("ERROR:"):
        try:
          excluded.add(qs.canonical_aux_key(translation))
        except Exception:  # pylint: disable=broad-except
          pass
  preferred_types = [
      qs.construction_type_key(str(record.get("translation") or ""))
      for record in records
      if str(record.get("translation") or "") and not str(record.get("translation")).startswith("ERROR:")
  ]
  raw_templates = qs.template_backfill_candidates(
      point_names,
      int(args.candidate_template_backfill_extra_slots),
      excluded if excluded else None,
      qs.goal_point_names(p_cur),
      preferred_types,
  )
  if not raw_templates:
    return records
  min_score = min([float(record.get("score", 0.0) or 0.0) for record in records] or [0.0])
  template_records = candidate_records_from_raws(
      [(raw, min_score - 5.0) for raw in raw_templates],
      graph,
      args,
      qs,
      "template_backfill",
      start_rank=len(records),
  )
  return records + template_records


def select_node_candidates_for_eval(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    qs: Any | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  limit = max(0, int(args.candidate_node_eval_limit or 0))
  type_cap = max(0, int(args.candidate_node_type_eval_cap or 0))
  if limit <= 0 and type_cap <= 0:
    return records, []
  selected = []
  pruned = []
  type_counts: dict[str, int] = {}
  for record in records:
    type_key = candidate_type_key(record, qs)
    if type_cap > 0 and type_counts.get(type_key, 0) >= type_cap:
      pruned_record = dict(record)
      pruned_record["_candidate_prune_reason"] = "node_type_cap"
      pruned.append(pruned_record)
      continue
    if limit > 0 and len(selected) >= limit:
      pruned_record = dict(record)
      pruned_record["_candidate_prune_reason"] = "node_eval_limit"
      pruned.append(pruned_record)
      continue
    selected.append(record)
    type_counts[type_key] = type_counts.get(type_key, 0) + 1
  return selected, pruned


def run_ag_problem(
    *,
    model: Any,
    pool: futures.ProcessPoolExecutor,
    args: argparse.Namespace,
    name: str,
    problem: pr.Problem,
    problem_dir: Path,
    events_path: Path,
    run_initial_ddar: bool,
) -> dict[str, Any]:
  start = time.time()
  out_file = problem_dir / "alphageometry_proof.txt"
  problem_dir.mkdir(parents=True, exist_ok=True)

  def timed_out() -> bool:
    return (
        args.problem_time_limit_sec > 0
        and time.time() - start > args.problem_time_limit_sec
    )

  write_jsonl(events_path, {"event": "ag_start", "problem": name})

  string = problem.setup_str_from_problem(ag.DEFINITIONS) + " {F1} x00"
  g, _ = gh.Graph.build_problem(problem, ag.DEFINITIONS)
  qs = getattr(args, "_qwen_search", None)
  if qs is not None and args.lm_fact_context_top_k > 0:
    try:
      root_fact = qs.run_ddar_once(
          g,
          problem,
          getattr(args, "_ag_ddar"),
          gh,
          args.root_fact_max_level,
          args.root_fact_timeout,
          str(events_path),
          "ag1_root_fact_context",
          args.lm_fact_context_top_k,
      )
      facts = list(root_fact.get("fact_context") or [])
      if facts:
        string = qs.build_lm_prompt(problem, ag.DEFINITIONS, facts)
      write_jsonl(
          events_path,
          {
              "event": "ag1_fact_context",
              "problem": name,
              "fact_context_count": len(facts),
              "facts": facts,
              "root_fact_solved": root_fact.get("solved"),
              "root_fact_status": root_fact.get("status"),
          },
      )
    except Exception as exc:  # pylint: disable=broad-except
      write_jsonl(
          events_path,
          {
              "event": "ag1_fact_context_error",
              "problem": name,
              "error": repr(exc),
              "traceback": traceback.format_exc(),
          },
      )

  if run_initial_ddar:
    initial_ddar = submit_ddar(
        pool,
        name,
        problem.txt(),
        out_file,
        problem_dir / "initial_ddar.log",
        keep_failed_log=True,
    ).result()
    write_jsonl(events_path, {"event": "ag_initial_ddar", **initial_ddar})
    if initial_ddar["solved"]:
      return {
          "problem": name,
          "solved": True,
          "method": "ddar_after_rename",
          "elapsed_sec": time.time() - start,
          "proof_file": str(out_file),
          "lm_calls": 0,
          "candidates_checked": 0,
          "status": "solved",
      }
  else:
    write_jsonl(
        events_path,
        {
            "event": "ag_initial_ddar_skipped",
            "problem": name,
            "reason": "already covered by DDAR prefilter",
        },
    )

  beam_queue = StableBeamQueue(max_size=args.beam_size)
  beam_queue.add(node=(g, string, problem.txt()), val=0.0)
  lm_calls = 0
  candidates_checked = 0
  adaptive_type_failures: dict[str, int] = {}

  for depth in range(args.search_depth):
    if timed_out():
      break
    write_jsonl(
        events_path,
        {
            "event": "ag_depth_start",
            "problem": name,
            "depth": depth,
            "nodes": len(beam_queue),
            "elapsed_sec": time.time() - start,
        },
    )
    new_queue = StableBeamQueue(max_size=args.beam_size)
    pending_candidates = []
    depth_seen_candidate_keys: set[str] = set()

    for node_index, (prev_score, (g_prev, prompt, pstring)) in enumerate(beam_queue):
      if timed_out():
        break

      p_cur = pr.Problem.from_txt(pstring, translate=False)
      decode_start = time.time()
      outputs = model.beam_decode(prompt, eos_tokens=[";"])
      lm_calls += 1
      records = candidate_records(outputs, g_prev, args, qs)
      records = add_template_backfill_records(
          records=records,
          graph=g_prev,
          p_cur=p_cur,
          args=args,
          qs=qs,
      )
      write_jsonl(
          events_path,
          {
              "event": "lm_decode",
              "problem": name,
              "depth": depth,
              "node_index": node_index,
              "elapsed_sec": time.time() - decode_start,
              "top_translation": records[0]["translation"] if records else "",
              "top_score": records[0]["score"] if records else None,
              "candidate_count_raw": len(records),
          },
      )

      valid = []
      seen_node_keys: set[str] = set()
      for record in records:
        if record["translation"].startswith("ERROR:"):
          note_adaptive_validation_failure(
              qs=qs,
              events_path=events_path,
              adaptive_type_failures=adaptive_type_failures,
              args=args,
              problem=name,
              depth=depth,
              node_index=node_index,
              record=record,
          )
          continue
        canonical_key = None
        if qs is not None and (
            args.candidate_canonical_dedup or args.candidate_depth_canonical_dedup
        ):
          try:
            canonical_key = qs.canonical_aux_key(record["translation"])
          except Exception:  # pylint: disable=broad-except
            canonical_key = record["translation"]
          if args.candidate_canonical_dedup and canonical_key in seen_node_keys:
            write_jsonl(
                events_path,
                {
                    "event": "candidate_filtered",
                    "problem": name,
                    "depth": depth,
                    "node_index": node_index,
                    "rank": record.get("rank"),
                    "raw": record.get("raw"),
                    "translation": record.get("translation"),
                    "reason": "duplicate_node_canonical",
                    "source": record.get("source"),
                },
            )
            continue
          if args.candidate_depth_canonical_dedup and canonical_key in depth_seen_candidate_keys:
            write_jsonl(
                events_path,
                {
                    "event": "candidate_filtered",
                    "problem": name,
                    "depth": depth,
                    "node_index": node_index,
                    "rank": record.get("rank"),
                    "raw": record.get("raw"),
                    "translation": record.get("translation"),
                    "reason": "duplicate_depth_canonical",
                    "source": record.get("source"),
                },
            )
            continue
          seen_node_keys.add(canonical_key)
          depth_seen_candidate_keys.add(canonical_key)
        candidate_pstring = ag.insert_aux_to_premise(pstring, record["translation"])
        valid.append({
            **record,
            "problem": name,
            "pstring": candidate_pstring,
        })

      if qs is not None and valid:
        valid = qs.rerank_candidate_records(
            valid,
            args.candidate_rerank,
            getattr(args, "_candidate_value_model", None),
            getattr(args, "_candidate_secondary_value_model", None),
            args.candidate_frontfill_limit,
            problem_type_bonus(
                getattr(args, "_candidate_static_type_bonus", {}),
                getattr(args, "_candidate_static_type_bonus_by_problem", {}),
                name,
            ),
        )
        write_jsonl(
            events_path,
            {
                "event": "candidate_rerank",
                "problem": name,
                "depth": depth,
                "node_index": node_index,
                "strategy": args.candidate_rerank,
                "candidate_count": len(valid),
                "top": [
                    {
                        "rank": cand.get("rank"),
                        "translation": cand.get("translation"),
                        "source": cand.get("source"),
                        "point_repaired": cand.get("point_repaired"),
                        "ag1_score": cand.get("score"),
                        "rerank_score": cand.get("_candidate_rerank_score"),
                        "base_rerank_score": cand.get("_candidate_base_rerank_score"),
                        "adaptive_penalty": cand.get(
                            "_candidate_adaptive_type_penalty"
                        ),
                        "rerank_phase": cand.get("_candidate_rerank_phase"),
                    }
                    for cand in valid[:8]
                ],
            },
        )
        valid = apply_adaptive_type_penalties(
            qs=qs,
            events_path=events_path,
            records=valid,
            adaptive_type_failures=adaptive_type_failures,
            args=args,
            problem=name,
            depth=depth,
            node_index=node_index,
        )

      eval_valid, pruned_valid = select_node_candidates_for_eval(valid, args, qs)
      for cand in pruned_valid:
        write_jsonl(
            events_path,
            {
                "event": "candidate_filtered",
                "problem": name,
                "depth": depth,
                "node_index": node_index,
                "rank": cand.get("rank"),
                "raw": cand.get("raw"),
                "translation": cand.get("translation"),
                "reason": cand.get("_candidate_prune_reason"),
                "source": cand.get("source"),
                "candidate_rerank_score": cand.get("_candidate_rerank_score"),
                "candidate_base_rerank_score": cand.get(
                    "_candidate_base_rerank_score"
                ),
                "candidate_adaptive_type_penalty": cand.get(
                    "_candidate_adaptive_type_penalty"
                ),
            },
        )

      for cand in eval_valid:
        candidates_checked += 1
        cand_dir = problem_dir / "candidates" / f"d{depth:02d}_n{node_index:04d}_r{cand['rank']:02d}"
        pending_candidates.append({
            "candidate": cand,
            "node_index": node_index,
            "prompt": prompt,
            "prev_score": prev_score,
            "future": submit_ddar(
                pool,
                name,
                cand["pstring"],
                cand_dir / "proof.txt",
                cand_dir / "ddar.log",
                args.keep_failed_candidate_logs,
                args.candidate_ddar_timeout_sec,
            ),
        })

    cand_results = []
    for item in pending_candidates:
      result = item["future"].result()
      cand_results.append((item, result))

    for item, result in cand_results:
      cand = item["candidate"]
      if result["solved"]:
        shutil.copyfile(result["out_file"], out_file)
        write_jsonl(
            events_path,
            {
                "event": "ag_solved",
                "problem": name,
                "depth": depth,
                "node_index": item["node_index"],
                "rank": cand["rank"],
                "score": cand["score"],
                "translation": cand["translation"],
                "elapsed_sec": time.time() - start,
                "lm_calls": lm_calls,
                "candidates_checked": candidates_checked,
                "candidate_proof_file": result["out_file"],
            },
        )
        return {
            "problem": name,
            "solved": True,
            "method": "alphageometry",
            "elapsed_sec": time.time() - start,
            "proof_file": str(out_file),
            "lm_calls": lm_calls,
            "candidates_checked": candidates_checked,
            "status": "solved",
        }

    for item, result in cand_results:
      cand = item["candidate"]
      if result["error"]:
        note_adaptive_ddar_failure(
            qs=qs,
            events_path=events_path,
            adaptive_type_failures=adaptive_type_failures,
            args=args,
            problem=name,
            depth=depth,
            node_index=item["node_index"],
            record=cand,
            error=result["error"],
        )
        write_jsonl(
          events_path,
          {
                "event": "candidate_error",
                "problem": name,
                "depth": depth,
                "node_index": item["node_index"],
                "rank": cand["rank"],
                "error": result["error"],
                "log_file": result["log_file"],
            },
        )
      p_new = pr.Problem.from_txt(cand["pstring"], translate=False)
      g_new, _ = gh.Graph.build_problem(p_new, ag.DEFINITIONS)
      new_queue.add(
          node=(
              g_new,
              item["prompt"] + " " + cand["lm_out"] + " x00",
              cand["pstring"],
          ),
          val=item["prev_score"]
          + float(cand.get("_candidate_rerank_score", cand["score"]) or 0.0),
      )

    beam_queue = new_queue

  status = "timeout" if timed_out() else "not_solved"
  write_jsonl(
      events_path,
      {
          "event": "ag_end",
          "problem": name,
          "status": status,
          "elapsed_sec": time.time() - start,
          "lm_calls": lm_calls,
          "candidates_checked": candidates_checked,
      },
  )
  return {
      "problem": name,
      "solved": False,
      "method": "alphageometry",
      "elapsed_sec": time.time() - start,
      "proof_file": "",
      "lm_calls": lm_calls,
      "candidates_checked": candidates_checked,
      "status": status,
  }


def main() -> int:
  args = parse_args()
  ensure_absl_flags_parsed()

  args.problems_file = str(Path(args.problems_file).resolve())
  args.defs_file = str(Path(args.defs_file).resolve())
  args.rules_file = str(Path(args.rules_file).resolve())
  args.ckpt_path = str(Path(args.ckpt_path).resolve())
  args.vocab_path = str(Path(args.vocab_path).resolve())
  args.meliad_path = str(Path(args.meliad_path).resolve())
  if args.qwen_search_path:
    args.qwen_search_path = str(Path(args.qwen_search_path).resolve())

  results_dir = Path(args.results_dir).resolve()
  proofs_dir = results_dir / "proofs"
  logs_dir = results_dir / "logs"
  results_dir.mkdir(parents=True, exist_ok=True)
  proofs_dir.mkdir(parents=True, exist_ok=True)
  logs_dir.mkdir(parents=True, exist_ok=True)
  events_path = results_dir / "events.jsonl"
  summary_path = results_dir / "summary.json"
  events_path.unlink(missing_ok=True)
  summary_path.unlink(missing_ok=True)

  ag.DEFINITIONS = pr.Definition.from_txt_file(args.defs_file, to_dict=True)
  ag.RULES = pr.Theorem.from_txt_file(args.rules_file, to_dict=True)
  args._qwen_search = load_qwen_search(args)
  if args._qwen_search is not None:
    import ddar as ag_ddar  # pylint: disable=import-error,import-outside-toplevel

    args._ag_ddar = ag_ddar
    args._qwen_search.DEFINITIONS = ag.DEFINITIONS
    args._qwen_search.RULES = ag.RULES
  else:
    args._ag_ddar = None
  args._candidate_value_model = (
      args._qwen_search.load_candidate_value_model(args.candidate_value_model)
      if args._qwen_search is not None
      else None
  )
  args._candidate_secondary_value_model = (
      args._qwen_search.load_candidate_value_model(
          args.candidate_secondary_value_model
      )
      if args._qwen_search is not None
      else None
  )
  (
      args._candidate_static_type_bonus,
      args._candidate_static_type_bonus_by_problem,
  ) = load_static_type_bonus(args.candidate_static_progress_type_bonus)

  problems_ddar = pr.Problem.from_txt_file(args.problems_file, to_dict=True, translate=False)
  problems_ag = pr.Problem.from_txt_file(args.problems_file, to_dict=True, translate=True)
  problem_texts = raw_problem_texts(args.problems_file)
  names = list(problems_ddar)
  if args.problem:
    requested = set(args.problem)
    names = [name for name in names if name in requested]

  write_jsonl(
      events_path,
      {
          "event": "benchmark_start",
          "problems": names,
          "batch_size": args.batch_size,
          "beam_size": args.beam_size,
          "search_depth": args.search_depth,
          "workers": args.workers,
          "candidate_rerank": args.candidate_rerank,
          "candidate_value_model": args.candidate_value_model,
          "candidate_secondary_value_model": args.candidate_secondary_value_model,
          "candidate_static_progress_type_bonus": args.candidate_static_progress_type_bonus,
          "lm_fact_context_top_k": args.lm_fact_context_top_k,
      },
  )

  ctx = mp.get_context("spawn")
  rows: list[dict[str, Any]] = []
  solved_by_ddar = set()

  with futures.ProcessPoolExecutor(
      max_workers=args.workers,
      mp_context=ctx,
      initializer=worker_init,
      initargs=(args.defs_file, args.rules_file),
  ) as pool:
    if not args.skip_ddar_prefilter:
      ddar_futures = {}
      for name in names:
        proof_dir = proofs_dir / name
        fut = submit_ddar(
            pool,
            name,
            problem_texts[name],
            proof_dir / "ddar_proof.txt",
            logs_dir / f"{name}.ddar.log",
            keep_failed_log=True,
        )
        ddar_futures[fut] = name

      for fut in futures.as_completed(ddar_futures):
        result = fut.result()
        write_jsonl(events_path, {"event": "ddar_result", **result})
        if result["solved"]:
          solved_by_ddar.add(result["problem"])
          rows.append({
              "problem": result["problem"],
              "solved": True,
              "method": "ddar",
              "elapsed_sec": result["elapsed_sec"],
              "proof_file": result["out_file"],
              "lm_calls": 0,
              "candidates_checked": 0,
              "status": "solved",
          })

    model, lm_info = load_lm(args)
    write_jsonl(events_path, {"event": "lm_loaded", **lm_info})

    for name in names:
      if name in solved_by_ddar:
        continue
      result = run_ag_problem(
          model=model,
          pool=pool,
          args=args,
          name=name,
          problem=problems_ag[name],
          problem_dir=proofs_dir / name,
          events_path=events_path,
          run_initial_ddar=not args.skip_initial_ddar,
      )
      rows.append(result)
      summary = {
          "solved": sum(1 for row in rows if row["solved"]),
          "total_finished": len(rows),
          "total_requested": len(names),
          "rows": sorted(rows, key=lambda row: names.index(row["problem"])),
          "lm_info": lm_info,
      }
      summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

  summary = {
      "solved": sum(1 for row in rows if row["solved"]),
      "total_finished": len(rows),
      "total_requested": len(names),
      "rows": sorted(rows, key=lambda row: names.index(row["problem"])),
  }
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
  write_jsonl(events_path, {"event": "benchmark_end", **summary})
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  sys.exit(main())
