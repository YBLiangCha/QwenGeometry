# Copyright 2026
#
# Utility script for reproducing the IMO-AG-30 benchmark with a single LM load
# and parallel DDAR checks. This keeps the original AlphaGeometry code path but
# avoids restarting the 150M language model once per problem.

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
from pathlib import Path
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
        solved = ag.run_ddar(g, p, out_file)
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
) -> futures.Future[dict[str, Any]]:
  return pool.submit(
      run_ddar_worker,
      problem_name,
      pstring,
      str(out_file),
      str(log_file),
      keep_failed_log,
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


def candidate_records(outputs: dict[str, Any], graph: gh.Graph) -> list[dict[str, Any]]:
  translations = []
  for output in outputs["seqs_str"]:
    try:
      translations.append(ag.try_translate_constrained_to_construct(output, graph))
    except Exception as exc:  # pylint: disable=broad-except
      translations.append(f"ERROR: translate_exception: {exc!r}")
  records = []
  for rank, (lm_out, translation, score) in enumerate(
      reversed(list(zip(outputs["seqs_str"], translations, outputs["scores"])))
  ):
    records.append({
        "rank": rank,
        "lm_out": lm_out,
        "translation": translation,
        "score": float(score),
    })
  return records


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

    for node_index, (prev_score, (g_prev, prompt, pstring)) in enumerate(beam_queue):
      if timed_out():
        break

      decode_start = time.time()
      outputs = model.beam_decode(prompt, eos_tokens=[";"])
      lm_calls += 1
      records = candidate_records(outputs, g_prev)
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
          },
      )

      valid = []
      for record in records:
        if record["translation"].startswith("ERROR:"):
          continue
        candidate_pstring = ag.insert_aux_to_premise(pstring, record["translation"])
        valid.append({
            **record,
            "raw": record["lm_out"],
            "lm_score": record["score"],
            "problem": name,
            "source": "ag1_lm",
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
                        "ag1_score": cand.get("score"),
                        "rerank_score": cand.get("_candidate_rerank_score"),
                        "rerank_phase": cand.get("_candidate_rerank_phase"),
                    }
                    for cand in valid[:8]
                ],
            },
        )

      for cand in valid:
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
