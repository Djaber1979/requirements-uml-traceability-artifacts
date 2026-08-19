#!/usr/bin/env python3
"""Prompt 9: frozen within-model run-resampling robustness analysis.

Uses only Prompt-4 Model×Run primary semantic metrics. No semantic model,
embedding, requirement scoring, or query ranking is recomputed.
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "artifacts/semantic_retrieval/model_run_primary_metrics.csv"
CORPUS = ROOT / "artifacts/semantic_retrieval/corpus_primary_metrics.json"
AMEND = ROOT / "PROTOCOL_AMENDMENTS.md"
ART = ROOT / "artifacts/run_resampling"
REPORTS = ROOT / "reports"
ART.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 20260819
N_REPS = 2000
RUNS = tuple(range(1, 11))
MODEL_ORDER = [
    "ChatGPT-4o",
    "ChatGPT-o3",
    "Claude3.7",
    "DeepSeek(R1)",
    "Gemini-2.5-Pro-Preview-05-06",
    "Grok3",
    "Llama4",
    "Mistral",
    "Qwen3",
]
METHODS = ("TFIDF", "MPNET")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def linear_quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "Mean": statistics.fmean(values),
        "Median": statistics.median(values),
        "SD": statistics.stdev(values),
        "P025": linear_quantile(values, 0.025),
        "P975": linear_quantile(values, 0.975),
        "Min": min(values),
        "Max": max(values),
    }


def main() -> int:
    amendment = AMEND.read_text(encoding="utf-8")
    if "## Amendment 007 — Prompt 9 run-resampling operationalization" not in amendment:
        raise SystemExit("Prompt-9 prospective Amendment 007 is missing")

    rows = read_csv(CELL)
    if len(rows) != 138:
        raise SystemExit(f"Expected 138 Prompt-4 method×eligible-cell rows, got {len(rows)}")

    by_method: dict[str, dict[tuple[str, int], tuple[float, float]]] = defaultdict(dict)
    model_eligible: dict[str, set[int]] = {m: set() for m in MODEL_ORDER}
    for r in rows:
        method = r["Method"]
        model = r["Model"]
        run = int(r["Run"])
        if method not in METHODS or model not in MODEL_ORDER or run not in RUNS:
            raise SystemExit(f"Unexpected cell identity: {method} {model} {run}")
        by_method[method][(model, run)] = (float(r["Hit1Mean"]), float(r["MRRMean"]))
        model_eligible[model].add(run)

    if set(by_method) != set(METHODS):
        raise SystemExit("Expected TFIDF and MPNET cell metrics")
    if set(by_method["TFIDF"]) != set(by_method["MPNET"]):
        raise SystemExit("TFIDF/MPNET eligible Model×Run identities differ")
    if len(by_method["TFIDF"]) != 69:
        raise SystemExit("Expected exactly 69 eligible Model×Run cells")

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    anchors = {
        "TFIDF": {
            "Hit1": float(corpus["methods"]["TFIDF"]["Hit1EqualCellMacro"]),
            "MRR": float(corpus["methods"]["TFIDF"]["MRREqualCellMacro"]),
        },
        "MPNET": {
            "Hit1": float(corpus["methods"]["MPNET"]["Hit1EqualCellMacro"]),
            "MRR": float(corpus["methods"]["MPNET"]["MRREqualCellMacro"]),
        },
    }

    # Generate one frozen schedule, shared across methods.
    rng = random.Random(SEED)
    schedule_rows: list[dict] = []
    schedule: dict[tuple[int, str], list[int]] = {}
    for rep in range(1, N_REPS + 1):
        for model in MODEL_ORDER:
            draws = [rng.choice(RUNS) for _ in range(10)]
            schedule[(rep, model)] = draws
            row = {"Replicate": rep, "Model": model}
            row.update({f"Draw{i+1}": x for i, x in enumerate(draws)})
            schedule_rows.append(row)
    write_csv(
        ART / "resampling_schedule.csv",
        ["Replicate", "Model"] + [f"Draw{i}" for i in range(1, 11)],
        schedule_rows,
    )

    overall_rows: list[dict] = []
    by_model_rows: list[dict] = []

    for method in METHODS:
        cells = by_method[method]
        for rep in range(1, N_REPS + 1):
            pooled_h1: list[float] = []
            pooled_mrr: list[float] = []
            contributing_models = 0
            for model in MODEL_ORDER:
                mh1: list[float] = []
                mmrr: list[float] = []
                draws = schedule[(rep, model)]
                for run in draws:
                    metric = cells.get((model, run))
                    if metric is None:
                        continue
                    h1, mrr = metric
                    mh1.append(h1)
                    mmrr.append(mrr)
                    pooled_h1.append(h1)
                    pooled_mrr.append(mrr)
                all_na = len(mh1) == 0
                if not all_na:
                    contributing_models += 1
                by_model_rows.append({
                    "Method": method,
                    "Replicate": rep,
                    "Model": model,
                    "Hit1": "" if all_na else statistics.fmean(mh1),
                    "MRR": "" if all_na else statistics.fmean(mmrr),
                    "NonNADraws": len(mh1),
                    "AllNA": int(all_na),
                })
            if not pooled_h1:
                overall_rows.append({
                    "Method": method,
                    "Replicate": rep,
                    "Hit1": "",
                    "MRR": "",
                    "NonNADraws": 0,
                    "ContributingModels": 0,
                })
            else:
                overall_rows.append({
                    "Method": method,
                    "Replicate": rep,
                    "Hit1": statistics.fmean(pooled_h1),
                    "MRR": statistics.fmean(pooled_mrr),
                    "NonNADraws": len(pooled_h1),
                    "ContributingModels": contributing_models,
                })

    write_csv(
        ART / "overall_replicates.csv",
        ["Method", "Replicate", "Hit1", "MRR", "NonNADraws", "ContributingModels"],
        overall_rows,
    )
    write_csv(
        ART / "by_model_replicates.csv",
        ["Method", "Replicate", "Model", "Hit1", "MRR", "NonNADraws", "AllNA"],
        by_model_rows,
    )

    # Overall summary.
    summary_rows: list[dict] = []
    for method in METHODS:
        group = [r for r in overall_rows if r["Method"] == method and r["Hit1"] != ""]
        if len(group) != N_REPS:
            raise SystemExit(f"Unexpected all-NA overall replicate for {method}")
        for metric in ("Hit1", "MRR"):
            vals = [float(r[metric]) for r in group]
            s = summarize(vals)
            summary_rows.append({
                "Method": method,
                "Metric": metric,
                "ObservedAnchor": anchors[method][metric],
                "Replicates": N_REPS,
                **s,
                "MedianMinusAnchor": s["Median"] - anchors[method][metric],
                "MeanMinusAnchor": s["Mean"] - anchors[method][metric],
            })
    write_csv(
        ART / "overall_summary.csv",
        ["Method", "Metric", "ObservedAnchor", "Replicates", "Mean", "Median", "SD", "P025", "P975", "Min", "Max", "MedianMinusAnchor", "MeanMinusAnchor"],
        summary_rows,
    )

    # By-model summary.
    model_summary_rows: list[dict] = []
    for method in METHODS:
        for model in MODEL_ORDER:
            group = [r for r in by_model_rows if r["Method"] == method and r["Model"] == model]
            all_na_n = sum(int(r["AllNA"]) for r in group)
            for metric in ("Hit1", "MRR"):
                vals = [float(r[metric]) for r in group if r[metric] != ""]
                s = summarize(vals)
                model_summary_rows.append({
                    "Method": method,
                    "Model": model,
                    "Metric": metric,
                    "OriginalEligibleRuns": len(model_eligible[model]),
                    "EvaluableReplicates": len(vals),
                    "AllNAReplicates": all_na_n,
                    "AllNARate": all_na_n / N_REPS,
                    **s,
                })
    write_csv(
        ART / "by_model_summary.csv",
        ["Method", "Model", "Metric", "OriginalEligibleRuns", "EvaluableReplicates", "AllNAReplicates", "AllNARate", "Mean", "Median", "SD", "P025", "P975", "Min", "Max"],
        model_summary_rows,
    )

    # Draw-coverage profile is method-invariant because the same eligibility grid/schedule is shared.
    first_method_rows = [r for r in overall_rows if r["Method"] == "TFIDF"]
    non_na_draws = [int(r["NonNADraws"]) for r in first_method_rows]
    contributing = [int(r["ContributingModels"]) for r in first_method_rows]
    coverage = {
        "seed": SEED,
        "replicates": N_REPS,
        "models": MODEL_ORDER,
        "source_runs_per_model": 10,
        "eligible_model_run_cells": 69,
        "zero_query_model_run_cells": 21,
        "non_na_draws": summarize([float(x) for x in non_na_draws]),
        "contributing_models": summarize([float(x) for x in contributing]),
        "replicates_with_all_9_models_contributing": sum(x == 9 for x in contributing),
        "replicates_with_zero_non_na_draws": sum(x == 0 for x in non_na_draws),
        "interpretation": "Perturbation distribution under within-model run resampling; not a confidence interval or p-value.",
    }
    (ART / "coverage_summary.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Reader-facing reports.
    lookup = {(r["Method"], r["Metric"]): r for r in summary_rows}
    lines = [
        "# Prompt 9 — Run-Resampling Robustness",
        "",
        "## Scope",
        "",
        f"Executed exactly {N_REPS:,} within-model run-resampling replicates with seed `{SEED}` using the frozen Prompt-4 Model×Run Hit@1/MRR metrics. The same schedule was used for TF-IDF and MPNet. Zero-query draws remained NA, repeated draws retained multiplicity, and no semantic model or retrieval score was recomputed.",
        "",
        "## Overall perturbation distributions",
        "",
        "| Method | Metric | Prompt-4 anchor | Median | Mean | SD | Central 95% perturbation range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for metric in ("Hit1", "MRR"):
            r = lookup[(method, metric)]
            lines.append(
                f"| {method} | {metric} | {float(r['ObservedAnchor']):.6f} | {float(r['Median']):.6f} | {float(r['Mean']):.6f} | {float(r['SD']):.6f} | [{float(r['P025']):.6f}, {float(r['P975']):.6f}] |"
            )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "These distributions quantify sensitivity to which of the ten observed stochastic runs are represented in repeated within-model draws. They are perturbation/resampling distributions, not confidence intervals for future tasks, future domains, or future model versions, and they are not inferential p-values.",
        "",
        "Generator-level summaries are descriptive only and must not be presented as a universal model leaderboard.",
    ]
    (REPORTS / "RQ4_RUN_RESAMPLING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    handoff = [
        "# HANDOFF 09 — RUN RESAMPLING",
        "",
        "## STATUS",
        "",
        "**PASS — pending independent Prompt-9 validator**",
        "",
        "## Scope",
        "",
        f"Executed the frozen {N_REPS:,}-replicate within-model run-resampling analysis with seed `{SEED}` from Prompt-4 Model×Run metrics only.",
        "",
        "## Gate",
        "",
        "Acceptance requires independent schedule/metric reconstruction, regression tests, deterministic rerun, and scientific-boundary checks.",
    ]
    (REPORTS / "HANDOFF_09_RUN_RESAMPLING.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")

    print(json.dumps({"status": "PROMPT9_EXECUTED", "coverage": coverage, "summary": summary_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
