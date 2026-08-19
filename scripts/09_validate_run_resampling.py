#!/usr/bin/env python3
"""Independent validator for Prompt 9 run-resampling robustness outputs."""
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
ART = ROOT / "artifacts/run_resampling"
REPORTS = ROOT / "reports"
SEED = 20260819
N_REPS = 2000
RUNS = list(range(1, 11))
MODELS = [
    "ChatGPT-4o", "ChatGPT-o3", "Claude3.7", "DeepSeek(R1)",
    "Gemini-2.5-Pro-Preview-05-06", "Grok3", "Llama4", "Mistral", "Qwen3",
]
METHODS = ["TFIDF", "MPNET"]


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def qlinear(values, q):
    xs = sorted(float(x) for x in values)
    p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    if lo == hi:
        return xs[lo]
    f = p - lo
    return xs[lo] * (1-f) + xs[hi] * f


def close(a, b, tol=1e-12):
    return abs(float(a)-float(b)) <= tol


def main():
    checks = 0
    failures = []
    def ck(name, cond):
        nonlocal checks
        checks += 1
        if not cond:
            failures.append(name)

    cell_rows = read_csv(CELL)
    cells = defaultdict(dict)
    for r in cell_rows:
        cells[r["Method"]][(r["Model"], int(r["Run"]))] = (float(r["Hit1Mean"]), float(r["MRRMean"]))
    ck("eligible_grid_equal", set(cells["TFIDF"]) == set(cells["MPNET"]))
    ck("eligible_grid_69", len(cells["TFIDF"]) == 69)

    schedule_rows = read_csv(ART / "resampling_schedule.csv")
    ck("schedule_row_count", len(schedule_rows) == N_REPS * len(MODELS))
    schedule = {}
    for r in schedule_rows:
        key = (int(r["Replicate"]), r["Model"])
        draws = [int(r[f"Draw{i}"]) for i in range(1,11)]
        schedule[key] = draws
        ck(f"schedule_runs_{key}", all(x in RUNS for x in draws))

    # Regenerate the complete schedule independently from the frozen RNG contract.
    rng = random.Random(SEED)
    for rep in range(1, N_REPS+1):
        for model in MODELS:
            expected = [rng.choice(RUNS) for _ in range(10)]
            ck(f"schedule_exact_{rep}_{model}", schedule[(rep, model)] == expected)

    overall = read_csv(ART / "overall_replicates.csv")
    by_model = read_csv(ART / "by_model_replicates.csv")
    summary = read_csv(ART / "overall_summary.csv")
    by_model_summary = read_csv(ART / "by_model_summary.csv")
    ck("overall_rows", len(overall) == N_REPS*2)
    ck("by_model_rows", len(by_model) == N_REPS*2*9)
    ck("summary_rows", len(summary) == 4)
    ck("by_model_summary_rows", len(by_model_summary) == 36)

    overall_map = {(r["Method"], int(r["Replicate"])): r for r in overall}
    model_map = {(r["Method"], int(r["Replicate"]), r["Model"]): r for r in by_model}

    # Recompute every replicate independently from the source cell table.
    for method in METHODS:
        for rep in range(1, N_REPS+1):
            pooled_h1, pooled_mrr = [], []
            contributing = 0
            for model in MODELS:
                mh1, mmrr = [], []
                for run in schedule[(rep, model)]:
                    v = cells[method].get((model, run))
                    if v is not None:
                        mh1.append(v[0]); mmrr.append(v[1])
                        pooled_h1.append(v[0]); pooled_mrr.append(v[1])
                observed = model_map[(method, rep, model)]
                ck(f"model_n_{method}_{rep}_{model}", int(observed["NonNADraws"]) == len(mh1))
                ck(f"model_allna_{method}_{rep}_{model}", int(observed["AllNA"]) == int(len(mh1)==0))
                if mh1:
                    contributing += 1
                    ck(f"model_h1_{method}_{rep}_{model}", close(observed["Hit1"], statistics.fmean(mh1)))
                    ck(f"model_mrr_{method}_{rep}_{model}", close(observed["MRR"], statistics.fmean(mmrr)))
                else:
                    ck(f"model_h1_blank_{method}_{rep}_{model}", observed["Hit1"] == "")
                    ck(f"model_mrr_blank_{method}_{rep}_{model}", observed["MRR"] == "")
            obs = overall_map[(method, rep)]
            ck(f"overall_n_{method}_{rep}", int(obs["NonNADraws"]) == len(pooled_h1))
            ck(f"overall_models_{method}_{rep}", int(obs["ContributingModels"]) == contributing)
            ck(f"overall_h1_{method}_{rep}", close(obs["Hit1"], statistics.fmean(pooled_h1)))
            ck(f"overall_mrr_{method}_{rep}", close(obs["MRR"], statistics.fmean(pooled_mrr)))

    corpus = json.loads(CORPUS.read_text())
    anchors = {
        ("TFIDF","Hit1"): float(corpus["methods"]["TFIDF"]["Hit1EqualCellMacro"]),
        ("TFIDF","MRR"): float(corpus["methods"]["TFIDF"]["MRREqualCellMacro"]),
        ("MPNET","Hit1"): float(corpus["methods"]["MPNET"]["Hit1EqualCellMacro"]),
        ("MPNET","MRR"): float(corpus["methods"]["MPNET"]["MRREqualCellMacro"]),
    }
    summary_map = {(r["Method"],r["Metric"]):r for r in summary}
    for method in METHODS:
        for metric in ["Hit1","MRR"]:
            vals = [float(overall_map[(method, rep)][metric]) for rep in range(1,N_REPS+1)]
            s = summary_map[(method,metric)]
            ck(f"anchor_{method}_{metric}", close(s["ObservedAnchor"], anchors[(method,metric)]))
            ck(f"mean_{method}_{metric}", close(s["Mean"], statistics.fmean(vals)))
            ck(f"median_{method}_{metric}", close(s["Median"], statistics.median(vals)))
            ck(f"sd_{method}_{metric}", close(s["SD"], statistics.stdev(vals)))
            ck(f"p025_{method}_{metric}", close(s["P025"], qlinear(vals,.025)))
            ck(f"p975_{method}_{metric}", close(s["P975"], qlinear(vals,.975)))
            ck(f"min_{method}_{metric}", close(s["Min"], min(vals)))
            ck(f"max_{method}_{metric}", close(s["Max"], max(vals)))

    # Model summaries: independently recompute all values and all-NA rates.
    bms = {(r["Method"],r["Model"],r["Metric"]):r for r in by_model_summary}
    for method in METHODS:
        for model in MODELS:
            elig = len({run for (m,run) in cells[method] if m==model})
            for metric in ["Hit1","MRR"]:
                grp = [model_map[(method,rep,model)] for rep in range(1,N_REPS+1)]
                vals = [float(r[metric]) for r in grp if r[metric] != ""]
                allna = sum(int(r["AllNA"]) for r in grp)
                s = bms[(method,model,metric)]
                ck(f"bm_elig_{method}_{model}_{metric}", int(s["OriginalEligibleRuns"]) == elig)
                ck(f"bm_eval_{method}_{model}_{metric}", int(s["EvaluableReplicates"]) == len(vals))
                ck(f"bm_allna_{method}_{model}_{metric}", int(s["AllNAReplicates"]) == allna)
                ck(f"bm_rate_{method}_{model}_{metric}", close(s["AllNARate"], allna/N_REPS))
                ck(f"bm_mean_{method}_{model}_{metric}", close(s["Mean"], statistics.fmean(vals)))
                ck(f"bm_median_{method}_{model}_{metric}", close(s["Median"], statistics.median(vals)))
                ck(f"bm_sd_{method}_{model}_{metric}", close(s["SD"], statistics.stdev(vals)))
                ck(f"bm_p025_{method}_{model}_{metric}", close(s["P025"], qlinear(vals,.025)))
                ck(f"bm_p975_{method}_{model}_{metric}", close(s["P975"], qlinear(vals,.975)))

    coverage = json.loads((ART / "coverage_summary.json").read_text())
    ck("coverage_seed", coverage["seed"] == SEED)
    ck("coverage_reps", coverage["replicates"] == N_REPS)
    ck("coverage_zero_overall", coverage["replicates_with_zero_non_na_draws"] == 0)

    report = REPORTS / "PROMPT9_VALIDATION.md"
    status = "PASS" if not failures else "FAIL"
    report.write_text(
        "# Prompt 9 Independent Validation\n\n"
        f"## Overall Assessment: {status}\n\n"
        f"Independent checks executed: **{checks:,}**.\n\n"
        f"Failures: **{len(failures)}**.\n\n"
        "Validated the exact 2,000-replicate schedule, every overall and by-model replicate metric, Prompt-4 anchors, distribution summaries, missingness handling, and frozen seed.\n\n"
        + ("Safe to accept Prompt 9 run-resampling analysis: **YES**\n" if not failures else "Safe to accept Prompt 9 run-resampling analysis: **NO**\n")
        + ("\nFailed checks:\n- " + "\n- ".join(failures[:50]) + "\n" if failures else ""),
        encoding="utf-8",
    )
    print(json.dumps({"status":status,"checks":checks,"failures":failures[:20]}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
