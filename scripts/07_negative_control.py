from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "derived/behaviors/primary_semantic_queries.csv"
SEM = ROOT / "artifacts/semantic_retrieval"
TFIDF = SEM / "tfidf_scores.csv"
MPNET = SEM / "mpnet_scores.csv"
OBSERVED = SEM / "corpus_primary_metrics.json"
AMENDMENTS = ROOT / "PROTOCOL_AMENDMENTS.md"
OUT = ROOT / "artifacts/negative_control"
REPORT = ROOT / "reports/RQ4_NEGATIVE_CONTROL.md"
HANDOFF = ROOT / "reports/HANDOFF_07_NEGATIVE_CONTROL.md"

B = 2000
SEED = 20260818
UC_COUNT = 21


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_queries():
    with QUERIES.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 2369:
        raise SystemExit(f"Expected 2369 primary queries, found {len(rows)}")
    prohibited = {"UC_Action", "Best_Match_Action", "SimilarityScore"}
    if prohibited.intersection(rows[0].keys()):
        raise SystemExit("Prohibited semantic/legacy fields present in primary query file")
    ids = [r["QueryID"] for r in rows]
    declared = np.asarray([int(r["DeclaredUC"]) - 1 for r in rows], dtype=np.int16)
    cells = []
    cell_map = {}
    for r in rows:
        key = (r["Model"], int(r["Run"]))
        if key not in cell_map:
            cell_map[key] = len(cell_map)
        cells.append(cell_map[key])
    cell_idx = np.asarray(cells, dtype=np.int16)
    if len(cell_map) != 69:
        raise SystemExit(f"Expected 69 eligible Model×Run cells, found {len(cell_map)}")
    return rows, ids, declared, cell_idx, cell_map


def read_score_matrix(path: Path, expected_ids):
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected_cols = ["QueryID"] + [f"UC{i}" for i in range(1, 22)]
        if reader.fieldnames != expected_cols:
            raise SystemExit(f"Unexpected score schema in {path}")
        ids = []
        vals = []
        for r in reader:
            ids.append(r["QueryID"])
            vals.append([float(r[f"UC{i}"]) for i in range(1, 22)])
    if ids != expected_ids:
        raise SystemExit(f"QueryID order mismatch in {path}")
    arr = np.asarray(vals, dtype=np.float64)
    if arr.shape != (2369, 21) or not np.isfinite(arr).all():
        raise SystemExit(f"Invalid score matrix {path}: {arr.shape}")
    return arr


def metric_for_mapping(scores, mapping, declared_idx, cell_idx, cell_counts):
    # mapping[label index] = frozen requirement-text score-column index.
    assigned = scores[:, mapping]
    row_index = np.arange(scores.shape[0])
    target = assigned[row_index, declared_idx]
    greater = np.sum(assigned > target[:, None], axis=1)
    label_idx = np.arange(UC_COUNT, dtype=np.int16)[None, :]
    lower_label_ties = np.sum((assigned == target[:, None]) & (label_idx < declared_idx[:, None]), axis=1)
    ranks = 1 + greater + lower_label_ties
    hits = (ranks == 1).astype(np.float64)
    rr = 1.0 / ranks.astype(np.float64)
    hit_by_cell = np.bincount(cell_idx, weights=hits, minlength=len(cell_counts)) / cell_counts
    rr_by_cell = np.bincount(cell_idx, weights=rr, minlength=len(cell_counts)) / cell_counts
    return float(np.mean(hit_by_cell)), float(np.mean(rr_by_cell))


def make_schedule():
    rng = random.Random(SEED)
    schedule = []
    base = list(range(UC_COUNT))
    for _ in range(B):
        p = base.copy()
        rng.shuffle(p)
        schedule.append(p)
    return schedule


def write_schedule(schedule):
    path = OUT / "permutation_schedule.csv"
    fields = ["Replicate"] + [f"UC{i}_AssignedTextUC" for i in range(1, 22)]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for i, p in enumerate(schedule, start=1):
            w.writerow([i] + [x + 1 for x in p])
    return path


def dist_summary(values, observed):
    a = np.asarray(values, dtype=np.float64)
    q025, med, q975 = np.quantile(a, [0.025, 0.5, 0.975], method="linear")
    return {
        "observed": float(observed),
        "permutation_median": float(med),
        "central_95_perturbation_range": [float(q025), float(q975)],
        "permutation_min": float(a.min()),
        "permutation_max": float(a.max()),
        "empirical_percentile_le_observed": float(np.mean(a <= observed)),
        "count_ge_observed": int(np.sum(a >= observed)),
        "observed_minus_permutation_median": float(observed - med),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    amendment_text = AMENDMENTS.read_text(encoding="utf-8")
    if "## Amendment 005 — Negative-control permutation reporting rules" not in amendment_text:
        raise SystemExit("Amendment 005 must be registered before Prompt 7 execution")

    rows, ids, declared, cell_idx, cell_map = read_queries()
    cell_counts = np.bincount(cell_idx, minlength=69).astype(np.float64)
    if np.any(cell_counts == 0):
        raise SystemExit("Unexpected empty eligible cell index")
    tfidf = read_score_matrix(TFIDF, ids)
    mpnet = read_score_matrix(MPNET, ids)

    frozen = json.loads(OBSERVED.read_text(encoding="utf-8"))["methods"]
    identity = np.arange(UC_COUNT, dtype=np.int16)
    obs_t_h, obs_t_m = metric_for_mapping(tfidf, identity, declared, cell_idx, cell_counts)
    obs_m_h, obs_m_m = metric_for_mapping(mpnet, identity, declared, cell_idx, cell_counts)
    expected = {
        "TFIDF": (float(frozen["TFIDF"]["Hit1EqualCellMacro"]), float(frozen["TFIDF"]["MRREqualCellMacro"])),
        "MPNET": (float(frozen["MPNET"]["Hit1EqualCellMacro"]), float(frozen["MPNET"]["MRREqualCellMacro"])),
    }
    recomputed = {"TFIDF": (obs_t_h, obs_t_m), "MPNET": (obs_m_h, obs_m_m)}
    for method in expected:
        for got, exp in zip(recomputed[method], expected[method]):
            if abs(got - exp) > 1e-12:
                raise SystemExit(f"Observed {method} metric did not reproduce Prompt 4: {got} vs {exp}")

    schedule = make_schedule()
    schedule_path = write_schedule(schedule)
    dist_rows = []
    t_hit = []; t_mrr = []; m_hit = []; m_mrr = []
    for rep, p in enumerate(schedule, start=1):
        mapping = np.asarray(p, dtype=np.int16)
        th, tm = metric_for_mapping(tfidf, mapping, declared, cell_idx, cell_counts)
        mh, mm = metric_for_mapping(mpnet, mapping, declared, cell_idx, cell_counts)
        t_hit.append(th); t_mrr.append(tm); m_hit.append(mh); m_mrr.append(mm)
        dist_rows.append([rep, th, tm, mh, mm])

    dist_path = OUT / "permutation_distribution.csv"
    with dist_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["Replicate", "TFIDF_Hit1Macro", "TFIDF_MRRMacro", "MPNET_Hit1Macro", "MPNET_MRRMacro"])
        w.writerows(dist_rows)

    summary = {
        "status": "PASS_PENDING_VALIDATOR",
        "design": {
            "permutations": B,
            "seed": SEED,
            "candidate_uc_count": 21,
            "primary_queries": 2369,
            "eligible_model_run_cells": 69,
            "zero_query_cells_excluded_as_NA": 21,
            "schedule_shared_across_methods": True,
            "score_matrices_reused_without_refit_or_rescoring": True,
            "tie_break": "ascending numeric assigned UC label after descending similarity",
            "aggregation": "query -> Model×Run cell -> equal-cell macro across 69 cells",
        },
        "inputs": {
            "queries_sha256": sha256(QUERIES),
            "tfidf_scores_sha256": sha256(TFIDF),
            "mpnet_scores_sha256": sha256(MPNET),
            "prompt4_observed_metrics_sha256": sha256(OBSERVED),
            "permutation_schedule_sha256": sha256(schedule_path),
        },
        "results": {
            "TFIDF": {
                "Hit1": dist_summary(t_hit, obs_t_h),
                "MRR": dist_summary(t_mrr, obs_t_m),
            },
            "MPNET": {
                "Hit1": dist_summary(m_hit, obs_m_h),
                "MRR": dist_summary(m_mrr, obs_m_m),
            },
        },
        "boundaries": {
            "new_semantic_retrieval_run": False,
            "embedding_model_loaded": False,
            "tfidf_refit": False,
            "legacy_semantic_fields_used": False,
            "UC_Action_used": False,
            "expert_or_llm_judge_used": False,
            "p_values_reported": False,
            "semantic_correctness_claimed": False,
        },
    }
    (OUT / "negative_control_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def line(method, metric):
        s = summary["results"][method][metric]
        lo, hi = s["central_95_perturbation_range"]
        return (f"- {method} {metric}: observed **{s['observed']:.6f}**; permutation median **{s['permutation_median']:.6f}**; "
                f"central 95% perturbation range **[{lo:.6f}, {hi:.6f}]**; empirical percentile **{s['empirical_percentile_le_observed']:.6f}**; "
                f"permutations ≥ observed **{s['count_ge_observed']}/{B}**.")

    report = "\n".join([
        "# RQ4 — Frozen Negative-Control Permutation Analysis",
        "",
        "## Scope",
        "The frozen Prompt-4 TF-IDF and MPNet similarity matrices were reused without retraining or rescoring. Exactly 2,000 shared UC-label/text permutations were generated with seed 20260818. Each replicate preserved the 2,369 behavior queries and 69 eligible Model×Run cells; the 21 zero-query cells remained NA.",
        "",
        "## Results",
        line("TFIDF", "Hit1"),
        line("TFIDF", "MRR"),
        line("MPNET", "Hit1"),
        line("MPNET", "MRR"),
        "",
        "## Interpretation boundary",
        "These permutation distributions are negative-control benchmarks for automated retrieval concordance under randomized UC-label/text assignments. They are not p-values, do not validate individual traces, and do not establish semantic correctness or an external ground truth.",
        "",
    ])
    REPORT.write_text(report, encoding="utf-8")

    handoff = "\n".join([
        "# HANDOFF 07 — FROZEN NEGATIVE CONTROL",
        "",
        "## STATUS",
        "",
        "**PASS — pending independent Prompt-7 validator**",
        "",
        "## Scope",
        "",
        "Executed exactly 2,000 shared UC-label/text permutations (seed 20260818) against the frozen Prompt-4 TF-IDF and MPNet score matrices. No semantic model was rerun and no score matrix was recomputed.",
        "",
        "## Headline results",
        line("TFIDF", "Hit1"),
        line("TFIDF", "MRR"),
        line("MPNET", "Hit1"),
        line("MPNET", "MRR"),
        "",
        "## Boundary",
        "",
        "The empirical percentiles and exceedance counts are descriptive negative-control benchmarks, not inferential p-values. Retrieval separation from randomized mappings is evidence of non-random automated semantic association under the frozen instruments, not proof of semantic correctness.",
        "",
        "## Gate",
        "",
        "Acceptance requires independent validation, regression tests, deterministic rerun, and scientific-boundary checks.",
        "",
    ])
    HANDOFF.write_text(handoff, encoding="utf-8")
    print(json.dumps(summary["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
