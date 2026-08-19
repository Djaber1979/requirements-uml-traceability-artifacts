from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "derived/behaviors/primary_semantic_queries.csv"
OUT = ROOT / "artifacts/negative_control"
SEM = ROOT / "artifacts/semantic_retrieval"
REPORT = ROOT / "reports/PROMPT7_VALIDATION.md"
B = 2000
SEED = 20260818


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_scores(path):
    rows = read_csv(path)
    ids = [r["QueryID"] for r in rows]
    scores = [[float(r[f"UC{i}"]) for i in range(1, 22)] for r in rows]
    return ids, scores


def linear_quantile(vals, q):
    x = sorted(vals)
    pos = (len(x) - 1) * q
    lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi:
        return x[lo]
    return x[lo] * (hi - pos) + x[hi] * (pos - lo)


def explicit_macro(scores, queries, mapping):
    by_cell_hit = defaultdict(list)
    by_cell_rr = defaultdict(list)
    for row, s in zip(queries, scores):
        declared = int(row["DeclaredUC"])
        # Assigned label j gets score from original requirement-text column mapping[j-1].
        assigned = [(label, s[mapping[label - 1]]) for label in range(1, 22)]
        assigned.sort(key=lambda z: (-z[1], z[0]))
        rank = next(i for i, (label, _) in enumerate(assigned, start=1) if label == declared)
        cell = (row["Model"], int(row["Run"]))
        by_cell_hit[cell].append(1.0 if rank == 1 else 0.0)
        by_cell_rr[cell].append(1.0 / rank)
    if len(by_cell_hit) != 69:
        raise AssertionError(len(by_cell_hit))
    cell_hit = [sum(v) / len(v) for v in by_cell_hit.values()]
    cell_rr = [sum(v) / len(v) for v in by_cell_rr.values()]
    return sum(cell_hit) / 69, sum(cell_rr) / 69


def main():
    checks = 0
    queries = read_csv(Q)
    assert len(queries) == 2369; checks += 1
    assert not {"UC_Action", "Best_Match_Action", "SimilarityScore"}.intersection(queries[0]); checks += 1

    schedule = read_csv(OUT / "permutation_schedule.csv")
    assert len(schedule) == B; checks += 1
    rng = random.Random(SEED)
    expected_schedule = []
    for rep in range(1, B + 1):
        p = list(range(21)); rng.shuffle(p); expected_schedule.append(p)
        row = schedule[rep - 1]
        got = [int(row[f"UC{i}_AssignedTextUC"]) - 1 for i in range(1, 22)]
        assert got == p
        assert sorted(got) == list(range(21))
        checks += 2

    qids = [r["QueryID"] for r in queries]
    tid, tscores = read_scores(SEM / "tfidf_scores.csv")
    mid, mscores = read_scores(SEM / "mpnet_scores.csv")
    assert tid == qids == mid; checks += 1

    frozen = json.loads((SEM / "corpus_primary_metrics.json").read_text(encoding="utf-8"))["methods"]
    identity = list(range(21))
    th, tm = explicit_macro(tscores, queries, identity)
    mh, mm = explicit_macro(mscores, queries, identity)
    assert abs(th - float(frozen["TFIDF"]["Hit1EqualCellMacro"])) < 1e-12; checks += 1
    assert abs(tm - float(frozen["TFIDF"]["MRREqualCellMacro"])) < 1e-12; checks += 1
    assert abs(mh - float(frozen["MPNET"]["Hit1EqualCellMacro"])) < 1e-12; checks += 1
    assert abs(mm - float(frozen["MPNET"]["MRREqualCellMacro"])) < 1e-12; checks += 1

    dist = read_csv(OUT / "permutation_distribution.csv")
    assert len(dist) == B; checks += 1
    spot = [1, 2, 3, 17, 101, 997, 2000]
    for rep in spot:
        p = expected_schedule[rep - 1]
        th2, tm2 = explicit_macro(tscores, queries, p)
        mh2, mm2 = explicit_macro(mscores, queries, p)
        r = dist[rep - 1]
        for got, key in [(th2, "TFIDF_Hit1Macro"), (tm2, "TFIDF_MRRMacro"), (mh2, "MPNET_Hit1Macro"), (mm2, "MPNET_MRRMacro")]:
            assert abs(got - float(r[key])) < 1e-12, (rep, key, got, r[key])
            checks += 1

    summary = json.loads((OUT / "negative_control_summary.json").read_text(encoding="utf-8"))
    assert summary["design"]["permutations"] == B and summary["design"]["seed"] == SEED; checks += 1
    assert summary["design"]["schedule_shared_across_methods"] is True; checks += 1
    assert summary["design"]["score_matrices_reused_without_refit_or_rescoring"] is True; checks += 1

    mapping = {
        ("TFIDF", "Hit1"): "TFIDF_Hit1Macro",
        ("TFIDF", "MRR"): "TFIDF_MRRMacro",
        ("MPNET", "Hit1"): "MPNET_Hit1Macro",
        ("MPNET", "MRR"): "MPNET_MRRMacro",
    }
    observed = {
        ("TFIDF", "Hit1"): th, ("TFIDF", "MRR"): tm,
        ("MPNET", "Hit1"): mh, ("MPNET", "MRR"): mm,
    }
    for pair, col in mapping.items():
        vals = [float(r[col]) for r in dist]
        s = summary["results"][pair[0]][pair[1]]
        med = linear_quantile(vals, 0.5)
        lo = linear_quantile(vals, 0.025)
        hi = linear_quantile(vals, 0.975)
        pct = sum(v <= observed[pair] for v in vals) / B
        nge = sum(v >= observed[pair] for v in vals)
        assert abs(float(s["observed"]) - observed[pair]) < 1e-12; checks += 1
        assert abs(float(s["permutation_median"]) - med) < 1e-12; checks += 1
        assert abs(float(s["central_95_perturbation_range"][0]) - lo) < 1e-12; checks += 1
        assert abs(float(s["central_95_perturbation_range"][1]) - hi) < 1e-12; checks += 1
        assert abs(float(s["empirical_percentile_le_observed"]) - pct) < 1e-12; checks += 1
        assert int(s["count_ge_observed"]) == nge; checks += 1

    assert all(summary["boundaries"][k] is False for k in [
        "new_semantic_retrieval_run", "embedding_model_loaded", "tfidf_refit",
        "legacy_semantic_fields_used", "UC_Action_used", "expert_or_llm_judge_used",
        "p_values_reported", "semantic_correctness_claimed",
    ]); checks += 1

    text = "\n".join([
        "# Prompt 7 Independent Validation",
        "",
        "## Overall Assessment: PASS",
        "",
        f"Independent checks passed: **{checks}**",
        "",
        "- Regenerated all 2,000 permutations from seed 20260818 and verified the saved shared schedule exactly.",
        "- Recomputed the observed Prompt-4 TF-IDF and MPNet equal-cell macro Hit@1/MRR using an explicit label-sorting implementation.",
        "- Independently recomputed seven dispersed permutation replicates for both methods and both metrics using explicit sorting.",
        "- Independently recomputed medians, central 95% perturbation ranges, empirical percentiles, and exceedance counts from the saved 2,000-replicate distributions.",
        "- Confirmed the 2,369-query/69-cell population and all scientific-boundary flags.",
        "",
        "### Interpretation caveat",
        "The negative control measures separation from randomized UC-label/text mappings under frozen automated retrieval instruments. It does not provide semantic ground truth, individual-trace correctness labels, or a population-level significance test.",
        "",
        "### Gate",
        "Safe to accept Prompt 7 negative-control analysis: **YES**.",
        "",
    ])
    REPORT.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
