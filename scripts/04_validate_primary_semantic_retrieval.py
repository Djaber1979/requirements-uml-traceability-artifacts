#!/usr/bin/env python3
"""Independent numerical and leakage validator for Prompt 4 outputs."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/semantic_retrieval"
REQ = ROOT / "derived/requirements/requirements_primary.csv"
QUERY = ROOT / "derived/behaviors/primary_semantic_queries.csv"
REPORT = ROOT / "reports/PROMPT4_VALIDATION.md"
FORBIDDEN = {"Best_Match_Action", "SimilarityScore", "UC_Action"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_scores(path: Path) -> tuple[list[str], np.ndarray]:
    rows = read_csv(path)
    qids = [r["QueryID"] for r in rows]
    arr = np.asarray([[float(r[f"UC{i}"]) for i in range(1, 22)] for r in rows], dtype=np.float64)
    return qids, arr


def recompute(scores: np.ndarray, declared_uc: int) -> dict:
    order = sorted(range(21), key=lambda i: (-float(scores[i]), i + 1))
    ranking = [i + 1 for i in order]
    rank = ranking.index(declared_uc) + 1
    max_score = max(float(x) for x in scores)
    ties = [i + 1 for i, x in enumerate(scores) if float(x) == max_score]
    return {
        "Top1UC": ranking[0],
        "DeclaredRank": rank,
        "Hit1": int(rank == 1),
        "Hit3": int(rank <= 3),
        "Hit5": int(rank <= 5),
        "ReciprocalRank": 1.0 / rank,
        "TieAwareTop1Credit": (1.0 / len(ties)) if declared_uc in ties else 0.0,
        "Top3UCs": "|".join(map(str, ranking[:3])),
        "Top5UCs": "|".join(map(str, ranking[:5])),
        "FullRanking": "|".join(map(str, ranking)),
    }


def approx(a, b, tol=1e-10):
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    failures: list[str] = []
    checks: list[str] = []

    required = [
        REQ, QUERY,
        ART / "tfidf_scores.csv", ART / "mpnet_scores.csv",
        ART / "tfidf_query_metrics.csv", ART / "mpnet_query_metrics.csv",
        ART / "model_run_primary_metrics.csv", ART / "by_model_primary_metrics.csv",
        ART / "corpus_primary_metrics.json", ART / "runtime_manifest.json",
        ART / "cross_method_query_consensus.csv", ART / "cross_method_cell_consensus.csv",
        ART / "cross_method_corpus_consensus.json",
        ROOT / "reports/HANDOFF_04_PRIMARY_SEMANTIC_RETRIEVAL.md",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing Prompt-4 outputs: {missing}")

    req_rows = read_csv(REQ)
    query_rows = read_csv(QUERY)
    if len(req_rows) != 21 or [int(r["UC_ID"]) for r in req_rows] != list(range(1, 22)):
        failures.append("Requirement materialization is not exactly UC1-UC21")
    else:
        checks.append("Requirement candidate universe is exactly UC1-UC21")
    if sum(int(r["MainScenarioStepCount"]) for r in req_rows) != 98:
        failures.append("Primary requirement step total is not 98")
    else:
        checks.append("Requirement parser reproduces 98 numbered MainScenario steps")

    if len(query_rows) != 2369:
        failures.append(f"Primary query count is {len(query_rows)}, expected 2369")
    else:
        checks.append("Primary query population contains 2369 queries")
    cells = {(r["Model"], int(r["Run"])) for r in query_rows}
    if len(cells) != 69:
        failures.append(f"Primary query cells are {len(cells)}, expected 69")
    else:
        checks.append("Primary query population spans 69 Model×Run cells")

    query_header = set(query_rows[0].keys()) if query_rows else set()
    leaked = FORBIDDEN & query_header
    if leaked:
        failures.append(f"Prohibited fields leaked into query dataset: {sorted(leaked)}")
    else:
        checks.append("No legacy semantic fields or UC_Action in primary query dataset")

    # Deterministic query-text reconstruction spot-check across all rows.
    def segment(text: str) -> str:
        import re
        s = (text or "").strip()
        s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
        s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
        s = re.sub(r"[_\-]+", " ", s)
        s = re.sub(r"[^A-Za-z0-9]+", " ", s)
        return " ".join(s.split())

    for r in query_rows:
        expected = f"Class: {segment(r['ClassCanonical'])}. Method: {segment(r['MethodName'])}. Signature: {segment(r['Signature'])}."
        if r["QueryText"] != expected:
            failures.append(f"Query-text reconstruction mismatch at {r['QueryID']}")
            break
    else:
        checks.append("All query texts reconstruct solely from Class+MethodName+Signature")

    tf_qids, tf_scores = load_scores(ART / "tfidf_scores.csv")
    mp_qids, mp_scores = load_scores(ART / "mpnet_scores.csv")
    expected_qids = [r["QueryID"] for r in query_rows]
    if tf_qids != expected_qids or mp_qids != expected_qids:
        failures.append("Score-matrix QueryID order differs from primary query dataset")
    else:
        checks.append("TF-IDF and MPNet score matrices preserve QueryID order")
    for label, arr in [("TF-IDF", tf_scores), ("MPNet", mp_scores)]:
        if arr.shape != (2369, 21) or not np.all(np.isfinite(arr)):
            failures.append(f"{label} score matrix has invalid shape or non-finite values")
        else:
            checks.append(f"{label} score matrix is finite with shape 2369×21")

    metric_files = {
        "TFIDF": (ART / "tfidf_query_metrics.csv", tf_scores),
        "MPNET": (ART / "mpnet_query_metrics.csv", mp_scores),
    }
    all_metric_rows: dict[str, list[dict[str, str]]] = {}
    declared_by_qid = {r["QueryID"]: int(r["DeclaredUC"]) for r in query_rows}
    for method, (path, scores) in metric_files.items():
        rows = read_csv(path)
        all_metric_rows[method] = rows
        if len(rows) != 2369:
            failures.append(f"{method} query-metric row count mismatch")
            continue
        for i, r in enumerate(rows):
            declared = declared_by_qid[r["QueryID"]]
            rec = recompute(scores[i], declared)
            exact_fields = ["Top1UC", "DeclaredRank", "Hit1", "Hit3", "Hit5", "Top3UCs", "Top5UCs", "FullRanking"]
            if any(str(r[k]) != str(rec[k]) for k in exact_fields):
                failures.append(f"{method} ranking-metric mismatch at {r['QueryID']}")
                break
            if not approx(r["ReciprocalRank"], rec["ReciprocalRank"]) or not approx(r["TieAwareTop1Credit"], rec["TieAwareTop1Credit"]):
                failures.append(f"{method} numeric ranking-metric mismatch at {r['QueryID']}")
                break
        else:
            checks.append(f"{method} rankings and declared-UC metrics independently recompute for all 2369 queries")

    # Independent cell and corpus aggregation from query metrics.
    cell_rows = read_csv(ART / "model_run_primary_metrics.csv")
    corpus = json.loads((ART / "corpus_primary_metrics.json").read_text(encoding="utf-8"))
    for method in ["TFIDF", "MPNET"]:
        rows = all_metric_rows[method]
        by_cell: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            by_cell[(r["Model"], int(r["Run"]))].append(r)
        if len(by_cell) != 69:
            failures.append(f"{method} has {len(by_cell)} evaluable cells, expected 69")
            continue
        recomputed_cells = {}
        for key, group in by_cell.items():
            recomputed_cells[key] = {
                "Hit1Mean": statistics.fmean(float(x["Hit1"]) for x in group),
                "MRRMean": statistics.fmean(float(x["ReciprocalRank"]) for x in group),
                "Hit3Mean": statistics.fmean(float(x["Hit3"]) for x in group),
                "Hit5Mean": statistics.fmean(float(x["Hit5"]) for x in group),
                "TieAwareTop1Mean": statistics.fmean(float(x["TieAwareTop1Credit"]) for x in group),
            }
        published = {(r["Model"], int(r["Run"])): r for r in cell_rows if r["Method"] == method}
        if set(published) != set(recomputed_cells):
            failures.append(f"{method} published cell keys mismatch")
        else:
            for key, vals in recomputed_cells.items():
                if any(not approx(published[key][k], v) for k, v in vals.items()):
                    failures.append(f"{method} cell aggregation mismatch at {key}")
                    break
            else:
                checks.append(f"{method} Model×Run aggregates independently recompute")

        macro_hit1 = statistics.fmean(v["Hit1Mean"] for v in recomputed_cells.values())
        macro_mrr = statistics.fmean(v["MRRMean"] for v in recomputed_cells.values())
        c = corpus["methods"][method]
        if not approx(c["Hit1EqualCellMacro"], macro_hit1) or not approx(c["MRREqualCellMacro"], macro_mrr):
            failures.append(f"{method} equal-cell corpus macro mismatch")
        else:
            checks.append(f"{method} corpus macro uses equal weighting across 69 eligible cells")

    # Cross-method consensus recomputation.
    cm_rows = read_csv(ART / "cross_method_query_consensus.csv")
    if len(cm_rows) != 2369:
        failures.append("Cross-method query consensus row count mismatch")
    else:
        tf_by = {r["QueryID"]: r for r in all_metric_rows["TFIDF"]}
        mp_by = {r["QueryID"]: r for r in all_metric_rows["MPNET"]}
        for i, r in enumerate(cm_rows):
            qid = r["QueryID"]
            t = tf_by[qid]
            m = mp_by[qid]
            top1 = int(int(t["Top1UC"]) == int(m["Top1UC"]))
            t3 = {int(x) for x in t["Top3UCs"].split("|")}
            m3 = {int(x) for x in m["Top3UCs"].split("|")}
            jac = len(t3 & m3) / len(t3 | m3)
            tr = rankdata(-tf_scores[i], method="average")
            mr = rankdata(-mp_scores[i], method="average")
            if float(np.std(tr)) == 0.0 or float(np.std(mr)) == 0.0:
                spear = None
            else:
                spear = float(np.corrcoef(tr, mr)[0, 1])
            if int(r["Top1ExactAgreement"]) != top1 or not approx(r["Top3SetJaccard"], jac):
                failures.append(f"Cross-method agreement mismatch at {qid}")
                break
            observed_spear = None if r["Full21SpearmanTieAware"] == "" else float(r["Full21SpearmanTieAware"])
            if (spear is None) != (observed_spear is None) or (spear is not None and not approx(spear, observed_spear)):
                failures.append(f"Cross-method Spearman mismatch at {qid}")
                break
        else:
            checks.append("Cross-method Top-1, Top-3 Jaccard, and tie-aware rank agreement independently recompute")

    runtime = json.loads((ART / "runtime_manifest.json").read_text(encoding="utf-8"))
    neural = runtime["primary_neural"]
    requested = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
    if neural["requested_revision"] != requested or neural["resolved_revision"] != requested:
        failures.append("MPNet requested/resolved revision differs from frozen revision")
    else:
        checks.append("MPNet resolved exactly to the frozen immutable revision")
    scope = runtime["execution_scope"]
    if not (scope["primary_tfidf_executed"] and scope["primary_mpnet_executed"] and scope["cross_method_primary_consensus_executed"]):
        failures.append("Required Prompt-4 primary execution flags are not all true")
    if scope["minilm_sensitivity_executed"] or scope["negative_control_executed"] or scope["run_resampling_executed"] or scope["rq3_fragmentation_executed"]:
        failures.append("Prompt 4 executed analyses reserved for later prompts")
    else:
        checks.append("Prompt 4 did not execute frozen later-stage sensitivity/robustness analyses")

    trunc_req = int(neural["requirement_token_profile"]["truncated_if_native_limit_applied"])
    trunc_query = int(neural["query_token_profile"]["truncated_if_native_limit_applied"])
    checks.append(f"MPNet truncation audit recorded: requirements={trunc_req}/21, queries={trunc_query}/2369")

    status = "PASS" if not failures else "FAIL"
    overall = "Ready for next research gate" if not failures else "Needs revision"
    report = [
        "# Prompt 4 Validation Report",
        "",
        f"## Overall Assessment: {overall}",
        "",
        f"**Status: {status}**",
        "",
        "### Methodology and Calculation Checks",
    ]
    report.extend(f"- VERIFIED: {x}" for x in checks)
    report.extend(["", "### Issues Found"])
    if failures:
        report.extend(f"- BLOCKER: {x}" for x in failures)
    else:
        report.append("- No material numerical, aggregation, population, revision, or leakage discrepancy found.")
    report.extend([
        "",
        "### Required Caveats",
        "- Retrieval concordance is not semantic correctness or an external gold standard.",
        "- The primary semantic population exists in 69/90 Model×Run cells; 21 zero-query cells remain structural NA and are not imputed as failures.",
        "- Generator-model comparisons must report eligible-run coverage because semantic coverage is uneven across models.",
        "- This validator checks internal reproducibility and protocol compliance; it does not provide human semantic validation.",
        "",
        "### Gate",
        f"Safe to accept Prompt 4 primary semantic retrieval: **{'YES' if not failures else 'NO'}**.",
        "",
    ])
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": status, "checks": len(checks), "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
