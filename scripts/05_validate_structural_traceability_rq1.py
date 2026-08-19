from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "derived/behaviors/source_without_legacy_semantics.csv"
SCAFFOLD = ROOT / "input_snapshot/provenance/Methodless.txt"
QUERIES = ROOT / "derived/behaviors/primary_semantic_queries.csv"
OUT = ROOT / "artifacts/structural_rq1"
REPORT = ROOT / "reports/PROMPT5_VALIDATION.md"
PRODUCTION = ROOT / "scripts/05_structural_traceability_rq1.py"

MODELS = ["ChatGPT-4o", "ChatGPT-o3", "Claude3.7", "DeepSeek(R1)", "Gemini-2.5-Pro-Preview-05-06", "Grok3", "Llama4", "Mistral", "Qwen3"]
VALID_UCS = set(range(1, 22))


def load_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_bool(v):
    x = (v or "").strip().lower()
    return True if x == "true" else False if x == "false" else None


def parse_uc(v):
    s = (v or "").strip()
    if re.fullmatch(r"\d+", s): return int(s)
    m = re.fullmatch(r"(?i:UC)\s*(\d+)", s)
    return int(m.group(1)) if m else None


def scaffold_ids(text):
    out = set()
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r'^(?:abstract\s+)?class\s+"([^"]+)"\s+as\s+([A-Za-z_]\w*)', line)
        if m:
            out.update([m.group(1), m.group(2)]); continue
        m = re.match(r"^(?:abstract\s+)?class\s+([A-Za-z_]\w*)", line)
        if m: out.add(m.group(1))
    return out


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def main():
    rows = load_csv(CLEAN)
    queries = load_csv(QUERIES)
    summary = json.loads((OUT / "corpus_structural_summary.json").read_text(encoding="utf-8"))
    cell_art = {(r["Model"], int(r["Run"])): r for r in load_csv(OUT / "model_run_structural_metrics.csv")}
    uc_art = {int(r["UC"].replace("UC", "")): r for r in load_csv(OUT / "uc_structural_coverage.csv")}
    scaffold = scaffold_ids(SCAFFOLD.read_text(encoding="utf-8"))

    checks = []
    def check(name, cond):
        if not cond: raise AssertionError(name)
        checks.append(name)

    check("row_count_3373", len(rows) == 3373)
    check("query_count_2369", len(queries) == 2369)
    check("scaffold_identity_count_21", len(scaffold) == 21)
    check("legacy_semantic_fields_absent", all(x not in rows[0] for x in ["Best_Match_Action", "SimilarityScore"]))

    full_counter = Counter(tuple(r[c] for c in rows[0].keys()) for r in rows)
    dup_excess = sum(max(0, n - 1) for n in full_counter.values())
    dup_by_cell = Counter()
    for key, n in full_counter.items():
        if n > 1:
            d = dict(zip(rows[0].keys(), key)); dup_by_cell[(d["Model"], int(d["Run"]))] += n - 1

    annotated = valid = invalid = action = complete = ref_only = off = 0
    invalid_values = Counter(); off_classes = Counter()
    by_cell = defaultdict(list); by_uc = defaultdict(list)
    for r in rows:
        model = r["Model"].strip(); run = int(r["Run"])
        ann = parse_bool(r["Has_UC_Annotation"])
        uc = parse_uc(r["UC_References"])
        valid_ref = uc in VALID_UCS if uc is not None else False
        invalid_ref = bool((r["UC_References"] or "").strip()) and not valid_ref
        action_present = bool((r["UC_Action"] or "").strip())
        on_scaffold = r["Class"].strip() in scaffold
        state = (ann is True, valid_ref, invalid_ref, action_present, on_scaffold, uc if valid_ref else None)
        by_cell[(model, run)].append(state)
        if valid_ref: by_uc[uc].append((model, run, ann is True and action_present, action_present))
        annotated += ann is True; valid += valid_ref; invalid += invalid_ref; action += action_present
        complete += ann is True and valid_ref and action_present
        ref_only += ann is True and valid_ref and not action_present
        off += not on_scaffold
        if invalid_ref: invalid_values[(r["UC_References"] or "").strip()] += 1
        if not on_scaffold: off_classes[r["Class"].strip()] += 1

    check("summary_all_rows", summary["population"]["all_rows"] == len(rows))
    check("summary_annotated", summary["population"]["annotated_rows"] == annotated)
    check("summary_valid", summary["population"]["valid_reference_rows"] == valid)
    check("summary_invalid", summary["population"]["invalid_reference_rows"] == invalid)
    check("summary_complete", summary["population"]["complete_explicit_trace_rows"] == complete)
    check("summary_reference_only", summary["population"]["reference_only_trace_rows"] == ref_only)
    check("summary_off_scaffold", summary["population"]["off_scaffold_rows"] == off)
    check("summary_duplicates", summary["population"]["exact_duplicate_excess_rows"] == dup_excess)
    check("invalid_reference_values", summary["invalid_reference_raw_values"] == dict(sorted(invalid_values.items())))
    check("off_scaffold_classes", summary["off_scaffold_classes"] == dict(sorted(off_classes.items())))

    query_counts = Counter((q["Model"], int(q["Run"])) for q in queries)
    zero_ann = zero_valid = zero_complete = zero_primary = 0
    for model in MODELS:
        for run in range(1, 11):
            states = by_cell[(model, run)]
            n = len(states); ann_n = sum(s[0] for s in states); valid_n = sum(s[1] for s in states)
            complete_n = sum(s[0] and s[1] and s[3] for s in states); invalid_n = sum(s[2] for s in states)
            off_n = sum(not s[4] for s in states); action_ann = sum(s[0] and s[3] for s in states)
            distinct_ucs = len({s[5] for s in states if s[1]})
            art = cell_art[(model, run)]
            check(f"cell_rows_{model}_{run}", int(art["Rows"]) == n)
            check(f"cell_annotation_{model}_{run}", int(art["AnnotatedRows"]) == ann_n and close(art["TraceAnnotationRate"], ann_n/n))
            check(f"cell_valid_{model}_{run}", int(art["ValidReferenceRows"]) == valid_n and close(art["ValidReferenceRate"], valid_n/n))
            check(f"cell_complete_{model}_{run}", int(art["CompleteTraceRows"]) == complete_n and close(art["CompleteExplicitTraceRate"], complete_n/n))
            check(f"cell_invalid_{model}_{run}", int(art["InvalidReferenceRows"]) == invalid_n)
            check(f"cell_off_{model}_{run}", int(art["OffScaffoldRows"]) == off_n)
            check(f"cell_dup_{model}_{run}", int(art["ExactDuplicateExcessRows"]) == dup_by_cell[(model, run)])
            check(f"cell_uc_coverage_{model}_{run}", int(art["DistinctValidUCs"]) == distinct_ucs and close(art["UCCoverage"], distinct_ucs/21))
            check(f"cell_primary_queries_{model}_{run}", int(art["PrimarySemanticQueries"]) == query_counts[(model, run)])
            if ann_n:
                check(f"cell_action_conditional_{model}_{run}", close(art["ActionTextCompletenessAmongAnnotated"], action_ann/ann_n))
            else:
                check(f"cell_action_conditional_na_{model}_{run}", art["ActionTextCompletenessAmongAnnotated"] == "")
            zero_ann += ann_n == 0; zero_valid += valid_n == 0; zero_complete += complete_n == 0; zero_primary += query_counts[(model, run)] == 0

    check("zero_annotation_cells", summary["cell_coverage"]["zero_annotation_cells"] == zero_ann)
    check("zero_valid_cells", summary["cell_coverage"]["zero_valid_reference_cells"] == zero_valid)
    check("zero_complete_cells", summary["cell_coverage"]["zero_complete_trace_cells"] == zero_complete)
    check("zero_primary_cells", summary["cell_coverage"]["zero_primary_semantic_query_cells"] == zero_primary == 21)
    check("primary_cells_69", summary["cell_coverage"]["primary_semantic_query_cells"] == 69)

    for uc in range(1, 22):
        states = by_uc[uc]; art = uc_art[uc]
        cells = {(x[0], x[1]) for x in states}; models = {x[0] for x in states}
        complete_uc = sum(x[2] for x in states); action_uc = sum(x[3] for x in states)
        check(f"uc_rows_{uc}", int(art["ValidReferenceRows"]) == len(states))
        check(f"uc_cells_{uc}", int(art["CellsWithReference"]) == len(cells) and close(art["CellCoverage"], len(cells)/90))
        check(f"uc_models_{uc}", int(art["ModelsWithReference"]) == len(models) and close(art["ModelCoverage"], len(models)/9))
        check(f"uc_complete_{uc}", int(art["CompleteExplicitTraceRows"]) == complete_uc)
        check(f"uc_action_{uc}", int(art["ActionTextPresentAmongValidRows"]) == action_uc)

    # Ensure production did not consume semantic retrieval outcomes.
    src = PRODUCTION.read_text(encoding="utf-8")
    check("no_tfidf_outcome_access", "tfidf_query_metrics" not in src and "tfidf_scores" not in src)
    check("no_mpnet_outcome_access", "mpnet_query_metrics" not in src and "mpnet_scores" not in src)
    check("no_inferential_test_code", all(token not in src for token in ["ttest", "mannwhitney", "anova", "pvalue", "p_value"]))

    report = [
        "# Prompt 5 Independent Validation", "", "## Overall Assessment: PASS", "",
        f"Independent checks passed: **{len(checks)}**", "",
        "Verified independently from the clean source and frozen Prompt-4 query identity file:",
        "- all 3,373 structural rows;",
        "- annotation, valid/invalid reference, action-presence, complete-trace, off-scaffold, and duplicate counts;",
        "- all 90 Model × Run structural metric rows;",
        "- all UC1–UC21 structural coverage rows;",
        "- the 69/90 primary-query cell reconciliation and 2,369-query identity population;",
        "- absence of TF-IDF/MPNet outcome consumption in the RQ1 production script;",
        "- descriptive-only/no-inferential-test boundary.", "",
        "### Interpretation caveat", "Structural completeness is metadata completeness, not semantic correctness or generator quality.", "",
        "### Gate", "Safe to accept Prompt 5 RQ1 structural analysis: **YES**.", "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": len(checks), "annotated": annotated, "valid": valid, "complete": complete, "zero_primary_cells": zero_primary}, sort_keys=True))


if __name__ == "__main__":
    main()
