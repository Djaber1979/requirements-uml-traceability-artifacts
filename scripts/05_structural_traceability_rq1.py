from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "derived/behaviors/source_without_legacy_semantics.csv"
SCAFFOLD = ROOT / "input_snapshot/provenance/Methodless.txt"
PROMPT4_QUERIES = ROOT / "derived/behaviors/primary_semantic_queries.csv"
PROMPT2_SUMMARY = ROOT / "artifacts/structural_audit/structural_audit_summary.json"
PROMPT3_POP = ROOT / "artifacts/protocol_freeze/primary_population_audit.json"
OUT = ROOT / "artifacts/structural_rq1"
REPORT = ROOT / "reports/RQ1_STRUCTURAL_TRACEABILITY.md"
HANDOFF = ROOT / "reports/HANDOFF_05_STRUCTURAL_TRACEABILITY.md"
AMENDMENTS = ROOT / "PROTOCOL_AMENDMENTS.md"

EXPECTED_MODELS = [
    "ChatGPT-4o", "ChatGPT-o3", "Claude3.7", "DeepSeek(R1)",
    "Gemini-2.5-Pro-Preview-05-06", "Grok3", "Llama4", "Mistral", "Qwen3",
]
UC_IDS = list(range(1, 22))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def parse_bool(value: str):
    v = (value or "").strip().casefold()
    if v == "true":
        return True
    if v == "false":
        return False
    return None


def parse_single_uc(value: str):
    raw = (value or "").strip()
    if not raw:
        return None, "blank"
    if re.fullmatch(r"\d+", raw):
        return int(raw), "single_integer"
    m = re.fullmatch(r"(?i:UC)\s*(\d+)", raw)
    if m:
        return int(m.group(1)), "single_prefixed"
    return None, "malformed_or_multi"


def scaffold_identities(text: str):
    ids = set()
    aliases = {}
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r'^(?:abstract\s+)?class\s+"([^"]+)"\s+as\s+([A-Za-z_]\w*)', line)
        if m:
            display, alias = m.group(1), m.group(2)
            ids.add(display); ids.add(alias); aliases[display] = alias
            continue
        m = re.match(r"^(?:abstract\s+)?class\s+([A-Za-z_]\w*)", line)
        if m:
            ids.add(m.group(1))
    return ids, aliases


def normalize_signature(value: str):
    text = (value or "").strip().casefold()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([(),:+\-])\s*", r"\1", text)
    return text


def quantile(values, q):
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def dist_summary(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {k: None for k in ["mean", "median", "q1", "q3", "min", "max"]}
    return {
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "q1": quantile(vals, 0.25),
        "q3": quantile(vals, 0.75),
        "min": min(vals),
        "max": max(vals),
    }


def fmt(x):
    if x is None:
        return "NA"
    return f"{float(x):.6f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_csv(CLEAN)
    if len(rows) != 3373:
        raise SystemExit(f"Expected 3373 clean rows, got {len(rows)}")
    if any(c in rows[0] for c in ["Best_Match_Action", "SimilarityScore"]):
        raise SystemExit("Legacy semantic fields present in clean source")
    if "## Amendment 004 — RQ1 structural-traceability operationalization" not in AMENDMENTS.read_text(encoding="utf-8"):
        raise SystemExit("Amendment 004 not frozen before Prompt 5")

    scaffold, alias_map = scaffold_identities(SCAFFOLD.read_text(encoding="utf-8"))
    if len(scaffold) != 21 or alias_map != {"RoleManager <<service>>": "RoleManager"}:
        raise SystemExit(f"Unexpected scaffold identity set: {len(scaffold)} / {alias_map}")

    # Exact duplicate excess rows at the full clean-row grain.
    fields = list(rows[0].keys())
    exact_counter = Counter(tuple(r[c] for c in fields) for r in rows)
    duplicate_excess_by_cell = Counter()
    duplicate_groups = 0
    for key, count in exact_counter.items():
        if count > 1:
            duplicate_groups += 1
            as_dict = dict(zip(fields, key))
            duplicate_excess_by_cell[(as_dict["Model"], int(as_dict["Run"]))] += count - 1
    exact_duplicate_excess = sum(max(0, c - 1) for c in exact_counter.values())

    row_states = []
    cell_rows = defaultdict(list)
    uc_rows = defaultdict(list)
    off_scaffold = Counter()
    invalid_ref_raw = Counter()
    consistency = Counter()

    for i, r in enumerate(rows, start=1):
        model = r["Model"].strip(); run = int(r["Run"])
        ann = parse_bool(r["Has_UC_Annotation"])
        uc, ref_kind = parse_single_uc(r["UC_References"])
        valid_ref = uc in UC_IDS if uc is not None else False
        invalid_ref = bool((r["UC_References"] or "").strip()) and not valid_ref
        action_present = bool((r["UC_Action"] or "").strip())
        on_scaffold = r["Class"].strip() in scaffold
        core_complete = all((r[c] or "").strip() for c in ["Class", "MethodName", "Signature"])
        complete_trace = ann is True and valid_ref and action_present
        reference_only = ann is True and valid_ref and not action_present

        if ann is True and not (r["UC_References"] or "").strip(): consistency["annotated_true_blank_reference"] += 1
        if ann is True and not action_present: consistency["annotated_true_blank_action"] += 1
        if ann is False and (r["UC_References"] or "").strip(): consistency["annotated_false_nonblank_reference"] += 1
        if ann is False and action_present: consistency["annotated_false_nonblank_action"] += 1
        if ann is True and invalid_ref: consistency["annotated_true_invalid_reference"] += 1
        if ann is None: consistency["unexpected_annotation_boolean"] += 1

        if invalid_ref:
            invalid_ref_raw[(r["UC_References"] or "").strip()] += 1
        if not on_scaffold:
            off_scaffold[r["Class"].strip()] += 1

        state = {
            "source_row": i,
            "model": model,
            "run": run,
            "annotation_present": ann is True,
            "annotation_absent": ann is False,
            "annotation_other": ann is None,
            "valid_uc_reference": valid_ref,
            "invalid_uc_reference": invalid_ref,
            "declared_uc": uc if valid_ref else None,
            "action_text_present": action_present,
            "complete_explicit_trace": complete_trace,
            "reference_only_trace": reference_only,
            "on_scaffold": on_scaffold,
            "core_complete": core_complete,
        }
        row_states.append(state)
        cell_rows[(model, run)].append(state)
        if valid_ref:
            uc_rows[uc].append(state)

    # Primary semantic funnel independently reconstructed from frozen rules.
    primary_by_key = {}
    primary_ref_by_key = defaultdict(set)
    for r, st in zip(rows, row_states):
        if not (st["annotation_present"] and st["valid_uc_reference"] and st["core_complete"] and st["on_scaffold"]):
            continue
        canonical_class = alias_map.get(r["Class"].strip(), r["Class"].strip())
        key = (r["Model"].strip(), int(r["Run"]), canonical_class, r["MethodName"].strip().casefold(), normalize_signature(r["Signature"]))
        primary_ref_by_key[key].add(int(st["declared_uc"]))
        primary_by_key.setdefault(key, st)
    conflicts = {k: v for k, v in primary_ref_by_key.items() if len(v) > 1}
    if conflicts:
        raise SystemExit(f"Primary duplicate conflicts unexpectedly present: {len(conflicts)}")

    prompt4_queries = read_csv(PROMPT4_QUERIES)
    if len(prompt4_queries) != len(primary_by_key):
        raise SystemExit(f"Prompt4 query count mismatch: {len(prompt4_queries)} vs {len(primary_by_key)}")
    prompt4_keys = {
        (q["Model"].strip(), int(q["Run"]), q["ClassCanonical"].strip(), q["MethodName"].strip().casefold(), normalize_signature(q["Signature"]))
        for q in prompt4_queries
    }
    if prompt4_keys != set(primary_by_key):
        raise SystemExit("Prompt4 query identity set does not exactly match independently reconstructed primary set")

    annotated_count = sum(s["annotation_present"] for s in row_states)
    valid_count = sum(s["valid_uc_reference"] for s in row_states)
    invalid_count = sum(s["invalid_uc_reference"] for s in row_states)
    action_count = sum(s["action_text_present"] for s in row_states)
    complete_count = sum(s["complete_explicit_trace"] for s in row_states)
    reference_only_count = sum(s["reference_only_trace"] for s in row_states)
    on_scaffold_valid_annotated_core = sum(
        s["annotation_present"] and s["valid_uc_reference"] and s["core_complete"] and s["on_scaffold"] for s in row_states
    )
    off_count = sum(not s["on_scaffold"] for s in row_states)

    # Cell-level metrics.
    cell_out = []
    for model in EXPECTED_MODELS:
        for run in range(1, 11):
            states = cell_rows[(model, run)]
            n = len(states)
            ann_n = sum(s["annotation_present"] for s in states)
            valid_n = sum(s["valid_uc_reference"] for s in states)
            complete_n = sum(s["complete_explicit_trace"] for s in states)
            invalid_n = sum(s["invalid_uc_reference"] for s in states)
            off_n = sum(not s["on_scaffold"] for s in states)
            action_ann_n = sum(s["annotation_present"] and s["action_text_present"] for s in states)
            distinct_ucs = len({s["declared_uc"] for s in states if s["valid_uc_reference"]})
            primary_n = sum(1 for k in primary_by_key if k[0] == model and k[1] == run)
            dup_n = duplicate_excess_by_cell[(model, run)]
            cell_out.append({
                "Model": model, "Run": run, "Rows": n,
                "AnnotatedRows": ann_n, "TraceAnnotationRate": ann_n / n,
                "ValidReferenceRows": valid_n, "ValidReferenceRate": valid_n / n,
                "CompleteTraceRows": complete_n, "CompleteExplicitTraceRate": complete_n / n,
                "ActionPresentAmongAnnotatedRows": action_ann_n,
                "ActionTextCompletenessAmongAnnotated": (action_ann_n / ann_n) if ann_n else "",
                "DistinctValidUCs": distinct_ucs, "UCCoverage": distinct_ucs / 21,
                "InvalidReferenceRows": invalid_n, "InvalidReferenceRate": invalid_n / n,
                "OffScaffoldRows": off_n, "OffScaffoldRate": off_n / n,
                "ExactDuplicateExcessRows": dup_n, "ExactDuplicateRate": dup_n / n,
                "PrimarySemanticQueries": primary_n,
                "ZeroAnnotation": int(ann_n == 0), "ZeroValidReference": int(valid_n == 0),
                "ZeroCompleteTrace": int(complete_n == 0), "ZeroPrimarySemanticQuery": int(primary_n == 0),
            })

    write_csv(OUT / "model_run_structural_metrics.csv", list(cell_out[0].keys()), cell_out)

    # Cell distribution summary.
    dist_metrics = [
        "TraceAnnotationRate", "ValidReferenceRate", "CompleteExplicitTraceRate",
        "ActionTextCompletenessAmongAnnotated", "UCCoverage", "InvalidReferenceRate",
        "OffScaffoldRate", "ExactDuplicateRate",
    ]
    dist_rows = []
    for metric in dist_metrics:
        values = []
        for r in cell_out:
            v = r[metric]
            if v != "": values.append(float(v))
        d = dist_summary(values)
        dist_rows.append({"Metric": metric, "NCells": len(values), **d})
    write_csv(OUT / "cell_distribution_summary.csv", list(dist_rows[0].keys()), dist_rows)

    # Generator summaries in fixed repository order, no performance sorting.
    model_out = []
    for model in EXPECTED_MODELS:
        rs = [r for r in cell_out if r["Model"] == model]
        def mean_metric(name):
            vals = [float(r[name]) for r in rs if r[name] != ""]
            return sum(vals) / len(vals) if vals else None
        def median_metric(name):
            vals = [float(r[name]) for r in rs if r[name] != ""]
            return statistics.median(vals) if vals else None
        model_out.append({
            "Model": model, "Runs": 10, "Rows": sum(int(r["Rows"]) for r in rs),
            "MeanTraceAnnotationRate": mean_metric("TraceAnnotationRate"),
            "MedianTraceAnnotationRate": median_metric("TraceAnnotationRate"),
            "MeanValidReferenceRate": mean_metric("ValidReferenceRate"),
            "MedianValidReferenceRate": median_metric("ValidReferenceRate"),
            "MeanCompleteExplicitTraceRate": mean_metric("CompleteExplicitTraceRate"),
            "MedianCompleteExplicitTraceRate": median_metric("CompleteExplicitTraceRate"),
            "MeanActionTextCompletenessAmongAnnotated": mean_metric("ActionTextCompletenessAmongAnnotated"),
            "MedianActionTextCompletenessAmongAnnotated": median_metric("ActionTextCompletenessAmongAnnotated"),
            "MeanUCCoverage": mean_metric("UCCoverage"), "MedianUCCoverage": median_metric("UCCoverage"),
            "InvalidReferenceRows": sum(int(r["InvalidReferenceRows"]) for r in rs),
            "OffScaffoldRows": sum(int(r["OffScaffoldRows"]) for r in rs),
            "ExactDuplicateExcessRows": sum(int(r["ExactDuplicateExcessRows"]) for r in rs),
            "ZeroAnnotationRuns": sum(int(r["ZeroAnnotation"]) for r in rs),
            "ZeroValidReferenceRuns": sum(int(r["ZeroValidReference"]) for r in rs),
            "ZeroCompleteTraceRuns": sum(int(r["ZeroCompleteTrace"]) for r in rs),
            "ZeroPrimarySemanticQueryRuns": sum(int(r["ZeroPrimarySemanticQuery"]) for r in rs),
            "PrimarySemanticQueries": sum(int(r["PrimarySemanticQueries"]) for r in rs),
        })
    write_csv(OUT / "model_structural_summary.csv", list(model_out[0].keys()), model_out)

    # UC structural coverage.
    uc_out = []
    for uc in UC_IDS:
        states = uc_rows[uc]
        cell_set = {(s["model"], s["run"]) for s in states}
        model_set = {s["model"] for s in states}
        complete = sum(s["complete_explicit_trace"] for s in states)
        action_present_valid = sum(s["action_text_present"] for s in states)
        uc_out.append({
            "UC": f"UC{uc}", "ValidReferenceRows": len(states),
            "CellsWithReference": len(cell_set), "CellCoverage": len(cell_set) / 90,
            "ModelsWithReference": len(model_set), "ModelCoverage": len(model_set) / 9,
            "CompleteExplicitTraceRows": complete,
            "ActionTextPresentAmongValidRows": action_present_valid,
            "ActionTextCompletenessAmongValidRows": (action_present_valid / len(states)) if states else "",
        })
    write_csv(OUT / "uc_structural_coverage.csv", list(uc_out[0].keys()), uc_out)

    # Funnel.
    funnel = [
        {"Stage": "All clean source rows", "Count": len(rows), "ShareOfAll": 1.0},
        {"Stage": "Has_UC_Annotation=True", "Count": annotated_count, "ShareOfAll": annotated_count / len(rows)},
        {"Stage": "Valid deterministic UC1-UC21 reference", "Count": valid_count, "ShareOfAll": valid_count / len(rows)},
        {"Stage": "Annotated + valid UC + core fields + on-scaffold", "Count": on_scaffold_valid_annotated_core, "ShareOfAll": on_scaffold_valid_annotated_core / len(rows)},
        {"Stage": "Unique primary semantic queries after within-run duplicate collapse", "Count": len(primary_by_key), "ShareOfAll": len(primary_by_key) / len(rows)},
    ]
    write_csv(OUT / "structural_funnel.csv", ["Stage", "Count", "ShareOfAll"], funnel)

    # Row-state counts.
    state_counts = [
        ("all_rows", len(rows)), ("annotation_present", annotated_count),
        ("valid_uc_reference", valid_count), ("invalid_uc_reference", invalid_count),
        ("action_text_present", action_count), ("complete_explicit_trace", complete_count),
        ("reference_only_trace", reference_only_count), ("off_scaffold", off_count),
        ("exact_duplicate_excess_rows", exact_duplicate_excess),
        ("primary_semantic_queries", len(primary_by_key)),
    ]
    write_csv(OUT / "row_state_counts.csv", ["State", "Count", "RateOfAllRows"], [
        {"State": name, "Count": count, "RateOfAllRows": count / len(rows)} for name, count in state_counts
    ])

    write_csv(OUT / "annotation_consistency_profile.csv", ["Issue", "Count", "RateOfAllRows"], [
        {"Issue": k, "Count": consistency.get(k, 0), "RateOfAllRows": consistency.get(k, 0) / len(rows)}
        for k in ["annotated_true_blank_reference", "annotated_true_blank_action", "annotated_false_nonblank_reference", "annotated_false_nonblank_action", "annotated_true_invalid_reference", "unexpected_annotation_boolean"]
    ])
    write_csv(OUT / "off_scaffold_profile.csv", ["Class", "Rows", "RateOfAllRows"], [
        {"Class": k, "Rows": v, "RateOfAllRows": v / len(rows)} for k, v in sorted(off_scaffold.items())
    ])
    write_csv(OUT / "invalid_reference_profile.csv", ["RawReference", "Rows", "RateOfAllRows"], [
        {"RawReference": k, "Rows": v, "RateOfAllRows": v / len(rows)} for k, v in sorted(invalid_ref_raw.items())
    ])
    write_csv(OUT / "duplicate_profile.csv", ["Metric", "Count", "RateOfAllRows"], [
        {"Metric": "exact_duplicate_groups", "Count": duplicate_groups, "RateOfAllRows": duplicate_groups / len(rows)},
        {"Metric": "exact_duplicate_excess_rows", "Count": exact_duplicate_excess, "RateOfAllRows": exact_duplicate_excess / len(rows)},
        {"Metric": "primary_duplicate_reference_conflicts", "Count": len(conflicts), "RateOfAllRows": 0.0},
    ])

    zero_annotation_cells = sum(int(r["ZeroAnnotation"]) for r in cell_out)
    zero_valid_cells = sum(int(r["ZeroValidReference"]) for r in cell_out)
    zero_complete_cells = sum(int(r["ZeroCompleteTrace"]) for r in cell_out)
    zero_primary_cells = sum(int(r["ZeroPrimarySemanticQuery"]) for r in cell_out)

    summary = {
        "status": "PASS",
        "population": {
            "all_rows": len(rows), "annotated_rows": annotated_count,
            "valid_reference_rows": valid_count, "invalid_reference_rows": invalid_count,
            "action_text_present_rows": action_count, "complete_explicit_trace_rows": complete_count,
            "reference_only_trace_rows": reference_only_count, "off_scaffold_rows": off_count,
            "exact_duplicate_excess_rows": exact_duplicate_excess,
            "primary_semantic_queries": len(primary_by_key),
        },
        "rates_of_all_rows": {
            "trace_annotation_rate": annotated_count / len(rows),
            "valid_reference_rate": valid_count / len(rows),
            "complete_explicit_trace_rate": complete_count / len(rows),
            "invalid_reference_rate": invalid_count / len(rows),
            "off_scaffold_rate": off_count / len(rows),
            "exact_duplicate_excess_rate": exact_duplicate_excess / len(rows),
        },
        "conditional_rates": {
            "valid_reference_among_annotated": valid_count / annotated_count if annotated_count else None,
            "action_text_completeness_among_annotated": sum(s["annotation_present"] and s["action_text_present"] for s in row_states) / annotated_count if annotated_count else None,
            "complete_trace_among_annotated": complete_count / annotated_count if annotated_count else None,
        },
        "cell_coverage": {
            "cells": 90, "zero_annotation_cells": zero_annotation_cells,
            "zero_valid_reference_cells": zero_valid_cells, "zero_complete_trace_cells": zero_complete_cells,
            "zero_primary_semantic_query_cells": zero_primary_cells,
            "primary_semantic_query_cells": 90 - zero_primary_cells,
        },
        "funnel": funnel,
        "invalid_reference_raw_values": dict(sorted(invalid_ref_raw.items())),
        "off_scaffold_classes": dict(sorted(off_scaffold.items())),
        "duplicate_groups": duplicate_groups,
        "primary_duplicate_reference_conflicts": len(conflicts),
        "cross_checks": {
            "prompt4_query_identity_exact": True,
            "prompt2_expected_rows": json.loads(PROMPT2_SUMMARY.read_text(encoding="utf-8"))["input"]["row_count"] == len(rows),
            "prompt3_primary_query_count": json.loads(PROMPT3_POP.read_text(encoding="utf-8"))["funnel"]["primary_queries_after_within_run_dedup"] == len(primary_by_key),
        },
        "interpretation_boundary": "Structural trace metadata completeness/coverage only; not semantic correctness or model quality.",
    }
    (OUT / "corpus_structural_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Publication-oriented report.
    dmap = {r["Metric"]: r for r in dist_rows}
    report = f"""# RQ1 Structural Traceability

## Status

**PASS — descriptive structural analysis only**

## Corpus-level structural findings

The RQ1 population contains all **{len(rows):,}** clean source rows across 9 generators × 10 runs. Explicit UC trace annotations occur in **{annotated_count:,}** rows ({annotated_count/len(rows):.2%}). A valid deterministic UC1–UC21 reference occurs in **{valid_count:,}** rows ({valid_count/len(rows):.2%} of all rows; {valid_count/annotated_count:.2%} of annotated rows). **{invalid_count}** rows carry a nonblank invalid reference; the observed invalid values are `{dict(sorted(invalid_ref_raw.items()))}`.

A structurally complete explicit trace — annotation present, valid UC1–UC21 reference, and nonblank `UC_Action` metadata — occurs in **{complete_count:,}** rows ({complete_count/len(rows):.2%} of all rows; {complete_count/annotated_count:.2%} of annotated rows). **{reference_only_count:,}** rows have a valid declared UC reference but blank action metadata. This is metadata completeness, not semantic correctness.

## Model × Run coverage

Across all 90 Model × Run cells, **{zero_annotation_cells}** contain no annotated rows, **{zero_valid_cells}** contain no valid UC1–UC21 reference, **{zero_complete_cells}** contain no structurally complete explicit trace, and **{zero_primary_cells}** contain no primary semantic query. The last value exactly reproduces the Prompt-3/Prompt-4 semantic coverage of **{90-zero_primary_cells}/90** eligible cells.

Equal-cell descriptive means are: trace-annotation rate **{fmt(dmap['TraceAnnotationRate']['mean'])}**, valid-reference rate **{fmt(dmap['ValidReferenceRate']['mean'])}**, complete-explicit-trace rate **{fmt(dmap['CompleteExplicitTraceRate']['mean'])}**, and UC coverage **{fmt(dmap['UCCoverage']['mean'])}**. These equal-cell summaries are descriptive and are not generator-quality estimates.

## Structural funnel

The frozen funnel is:

1. all clean rows: **{len(rows):,}**;
2. `Has_UC_Annotation=True`: **{annotated_count:,}**;
3. valid deterministic UC1–UC21 reference: **{valid_count:,}**;
4. annotated + valid reference + complete core fields + on-scaffold: **{on_scaffold_valid_annotated_core:,}**;
5. unique primary semantic queries after within-run duplicate collapse: **{len(primary_by_key):,}**.

The final query identity set exactly matches the Prompt-4 primary semantic query file.

## Anomalies and data-quality findings

- Invalid references: **{invalid_count}** rows; raw values `{dict(sorted(invalid_ref_raw.items()))}`.
- Off-scaffold rows: **{off_count}** ({off_count/len(rows):.2%}); classes `{dict(sorted(off_scaffold.items()))}`.
- Exact duplicate excess rows: **{exact_duplicate_excess}** across **{duplicate_groups}** duplicate groups.
- Conflicting declared UC references among primary duplicate behavior groups: **{len(conflicts)}**.
- Annotated rows with blank `UC_Action`: **{consistency.get('annotated_true_blank_action',0):,}**.

## Interpretation boundary

RQ1 describes the presence, completeness, validity, coverage, and internal consistency of explicit trace metadata. It does not establish semantic correctness, causal trace provenance, or intrinsic generator superiority. Per-generator tables are retained in fixed repository order and must not be presented as a leaderboard.
"""
    REPORT.write_text(report, encoding="utf-8")

    handoff = f"""# HANDOFF 05 — RQ1 STRUCTURAL TRACEABILITY

## STATUS

**PASS — pending independent Prompt-5 validator**

## Scope executed

Prompt 5 executed the frozen descriptive RQ1 structural-traceability analysis over all **{len(rows):,}** clean rows. No new semantic retrieval, sensitivity encoder, negative control, resampling, fragmentation, or inferential test was run.

## Headline structural results

- Trace annotation present: **{annotated_count:,}/{len(rows):,} = {annotated_count/len(rows):.6f}**.
- Valid UC1–UC21 reference: **{valid_count:,}/{len(rows):,} = {valid_count/len(rows):.6f}**.
- Valid reference among annotated rows: **{valid_count:,}/{annotated_count:,} = {valid_count/annotated_count:.6f}**.
- Complete explicit trace metadata: **{complete_count:,}/{len(rows):,} = {complete_count/len(rows):.6f}**.
- Complete trace among annotated rows: **{complete_count:,}/{annotated_count:,} = {complete_count/annotated_count:.6f}**.
- Invalid nonblank references: **{invalid_count}**, values `{dict(sorted(invalid_ref_raw.items()))}`.
- Off-scaffold rows: **{off_count}**.
- Exact duplicate excess rows: **{exact_duplicate_excess}**.
- Primary duplicate reference conflicts: **{len(conflicts)}**.

## Model × Run coverage

- Zero-annotation cells: **{zero_annotation_cells}/90**.
- Zero-valid-reference cells: **{zero_valid_cells}/90**.
- Zero-complete-trace cells: **{zero_complete_cells}/90**.
- Zero-primary-semantic-query cells: **{zero_primary_cells}/90**.
- Primary semantic-query cells: **{90-zero_primary_cells}/90**.

## Population reconciliation

Prompt-5 independently reconstructed the primary semantic query identity set and matched Prompt 4 exactly: **{len(primary_by_key):,} queries**.

## Required interpretation boundary

`complete_explicit_trace` is structural metadata completeness only. RQ1 does not establish semantic correctness, accuracy, causal trace provenance, or universal generator quality. Per-model results are descriptive and unsorted by outcome.

## Gate

Acceptance requires `scripts/05_validate_structural_traceability_rq1.py` and the Prompt-5 regression suite to pass. Until then, this handoff is provisional.
"""
    HANDOFF.write_text(handoff, encoding="utf-8")

    manifest_files = [
        OUT / "model_run_structural_metrics.csv", OUT / "cell_distribution_summary.csv",
        OUT / "model_structural_summary.csv", OUT / "uc_structural_coverage.csv",
        OUT / "structural_funnel.csv", OUT / "row_state_counts.csv",
        OUT / "annotation_consistency_profile.csv", OUT / "off_scaffold_profile.csv",
        OUT / "invalid_reference_profile.csv", OUT / "duplicate_profile.csv",
        OUT / "corpus_structural_summary.json", REPORT, HANDOFF,
    ]
    manifest = {
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {"clean_source": sha256(CLEAN), "scaffold": sha256(SCAFFOLD), "prompt4_queries": sha256(PROMPT4_QUERIES)},
        "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in manifest_files},
    }
    (OUT / "structural_rq1_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "PASS", "rows": len(rows), "annotated": annotated_count,
        "valid_reference": valid_count, "complete_trace": complete_count,
        "invalid_reference": invalid_count, "off_scaffold": off_count,
        "duplicate_excess": exact_duplicate_excess, "primary_queries": len(primary_by_key),
        "zero_annotation_cells": zero_annotation_cells, "zero_complete_cells": zero_complete_cells,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
