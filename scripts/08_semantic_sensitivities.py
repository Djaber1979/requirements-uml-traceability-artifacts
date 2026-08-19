#!/usr/bin/env python3
"""Prompt 8: execute the Prompt-3-frozen semantic sensitivity analyses.

Design is prospectively operationalized in Amendment 006. This script runs only:
- one-factor-at-a-time requirement representation sensitivities;
- one-factor-at-a-time behavior representation sensitivities;
- frozen MiniLM sensitivity;
- off-scaffold inclusion sensitivity;
- raw-row/no-dedup sensitivity;
- equal-cell macro vs row-weighted micro summaries.

It does not run new permutation nulls, run resampling, scenario-step retrieval,
expert/LLM judging, FEER/voting comparisons, or generator ranking.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "derived/behaviors/source_without_legacy_semantics.csv"
PRIMARY_QUERIES = ROOT / "derived/behaviors/primary_semantic_queries.csv"
PRIMARY_REQ = ROOT / "derived/requirements/requirements_primary.csv"
UCS = ROOT / "input_snapshot/provenance/UCs.txt"
SCAFFOLD = ROOT / "input_snapshot/provenance/Methodless.txt"
RETRIEVAL_CFG = ROOT / "config/retrieval_rules.yaml"
MODEL_CFG = ROOT / "config/semantic_models.yaml"
PREPROCESS_CFG = ROOT / "config/preprocessing.yaml"
AMENDMENTS = ROOT / "PROTOCOL_AMENDMENTS.md"
PROMPT4_CORPUS = ROOT / "artifacts/semantic_retrieval/corpus_primary_metrics.json"
PROMPT4_TFIDF = ROOT / "artifacts/semantic_retrieval/tfidf_scores.csv"
PROMPT4_MPNET = ROOT / "artifacts/semantic_retrieval/mpnet_scores.csv"
OUT = ROOT / "artifacts/semantic_sensitivities"
REPORT = ROOT / "reports/RQ4_SEMANTIC_SENSITIVITIES.md"
HANDOFF = ROOT / "reports/HANDOFF_08_SEMANTIC_SENSITIVITIES.md"
OUT.mkdir(parents=True, exist_ok=True)

VALID_UCS = tuple(range(1, 22))
PRIMARY_N = 2369
RAW_ON_SCAFFOLD_N = 2384
PRIMARY_CELLS = 69
FORBIDDEN = {"Best_Match_Action", "SimilarityScore"}
CONDITION_ORDER = [
    "primary_anchor",
    "req_title_description",
    "req_with_extensions",
    "behavior_method_only",
    "behavior_class_method",
    "minilm_primary",
    "include_off_scaffold",
    "raw_rows_no_dedup",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def parse_bool(v: str) -> bool:
    return (v or "").strip().casefold() == "true"


def parse_uc(v: str):
    x = (v or "").strip()
    if re.fullmatch(r"\d+", x):
        return int(x)
    m = re.fullmatch(r"(?i:UC)\s*(\d+)", x)
    return int(m.group(1)) if m else None


def scaffold_identities(text: str) -> set[str]:
    out: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r'^(?:abstract\s+)?class\s+"([^"]+)"\s+as\s+([A-Za-z_]\w*)', line)
        if m:
            out.update([m.group(1), m.group(2)]); continue
        m = re.match(r"^(?:abstract\s+)?class\s+([A-Za-z_]\w*)", line)
        if m:
            out.add(m.group(1))
    return out


def canonical_class(v: str) -> str:
    x = (v or "").strip()
    return "RoleManager" if x == "RoleManager <<service>>" else x


def canonical_signature(v: str) -> str:
    x = " ".join((v or "").strip().split()).casefold()
    return re.sub(r"\s*([(),:\[\]<>])\s*", r"\1", x)


def behavior_key(row: dict) -> tuple:
    return (
        row["Model"].strip(), int(row["Run"]), canonical_class(row["Class"]),
        row["MethodName"].strip().casefold(), canonical_signature(row["Signature"]),
    )


def segment_text(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return " ".join(s.split())


def render_behavior(q: dict, representation: str) -> str:
    c = segment_text(q["ClassCanonical"])
    m = segment_text(q["MethodName"])
    s = segment_text(q["Signature"])
    if representation == "method_only":
        return f"Method: {m}."
    if representation == "class_method":
        return f"Class: {c}. Method: {m}."
    if representation == "full":
        return f"Class: {c}. Method: {m}. Signature: {s}."
    raise ValueError(representation)


def parse_requirements(text: str) -> list[dict]:
    known = {"Actor", "Description", "PreConditions", "Triggers", "MainScenario", "PostConditions", "Extensions"}
    rows: list[dict] = []
    cur = None
    section = None

    def finish():
        nonlocal cur
        if cur is not None and cur.get("id") is not None:
            rows.append(cur)
        cur = None

    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("Title:"):
            finish()
            cur = {"id": None, "title": stripped.split(":", 1)[1].strip(), "description": "", "main": [], "extensions": []}
            section = "Title"; continue
        if cur is None:
            continue
        if stripped.startswith("ID:"):
            m = re.fullmatch(r"ID:\s*UC(\d+)", stripped, flags=re.I)
            if not m:
                raise ValueError(f"Malformed UC ID: {stripped}")
            cur["id"] = int(m.group(1)); section = "ID"; continue
        hm = re.match(r"^([A-Za-z]+(?:[A-Za-z ]+)?):\s*(.*)$", stripped)
        if hm and hm.group(1) in known:
            section = hm.group(1)
            payload = hm.group(2).strip()
            if section == "Description": cur["description"] = payload
            elif section == "Extensions" and payload:
                cur["extensions"].append(re.sub(r"^[–-]\s*", "", payload))
            continue
        if not stripped:
            continue
        if section == "Description" and not stripped.startswith("–"):
            cur["description"] = (cur["description"] + " " + stripped).strip()
        elif section == "MainScenario":
            step = re.sub(r"^\d+\.\s*", "", stripped)
            if step: cur["main"].append(step)
        elif section == "Extensions":
            ext = re.sub(r"^[–-]\s*", "", stripped)
            if ext: cur["extensions"].append(ext)
    finish()
    rows.sort(key=lambda x: x["id"])
    if [r["id"] for r in rows] != list(VALID_UCS):
        raise ValueError("Requirement parser did not recover exactly UC1-UC21")
    for r in rows:
        if not r["title"] or not r["description"] or not r["main"]:
            raise ValueError(f"Incomplete UC{r['id']} representation")
        r["title_description"] = f"Title: {r['title']}. Description: {r['description']}."
        r["primary"] = f"Title: {r['title']}. Description: {r['description']}. MainScenario: {' '.join(r['main'])}"
        ext_text = " ".join(r["extensions"])
        r["with_extensions"] = r["primary"] + (f" Extensions: {ext_text}" if ext_text else " Extensions:")
    return rows


def load_primary_queries() -> list[dict]:
    rows = read_csv(PRIMARY_QUERIES)
    if len(rows) != PRIMARY_N:
        raise ValueError(f"Primary query count drift: {len(rows)}")
    if rows and ({"UC_Action"} | FORBIDDEN).intersection(rows[0]):
        raise ValueError("Prohibited field leaked into primary query file")
    return rows


def materialize_population(include_off_scaffold: bool, collapse_duplicates: bool) -> list[dict]:
    scaffold = scaffold_identities(SCAFFOLD.read_text(encoding="utf-8"))
    source = read_csv(CLEAN)
    if not source or FORBIDDEN.intersection(source[0]):
        raise ValueError("Clean source schema is not safe")
    eligible: list[dict] = []
    for idx, r in enumerate(source, start=1):
        if not parse_bool(r["Has_UC_Annotation"]):
            continue
        uc = parse_uc(r["UC_References"])
        if uc not in VALID_UCS:
            continue
        if not all((r[k] or "").strip() for k in ["Class", "MethodName", "Signature"]):
            continue
        if not include_off_scaffold and r["Class"].strip() not in scaffold:
            continue
        q = {
            "Model": r["Model"].strip(), "Run": int(r["Run"]), "File": r["File"],
            "ClassRaw": r["Class"], "ClassCanonical": canonical_class(r["Class"]),
            "MethodName": r["MethodName"], "Signature": r["Signature"],
            "DeclaredUC": int(uc), "FirstSourceRow": idx,
        }
        eligible.append(q)
    if not collapse_duplicates:
        for n, q in enumerate(eligible, start=1):
            q["QueryID"] = f"RAW{n:06d}"
        return eligible
    groups: dict[tuple, list[dict]] = defaultdict(list)
    order: list[tuple] = []
    for q in eligible:
        key = (
            q["Model"], int(q["Run"]), q["ClassCanonical"],
            q["MethodName"].strip().casefold(), canonical_signature(q["Signature"]),
        )
        if key not in groups: order.append(key)
        groups[key].append(q)
    out = []
    for n, key in enumerate(order, start=1):
        group = groups[key]
        targets = {int(x["DeclaredUC"]) for x in group}
        if len(targets) != 1:
            raise ValueError(f"Conflicting duplicate targets in sensitivity population: {key} -> {targets}")
        q = dict(group[0]); q["QueryID"] = f"INC{n:06d}"; q["CollapsedSourceRows"] = len(group)
        out.append(q)
    return out


def load_score_matrix(path: Path, expected_ids: list[str]) -> np.ndarray:
    rows = read_csv(path)
    ids = [r["QueryID"] for r in rows]
    if ids != expected_ids:
        raise ValueError(f"Score-matrix query identity drift: {path}")
    arr = np.asarray([[float(r[f"UC{i}"]) for i in VALID_UCS] for r in rows], dtype=np.float64)
    if arr.shape != (len(expected_ids), 21) or not np.all(np.isfinite(arr)):
        raise ValueError(f"Invalid score matrix {path}: {arr.shape}")
    return arr


def make_tfidf(requirement_texts: list[str], query_texts: list[str], cfg: dict) -> np.ndarray:
    lex = cfg["lexical_primary"]
    v = TfidfVectorizer(
        analyzer=lex["analyzer"], ngram_range=(int(lex["ngram_min"]), int(lex["ngram_max"])),
        lowercase=bool(lex["lowercase"]), min_df=int(lex["min_df"]), max_df=float(lex["max_df"]),
        sublinear_tf=bool(lex["sublinear_tf"]), smooth_idf=bool(lex["smooth_idf"]),
        norm=lex["norm"], stop_words=lex.get("stop_words"),
    )
    req = v.fit_transform(requirement_texts)
    qry = v.transform(query_texts)
    return np.asarray((qry @ req.T).toarray(), dtype=np.float64)


def rank_row(scores: np.ndarray, declared: int) -> dict:
    order = sorted(range(21), key=lambda i: (-float(scores[i]), i + 1))
    ranking = [i + 1 for i in order]
    rank = ranking.index(int(declared)) + 1
    return {
        "Top1UC": ranking[0], "DeclaredRank": rank, "Hit1": int(rank == 1),
        "Hit3": int(rank <= 3), "Hit5": int(rank <= 5), "ReciprocalRank": 1.0 / rank,
    }


def summarize(condition: str, method: str, scores: np.ndarray, queries: list[dict]) -> tuple[list[dict], list[dict], dict]:
    if scores.shape != (len(queries), 21):
        raise ValueError(f"{condition}/{method} score shape {scores.shape} != {(len(queries), 21)}")
    qrows = []
    for q, score in zip(queries, scores):
        m = rank_row(score, int(q["DeclaredUC"]))
        qrows.append({
            "Condition": condition, "Method": method, "QueryID": q["QueryID"],
            "Model": q["Model"], "Run": int(q["Run"]), "DeclaredUC": int(q["DeclaredUC"]), **m,
        })
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in qrows: by_cell[(r["Model"], int(r["Run"]))].append(r)
    cells = []
    for (model, run), g in sorted(by_cell.items(), key=lambda x: (x[0][0], x[0][1])):
        cells.append({
            "Condition": condition, "Method": method, "Model": model, "Run": run, "NQueries": len(g),
            "Hit1Mean": statistics.fmean(float(x["Hit1"]) for x in g),
            "MRRMean": statistics.fmean(float(x["ReciprocalRank"]) for x in g),
            "Hit3Mean": statistics.fmean(float(x["Hit3"]) for x in g),
            "Hit5Mean": statistics.fmean(float(x["Hit5"]) for x in g),
        })
    summary = {
        "Condition": condition, "Method": method, "QueryCount": len(qrows), "EligibleCells": len(cells),
        "ZeroQueryCells": 90 - len(cells),
        "Hit1EqualCellMacro": statistics.fmean(float(x["Hit1Mean"]) for x in cells),
        "MRREqualCellMacro": statistics.fmean(float(x["MRRMean"]) for x in cells),
        "Hit3EqualCellMacro": statistics.fmean(float(x["Hit3Mean"]) for x in cells),
        "Hit5EqualCellMacro": statistics.fmean(float(x["Hit5Mean"]) for x in cells),
        "Hit1Micro": statistics.fmean(float(x["Hit1"]) for x in qrows),
        "MRRMicro": statistics.fmean(float(x["ReciprocalRank"]) for x in qrows),
    }
    return qrows, cells, summary


def token_profile(model: SentenceTransformer, texts: list[str]) -> dict:
    lengths = []
    for text in texts:
        enc = model.tokenizer(text, add_special_tokens=True, truncation=False, padding=False)
        lengths.append(len(enc["input_ids"]))
    max_len = int(model.max_seq_length)
    return {
        "count": len(texts), "model_max_seq_length": max_len,
        "max_untruncated_wordpieces": max(lengths) if lengths else 0,
        "truncated_if_native_limit_applied": sum(x > max_len for x in lengths),
    }


def load_encoder(role: str, cfg: dict) -> tuple[SentenceTransformer, dict]:
    spec = next(x for x in cfg["models"] if x["role"] == role)
    model_id = str(spec["model_id"]); rev = str(spec["revision"])
    resolved = str(HfApi().model_info(repo_id=model_id, revision=rev).sha)
    if resolved != rev:
        raise RuntimeError(f"Frozen revision mismatch for {role}: {rev} -> {resolved}")
    model = SentenceTransformer(model_id, revision=rev, device="cpu")
    expected = int(spec["expected_default_max_wordpieces"])
    if int(model.max_seq_length) != expected:
        raise RuntimeError(f"Frozen max sequence length mismatch for {role}: {model.max_seq_length} != {expected}")
    return model, {"model_id": model_id, "requested_revision": rev, "resolved_revision": resolved, "model_max_seq_length": expected}


def encode(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    return np.asarray(model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True))


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if "## Amendment 006 — Prompt 8 semantic-sensitivity operationalization" not in AMENDMENTS.read_text(encoding="utf-8"):
        raise SystemExit("Prospective Amendment 006 is missing")
    retrieval = yaml.safe_load(RETRIEVAL_CFG.read_text(encoding="utf-8"))
    models = yaml.safe_load(MODEL_CFG.read_text(encoding="utf-8"))
    preprocess = yaml.safe_load(PREPROCESS_CFG.read_text(encoding="utf-8"))
    if not (retrieval.get("frozen") and models.get("frozen") and preprocess.get("frozen")):
        raise SystemExit("Prompt-3 semantic configuration is not frozen")

    import torch
    torch.manual_seed(20260818)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)

    req = parse_requirements(UCS.read_text(encoding="utf-8"))
    primary_req_saved = read_csv(PRIMARY_REQ)
    if [r["primary"] for r in req] != [r["PrimaryText"] for r in primary_req_saved]:
        raise SystemExit("Primary requirement parser no longer matches Prompt 4 materialization")
    req_text = {
        "title_description": [r["title_description"] for r in req],
        "primary": [r["primary"] for r in req],
        "with_extensions": [r["with_extensions"] for r in req],
    }
    write_csv(OUT / "requirement_representations.csv",
              ["UC", "TitleDescription", "Primary", "WithExtensions"],
              [{"UC": f"UC{r['id']}", "TitleDescription": r["title_description"], "Primary": r["primary"], "WithExtensions": r["with_extensions"]} for r in req])

    primary = load_primary_queries()
    primary_ids = [q["QueryID"] for q in primary]
    primary_full = [render_behavior(q, "full") for q in primary]
    if primary_full != [q["QueryText"] for q in primary]:
        raise SystemExit("Primary behavior rendering drifted from Prompt 4")
    method_only = [render_behavior(q, "method_only") for q in primary]
    class_method = [render_behavior(q, "class_method") for q in primary]

    raw = materialize_population(include_off_scaffold=False, collapse_duplicates=False)
    if len(raw) != RAW_ON_SCAFFOLD_N:
        raise SystemExit(f"Frozen raw on-scaffold population drift: {len(raw)}")
    inclusive = materialize_population(include_off_scaffold=True, collapse_duplicates=True)
    raw_full = [render_behavior(q, "full") for q in raw]
    inclusive_full = [render_behavior(q, "full") for q in inclusive]

    # Primary anchors are reused from Prompt 4, never overwritten.
    saved_tf = load_score_matrix(PROMPT4_TFIDF, primary_ids)
    saved_mp = load_score_matrix(PROMPT4_MPNET, primary_ids)
    all_qrows: list[dict] = []
    all_cells: list[dict] = []
    summaries: list[dict] = []
    for method, scores in [("TFIDF", saved_tf), ("MPNET", saved_mp)]:
        q, c, s = summarize("primary_anchor", method, scores, primary)
        all_qrows += q; all_cells += c; summaries.append(s)

    # Lexical sensitivities.
    tf_primary_recomputed = make_tfidf(req_text["primary"], primary_full, retrieval)
    if not np.allclose(tf_primary_recomputed, saved_tf, rtol=0, atol=1e-12):
        raise SystemExit("Recomputed primary TF-IDF scores do not match Prompt 4")
    tf_conditions = [
        ("req_title_description", make_tfidf(req_text["title_description"], primary_full, retrieval), primary),
        ("req_with_extensions", make_tfidf(req_text["with_extensions"], primary_full, retrieval), primary),
        ("behavior_method_only", make_tfidf(req_text["primary"], method_only, retrieval), primary),
        ("behavior_class_method", make_tfidf(req_text["primary"], class_method, retrieval), primary),
        ("include_off_scaffold", make_tfidf(req_text["primary"], inclusive_full, retrieval), inclusive),
        ("raw_rows_no_dedup", make_tfidf(req_text["primary"], raw_full, retrieval), raw),
    ]
    for condition, scores, qs in tf_conditions:
        q, c, s = summarize(condition, "TFIDF", scores, qs); all_qrows += q; all_cells += c; summaries.append(s)

    # Primary MPNet and representation/population sensitivities.
    mpnet, mp_info = load_encoder("primary_neural", models)
    mp_req = {k: encode(mpnet, v, 16) for k, v in req_text.items()}
    mp_q_primary = encode(mpnet, primary_full)
    primary_recomputed = np.asarray(mp_q_primary @ mp_req["primary"].T, dtype=np.float64)
    primary_rank_changes = 0
    max_abs_primary_score_diff = float(np.max(np.abs(primary_recomputed - saved_mp)))
    for a, b, q in zip(primary_recomputed, saved_mp, primary):
        if rank_row(a, int(q["DeclaredUC"]))["DeclaredRank"] != rank_row(b, int(q["DeclaredUC"]))["DeclaredRank"]:
            primary_rank_changes += 1
    if primary_rank_changes != 0 or max_abs_primary_score_diff > 1e-5:
        raise SystemExit(f"MPNet primary numerical-reproducibility guard failed: rank_changes={primary_rank_changes}, max_abs={max_abs_primary_score_diff}")

    mp_conditions = [
        ("req_title_description", np.asarray(mp_q_primary @ mp_req["title_description"].T, dtype=np.float64), primary),
        ("req_with_extensions", np.asarray(mp_q_primary @ mp_req["with_extensions"].T, dtype=np.float64), primary),
        ("behavior_method_only", np.asarray(encode(mpnet, method_only) @ mp_req["primary"].T, dtype=np.float64), primary),
        ("behavior_class_method", np.asarray(encode(mpnet, class_method) @ mp_req["primary"].T, dtype=np.float64), primary),
        ("include_off_scaffold", np.asarray(encode(mpnet, inclusive_full) @ mp_req["primary"].T, dtype=np.float64), inclusive),
        ("raw_rows_no_dedup", np.asarray(encode(mpnet, raw_full) @ mp_req["primary"].T, dtype=np.float64), raw),
    ]
    for condition, scores, qs in mp_conditions:
        q, c, s = summarize(condition, "MPNET", scores, qs); all_qrows += q; all_cells += c; summaries.append(s)

    # Frozen second encoder sensitivity at the primary representation/population only.
    minilm, mini_info = load_encoder("sensitivity_neural", models)
    mini_req_emb = encode(minilm, req_text["primary"], 16)
    mini_q_emb = encode(minilm, primary_full)
    mini_scores = np.asarray(mini_q_emb @ mini_req_emb.T, dtype=np.float64)
    q, c, s = summarize("minilm_primary", "MINILM", mini_scores, primary)
    all_qrows += q; all_cells += c; summaries.append(s)

    # Add anchor-relative deltas after all summaries exist.
    anchors = {(x["Method"]): x for x in summaries if x["Condition"] == "primary_anchor"}
    for s in summaries:
        anchor_method = "MPNET" if s["Method"] == "MINILM" else s["Method"]
        a = anchors[anchor_method]
        s["AnchorMethod"] = anchor_method
        s["DeltaHit1Macro"] = float(s["Hit1EqualCellMacro"]) - float(a["Hit1EqualCellMacro"])
        s["DeltaMRRMacro"] = float(s["MRREqualCellMacro"]) - float(a["MRREqualCellMacro"])
        s["DeltaHit1Micro"] = float(s["Hit1Micro"]) - float(a["Hit1Micro"])
        s["DeltaMRRMicro"] = float(s["MRRMicro"]) - float(a["MRRMicro"])

    order_index = {name: i for i, name in enumerate(CONDITION_ORDER)}
    method_index = {"TFIDF": 0, "MPNET": 1, "MINILM": 2}
    summaries.sort(key=lambda x: (order_index[x["Condition"]], method_index[x["Method"]]))
    all_qrows.sort(key=lambda x: (order_index[x["Condition"]], method_index[x["Method"]], x["QueryID"]))
    all_cells.sort(key=lambda x: (order_index[x["Condition"]], method_index[x["Method"]], x["Model"], int(x["Run"])))

    summary_fields = [
        "Condition", "Method", "AnchorMethod", "QueryCount", "EligibleCells", "ZeroQueryCells",
        "Hit1EqualCellMacro", "MRREqualCellMacro", "Hit3EqualCellMacro", "Hit5EqualCellMacro",
        "Hit1Micro", "MRRMicro", "DeltaHit1Macro", "DeltaMRRMacro", "DeltaHit1Micro", "DeltaMRRMicro",
    ]
    query_fields = ["Condition", "Method", "QueryID", "Model", "Run", "DeclaredUC", "Top1UC", "DeclaredRank", "Hit1", "Hit3", "Hit5", "ReciprocalRank"]
    cell_fields = ["Condition", "Method", "Model", "Run", "NQueries", "Hit1Mean", "MRRMean", "Hit3Mean", "Hit5Mean"]
    write_csv(OUT / "sensitivity_summary.csv", summary_fields, summaries)
    write_csv(OUT / "sensitivity_query_metrics.csv", query_fields, all_qrows)
    write_csv(OUT / "sensitivity_cell_metrics.csv", cell_fields, all_cells)

    off_scaffold_raw = 2408 - RAW_ON_SCAFFOLD_N
    pop = {
        "primary_queries": len(primary), "primary_cells": len({(q['Model'], int(q['Run'])) for q in primary}),
        "raw_on_scaffold_queries": len(raw), "raw_on_scaffold_cells": len({(q['Model'], int(q['Run'])) for q in raw}),
        "inclusive_deduplicated_queries": len(inclusive), "inclusive_cells": len({(q['Model'], int(q['Run'])) for q in inclusive}),
        "additional_queries_vs_primary": len(inclusive) - len(primary),
        "otherwise_eligible_off_scaffold_raw_rows_from_frozen_audit": off_scaffold_raw,
    }
    (OUT / "population_manifest.json").write_text(json.dumps(pop, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime = {
        "status": "PROMPT8_SEMANTIC_SENSITIVITIES_EXECUTED",
        "design": "one_factor_at_a_time",
        "primary_anchor_reused": True,
        "mpnet_primary_reembedding_max_abs_score_difference": max_abs_primary_score_diff,
        "mpnet_primary_reembedding_declared_rank_changes": primary_rank_changes,
        "primary_neural": mp_info,
        "sensitivity_neural": mini_info,
        "truncation_profiles": {
            "MPNET": {
                "req_title_description": token_profile(mpnet, req_text["title_description"]),
                "req_primary": token_profile(mpnet, req_text["primary"]),
                "req_with_extensions": token_profile(mpnet, req_text["with_extensions"]),
                "query_primary": token_profile(mpnet, primary_full),
                "query_method_only": token_profile(mpnet, method_only),
                "query_class_method": token_profile(mpnet, class_method),
                "query_include_off_scaffold": token_profile(mpnet, inclusive_full),
                "query_raw_rows": token_profile(mpnet, raw_full),
            },
            "MINILM": {
                "req_primary": token_profile(minilm, req_text["primary"]),
                "query_primary": token_profile(minilm, primary_full),
            },
        },
        "packages": {name: pkg_version(name) for name in ["numpy", "scipy", "scikit-learn", "sentence-transformers", "transformers", "torch", "huggingface-hub", "PyYAML"]},
        "boundaries": {
            "legacy_semantic_fields_used": False, "uc_action_used": False, "new_permutation_null_run": False,
            "run_resampling_run": False, "scenario_step_retrieval_run": False, "generator_quality_ranking": False,
            "significance_tests_run": False, "model_selection_from_sensitivity_outcomes": False,
        },
    }
    (OUT / "runtime_manifest.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "sensitivity_summary.json").write_text(json.dumps({"population": pop, "summaries": summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Human-readable report, deliberately descriptive and unsorted by performance.
    lines = [
        "# RQ4 — Frozen Semantic Sensitivity Analyses", "",
        "## Scope", "",
        "Prompt 8 evaluates the Prompt-3-frozen robustness perturbations one factor at a time. The Prompt-4 primary analysis remains the anchor; no sensitivity is promoted to a new primary analysis.", "",
        "## Sensitivity results", "",
        "| Condition | Method | N | Cells | Hit@1 macro | MRR macro | Δ Hit@1 | Δ MRR | Hit@1 micro | MRR micro |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['Condition']} | {s['Method']} | {s['QueryCount']} | {s['EligibleCells']} | "
            f"{s['Hit1EqualCellMacro']:.6f} | {s['MRREqualCellMacro']:.6f} | {s['DeltaHit1Macro']:+.6f} | {s['DeltaMRRMacro']:+.6f} | "
            f"{s['Hit1Micro']:.6f} | {s['MRRMicro']:.6f} |"
        )
    lines += [
        "", "## Population sensitivities", "",
        f"- Primary deduplicated on-scaffold population: **{pop['primary_queries']}** queries in **{pop['primary_cells']}** cells.",
        f"- Raw-row/no-dedup population: **{pop['raw_on_scaffold_queries']}** queries in **{pop['raw_on_scaffold_cells']}** cells.",
        f"- Off-scaffold-inclusive deduplicated population: **{pop['inclusive_deduplicated_queries']}** queries in **{pop['inclusive_cells']}** cells, **{pop['additional_queries_vs_primary']:+d}** versus primary.",
        "", "## Neural execution", "",
        f"- MPNet resolved revision: `{mp_info['resolved_revision']}`.",
        f"- MiniLM resolved revision: `{mini_info['resolved_revision']}`.",
        f"- MPNet re-embedding check versus saved Prompt-4 primary scores: max |Δ| = **{max_abs_primary_score_diff:.3e}**, declared-rank changes = **{primary_rank_changes}**.",
        "", "## Interpretation boundary", "",
        "These are prespecified robustness perturbations of automated retrieval concordance. Differences do not identify a semantically correct representation or encoder. No significance test, winner selection, semantic-ground-truth claim, or generator-quality ranking is made.", "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    HANDOFF.write_text(
        "# HANDOFF 08 — SEMANTIC SENSITIVITIES\n\n## STATUS\n\n**PASS — pending independent Prompt-8 validator**\n\n"
        "## Scope\n\nExecuted the Prompt-3-frozen one-factor-at-a-time requirement/behavior representation sensitivities, frozen MiniLM encoder sensitivity, off-scaffold inclusion, raw-row/no-dedup sensitivity, and micro-vs-equal-cell aggregation.\n\n"
        "## Gate\n\nAcceptance requires independent validation, regression tests, deterministic rerun, exact model-revision checks, and scientific-boundary checks.\n",
        encoding="utf-8",
    )

    print(json.dumps({"population": pop, "summaries": summaries, "runtime": {"mpnet": mp_info, "minilm": mini_info}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
