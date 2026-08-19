#!/usr/bin/env python3
"""Prompt 4: execute the frozen primary semantic-retrieval protocol.

This script implements only the Prompt-3-frozen primary semantic stage:
- deterministic UC1-UC21 requirement materialization;
- deterministic primary behavior-query materialization;
- requirement-fitted TF-IDF cosine retrieval;
- exact-revision MPNet cosine retrieval;
- frozen declared-UC ranking measures;
- frozen primary cross-method consensus measures.

It does NOT run MiniLM sensitivity, representation sensitivities, off-scaffold
sensitivity, negative controls, run resampling, RQ3 fragmentation, or any
legacy semantic field.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import HfApi
from scipy.stats import rankdata
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "derived/behaviors/source_without_legacy_semantics.csv"
UCS = ROOT / "input_snapshot/provenance/UCs.txt"
SCAFFOLD = ROOT / "input_snapshot/provenance/Methodless.txt"
PREPROCESS = ROOT / "config/preprocessing.yaml"
RETRIEVAL = ROOT / "config/retrieval_rules.yaml"
SEM_MODELS = ROOT / "config/semantic_models.yaml"
PROTOCOL = ROOT / "STUDY_PROTOCOL.md"
PROMPT3_AUDIT = ROOT / "artifacts/protocol_freeze/primary_population_audit.json"

REQ_OUT = ROOT / "derived/requirements/requirements_primary.csv"
QUERY_OUT = ROOT / "derived/behaviors/primary_semantic_queries.csv"
ART = ROOT / "artifacts/semantic_retrieval"
REPORTS = ROOT / "reports"
ART.mkdir(parents=True, exist_ok=True)
REQ_OUT.parent.mkdir(parents=True, exist_ok=True)
QUERY_OUT.parent.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

PROMPT3_MERGE_COMMIT = "4feddb466b7843af9e872cc44d88aa8d76efccbc"
VALID_UCS = tuple(range(1, 22))
EXPECTED_CLEAN_COLUMNS = [
    "Model", "Run", "File", "Class", "MethodName", "Signature",
    "Has_UC_Annotation", "UC_References", "UC_Action",
]
FORBIDDEN_FIELDS = {"Best_Match_Action", "SimilarityScore"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def parse_bool(v: str) -> bool:
    return (v or "").strip().lower() == "true"


def parse_single_uc(v: str):
    s = (v or "").strip()
    if re.fullmatch(r"\d+", s):
        return int(s)
    m = re.fullmatch(r"(?i:UC)\s*(\d+)", s)
    return int(m.group(1)) if m else None


def scaffold_identities(text: str) -> set[str]:
    out: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r'^(?:abstract\s+)?class\s+"([^"]+)"\s+as\s+([A-Za-z_]\w*)', line)
        if m:
            out.update([m.group(1), m.group(2)])
            continue
        m = re.match(r"^(?:abstract\s+)?class\s+([A-Za-z_]\w*)", line)
        if m:
            out.add(m.group(1))
    return out


def canonical_class(v: str) -> str:
    x = (v or "").strip()
    return "RoleManager" if x == "RoleManager <<service>>" else x


def canonical_signature(v: str) -> str:
    s = " ".join((v or "").strip().split()).casefold()
    s = re.sub(r"\s*([(),:\[\]<>])\s*", r"\1", s)
    return s


def behavior_key(row: dict[str, str]) -> tuple:
    return (
        row["Model"].strip(),
        int(row["Run"]),
        canonical_class(row["Class"]),
        row["MethodName"].strip().casefold(),
        canonical_signature(row["Signature"]),
    )


def segment_text(text: str) -> str:
    """Deterministically segment identifiers and punctuation for retrieval text."""
    s = (text or "").strip()
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return " ".join(s.split())


def render_query(class_name: str, method: str, signature: str) -> str:
    return (
        f"Class: {segment_text(class_name)}. "
        f"Method: {segment_text(method)}. "
        f"Signature: {segment_text(signature)}."
    )


def parse_uc_spec(text: str) -> list[dict]:
    """Parse Title, Description, and numbered MainScenario steps without rewriting."""
    rows: list[dict] = []
    current: dict | None = None
    section: str | None = None

    def finish():
        nonlocal current
        if current is None:
            return
        if current.get("id") is not None:
            rows.append(current)
        current = None

    known_headers = {
        "Actor", "Description", "PreConditions", "Triggers", "MainScenario",
        "PostConditions", "Extensions",
    }

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("Title:"):
            finish()
            current = {
                "id": None,
                "title": stripped.split(":", 1)[1].strip(),
                "description": "",
                "main_steps": [],
            }
            section = "Title"
            continue
        if current is None:
            continue
        if stripped.startswith("ID:"):
            m = re.fullmatch(r"ID:\s*UC(\d+)", stripped, flags=re.I)
            if not m:
                raise ValueError(f"Malformed UC ID line: {stripped}")
            current["id"] = int(m.group(1))
            section = "ID"
            continue
        header_match = re.match(r"^([A-Za-z]+(?:[A-Za-z ]+)?):\s*(.*)$", stripped)
        if header_match and header_match.group(1) in known_headers:
            section = header_match.group(1)
            payload = header_match.group(2).strip()
            if section == "Description":
                current["description"] = payload
            continue
        if section == "MainScenario" and stripped:
            step = re.sub(r"^\d+\.\s*", "", stripped)
            if step:
                current["main_steps"].append(step)
        elif section == "Description" and stripped and not stripped.startswith("–"):
            # Safe continuation support; no semantic rewriting.
            current["description"] = (current["description"] + " " + stripped).strip()
    finish()

    rows.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in rows]
    if ids != list(VALID_UCS):
        raise ValueError(f"Expected exactly UC1-UC21; observed {ids}")
    for r in rows:
        if not r["title"] or not r["description"] or not r["main_steps"]:
            raise ValueError(f"Incomplete primary requirement representation for UC{r['id']}")
        r["primary_text"] = (
            f"Title: {r['title']}. Description: {r['description']}. "
            f"MainScenario: {' '.join(r['main_steps'])}"
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_primary_queries() -> list[dict]:
    scaffold = scaffold_identities(SCAFFOLD.read_text(encoding="utf-8"))
    if len(scaffold) != 21:
        raise ValueError(f"Expected 21 accepted exact scaffold identities; got {len(scaffold)}")

    with CLEAN.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != EXPECTED_CLEAN_COLUMNS:
            raise ValueError(f"Unexpected clean schema: {reader.fieldnames}")
        if FORBIDDEN_FIELDS.intersection(reader.fieldnames or []):
            raise ValueError("Legacy semantic fields leaked into clean analytical input")
        source_rows = list(reader)

    eligible: list[dict] = []
    for index, row in enumerate(source_rows, start=1):
        if not parse_bool(row["Has_UC_Annotation"]):
            continue
        uid = parse_single_uc(row["UC_References"])
        if uid not in VALID_UCS:
            continue
        if not all((row[c] or "").strip() for c in ["Class", "MethodName", "Signature"]):
            continue
        if row["Class"].strip() not in scaffold:
            continue
        rr = dict(row)
        rr["_source_index"] = index
        rr["_declared_uc"] = uid
        eligible.append(rr)

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    order: list[tuple] = []
    for row in eligible:
        key = behavior_key(row)
        if key not in grouped:
            order.append(key)
        grouped[key].append(row)

    queries: list[dict] = []
    for qn, key in enumerate(order, start=1):
        group = grouped[key]
        targets = {int(r["_declared_uc"]) for r in group}
        if len(targets) != 1:
            raise ValueError(f"Conflicting declared targets in duplicate group {key}: {targets}")
        first = group[0]
        class_canonical = canonical_class(first["Class"])
        qtext = render_query(class_canonical, first["MethodName"], first["Signature"])
        queries.append({
            "QueryID": f"Q{qn:06d}",
            "Model": first["Model"].strip(),
            "Run": int(first["Run"]),
            "File": first["File"],
            "ClassRaw": first["Class"],
            "ClassCanonical": class_canonical,
            "MethodName": first["MethodName"],
            "Signature": first["Signature"],
            "DeclaredUC": next(iter(targets)),
            "CollapsedSourceRows": len(group),
            "FirstSourceRow": first["_source_index"],
            "QueryText": qtext,
        })

    audit = json.loads(PROMPT3_AUDIT.read_text(encoding="utf-8"))
    expected_queries = audit["funnel"]["primary_queries_after_within_run_dedup"]
    expected_cells = audit["model_run_cells_with_primary_queries"]
    cells = {(q["Model"], int(q["Run"])) for q in queries}
    if len(queries) != expected_queries or len(cells) != expected_cells:
        raise ValueError(
            f"Prompt-3 population mismatch: queries {len(queries)}/{expected_queries}, "
            f"cells {len(cells)}/{expected_cells}"
        )
    return queries


def rank_metrics(scores: np.ndarray, declared_uc: int) -> dict:
    if scores.shape != (21,):
        raise ValueError(f"Expected 21 scores, got {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite retrieval score detected")
    ordered_idx = sorted(range(21), key=lambda i: (-float(scores[i]), i + 1))
    ranking = [i + 1 for i in ordered_idx]
    rank = ranking.index(int(declared_uc)) + 1
    max_score = max(float(x) for x in scores)
    exact_ties = [i + 1 for i, x in enumerate(scores) if float(x) == max_score]
    tie_credit = (1.0 / len(exact_ties)) if declared_uc in exact_ties else 0.0
    return {
        "Top1UC": ranking[0],
        "DeclaredRank": rank,
        "Hit1": int(rank <= 1),
        "Hit3": int(rank <= 3),
        "Hit5": int(rank <= 5),
        "ReciprocalRank": 1.0 / rank,
        "TieAwareTop1Credit": tie_credit,
        "Top3UCs": "|".join(str(x) for x in ranking[:3]),
        "Top5UCs": "|".join(str(x) for x in ranking[:5]),
        "FullRanking": "|".join(str(x) for x in ranking),
        "DeclaredScore": float(scores[declared_uc - 1]),
        "Top1Score": float(scores[ranking[0] - 1]),
    }


def build_method_outputs(method: str, scores: np.ndarray, queries: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    query_rows: list[dict] = []
    for i, q in enumerate(queries):
        m = rank_metrics(scores[i], int(q["DeclaredUC"]))
        query_rows.append({
            "QueryID": q["QueryID"],
            "Method": method,
            "Model": q["Model"],
            "Run": q["Run"],
            "DeclaredUC": q["DeclaredUC"],
            **m,
        })

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in query_rows:
        by_cell[(r["Model"], int(r["Run"]))].append(r)

    cell_rows: list[dict] = []
    for (model, run), group in sorted(by_cell.items(), key=lambda x: (x[0][0], x[0][1])):
        cell_rows.append({
            "Method": method,
            "Model": model,
            "Run": run,
            "NQueries": len(group),
            "Hit1Mean": statistics.fmean(float(r["Hit1"]) for r in group),
            "MRRMean": statistics.fmean(float(r["ReciprocalRank"]) for r in group),
            "Hit3Mean": statistics.fmean(float(r["Hit3"]) for r in group),
            "Hit5Mean": statistics.fmean(float(r["Hit5"]) for r in group),
            "TieAwareTop1Mean": statistics.fmean(float(r["TieAwareTop1Credit"]) for r in group),
            "MedianDeclaredRank": statistics.median(int(r["DeclaredRank"]) for r in group),
        })

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in cell_rows:
        by_model[r["Model"]].append(r)
    model_rows: list[dict] = []
    for model, group in sorted(by_model.items()):
        model_rows.append({
            "Method": method,
            "Model": model,
            "EligibleRuns": len(group),
            "SourceRuns": 10,
            "TotalQueries": sum(int(r["NQueries"]) for r in group),
            "Hit1MacroEligibleRuns": statistics.fmean(float(r["Hit1Mean"]) for r in group),
            "MRRMacroEligibleRuns": statistics.fmean(float(r["MRRMean"]) for r in group),
            "Hit3MacroEligibleRuns": statistics.fmean(float(r["Hit3Mean"]) for r in group),
            "Hit5MacroEligibleRuns": statistics.fmean(float(r["Hit5Mean"]) for r in group),
            "TieAwareTop1MacroEligibleRuns": statistics.fmean(float(r["TieAwareTop1Mean"]) for r in group),
        })

    corpus = {
        "Method": method,
        "EligibleModelRunCells": len(cell_rows),
        "ZeroQueryModelRunCells": 90 - len(cell_rows),
        "PrimaryQueries": len(query_rows),
        "Hit1EqualCellMacro": statistics.fmean(float(r["Hit1Mean"]) for r in cell_rows),
        "MRREqualCellMacro": statistics.fmean(float(r["MRRMean"]) for r in cell_rows),
        "Hit3EqualCellMacro": statistics.fmean(float(r["Hit3Mean"]) for r in cell_rows),
        "Hit5EqualCellMacro": statistics.fmean(float(r["Hit5Mean"]) for r in cell_rows),
        "TieAwareTop1EqualCellMacro": statistics.fmean(float(r["TieAwareTop1Mean"]) for r in cell_rows),
        "Hit1MicroSensitivity": statistics.fmean(float(r["Hit1"]) for r in query_rows),
        "MRRMicroSensitivity": statistics.fmean(float(r["ReciprocalRank"]) for r in query_rows),
    }
    return query_rows, cell_rows, model_rows, corpus


def write_score_matrix(path: Path, scores: np.ndarray, queries: list[dict]) -> None:
    fieldnames = ["QueryID"] + [f"UC{i}" for i in VALID_UCS]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for q, row in zip(queries, scores):
            w.writerow([q["QueryID"]] + [format(float(x), ".17g") for x in row])


def full_rank_average(scores: np.ndarray) -> np.ndarray:
    # Higher similarity receives lower rank number; ties receive average ranks.
    return rankdata(-scores, method="average")


def cross_method_consensus(
    tfidf_scores: np.ndarray,
    mpnet_scores: np.ndarray,
    tfidf_query: list[dict],
    mpnet_query: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    tf_by_id = {r["QueryID"]: r for r in tfidf_query}
    mp_by_id = {r["QueryID"]: r for r in mpnet_query}
    if set(tf_by_id) != set(mp_by_id):
        raise ValueError("Cross-method query IDs differ")

    query_rows: list[dict] = []
    for i, qid in enumerate(tf_by_id):
        t = tf_by_id[qid]
        m = mp_by_id[qid]
        t3 = {int(x) for x in str(t["Top3UCs"]).split("|")}
        m3 = {int(x) for x in str(m["Top3UCs"]).split("|")}
        jacc = len(t3 & m3) / len(t3 | m3)
        tr = full_rank_average(tfidf_scores[i])
        mr = full_rank_average(mpnet_scores[i])
        if float(np.std(tr)) == 0.0 or float(np.std(mr)) == 0.0:
            spearman = math.nan
        else:
            spearman = float(np.corrcoef(tr, mr)[0, 1])
        th = int(t["Hit1"])
        mh = int(m["Hit1"])
        category = "both" if th and mh else ("exactly_one" if th or mh else "neither")
        query_rows.append({
            "QueryID": qid,
            "Model": t["Model"],
            "Run": t["Run"],
            "DeclaredUC": t["DeclaredUC"],
            "Top1ExactAgreement": int(int(t["Top1UC"]) == int(m["Top1UC"])),
            "Top3SetJaccard": jacc,
            "Full21SpearmanTieAware": "" if math.isnan(spearman) else spearman,
            "DualMethodDeclaredTop1Category": category,
        })

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in query_rows:
        by_cell[(r["Model"], int(r["Run"]))].append(r)
    cell_rows: list[dict] = []
    for (model, run), group in sorted(by_cell.items(), key=lambda x: (x[0][0], x[0][1])):
        spears = [float(r["Full21SpearmanTieAware"]) for r in group if r["Full21SpearmanTieAware"] != ""]
        cell_rows.append({
            "Model": model,
            "Run": run,
            "NQueries": len(group),
            "Top1ExactAgreementMean": statistics.fmean(float(r["Top1ExactAgreement"]) for r in group),
            "Top3SetJaccardMean": statistics.fmean(float(r["Top3SetJaccard"]) for r in group),
            "Full21SpearmanTieAwareMean": statistics.fmean(spears) if spears else "",
            "SpearmanEvaluableQueries": len(spears),
            "BothDeclaredTop1Share": sum(r["DualMethodDeclaredTop1Category"] == "both" for r in group) / len(group),
            "ExactlyOneDeclaredTop1Share": sum(r["DualMethodDeclaredTop1Category"] == "exactly_one" for r in group) / len(group),
            "NeitherDeclaredTop1Share": sum(r["DualMethodDeclaredTop1Category"] == "neither" for r in group) / len(group),
        })

    cell_spears = [float(r["Full21SpearmanTieAwareMean"]) for r in cell_rows if r["Full21SpearmanTieAwareMean"] != ""]
    corpus = {
        "EligibleModelRunCells": len(cell_rows),
        "PrimaryQueries": len(query_rows),
        "Top1ExactAgreementEqualCellMacro": statistics.fmean(float(r["Top1ExactAgreementMean"]) for r in cell_rows),
        "Top3SetJaccardEqualCellMacro": statistics.fmean(float(r["Top3SetJaccardMean"]) for r in cell_rows),
        "Full21SpearmanTieAwareEqualCellMacro": statistics.fmean(cell_spears) if cell_spears else None,
        "SpearmanEvaluableCells": len(cell_spears),
        "BothDeclaredTop1EqualCellMacro": statistics.fmean(float(r["BothDeclaredTop1Share"]) for r in cell_rows),
        "ExactlyOneDeclaredTop1EqualCellMacro": statistics.fmean(float(r["ExactlyOneDeclaredTop1Share"]) for r in cell_rows),
        "NeitherDeclaredTop1EqualCellMacro": statistics.fmean(float(r["NeitherDeclaredTop1Share"]) for r in cell_rows),
    }
    return query_rows, cell_rows, corpus


def token_length_profile(model: SentenceTransformer, texts: list[str]) -> dict:
    tokenizer = model.tokenizer
    max_len = int(model.max_seq_length)
    lengths: list[int] = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=True, truncation=False, padding=False)
        ids = encoded["input_ids"]
        lengths.append(len(ids))
    return {
        "count": len(lengths),
        "model_max_seq_length": max_len,
        "max_untruncated_wordpieces": max(lengths) if lengths else 0,
        "truncated_if_native_limit_applied": sum(x > max_len for x in lengths),
    }


def main() -> int:
    random.seed(20260818)
    np.random.seed(20260818)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Guard against post-freeze semantic configuration drift.
    protocol_text = PROTOCOL.read_text(encoding="utf-8")
    if "Status: **FROZEN — Prompt 3 protocol/metric freeze**" not in protocol_text:
        raise SystemExit("Prompt-3 protocol is not frozen")
    if "69 Model × Run cells" not in protocol_text or "21 cells" not in protocol_text:
        raise SystemExit("Prompt-3 zero-query-cell amendment is not present")

    preprocess_cfg = yaml.safe_load(PREPROCESS.read_text(encoding="utf-8"))
    retrieval_cfg = yaml.safe_load(RETRIEVAL.read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(SEM_MODELS.read_text(encoding="utf-8"))
    if not (preprocess_cfg.get("frozen") and retrieval_cfg.get("frozen") and model_cfg.get("frozen")):
        raise SystemExit("Prompt-3 semantic configuration is not frozen")

    requirements = parse_uc_spec(UCS.read_text(encoding="utf-8"))
    req_rows = [{
        "UC_ID": r["id"],
        "Title": r["title"],
        "Description": r["description"],
        "MainScenarioStepCount": len(r["main_steps"]),
        "MainScenario": " | ".join(r["main_steps"]),
        "PrimaryText": r["primary_text"],
    } for r in requirements]
    write_csv(
        REQ_OUT,
        ["UC_ID", "Title", "Description", "MainScenarioStepCount", "MainScenario", "PrimaryText"],
        req_rows,
    )

    queries = load_primary_queries()
    write_csv(
        QUERY_OUT,
        [
            "QueryID", "Model", "Run", "File", "ClassRaw", "ClassCanonical",
            "MethodName", "Signature", "DeclaredUC", "CollapsedSourceRows",
            "FirstSourceRow", "QueryText",
        ],
        queries,
    )

    requirement_texts = [r["PrimaryText"] for r in req_rows]
    query_texts = [q["QueryText"] for q in queries]

    # ------------------------------------------------------------------
    # Primary lexical method: frozen TF-IDF configuration.
    # ------------------------------------------------------------------
    lex = retrieval_cfg["lexical_primary"]
    vectorizer = TfidfVectorizer(
        analyzer=lex["analyzer"],
        ngram_range=(int(lex["ngram_min"]), int(lex["ngram_max"])),
        lowercase=bool(lex["lowercase"]),
        min_df=int(lex["min_df"]),
        max_df=float(lex["max_df"]),
        sublinear_tf=bool(lex["sublinear_tf"]),
        smooth_idf=bool(lex["smooth_idf"]),
        norm=lex["norm"],
        stop_words=lex.get("stop_words"),
    )
    req_tfidf = vectorizer.fit_transform(requirement_texts)
    query_tfidf = vectorizer.transform(query_texts)
    tfidf_scores = (query_tfidf @ req_tfidf.T).toarray()
    if tfidf_scores.shape != (2369, 21):
        raise SystemExit(f"Unexpected TF-IDF score shape: {tfidf_scores.shape}")

    # ------------------------------------------------------------------
    # Primary neural method: frozen MPNet revision, CPU deterministic path.
    # ------------------------------------------------------------------
    primary_neural = next(x for x in model_cfg["models"] if x["role"] == "primary_neural")
    model_id = str(primary_neural["model_id"])
    requested_revision = str(primary_neural["revision"])
    api_info = HfApi().model_info(repo_id=model_id, revision=requested_revision)
    resolved_revision = str(api_info.sha)
    if resolved_revision != requested_revision:
        raise SystemExit(
            f"Frozen model revision did not resolve exactly: requested={requested_revision}, resolved={resolved_revision}"
        )

    import torch
    torch.manual_seed(20260818)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)

    mpnet = SentenceTransformer(model_id, revision=requested_revision, device="cpu")
    expected_max = int(primary_neural["expected_default_max_wordpieces"])
    if int(mpnet.max_seq_length) != expected_max:
        raise SystemExit(
            f"Frozen MPNet max sequence length mismatch: expected={expected_max}, observed={mpnet.max_seq_length}"
        )
    req_token_profile = token_length_profile(mpnet, requirement_texts)
    query_token_profile = token_length_profile(mpnet, query_texts)

    req_emb = mpnet.encode(
        requirement_texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_emb = mpnet.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    mpnet_scores = np.asarray(query_emb @ req_emb.T, dtype=np.float64)
    if mpnet_scores.shape != (2369, 21):
        raise SystemExit(f"Unexpected MPNet score shape: {mpnet_scores.shape}")

    # ------------------------------------------------------------------
    # Frozen ranking metrics and aggregation.
    # ------------------------------------------------------------------
    tf_q, tf_cell, tf_model, tf_corpus = build_method_outputs("TFIDF", tfidf_scores, queries)
    mp_q, mp_cell, mp_model, mp_corpus = build_method_outputs("MPNET", mpnet_scores, queries)

    query_fields = [
        "QueryID", "Method", "Model", "Run", "DeclaredUC", "Top1UC",
        "DeclaredRank", "Hit1", "Hit3", "Hit5", "ReciprocalRank",
        "TieAwareTop1Credit", "Top3UCs", "Top5UCs", "FullRanking",
        "DeclaredScore", "Top1Score",
    ]
    cell_fields = [
        "Method", "Model", "Run", "NQueries", "Hit1Mean", "MRRMean",
        "Hit3Mean", "Hit5Mean", "TieAwareTop1Mean", "MedianDeclaredRank",
    ]
    model_fields = [
        "Method", "Model", "EligibleRuns", "SourceRuns", "TotalQueries",
        "Hit1MacroEligibleRuns", "MRRMacroEligibleRuns", "Hit3MacroEligibleRuns",
        "Hit5MacroEligibleRuns", "TieAwareTop1MacroEligibleRuns",
    ]
    write_csv(ART / "tfidf_query_metrics.csv", query_fields, tf_q)
    write_csv(ART / "mpnet_query_metrics.csv", query_fields, mp_q)
    write_csv(ART / "model_run_primary_metrics.csv", cell_fields, tf_cell + mp_cell)
    write_csv(ART / "by_model_primary_metrics.csv", model_fields, tf_model + mp_model)
    write_score_matrix(ART / "tfidf_scores.csv", tfidf_scores, queries)
    write_score_matrix(ART / "mpnet_scores.csv", mpnet_scores, queries)

    corpus_metrics = {
        "status": "PROMPT4_PRIMARY_RETRIEVAL_EXECUTED",
        "prompt3_merge_commit": PROMPT3_MERGE_COMMIT,
        "primary_population_queries": len(queries),
        "eligible_model_run_cells": 69,
        "zero_query_model_run_cells": 21,
        "methods": {"TFIDF": tf_corpus, "MPNET": mp_corpus},
    }
    (ART / "corpus_primary_metrics.json").write_text(
        json.dumps(corpus_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Frozen primary cross-method consensus.
    # ------------------------------------------------------------------
    cm_q, cm_cell, cm_corpus = cross_method_consensus(tfidf_scores, mpnet_scores, tf_q, mp_q)
    write_csv(
        ART / "cross_method_query_consensus.csv",
        [
            "QueryID", "Model", "Run", "DeclaredUC", "Top1ExactAgreement",
            "Top3SetJaccard", "Full21SpearmanTieAware",
            "DualMethodDeclaredTop1Category",
        ],
        cm_q,
    )
    write_csv(
        ART / "cross_method_cell_consensus.csv",
        [
            "Model", "Run", "NQueries", "Top1ExactAgreementMean",
            "Top3SetJaccardMean", "Full21SpearmanTieAwareMean",
            "SpearmanEvaluableQueries", "BothDeclaredTop1Share",
            "ExactlyOneDeclaredTop1Share", "NeitherDeclaredTop1Share",
        ],
        cm_cell,
    )
    (ART / "cross_method_corpus_consensus.json").write_text(
        json.dumps(cm_corpus, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    runtime_manifest = {
        "status": "PROMPT4_RUNTIME_MANIFEST",
        "prompt3_merge_commit": PROMPT3_MERGE_COMMIT,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": pkg_version("numpy"),
            "scipy": pkg_version("scipy"),
            "scikit-learn": pkg_version("scikit-learn"),
            "sentence-transformers": pkg_version("sentence-transformers"),
            "transformers": pkg_version("transformers"),
            "torch": pkg_version("torch"),
            "huggingface-hub": pkg_version("huggingface-hub"),
            "PyYAML": pkg_version("PyYAML"),
        },
        "primary_neural": {
            "model_id": model_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "device": "cpu",
            "normalize_embeddings": True,
            "model_max_seq_length": int(mpnet.max_seq_length),
            "requirement_token_profile": req_token_profile,
            "query_token_profile": query_token_profile,
        },
        "tfidf": {
            "vocabulary_size": len(vectorizer.vocabulary_),
            "requirement_matrix_shape": list(req_tfidf.shape),
            "query_matrix_shape": list(query_tfidf.shape),
        },
        "execution_scope": {
            "primary_tfidf_executed": True,
            "primary_mpnet_executed": True,
            "cross_method_primary_consensus_executed": True,
            "minilm_sensitivity_executed": False,
            "negative_control_executed": False,
            "run_resampling_executed": False,
            "rq3_fragmentation_executed": False,
        },
        "input_hashes": {
            "clean_behaviors_sha256": sha256(CLEAN),
            "ucs_sha256": sha256(UCS),
            "scaffold_sha256": sha256(SCAFFOLD),
            "preprocessing_config_sha256": sha256(PREPROCESS),
            "retrieval_config_sha256": sha256(RETRIEVAL),
            "semantic_models_config_sha256": sha256(SEM_MODELS),
        },
    }
    (ART / "runtime_manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    handoff = f"""# HANDOFF 04 — PRIMARY SEMANTIC RETRIEVAL\n\n## STATUS\n\n**PASS — pending independent Prompt-4 validator**\n\n## Scope executed\n\nPrompt 4 executed the Prompt-3-frozen primary lexical and primary neural semantic retrieval only, plus the frozen primary cross-method consensus. No sensitivity encoder, representation sensitivity, negative control, resampling, or RQ3 fragmentation analysis was run.\n\n## Frozen population reproduced\n\n- Primary behavior queries: **{len(queries)}**\n- Eligible Model × Run cells: **69/90**\n- Zero-query Model × Run cells: **21/90**, retained as NA for semantic metrics.\n- Requirement candidates: **21 (UC1–UC21)**\n\n## Primary retrieval results\n\n### TF-IDF\n- Equal-cell macro Hit@1: **{tf_corpus['Hit1EqualCellMacro']:.6f}**\n- Equal-cell macro MRR: **{tf_corpus['MRREqualCellMacro']:.6f}**\n- Equal-cell macro Hit@3: **{tf_corpus['Hit3EqualCellMacro']:.6f}**\n- Equal-cell macro Hit@5: **{tf_corpus['Hit5EqualCellMacro']:.6f}**\n\n### MPNet\n- Exact model revision: `{resolved_revision}`\n- Equal-cell macro Hit@1: **{mp_corpus['Hit1EqualCellMacro']:.6f}**\n- Equal-cell macro MRR: **{mp_corpus['MRREqualCellMacro']:.6f}**\n- Equal-cell macro Hit@3: **{mp_corpus['Hit3EqualCellMacro']:.6f}**\n- Equal-cell macro Hit@5: **{mp_corpus['Hit5EqualCellMacro']:.6f}**\n- Requirement texts exceeding native max length: **{req_token_profile['truncated_if_native_limit_applied']}/{req_token_profile['count']}**\n- Query texts exceeding native max length: **{query_token_profile['truncated_if_native_limit_applied']}/{query_token_profile['count']}**\n\n## Cross-method primary consensus\n\n- Equal-cell macro exact Top-1 UC agreement: **{cm_corpus['Top1ExactAgreementEqualCellMacro']:.6f}**\n- Equal-cell macro Top-3 set Jaccard: **{cm_corpus['Top3SetJaccardEqualCellMacro']:.6f}**\n- Equal-cell macro tie-aware full-rank Spearman: **{cm_corpus['Full21SpearmanTieAwareEqualCellMacro'] if cm_corpus['Full21SpearmanTieAwareEqualCellMacro'] is not None else 'NA'}**\n- Dual-method declared-UC Top-1: both **{cm_corpus['BothDeclaredTop1EqualCellMacro']:.6f}**, exactly one **{cm_corpus['ExactlyOneDeclaredTop1EqualCellMacro']:.6f}**, neither **{cm_corpus['NeitherDeclaredTop1EqualCellMacro']:.6f}**.\n\n## Interpretation boundary\n\nThese are automated retrieval-concordance and cross-method agreement results. They are not semantic correctness, accuracy, precision, recall, a gold standard, or evidence of causal trace provenance. Zero-query runs are structural absence/NA, not semantic failures.\n\n## Prohibited inputs\n\n`Best_Match_Action`, `SimilarityScore`, FEER/voting references, expert mappings, LLM-judge labels, and `UC_Action` in query construction were not used.\n\n## Gate\n\nThe generated numerical results must pass `scripts/04_validate_primary_semantic_retrieval.py` and unit tests before Prompt 4 is accepted.\n"""
    (REPORTS / "HANDOFF_04_PRIMARY_SEMANTIC_RETRIEVAL.md").write_text(handoff, encoding="utf-8")

    summary = {
        "status": "PASS_PENDING_VALIDATOR",
        "queries": len(queries),
        "eligible_cells": 69,
        "tfidf_hit1_macro": tf_corpus["Hit1EqualCellMacro"],
        "tfidf_mrr_macro": tf_corpus["MRREqualCellMacro"],
        "mpnet_hit1_macro": mp_corpus["Hit1EqualCellMacro"],
        "mpnet_mrr_macro": mp_corpus["MRREqualCellMacro"],
        "top1_cross_method_macro": cm_corpus["Top1ExactAgreementEqualCellMacro"],
        "mpnet_requirement_truncations": req_token_profile["truncated_if_native_limit_applied"],
        "mpnet_query_truncations": query_token_profile["truncated_if_native_limit_applied"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
