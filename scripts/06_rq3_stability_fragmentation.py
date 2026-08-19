from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "derived/behaviors/source_without_legacy_semantics.csv"
SCAFFOLD = ROOT / "input_snapshot/provenance/Methodless.txt"
PROMPT4_QUERIES = ROOT / "derived/behaviors/primary_semantic_queries.csv"
PROTOCOL = ROOT / "STUDY_PROTOCOL.md"
OUT = ROOT / "artifacts/rq3_stability_fragmentation"
REPORT = ROOT / "reports/RQ3_STABILITY_FRAGMENTATION.md"
HANDOFF = ROOT / "reports/HANDOFF_06_RQ3_STABILITY_FRAGMENTATION.md"

MODELS = [
    "ChatGPT-4o", "ChatGPT-o3", "Claude3.7", "DeepSeek(R1)",
    "Gemini-2.5-Pro-Preview-05-06", "Grok3", "Llama4", "Mistral", "Qwen3",
]
UCS = list(range(1, 22))
RUNS = list(range(1, 11))


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def parse_bool(v):
    x = (v or "").strip().casefold()
    return True if x == "true" else False if x == "false" else None


def parse_uc(v):
    x = (v or "").strip()
    if re.fullmatch(r"\d+", x): return int(x)
    m = re.fullmatch(r"(?i:UC)\s*(\d+)", x)
    return int(m.group(1)) if m else None


def scaffold_identities(text: str):
    ids, aliases = set(), {}
    for raw in text.splitlines():
        s = raw.strip()
        m = re.match(r'^(?:abstract\s+)?class\s+"([^"]+)"\s+as\s+([A-Za-z_]\w*)', s)
        if m:
            display, alias = m.group(1), m.group(2)
            ids.update([display, alias]); aliases[display] = alias; continue
        m = re.match(r"^(?:abstract\s+)?class\s+([A-Za-z_]\w*)", s)
        if m: ids.add(m.group(1))
    return ids, aliases


def norm_sig(v):
    x = (v or "").strip().casefold()
    x = re.sub(r"\s+", " ", x)
    return re.sub(r"\s*([(),:+\-])\s*", r"\1", x)


def behavior_id(cls, method, sig):
    return f"{cls}::{method.strip().casefold()}::{norm_sig(sig)}"


def jaccard(a, b):
    u = a | b
    return None if not u else len(a & b) / len(u)


def qtile(vals, q):
    vals = sorted(vals)
    if not vals: return None
    if len(vals) == 1: return vals[0]
    pos = (len(vals)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    return vals[lo] if lo==hi else vals[lo]*(hi-pos)+vals[hi]*(pos-lo)


def summary(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals: return {k: None for k in ["n","mean","median","q1","q3","min","max"]}
    return {"n":len(vals),"mean":sum(vals)/len(vals),"median":statistics.median(vals),
            "q1":qtile(vals,.25),"q3":qtile(vals,.75),"min":min(vals),"max":max(vals)}


def entropy(counts):
    n = sum(counts.values()); k = len(counts)
    if n == 0: return None
    if k == 1: return 0.0
    h = -sum((c/n)*math.log(c/n) for c in counts.values())
    return h / math.log(k)


def fmt(v): return "NA" if v is None else f"{float(v):.6f}"


def main():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    if "## RQ3 stability and fragmentation" not in protocol:
        raise SystemExit("Frozen RQ3 protocol not found")
    amendments = (ROOT / "PROTOCOL_AMENDMENTS.md").read_text(encoding="utf-8")
    if "Exact-behavior Jaccard" not in amendments or "normalized Shannon entropy" not in amendments:
        raise SystemExit("Frozen RQ3 edge rules not found")

    rows = read_csv(CLEAN)
    if len(rows) != 3373: raise SystemExit("Unexpected clean-row count")
    if any(x in rows[0] for x in ["Best_Match_Action","SimilarityScore"]):
        raise SystemExit("Prohibited legacy fields present")
    scaffold, aliases = scaffold_identities(SCAFFOLD.read_text(encoding="utf-8"))
    if len(scaffold) != 21 or aliases != {"RoleManager <<service>>":"RoleManager"}:
        raise SystemExit("Unexpected scaffold identities")

    # Materialize frozen RQ3 population and collapse exact within-run behaviors.
    groups = defaultdict(list)
    for idx, r in enumerate(rows, 1):
        if parse_bool(r["Has_UC_Annotation"]) is not True: continue
        uc = parse_uc(r["UC_References"])
        if uc not in UCS: continue
        if not all((r[c] or "").strip() for c in ["Class","MethodName","Signature"]): continue
        raw_cls = r["Class"].strip()
        if raw_cls not in scaffold: continue
        cls = aliases.get(raw_cls, raw_cls)
        bid = behavior_id(cls, r["MethodName"], r["Signature"])
        groups[(r["Model"].strip(), int(r["Run"]), bid)].append((uc, cls, idx))

    conflicts = []
    queries = []
    for (model, run, bid), members in sorted(groups.items()):
        ucs = {m[0] for m in members}; classes = {m[1] for m in members}
        if len(ucs) != 1 or len(classes) != 1:
            conflicts.append((model,run,bid,sorted(ucs),sorted(classes)))
            continue
        queries.append({"Model":model,"Run":run,"UC":next(iter(ucs)),"Class":next(iter(classes)),"BehaviorID":bid,"CollapsedRows":len(members)})
    if conflicts: raise SystemExit(f"Duplicate trace conflicts found: {conflicts[:3]}")
    if len(queries) != 2369: raise SystemExit(f"Expected 2369 RQ3 queries, got {len(queries)}")

    # Exact identity reconciliation with Prompt 4.
    p4 = read_csv(PROMPT4_QUERIES)
    p4keys={(r['Model'],int(r['Run']),int(r['DeclaredUC']),behavior_id(r['ClassCanonical'],r['MethodName'],r['Signature'])) for r in p4}
    qkeys={(r['Model'],r['Run'],r['UC'],r['BehaviorID']) for r in queries}
    if qkeys != p4keys: raise SystemExit(f"Prompt4/RQ3 query identity mismatch: {len(qkeys^p4keys)}")

    run_sets = {(m,r,u):set() for m in MODELS for r in RUNS for u in UCS}
    owner_counts = {(m,r,u):Counter() for m in MODELS for r in RUNS for u in UCS}
    for q in queries:
        k=(q['Model'],q['Run'],q['UC']); run_sets[k].add(q['BehaviorID']); owner_counts[k][q['Class']]+=1

    owner_rows=[]
    for m in MODELS:
        for r in RUNS:
            for u in UCS:
                counts=owner_counts[(m,r,u)]; n=sum(counts.values()); k=len(counts)
                if n:
                    maxn=max(counts.values()); dom=sorted(c for c,v in counts.items() if v==maxn)[0]
                    share=maxn/n; ent=entropy(counts)
                else: dom=""; share=None; ent=None
                owner_rows.append({"Model":m,"Run":r,"UC":f"UC{u}","BehaviorCount":n,"OwnerClassCount":k,
                                   "DominantClass":dom,"DominantClassShare":"" if share is None else share,
                                   "NormalizedShannonEntropy":"" if ent is None else ent})

    model_uc_rows=[]
    recurrence_rows=[]
    model_unions={}
    model_support2={}
    for m in MODELS:
        for u in UCS:
            sets=[run_sets[(m,r,u)] for r in RUNS]
            js=[]; empty_pairs=0
            for i,j in combinations(range(10),2):
                val=jaccard(sets[i],sets[j])
                if val is None: empty_pairs+=1
                else: js.append(val)
            union=set().union(*sets)
            inter=set(sets[0])
            for s in sets[1:]: inter &= s
            support=Counter(b for s in sets for b in s)
            support_counts={k:sum(1 for v in support.values() if v==k) for k in RUNS}
            for k in RUNS:
                recurrence_rows.append({"Model":m,"UC":f"UC{u}","SupportRuns":k,"BehaviorCount":support_counts[k]})
            observed=sum(bool(s) for s in sets)
            singleton=support_counts[1]; recurrent=sum(v for k,v in support_counts.items() if k>=5)
            owner_obs=[x for x in owner_rows if x['Model']==m and x['UC']==f"UC{u}" and x['BehaviorCount']>0]
            dom_vals=[float(x['DominantClassShare']) for x in owner_obs]
            ent_vals=[float(x['NormalizedShannonEntropy']) for x in owner_obs]
            all_owner=Counter()
            for r in RUNS: all_owner.update(owner_counts[(m,r,u)])
            model_unions[(m,u)] = union
            model_support2[(m,u)] = {b for b,c in support.items() if c>=2}
            model_uc_rows.append({
                "Model":m,"UC":f"UC{u}","RunsWithBehavior":observed,"EvaluableRunPairs":len(js),"EmptyEmptyRunPairs":empty_pairs,
                "MeanPairwiseJaccard":"" if not js else sum(js)/len(js),"MedianPairwiseJaccard":"" if not js else statistics.median(js),
                "UnionBehaviorCount":len(union),"IntersectionAll10RunsCount":len(inter),"SingletonBehaviorCount":singleton,
                "SingletonShare":"" if not union else singleton/len(union),"RecurrentSupportGE5Count":recurrent,
                "RecurrentSupportGE5Share":"" if not union else recurrent/len(union),"DistinctOwnerClassesAcrossRuns":len(all_owner),
                "ObservedOwnerRunCells":len(owner_obs),"MeanDominantClassShare":"" if not dom_vals else sum(dom_vals)/len(dom_vals),
                "MeanNormalizedOwnerEntropy":"" if not ent_vals else sum(ent_vals)/len(ent_vals)
            })

    pair_rows=[]; pair2_rows=[]; uc_cross_rows=[]
    for u in UCS:
        vals=[]; vals2=[]; empty=0; empty2=0
        for a,b in combinations(MODELS,2):
            v=jaccard(model_unions[(a,u)],model_unions[(b,u)])
            v2=jaccard(model_support2[(a,u)],model_support2[(b,u)])
            if v is None: empty+=1
            else: vals.append(v)
            if v2 is None: empty2+=1
            else: vals2.append(v2)
            pair_rows.append({"UC":f"UC{u}","ModelA":a,"ModelB":b,"Jaccard":"" if v is None else v})
            pair2_rows.append({"UC":f"UC{u}","ModelA":a,"ModelB":b,"JaccardSupportGE2":"" if v2 is None else v2})
        global_beh=set().union(*(model_unions[(m,u)] for m in MODELS))
        uc_q=[q for q in queries if q['UC']==u]
        owners={q['Class'] for q in uc_q}
        uc_cross_rows.append({"UC":f"UC{u}","QueryOccurrences":len(uc_q),"DistinctBehaviors":len(global_beh),"DistinctOwnerClasses":len(owners),
                              "ModelsWithBehavior":sum(bool(model_unions[(m,u)]) for m in MODELS),
                              "RunsWithBehavior":sum(bool(run_sets[(m,r,u)]) for m in MODELS for r in RUNS),
                              "CrossModelEvaluablePairs":len(vals),"CrossModelEmptyEmptyPairs":empty,
                              "MeanCrossModelJaccard":"" if not vals else sum(vals)/len(vals),
                              "MedianCrossModelJaccard":"" if not vals else statistics.median(vals),
                              "SupportGE2EvaluablePairs":len(vals2),"SupportGE2EmptyEmptyPairs":empty2,
                              "MeanCrossModelJaccardSupportGE2":"" if not vals2 else sum(vals2)/len(vals2)})

    # Global descriptive summaries, without inferential tests or generator ranking.
    eval_muc=[r for r in model_uc_rows if r['MeanPairwiseJaccard']!=""]
    owner_obs=[r for r in owner_rows if r['BehaviorCount']>0]
    summary_json={
        "status":"PASS_PENDING_VALIDATOR",
        "population":{"clean_rows":3373,"rq3_primary_queries":len(queries),"model_run_uc_cells":len(owner_rows),"model_uc_units":len(model_uc_rows)},
        "identity":{"prompt4_query_identity_match":True,"duplicate_reference_conflicts":0},
        "run_stability":{"model_uc_units_with_evaluable_pairs":len(eval_muc),
                         "mean_of_model_uc_mean_jaccards":summary([float(r['MeanPairwiseJaccard']) for r in eval_muc]),
                         "total_evaluable_run_pairs":sum(int(r['EvaluableRunPairs']) for r in model_uc_rows),
                         "total_empty_empty_run_pairs":sum(int(r['EmptyEmptyRunPairs']) for r in model_uc_rows)},
        "fragmentation":{"global_distinct_behavior_identities":len({q['BehaviorID'] for q in queries}),
                         "union_behavior_count_distribution_by_model_uc":summary([int(r['UnionBehaviorCount']) for r in model_uc_rows if int(r['UnionBehaviorCount'])>0]),
                         "singleton_share_distribution":summary([float(r['SingletonShare']) for r in model_uc_rows if r['SingletonShare']!=""]),
                         "recurrent_ge5_share_distribution":summary([float(r['RecurrentSupportGE5Share']) for r in model_uc_rows if r['RecurrentSupportGE5Share']!=""])},
        "ownership":{"observed_model_run_uc_cells":len(owner_obs),
                     "dominant_share_distribution":summary([float(r['DominantClassShare']) for r in owner_obs]),
                     "normalized_entropy_distribution":summary([float(r['NormalizedShannonEntropy']) for r in owner_obs])},
        "cross_model":{"uc_count":21,"primary_mean_jaccard_distribution":summary([float(r['MeanCrossModelJaccard']) for r in uc_cross_rows if r['MeanCrossModelJaccard']!=""]),
                       "support_ge2_mean_jaccard_distribution":summary([float(r['MeanCrossModelJaccardSupportGE2']) for r in uc_cross_rows if r['MeanCrossModelJaccardSupportGE2']!=""])},
        "analysis_boundary":{"semantic_outcomes_consumed":False,"inferential_tests_run":False,"generator_quality_ranking":False,
                             "Best_Match_Action_used":False,"SimilarityScore_used":False,"UC_Action_used":False}
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT/"model_uc_stability_fragmentation.csv", list(model_uc_rows[0]), model_uc_rows)
    write_csv(OUT/"model_run_uc_owner_metrics.csv", list(owner_rows[0]), owner_rows)
    write_csv(OUT/"model_uc_recurrence_distribution.csv", list(recurrence_rows[0]), recurrence_rows)
    write_csv(OUT/"cross_model_uc_pairwise_jaccard.csv", list(pair_rows[0]), pair_rows)
    write_csv(OUT/"cross_model_uc_pairwise_jaccard_support_ge2.csv", list(pair2_rows[0]), pair2_rows)
    write_csv(OUT/"uc_fragmentation_cross_model_summary.csv", list(uc_cross_rows[0]), uc_cross_rows)
    (OUT/"rq3_summary.json").write_text(json.dumps(summary_json,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    # Publication-oriented descriptive report.
    top_fragmented=sorted(uc_cross_rows,key=lambda r:(-int(r['DistinctBehaviors']),int(r['UC'][2:])))[:5]
    highest_cross=sorted([r for r in uc_cross_rows if r['MeanCrossModelJaccard']!=""],key=lambda r:(-float(r['MeanCrossModelJaccard']),int(r['UC'][2:])))[:5]
    lowest_cross=sorted([r for r in uc_cross_rows if r['MeanCrossModelJaccard']!=""],key=lambda r:(float(r['MeanCrossModelJaccard']),int(r['UC'][2:])))[:5]
    rep=summary_json['run_stability']['mean_of_model_uc_mean_jaccards']
    ent=summary_json['ownership']['normalized_entropy_distribution']
    lines=["# RQ3 — Stability and Fragmentation","",
           "## Scope",f"The frozen RQ3 population contains **{len(queries):,}** deduplicated on-scaffold behaviors with valid UC1–UC21 traces, exactly matching the Prompt-4 primary query identity set.","",
           "## Run-to-run exact-behavior stability",
           f"Model×UC units with at least one evaluable run pair: **{len(eval_muc)}/189**. The descriptive mean of Model×UC mean pairwise Jaccards is **{fmt(rep['mean'])}** (median **{fmt(rep['median'])}**). Empty–empty run pairs are NA by the frozen protocol.","",
           "## Fragmentation and ownership",
           f"Distinct exact behavior identities across the RQ3 population: **{summary_json['fragmentation']['global_distinct_behavior_identities']}**. Observed Model×Run×UC owner cells: **{len(owner_obs)}/1,890**. Mean normalized owner entropy across observed cells is **{fmt(ent['mean'])}** (median **{fmt(ent['median'])}**).", "",
           "Most fragmented UCs by distinct exact behaviors: " + ", ".join(f"{r['UC']} ({r['DistinctBehaviors']})" for r in top_fragmented) + ".", "",
           "Highest descriptive cross-model exact-behavior overlap: " + ", ".join(f"{r['UC']} ({float(r['MeanCrossModelJaccard']):.3f})" for r in highest_cross) + ".", "",
           "Lowest descriptive cross-model exact-behavior overlap: " + ", ".join(f"{r['UC']} ({float(r['MeanCrossModelJaccard']):.3f})" for r in lowest_cross) + ".", "",
           "## Interpretation boundary","Exact-name/signature fragmentation is not semantic disagreement. Jaccard and owner-concentration measures describe reproducibility and responsibility allocation within this frozen corpus; they do not establish semantic correctness or universal generator quality."]
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    HANDOFF.write_text("\n".join(["# HANDOFF 06 — RQ3 STABILITY AND FRAGMENTATION","","## STATUS","","**PASS — pending independent Prompt-6 validator**","","## Scope","",f"Frozen RQ3 analysis executed on **{len(queries):,}** exact primary behavior queries with no semantic-score consumption.","","## Gate","","Acceptance requires the independent validator, regression tests, deterministic rerun, and scientific-boundary checks to pass."])+"\n",encoding="utf-8")
    print(json.dumps({"queries":len(queries),"model_uc_evaluable":len(eval_muc),"distinct_behaviors":summary_json['fragmentation']['global_distinct_behavior_identities'],"owner_cells":len(owner_obs),"mean_model_uc_jaccard":rep['mean']},indent=2))

if __name__ == "__main__": main()
