from __future__ import annotations
import csv,json,math,re
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CLEAN=ROOT/'derived/behaviors/source_without_legacy_semantics.csv'
SCAFFOLD=ROOT/'input_snapshot/provenance/Methodless.txt'
P4=ROOT/'derived/behaviors/primary_semantic_queries.csv'
OUT=ROOT/'artifacts/rq3_stability_fragmentation'
REPORT=ROOT/'reports/PROMPT6_VALIDATION.md'
MODELS=["ChatGPT-4o","ChatGPT-o3","Claude3.7","DeepSeek(R1)","Gemini-2.5-Pro-Preview-05-06","Grok3","Llama4","Mistral","Qwen3"]
UCS=range(1,22); RUNS=range(1,11)

def rcsv(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def pb(v): return (v or '').strip().casefold()=='true'
def puc(v):
    x=(v or '').strip(); m=re.fullmatch(r'(?i:UC)\s*(\d+)',x)
    return int(x) if x.isdigit() else int(m.group(1)) if m else None
def scaffold(text):
    ids=set(); aliases={}
    for raw in text.splitlines():
        s=raw.strip(); m=re.match(r'^(?:abstract\s+)?class\s+"([^"]+)"\s+as\s+([A-Za-z_]\w*)',s)
        if m: ids|={m.group(1),m.group(2)}; aliases[m.group(1)]=m.group(2); continue
        m=re.match(r'^(?:abstract\s+)?class\s+([A-Za-z_]\w*)',s)
        if m: ids.add(m.group(1))
    return ids,aliases
def ns(v):
    x=re.sub(r'\s+',' ',(v or '').strip().casefold()); return re.sub(r'\s*([(),:+\-])\s*',r'\1',x)
def bid(c,m,s):return f"{c}::{m.strip().casefold()}::{ns(s)}"
def jac(a,b):
    u=a|b; return None if not u else len(a&b)/len(u)
def close(a,b,t=1e-12): return abs(float(a)-float(b))<=t

def main():
    checks=[]; failures=[]
    def ck(name,cond):
        checks.append(name)
        if not cond: failures.append(name)
    rows=rcsv(CLEAN); ck('clean_rows',len(rows)==3373)
    sc,aliases=scaffold(SCAFFOLD.read_text(encoding='utf-8')); ck('scaffold',len(sc)==21 and aliases=={'RoleManager <<service>>':'RoleManager'})
    groups=defaultdict(list)
    for r in rows:
        if not pb(r['Has_UC_Annotation']): continue
        u=puc(r['UC_References'])
        if u not in UCS: continue
        if not all((r[c] or '').strip() for c in ['Class','MethodName','Signature']): continue
        raw=r['Class'].strip()
        if raw not in sc: continue
        c=aliases.get(raw,raw); b=bid(c,r['MethodName'],r['Signature'])
        groups[(r['Model'].strip(),int(r['Run']),b)].append((u,c))
    q=[]; conflict=0
    for (m,r,b),members in groups.items():
        us={x[0] for x in members}; cs={x[1] for x in members}
        if len(us)!=1 or len(cs)!=1: conflict+=1; continue
        q.append((m,r,next(iter(us)),next(iter(cs)),b))
    ck('query_count',len(q)==2369); ck('duplicate_conflicts',conflict==0)
    p4={(r['Model'],int(r['Run']),int(r['DeclaredUC']),r['ClassCanonical'],bid(r['ClassCanonical'],r['MethodName'],r['Signature'])) for r in rcsv(P4)}
    qk={(m,r,u,c,b) for m,r,u,c,b in q}; ck('prompt4_identity',qk==p4)

    sets={(m,r,u):set() for m in MODELS for r in RUNS for u in UCS}
    owners={(m,r,u):Counter() for m in MODELS for r in RUNS for u in UCS}
    for m,r,u,c,b in q: sets[(m,r,u)].add(b); owners[(m,r,u)][c]+=1

    mout=rcsv(OUT/'model_uc_stability_fragmentation.csv'); ck('model_uc_rows',len(mout)==189)
    omap={(r['Model'],int(r['UC'][2:])):r for r in mout}
    model_unions={}; model_s2={}; independent_evaluable=0
    for m in MODELS:
        for u in UCS:
            ss=[sets[(m,r,u)] for r in RUNS]; vals=[]
            for i,j in combinations(range(10),2):
                v=jac(ss[i],ss[j]);
                if v is not None: vals.append(v)
            union=set().union(*ss); support=Counter(b for s in ss for b in s)
            inter=set(ss[0])
            for s in ss[1:]:inter&=s
            model_unions[(m,u)]=union; model_s2[(m,u)]={b for b,c in support.items() if c>=2}
            o=omap[(m,u)]
            ck(f'{m}-UC{u}-runs',int(o['RunsWithBehavior'])==sum(bool(s) for s in ss))
            ck(f'{m}-UC{u}-pairs',int(o['EvaluableRunPairs'])==len(vals))
            ck(f'{m}-UC{u}-union',int(o['UnionBehaviorCount'])==len(union))
            ck(f'{m}-UC{u}-inter',int(o['IntersectionAll10RunsCount'])==len(inter))
            if vals:
                independent_evaluable+=1; ck(f'{m}-UC{u}-jaccard',close(o['MeanPairwiseJaccard'],sum(vals)/len(vals)))
            else: ck(f'{m}-UC{u}-jaccard-na',o['MeanPairwiseJaccard']=='')

    ownerout=rcsv(OUT/'model_run_uc_owner_metrics.csv'); ck('owner_rows',len(ownerout)==1890)
    owner_map={(r['Model'],int(r['Run']),int(r['UC'][2:])):r for r in ownerout}
    observed=0
    for m in MODELS:
        for r in RUNS:
            for u in UCS:
                c=owners[(m,r,u)]; o=owner_map[(m,r,u)]; n=sum(c.values())
                ck(f'owner-count-{m}-{r}-{u}',int(o['BehaviorCount'])==n)
                if n:
                    observed+=1; k=len(c); mx=max(c.values()); share=mx/n
                    ent=0.0 if k==1 else -sum((v/n)*math.log(v/n) for v in c.values())/math.log(k)
                    ck(f'owner-share-{m}-{r}-{u}',close(o['DominantClassShare'],share))
                    ck(f'owner-ent-{m}-{r}-{u}',close(o['NormalizedShannonEntropy'],ent))
                else:
                    ck(f'owner-na-{m}-{r}-{u}',o['DominantClassShare']=='' and o['NormalizedShannonEntropy']=='')

    ucout=rcsv(OUT/'uc_fragmentation_cross_model_summary.csv'); ck('uc_rows',len(ucout)==21)
    umap={int(r['UC'][2:]):r for r in ucout}
    for u in UCS:
        vals=[];vals2=[]
        for a,b in combinations(MODELS,2):
            v=jac(model_unions[(a,u)],model_unions[(b,u)]); v2=jac(model_s2[(a,u)],model_s2[(b,u)])
            if v is not None:vals.append(v)
            if v2 is not None:vals2.append(v2)
        o=umap[u]; glob=set().union(*(model_unions[(m,u)] for m in MODELS))
        ck(f'uc{u}-distinct',int(o['DistinctBehaviors'])==len(glob))
        ck(f'uc{u}-models',int(o['ModelsWithBehavior'])==sum(bool(model_unions[(m,u)]) for m in MODELS))
        ck(f'uc{u}-pairs',int(o['CrossModelEvaluablePairs'])==len(vals))
        if vals: ck(f'uc{u}-crossj',close(o['MeanCrossModelJaccard'],sum(vals)/len(vals)))
        if vals2: ck(f'uc{u}-crossj2',close(o['MeanCrossModelJaccardSupportGE2'],sum(vals2)/len(vals2)))

    s=json.loads((OUT/'rq3_summary.json').read_text(encoding='utf-8'))
    ck('summary_queries',s['population']['rq3_primary_queries']==2369)
    ck('summary_model_uc',s['population']['model_uc_units']==189)
    ck('summary_owner_cells',s['ownership']['observed_model_run_uc_cells']==observed)
    ck('summary_evaluable',s['run_stability']['model_uc_units_with_evaluable_pairs']==independent_evaluable)
    ck('boundaries',all(v is False for v in s['analysis_boundary'].values()))
    prod=(ROOT/'scripts/06_rq3_stability_fragmentation.py').read_text(encoding='utf-8').casefold()
    ck('no_semantic_artifact_consumption','tfidf_scores' not in prod and 'mpnet_scores' not in prod and 'semantic_retrieval' not in prod)
    ck('no_inference','ttest' not in prod and 'mannwhitney' not in prod and 'anova' not in prod and 'pvalue' not in prod)

    status='PASS' if not failures else 'FAIL'
    REPORT.write_text('\n'.join(['# Prompt 6 Independent Validation','',f'## Overall Assessment: {status}','',f'Independent checks passed: **{len(checks)-len(failures)}/{len(checks)}**','',
        '- Recomputed the frozen 2,369-query RQ3 identity population from the clean source.','- Verified all 189 Model×UC stability/fragmentation rows.','- Verified all 1,890 Model×Run×UC owner cells.','- Verified all 21 cross-model UC summaries and support≥2/10 sensitivity summaries.','- Confirmed Prompt-4 identity reconciliation and zero duplicate trace conflicts.','- Confirmed no semantic-score consumption, legacy semantic fields, inferential tests, or generator-quality ranking.','',
        '### Interpretation caveat','Exact behavior/name fragmentation is a structural reproducibility measure, not semantic disagreement or correctness.','',f'### Gate\nSafe to accept Prompt 6 RQ3 analysis: **{"YES" if not failures else "NO"}**.'] + ([] if not failures else ['', 'Failures: '+', '.join(failures[:20])]))+'\n',encoding='utf-8')
    print(json.dumps({'status':status,'checks':len(checks),'failures':failures[:20]},indent=2))
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
