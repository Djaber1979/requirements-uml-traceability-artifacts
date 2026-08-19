#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, re
from collections import Counter
from pathlib import Path

COLUMNS = ["Model","Run","File","Class","MethodName","Signature","Has_UC_Annotation","UC_References","UC_Action"]
MOJIBAKE = "â€™"
UNICODE_APOSTROPHE = "’"

def refs(v):
    if v is None: return ""
    if isinstance(v, list): return ",".join(str(x).strip() for x in v if str(x).strip())
    return str(v).strip()

def collect(src: Path):
    out=[]
    for p in sorted(src.glob("*.json"), key=lambda q:q.name):
        m=re.match(r"^(.*)_run(\d+)$", p.stem)
        if not m: raise RuntimeError(p.name)
        model,run=m.group(1),str(int(m.group(2)))
        obj=json.loads(p.read_text(encoding="utf-8"))
        for cls in obj.get("classes",[]):
            for meth in cls.get("methods",[]):
                if not meth.get("name"): continue
                ann=meth.get("annotation") or {}
                out.append({"Model":model,"Run":run,"File":p.name,"Class":str(cls.get("name","")),"MethodName":str(meth.get("name","")),"Signature":str(meth.get("signature","")),"Has_UC_Annotation":"True" if ann else "False","UC_References":refs(ann.get("uc_references",ann.get("uc_reference"))),"UC_Action":str(ann.get("uc_action",ann.get("action","")) or "").strip()})
    return out

def key(row, normalize_encoding=False):
    vals=[]
    for c in COLUMNS:
        v=str(row.get(c,""))
        if normalize_encoding and c=="UC_Action": v=v.replace(MOJIBAKE,UNICODE_APOSTROPHE)
        vals.append(v)
    return tuple(vals)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--json-dir", default="historical_generation/json")
    ap.add_argument("--out", default="reconstructed_clean_source.csv")
    ap.add_argument("--compare", default="derived/behaviors/source_without_legacy_semantics.csv")
    a=ap.parse_args()
    rows=collect(Path(a.json_dir))
    with open(a.out,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS); w.writeheader(); w.writerows(rows)
    if not a.compare:
        print(f"Wrote {len(rows)} archive-native reconstructed rows"); return
    with open(a.compare,newline="",encoding="utf-8") as f: frozen=list(csv.DictReader(f))
    native_r=Counter(key(r) for r in rows); native_f=Counter(key(r) for r in frozen)
    norm_r=Counter(key(r,True) for r in rows); norm_f=Counter(key(r,True) for r in frozen)
    only_r=native_r-native_f; only_f=native_f-native_r
    if sum(only_r.values()) != 1 or sum(only_f.values()) != 1 or norm_r != norm_f:
        raise SystemExit("FAIL: mismatch is not the documented one-row UC_Action encoding artifact")
    print(f"PASS: {len(rows)} rows; native exact match=False (1 documented UC_Action mojibake row); match after documented encoding normalization=True")
if __name__ == "__main__": main()
