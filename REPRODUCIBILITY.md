# Reproducibility Guide

## Environment

Python 3.10+ is recommended. The analytical scripts use NumPy, SciPy, scikit-learn, PyYAML, Hugging Face Hub, and Sentence-Transformers. Neural stages require network access on first model download. Exact model identifiers/revisions are frozen in `config/semantic_models.yaml`.

## Suggested execution order

```bash
python scripts/05_structural_traceability_rq1.py
python scripts/05_validate_structural_traceability_rq1.py

python scripts/04_primary_semantic_retrieval.py
python scripts/04_validate_primary_semantic_retrieval.py

python scripts/06_rq3_stability_fragmentation.py
python scripts/06_validate_rq3_stability_fragmentation.py

python scripts/07_negative_control.py
python scripts/07_validate_negative_control.py

python scripts/08_semantic_sensitivities.py
python scripts/08_validate_semantic_sensitivities.py

python scripts/09_run_resampling.py
python scripts/09_validate_run_resampling.py
```

The included artifact summaries provide frozen comparison points. Key anchors are recorded in `artifacts/structural_rq1/corpus_structural_summary.json`, `artifacts/semantic_retrieval/corpus_primary_metrics.json`, `artifacts/semantic_retrieval/cross_method_corpus_consensus.json`, `artifacts/rq3_stability_fragmentation/rq3_summary.json`, `artifacts/negative_control/negative_control_summary.json`, `artifacts/semantic_sensitivities/sensitivity_summary.json`, and `artifacts/run_resampling/overall_summary.csv`.

## Historical source-artifact reconstruction

The recovered source-artifact layer is documented in `HISTORICAL_SOURCE_PROVENANCE.md`. `scripts/reconstruct_clean_source_from_historical_json.py` reconstructs the nine permitted extraction fields from the 90 archived JSON files and can compare the reconstructed row multiset with the frozen 3,373-row clean analytical table. The historical provider-call layer itself is not reproducible because per-call request/response logs and generation-time decoding settings were not located.

Exactly one `UC_Action` string differs natively between the archived JSON reconstruction and the frozen CSV: the archive contains the correct Unicode smart apostrophe in `user’s`, whereas the frozen CSV contains the UTF-8 mojibake form `userâ€™s`. All other fields and rows match, and normalizing only this documented encoding artifact yields full nine-field multiset equality. `UC_Action` is not part of semantic query construction.

## Manuscript independence

No manuscript or journal-specific submission package is required to rerun or audit the analyses. Manuscript versions are intentionally omitted from this artifact repository so journal retargeting cannot make the reproducibility package stale.
