# Historical Source Provenance

## Pinned source

- Repository: `Djaber1979/-Behavioral-Augmentation-of-UML-Class-Diagrams`
- Commit: `b8e7814bdab4b863c0d679952a18dafff21c84c4`
- Recovery scope: 90 archived generated PlantUML diagrams (9 recorded model labels × 10 nominal runs), 90 parsed JSON representations, the standardized generation prompt, methodless scaffold, 21-use-case corpus, and the archived PlantUML parser.

## Recovered chain

`Main.txt + Methodless.txt + UCs.txt` → archived generated PlantUML artifacts → `puml_to_json.py` → per-run JSON → nine-field clean reconstruction → frozen 3,373-row Phase-2 analytical table.

The included `scripts/reconstruct_clean_source_from_historical_json.py` reconstructs only `Model, Run, File, Class, MethodName, Signature, Has_UC_Annotation, UC_References, UC_Action`. Archive-native reconstruction yields 3,373 rows. Exactly one row differs from the frozen CSV, solely in `UC_Action`: the archive contains `System updates the user’s role (actual state change)`, whereas the frozen CSV contains the UTF-8 mojibake form `System updates the userâ€™s role (actual state change)`. After normalizing this documented text-encoding artifact, the complete nine-field row multiset matches. No dataset is silently rewritten; both the native mismatch and normalized equivalence are recorded in `artifacts/provenance/historical_reconstruction_audit.json`.

## Boundaries

The archived `.puml` files are described as **archived generated PlantUML diagrams**, not raw provider API responses. The historical repository also preserves a narrow PlantUML post-processing utility (`Modify_EmptyStrings_In_Puml.py`), so provider-response byte identity is not asserted.

The historical repository contains downstream analysis code that produced the former 11-column combined table, but that code and table also involve the legacy semantic fields `Best_Match_Action` and `SimilarityScore`. They are deliberately not imported into this clean publication archive. The independent reconstruction script above bypasses those legacy semantic outputs.

No per-call provider request/response logs, generation-time temperature/top-p/token settings, immutable provider-side model revisions, generation seeds, or controlled run-independence records were located. Thus Model × Run identifiers remain nominal repeated-run labels rather than evidence of verified independent stochastic draws.
