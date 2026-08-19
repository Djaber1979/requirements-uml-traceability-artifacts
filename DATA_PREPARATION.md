# Data Preparation and Redaction Boundary

The public analytical table is `derived/behaviors/source_without_legacy_semantics.csv`.

It was produced from the frozen 3,373-row extraction table by **dropping two columns only**:

- `Best_Match_Action`
- `SimilarityScore`

No rows were removed during this cleaning step. Row order was preserved, and all retained values were verified exactly. The resulting public schema is:

`Model, Run, File, Class, MethodName, Signature, Has_UC_Annotation, UC_References, UC_Action`

The excluded fields are legacy semantic outputs and are not part of the independent semantic retrieval path used in this study. Their values are intentionally absent from this public archive. The exact transformation metadata and source/clean checksums are recorded in `derived/behaviors/CLEAN_COPY_MANIFEST.json`.

The original 11-column CSV is not on the whitelist and cannot be copied by the public-export builder.
