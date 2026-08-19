# Analysis Specification

This file summarizes the frozen analytical boundaries represented by the included code/configuration.

## Population

RQ1 describes all 3,373 cleaned source rows. Primary semantic analyses use rows with a valid single UC1–UC21 recorded target, nonblank Class/MethodName/Signature, and an accepted exact scaffold class identity. Exact within-Model × Run duplicate behaviors are collapsed after checking target consistency. This yields 2,369 queries in 69 eligible cells; the other 21 cells are NA for RQ2/RQ4 rather than scored as zero.

## Primary estimand

Primary semantic reporting is the equal-cell mean across the 69 eligible Model × Run cells. It is not a row-weighted mean and is not an equal-model macro-estimand.

## Retrieval instruments

Behavior representation: Class + MethodName + Signature with deterministic identifier segmentation. Requirement representation: Title + Description + MainScenario. The recorded UC target is ranked among all 21 candidates using requirement-fitted TF-IDF cosine retrieval and the frozen MPNet-family sentence encoder in `config/semantic_models.yaml`.

These rankings measure **recorded-target concordance**, not semantic correctness or a gold-standard trace matrix.

## RQ3

Behavior recurrence uses exact Class + MethodName + Signature identities within the valid-trace population. Low exact overlap may reflect identity changes and/or absence of eligible traced behaviors and is not interpreted as pure API instability.

## Negative control and robustness

The negative control uses the frozen randomized UC-label/text mappings recorded under `artifacts/negative_control/`. The 0/2,000 exceedance result is descriptive finite-schedule separation, not a p-value. Run-resampling ranges are perturbation ranges, not confidence intervals. Representation sensitivity is material; the other prespecified perturbations are comparatively small for this corpus.

## Generation provenance limitation

The recovered historical source archive preserves all 90 archived generated PlantUML diagrams, the standardized prompt/scaffold/use-case inputs, corresponding parsed JSON representations, and the PlantUML-to-JSON/downstream extraction code. Per-call provider request/response logs, generation-time decoding parameters, immutable provider-side model revisions, and evidence of controlled run independence were not located. Model/run labels are therefore treated as nominal repeated-run identifiers rather than verified independent stochastic draws.
