# Requirements-to-UML Traceability Study Artifacts

This artifact-only repository accompanies the study **Structural and Semantic Concordance of Requirements-to-UML Traceability in LLM-Generated Behavioral Models**.

## What is included

- `derived/`: the cleaned 3,373-row analytical table, the 2,369-query primary semantic population, and the 21 materialized requirement documents.
- `input_snapshot/provenance/`: the Phase-2 use-case specification, methodless UML scaffold, and expected-output index.
- `historical_generation/`: the recovered 90 archived generated PlantUML diagrams, their 90 parsed JSON representations, the exact historical prompt/scaffold/use-case inputs, and the archived PlantUML parser, all pinned to a recorded upstream commit.
- `config/`: frozen preprocessing, retrieval, model, exclusion, and robustness configuration.
- `scripts/`: the primary RQ/robustness scripts, independent validators, and the historical clean-source reconstruction helper.
- `artifacts/`: frozen analytical summaries plus the negative-control schedule/distribution and provenance audit needed to check the reported results.
- `PUBLIC_WHITELIST.json`: the exact source-to-public-file allowlist used to construct this artifact release.
- `DATA_PREPARATION.md`, `ANALYSIS_SPECIFICATION.md`, `REPRODUCIBILITY.md`, and `HISTORICAL_SOURCE_PROVENANCE.md`: the public methodological and provenance documentation.
- `CHECKSUMS.sha256`: SHA-256 digests for the artifact-release contents.

## Manuscript-version policy

No journal-specific manuscript source, manuscript PDF, publisher template, submission bundle, or superseded article version is included in this repository. This repository is the reproducibility/artifact record; it is deliberately kept independent of journal retargeting and manuscript-version changes.

## Deliberately excluded

The original 11-column raw extraction CSV is **not** included. Its two legacy semantic fields (`Best_Match_Action` and `SimilarityScore`) were excluded before this study's independent semantic analysis. Internal drafting files, review notes, workflow handoffs, working-stage reports, publication-package staging, journal submission materials, and Git/pull-request/workflow history are also excluded.

## Core frozen quantities

- 3,373 UML behavior rows in the cleaned extraction table.
- 2,369 primary semantic queries across 69 eligible Model × Run cells.
- TF-IDF Hit@1/MRR: 0.559/0.676.
- MPNet Hit@1/MRR: 0.633/0.723.
- Exact Top-1 cross-method agreement: 0.686.
- 1,260 distinct exact valid-traced behaviors; mean run-pair Jaccard 0.100; mean singleton share 0.730.
- Four primary negative-control comparisons, each exceeding all 2,000 randomized mappings.

See `ANALYSIS_SPECIFICATION.md`, `REPRODUCIBILITY.md`, and `HISTORICAL_SOURCE_PROVENANCE.md` for interpretation and provenance boundaries.

## License

Repository-authored analysis code, cleaned analytical data, configuration, result summaries, and documentation are available under the MIT License. Imported historical-generation artifacts are not automatically relicensed by the root MIT grant; see `LICENSE_SCOPE.md`.
