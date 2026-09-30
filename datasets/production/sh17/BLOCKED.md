# SH17 — ACQUISITION BLOCKED (full dataset)

**Status**: NOT ACQUIRED. Investigated and verified to the extent possible without credentials or disk space. Kept separate (nothing was merged).

## What was verified (2026-10-01)

| Field | Value |
|---|---|
| Dataset | SH17 — "Safe Human" PPE dataset, 17 classes |
| Official repo | https://github.com/ahmadmughees/SH17dataset (verified live) |
| Paper | arXiv:2407.04590; Journal of Safety Science and Resilience (2024) |
| Scale | 8,099 annotated images, 75,994 instances, 17 classes (per official README) |
| Official class order | Captured from the repo's `sh17.yaml` — archived in `class_order_official.yaml` |
| Distribution channel | Kaggle: `mugheesahmad/sh17-dataset-for-ppe-detection` — images AND labels are ONLY on Kaggle (the GitHub repo holds no label files) |
| Kaggle metadata (anonymous API) | `totalBytes = 14,237,350,242` (14.2 GB); `licenseName = "CC BY-NC-SA 4.0"` |
| License finding | **Kaggle declares CC BY-NC-SA 4.0 — this CONFIRMS the historically documented license** (recorded as [HISTORICAL] in `datasets/DATASET_STRATEGY.md`). Note: the GitHub README separately states "educational, research and analysis purposes only" + Pexels license terms. Both regimes are non-commercial; the formal license citation should be CC BY-NC-SA 4.0 per the source's own Kaggle metadata. |

## Why blocked

1. **Disk**: 14.2 GB compressed vs ~3.1 GB free at acquisition time (D-Fire + Hard Hats occupy ~4.2 GB). Extraction would need ~28 GB. No partial-download option exists for the Kaggle archive, and labels are not downloadable separately.
2. **Credentials**: Kaggle download requires authentication (none provided this session).

## Consequence for the 5-class PPE plan

`safety_vest` and `gloves` remain UNCOVERED until SH17 (or an alternative) is acquired. Options:
- Provide Kaggle credentials + ≥ 30 GB free storage, or
- Approve an alternative vest/gloves source (e.g., GDUT-HD — license unclear, strategy doc §4).

Per the no-false-negative rule (strategy doc F5), when SH17 is acquired it must be source-gated per class and kept as its own source, never silently merged.
