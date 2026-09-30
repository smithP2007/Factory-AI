# Voxel51 Hard-Hat Reference — VERIFICATION (NOT a training source)

**Status**: NOT acquired as a dataset (images not downloaded — deliberately). Only the 8.9 MB `samples.json` annotation manifest was inspected (CC0-1.0) to **independently verify Day 2 finding F1**. See `VERIFICATION.json` for the machine-readable evidence.

## Result [VERIFIED-AGAINST-SOURCE 2026-10-01]

- 5,000 images; detections: helmet 18,966 / head 5,785 / **person 751**.
- `person` boxes appear on only **158 / 5,000 images (3.16%)** — model-generated annotations (FiftyOne "quick start" semantic), essentially absent from 96.8% of images.
- **Confirms finding F1** in `datasets/DATASET_STRATEGY.md` §3: using this source for `person` supervision would turn 96.8% of visible people into background supervision (false-negative poisoning). The Day 2 exclusion decision was correct and is now evidence-backed.
- License CC0-1.0 (no restrictions); still excluded as a person source.

## Relation to historical PPE statistics

helmet 18,966 / head 5,785 here vs the historical production numbers (helmet 19,071 / no_helmet 5,891): close but NOT equal (deltas 105/106). Plausibly the lost production set combined this family with another source or a variant. This does NOT verify the historical numbers; it only shows provenance plausibility.

## Usage rule going forward

This reference must never be merged into training data. It exists as audit evidence for the source-gating policy (strategy doc F5).
