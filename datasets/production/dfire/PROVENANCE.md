# D-Fire — Acquisition Provenance (Production Fire/Smoke)

**Acquired**: 2026-10-01 by PS6 reconstruction (dataset reacquisition phase)
**Status**: ACQUIRED, validated separately. NOT merged with demo or other sources.

## Source

| Field | Value |
|---|---|
| Dataset | D-Fire: an image dataset for fire and smoke detection |
| Official repository | https://github.com/gaia-solutions-on-demand/DFireDataset (GAIA — solutions on demand) |
| File obtained | "D-Fire dataset (only images and labels)" — official OneDrive link from the repo README |
| Direct download URL | `https://my.microsoftpersonalcontent.com/personal/C0BD25B6B048B01D/_layouts/15/download.aspx?share=EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw` (resolved from the official 1drv.ms shortlink in the README) |
| File size | 3,036,222,313 bytes (verified = HTTP content-length before extraction) |
| Archive timestamp | 2022-04-29 (file mtimes inside the zip) |
| License | **No explicit license file/declaration in the official repo.** Public research release; README requests citation: Venâncio, Lisboa, Barbosa — "An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices", Neural Computing and Applications, 2022 (https://link.springer.com/article/10.1007/s00521-022-07467-z) |
| Access requirements | None (public direct link; no Kaggle auth needed for this copy) |

## Official statistics (from repo README) vs measured here

| Item | Official README | Measured on acquired copy | Match |
|---|---:|---:|---|
| Images total | 21,527 | 21,527 (17,221 train + 4,306 test) | YES |
| Only-fire images | 1,164 | 1,164 | YES |
| Only-smoke images | 5,867 | 5,867 | YES |
| Fire+smoke images | 4,658 | 4,658 | YES |
| None (negatives) | 9,838 | 9,838 (empty label files) | YES |
| Fire boxes | 14,692 | 14,692 | YES |
| Smoke boxes | 11,865 | 11,865 | YES |

## CRITICAL: class index order

**The official D-Fire archive uses `0 = smoke`, `1 = fire`.**

Evidence (measured, not assumed):
- Images labeled ONLY class 0: 5,867 = README "Only smoke" row.
- Images labeled ONLY class 1: 1,164 = README "Only fire" row.
- Box totals: class 0 = 11,865 = README "Smoke" row; class 1 = 14,692 = README "Fire" row.

**This is the OPPOSITE of `datasets/fire_smoke/fire_smoke.yaml` in this repo (0=fire, 1=smoke)**, and the opposite of the "class indices intentionally match D-Fire" statement in `datasets/DATASET_REPORT.md` (Day 2). The Day 2 claim was wrong for this official archive (the lost-workspace production set may have been relabeled, or derived from a Kaggle re-pack that reordered classes — unverifiable).

**Decision (per no-blind-merge instruction)**: the acquired copy keeps NATIVE indices `0=smoke, 1=fire` in `dfire_native.yaml`. No relabeling, no merging. A class-remap decision (native -> repo schema `0=fire,1=smoke`) is deferred to the pre-training phase and must be applied explicitly if/when the user approves use of this dataset.

## Splits (kept as distributed — not re-split)

| Split | Images | Labels |
|---|---:|---:|
| train | 17,221 | 17,221 |
| test | 4,306 | 4,306 |
| total | 21,527 | 21,527 |

No official validation split in this archive (train/test only). The repo's `split_dataset.py` was NOT run (would re-split; deferred).

## Relation to lost-workspace historical numbers [HISTORICAL]

User-documented production stats (17,806 images / 13,696 fire / 10,512 smoke / 7,665 negatives) do NOT match this official copy (21,527 / 14,692 / 11,865 / 9,838). The lost production set was a filtered/re-labeled derivative (documented night-frame noise removal + likely class remap). The filter criteria are lost; re-deriving them is a pre-training decision.
