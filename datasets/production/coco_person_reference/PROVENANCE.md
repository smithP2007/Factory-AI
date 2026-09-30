# COCO val2017 Person Reference — Provenance

**Acquired**: 2026-10-01 (dataset correction phase, person-class task 5)
**Status**: ACQUIRED + validated. Role: **EVALUATION REFERENCE ONLY** — never
merged into a training set.

## Why this source (person class task)

The reacquisition confirmed `person` is NOT adequately supervised by any
acquired PPE dataset (Hard Hats v2 has no person class; the Voxel51 variant
labels person on only 158/5,000 images = 3.16%). COCO is the opposite regime:
**person is a core COCO category and every visible instance is annotated in
every labeled image — person supervision is complete by construction.** This
makes it the reference source for verifying/sampling person detection rather
than for PPE-specific training.

## Source

| Field | Value |
|---|---|
| Images | COCO val2017 — `http://images.cocodataset.org/zips/val2017.zip`, 815,585,330 bytes (official size) |
| Annotations | `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`, 252,907,541 bytes (official size) |
| Access | None (public direct download) |
| License | Annotations CC BY 4.0 (COCO); images under their respective Flickr terms (customary research use) — attribution obligations apply |
| Class | 0 = person (single class; extracted from COCO category "person") |

## Measured statistics

| Item | Value |
|---|---:|
| Images | 5,000 (all kept; 2,307 are person-free negatives) |
| Person boxes (YOLO, id 0) | 10,777 |
| iscrowd=1 instances dropped | 227 (RLE crowd regions are not box supervision) |
| Unreadable images | 0 |
| Visual QA | `person_reference_preview.jpg` — boxes verified on all visible persons incl. partial occlusions |

Full report: `PERSON_REFERENCE_REPORT.json`. Conversion:
`scripts/build_coco_person_reference.py`.

## Role in the architecture (decision recorded in ../PPE_ARCHITECTURE_DECISION.md)

- `person` at inference time comes from the **COCO-pretrained yolo26n base
  detector** (already trained on COCO train2017's complete person supervision).
- This val2017 reference is the held-out evidence base for evaluating that
  person input if/when the compliance engine needs quantified person recall.
- NOT used as training data (would be redundant with pretraining and is a
  different annotation regime than PPE sources).
