# PPE Architecture Decision (post-reacquisition corrections, 2026-10-01)

**Decision**: **Option C — multi-model architecture.** A single unified 5-class
PPE detector is NOT constructible from safely-obtainable data in this
environment. Training is approved-candidate ONLY for the sub-models below;
`no_safety_vest` remains compliance-engine logic (unchanged).

## 1. Why not a unified 5-class detector (evidence, not opinion)

| Requirement | Evidence |
|---|---|
| person | Hard Hats v2: no person class. Voxel51 variant: person on 3.16% of images (verified). Construction-safety: 69 person boxes total (incomplete-supervision risk, too small). COCO person is complete but a different annotation regime (80-class scenes). |
| helmet / no_helmet | COVERED by Hard Hats v2 (42,428 / 12,965 boxes). |
| safety_vest | Hard Hats v2: absent (vests VISIBLE but UNLABELED in many frames — visually confirmed). SH17 (16-class, vest+gloves): BLOCKED (14.2 GB vs disk, Kaggle auth). Construction-safety: 53 boxes — far below training scale. |
| gloves | SH17 blocked; construction-safety 18 boxes; no other no-auth licensed source found (HF search returned no gloves detection dataset with license). |
| no_safety_vest | No acquired source offers production-scale supervision; construction-safety annotates a `no-safety vest` class on 38 boxes — recorded as evidence that a detector class is *possible* with a proper source, but it cannot be trained here. Stays compliance-engine logic per the accepted strategy. |

Merging is additionally PROVEN unsafe between the two Roboflow-derived sets:
**236 exact duplicate images (dHash d=0) between Hard Hats v2 and
construction-safety** (`construction_safety/cross_dataset_overlap_vs_hardhats.json`).

## 2. Architecture (Option C)

| Model | Data (leakage-aware split) | Classes | Status |
|---|---|---|---|
| **M1 fire/smoke** | D-Fire production (`dfire/dfire_native.yaml`, 17,377 imgs after dedup) | 0=smoke, 1=fire | DATA READY (mapping verified) |
| **M2 helmet** | Hard Hats v2 (`hardhats/hardhats_native.yaml`, 17,427 imgs after dedup) | 0=hardhat, 1=no-hardhat → target helmet/no_helmet | DATA READY |
| **M0 person (frozen)** | COCO-pretrained `yolo26n.pt` (base model) | COCO person | NO TRAINING NEEDED; eval reference acquired (`coco_person_reference/`, 5,000 imgs, 10,777 person boxes) |
| safety_vest / gloves | — | — | **UNRESOLVED** (blocked SH17; no obtainable alternative) |
| no_safety_vest | compliance-engine derivation from M0 person + future vest source | — | unchanged (logic, not detector class) |

Compliance logic at inference: persons from M0 (COCO), heads from M2; helmet
compliance = head-level M2 result mapped onto M0 persons; vest/gloves checks
stay DISABLED until a verified source exists — the engine must not infer
compliance for classes it cannot see.

## 3. Person-class decision (task 5 resolution)

- Person detection = **separate reliable detector** (COCO-pretrained base
  model) — per the approved option "use an appropriate architecture where
  person detection comes from a separate reliable detector".
- Hard Hats v2 person annotations are NOT used (they do not exist; the Voxel51
  3.16% finding stands).
- COCO val2017 person reference acquired as the evaluation base for M0 person
  output when the compliance engine is built (NOT training data here).

## 4. Dataset rules enforced (final rule 6)

- No two sources are merged. Every training view is single-source.
- No image participates as negative/background supervision for a class that is
  visible-but-unlabeled in its source (vests in Hard Hats v2 confirmed visually;
  hence vest supervision cannot come from Hard Hats v2 imagery).
- SH17 remains blocked and separate (`sh17/BLOCKED.md`); acquisition path
  requires Kaggle credentials + >= 30 GB disk.

## 5. Answer to "can the project proceed with A/B/C?"

**YES — proceed with C (multi-model)**: M1 + M2 trained on their own
leakage-aware splits; person from the frozen COCO base; vest/gloves deferred
with the compliance engine degrading gracefully. Options A and B alone are
insufficient (A lacks person/vest/gloves; B: no licensed production-scale
vest/gloves source is obtainable without credentials in this environment —
verified by acquisition attempts and HF metadata search 2026-10-01).
