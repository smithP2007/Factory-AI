# DATASET_READY_FOR_TRAINING.md

**Date**: 2026-10-01 (UTC+8) · **Phase**: post-reacquisition corrections · **Gate**: awaiting explicit training approval

---

## 1. Are the datasets ready for training?

**YES — for exactly two models, under the conditions below. NOT ready for a unified 5-class PPE detector.**

| Model | Data | Ready? | Conditions already satisfied |
|---|---|---|---|
| **M1 — fire/smoke detector** | D-Fire, leakage-aware split | **READY** | mapping verified (statistical + visual); leakage removed; validated PASS |
| **M2 — helmet detector (head-level)** | Hard Hats v2, leakage-aware split | **READY** | leakage removed; validated PASS; kept separate |
| Unified 5-class PPE detector | — | **NOT READY** | person / safety_vest / gloves have no sufficient source (see §3, §7) |
| Person detector | frozen COCO-pretrained yolo26n | **NO TRAINING NEEDED** | eval reference acquired (`production/coco_person_reference/`) |

`DATASET_VALIDATION_FINAL.json`: **OVERALL PASS** (0 invalid images across 34,804
split-listed images, 0 label-syntax errors, class IDs within mapping, demo
datasets untouched, credential scan clean).

## 2. What exact models/classes will be trained?

1. **M1**: fine-tune `models/yolo26n.pt` on D-Fire leakage-aware split,
   2 classes: **0 = smoke, 1 = fire** (final verified production mapping —
   NOT the demo's 0=fire,1=smoke; see `production/dfire/CLASS_MAPPING_VERIFICATION.md`).
2. **M2**: fine-tune `models/yolo26n.pt` on Hard Hats v2 leakage-aware split,
   2 head-level classes: **0 = hardhat → helmet, 1 = no-hardhat → no_helmet**
   (target-schema mapping applied at integration, classes renamed then).
3. No other detector is trained. `no_safety_vest` is NOT a trained class.

## 3. Which dataset supplies each class?

| Target class | Supplier | Evidence |
|---|---|---|
| fire (1) | D-Fire (official archive) | 13,776 fire boxes in the leakage-aware split |
| smoke (0) | D-Fire | 10,376 smoke boxes in the leakage-aware split |
| helmet | Hard Hats v2 (`hardhat`) | 39,199 boxes in the leakage-aware split |
| no_helmet | Hard Hats v2 (`no-hardhat`) | 10,905 boxes in the leakage-aware split |
| person | **COCO-pretrained yolo26n (frozen)** — person supervision complete in COCO by construction | eval reference: 5,000 imgs / 10,777 person boxes (`coco_person_reference/`) |
| safety_vest | **NONE — unresolved** | SH17 blocked (14.2 GB, Kaggle auth); construction-safety only 53 boxes (evidence-only) |
| gloves | **NONE — unresolved** | SH17 blocked; no licensed obtainable alternative found |
| no_safety_vest | compliance-engine derivation (NOT a detector class) | unchanged from accepted strategy; construction-safety's 38-box `no-safety vest` class recorded as evidence, not schema change |

## 4. How is missing PPE inferred?

- **person**: from the frozen COCO-pretrained base detector (already trained on
  complete COCO train2017 person supervision). Never from Hard Hats imagery
  (Hard Hats v2 has no person class; the Voxel51 3.16% finding stands).
- **helmet/no_helmet → person**: at inference, M2 head boxes are assigned to M0
  person boxes (containment/IoU assignment); a person with a `no_helmet` head is
  flagged. Assignment thresholds are integration-time parameters, to be tuned
  and documented when the compliance engine is approved.
- **safety_vest / gloves**: **not inferred at all** — the engine must not
  output compliance verdicts for classes it has no detector for. Checks stay
  disabled until a verified source exists.
- **no_safety_vest**: derived only when a vest detector exists (person box
  without overlapping vest box). Today it would be meaningless, so it is not computed.

## 5. How is leakage prevented?

Method (`scripts/leakage_aware_split.py`, seed 20261001, deterministic):

1. **Exact-duplicate quarantine**: 64-bit dHash; images at Hamming distance 0
   are clustered and only one representative is kept (D-Fire: 4,150 quarantined;
   Hard Hats: 2,318). Manifests: `production/*/leakage_aware/quarantine_manifest.csv`.
2. **Group-aware assignment**: remaining images joined into groups by
   (a) shared source key (Roboflow base name; D-Fire series prefixes),
   (b) perceptual near-duplicates (dHash Hamming ≤ 8),
   (c) temporal adjacency for D-Fire video series (same prefix, sequence gap ≤ 2,
   Hamming ≤ 16). Whole groups go to one split — **no random image-level split**.
3. **Stratified group-level split**: greedy largest-group-first against
   per-profile ratio targets (D-Fire 80/10/10; Hard Hats 70/20/10), seeded.
4. **Verification**: zero cross-split near-duplicate pairs at Hamming ≤ 8 in
   BOTH datasets (official splits had 35,733 and 17,424 pairs at ≤ 2 respectively).

| Dataset | Images used | Groups | Train / val / test (imgs) | Quarantined d=0 | Cross-split ≤8 NEW |
|---|---:|---:|---|---:|---:|
| D-Fire | 17,377 | 8,635 | 13,901 (5,337 gr) / 1,739 (1,650 gr) / 1,737 (1,648 gr) | 4,150 | **0** |
| Hard Hats v2 | 17,427 | 13,701 | 12,199 (8,473 gr) / 3,486 (3,486 gr) / 1,742 (1,742 gr) | 2,318 | **0** |

Final class counts (train/val/test):
- D-Fire — smoke 8,796/786/794 · fire 11,812/967/997 · negatives 5,315/1,014/1,010
- Hard Hats — hardhat 26,727/8,301/4,171 · no-hardhat 6,765/2,738/1,402 · negatives 208/60/30

Intra-split same-scene near-duplicates remain (video-frame nature; harmless —
they never straddle splits). Split lists reference original files (no copies);
training YAMLs point at `leakage_aware/{train,val,test}.txt`.

## 6. What licenses apply?

| Dataset | License | Obligation |
|---|---|---|
| D-Fire | **None declared**; public research release | Citation required: Venâncio, Lisboa, Barbosa, *Neural Computing and Applications* (2022) |
| Hard Hats v2 (Roboflow export) | **CC BY 4.0** | Attribution |
| construction-safety (evidence-only) | **CC BY 4.0** | Attribution |
| COCO val2017 annotations | **CC BY 4.0** | Attribution; images under respective Flickr terms (research use) |
| SH17 | CC BY-NC-SA 4.0 | **Non-commercial** — blocked, NOT used |
| Demo datasets (Wikimedia) | per-image sidecars (`.license.json`) | untouched, unchanged |

## 7. What remains unresolved?

1. **safety_vest + gloves training source** — SH17 blocked (needs Kaggle
   credentials + ≥ 30 GB disk). Until then M1+M2+M0 cover fire/smoke and
   helmet compliance only; vest/gloves checks disabled.
2. **D-Fire night-frame filter** — the lost production set had a night-noise
   filter whose criteria are unrecoverable; the official archive (used here)
   contains its original night frames. Re-derivation is a separate approved task.
3. **26 degenerate zero-area boxes** in official D-Fire annotations — left
   as-is (source-inherent; Ultralytics drops zero-area boxes at load time).
4. **Hard Hats source noise** — a small number of compilation-style frames
   (e.g., mask-overlay overlays) exist; kept (documented via previews).
5. **Person↔head assignment thresholds** — integration-time parameter, not yet tuned.
6. **GitHub push** — commits through this phase are local only; needs a fresh
   PAT (the old one must stay revoked).
7. **`no_safety_vest` as detector class vs engine logic** — stays engine logic;
   revisit only with a real vest source.

## 8. Approved training commands (RUN ONLY AFTER EXPLICIT APPROVAL)

```bash
# M1 fire/smoke (2 classes: 0=smoke, 1=fire)
yolo train model=models/yolo26n.pt data=datasets/production/dfire/dfire_native.yaml \
     epochs=100 imgsz=640 name=m1_fire_smoke

# M2 helmet (2 classes: 0=hardhat->helmet, 1=no-hardhat->no_helmet)
yolo train model=models/yolo26n.pt data=datasets/production/hardhats/hardhats_native.yaml \
     epochs=100 imgsz=640 name=m2_helmet
```

## 9. STOP status

Per instruction: corrections complete, validation PASS, this document written.
**No training started. No tracking / FastAPI / database / dashboard / alerts.**
Awaiting explicit approval.
