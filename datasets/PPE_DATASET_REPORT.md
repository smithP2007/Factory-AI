# PS6 Core PPE Dataset Report

**Dataset**: `datasets/ppe_core` — four mandatory PPE classes for the PS6 compliance pipeline
**Generated**: 2026-10-02 · **Prepared by**: `scripts/prepare_ppe_core.py` (validated by an independent post-quarantine audit)
**Status**: **VALIDATION PASS · VISUAL INSPECTION PASS (with documented limitations)**

---

## 1. Source and provenance

| Item | Value |
|---|---|
| Source | Hugging Face dataset [`51ddhesh/PPE_Detection`](https://huggingface.co/datasets/51ddhesh/PPE_Detection) |
| Upstream origin | Roboflow Universe project `tanish-y7iqo/ppe-detection_data-adcya`, version 3 (`PPE.zip`, exported 2025-11-21) |
| Declared license | **CC BY 4.0** (stated both on the HF dataset card and in the embedded `README.roboflow.txt`) |
| Downloaded artifact | `PPE.zip` — 667,690,668 bytes, SHA-256 `4b10c6da2abc0f9b5ddf0b37513879ae16922b9f3f21e8b75de12f45cd3de9ac` |
| Raw preservation | Zip stored byte-exact at `datasets/raw/ppe/51ddhesh_PPE_Detection/PPE.zip` (gitignored, never modified); extraction to `extracted/` performed read-only with CRC check (`ZIP_ENTRIES=24168`, no errors) |
| Upstream preprocessing | Auto-orientation (EXIF stripped) + resize to 640×640 (stretch). **No image augmentation** applied upstream — confirmed in `README.roboflow.txt` |

The raw source is preserved unchanged; every derived file in `datasets/ppe_core/` was produced by `scripts/prepare_ppe_core.py` from that raw source.

## 2. Raw source audit (before filtering)

- **Images**: 12,078 (train 8,774 / valid 2,070 / test 1,234) — every image has exactly one label file and vice versa (0 missing, 0 orphan).
- **Class schema** (from `data.yaml`, six classes): `0=Gloves, 1=Vest, 2=goggles, 3=helmet, 4=mask, 5=safety_shoe`.
  ⚠️ The actual export ordering differs from the ordering implied by the HF dataset card; mapping is therefore done **by normalized class name, never by assumed index**.
- **Object counts vs card declaration**:

| Source class | Actual | Declared | Diff |
|---|---:|---:|---:|
| Gloves | 2,693 | 2,693 | 0 |
| Vest | 4,418 | 4,418 | 0 |
| goggles | 1,431 | 1,431 | 0 |
| helmet | 2,702 | 2,703 | −1 |
| mask | 2,761 | 2,763 | −2 |
| safety_shoe | 2,006 | 2,006 | 0 |
| **Total** | **16,011** | **16,014** | **−3** |

- The −1 helmet is fully explained: one malformed label line (a 9-field polygon-style row in `test/VID-20240831-WA0012_mp4-0011_jpg.rf.73aaa67d….txt`) that is skipped by the parser and logged in the quarantine manifest.
- The −2 mask is card-vs-export drift (0.02% of objects); all class IDs in the export are within 0–5 and every line is well-formed apart from the single helmet row above.

## 3. Target class mapping and exclusions

Final detector schema (exactly four classes, per the PS6 core requirement):

| Target id | Name | Kept from source |
|---|---|---|
| 0 | `helmet` | source `helmet` (id 3) |
| 1 | `gloves` | source `Gloves` (id 0) |
| 2 | `goggles` | source `goggles` (id 2) |
| 3 | `vest` | source `Vest` (id 1) |

**Excluded (annotations removed, never renamed)**: `mask` (id 4), `safety_shoe` (id 5). They are not part of the PS6 mandatory set.

**No fabricated classes**: no `no_helmet` / `no_gloves` / `no_goggles` / `no_vest` detector classes exist anywhere in this dataset. Missing PPE is inferred downstream by `scripts/compliance_engine.py` from worker↔PPE association, per the approved architecture. Images whose labels become empty after filtering are retained deliberately as **valid background negatives** for the four-class detector (3,756 raw images had zero annotations of any class).

## 4. Duplicate quarantine (leakage control)

Cross-split leakage is severe in the raw export because a large share of images are **video frames** (filename pattern `VID-*.mp4-NNNN_jpg.rf.*`) that were distributed across train/valid/test by the upstream splitter. Frame-adjacent near-copies of the same scene therefore appear in multiple splits.

Method (all cross-split, priority keep test > val > train):

1. **Exact duplicates** — full-file SHA-256; identical files in more than one split.
2. **Near duplicates** — 64-bit dHash, Hamming distance ≤ 8; candidate search via exact multi-index (pigeonhole) bucketing, verified pair-set-identical to a brute-force scan on a 600-image real-data sample (81/81 pairs). Threshold follows `datasets/PPE_DATASET_STRATEGY.md` §6.

| Outcome | Count |
|---|---:|
| Exact cross-split duplicates quarantined | 1,137 |
| Near cross-split duplicates quarantined (distance 0–8, mean 4.1) | 2,452 |
| Malformed label lines skipped | 1 |
| **Images quarantined in total** | **3,589** (train 3,087 · val 495 · test 7) |
| **Images retained** | **8,489** (12,078 − 3,589) |

The priority rule kept the test split 99.4% intact (only 7 removals), so the final evaluation set remains close to the upstream distribution. Full audit trail: `datasets/ppe_core/duplicate_quarantine_manifest.json` (every removal lists type, source path, kept counterpart, split, and distance where applicable).

## 5. Final dataset composition (post-quarantine)

| Split | Images | helmet | gloves | goggles | vest | Total objects |
|---|---:|---:|---:|---:|---:|---:|
| train | 5,687 | 1,893 | 1,267 | 961 | 1,854 | 5,975 |
| val | 1,575 | 181 | 422 | 294 | 632 | 1,529 |
| test | 1,227 | 596 | 298 | 139 | 471 | 1,504 |
| **total** | **8,489** | **2,670** | **1,987** | **1,394** | **2,957** | **9,008** |

Images containing each class (coverage):

| Split | helmet | gloves | goggles | vest |
|---|---:|---:|---:|---:|
| train | 790 | 673 | 862 | 1,029 |
| val | 145 | 221 | 251 | 346 |
| test | 283 | 164 | 126 | 244 |

Independent post-quarantine re-scan (fresh SHA-256 + dHash pass over the **output** dataset, not the manifest):

- Exact cross-split duplicates: **0**
- Near cross-split duplicates (≤ 8): **0**
- Foreign images not present in the raw source: **0**
- Label syntax: 100% of lines have exactly 5 numeric fields, class ∈ {0,1,2,3}, center ∈ [0,1]², 0 < w,h ≤ 1
- Image integrity: all 8,489 images open via PIL verification

Structural evidence: `datasets/PPE_CORE_VALIDATION.json` (machine-readable validator output), `datasets/ppe_previews/PREVIEWS_MANIFEST.json` (24 annotated previews, ~1.9 MB, committed for review).

## 6. Visual inspection findings

24 annotated previews (`datasets/ppe_previews/`) spanning all three splits and all four classes were inspected. Findings, stated plainly:

**Correct and useful**
- Box geometry is tight and correctly classified wherever annotations exist; no systematic coordinate errors.
- The WhatsApp-camera workshop/factory subset (`IMG-*`/`VID-*` filenames) matches the deployment domain well: real workshops, multiple workers per frame, small PPE targets — good multi-worker association test material.

**Limitations (accepted for the baseline, documented honestly)**
1. **`goggles` is the weakest class.** A substantial part of its images are web-scraped product/portrait shots — including ordinary eyeglasses and even browser screenshots (one preview contains a Google search page) — rather than safety goggles worn in industrial scenes. The class will likely underperform and needs cautious interpretation; treat goggle detections with lower trust until more in-domain data exists.
2. **Partial per-image annotation.** Many images show PPE items that are visible but not annotated (e.g., a worker with labeled gloves whose helmet and vest are unlabeled). Unannotated visible PPE acts as label noise (false negatives) during training. This is inherent to the source and is the reason compliance decisions must be thresholded conservatively.
3. **Watermarked stock photography** (Dreamstime/iStock/Colourbox) forms a sizable share of helmet/vest/gloves imagery — industrial scenes, but not CCTV-like. Domain shift vs the target factory cameras must be expected.
4. **Occasional annotation noise**: an occasional double box on one glove (tight + loose), and rare boxes on ambiguous regions (apron edge labeled as glove).
5. **Empty-label negatives (3,756 raw → retained post-quarantine)** are frames with no annotated objects at all; they are retained as background but may still contain unlabeled PPE, which the detector should not be punished for finding — acceptable noise level for a baseline.

**Verdict**: usable for the approved baseline training run. No fabrication of metrics or data anywhere in this pipeline; limitations above must be repeated in the model report.

## 7. Licensing note

The source is CC BY 4.0, which permits use and redistribution with attribution (attributed in `SOURCE_DATASET.txt` and here). Watermarked stock preview images inside the source inherit that upstream declaration; we redistribute them only as part of the cited dataset for research purposes. `Ultralytics Construction-PPE` (which has genuine missing-gear labels) remains **excluded** pending license review (AGPL-3.0), per strategy §10.

## 8. Reproducibility

```bash
# 1. raw download (byte-exact, CRC-checked)
python3 scripts/download_ppe_raw.py            # or: hf download 51ddhesh/PPE_Detection
# 2. filter + remap + quarantine (never modifies raw)
python3 scripts/prepare_ppe_core.py \
  --src-root datasets/raw/ppe/51ddhesh_PPE_Detection/extracted \
  --out-root datasets/ppe_core --near-threshold 8
# 3. independent validation gate (must print verdict PASS)
python3 /home/z/my-project/scripts/validate_ppe_core.py
# 4. previews
python3 /home/z/my-project/scripts/make_ppe_previews.py
```

Deterministic: no random splits are generated anywhere; the upstream split semantics are preserved as train/val/test, and quarantine decisions are order-deterministic with fixed priority test > val > train.
