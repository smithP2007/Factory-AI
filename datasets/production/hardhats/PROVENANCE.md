# Hard Hats — Acquisition Provenance (Production PPE)

**Acquired**: 2026-10-01 by PS6 reconstruction (dataset reacquisition phase)
**Status**: ACQUIRED, validated separately. NOT merged with demo or other sources.

## Source

| Field | Value |
|---|---|
| Dataset | "Hard Hats" v2 — `resized640_noAugmentation-FAST` export |
| Origin | Roboflow Universe: https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5 |
| Mirror used | Hugging Face `keremberke/hard-hat-detection` (verbatim Roboflow export incl. `README.roboflow.txt`, CC BY 4.0 declared) |
| Download URLs | `https://huggingface.co/datasets/keremberke/hard-hat-detection/resolve/main/data/{train,valid,test}.zip` |
| Export timestamp | 2023-01-16 21:17 GMT (per embedded README.roboflow.txt) |
| License | **CC BY 4.0** (declared in `README.dataset.txt` AND in each COCO `licenses` entry: https://creativecommons.org/licenses/by/4.0/) |
| Access requirements | None (public mirror; no Roboflow API key needed) |
| Pre-processing by source | auto-orientation + resize to 640x640 (stretch); no augmentation |

## Measured statistics (this acquisition, post COCO→YOLO conversion)

| Split | Images | Labels | hardhat boxes | no-hardhat boxes | Total boxes |
|---|---:|---:|---:|---:|---:|
| train | 13,782 | 13,782 | 28,996 | 9,705 | 38,701 |
| valid | 3,962 | 3,962 | 8,952 | 2,222 | 11,174 |
| test | 2,001 | 2,001 | 4,480 | 1,038 | 5,518 |
| **total** | **19,745** | **19,745** | **42,428** | **12,965** | **55,393** |

Conversion: `scripts/coco_to_yolo_hardhats.py` (COCO bbox x,y,w,h → YOLO center format; native class order preserved: 0=hardhat, 1=no-hardhat). 0 missing files, 0 failed conversions.

## QA results (qa_acquired_dataset.py + cross-split audit)

- Image integrity: 19,745/19,745 decode via OpenCV — 0 unreadable.
- Label pairing: 0 images without label; 0 labels with syntax errors.
- Within-split near-duplicates (dHash 64-bit, Hamming ≤ 8): 153,893 pairs total (7,111 at distance 0). Distance distribution healthy (random-pair mean 31.6 ≈ 32 expected), i.e. these are genuinely similar images, not a hashing artifact. Aggressive cluster chaining observed — dedup/clustering decisions DEFERRED to pre-training approval.
- **Cross-split leakage audit: train↔valid 10,974 pairs ≤2; train↔test 5,438 pairs ≤2 (10,855 ≤4); valid↔test 1,012 ≤2.** The source's train/valid/test split does NOT separate near-identical frames. Evidence grid: `outputs/dataset_preview/production/hardhats_cross_split_dup_evidence.jpg`. Do NOT treat test mAP as unbiased without re-splitting/dedup.
- Visual QA previews: `outputs/dataset_preview/production/hardhats/{train,valid,test}_preview.jpg` — boxes verified aligned.
- Credential scan: clean.

## Relation to the 5-class PPE target [PROPOSED schema]

| Target class | Coverage here | Mapping |
|---|---|---|
| 0 person | NOT present | not a source class (see compliance-engine note) |
| 1 helmet | COVERED (42,428 boxes) | `hardhat` → `helmet` |
| 2 no_helmet | COVERED (12,965 boxes) | `no-hardhat` → `no_helmet` |
| 3 safety_vest | NOT present | requires SH17 (blocked, see `../sh17/BLOCKED.md`) |
| 4 gloves | NOT present | requires SH17 (blocked) |

`no_safety_vest` remains a **compliance-engine derivation** (person without overlapping vest box) — nothing acquired contradicts this; no acquired dataset even contains a usable `no_safety_vest` detector class. Nothing here "proves otherwise", so the architecture stands as proposed.

Person supervision note: `person` is deliberately NOT mapped from any acquired source (the Voxel51 variant of this dataset carries model-generated `person` boxes on only 158/5,000 images — see `../voxel51_hardhat_reference/VERIFICATION.md` — which independently verifies the Day 2 finding F1). At inference time the compliance engine can source `person` from the COCO-pretrained base model instead of training it on these images.
