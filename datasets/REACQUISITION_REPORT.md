# Dataset Reacquisition Report — PS6 Factory-AI (Post-Reconstruction)

**Date**: 2026-10-01 (UTC+8)
**Scope**: Dataset reacquisition + validation ONLY (per approval). **No training, no model building, no app infrastructure (tracking/FastAPI/db/dashboard/alerts), no merging.**
**Baseline**: reconstruction approved at commit `55739a1` (history `5456d25 → a1f0cba → 55739a1` preserved; this phase adds new commits on top only).

---

## 0. Executive summary

| Dataset | Status | Images | Boxes | License |
|---|---|---:|---:|---|
| D-Fire (fire/smoke) | **ACQUIRED + validated** | 21,527 | 26,557 (14,692 fire / 11,865 smoke) | none declared; citation required |
| Hard Hats v2 (PPE helmet family) | **ACQUIRED + validated** | 19,745 | 55,393 (42,428 hardhat / 12,965 no-hardhat) | CC BY 4.0 |
| SH17 (PPE vest/gloves) | **BLOCKED** (14.2 GB > disk; Kaggle auth) | — | — | CC BY-NC-SA 4.0 (confirmed from Kaggle metadata) |
| GDUT-HD | NOT attempted (optional; license unclear) | — | — | unclear |
| Voxel51 hard-hat | Evidence-only (annotations manifest inspected; images not taken) | 5,000 | 25,502 | CC0-1.0 |

**5-class PPE verification**: `helmet` + `no_helmet` are covered by Hard Hats v2; `person`, `safety_vest`, `gloves` are NOT covered by what could be acquired. `no_safety_vest` remains a compliance-engine derivation — **no dataset evidence contradicts the proposed architecture**; in fact, no acquired source even offers a usable `no_safety_vest` detector class, and the one `person`-labeled variant found (Voxel51) has person boxes on just 3.16% of images (see §3.3).

**Two critical data-quality findings** (deferred decisions, no data was altered):
1. **D-Fire native class order is `0=smoke, 1=fire`** — the OPPOSITE of this repo's demo `fire_smoke.yaml`. The Day 2 "drop-in compatible schema" claim is wrong for the official archive.
2. **Cross-split near-duplicate leakage exists in both acquired datasets** (D-Fire train↔test: 35,733 near-identical pairs at Hamming ≤ 2; Hard Hats train↔test: 5,438). The official splits must NOT be treated as unbiased test sets without a re-split/dedup decision.

---

## 1. Environment re-provisioning (task 1) — PASS

- CPU-first install order (documented Day 1 lesson): `torch==2.14.0+cpu`, `torchvision==0.29.0+cpu` from the PyTorch CPU index, then `requirements.txt`.
- Verified imports: torch 2.14.0+cpu · torchvision 0.29.0+cpu · ultralytics 8.4.165 · OpenCV 5.0.0 · numpy 2.5.2 — exact Day 1 pinned baseline.
- Pre-existing sandbox scipy/numba version warnings re-confirmed as unrelated to this project.
- Ultralytics settings initialized at `~/.config/Ultralytics/settings.json` (writable this time; the old `YOLO_CONFIG_DIR` workaround was not needed).

## 2. YOLO26n + Day 1 functionality (task 2) — PASS

- `scripts/check_environment.py`: all sections pass; device = cpu (intended).
- `scripts/detect.py --source images/test/bus.jpg`: ran successfully; annotated outputs written (`outputs/image_test/first_test_bus.jpg`, `outputs/image_test/detect/bus.jpg`). Model `models/yolo26n.pt` (5,544,453 bytes) intact from git.

## 3. Per-dataset reports (tasks 3–4, 8–9)

### 3.1 D-Fire — production fire/smoke — ACQUIRED

| Item | Value |
|---|---|
| Source | Official repo `gaia-solutions-on-demand/DFireDataset` (GAIA), "D-Fire dataset (only images and labels)" link |
| Exact URL | 1drv.ms shortlink in README → resolved to `https://my.microsoftpersonalcontent.com/personal/C0BD25B6B048B01D/_layouts/15/download.aspx?share=EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw` (public, no auth) |
| Version | Archive dated 2022-04-29 (inside-zip mtimes); 3,036,222,313 bytes (byte-verified) |
| License | **No license file/declaration** in official repo; public research release; citation required: Venâncio, Lisboa, Barbosa, Neural Computing and Applications (2022), https://link.springer.com/article/10.1007/s00521-022-07467-z |
| Number of images | **21,527** (17,221 train + 4,306 test) — matches official README exactly |
| Classes | NATIVE: **0 = smoke, 1 = fire** (see critical note) |
| Annotation format | YOLO (normalized), as distributed; official pre-split train/test (no val) |
| Boxes | fire 14,692 / smoke 11,865 = 26,557 total — matches official README; validator-counted 26,531 valid + 26 degenerate zero-area boxes **present in the official annotations** (`0 0.75… 0.25… 0.0 0.0`), left untouched, itemized in QA report |
| Annotation completeness | Negatives are explicit empty label files: 9,838 (matches official "None" row). Fire/smoke images: mixed per-image composition matches official rows exactly (only-fire 1,164 / only-smoke 5,867 / both 4,658) |
| Duplicate count (within-split, dHash64 Hamming ≤ 8) | 536,684 pairs (16,040 at distance 0) — surveillance-video-frame character |
| **Cross-split leakage** | **train↔test: 35,733 pairs ≤ 2 (82,530 ≤ 4)** — official split leaks near-identical frames; evidence grid `outputs/dataset_preview/production/dfire_cross_split_dup_evidence.jpg` |
| Train/val/test | 17,221 / 0 (none shipped) / 4,306 — kept as distributed, NOT re-split |
| Validation result | PASS with exceptions: 0 unreadable images, 0 pairing errors, 26 degenerate-box label lines (source-inherent; documented, not modified) |
| Visual QA | Previews: `outputs/dataset_preview/production/dfire/{train,test}_preview.jpg` — box alignment visually confirmed |
| Full QA JSON | `datasets/production/dfire/dfire_qa_report.json` |
| Provenance / config | `datasets/production/dfire/PROVENANCE.md`, `dfire_native.yaml` |

**Relation to lost-workspace historical numbers**: the historical production stats (17,806 imgs / 13,696 fire / 10,512 smoke / 7,665 negatives) do NOT match the official current archive (21,527 / 14,692 / 11,865 / 9,838). The lost set was a filtered/re-labeled derivative (documented night-frame noise removal). Re-deriving that filter is a pre-training decision requiring approval.

### 3.2 Hard Hats v2 — production PPE (helmet family) — ACQUIRED

| Item | Value |
|---|---|
| Source | Roboflow Universe "Hard Hats" v2 (`roboflow-universe-projects/hard-hats-fhbh5`), export `resized640_noAugmentation-FAST` (2023-01-16), via Hugging Face mirror `keremberke/hard-hat-detection` (verbatim export, embedded `README.roboflow.txt`) |
| Exact URLs | `https://huggingface.co/datasets/keremberke/hard-hat-detection/resolve/main/data/{train,valid,test}.zip` (public, no API key) |
| License | **CC BY 4.0** — declared in `README.dataset.txt` AND inside every COCO `licenses` entry |
| Number of images | **19,745** (13,782 / 3,962 / 2,001) |
| Classes | NATIVE: **0 = hardhat, 1 = no-hardhat** |
| Annotation format | COCO (Roboflow export) → converted to YOLO with `scripts/coco_to_yolo_hardhats.py` (native order kept); 0 missing, 0 failed |
| Annotation completeness | 55,393 boxes; every image labeled; source pre-processed to 640×640, no augmentation |
| Duplicate count (within-split, ≤ 8) | 153,893 pairs (7,111 at distance 0); hash-space sanity-checked (random-pair mean 31.6 ≈ 32) — real visual similarity, not an artifact |
| **Cross-split leakage** | train↔test **5,438 pairs ≤ 2**; train↔valid 10,974 ≤ 2; valid↔test 1,012 ≤ 2 — evidence grid `outputs/dataset_preview/production/hardhats_cross_split_dup_evidence.jpg` |
| Train/val/test | 13,782 / 3,962 / 2,001 (as distributed) |
| Validation result | **PASS — clean**: 0 unreadable, 0 pairing errors, 0 syntax errors |
| Visual QA | `outputs/dataset_preview/production/hardhats/{train,valid,test}_preview.jpg` — boxes aligned |
| Full QA JSON | `datasets/production/hardhats/hardhats_qa_report.json` |
| Provenance / config | `datasets/production/hardhats/PROVENANCE.md`, `hardhats_native.yaml` |

### 3.3 PPE 5-class architecture verification (per instruction)

| Target class | Evidence from acquired data | Verdict |
|---|---|---|
| `0 person` | Not a class in Hard Hats v2. The Voxel51 variant carries `person` on only **158/5,000 images (3.16%)** — independently verified from its CC0 manifest (`production/voxel51_hardhat_reference/VERIFICATION.json`). | NOT covered by acquired data; must come from COCO-pretrained base model (compliance-engine input) or a future source |
| `1 helmet` | Hard Hats v2 `hardhat`: 42,428 boxes | COVERED |
| `2 no_helmet` | Hard Hats v2 `no-hardhat`: 12,965 boxes | COVERED |
| `3 safety_vest` | Not in Hard Hats v2; SH17 (`16: safety-vest`) is disk-blocked | NOT covered yet — SH17 acquisition required |
| `4 gloves` | Not in Hard Hats v2; SH17 (`9: gloves`) is disk-blocked | NOT covered yet |
| `no_safety_vest` | No acquired dataset provides a usable detector class; nothing contradicts the compliance-engine derivation | **Stays compliance-engine logic** (unchanged) |

### 3.4 SH17 — BLOCKED (kept separate)

- Verified: official repo live; 8,099 images / 75,994 instances / 17 classes; labels only on Kaggle `mugheesahmad/sh17-dataset-for-ppe-detection`; **Kaggle metadata declares `CC BY-NC-SA 4.0` — this CONFIRMS the historically documented license** (GitHub README separately states educational/research-only + Pexels terms; both non-commercial).
- Blocked by: 14.2 GB archive vs ~3.1 GB free disk + Kaggle auth requirement. Details: `datasets/production/sh17/BLOCKED.md`; official class order archived at `datasets/production/sh17/class_order_official.yaml`.

### 3.5 GDUT-HD — NOT attempted

Optional source with unclear license per strategy doc §4; see `datasets/production/gdut_hd/NOT_ATTEMPTED.md`.

---

## 4. Preservation checks (tasks 5–7)

- Demo datasets untouched: `datasets/ppe` (17 imgs), `datasets/fire_smoke` (20 imgs), `datasets/raw/**`, `datasets/invalid/**` — no writes this phase (git-tracked; clean at commit time).
- Reconstruction documentation preserved verbatim: `RECONSTRUCTION_REPORT.md`, `datasets/DATASET_STRATEGY.md`, both existing `DATASET_REPORT.md` and all Day 1/Day 2 reports.
- No source-gated training views were created (no training data was assembled at all). The gating rule (strategy doc F5) remains designed-but-unimplemented until an approved training-view phase.

## 5. Credential / API-key scan (task 10) — CLEAN

- Text scan of `datasets/production/**` (`*.txt/*.yaml/*.yml/*.json/*.md/*.py`) for `api_key/apikey/secret/password/token/kaggle.json/AKIA/ghp_/gho_/github_pat_/sk-…/Bearer …`: **0 hits**.
- No `.env`, `kaggle.json`, `credentials*`, `*.key`, `*.pem` files anywhere under acquired data.
- No credentials were embedded in download URLs committed to the repo (OneDrive share token is a public un-gated link published by the dataset authors in their own README).

## 6. Tools added this phase (no existing pipeline script modified)

- `scripts/qa_acquired_dataset.py` (committed) — parameterized validator mirroring `scripts/validate_dataset.py` checks (integrity, pairing, syntax, class coverage) + dHash duplicate detection (within-split and cross-split) + preview grids; the repo's own validator remains hard-coded to the demo datasets and was left untouched.
- `scripts/coco_to_yolo_hardhats.py` (committed) — one-shot idempotent COCO→YOLO conversion for the Hard Hats export (native class order preserved).

## 7. Decisions required before any training (NOT decided here)

1. **Class order remap for D-Fire** (native 0=smoke/1=fire → repo schema 0=fire/1=smoke) or adopt native order everywhere.
2. **Night-frame / leakage-aware re-split** for D-Fire (re-derive the lost filter; de-duplicate near-identical frames ACROSS splits at minimum).
3. **Dedup policy** for Hard Hats v2 (same-scene clusters) + cross-split cleanup.
4. **SH17 acquisition path** (Kaggle credentials + ≥ 30 GB storage) or an alternative vest/gloves source.
5. **Person supervision strategy** for the compliance engine (COCO base model vs. a properly annotated person source).

## 8. STOP status

Per instruction: acquisition and validation are complete; **everything is stopped here**. No training, no tracking, no FastAPI, no database, no dashboard, no alerts. Awaiting explicit approval of the reacquired dataset state.
