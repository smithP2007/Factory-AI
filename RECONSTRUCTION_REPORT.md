# Factory-AI Reconstruction Report (Post-Workspace-Reset)

**Date**: 2026-10-01 (UTC+8)
**Basis of reconstruction**: surviving GitHub repository `smithP2007/Factory-AI` @ `main` = `a1f0cba` + surviving local plan documents (`upload/PS6_Day_1_Implementation_Plan-2.txt`, `upload/PS6_Day_2_Implementation_Plan-1.txt`) + user-provided historical documentation of the lost workspace.
**Method**: fresh verification of live remote HEAD → full tree inventory of `a1f0cba` (315 tracked files) → full-history keyword/number search → on-disk cross-check of every documented count. No files were invented; every claim below carries a provenance tag.

**Provenance legend**

| Tag | Meaning |
|---|---|
| [VERIFIED] | Present in surviving repo at `a1f0cba`; byte/count cross-checked in this pass |
| [HISTORICAL] | User-documented fact about the lost workspace; **not** independently verifiable from surviving artifacts |
| [ANALYSIS] | Inference drawn in this report; explicitly not verification |
| [PROPOSED] | Under consideration; requires explicit approval before adoption |

---

## 1. What Survived

### 1.1 Git state [VERIFIED]

- Live remote `refs/heads/main` = `a1f0cba` (checked via `git ls-remote` against github.com at reconstruction time; repo is public, anonymous read works).
- Full history is exactly **2 commits**: `5456d25` "Initial commit" → `a1f0cba` "PS6 Safety AI - Day 1 + Day 2 complete".
- No stash, no dangling/unreachable objects, no other refs. **Nothing is recoverable from git beyond `a1f0cba`.**
- History is fully preserved; this reconstruction commits on top of `a1f0cba` (no force-push, no rewrite).

### 1.2 Surviving artifacts by category [VERIFIED]

| Category | Contents | Cross-check result |
|---|---|---|
| Pipeline scripts (9) | `detect.py`, `check_environment.py`, `benchmark.py` (Day 1); `download_wikimedia.py`, `auto_label.py`, `split_dataset.py`, `validate_dataset.py`, `visualize_dataset.py`, `quarantine_duplicates.py` (Day 2) | All present, tracked |
| PPE demo dataset | 17 images + 17 labels, split 14/2/1 (train/val/test); 34 `person` boxes (31/2/1 per split) | Matches `datasets/ppe/validation_report.json` exactly; on-disk counts match |
| Fire/smoke demo dataset | 20 images + 20 labels, split 16/2/2; 10 `fire` + 10 `smoke` boxes (train: 6 fire + 10 smoke; val/test: 2 fire each) | Matches `datasets/fire_smoke/validation_report.json` exactly; on-disk counts match |
| Raw downloads | 50 raw images + **50 `.license.json` provenance sidecars** under `datasets/raw/{ppe,fire_smoke}/raw_images/<class>/` | All tracked in git |
| Quarantine | `datasets/invalid/ppe/duplicates/` — 7 near-duplicate images + `quarantine_manifest.json` | Manifest present and tracked |
| Configs | `datasets/ppe/ppe.yaml` (MVP: `0: person`; full 6-class schema in comments), `datasets/fire_smoke/fire_smoke.yaml` (`0: fire`, `1: smoke`, D-Fire-compatible), `datasets/class_mapping.yaml` | Present |
| Reports | `datasets/DATASET_REPORT.md`, `outputs/day1_report.json`, `outputs/day2_report.json`, `datasets/auto_label_summary.json`, 2× `validation_report.json`, `outputs/benchmark_results.json` | All parsed successfully in this pass |
| Previews & outputs | 17 annotated preview PNGs under `outputs/dataset_preview/`; Day 1 detection outputs (`bus.jpg`, `people-detection.avi`) | Present |
| Model | `models/yolo26n.pt` (5,544,453 bytes, COCO-pretrained base model) | Present |
| Media & placeholders | `images/test/bus.jpg`, `videos/test/{people-detection.mp4, oceans.mp4}`, `backend/.gitkeep`, `frontend/.gitkeep`, `tests/.gitkeep`, `README.md`, `LICENSE`, `requirements.txt`, `.gitignore` | Present |
| Outside the repo (local) | Day 1 + Day 2 implementation plans (`upload/`); `download/source.zip` (97 MB, 380 files = snapshot of `a1f0cba` + plans) | Present |

**Conclusion**: the *entire demo-scale Day 1 + Day 2 pipeline* (scripts, demo data, configs, validation reports, documentation) survived on GitHub and is intact.

### 1.3 Surviving documented Day 2 statistics [VERIFIED]

These are the **only** dataset statistics that survive, and they describe the **demo** subset — not production:

- PPE demo: **17 images**, **34 person boxes**; helmet/vest/gloves classes deferred (no clean CC source at the time).
- Fire/smoke demo: **20 images**, **10 fire + 10 smoke** whole-image boxes.
- 7 PPE near-duplicates quarantined; 6 images skipped (no detectable person).
- Both datasets: validation **PASS**, 0 errors.

---

## 2. What Was Lost

### 2.1 Production-scale datasets [HISTORICAL — lost, never pushed to GitHub]

The following existed only in the lost workspace and are **absent from every surviving artifact** (full-history search: 0 hits for all figures below):

- **PPE production set**: 5,096 images; instances — helmet 19,071 / no_helmet 5,891 / safety_vest 126 / no_safety_vest 443 / gloves 191 / person 595.
- **Fire/smoke production set**: 17,806 images; 13,696 fire boxes / 10,512 smoke boxes / 7,665 negatives.

### 2.2 Lost process knowledge [HISTORICAL]

- The Day 2 remediation decisions and findings (Voxel51 person-annotation incompleteness, unlabeled-vest problem, D-Fire night-frame noise, SH17 investigation and licensing).
- Any workspace-only scripts, filter lists, or notes used to produce the production sets (e.g., whatever filtering reduced D-Fire to 17,806 images — see §5.2).
- The Python ML environment in this workspace also did not survive the reset (torch/ultralytics not importable post-reset) and must be re-provisioned.

### 2.3 Never existed (not lost)

- Trained PPE/fire-smoke model weights (`ppe_best.pt`, `fire_smoke_best.pt`) — training was explicitly planned for Day 3 and never started.

---

## 3. What Can Be Recreated Exactly — and What Was Done Now

| Item | Status |
|---|---|
| Dataset directory structure & split/config files | **Already survived intact** — verified against reports; no recreation needed. Nothing was fabricated to fill gaps. |
| Pipeline reproducibility (demo scale) | [VERIFIED] Scripts survived; documented re-run commands are in `datasets/DATASET_REPORT.md` §12. |
| Day 2 dataset strategy documentation | **Recreated now** as `datasets/DATASET_STRATEGY.md` — every fact tagged [VERIFIED] / [HISTORICAL] / [PROPOSED]; no invented statistics. |
| This reconstruction report | Created now (`RECONSTRUCTION_REPORT.md`). |

---

## 4. What Must Be Reacquired

1. **Production PPE dataset** (~5,096 images) [HISTORICAL scale]. Candidate sources with constraints (from surviving docs + user history):
   - Hard Hat Workers v2 (Roboflow Universe; API key required) — covers helmet / no_helmet.
   - SH17 (CC BY-NC-SA 4.0 [HISTORICAL]) — PPE supplement; **non-commercial** restriction; usable only with per-source gating (see strategy doc §3, F5).
   - GDUT-HD — gloves/vest coverage; license unclear (flagged in surviving `DATASET_REPORT.md`).
2. **Production fire/smoke dataset** (~17,806 images [HISTORICAL]) — D-Fire via Kaggle (authentication required; documented in surviving `DATASET_REPORT.md` §12), plus re-derivation of whatever filtering produced the final set.
3. **Annotation-completeness decisions** — how person/vest incompleteness (Voxel51, unlabeled vests) will be handled per source.
4. **Python environment rebuild** — `torch` (CPU build first, per documented Day 1 lesson), then `pip install -r requirements.txt`.

No dataset downloads were performed in this reconstruction (per instructions).

---

## 5. Which Historical Statistics Are Independently Verifiable

### 5.1 Verification result: **none of the production-scale numbers are verifiable** [VERIFIED NEGATIVE]

A full search of the working tree **and complete git history** at `a1f0cba` for `5096 / 19071 / 5891 / 126* / 443 / 191* / 595 / 17806 / 13696 / 10512 / 7665` (and comma-formatted variants) returned **zero hits**. Keywords `Voxel51`, `SH17`, `NC-SA` also return zero hits. The surviving `day2_report.json` documents only the demo run (17 + 20 images).

\* 126/191/595/443 as bare numbers can occur coincidentally in binary blobs; the statement holds for every dataset-report, YAML, JSON, and Markdown artifact.

### 5.2 Consistency analysis (context, not verification) [ANALYSIS]

- The fire/smoke historical figures (smoke **10,512**, negatives **7,665**, fire **13,696**, images **17,806**) closely track D-Fire's commonly published statistics (≈13.7k fire boxes, 10,512 smoke boxes, 7,665 negatives, 21,175 images). The image shortfall (17,806 < 21,175) is consistent with the documented night-frame noise removal. This suggests the lost production fire/smoke set was **D-Fire-derived after filtering**, but only re-acquisition and re-counting can confirm it.
- The PPE historical figures are internally consistent with the documented findings: `person` (595) ≪ `helmet` (19,071) matches the "incomplete person annotations were excluded" finding; very low `safety_vest` (126) / `no_safety_vest` (443) / `gloves` (191) counts match the "unlabeled vest problems" finding.

**Ruling**: all production-scale numbers remain [HISTORICAL-UNVERIFIED] until the datasets are re-acquired and recounted. They are recorded as historical documentation in `datasets/DATASET_STRATEGY.md` and must not be treated as current dataset state.

---

## 6. Config / Documentation Drift Found During Reconstruction

1. `ppe.yaml` and `fire_smoke.yaml` still carry `path: /home/z/my-project/PS6-Safety-AI/...` from the original workspace layout. **Must be updated before any Day 3 training.** Left unchanged deliberately (Day 3 is not approved).
2. `DATASET_REPORT.md` §10 says sidecars sit next to images in `raw_all/images/`; in the surviving tree they are under `datasets/raw/*/raw_images/<class>/` (50 tracked). Documentation drift only; the sidecar files themselves are intact.
3. `auto_label_summary.json` reports 43 person boxes at the auto-label stage vs 34 in the final splits — explained by the documented 7 quarantined duplicates + 2 val/test reassignments; noted for auditability.

---

## 7. Actions Taken in This Reconstruction

- Verified live GitHub HEAD (`a1f0cba`) and full history integrity.
- Inventoried all 315 tracked files; cross-checked every documented count against disk.
- Recreated `datasets/DATASET_STRATEGY.md` (latest Day 2 dataset strategy, provenance-tagged).
- Created this report.
- Committed both documents **on top of** `a1f0cba` (history preserved; no force, no rewrite). **Not pushed** — no credentials available, and the previously exposed PAT should remain revoked.
- Did **not**: start Day 3, train anything, download any dataset, or alter any existing config.

## 8. Awaiting Explicit Approval

Per instruction: **STOP.** No Day 3 work (schema migration, dataset acquisition, source-gated training, compliance engine, environment rebuild for training) begins until the reconstructed state is explicitly approved.
