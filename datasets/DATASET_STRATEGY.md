# PS6 Dataset Strategy — v2 (Reconstructed)

**Status**: Reconstructed 2026-10-01 after workspace reset. Supersedes nothing and rewrites nothing — it documents the **latest approved Day 2 dataset strategy** as recovered from user-provided historical records, anchored to the surviving repo state at `a1f0cba`.

**Provenance legend**: [VERIFIED] = present in surviving repo · [HISTORICAL] = user-documented, not independently verifiable · [ANALYSIS] = inference, not verification · [PROPOSED] = under consideration, subject to verification.

---

## 1. Target Detection Architecture

### 1.1 Fire/Smoke — final [VERIFIED]

```
0: fire
1: smoke
```

Class indices intentionally match the D-Fire schema (documented in surviving `fire_smoke.yaml` and `class_mapping.yaml`) so production data drops in without renumbering.

### 1.2 PPE — architecture under consideration [PROPOSED — subject to verification]

```
0: person
1: helmet
2: no_helmet
3: safety_vest
4: gloves
```

**`no_safety_vest` is NOT a detector class.** Non-compliance (person without vest) is derived by the **compliance engine**: a `person` instance with no sufficiently overlapping `safety_vest` box is flagged non-compliant (overlap rule / IoU threshold TBD).

Rationale for the compliance-engine approach:

1. **Label sparsity**: historically only 126 `safety_vest` + 443 `no_safety_vest` instances existed versus 19,071 `helmet` [HISTORICAL]. A detector class trained on that few positives — and on negatives that are mostly *unlabeled* rather than truly negative — is unreliable.
2. **False-negative control**: the same principle adopted for SH17 source-gating (§3, F5) applies: training a "no_vest" class against images where visible vests are unlabeled teaches the detector to miss real objects. Deriving non-compliance at inference time from `person` + missing `safety_vest` avoids injecting that noise into the detector.
3. **Subject to verification** because: the compliance rule needs an overlap threshold validated on re-acquired data, and `person`-recall errors propagate directly into non-compliance events.

**Migration note**: the surviving `ppe.yaml` comments carry an older 6-class schema (`0..5` incl. `no_safety_vest` as class 4). Adopting the 5-class schema above is a config migration that happens **only after approval** — indices for `gloves` shift from 5 to 4, so any legacy labels must be remapped.

---

## 2. Production Dataset Inventory (historical, unverified)

All figures in this section are [HISTORICAL — lost workspace documentation]. They were **not found in any surviving artifact** (see `RECONSTRUCTION_REPORT.md` §5) and must be re-established by re-acquisition and re-count.

### PPE production set

| Item | Value [HISTORICAL] |
|---|---:|
| Images | 5,096 |
| helmet instances | 19,071 |
| no_helmet instances | 5,891 |
| safety_vest instances | 126 |
| no_safety_vest instances | 443 |
| gloves instances | 191 |
| person instances | 595 |

Known quality issues [HISTORICAL]: incomplete Voxel51 person annotations (excluded), unlabeled-vest problem (F2 below).

### Fire/smoke production set

| Item | Value [HISTORICAL] |
|---|---:|
| Images | 17,806 |
| fire boxes | 13,696 |
| smoke boxes | 10,512 |
| negatives (no fire/smoke) | 7,665 |

Known quality issues [HISTORICAL]: D-Fire night-frame noise (F3 below). [ANALYSIS] The smoke/negative counts exactly match D-Fire's commonly published figures and the image count is lower, consistent with a D-Fire-derived set after filtering — to be confirmed on re-acquisition.

### Demo datasets (surviving, current state) [VERIFIED]

| Dataset | Images | Boxes | Status |
|---|---:|---|---|
| ppe (Wikimedia demo) | 17 | 34 person | PASS validation; placeholder only |
| fire_smoke (Wikimedia demo) | 20 | 10 fire + 10 smoke | PASS validation; D-Fire-compatible schema |

---

## 3. Day 2 Remediation Findings (reconstructed from user documentation)

All findings below are [HISTORICAL] (they appear in no surviving artifact) but are binding project knowledge going forward.

- **F1 — Voxel51 person annotations incomplete → excluded.** The person-annotation subset from the Voxel51 source was incomplete and was excluded rather than risk corrupting `person` supervision. Consequence: `person` instance count (595) is far below `helmet` (19,071) [consistent with §2 — ANALYSIS].
- **F2 — Unlabeled vest problems identified.** Many images show visible vests with no vest boxes. Treating those as negatives would poison `safety_vest` training; this drove both the low vest counts and the compliance-engine proposal (§1.2).
- **F3 — D-Fire night-frame noise documented.** Night images in D-Fire are noisy for fire/smoke detection; a filtering pass was applied in the lost workspace (17,806 kept). The filter criteria must be re-derived and re-applied on re-acquisition.
- **F4 — SH17 investigated as PPE supplement; license CC BY-NC-SA 4.0.** SH17 can supplement PPE classes (incl. vest/gloves coverage) but is **NonCommercial-ShareAlike**: acceptable for PS6 academic/non-commercial use, **not** usable in any commercial deployment; ShareAlike obligations apply to redistribution. Per-source license provenance must be preserved (same discipline as the surviving `.license.json` sidecars [VERIFIED pattern]).
- **F5 — Source-gated SH17 training rule.** SH17 images may only contribute to a class on images where all visible instances of that class are labeled. If visible objects of a class are unlabeled in an SH17 image, that image must be excluded for that class — otherwise the model is trained to produce **false negatives** on real objects. Gate is per-source, per-class (a per-image/per-class allowlist).

---

## 4. Acquisition Roadmap (Day 3 gate — blocked until approval)

| Priority | Source | Covers | Access | License | Blocker |
|---|---|---|---|---|---|
| 1 | D-Fire (re-acquire) | fire, smoke, negatives | Kaggle, auth required | As published (verify at download) | Kaggle credentials; re-derive night filter |
| 2 | Hard Hat Workers v2 | helmet, no_helmet, person | Roboflow Universe API key | Check at download | API key |
| 3 | SH17 (source-gated) | safety_vest, gloves, PPE supplement | Public download | **CC BY-NC-SA 4.0** [HISTORICAL] | Non-commercial + F5 gating implementation |
| Optional | GDUT-HD | vest, gloves | Public | Unclear (flagged in surviving DATASET_REPORT.md) | License clarification before use |

Person-annotation strategy must be re-decided before merging (F1: the incomplete Voxel51 source stays excluded unless a complete source is found).

---

## 5. Pipeline Re-Run Plan (surviving scripts) [VERIFIED]

The Day 2 pipeline scripts all survived and are dataset-agnostic. After any production dataset is placed under `datasets/raw/<name>/`:

```bash
python scripts/split_dataset.py --name <name>
python scripts/validate_dataset.py --name <name>
python scripts/visualize_dataset.py --name <name>
python scripts/quarantine_duplicates.py   # if duplicates are flagged by the validator
```

To be added in Day 3 (not yet written): a **source-gating pre-processor** implementing F5 (per-source, per-class allowlist) and the **compliance engine** module (§1.2).

---

## 6. Config Migration Notes (blocked until approval)

1. `ppe.yaml`: update `path:` (currently points at the pre-reset workspace `/home/z/my-project/PS6-Safety-AI/...`), then adopt the approved schema — either the 6-class commented schema or the 5-class §1.2 proposal (with gloves index remap).
2. `fire_smoke.yaml`: update `path:`; schema already correct.
3. `class_mapping.yaml`: extend per-source mappings (HHW `head` → `no_helmet`, etc.) once sources are chosen.

---

## 7. Explicit Non-Goals Until Approval

No Day 3 training, no dataset downloads/merging, no schema/config rewrites, no compliance-engine implementation. The demo datasets remain the only on-disk data and are placeholders, not training sets.
