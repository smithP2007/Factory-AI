# Construction Safety — Acquisition Provenance (EVIDENCE-ONLY, NOT approved for training)

**Acquired**: 2026-10-01 (dataset correction phase, per reviewer instruction to
investigate obtainable vest/gloves/person sources)
**Status**: ACQUIRED + validated. **NOT approved for training** — see role below.

## Source

| Field | Value |
|---|---|
| Dataset | "Construction Site Safety" v1 (original_raw-images), 398 images |
| Origin | Roboflow Universe `roboflow-universe-projects/construction-site-safety` (composite: images cloned from safety-vests / people-and-ladders / personal-protective-equipment-combined-model / people-detection-general / construction-madness + 2 YouTube videos + MIT Indoor nulls) |
| Mirror | Hugging Face `keremberke/construction-safety-object-detection` (roboflow2huggingface verbatim export) |
| URLs | `.../resolve/main/data/{train,valid,test}.zip` (27.4 MB total) |
| License | **CC BY 4.0** (README.dataset.txt) |
| Access | None (public, no API key) |
| Splits | 307 / 57 / 34 (official) |

## Measured statistics (post COCO→YOLO conversion, native alphabetical class order)

17 classes. Boxes relevant to the PS6 5-class target:

| Target class | Boxes here | Verdict |
|---|---:|---|
| person | 69 (42/23/4) | present but tiny |
| safety_vest | 53 (45/6/2) | present but tiny |
| gloves | 18 (11/5/2) | present but tiny |
| no_safety_vest | 38 (21/14/3) | present but tiny — see note |
| helmet / no_helmet | 436 / 108 | redundant with Hard Hats v2 |

QA: 0 unreadable, 0 missing labels, 0 syntax errors, 16 within-split near-dup
pairs (dHash <= 8). Full report: `construction_safety_qa_report.json`.

## WHY NOT APPROVED FOR TRAINING

1. **Scale**: 53 vest / 18 glove / 69 person boxes cannot train (or even
   evaluate) production classes; it would memorize 398 images.
2. **Cross-dataset duplication with Hard Hats v2 (measured)**:
   `cross_dataset_overlap_vs_hardhats.json` — **236 exact duplicates (dHash d=0)**,
   864 pairs <= 2, 4,724 pairs <= 8 across the two datasets (shared Roboflow
   Universe ancestry; identical base filenames). Merging or cross-evaluating
   these two sources without overlap filtering would corrupt both training and
   evaluation.
3. **Visible-but-unlabeled risk**: composite construction scenes; annotation
   policy for vest/gloves completeness across ALL persons cannot be verified at
   this scale (Final Dataset Rule 6).

## VALUE (why it was kept)

Concrete evidence that (a) `no-safety vest` exists as an annotated detector
class in at least one licensed source — recorded as data-point for the
compliance-engine vs detector-class question, without changing our schema —
and (b) no obtainable no-auth source provides production-scale
person/vest/gloves supervision. Kept separate; never merged.
