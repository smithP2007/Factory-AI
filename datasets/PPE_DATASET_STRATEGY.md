# PS6 Core PPE Dataset Strategy

## 1. Mandatory classes

The PS6 discussion materials make these the main required PPE categories:

1. Helmet
2. Gloves
3. Goggles
4. Safety vest

Fire/smoke remains an optional hazard module.

## 2. Primary dataset candidate

**51ddhesh/PPE_Detection**

- Source: Hugging Face dataset `51ddhesh/PPE_Detection`
- Declared license: CC BY 4.0
- Format: YOLO-style object detection
- Declared object counts:
  - Vest: 4,418
  - Safety Shoe: 2,006
  - Mask: 2,763
  - Helmet: 2,703
  - Goggles: 1,431
  - Gloves: 2,693
  - Total: 16,014 objects
- Declared source structure: train / valid / test
- The source is described as manually annotated.

We use it as the starting point because it contains all four required PS6 PPE classes in one source and avoids having to create a synthetic mapping from unrelated object categories.

## 3. Target class mapping

Source → PS6 target:

- Helmet → helmet
- Gloves → gloves
- Goggles → goggles
- Vest → vest

Source classes intentionally excluded from the detector:

- Safety Shoe
- Mask

They are not part of the current PS6 core requirement. Their annotations are removed from the training labels rather than being silently renamed to a target class.

## 4. Person detection is separate

The source dataset does not provide a person class in the stated schema. The PS6 system therefore uses a frozen COCO-pretrained YOLO26n person detector for worker boxes. PPE detections are associated to those worker boxes in the compliance engine.

This preserves a clean separation:

- Detector A: person
- Detector B: helmet / gloves / goggles / vest
- Engine: worker-level compliance

## 5. Do not create no_* detector classes from absence

A missing helmet, glove, goggle, or vest is a relationship/compliance decision, not a physical object bounding box.

Do not label an empty region as `no_helmet` or treat an image without a vest annotation as a `no_vest` object.

The engine instead evaluates:

```text
worker + required PPE evidence
        ↓
compliant / non-compliant
        ↓
missing PPE list
```

## 6. Leakage control

Before training:

1. Verify every image opens.
2. Verify every label has valid YOLO syntax.
3. Reject invalid class IDs or invalid coordinates.
4. Compute exact image hashes across all splits.
5. Compute perceptual hashes where possible.
6. Quarantine cross-split duplicate / near-duplicate images.
7. Preserve a manifest of every removed item and its original split.
8. Re-run validation after quarantine.

For near-duplicates, start with dHash Hamming distance ≤8 as a quarantine candidate threshold. Do not report a zero-leakage claim until the post-quarantine scan actually returns zero cross-split matches at the chosen threshold.

## 7. Training classes

Final detector:

```yaml
names:
  0: helmet
  1: gloves
  2: goggles
  3: vest
```

## 8. Evaluation

Report per-class and overall:

- precision
- recall
- mAP50
- mAP50-95
- inference latency / FPS on the intended deployment hardware

Also inspect qualitatively:

- crowded scenes
- small goggles
- small gloves
- partial occlusion
- side views
- workers with overlapping bounding boxes
- workers with missing PPE

The four classes should not be summarized only by a single aggregate mAP number because goggles and gloves are substantially smaller targets than vests.

## 9. Compliance rule for the MVP

For a zone requiring all four:

```text
helmet = required
gloves = required
goggles = required
vest = required
```

A worker is compliant only when all four are associated with that worker.

For gloves, the initial MVP treats at least one confidently associated glove detection as evidence of gloves present. This is a documented simplification; later versions can model left/right hand compliance separately.

## 10. Optional future dataset

Ultralytics Construction-PPE contains helmet, gloves, vest, goggles and explicit missing-gear labels, but its dataset documentation declares AGPL-3.0. Do not mix it into the core training corpus until the team has reviewed the licensing implications for the intended use.
