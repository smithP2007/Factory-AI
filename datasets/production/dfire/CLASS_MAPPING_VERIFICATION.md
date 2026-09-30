# D-Fire Class Mapping Verification (FINAL)

**Final production mapping: `0 = smoke`, `1 = fire`.**

This reverses the repo's earlier assumption (`datasets/fire_smoke/fire_smoke.yaml`,
Day 2: `0 = fire`, `1 = smoke`), which does NOT match the official archive.

## Layer 1 - Statistical verification vs official README table

| Quantity | Official README | Measured on acquired archive | Match |
|---|---:|---:|---|
| Images containing only class 0 (= official "Only smoke") | 5867 | 5867 | True |
| Images containing only class 1 (= official "Only fire") | 1164 | 1164 | True |
| Images with both classes | 4658 | 4658 | True |
| Negatives (empty labels) | 9838 | 9838 | True |
| Boxes of class 0 (= official "Smoke") | 11865 | 11865 | True |
| Boxes of class 1 (= official "Fire") | 14692 | 14692 | True |
| Total images | 21,527 | 21527 | True |

**Layer 1 result: PASS - the composition of class-0-only and class-1-only images cannot match the official semantic rows unless 0=smoke and 1=fire.**

## Layer 2 - Visual verification (human-readable grids)

Deterministic samples rendered with class-colored boxes
(class 0 = cyan `0:smoke`, class 1 = orange `1:fire`):

1. `outputs/dataset_preview/production/dfire_mapping_verification/class0_only_expect_smoke.jpg`
   - images whose labels contain ONLY class 0; every box must sit on a smoke plume/plume region, none on flames.
2. `outputs/dataset_preview/production/dfire_mapping_verification/class1_only_expect_fire.jpg`
   - images whose labels contain ONLY class 1; every box must sit on flames/fire, none on smoke.
3. `outputs/dataset_preview/production/dfire_mapping_verification/both_expect_smoke_cyan_fire_orange.jpg`
   - mixed images; cyan must cover smoke, orange must cover fire in the same frame.

Recorded verdicts (reviewer: project owner + AI reviewer, 2026-10-01):

| Grid | Expected | Verdict |
|---|---|---|
| class 0 only | cyan boxes on smoke | VERIFIED - cyan encloses smoke plumes; no flames inside class-0 boxes |
| class 1 only | orange boxes on fire | VERIFIED - orange encloses flames/burning objects; no plume-only boxes |
| both classes | cyan=smoke, orange=fire | VERIFIED - consistent in mixed frames |

## Consequence for configuration

- `datasets/production/dfire/dfire_native.yaml` is the production fire/smoke config and uses `names: {0: smoke, 1: fire}`.
- The demo `datasets/fire_smoke/fire_smoke.yaml` (0=fire, 1=smoke) is a self-consistent legacy demo schema and was NOT relabeled;
  production training must use the production YAML, never the demo one.

Degenerate (zero-area) boxes present in the official annotations, unchanged: 18.

## Reproduce

```
python scripts/verify_dfire_class_mapping.py
```
