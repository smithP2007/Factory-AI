# Leakage-aware re-split — dfire

Method: dHash64 perceptual grouping (<= 8) + source-key grouping + temporal adjacency (D-Fire series); exact duplicates (d=0) quarantined; whole groups assigned to splits, stratified greedy largest-first (seed 20261001). No images copied - split lists reference originals.

| Metric | Value |
|---|---:|
| Images (official, all splits) | 21527 |
| Exact duplicates quarantined (dHash d=0) | 4150 |
| Images used | 17377 |
| Groups (near-dup/source-connected) | 8635 |
| train: images / groups | 13901 / 5337 |
| train: boxes {'smoke': 8796, 'fire': 11812} | |
| val: images / groups | 1739 / 1650 |
| val: boxes {'smoke': 786, 'fire': 967} | |
| test: images / groups | 1737 / 1648 |
| test: boxes {'smoke': 794, 'fire': 997} | |

## Leakage verification

| Cross-split near-duplicate pairs | Official split (OLD) | Leakage-aware (NEW) |
|---|---:|---:|
| Hamming <= 2 | 35733 | 0 |
| Hamming <= 8 | 248594 | 0 |

(OLD column counts the official split as distributed, all images - it reproduces the reacquisition audit. Post-quarantine OLD at <=2/<=8: 3759 / 60458 - the exact-duplicate quarantine alone removes most of the leakage, grouping removes the rest.)

**Verdict: PASS - zero cross-split near-duplicate pairs at Hamming <= 8**

Intra-split near-duplicate pairs (same scene, same split, <= 8): 145733 - expected for video-frame data, harmless for training/evaluation integrity.

## Files

- `train.txt` / `val.txt` / `test.txt` - absolute image paths
- `quarantine_manifest.csv` - every exact duplicate removed, with its kept representative
- `leakage_aware_report.json` - full machine-readable report
