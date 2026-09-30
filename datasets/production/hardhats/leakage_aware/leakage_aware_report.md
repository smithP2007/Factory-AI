# Leakage-aware re-split — hardhats

Method: dHash64 perceptual grouping (<= 8) + source-key grouping + temporal adjacency (D-Fire series); exact duplicates (d=0) quarantined; whole groups assigned to splits, stratified greedy largest-first (seed 20261001). No images copied - split lists reference originals.

| Metric | Value |
|---|---:|
| Images (official, all splits) | 19745 |
| Exact duplicates quarantined (dHash d=0) | 2318 |
| Images used | 17427 |
| Groups (near-dup/source-connected) | 13701 |
| train: images / groups | 12199 / 8473 |
| train: boxes {'hardhat': 26727, 'no-hardhat': 6765} | |
| val: images / groups | 3486 / 3486 |
| val: boxes {'hardhat': 8301, 'no-hardhat': 2738} | |
| test: images / groups | 1742 / 1742 |
| test: boxes {'hardhat': 4171, 'no-hardhat': 1402} | |

## Leakage verification

| Cross-split near-duplicate pairs | Official split (OLD) | Leakage-aware (NEW) |
|---|---:|---:|
| Hamming <= 2 | 17424 | 0 |
| Hamming <= 8 | 83578 | 0 |

(OLD column counts the official split as distributed, all images - it reproduces the reacquisition audit. Post-quarantine OLD at <=2/<=8: 1159 / 15954 - the exact-duplicate quarantine alone removes most of the leakage, grouping removes the rest.)

**Verdict: PASS - zero cross-split near-duplicate pairs at Hamming <= 8**

Intra-split near-duplicate pairs (same scene, same split, <= 8): 45043 - expected for video-frame data, harmless for training/evaluation integrity.

## Files

- `train.txt` / `val.txt` / `test.txt` - absolute image paths
- `quarantine_manifest.csv` - every exact duplicate removed, with its kept representative
- `leakage_aware_report.json` - full machine-readable report
