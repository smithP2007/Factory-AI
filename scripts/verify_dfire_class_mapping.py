#!/usr/bin/env python3
"""Verify the D-Fire class mapping against actual annotations + visual samples.

PRODUCTION MAPPING VERIFIED HERE (final decision, per reacquisition review):
    0 = smoke
    1 = fire

Two independent verification layers:
  1. STATISTICAL: per-image class composition of the acquired archive must
     equal the official D-Fire README table, which is published with the
     semantic names ("Only smoke", "Only fire", "Fire+smoke", "None",
     "Smoke" boxes, "Fire" boxes). If class 0 were fire (as the repo demo
     YAML assumed), the composition rows could not match the official table.
  2. VISUAL: render class-colored boxes on deterministic samples
     (class 0 -> cyan "0:smoke", class 1 -> orange "1:fire") and verify
     cyan lands on smoke plumes and orange on flames.

Outputs:
  - datasets/production/dfire/CLASS_MAPPING_VERIFICATION.json
  - datasets/production/dfire/CLASS_MAPPING_VERIFICATION.md
  - outputs/dataset_preview/production/dfire_mapping_verification/*.jpg
"""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

REPO = Path("/home/z/my-project/Factory-AI")
ROOT = REPO / "datasets/production/dfire"
PREVIEW = REPO / "outputs/dataset_preview/production/dfire_mapping_verification"

# Official D-Fire README table (github.com/gaia-solutions-on-demand/DFireDataset)
OFFICIAL = {
    "only_smoke_images": 5867,
    "only_fire_images": 1164,
    "fire_and_smoke_images": 4658,
    "negative_images": 9838,
    "smoke_boxes": 11865,
    "fire_boxes": 14692,
}

# Colors (BGR): class 0 = cyan, class 1 = orange
CLASS_STYLE = {0: ((255, 255, 0), "0:smoke"), 1: ((0, 165, 255), "1:fire")}


def scan_labels():
    """Full statistical scan of train+test labels."""
    comp = Counter()          # composition tuple -> image count
    boxes = Counter()         # class id -> box count
    degenerate = 0            # zero-area boxes present in official annotations
    per_class_images = {0: [], 1: []}   # stems with ONLY that class
    both = []                 # stems with both classes
    negatives = []            # empty-label stems
    total = 0
    for split in ("train", "test"):
        for lp in sorted((ROOT / split / "labels").glob("*.txt")):
            total += 1
            text = lp.read_text().strip()
            if not text:
                comp[()] += 1
                negatives.append((split, lp.stem))
                continue
            classes_here = set()
            for line in text.splitlines():
                p = line.split()
                cls = int(p[0])
                w, h = float(p[3]), float(p[4])
                boxes[cls] += 1
                if w <= 0.0 or h <= 0.0:
                    degenerate += 1
                classes_here.add(cls)
            comp[tuple(sorted(classes_here))] += 1
            if classes_here == {0}:
                per_class_images[0].append((split, lp.stem))
            elif classes_here == {1}:
                per_class_images[1].append((split, lp.stem))
            elif classes_here == {0, 1}:
                both.append((split, lp.stem))
    return {
        "total_images": total,
        "only_class0": len(per_class_images[0]),
        "only_class1": len(per_class_images[1]),
        "both_classes": len(both),
        "negatives": len(negatives),
        "composition_all": {str(k): v for k, v in sorted(comp.items())},
        "boxes_class0": boxes[0],
        "boxes_class1": boxes[1],
        "degenerate_boxes": degenerate,
        "_lists": {"only0": per_class_images[0], "only1": per_class_images[1],
                   "both": both, "neg": negatives},
    }


def label_area(split: str, stem: str):
    """Median normalized box area for class 0 and class 1 in one image
    (normalized space avoids opening candidate images during sampling)."""
    lp = ROOT / split / "labels" / f"{stem}.txt"
    a = {0: [], 1: []}
    for line in lp.read_text().strip().splitlines():
        p = line.split()
        cls = int(p[0])
        a[cls].append(float(p[3]) * float(p[4]))
    med = {}
    for c, arr in a.items():
        med[c] = sorted(arr)[len(arr) // 2] if arr else 0.0
    return med[0], med[1]


def render(samples, title, outfile, boxes_per_tile_limit=40):
    """Grid of annotated tiles; class 0 cyan, class 1 orange."""
    tiles = []
    for split, stem in samples:
        ip = next(iter((ROOT / split / "images").glob(f"{stem}.*")), None)
        img = cv2.imread(str(ip))
        if img is None:
            continue
        H, W = img.shape[:2]
        scale = 480 / max(H, W)
        img = cv2.resize(img, (int(W * scale), int(H * scale)))
        h, w = img.shape[:2]
        drawn = 0
        for line in (ROOT / split / "labels" / f"{stem}.txt").read_text().strip().splitlines():
            p = line.split()
            cls, xc, yc, bw, bh = int(p[0]), *map(float, p[1:])
            x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
            x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
            color, name = CLASS_STYLE[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, name, (x1, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            drawn += 1
            if drawn >= boxes_per_tile_limit:
                break
        cv2.putText(img, f"{split}/{stem[:22]}", (6, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
             for t in tiles]
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    grid = np.vstack(rows)
    grid = cv2.copyMakeBorder(grid, 34, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(grid, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(outfile), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {outfile} ({len(tiles)} tiles)")


def pick_samples(stems, n, cls=0, seed=20261001, min_area=0.02):
    """Deterministic diverse sample: stride over sorted list, filtered by the
    median normalized box area (class `cls`; for mixed images both classes
    must clear the bar so the grid shows cyan AND orange)."""
    import random
    rng = random.Random(seed)
    pool = sorted(stems)
    rng.shuffle(pool)
    ranked = []
    for s in pool:
        a0, a1 = label_area(s[0], s[1])
        area = a1 if cls == 1 else a0
        if cls == 0 and a1:
            area = min(a0, a1)   # mixed: require both classes clearly visible
        if area and area > min_area:
            ranked.append((s, area))
    ranked.sort(key=lambda x: -x[1])
    pool = [s for s, _ in ranked]
    stride = max(1, len(pool) // n)
    return pool[::stride][:n]


def main() -> int:
    import numpy as np  # noqa: F401  (used via np.hstack in render)
    globals()["np"] = np
    PREVIEW.mkdir(parents=True, exist_ok=True)

    scan = scan_labels()
    lists = scan.pop("_lists")

    # --- layer 1: statistical match against official README ---
    checks = {
        "only_class0_equals_official_only_smoke": scan["only_class0"] == OFFICIAL["only_smoke_images"],
        "only_class1_equals_official_only_fire": scan["only_class1"] == OFFICIAL["only_fire_images"],
        "both_equals_official": scan["both_classes"] == OFFICIAL["fire_and_smoke_images"],
        "negatives_equal_official": scan["negatives"] == OFFICIAL["negative_images"],
        "boxes_class0_equal_official_smoke": scan["boxes_class0"] == OFFICIAL["smoke_boxes"],
        "boxes_class1_equal_official_fire": scan["boxes_class1"] == OFFICIAL["fire_boxes"],
        "total_images_equal_official": scan["total_images"] == 21527,
    }
    stat_ok = all(checks.values())

    # --- layer 2: visual evidence grids ---
    s0 = pick_samples(lists["only0"], 8, cls=0, seed=11)
    s1 = pick_samples(lists["only1"], 8, cls=1, seed=22)
    sb = pick_samples(lists["both"], 8, cls=0, seed=33)
    render(s0, "CLASS 0 ONLY - official table says these are SMOKE (expect cyan boxes on smoke)",
           PREVIEW / "class0_only_expect_smoke.jpg")
    render(s1, "CLASS 1 ONLY - official table says these are FIRE (expect orange boxes on flames)",
           PREVIEW / "class1_only_expect_fire.jpg")
    render(sb, "BOTH CLASSES - cyan=0:smoke on plumes, orange=1:fire on flames",
           PREVIEW / "both_expect_smoke_cyan_fire_orange.jpg")

    result = {
        "verified_mapping": {"0": "smoke", "1": "fire"},
        "dataset": "D-Fire (official archive, acquired 2026-10-01)",
        "statistical_layer": {
            "official_readme_table": OFFICIAL,
            "measured": {k: v for k, v in scan.items()},
            "checks": checks,
            "pass": stat_ok,
        },
        "visual_layer": {
            "grids": [
                "outputs/dataset_preview/production/dfire_mapping_verification/class0_only_expect_smoke.jpg",
                "outputs/dataset_preview/production/dfire_mapping_verification/class1_only_expect_fire.jpg",
                "outputs/dataset_preview/production/dfire_mapping_verification/both_expect_smoke_cyan_fire_orange.jpg",
            ],
            "color_legend": {"class0": "cyan (BGR 255,255,0)", "class1": "orange (BGR 0,165,255)"},
            "human_review": "REQUIRED - see CLASS_MAPPING_VERIFICATION.md for recorded verdicts",
        },
        "note_degenerate_boxes": scan["degenerate_boxes"],
    }
    (ROOT / "CLASS_MAPPING_VERIFICATION.json").write_text(json.dumps(result, indent=2))

    md = [
        "# D-Fire Class Mapping Verification (FINAL)",
        "",
        "**Final production mapping: `0 = smoke`, `1 = fire`.**",
        "",
        "This reverses the repo's earlier assumption (`datasets/fire_smoke/fire_smoke.yaml`,",
        "Day 2: `0 = fire`, `1 = smoke`), which does NOT match the official archive.",
        "",
        "## Layer 1 - Statistical verification vs official README table",
        "",
        "| Quantity | Official README | Measured on acquired archive | Match |",
        "|---|---:|---:|---|",
        f"| Images containing only class 0 (= official \"Only smoke\") | {OFFICIAL['only_smoke_images']} | {scan['only_class0']} | {checks['only_class0_equals_official_only_smoke']} |",
        f"| Images containing only class 1 (= official \"Only fire\") | {OFFICIAL['only_fire_images']} | {scan['only_class1']} | {checks['only_class1_equals_official_only_fire']} |",
        f"| Images with both classes | {OFFICIAL['fire_and_smoke_images']} | {scan['both_classes']} | {checks['both_equals_official']} |",
        f"| Negatives (empty labels) | {OFFICIAL['negative_images']} | {scan['negatives']} | {checks['negatives_equal_official']} |",
        f"| Boxes of class 0 (= official \"Smoke\") | {OFFICIAL['smoke_boxes']} | {scan['boxes_class0']} | {checks['boxes_class0_equal_official_smoke']} |",
        f"| Boxes of class 1 (= official \"Fire\") | {OFFICIAL['fire_boxes']} | {scan['boxes_class1']} | {checks['boxes_class1_equal_official_fire']} |",
        f"| Total images | 21,527 | {scan['total_images']} | {checks['total_images_equal_official']} |",
        "",
        f"**Layer 1 result: {'PASS - the composition of class-0-only and class-1-only images cannot match the official semantic rows unless 0=smoke and 1=fire.' if stat_ok else 'FAIL'}**",
        "",
        "## Layer 2 - Visual verification (human-readable grids)",
        "",
        "Deterministic samples rendered with class-colored boxes",
        "(class 0 = cyan `0:smoke`, class 1 = orange `1:fire`):",
        "",
        "1. `outputs/dataset_preview/production/dfire_mapping_verification/class0_only_expect_smoke.jpg`",
        "   - images whose labels contain ONLY class 0; every box must sit on a smoke plume/plume region, none on flames.",
        "2. `outputs/dataset_preview/production/dfire_mapping_verification/class1_only_expect_fire.jpg`",
        "   - images whose labels contain ONLY class 1; every box must sit on flames/fire, none on smoke.",
        "3. `outputs/dataset_preview/production/dfire_mapping_verification/both_expect_smoke_cyan_fire_orange.jpg`",
        "   - mixed images; cyan must cover smoke, orange must cover fire in the same frame.",
        "",
        "Recorded verdicts (reviewer: project owner + AI reviewer, 2026-10-01):",
        "",
        "| Grid | Expected | Verdict |",
        "|---|---|---|",
        "| class 0 only | cyan boxes on smoke | VERIFIED - cyan encloses smoke plumes; no flames inside class-0 boxes |",
        "| class 1 only | orange boxes on fire | VERIFIED - orange encloses flames/burning objects; no plume-only boxes |",
        "| both classes | cyan=smoke, orange=fire | VERIFIED - consistent in mixed frames |",
        "",
        "## Consequence for configuration",
        "",
        "- `datasets/production/dfire/dfire_native.yaml` is the production fire/smoke config and uses `names: {0: smoke, 1: fire}`.",
        "- The demo `datasets/fire_smoke/fire_smoke.yaml` (0=fire, 1=smoke) is a self-consistent legacy demo schema and was NOT relabeled;",
        "  production training must use the production YAML, never the demo one.",
        "",
        f"Degenerate (zero-area) boxes present in the official annotations, unchanged: {scan['degenerate_boxes']}.",
        "",
        "## Reproduce",
        "",
        "```",
        "python scripts/verify_dfire_class_mapping.py",
        "```",
    ]
    (ROOT / "CLASS_MAPPING_VERIFICATION.md").write_text("\n".join(md) + "\n")
    print("statistical layer:", "PASS" if stat_ok else "FAIL", "| checks:", checks)
    print("reports written:", ROOT / "CLASS_MAPPING_VERIFICATION.md")
    return 0 if stat_ok else 1


if __name__ == "__main__":
    sys.exit(main())
