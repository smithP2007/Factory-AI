#!/usr/bin/env python3
"""Build COCO val2017 PERSON reference (evaluation reference for the person
detector used by the compliance engine).

Why this source: COCO annotates every visible instance of its 80 categories in
every annotated image - `person` supervision is complete by construction
(unlike the Hard Hats / Voxel51 person problem). License: annotations CC BY 4.0;
images under their respective Flickr terms (research use).

Policy:
  - class id 0 = person (only class)
  - iscrowd=1 instances EXCLUDED (RLE crowd regions are not box-trainable)
    and counted
  - images with zero person boxes are KEPT as negatives (COCO labels all
    person instances; a missing box means no unannotated person visible)
Output: images/ (moved from val2017/), labels/, PERSON_REFERENCE_REPORT.json,
preview grid. Idempotent.
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/home/z/my-project/Factory-AI")
ROOT = REPO / "datasets/production/coco_person_reference"


def main() -> int:
    coco = json.loads((ROOT / "instances_val2017.json").read_text())
    imgs = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    person_ids = [cid for cid, n in cats.items() if n == "person"]
    assert len(person_ids) == 1, cats
    person_id = person_ids[0]

    per_img = {}
    iscrowd_dropped = 0
    for ann in coco["annotations"]:
        if ann["category_id"] != person_id:
            continue
        if ann.get("iscrowd", 0) == 1:
            iscrowd_dropped += 1
            continue
        per_img.setdefault(ann["image_id"], []).append(ann)

    (ROOT / "images").mkdir(exist_ok=True)
    (ROOT / "labels").mkdir(exist_ok=True)

    stats = {"images": 0, "labels": 0, "person_boxes": 0, "negatives": 0,
             "iscrowd_dropped": iscrowd_dropped, "unreadable": 0,
             "boxes_per_image_hist": Counter()}
    previews = []
    for im_id, im in sorted(imgs.items()):
        fname = Path(im["file_name"]).name
        src = ROOT / "val2017" / fname
        dst = ROOT / "images" / fname
        if not dst.exists():
            if not src.exists():
                stats["unreadable"] += 1
                continue
            shutil.move(str(src), str(dst))
        H, W = im["height"], im["width"]
        lines = []
        for ann in sorted(per_img.get(im_id, []), key=lambda a: a["id"]):
            x, y, bw, bh = ann["bbox"]
            cx, cy = (x + bw / 2) / W, (y + bh / 2) / H
            nw, nh = bw / W, bh / H
            if not (0 < nw <= 1 and 0 < nh <= 1):
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (ROOT / "labels" / (dst.stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        stats["images"] += 1
        stats["labels"] += 1
        stats["person_boxes"] += len(lines)
        stats["boxes_per_image_hist"][len(lines)] += 1
        if not lines:
            stats["negatives"] += 1
        elif len(previews) < 8 and len(lines) >= 3:
            previews.append(dst)

    # decode-integrity check + preview grid
    bad = 0
    tiles = []
    for p in sorted((ROOT / "images").iterdir()):
        img = cv2.imread(str(p))
        if img is None:
            bad += 1
    for p in previews:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        s = 416 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
        hh, ww = img.shape[:2]
        for line in (ROOT / "labels" / (p.stem + ".txt")).read_text().splitlines():
            _, xc, yc, bw, bh = map(float, line.split())
            cv2.rectangle(img, (int((xc - bw / 2) * ww), int((yc - bh / 2) * hh)),
                          (int((xc + bw / 2) * ww), int((yc + bh / 2) * hh)),
                          (255, 80, 80), 2)
        tiles.append(img)
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
             for t in tiles]
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT)
            for r in rows]
    grid = np.vstack(rows)
    grid = cv2.copyMakeBorder(grid, 34, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(grid, "COCO val2017 person reference - red boxes = person (id 0)",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    (ROOT / "person_reference_preview.jpg").write_bytes(
        cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())

    stats["unreadable_decode_check"] = bad
    stats["boxes_per_image_hist"] = dict(sorted(stats["boxes_per_image_hist"].items()))
    report = {
        "dataset": "COCO val2017 person reference",
        "source": "http://images.cocodataset.org/zips/val2017.zip (815,585,330 bytes) + annotations_trainval2017.zip (252,907,541 bytes)",
        "license": "annotations: CC BY 4.0 (COCO); images: respective Flickr terms (research use)",
        "role": "EVALUATION REFERENCE for the COCO-pretrained person detector; NOT merged with any training set",
        "class_mapping": {"0": "person"},
        **stats,
    }
    (ROOT / "PERSON_REFERENCE_REPORT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
