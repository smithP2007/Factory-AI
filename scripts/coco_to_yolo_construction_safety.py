#!/usr/bin/env python3
"""Convert Roboflow COCO export (keremberke/construction-safety-object-detection,
CC BY 4.0) to YOLO format. Keeps the same native-order convention as the Hard
Hats converter: class ids = alphabetical order of COCO category names.

Reorganizes construction_safety/{train,valid,test}/ into images/ + labels/.
Idempotent: moves images once; regenerates labels every run.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/home/z/my-project/Factory-AI/datasets/production/construction_safety")
SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def convert_split(split: str) -> dict:
    coco_path = ROOT / split / "_annotations.coco.json"
    d = json.loads(coco_path.read_text())
    imgs = {im["id"]: im for im in d["images"]}
    cats = {c["id"]: c["name"] for c in d["categories"]}
    native_names = [cats[cid] for cid in sorted(cats)]
    remap = {cid: sorted(cats).index(cid) for cid in cats}

    boxes_per_image: dict[int, list] = {}
    for ann in d["annotations"]:
        boxes_per_image.setdefault(ann["image_id"], []).append(ann)

    stats = {"images": 0, "labels": 0, "boxes": 0, "missing_files": 0,
             "empty_label_images": 0,
             "class_counts": {name: 0 for name in native_names}}

    (ROOT / split / "images").mkdir(exist_ok=True)
    (ROOT / split / "labels").mkdir(exist_ok=True)

    for im_id, im in imgs.items():
        file_name = Path(im["file_name"]).name
        src = ROOT / split / file_name
        if not src.exists():
            cand = list((ROOT / split).glob(f"*/{file_name}")) or list((ROOT / split).glob(file_name))
            if not cand:
                stats["missing_files"] += 1
                continue
            src = cand[0]
        w, h = im["width"], im["height"]
        lines = []
        for ann in boxes_per_image.get(im_id, []):
            x, y, bw, bh = ann["bbox"]
            cx = min(max((x + bw / 2) / w, 0.0), 1.0)
            cy = min(max((y + bh / 2) / h, 0.0), 1.0)
            nw = min(max(bw / w, 0.0), 1.0)
            nh = min(max(bh / h, 0.0), 1.0)
            cls = remap[ann["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats["class_counts"][native_names[cls]] += 1
        dst = ROOT / split / "images" / file_name
        if src.exists() and src.parent != dst.parent:
            shutil.move(str(src), str(dst))
        if dst.exists():
            stats["images"] += 1
            lp = ROOT / split / "labels" / (dst.stem + ".txt")
            lp.write_text("\n".join(lines) + ("\n" if lines else ""))
            stats["labels"] += 1
            stats["boxes"] += len(lines)
            if not lines:
                stats["empty_label_images"] += 1

    stats["native_class_order"] = {i: n for i, n in enumerate(native_names)}
    return stats


def main() -> int:
    total = {"images": 0, "boxes": 0, "missing_files": 0}
    for s in SPLITS:
        st = convert_split(s)
        print(s, json.dumps(st, indent=2))
        for k in ("images", "boxes", "missing_files"):
            total[k] += st[k]
    print("TOTAL:", total)
    print("native class order (alphabetical):", st["native_class_order"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
