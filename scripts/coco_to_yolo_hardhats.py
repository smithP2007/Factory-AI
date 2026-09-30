#!/usr/bin/env python3
"""Convert Roboflow COCO export (keremberke/hard-hat-detection, CC BY 4.0)
to YOLO format. Keeps NATIVE class order (0=hardhat, 1=no-hardhat).

Reorganizes hardhats/{train,valid,test}/ into hardhats/{split}/{images,labels}/.
Idempotent: skips images already moved; regenerates labels from COCO every run.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/home/z/my-project/Factory-AI/datasets/production/hardhats")
SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def convert_split(split: str) -> dict:
    coco_path = ROOT / split / "_annotations.coco.json"
    d = json.loads(coco_path.read_text())
    imgs = {im["id"]: im for im in d["images"]}
    cats = {c["id"]: c["name"] for c in d["categories"]}
    native_names = [cats[cid] for cid in sorted(cats)]
    remap = {cid: sorted(cats).index(cid) for cid in cats}  # keep native order

    boxes_per_image: dict[int, list] = {}
    for ann in d["annotations"]:
        boxes_per_image.setdefault(ann["image_id"], []).append(ann)

    stats = {"images": 0, "labels": 0, "boxes": 0, "missing_files": 0,
             "class_counts": {name: 0 for name in native_names}}

    (ROOT / split / "images").mkdir(exist_ok=True)
    (ROOT / split / "labels").mkdir(exist_ok=True)

    for im_id, im in imgs.items():
        file_name = Path(im["file_name"]).name
        src = ROOT / split / file_name
        if not src.exists():
            # some exports nest file_name with a folder prefix
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
        # move image into images/ (idempotent)
        dst = ROOT / split / "images" / file_name
        if src.exists() and src.parent != dst.parent:
            shutil.move(str(src), str(dst))
        stats["images"] += 1
        # write label
        lbl = ROOT / split / "labels" / (Path(file_name).stem + ".txt")
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["labels"] += 1
        stats["boxes"] += len(lines)

    return stats


def main() -> int:
    totals = {"images": 0, "labels": 0, "boxes": 0,
              "class_counts": {"hardhat": 0, "no-hardhat": 0}}
    for split in SPLITS:
        s = convert_split(split)
        print(f"[{split}] images={s['images']} labels={s['labels']} boxes={s['boxes']} "
              f"missing={s['missing_files']} classes={s['class_counts']}")
        totals["images"] += s["images"]
        totals["labels"] += s["labels"]
        totals["boxes"] += s["boxes"]
        for k, v in s["class_counts"].items():
            totals["class_counts"][k] += v
    print("[TOTAL]", json.dumps(totals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
