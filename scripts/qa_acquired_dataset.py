#!/usr/bin/env python3
"""QA pipeline for reacquired production datasets (no training, no merging).

For a dataset root with splits containing images/ and labels/:
  1. per-split image/label counts + per-class box counts
  2. image integrity via OpenCV decode
  3. YOLO label syntax validation (5 fields, class id, coord ranges)
  4. pairing checks (image without label, label without image)
  5. near-duplicate detection via 64-bit dHash (Hamming <= threshold)
  6. visual QA preview grid (annotated samples) per split

Writes a JSON report next to the dataset root and PNG previews under
outputs/dataset_preview/production/<name>/.

Usage:
  python qa_acquired_dataset.py --name dfire --root datasets/production/dfire \
      --classes 0:smoke,1:fire --splits train,test [--dup-threshold 8]
"""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/home/z/my-project/Factory-AI")


def dhash64(img) -> int:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    diff = (g[:, 1:] > g[:, :-1]).flatten()
    return int.from_bytes(np.packbits(diff).tobytes(), "big")


def process_image(args):
    img_path, label_path, expected_classes = args
    rec = {"image": img_path.name, "split_ok": True, "readable": False,
           "label_exists": label_path.exists(), "label_errors": [], "boxes": 0,
           "class_counts": Counter(), "hash": None}
    img = cv2.imread(str(img_path))
    if img is None:
        rec["split_ok"] = False
        return rec
    rec["readable"] = True
    rec["hash"] = dhash64(img)
    if not rec["label_exists"]:
        rec["label_errors"].append("missing_label_file")
        return rec
    try:
        text = label_path.read_text().strip()
    except Exception as e:  # noqa: BLE001
        rec["label_errors"].append(f"unreadable:{e}")
        return rec
    if not text:
        rec["boxes"] = 0  # negative image (no objects) - valid
        return rec
    for i, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            rec["label_errors"].append(f"line{i}:not_5_fields")
            continue
        try:
            cls = int(parts[0]); xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            rec["label_errors"].append(f"line{i}:non_numeric")
            continue
        if cls not in expected_classes:
            rec["label_errors"].append(f"line{i}:class_id_{cls}_unexpected")
            continue
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            rec["label_errors"].append(f"line{i}:center_out_of_range"); continue
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            rec["label_errors"].append(f"line{i}:size_out_of_range"); continue
        rec["boxes"] += 1
        rec["class_counts"][cls] += 1
    return rec


def pair_names(split_dir: Path):
    images = {p.stem: p for p in (split_dir / "images").iterdir()
              if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    labels = {p.stem: p for p in (split_dir / "labels").iterdir() if p.suffix == ".txt"}
    return images, labels


def hamming_pairs(hashes: dict[str, int], threshold: int):
    """dHash 64-bit, Hamming distance via np.bitwise_count (numpy>=2.0).
    Memory-bounded: XOR chunk x full array, upper-triangle only."""
    names = list(hashes)
    arr = np.array([hashes[n] for n in names], dtype=np.uint64)
    n = len(names)
    dupes = []
    CH = 512
    for i in range(0, n, CH):
        chunk = arr[i:i + CH]
        x = chunk[:, None] ^ arr[None, :]
        hd = np.bitwise_count(x)  # uint64 popcount per element
        for r in range(x.shape[0]):
            g = i + r
            row = hd[r]
            for c in np.where(row[g + 1:] <= threshold)[0]:
                c = int(c) + g + 1
                dupes.append((names[g], names[c], int(row[c])))
        del x, hd
    return dupes


def preview_grid(root: Path, name: str, split: str, images, labels,
                 class_names: dict, n: int = 12):
    out_dir = REPO / "outputs" / "dataset_preview" / "production" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = sorted(images)[:n]
    tiles = []
    for stem in picks:
        img = cv2.imread(str(images[stem]))
        if img is None:
            continue
        img = cv2.resize(img, (416, 416))
        h, w = img.shape[:2]
        lp = labels.get(stem)
        if lp and lp.exists():
            for line in lp.read_text().strip().splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                cls, xc, yc, bw, bh = int(p[0]), *map(float, p[1:])
                x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
                x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(img, class_names.get(cls, str(cls)), (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(img)
    if not tiles:
        return None
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    grid = np.vstack(rows)
    out = out_dir / f"{split}_preview.jpg"
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--classes", required=True, help="e.g. 0:smoke,1:fire")
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--dup-threshold", type=int, default=8)
    args = ap.parse_args()

    root = REPO / args.root
    class_names = {int(k): v for k, v in
                   (p.split(":") for p in args.classes.split(","))}
    expected = set(class_names)
    report = {"dataset": args.name, "root": str(root), "classes": class_names,
              "splits": {}, "duplicates": {"threshold": args.dup_threshold, "pairs": []},
              "errors_total": 0}

    for split in [s for s in args.splits.split(",") if s]:
        sd = root / split
        images, labels = pair_names(sd)
        tasks = [(images[k], labels.get(k, sd / "labels" / f"{k}.txt"), expected)
                 for k in sorted(images)]
        with ProcessPoolExecutor(max_workers=2) as ex:
            recs = []
            for i, r in enumerate(ex.map(process_image, tasks, chunksize=128)):
                recs.append(r)
                if (i + 1) % 5000 == 0:
                    print(f"  [{split}] {i + 1}/{len(tasks)} processed", flush=True)

        unreadable = [r for r in recs if not r["readable"]]
        lbl_missing = [r["image"] for r in recs if "missing_label_file" in r["label_errors"]]
        bad_labels = [r for r in recs
                      if any(not e.startswith("missing_label_file") for e in r["label_errors"])]
        class_counts = Counter()
        for r in recs:
            class_counts.update(r["class_counts"])
        hashes = {r["image"]: r["hash"] for r in recs if r["hash"] is not None}
        dupes = hamming_pairs(hashes, args.dup_threshold)

        report["splits"][split] = {
            "images": len(images), "labels_on_disk": len(labels),
            "unreadable_images": len(unreadable),
            "images_without_label": len(lbl_missing),
            "labels_with_syntax_errors": len(bad_labels),
            "boxes": int(sum(class_counts.values())),
            "class_counts": {class_names[k]: int(v) for k, v in sorted(class_counts.items())},
            "duplicate_pairs": len(dupes),
            "duplicate_examples": dupes[:10],
        }
        report["errors_total"] += (len(unreadable) + len(lbl_missing) + len(bad_labels))
        report["duplicates"]["pairs"].extend(
            {"split": split, "a": a, "b": b, "hamming": h} for a, b, h in dupes)
        pv = preview_grid(root, args.name, split, images, labels, class_names)
        if pv:
            report["splits"][split]["preview"] = pv

    report["duplicates"]["total_pairs"] = len(report["duplicates"]["pairs"])
    # keep repo-lean: only first 20 pairs inline + distribution; full list regenerable
    from collections import Counter as _C
    _hd = _C(p["hamming"] for p in report["duplicates"]["pairs"])
    report["duplicates"]["hamming_distribution"] = {str(k): v for k, v in sorted(_hd.items())}
    report["duplicates"]["kept_examples"] = 20
    report["duplicates"]["note"] = ("pairs truncated to first 20; re-run for the full list")
    report["duplicates"]["pairs"] = report["duplicates"]["pairs"][:20]
    out = root / f"{args.name}_qa_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report["splits"].items()}, indent=2))
    print("duplicates:", report["duplicates"]["total_pairs"], "| errors_total:",
          report["errors_total"], "| report:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
