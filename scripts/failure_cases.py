#!/usr/bin/env python3
"""Failure-case analysis on the leakage-aware TEST split (GPU required).

For every test image: greedy class-matched IoU>=0.5 matching between
predictions (conf>=0.25) and ground-truth boxes; counts FN/FP per image,
ranks worst offenders, saves annotated evidence for the top cases and a JSON
summary. Evidence drawing: green=TP, red=FN (missed GT), orange=FP.

Usage:
    python3 scripts/failure_cases.py <model.pt> <data.yaml> <tag> [top_n]
"""
import json
import pathlib
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

OUT_ROOT = pathlib.Path("outputs/training")
CONF = 0.25
IOU_MATCH = 0.5


def load_gt(label_path: pathlib.Path):
    boxes = []
    if not label_path.exists():
        return boxes
    for ln in label_path.read_text().splitlines():
        parts = ln.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        boxes.append((c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def xywh_to_xyxy(pred):
    # pred.boxes.xyxy is tensor [N,4] (pixels); conf/cls tensors [N]
    return pred.boxes.xyxy.cpu().numpy(), pred.boxes.conf.cpu().numpy(), \
        pred.boxes.cls.cpu().numpy().astype(int)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
    bx1, by1, bx2, by2 = b[:, 0][None, :], b[:, 1][None, :], b[:, 2][None, :], b[:, 3][None, :]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def main():
    if not torch.cuda.is_available():
        sys.exit("REFUSING: failure analysis is GPU-only per approved spec.")
    model_path, data, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    top_n = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    root = test_rel = None
    for line in pathlib.Path(data).read_text().splitlines():
        if line.startswith("path:"):
            root = pathlib.Path(line.split(":", 1)[1].strip())
        elif line.startswith("test:"):
            test_rel = line.split(":", 1)[1].strip()
    images = [pathlib.Path(p) for p in (root / test_rel).read_text().split()]

    model = YOLO(model_path)
    names = model.names
    per_image = []
    agg = {"true_positives": 0, "false_negatives": 0, "false_positives": 0,
           "fn_by_class": {}, "fp_by_class": {}}

    out_dir = OUT_ROOT / tag / "failure_cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        label_path = pathlib.Path(str(img_path).replace("/images/", "/labels/")
                                  .rsplit(".", 1)[0] + ".txt")
        gt = load_gt(label_path)
        res = model.predict(str(img_path), imgsz=640, device=0, conf=CONF,
                            verbose=False)[0]
        px, pc, pcls = xywh_to_xyxy(res)
        g = np.array([[b[1], b[2], b[3], b[4]] for b in gt]).reshape(-1, 4)
        gcls = [b[0] for b in gt]

        matched_gt, matched_pred = set(), set()
        if len(g) and len(px):
            m = iou_matrix(px, g)
            # greedy highest-IoU first, class-matched
            order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
            for pi, gi in order:
                if pi in matched_pred or gi in matched_gt:
                    continue
                if pcls[pi] != gcls[gi] or m[pi, gi] < IOU_MATCH:
                    continue
                matched_pred.add(int(pi))
                matched_gt.add(int(gi))

        fn = [i for i in range(len(g)) if i not in matched_gt]
        fp = [i for i in range(len(px)) if i not in matched_pred]
        for i in fn:
            c = names[gcls[i]]
            agg["fn_by_class"][c] = agg["fn_by_class"].get(c, 0) + 1
        for i in fp:
            c = names[pcls[i]]
            agg["fp_by_class"][c] = agg["fp_by_class"].get(c, 0) + 1
        agg["true_positives"] += len(matched_gt)
        agg["false_negatives"] += len(fn)
        agg["false_positives"] += len(fp)
        if fn or fp:
            per_image.append({
                "image": str(img_path), "gt": len(g), "pred": len(px),
                "fn": len(fn), "fp": len(fp), "fn_classes": [names[gcls[i]] for i in fn],
                "fp_classes": [names[pcls[i]] for i in fp],
            })

    per_image.sort(key=lambda r: -(r["fn"] + r["fp"]))
    summary = {
        "model": model_path, "data": data, "conf": CONF, "iou_match": IOU_MATCH,
        "test_images": len(images),
        "aggregate": agg,
        "worst_images": per_image[:50],
        "images_with_any_error": len(per_image),
    }
    (out_dir / "failure_summary.json").write_text(json.dumps(summary, indent=2))

    # annotated evidence for the worst N images
    for rank, rec in enumerate(per_image[:top_n], 1):
        img_path = pathlib.Path(rec["image"])
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        label_path = pathlib.Path(str(img_path).replace("/images/", "/labels/")
                                  .rsplit(".", 1)[0] + ".txt")
        gt = load_gt(label_path)
        res = model.predict(str(img_path), imgsz=640, device=0, conf=CONF,
                            verbose=False)[0]
        px, pc, pcls = xywh_to_xyxy(res)
        g = np.array([[b[1], b[2], b[3], b[4]] for b in gt]).reshape(-1, 4)
        gcls = [b[0] for b in gt]
        matched_gt, matched_pred = set(), set()
        if len(g) and len(px):
            m = iou_matrix(px, g)
            order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
            for pi, gi in order:
                if pi in matched_pred or gi in matched_gt:
                    continue
                if pcls[pi] != gcls[gi] or m[pi, gi] < IOU_MATCH:
                    continue
                matched_pred.add(int(pi))
                matched_gt.add(int(gi))
        for i, (x1, y1, x2, y2) in enumerate(g):
            color = (0, 255, 0) if i in matched_gt else (0, 0, 255)  # green TP / red FN
            cv2.rectangle(img, (int(x1*W), int(y1*H)), (int(x2*W), int(y2*H)), color, 2)
            if i not in matched_gt:
                cv2.putText(img, f"FN {names[gcls[i]]}", (int(x1*W), max(12, int(y1*H)-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        for i in range(len(px)):
            if i in matched_pred:
                continue
            x1, y1, x2, y2 = px[i]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
            cv2.putText(img, f"FP {names[pcls[i]]} {pc[i]:.2f}", (int(x1), max(12, int(y1)-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.imwrite(str(out_dir / f"rank{rank:02d}_{img_path.stem}.jpg"), img)

    print(json.dumps(agg, indent=2))
    print(f"[failure-cases] {len(per_image)} test images with errors; "
          f"evidence -> {out_dir}")


if __name__ == "__main__":
    main()
