#!/usr/bin/env python3
"""Qualitative runs on UNSEEN images and videos for approved M1/M2 (GPU required).

Unseen sources (none are part of the production leakage-aware splits):
  M1 (fire/smoke): demo fire_smoke dataset images + images/test/bus.jpg
                   (negative control) + videos/test/oceans.mp4 (negative video)
  M2 (helmet):     demo ppe dataset images + images/test/bus.jpg +
                   videos/test/people-detection.mp4

Usage:
    python3 scripts/qualitative_approved.py <model.pt> <tag> m1|m2
"""
import glob
import os
import pathlib
import sys

import torch
from ultralytics import YOLO

OUT_ROOT = pathlib.Path("outputs/training")
# Optional relocation of the unseen-asset tree (used on Kaggle where the
# package mounts at /kaggle/input/<slug>/unseen_assets; locally defaults to repo-relative paths)
UNSEEN_ROOT = os.environ.get("UNSEEN_ROOT", "")


def main():
    if not torch.cuda.is_available():
        sys.exit("REFUSING: qualitative runs are GPU-only per approved spec.")
    model_path, tag, which = sys.argv[1], sys.argv[2], sys.argv[3]
    out_dir = OUT_ROOT / tag / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    if which == "m1":
        img_pats = ["datasets/fire_smoke/images/test/*.jpg",
                    "datasets/fire_smoke/images/val/*.jpg",
                    "datasets/fire_smoke/images/train/*.jpg",
                    "images/test/*.jpg"]
        videos = ["videos/test/oceans.mp4"]
    else:
        img_pats = ["datasets/ppe/images/test/*.jpg",
                    "datasets/ppe/images/val/*.jpg",
                    "datasets/ppe/images/train/*.jpg",
                    "images/test/*.jpg"]
        videos = ["videos/test/people-detection.mp4"]
    if UNSEEN_ROOT:
        img_pats = [f"{UNSEEN_ROOT}/{p}" for p in img_pats]
        videos = [f"{UNSEEN_ROOT}/{v}" for v in videos]

    images = []
    for pat in img_pats:
        images.extend(sorted(glob.glob(pat)))
    # cap per source to 3 images each for a compact evidence set
    selected = []
    by_dir = {}
    for p in images:
        d = str(pathlib.Path(p).parent)
        if by_dir.get(d, 0) < 3:
            selected.append(p)
            by_dir[d] = by_dir.get(d, 0) + 1

    print(f"[qualitative] {len(selected)} unseen images, {len(videos)} unseen videos")
    if selected:
        for res in model.predict(selected, imgsz=640, device=0, conf=0.25,
                                 save=True, project=str(out_dir),
                                 name="images", exist_ok=True, stream=True):
            boxes = len(res.boxes)
            print(f"  {pathlib.Path(res.path).name}: {boxes} detections "
                  f"({', '.join(model.names[int(c)] for c in res.boxes.cls) or '-'})")

    for vid in videos:
        if not pathlib.Path(vid).exists():
            print(f"  [warn] video missing: {vid}")
            continue
        print(f"  video: {vid}")
        frames = 0
        for _ in model.predict(vid, imgsz=640, device=0, conf=0.25, save=True,
                               project=str(out_dir), name="videos",
                               exist_ok=True, stream=True):
            frames += 1
        print(f"    processed {frames} frames -> {out_dir/'videos'}")

    print(f"[qualitative] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
