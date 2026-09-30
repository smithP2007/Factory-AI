#!/usr/bin/env python3
"""Inference latency/FPS benchmark for approved M1/M2 models (GPU required).

Benchmarks on leakage-aware TEST-split images (unseen in training):
  - warmup pass, then N timed images end-to-end (predict pipeline)
  - Ultralytics speed dict (preprocess / inference / postprocess ms per image)
Usage:
    python3 scripts/benchmark_inference.py <model.pt> <data.yaml> <tag> [n_images]
"""
import json
import pathlib
import random
import sys
import time

import torch
from ultralytics import YOLO

OUT_ROOT = pathlib.Path("outputs/training")


def main():
    if not torch.cuda.is_available():
        sys.exit("REFUSING: benchmark is GPU-only per approved spec.")
    model_path, data, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    warmup_n = 10

    # resolve test list from the data yaml
    yaml_text = pathlib.Path(data).read_text()
    root = None
    test_rel = None
    for line in yaml_text.splitlines():
        if line.startswith("path:"):
            root = pathlib.Path(line.split(":", 1)[1].strip())
        elif line.startswith("test:"):
            test_rel = line.split(":", 1)[1].strip()
    if root is None or test_rel is None:
        sys.exit("cannot resolve path/test from yaml")
    test_list = root / test_rel
    images = [pathlib.Path(p) for p in test_list.read_text().split()]
    random.seed(20261001)
    random.shuffle(images)
    if len(images) < n + warmup_n:
        n = max(1, len(images) - warmup_n)
    files = [str(p) for p in images[:n]]
    warmup_files = [str(p) for p in images[n:n + warmup_n]]

    model = YOLO(model_path)

    # warmup (GPU kernels, allocator)
    for _ in model.predict(warmup_files, imgsz=640, device=0, verbose=False,
                           stream=True):
        pass

    speeds = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    count = 0
    t0 = time.perf_counter()
    for res in model.predict(files, imgsz=640, device=0, verbose=False,
                             stream=True):
        for k in speeds:
            speeds[k] += res.speed.get(k, 0.0)
        count += 1
    wall = time.perf_counter() - t0

    for k in speeds:
        speeds[k] /= max(count, 1)
    payload = {
        "model": model_path,
        "gpu": torch.cuda.get_device_name(0),
        "imgsz": 640,
        "images_timed": count,
        "end_to_end_ms_per_image": wall * 1000.0 / max(count, 1),
        "end_to_end_FPS": count / wall,
        "preprocess_ms": speeds["preprocess"],
        "inference_ms": speeds["inference"],
        "postprocess_ms": speeds["postprocess"],
        "inference_only_FPS": 1000.0 / max(speeds["inference"], 1e-9),
    }
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
