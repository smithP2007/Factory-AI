#!/usr/bin/env python3
"""Post-training evaluation for approved M1/M2 models (GPU required).

Runs Ultralytics validation on the approved leakage-aware val AND test splits,
captures overall + per-class metrics to JSON/text, and copies confusion
matrices. Usage:
    python3 scripts/eval_approved_model.py <model.pt> <data.yaml> <tag>
Example:
    python3 scripts/eval_approved_model.py models/m1_fire_smoke_best.pt \
        datasets/production/dfire/dfire_native.yaml m1_fire_smoke
"""
import json
import pathlib
import shutil
import sys

from ultralytics import YOLO

import torch

OUT_ROOT = pathlib.Path("outputs/training")


def require_gpu():
    if not torch.cuda.is_available():
        sys.exit("REFUSING: evaluation is GPU-only per approved spec (CPU eval would misstate latency).")


def run_split(model: YOLO, data: str, split: str, tag: str, out_dir: pathlib.Path):
    res = model.val(data=data, split=split, imgsz=640, device=0, plots=True,
                    project="runs/val", name=f"{tag}_{split}")
    box = res.box
    per_class = []
    names = res.names
    idxs = list(box.ap_class_index)
    ap50s = list(box.ap50)
    aps = list(box.ap)          # per-class mAP50-95
    for i, c in enumerate(idxs):
        per_class.append({
            "class_id": int(c),
            "class_name": names[int(c)],
            "AP50": float(ap50s[i]) if i < len(ap50s) else None,
            "AP50-95": float(aps[i]) if i < len(aps) else None,
        })
    payload = {
        "split": split,
        "images": int(getattr(res, "count", 0)) if hasattr(res, "count") else None,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "speed_ms_per_image": res.speed,
        "per_class": per_class,
        "class_names": names,
    }
    (out_dir / f"val_{split}.json").write_text(json.dumps(payload, indent=2))
    lines = [f"split={split} images={payload['images']}",
             f"P={payload['precision']:.4f} R={payload['recall']:.4f} "
             f"mAP50={payload['mAP50']:.4f} mAP50-95={payload['mAP50-95']:.4f}"]
    for pc in per_class:
        lines.append(f"  class {pc['class_id']} {pc['class_name']}: "
                     f"AP50={pc['AP50']:.4f} AP50-95={pc['AP50-95']:.4f}")
    (out_dir / f"val_{split}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    # copy confusion matrices produced by this val run
    run_dir = pathlib.Path(res.save_dir)
    for cm in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = run_dir / cm
        if src.exists():
            shutil.copy2(src, out_dir / f"{split}_{cm}")
    return payload


def main():
    require_gpu()
    model_path, data, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    result = {"model": model_path, "data": data, "tag": tag,
              "val": run_split(model, data, "val", tag, out_dir),
              "test": run_split(model, data, "test", tag, out_dir)}
    (out_dir / "eval_summary.json").write_text(json.dumps(result, indent=2))
    print(f"[eval] summary -> {out_dir/'eval_summary.json'}")


if __name__ == "__main__":
    main()
