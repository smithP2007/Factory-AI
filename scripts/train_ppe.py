#!/usr/bin/env python3
"""Train the PS6 four-class PPE detector on a GPU host (Kaggle/Colab/RTX 4060)."""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to ppe_core.yaml")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="ps6_ppe_core")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        amp=True,
        save=True,
        save_period=10,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=42,
        verbose=True,
    )
    save_dir = Path(getattr(results, "save_dir", args.project))
    print(f"Training finished. Save directory: {save_dir}")
    print(f"Best checkpoint: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
