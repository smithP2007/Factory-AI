#!/usr/bin/env python3
"""Visual QA grids for leakage-aware split lists (annotated samples per split)."""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/home/z/my-project/Factory-AI")

# class colors (BGR)
PALETTES = {
    "dfire": {0: ((255, 255, 0), "0:smoke"), 1: ((0, 165, 255), "1:fire")},
    "hardhats": {0: ((0, 200, 0), "0:hardhat"), 1: ((0, 0, 255), "1:no-hardhat"),
                 2: ((255, 0, 255), "2:person"), 3: ((255, 160, 0), "3:vest")},
}


def render_split(ds, split, paths, n=12):
    style = PALETTES[ds]
    out_dir = REPO / "outputs/dataset_preview/production" / f"{ds}_leakage_aware"
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(paths) // n)
    picks = paths[::step][:n]
    tiles = []
    for p in picks:
        img = cv2.imread(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        scale = 416 / max(H, W)
        img = cv2.resize(img, (int(W * scale), int(H * scale)))
        h, w = img.shape[:2]
        lp = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
        if lp.exists() and lp.read_text().strip():
            for line in lp.read_text().strip().splitlines():
                q = line.split()
                if len(q) != 5:
                    continue
                cls, xc, yc, bw, bh = int(q[0]), *map(float, q[1:])
                x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
                x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
                color, name = style.get(cls, ((255, 255, 255), str(cls)))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, name, (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "negative (no boxes)", (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{Path(p).parent.parent.name}/{Path(p).name[:20]}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
             for t in tiles]
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    grid = np.vstack(rows)
    grid = cv2.copyMakeBorder(grid, 34, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(grid, f"{ds} leakage-aware {split} - {len(picks)} of {len(paths)} images "
                     f"(deterministic stride sample)", (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    out = out_dir / f"{split}_preview.jpg"
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", out)
    return out


def main() -> int:
    ds = sys.argv[1] if len(sys.argv) > 1 else "dfire"
    root = REPO / "datasets/production" / ds / "leakage_aware"
    for split in ("train", "val", "test"):
        paths = [l for l in (root / f"{split}.txt").read_text().splitlines() if l]
        render_split(ds, split, paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
