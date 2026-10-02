#!/usr/bin/env python3
"""Prepare the PS6 four-class PPE dataset from a YOLO dataset.

Expected source structure (one of):
  train/valid/test
  train/val/test
Each split should contain images/ and labels/, plus a data.yaml at source root.

The script:
- reads the source class names from data.yaml
- keeps only helmet, gloves, goggles, vest
- remaps them to 0..3
- preserves source splits
- computes SHA-256 hashes for exact duplicates
- optionally computes dHash for near-duplicates when imagehash is installed
- quarantines duplicate copies, preferring test > val > train
- writes a machine-readable manifest

It never fabricates missing-PPE annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Pillow is required: pip install pillow") from exc

try:
    import imagehash
except ImportError:
    imagehash = None

TARGET = {"helmet": 0, "gloves": 1, "goggles": 2, "vest": 3, "safety vest": 3, "safety-vest": 3}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
PRIORITY = {"test": 3, "val": 2, "train": 1}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm_name(x: str) -> str:
    return " ".join(str(x).strip().lower().replace("_", " ").replace("-", " ").split())


def read_names(data_yaml: Path) -> List[str]:
    obj = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = obj.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda z: int(z))]
    raise ValueError("data.yaml does not contain a supported 'names' field")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path):
    if imagehash is None:
        return None
    with Image.open(path) as im:
        return imagehash.dhash(im.convert("RGB"))


def find_image(label_path: Path, images_dir: Path) -> Path | None:
    stem = label_path.stem
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_label(label_path: Path, src_names: List[str]) -> Tuple[List[str], int]:
    out: List[str] = []
    bad = 0
    for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            bad += 1
            continue
        try:
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            bad += 1
            continue
        if cls < 0 or cls >= len(src_names):
            bad += 1
            continue
        if any(v < 0 or v > 1 for v in coords) or coords[2] <= 0 or coords[3] <= 0:
            bad += 1
            continue
        n = norm_name(src_names[cls])
        target = TARGET.get(n)
        if target is None:
            continue
        # Target classes are normalized to the exact four-class schema.
        out.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(target, *coords))
    return out, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--near-threshold", type=int, default=8)
    args = ap.parse_args()

    src = args.src_root.resolve()
    out = args.out_root.resolve()
    data_yaml = src / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"Missing source data.yaml: {data_yaml}")

    names = read_names(data_yaml)
    norm_names = {norm_name(n): i for i, n in enumerate(names)}
    found = {}
    for target in ("helmet", "gloves", "goggles", "vest"):
        hits = [i for n, i in norm_names.items() if n == target or (target == "vest" and n in {"safety vest", "safety vest on", "safety-vest"})]
        if not hits:
            raise SystemExit(f"Required source class not found: {target}; source names={names}")
        found[target] = hits[0]

    if out.exists():
        raise SystemExit(f"Output already exists: {out}. Use a new directory to avoid accidental overwrite.")

    records = []
    counts = {"train": 0, "val": 0, "test": 0}
    class_counts = {"helmet": 0, "gloves": 0, "goggles": 0, "vest": 0}
    exact = {}
    split_files: Dict[str, List[Tuple[Path, Path]]] = {"train": [], "val": [], "test": []}

    for src_split, canonical in SPLIT_ALIASES.items():
        sroot = src / src_split
        if not sroot.exists():
            continue
        imgdir = sroot / "images"
        lbldir = sroot / "labels"
        if not imgdir.exists() or not lbldir.exists():
            continue
        for img in sorted(p for p in imgdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS):
            lbl = lbldir / f"{img.stem}.txt"
            if not lbl.exists():
                records.append({"status": "missing_label", "split": canonical, "image": str(img)})
                continue
            try:
                lines, bad = parse_label(lbl, names)
            except Exception as exc:
                records.append({"status": "label_error", "split": canonical, "image": str(img), "error": str(exc)})
                continue
            # Copy even if target list is empty: such images are valid negatives for the four-class detector.
            rel_name = img.name
            split_files[canonical].append((img, lbl))
            counts[canonical] += 1
            for line in lines:
                target = int(line.split()[0])
                class_counts[{0:"helmet",1:"gloves",2:"goggles",3:"vest"}[target]] += 1
            h = sha256_file(img)
            entry = {"split": canonical, "image": img, "label": lbl, "sha256": h, "lines": lines, "bad_lines": bad, "dhash": None}
            try:
                entry["dhash"] = dhash(img)
            except Exception:
                entry["dhash"] = None
            records.append(entry)
            exact.setdefault(h, []).append(entry)

    out.mkdir(parents=True)
    quarantine = out / "quarantine"
    quarantine.mkdir()

    manifest = {"source_root": str(src), "source_names": names, "target_names": ["helmet", "gloves", "goggles", "vest"], "near_threshold": args.near_threshold, "records": []}

    # Exact duplicates: preserve only the copy in the highest-priority split.
    removed = set()
    for h, group in exact.items():
        if len(group) <= 1:
            continue
        keep = max(group, key=lambda e: PRIORITY[e["split"]])
        for e in group:
            if e is keep:
                continue
            removed.add((str(e["image"]), e["split"]))
            manifest["records"].append({"type": "exact_duplicate", "source": str(e["image"]), "kept": str(keep["image"]), "split": e["split"]})

    # Optional near-duplicate pass. Only cross-split matches are quarantined automatically.
    dhash_records = [r for r in records if "dhash" in r and r.get("dhash") is not None]
    for i, a in enumerate(dhash_records):
        for b in dhash_records[i + 1 :]:
            if a["split"] == b["split"]:
                continue
            if (str(a["image"]), a["split"]) in removed or (str(b["image"]), b["split"]) in removed:
                continue
            try:
                dist = a["dhash"] - b["dhash"]
            except Exception:
                continue
            if dist <= args.near_threshold:
                keep = a if PRIORITY[a["split"]] >= PRIORITY[b["split"]] else b
                drop = b if keep is a else a
                removed.add((str(drop["image"]), drop["split"]))
                manifest["records"].append({"type": "near_duplicate", "distance": int(dist), "source": str(drop["image"]), "kept": str(keep["image"]), "split": drop["split"]})

    for r in records:
        if "image" not in r:
            continue
        key = (str(r["image"]), r["split"])
        if key in removed:
            continue
        src_img = Path(r["image"])
        src_lbl = Path(r["label"])
        dst_img_dir = out / "images" / r["split"]
        dst_lbl_dir = out / "labels" / r["split"]
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img_dir / src_img.name)
        (dst_lbl_dir / f"{src_img.stem}.txt").write_text("\n".join(r["lines"]) + ("\n" if r["lines"] else ""), encoding="utf-8")
        if r["bad_lines"]:
            manifest["records"].append({"type": "invalid_label_lines_skipped", "source": str(src_lbl), "count": int(r["bad_lines"])})

    # Write the final YAML and license/source manifest.
    (out / "ppe_core.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: helmet\n  1: gloves\n  2: goggles\n  3: vest\n",
        encoding="utf-8",
    )
    (out / "SOURCE_DATASET.txt").write_text(
        "Source: https://huggingface.co/datasets/51ddhesh/PPE_Detection\n"
        "Declared license: CC BY 4.0\n"
        "Target classes retained: helmet, gloves, goggles, vest\n"
        "Source classes omitted: safety shoe, mask\n",
        encoding="utf-8",
    )
    (out / "duplicate_quarantine_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": "PASS", "source_counts": counts, "target_object_counts": class_counts, "quarantined_items": len(removed), "imagehash_available": imagehash is not None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
