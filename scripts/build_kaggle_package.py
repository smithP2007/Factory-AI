#!/usr/bin/env python3
"""Build the Kaggle transfer package for the exact approved M1/M2 datasets.

STREAMING builder: writes zip entries directly from the original files
(ZIP_STORED — jpgs are already compressed; no staging copy, minimal disk use).

Package layout (Kaggle dataset slug ps6-m1-dfire / ps6-m2-hardhats):
  <ds>/{train,test}/images/... , <ds>/{train,test}/labels/...   # ONLY list-referenced files
  leakage_aware/{train,val,test}.txt                            # rewritten: Kaggle-relative
  leakage_aware/original_lists/{train,val,test}.txt             # byte-original provenance
  leakage_aware/{quarantine.txt,quarantine_manifest.csv,leakage_aware_report.json,leakage_aware_report.md}
  <ds>_kaggle.yaml  (+ native yaml as provenance)
  PROVENANCE.md (+ CLASS_MAPPING_VERIFICATION.* for m1)
  models/yolo26n.pt
  scripts/ (approved training + eval tooling incl. kaggle_train.sh)
  unseen_assets/ (demo images + negative + unseen video for qualitative runs)
  shared reports + MANIFEST.json (counts, bytes, sha256)

Usage:
    python3 scripts/build_kaggle_package.py m1 [--out PATH]
    python3 scripts/build_kaggle_package.py m2 [--out PATH]

SPLIT INTEGRITY GUARANTEE: split MEMBERSHIP (which images belong to train/val/
test) is preserved byte-for-byte; only the path PREFIX is rewritten for the
Kaggle mount. The original absolute-path lists are shipped under
leakage_aware/original_lists/ for provenance. No image is added, removed,
or moved between splits. Approved splits are NOT modified at the source.
"""
import hashlib
import json
import pathlib
import sys
import zipfile

REPO = pathlib.Path("/home/z/my-project/Factory-AI")

CFG = {
    "m1": {
        "slug": "ps6-m1-dfire",
        "ds_root": REPO / "datasets/production/dfire",
        "ds_name": "dfire",
        "lists": ["train", "val", "test"],
        "native_yaml": "dfire_native.yaml",
        "kaggle_yaml": "dfire_kaggle.yaml",
        "extra_docs": ["PROVENANCE.md", "CLASS_MAPPING_VERIFICATION.md", "CLASS_MAPPING_VERIFICATION.json"],
        "unseen_globs": ["datasets/fire_smoke/images/*/*.jpg", "images/test/*.jpg"],
        "unseen_videos": ["videos/test/oceans.mp4"],
    },
    "m2": {
        "slug": "ps6-m2-hardhats",
        "ds_root": REPO / "datasets/production/hardhats",
        "ds_name": "hardhats",
        "lists": ["train", "val", "test"],
        "native_yaml": "hardhats_native.yaml",
        "kaggle_yaml": "hardhats_kaggle.yaml",
        "extra_docs": ["PROVENANCE.md"],
        "unseen_globs": ["datasets/ppe/images/*/*.jpg", "images/test/*.jpg"],
        "unseen_videos": ["videos/test/people-detection.mp4"],
    },
}
SCRIPTS = ["train_approved.sh", "run_approved_pipeline.sh", "kaggle_train.sh",
           "eval_approved_model.py", "benchmark_inference.py",
           "failure_cases.py", "qualitative_approved.py"]
SHARED = ["DATASET_READY_FOR_TRAINING.md", "requirements.txt",
          "datasets/production/DATASET_VALIDATION_FINAL.json",
          "datasets/production/PPE_ARCHITECTURE_DECISION.md",
          "datasets/REACQUISITION_REPORT.md", "datasets/DATASET_REPORT.md",
          "datasets/DATASET_STRATEGY.md"]


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kaggle_rel(img: pathlib.Path, ds_root: pathlib.Path, ds_name: str) -> str:
    rel = img.relative_to(ds_root)
    return f"{ds_name}/{rel}"


def label_for(img: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(str(img).replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which not in CFG:
        sys.exit("usage: build_kaggle_package.py m1|m2 [--out PATH]")
    cfg = CFG[which]
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else \
        pathlib.Path(f"/home/z/my-project/download/{cfg['slug']}.zip")

    ds_root, ds_name = cfg["ds_root"], cfg["ds_name"]
    la_dir = ds_root / "leakage_aware"

    # resolve referenced files
    split = {}
    for s in cfg["lists"]:
        imgs = [pathlib.Path(p) for p in (la_dir / f"{s}.txt").read_text().split()]
        missing = [p for p in imgs if not p.is_file()]
        if missing:
            sys.exit(f"ABORT: {len(missing)} images missing from {s} list (e.g. {missing[0]})")
        split[s] = imgs
    all_imgs = [p for s in cfg["lists"] for p in split[s]]
    labels = [label_for(p) for p in all_imgs]
    missing_lbl = [p for p in labels if not p.is_file()]
    if missing_lbl:
        sys.exit(f"ABORT: {len(missing_lbl)} labels missing")

    payload_bytes = sum(p.stat().st_size for p in all_imgs) + \
        sum(p.stat().st_size for p in labels)
    support = ([ds_root / cfg["native_yaml"]] +
               [ds_root / f for f in cfg["extra_docs"]] +
               [la_dir / f for f in ["quarantine.txt", "quarantine_manifest.csv",
                                     "leakage_aware_report.json", "leakage_aware_report.md"]] +
               [REPO / "scripts" / s for s in SCRIPTS] +
               [REPO / s for s in SHARED] +
               [REPO / "models/yolo26n.pt"])
    unseen = [pathlib.Path(p) for g in cfg["unseen_globs"] for p in sorted(REPO.glob(g))]
    unseen_vids = [p for p in (REPO / v for v in cfg["unseen_videos"]) if p.is_file()]
    support_bytes = sum(p.stat().st_size for p in support) + \
        sum(p.stat().st_size for p in unseen + unseen_vids)
    est = payload_bytes + support_bytes + 20_000_000

    free = pathlib.Path("/home/z").stat
    import shutil
    free_b = shutil.disk_usage("/home/z").free
    if free_b < est + 200_000_000:
        sys.exit(f"ABORT: free disk {free_b/1e9:.2f} GB < needed ~{(est+2e8)/1e9:.2f} GB. "
                 f"Free space first (see transfer plan) or choose another output location.")

    manifest = {"slug": cfg["slug"], "which": which,
                "splits": {s: {"images": len(v), "bytes": sum(p.stat().st_size for p in v)}
                           for s, v in split.items()},
                "total_images": len(all_imgs), "total_labels": len(labels),
                "payload_bytes": payload_bytes, "support_bytes": support_bytes,
                "sha256": {}, "split_membership_unchanged": True,
                "list_prefix_rewritten_to": f"<kaggle_input>/{cfg['slug']}"}

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        # 1) images + labels (exactly the approved split payload)
        written = set()
        for s in cfg["lists"]:
            for img in split[s]:
                arc = kaggle_rel(img, ds_root, ds_name)
                if arc in written:
                    sys.exit(f"ABORT: image shared across splits would break grouping: {arc}")
                written.add(arc)
                z.write(img, arc)
                z.write(label_for(img), arc.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
        # 2) leakage_aware lists: rewritten + originals
        for s in cfg["lists"]:
            rewritten = "\n".join(kaggle_rel(p, ds_root, ds_name) for p in split[s]) + "\n"
            z.writestr(f"leakage_aware/{s}.txt", rewritten)
            z.write(la_dir / f"{s}.txt", f"leakage_aware/original_lists/{s}.txt")
            manifest["sha256"][f"original_lists/{s}.txt"] = sha256(la_dir / f"{s}.txt")
        # 3) quarantine + method provenance
        for f in ["quarantine.txt", "quarantine_manifest.csv",
                  "leakage_aware_report.json", "leakage_aware_report.md"]:
            z.write(la_dir / f, f"leakage_aware/{f}")
        # 4) yamls
        native = (ds_root / cfg["native_yaml"]).read_text()
        z.writestr(cfg["kaggle_yaml"],
                   native.replace(f"path: {ds_root}", f"path: /kaggle/input/{cfg['slug']}/{ds_name}"))
        z.write(ds_root / cfg["native_yaml"], cfg["native_yaml"])
        # 5) docs, scripts, base model
        for f in cfg["extra_docs"]:
            z.write(ds_root / f, f)
        for s in SCRIPTS:
            z.write(REPO / "scripts" / s, f"scripts/{s}")
        for s in SHARED:
            z.write(REPO / s, s.split("/")[-1] if not s.startswith("datasets") else s)
        z.write(REPO / "models/yolo26n.pt", "models/yolo26n.pt")
        # 6) unseen assets for qualitative runs
        for p in unseen + unseen_vids:
            z.write(p, f"unseen_assets/{p.relative_to(REPO)}")
        # 7) manifest
        manifest["zip_bytes_estimated"] = est
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

    # post-verify
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        n_img = sum(1 for n in z.namelist() if "/images/" in n and not n.endswith("/"))
        n_lbl = sum(1 for n in z.namelist() if "/labels/" in n and not n.endswith("/"))
    print(json.dumps({"zip": str(out), "bytes": out.stat().st_size,
                      "GB": round(out.stat().st_size / 1e9, 2),
                      "testzip": bad, "image_entries": n_img, "label_entries": n_lbl,
                      "expected_images": len(all_imgs), "expected_labels": len(labels)},
                     indent=2))
    if bad or n_img != len(all_imgs) or n_lbl != len(labels):
        sys.exit("ABORT: verification failed")


if __name__ == "__main__":
    main()
