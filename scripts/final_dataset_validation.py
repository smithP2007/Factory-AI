#!/usr/bin/env python3
"""Final pre-training dataset validation (correction phase, 2026-10-01).

Checks:
  1. leakage-aware split lists: every image exists + decodes, label pairing,
     YOLO syntax, class ids within the declared mapping
  2. YAML class-name assertions (dfire 0=smoke,1=fire; hardhats 0=hardhat,1=no-hardhat)
  3. leakage reports: zero cross-split pairs at Hamming <= 8 (both datasets)
  4. credential scan over production text artifacts
  5. demo datasets untouched (image counts unchanged)
Writes datasets/production/DATASET_VALIDATION_FINAL.json
"""
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import yaml

REPO = Path("/home/z/my-project/Factory-AI")
PROD = REPO / "datasets/production"

DATASETS = {
    "dfire": {"root": PROD / "dfire", "names": {0: "smoke", 1: "fire"}},
    "hardhats": {"root": PROD / "hardhats", "names": {0: "hardhat", 1: "no-hardhat"}},
}

CRED_PAT = re.compile(
    r"(api[_-]?key|apikey|secret|password|token\s*[:=]|kaggle\.json|AKIA[0-9A-Z]{16}"
    r"|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]{20,})", re.IGNORECASE)


def check_image(args):
    img_path, label_path, allowed = args
    rec = {"image": img_path, "ok": True, "errors": []}
    p = Path(img_path)
    if not p.exists():
        rec["ok"] = False
        rec["errors"].append("missing_image")
        return rec
    img = cv2.imread(str(p))
    if img is None:
        rec["ok"] = False
        rec["errors"].append("unreadable_image")
    lp = Path(label_path)
    if not lp.exists():
        rec["ok"] = False
        rec["errors"].append("missing_label")
        return rec
    text = lp.read_text().strip()
    if text:
        for i, line in enumerate(text.splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                rec["errors"].append(f"line{i}:not_5_fields"); rec["ok"] = False; continue
            try:
                cls = int(parts[0]); xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                rec["errors"].append(f"line{i}:non_numeric"); rec["ok"] = False; continue
            if cls not in allowed:
                rec["errors"].append(f"line{i}:class_{cls}_not_in_mapping"); rec["ok"] = False
            if not (0 <= xc <= 1 and 0 <= yc <= 1):
                rec["errors"].append(f"line{i}:coord_range"); rec["ok"] = False
            elif not (0 < w <= 1 and 0 < h <= 1):
                # zero-area boxes exist in the OFFICIAL D-Fire annotations
                # (26 lines, documented in dfire_qa_report.json / PROVENANCE.md);
                # counted separately, not a failure
                rec["degenerate"] = rec.get("degenerate", 0) + 1
    return rec


def main() -> int:
    result = {"date": "2026-10-01", "datasets": {}, "demo_untouched": {},
              "credential_scan": {}, "overall": None}
    ok_all = True

    for name, cfg in DATASETS.items():
        root = cfg["root"]
        allowed = set(cfg["names"])
        ds_res = {"splits": {}, "errors": 0}

        # --- YAML assertion ---
        yname = "dfire_native.yaml" if name == "dfire" else "hardhats_native.yaml"
        y = yaml.safe_load((root / yname).read_text())
        yaml_names = {int(k): v for k, v in y["names"].items()}
        ds_res["yaml_names"] = yaml_names
        ds_res["yaml_matches_verified_mapping"] = yaml_names == cfg["names"]
        ds_res["split_lists_used"] = {"train": y.get("train"), "val": y.get("val"),
                                      "test": y.get("test")}
        ok_all &= ds_res["yaml_matches_verified_mapping"]

        # --- split list validation ---
        for split in ("train", "val", "test"):
            lst = root / "leakage_aware" / f"{split}.txt"
            paths = [l for l in lst.read_text().splitlines() if l]
            tasks = [(p, str(str(p).replace("/images/", "/labels/")).rsplit(".", 1)[0] + ".txt",
                      allowed) for p in paths]
            n_boxes = Counter()
            n_neg = 0
            n_err = 0
            with ProcessPoolExecutor(max_workers=2) as ex:
                for r in ex.map(check_image, tasks, chunksize=64):
                    if not r["ok"]:
                        n_err += 1
            # recount boxes/negatives cheaply
            for p in paths:
                lp = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
                t = lp.read_text().strip()
                if not t:
                    n_neg += 1
                    continue
                for line in t.splitlines():
                    q = line.split()
                    n_boxes[int(q[0])] += 1
            ds_res["splits"][split] = {
                "images": len(paths),
                "invalid_images": n_err,
                "boxes": sum(n_boxes.values()),
                "class_counts": {cfg["names"][k]: int(v) for k, v in sorted(n_boxes.items())},
                "negatives": n_neg,
            }
            ok_all &= n_err == 0
            print(f"[{name}/{split}] {len(paths)} imgs, {sum(n_boxes.values())} boxes, "
                  f"{n_err} invalid, classes={dict(ds_res['splits'][split]['class_counts'])}")

        # --- leakage verification ---
        lr = json.loads((root / "leakage_aware" / "leakage_aware_report.json").read_text())
        lv = lr["leakage_verification"]
        ds_res["leakage"] = {
            "cross_split_le8_NEW": lv["cross_split_pairs_hamming<=8_NEW"],
            "cross_split_le2_OLD_official": lv["cross_split_pairs_hamming<=2_OFFICIAL_OLD"],
            "groups_total": lr["groups_total"],
            "quarantined_exact_dups": lr["exact_duplicates_quarantined"],
        }
        ok_all &= lv["cross_split_pairs_hamming<=8_NEW"] == 0
        result["datasets"][name] = ds_res

    # --- construction_safety + coco person reference summary ---
    cs = json.loads((PROD / "construction_safety" / "construction_safety_qa_report.json").read_text())
    result["datasets"]["construction_safety"] = {
        "status": "EVIDENCE_ONLY - not approved for training",
        "errors_total": cs["errors_total"],
        "overlap_vs_hardhats_d0": json.loads(
            (PROD / "construction_safety" / "cross_dataset_overlap_vs_hardhats.json").read_text()
        )["pairs_hamming<=0"],
    }
    pr = json.loads((PROD / "coco_person_reference" / "PERSON_REFERENCE_REPORT.json").read_text())
    result["datasets"]["coco_person_reference"] = {
        "status": "EVAL_REFERENCE_ONLY", "images": pr["images"],
        "person_boxes": pr["person_boxes"], "unreadable": pr["unreadable_decode_check"],
    }

    # --- demo untouched (count only the split image dirs) ---
    for demo, expected in (("ppe", 17), ("fire_smoke", 20)):
        n = sum(1 for _ in (REPO / "datasets" / demo / "images").rglob("*.jpg")) \
            + sum(1 for _ in (REPO / "datasets" / demo / "images").rglob("*.png"))
        result["demo_untouched"][demo] = {"images": n, "expected": expected,
                                          "ok": n == expected}
        ok_all &= n == expected

    # --- credential scan ---
    hits = []
    for pat in ("*.txt", "*.yaml", "*.yml", "*.json", "*.md"):
        for f in PROD.rglob(pat):
            if f.name.startswith("."):
                continue
            try:
                if CRED_PAT.search(f.read_text(errors="ignore")):
                    hits.append(str(f.relative_to(REPO)))
            except Exception:
                pass
    for f in PROD.rglob("*"):
        if f.suffix.lower() in {".env", ".key", ".pem"} or f.name == "kaggle.json":
            hits.append(str(f.relative_to(REPO)))
    result["credential_scan"] = {"files_scanned_root": str(PROD), "hits": hits}
    ok_all &= not hits

    result["overall"] = "PASS" if ok_all else "FAIL"
    (PROD / "DATASET_VALIDATION_FINAL.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("demo_untouched", "credential_scan",
                                             "overall")}, indent=2))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
