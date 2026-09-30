#!/usr/bin/env python3
"""Kaggle transfer audit — READ-ONLY verification of the exact approved M1/M2 data.

Verifies, for both approved datasets:
  1. every image referenced by the leakage-aware train/val/test lists exists
  2. every corresponding label file exists
  3. full class-ID scan of all referenced labels (must be {0,1} per dataset)
  4. exact byte totals of the required image+label payload
  5. support-file inventory (lists, YAMLs, manifests, reports)
  6. disk headroom vs estimated transfer archive size
No file is modified. Output: outputs/training/kaggle_transfer_audit.json + stdout.
"""
import json
import pathlib
import shutil

REPO = pathlib.Path("/home/z/my-project/Factory-AI")
DATASETS = {
    "m1_dfire": {
        "yaml": REPO / "datasets/production/dfire/dfire_native.yaml",
        "la_dir": REPO / "datasets/production/dfire/leakage_aware",
        "class_ids_expected": [0, 1],
        "class_semantics": {"0": "smoke", "1": "fire"},
    },
    "m2_hardhats": {
        "yaml": REPO / "datasets/production/hardhats/hardhats_native.yaml",
        "la_dir": REPO / "datasets/production/hardhats/leakage_aware",
        "class_ids_expected": [0, 1],
        "class_semantics": {"0": "hardhat (report as helmet)", "1": "no-hardhat (report as no_helmet)"},
    },
}
SUPPORT_FILES = {
    "m1_dfire": [
        "datasets/production/dfire/dfire_native.yaml",
        "datasets/production/dfire/leakage_aware/train.txt",
        "datasets/production/dfire/leakage_aware/val.txt",
        "datasets/production/dfire/leakage_aware/test.txt",
        "datasets/production/dfire/leakage_aware/quarantine.txt",
        "datasets/production/dfire/leakage_aware/quarantine_manifest.csv",
        "datasets/production/dfire/leakage_aware/leakage_aware_report.json",
        "datasets/production/dfire/leakage_aware/leakage_aware_report.md",
        "datasets/production/dfire/PROVENANCE.md",
        "datasets/production/dfire/CLASS_MAPPING_VERIFICATION.md",
        "datasets/production/dfire/CLASS_MAPPING_VERIFICATION.json",
    ],
    "m2_hardhats": [
        "datasets/production/hardhats/hardhats_native.yaml",
        "datasets/production/hardhats/leakage_aware/train.txt",
        "datasets/production/hardhats/leakage_aware/val.txt",
        "datasets/production/hardhats/leakage_aware/test.txt",
        "datasets/production/hardhats/leakage_aware/quarantine.txt",
        "datasets/production/hardhats/leakage_aware/quarantine_manifest.csv",
        "datasets/production/hardhats/leakage_aware/leakage_aware_report.json",
        "datasets/production/hardhats/leakage_aware/leakage_aware_report.md",
        "datasets/production/hardhats/PROVENANCE.md",
    ],
}
SHARED_REPORTS = [
    "DATASET_READY_FOR_TRAINING.md",
    "datasets/production/DATASET_VALIDATION_FINAL.json",
    "datasets/production/PPE_ARCHITECTURE_DECISION.md",
    "datasets/REACQUISITION_REPORT.md",
    "datasets/DATASET_REPORT.md",
    "datasets/DATASET_STRATEGY.md",
    "requirements.txt",
    "scripts/train_approved.sh",
    "scripts/run_approved_pipeline.sh",
    "scripts/eval_approved_model.py",
    "scripts/benchmark_inference.py",
    "scripts/failure_cases.py",
    "scripts/qualitative_approved.py",
]
BASE_MODEL = "models/yolo26n.pt"


def audit_dataset(name: str, cfg: dict):
    la_dir = cfg["la_dir"]
    result = {"dataset": name, "splits": {}, "images_missing": [], "labels_missing": [],
              "label_class_ids_seen": set(), "label_parse_errors": 0,
              "images_bytes": 0, "labels_bytes": 0, "labels_scanned": 0}
    for split in ("train", "val", "test"):
        lst = la_dir / f"{split}.txt"
        imgs = [pathlib.Path(p) for p in lst.read_text().split()]
        missing = [str(p) for p in imgs if not p.is_file()]
        nbytes = sum(p.stat().st_size for p in imgs if p.is_file())
        result["splits"][split] = {"list": str(lst), "images": len(imgs),
                                   "missing": len(missing), "bytes": nbytes}
        result["images_missing"].extend(missing[:20])
        result["images_bytes"] += nbytes
        for p in imgs:
            lbl = pathlib.Path(str(p).replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
            if not lbl.is_file():
                result["labels_missing"].append(str(lbl))
                continue
            result["labels_bytes"] += lbl.stat().st_size
            result["labels_scanned"] += 1
            try:
                for ln in lbl.read_text().splitlines():
                    if ln.strip():
                        result["label_class_ids_seen"].add(int(ln.split()[0]))
            except Exception:
                result["label_parse_errors"] += 1
    result["label_class_ids_seen"] = sorted(result["label_class_ids_seen"])
    result["class_ids_ok"] = result["label_class_ids_seen"] == cfg["class_ids_expected"]
    result["class_semantics"] = cfg["class_semantics"]

    report = json.loads((la_dir / "leakage_aware_report.json").read_text())
    result["split_method"] = {
        "method": report.get("method"), "seed": report.get("seed"),
        "parameters": report.get("parameters"),
        "groups_total": report.get("groups_total"),
        "images_used": report.get("images_used"),
        "exact_duplicates_quarantined": report.get("exact_duplicates_quarantined"),
        "leakage_verification": report.get("leakage_verification"),
    }
    return result


def main():
    audit = {"datasets": {}, "support_files": {}, "shared": {}}
    total_payload = 0
    for name, cfg in DATASETS.items():
        res = audit_dataset(name, cfg)
        audit["datasets"][name] = res
        total_payload += res["images_bytes"] + res["labels_bytes"]
        sup = {}
        for rel in SUPPORT_FILES[name]:
            p = REPO / rel
            sup[rel] = {"exists": p.is_file(), "bytes": p.stat().st_size if p.is_file() else None}
        audit["support_files"][name] = sup

    for rel in SHARED_REPORTS + [BASE_MODEL]:
        p = REPO / rel
        audit["shared"][rel] = {"exists": p.is_file(), "bytes": p.stat().st_size if p.is_file() else None}

    free = shutil.disk_usage("/home/z").free
    audit["disk"] = {"free_bytes": free, "free_GB": round(free / 1e9, 2)}
    # stored-mode zip of jpg payload ~= payload size (jpgs are incompressible)
    audit["transfer_estimate"] = {
        "images_labels_payload_bytes": total_payload,
        "payload_GB": round(total_payload / 1e9, 2),
        "zip_estimate_GB": round(total_payload / 1e9 * 1.01, 2),
        "note": "jpg payloads are already compressed; stored zip ~= payload size",
    }
    out = REPO / "outputs/training/kaggle_transfer_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, default=str))

    for name, res in audit["datasets"].items():
        s = res["splits"]
        print(f"=== {name} ===")
        for k in ("train", "val", "test"):
            print(f"  {k}: {s[k]['images']} imgs, missing={s[k]['missing']}, {s[k]['bytes']/1e9:.2f} GB")
        print(f"  labels scanned: {res['labels_scanned']}, missing: {len(res['labels_missing'])}, "
              f"parse errors: {res['label_parse_errors']}")
        print(f"  class IDs seen: {res['label_class_ids_seen']} ok={res['class_ids_ok']} "
              f"({res['class_semantics']})")
        m = res["split_method"]
        print(f"  method: {m['method']}, seed {m['seed']}, groups {m['groups_total']}, "
              f"quarantined {sum(v for v in m['exact_duplicates_quarantined'].values()) if isinstance(m['exact_duplicates_quarantined'], dict) else m['exact_duplicates_quarantined']}")
        lv = m["leakage_verification"]
        print(f"  leakage verification: {json.dumps(lv)[:220]}")
    print(f"payload total: {audit['transfer_estimate']['payload_GB']} GB; "
          f"zip est {audit['transfer_estimate']['zip_estimate_GB']} GB; free disk {audit['disk']['free_GB']} GB")
    print(f"audit -> {out}")


if __name__ == "__main__":
    main()
