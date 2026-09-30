#!/usr/bin/env python3
"""Leakage-aware, group-aware re-split for the reacquired production datasets.

Problem (documented in the reacquisition QA):
  official splits leak near-identical video frames across train/val/test
  (D-Fire train<->test: 35,733 pairs at Hamming <= 2; Hard Hats train<->test
  5,438 + train<->valid 10,974). Random image-level splitting is FORBIDDEN.

Method:
  1. dHash64 perceptual hash for every image (same hash as qa_acquired_dataset.py)
  2. exact-duplicate quarantine: d == 0 cliques keep ONE representative
  3. union-find grouping:
       Rule A (source key): D-Fire stem; Hard Hats Roboflow base name
                            (stem minus ".rf.<32-hex>" - same base = same
                            original image exported into multiple splits)
       Rule B (perceptual): any pair with Hamming <= 8 joins one group
       Rule C (temporal, D-Fire only): same numeric-series prefix, sequence
                            gap <= 2 AND Hamming <= 16 (slow-video drift)
  4. group-level stratified split: whole groups assigned to train/val/test,
     greedy largest-first against per-profile ratio targets (seeded, deterministic)
  5. verification: every d<=8 pair must be intra-split; zero cross-split pairs

Outputs per dataset (under datasets/production/<name>/leakage_aware/):
  train.txt / val.txt / test.txt   absolute image paths (no files copied)
  quarantine_manifest.csv          exact duplicates removed from use
  leakage_aware_report.json / .md  groups, counts, class counts, verification
Hash cache: .hash_cache.npz (gitignored, regenerable).

Usage:
  python scripts/leakage_aware_split.py --name dfire \
      --root datasets/production/dfire --classes 0:smoke,1:fire \
      --official-splits train,test --ratios 0.8,0.1,0.1
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/home/z/my-project/Factory-AI")
SEED = 20261001
IMG_EXT = {".jpg", ".jpeg", ".png"}


def dhash64(img) -> int:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    diff = (g[:, 1:] > g[:, :-1]).flatten()
    return int.from_bytes(np.packbits(diff).tobytes(), "big")


def hash_one(path_str: str):
    img = cv2.imread(path_str)
    if img is None:
        return path_str, None
    return path_str, dhash64(img)


def hamming_pairs(hashes: dict, threshold: int):
    """All pairs with Hamming distance <= threshold. Chunked numpy popcount."""
    names = list(hashes)
    arr = np.array([hashes[n] for n in names], dtype=np.uint64)
    n = len(names)
    out = []
    CH = 512
    for i in range(0, n, CH):
        chunk = arr[i:i + CH]
        x = chunk[:, None] ^ arr[None, :]
        hd = np.bitwise_count(x)
        for r in range(x.shape[0]):
            g = i + r
            row = hd[r]
            for c in np.where(row[g + 1:] <= threshold)[0]:
                c = int(c) + g + 1
                out.append((names[g], names[c], int(row[c])))
        del x, hd
        if (i // CH) % 10 == 0:
            print(f"    pair-scan {i + len(chunk)}/{n}", flush=True)
    return out


class UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def load_hashes(root, official_splits, cache: Path):
    names, hashes = [], []
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        names, hashes = list(z["names"]), list(z["hashes"])
        print(f"  hash cache hit: {len(names)} entries")
    else:
        files = []
        for s in official_splits:
            for p in sorted((root / s / "images").iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    files.append(str(p))
        with ProcessPoolExecutor(max_workers=2) as ex:
            for i, (p, h) in enumerate(ex.map(hash_one, files, chunksize=64)):
                if h is None:
                    print(f"  WARN unreadable: {p}")
                    continue
                names.append(p)
                hashes.append(int(h))
                if (i + 1) % 5000 == 0:
                    print(f"  hashed {i + 1}/{len(files)}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, names=np.array(names), hashes=np.array(hashes, dtype=np.uint64))
    return names, np.array(hashes, dtype=np.uint64)


def base_name(path: str, dataset: str) -> str:
    stem = Path(path).stem
    if dataset == "hardhats":
        return re.sub(r"\.rf\.[0-9a-f]{32}$", "", stem)
    return stem


def series_key(path: str, dataset: str):
    """(prefix, seq) for temporal adjacency (Rule C). D-Fire numbered series only.
    Returns None when the stem carries no trustworthy series semantics."""
    stem = Path(path).stem
    if dataset != "dfire":
        return None
    m = re.match(r"^([A-Za-z]+)(\d+)$", stem)
    if not m:
        return None
    prefix, seq = m.group(1), int(m.group(2))
    if prefix == "WEB":       # web-crawl batch numbering has no video semantics
        return None
    return prefix, seq


def label_profile(path: str):
    lp = Path(str(path).replace("/images/", "/labels/")).with_suffix(".txt")
    classes = set()
    if lp.exists():
        for line in lp.read_text().strip().splitlines():
            p = line.split()
            if len(p) == 5:
                classes.add(int(p[0]))
    return tuple(sorted(classes))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--classes", required=True, help="0:smoke,1:fire")
    ap.add_argument("--official-splits", default="train,valid,test")
    ap.add_argument("--ratios", required=True, help="train,val,test e.g. 0.8,0.1,0.1")
    ap.add_argument("--group-threshold", type=int, default=8)
    ap.add_argument("--quarantine-threshold", type=int, default=0)
    args = ap.parse_args()

    root = REPO / args.root
    out = root / "leakage_aware"
    out.mkdir(parents=True, exist_ok=True)
    class_names = {int(k): v for k, v in
                   (p.split(":") for p in args.classes.split(","))}
    splits = ["train", "val", "test"]
    ratios = [float(x) for x in args.ratios.split(",")]
    official = [s for s in args.official_splits.split(",") if s]
    ds = args.name

    # ---------- 1. hashes ----------
    print(f"[{ds}] hashing images ...")
    names, arr = load_hashes(root, official, out / ".hash_cache.npz")
    hashes = dict(zip(names, (int(v) for v in arr)))
    print(f"[{ds}] {len(names)} images hashed")

    # ---------- 2. exact-duplicate quarantine (d == 0) ----------
    print(f"[{ds}] exact-duplicate scan (d == 0) ...")
    pairs8 = hamming_pairs(hashes, args.group_threshold)
    pairs0 = [(a, b, h) for a, b, h in pairs8 if h <= args.quarantine_threshold]
    uf0 = UF()
    for a, b, _ in pairs0:
        uf0.union(a, b)
    rep_of = defaultdict(list)
    for nme in names:
        rep_of[uf0.find(nme)].append(nme)
    quarantined = {}
    for r, members in rep_of.items():
        if len(members) < 2:
            continue
        keep = min(members)  # deterministic representative
        for m in sorted(members):
            if m != keep:
                quarantined[m] = keep
    print(f"[{ds}] exact duplicates quarantined: {len(quarantined)} "
          f"(kept {len(names) - len(quarantined)})")

    # ---------- 3. grouping on kept images ----------
    print(f"[{ds}] grouping (rules A/B/C, threshold <= {args.group_threshold}) ...")
    kept = [n for n in names if n not in quarantined]
    kept_set = set(kept)
    uf = UF()
    for nme in kept:
        uf.find(nme)
    # Rule A: shared source key
    by_base = defaultdict(list)
    for nme in kept:
        by_base[base_name(nme, ds)].append(nme)
    for base, members in by_base.items():
        for m in members[1:]:
            uf.union(members[0], m)
    # Rule B: perceptual near-duplicate pairs
    n_pairs_b = 0
    for a, b, h in pairs8:
        if a in kept_set and b in kept_set:
            uf.union(a, b)
            n_pairs_b += 1
    # Rule C: temporal adjacency within numbered series (D-Fire only)
    by_series = defaultdict(list)
    for nme in kept:
        k = series_key(nme, ds)
        if k:
            by_series[k[0]].append((k[1], nme))
    n_pairs_c = 0
    for prefix, seqs in by_series.items():
        seqs.sort()
        for (s1, n1), (s2, n2) in zip(seqs, seqs[1:]):
            if s2 - s1 <= 2 and hashes[n1] ^ hashes[n2] != 0:
                d = int(np.bitwise_count(np.uint64(hashes[n1] ^ hashes[n2])))
                if d <= 16:
                    uf.union(n1, n2)
                    n_pairs_c += 1
    groups = defaultdict(list)
    for nme in kept:
        groups[uf.find(nme)].append(nme)
    print(f"[{ds}] groups: {len(groups)} (ruleB pairs joined: {n_pairs_b}, "
          f"ruleC joins: {n_pairs_c})")

    # ---------- 4. group-level stratified split ----------
    print(f"[{ds}] group-level stratified split (ratios {ratios}) ...")
    rng = random.Random(SEED)
    prof_names = {"dfire": {(): "negative", (0,): "smoke_only", (1,): "fire_only",
                            (0, 1): "fire_and_smoke"},
                  "hardhats": {(): "negative", (0,): "hardhat_only",
                               (1,): "no_hardhat_only", (0, 1): "mixed"}}
    pname = prof_names[ds]
    g_profile, g_size = {}, {}
    for gid, members in groups.items():
        profs = Counter(pname.get(label_profile(m), f"classes{label_profile(m)}")
                        for m in members)
        g_profile[gid] = max(profs, key=profs.get)   # dominant profile
        g_size[gid] = len(members)
    by_prof = defaultdict(list)
    for gid in groups:
        by_prof[g_profile[gid]].append(gid)
    total_kept = len(kept)
    assign = {}
    prof_report = {}
    for prof in sorted(by_prof, key=lambda p: -sum(g_size[g] for g in by_prof[p])):
        gids = sorted(by_prof[prof], key=lambda g: (-g_size[g], g))
        rng.shuffle(gids)
        gids.sort(key=lambda g: -g_size[g])   # seeded order, largest first
        ptot = sum(g_size[g] for g in gids)
        target = {s: ratios[i] * ptot for i, s in enumerate(splits)}
        cur = {s: 0 for s in splits}
        for gid in gids:
            deficits = {s: target[s] - cur[s] for s in splits}
            best = max(splits, key=lambda s: (deficits[s], -splits.index(s)))
            assign[gid] = best
            cur[best] += g_size[gid]
        prof_report[prof] = {"images": ptot, "groups": len(gids),
                             **{f"{s}_images": cur[s] for s in splits}}

    # ---------- 5. write split lists ----------
    members_of = {s: [] for s in splits}
    for gid, s in assign.items():
        members_of[s].extend(sorted(groups[gid]))
    for s in splits:
        members_of[s].sort()
        (out / f"{s}.txt").write_text("\n".join(members_of[s]) + "\n")
    qpath = sorted(quarantined)
    (out / "quarantine.txt").write_text("\n".join(qpath) + "\n")
    with (out / "quarantine_manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quarantined_image", "kept_representative", "reason"])
        for m in qpath:
            w.writerow([m, quarantined[m], "exact_duplicate_dhash_d0"])

    # ---------- 6. verification + statistics ----------
    print(f"[{ds}] verification ...")
    split_of = {nme: s for s in splits for nme in members_of[s]}
    cross = Counter()          # hamming -> count of cross-split pairs
    intra = Counter()
    for a, b, h in pairs8:
        if a in quarantined or b in quarantined:
            continue
        if split_of.get(a) != split_of.get(b):
            cross[h] += 1
        else:
            intra[h] += 1
    old_cross = Counter()          # official splits, ALL images (matches reacquisition audit)
    old_cross_postq = Counter()    # official splits, post-quarantine (apples-to-apples with NEW)
    off_of = {}
    for s in official:
        for p in (root / s / "images").iterdir():
            if p.suffix.lower() in IMG_EXT:
                off_of[str(p)] = s
    for a, b, h in pairs8:
        if off_of.get(a) != off_of.get(b):
            old_cross[h] += 1
            if a not in quarantined and b not in quarantined:
                old_cross_postq[h] += 1

    def class_counts(lst):
        cc = Counter()
        for nme in lst:
            for line in Path(str(nme).replace("/images/", "/labels/")).with_suffix(".txt") \
                    .read_text().strip().splitlines():
                p = line.split()
                if len(p) == 5:
                    cc[int(p[0])] += 1
        return cc

    split_stats = {}
    for s in splits:
        cc = class_counts(members_of[s])
        split_stats[s] = {
            "images": len(members_of[s]),
            "groups": sum(1 for gid in assign if assign[gid] == s),
            "boxes": sum(cc.values()),
            "class_counts": {class_names[k]: int(v) for k, v in sorted(cc.items())},
        }
    neg = {s: sum(1 for nme in members_of[s] if not label_profile(nme))
           for s in splits}

    report = {
        "dataset": ds,
        "method": "leakage-aware group-aware re-split (dHash64 + source keys + temporal adjacency)",
        "seed": SEED,
        "parameters": {
            "group_threshold_hamming": args.group_threshold,
            "temporal_rule": "D-Fire only: same prefix, seq gap<=2, hamming<=16",
            "ratios_target": dict(zip(splits, ratios)),
            "hash_cache": str(out / ".hash_cache.npz"),
        },
        "images_total_official": len(names),
        "exact_duplicates_quarantined": len(quarantined),
        "images_used": total_kept,
        "groups_total": len(groups),
        "group_size_distribution": {
            "1": sum(1 for g in groups.values() if len(g) == 1),
            "2-5": sum(1 for g in groups.values() if 2 <= len(g) <= 5),
            "6-50": sum(1 for g in groups.values() if 6 <= len(g) <= 50),
            "51-500": sum(1 for g in groups.values() if 51 <= len(g) <= 500),
            ">500": sum(1 for g in groups.values() if len(g) > 500),
        },
        "largest_groups": sorted((len(g) for g in groups.values()), reverse=True)[:10],
        "profile_assignment": prof_report,
        "splits": split_stats,
        "negatives_per_split": neg,
        "leakage_verification": {
            "cross_split_pairs_hamming<=8_NEW": sum(v for h, v in cross.items() if h <= 8),
            "cross_split_pairs_hamming<=4_NEW": sum(v for h, v in cross.items() if h <= 4),
            "cross_split_pairs_hamming<=2_NEW": sum(v for h, v in cross.items() if h <= 2),
            "cross_split_pairs_hamming<=8_OFFICIAL_OLD": sum(
                v for h, v in old_cross.items() if h <= 8),
            "cross_split_pairs_hamming<=2_OFFICIAL_OLD": sum(
                v for h, v in old_cross.items() if h <= 2),
            "cross_split_pairs_hamming<=8_OFFICIAL_OLD_post_quarantine": sum(
                v for h, v in old_cross_postq.items() if h <= 8),
            "cross_split_pairs_hamming<=2_OFFICIAL_OLD_post_quarantine": sum(
                v for h, v in old_cross_postq.items() if h <= 2),
            "verdict": ("PASS - zero cross-split near-duplicate pairs at Hamming <= 8"
                        if sum(v for h, v in cross.items() if h <= 8) == 0
                        else "FAIL - leakage remains"),
        },
        "intra_split_near_dup_pairs<=8": sum(intra.values()),
        "outputs": {s: str(out / f"{s}.txt") for s in splits},
    }
    (out / "leakage_aware_report.json").write_text(json.dumps(report, indent=2))

    md = [
        f"# Leakage-aware re-split — {ds}",
        "",
        f"Method: dHash64 perceptual grouping (<= {args.group_threshold}) + source-key "
        "grouping + temporal adjacency (D-Fire series); exact duplicates (d=0) quarantined; "
        "whole groups assigned to splits, stratified greedy largest-first "
        f"(seed {SEED}). No images copied - split lists reference originals.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Images (official, all splits) | {len(names)} |",
        f"| Exact duplicates quarantined (dHash d=0) | {len(quarantined)} |",
        f"| Images used | {total_kept} |",
        f"| Groups (near-dup/source-connected) | {len(groups)} |",
    ]
    for s in splits:
        st = split_stats[s]
        md.append(f"| {s}: images / groups | {st['images']} / {st['groups']} |")
        md.append(f"| {s}: boxes {st['class_counts']} | |")
    lv = report["leakage_verification"]
    md += [
        "",
        "## Leakage verification",
        "",
        "| Cross-split near-duplicate pairs | Official split (OLD) | Leakage-aware (NEW) |",
        "|---|---:|---:|",
        f"| Hamming <= 2 | {lv['cross_split_pairs_hamming<=2_OFFICIAL_OLD']} | {lv['cross_split_pairs_hamming<=2_NEW']} |",
        f"| Hamming <= 8 | {lv['cross_split_pairs_hamming<=8_OFFICIAL_OLD']} | {lv['cross_split_pairs_hamming<=8_NEW']} |",
        "",
        f"(OLD column counts the official split as distributed, all images - it reproduces the "
        f"reacquisition audit. Post-quarantine OLD at <=2/<=8: "
        f"{lv['cross_split_pairs_hamming<=2_OFFICIAL_OLD_post_quarantine']} / "
        f"{lv['cross_split_pairs_hamming<=8_OFFICIAL_OLD_post_quarantine']} - the exact-duplicate "
        "quarantine alone removes most of the leakage, grouping removes the rest.)",
        "",
        f"**Verdict: {lv['verdict']}**",
        "",
        f"Intra-split near-duplicate pairs (same scene, same split, <= 8): "
        f"{report['intra_split_near_dup_pairs<=8']} - expected for video-frame data, "
        "harmless for training/evaluation integrity.",
        "",
        "## Files",
        "",
        f"- `train.txt` / `val.txt` / `test.txt` - absolute image paths",
        f"- `quarantine_manifest.csv` - every exact duplicate removed, with its kept representative",
        f"- `leakage_aware_report.json` - full machine-readable report",
    ]
    (out / "leakage_aware_report.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"splits": split_stats, "groups": len(groups),
                      "quarantined": len(quarantined),
                      "verification": lv}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
