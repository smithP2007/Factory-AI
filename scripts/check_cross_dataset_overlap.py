#!/usr/bin/env python3
"""Cross-dataset near-duplicate check (evidence for keep-separate decisions)."""
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/z/my-project/Factory-AI")
sys.path.insert(0, str(REPO / "scripts"))
from leakage_aware_split import load_hashes, hamming_pairs  # reuse


def main() -> int:
    a_root = REPO / "datasets/production/hardhats"
    b_root = REPO / "datasets/production/construction_safety"
    na, ha = load_hashes(a_root, ["train", "valid", "test"],
                         a_root / "leakage_aware/.hash_cache.npz")
    nb, hb = load_hashes(b_root, ["train", "valid", "test"],
                         b_root / ".hash_cache_tmp.npz")
    ha = np.array([int(x) for x in ha], dtype=np.uint64)
    hb = np.array([int(x) for x in hb], dtype=np.uint64)
    x = ha[:, None] ^ hb[None, :]
    hd = np.bitwise_count(x)
    res = {}
    for t in (0, 2, 4, 8):
        idx = np.argwhere(hd <= t)
        res[f"pairs_hamming<={t}"] = int(len(idx))
        if t == 0 and len(idx):
            res["d0_examples"] = [
                {"hardhats": Path(na[i]).name, "construction_safety": Path(nb[j]).name}
                for i, j in idx.tolist()[:10]]
    print(f"hardhats {len(na)} x construction_safety {len(nb)}")
    import json
    print(json.dumps(res, indent=2))
    out = b_root / "cross_dataset_overlap_vs_hardhats.json"
    out.write_text(json.dumps({"hardhats_images": len(na),
                               "construction_safety_images": len(nb), **res}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
