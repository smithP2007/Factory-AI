#!/usr/bin/env bash
# Kaggle-side training adapter for approved M1/M2 (GPU-gated, hard refusal).
# Runs INSIDE a Kaggle notebook:  !bash kaggle_train.sh m1
# Data root follows the Kaggle input mounts produced by build_kaggle_package.py:
#   /kaggle/input/ps6-m1-dfire/    (dfire payload + yaml + scripts + base model + unseen assets)
#   /kaggle/input/ps6-m2-hardhats/ (hardhats payload + yaml + scripts + base model + unseen assets)
set -euo pipefail

TARGET="${1:?usage: kaggle_train.sh m1|m2}"
KAGGLE_INPUT="${KAGGLE_INPUT:-/kaggle/input}"

case "$TARGET" in
  m1) SLUG=ps6-m1-dfire;    TAG=m1_fire_smoke; YAML="$KAGGLE_INPUT/ps6-m1-dfire/dfire_kaggle.yaml";    BEST=m1_fire_smoke_best.pt ;;
  m2) SLUG=ps6-m2-hardhats; TAG=m2_helmet;     YAML="$KAGGLE_INPUT/ps6-m2-hardhats/hardhats_kaggle.yaml"; BEST=m2_helmet_best.pt ;;
  *) echo "unknown target $TARGET"; exit 1 ;;
esac
PKG="$KAGGLE_INPUT/$SLUG"

# HARD GPU GUARD (identical policy to scripts/train_approved.sh)
python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("REFUSING TO TRAIN: no CUDA accelerator available on this Kaggle session. "
             "Enable GPU (Settings -> Accelerator) and re-run.")
props = torch.cuda.get_device_properties(0)
print(f"GPU OK: {props.name} ({props.total_memory/1e9:.1f} GB)")
PY

export YOLO_CONFIG_DIR=/tmp/ultralytics
export UNSEEN_ROOT="$PKG/unseen_assets"
mkdir -p /tmp/ultralytics /kaggle/working/models /kaggle/working/outputs/training

echo "=== [$TARGET] approved training START $(date -Iseconds) ==="
python3 - "$YAML" "$TAG" "$PKG" <<'PY'
import sys
from ultralytics import YOLO
data, name, pkg = sys.argv[1], sys.argv[2], sys.argv[3]
model = YOLO(f"{pkg}/models/yolo26n.pt")   # pretrained base shipped in package; file is never modified
model.train(data=data, epochs=100, imgsz=640, device=0,
            project='/kaggle/working/runs/detect', name=name)
PY

BEST_RUN="/kaggle/working/runs/detect/$TAG/weights/best.pt"
test -f "$BEST_RUN" || { echo "ERROR: best.pt not produced"; exit 1; }
cp "$BEST_RUN" "/kaggle/working/models/$BEST"
echo "=== [$TARGET] DONE $(date -Iseconds); best -> /kaggle/working/models/$BEST ==="

echo "--- approved post-training evaluation (val+test, benchmark, failure cases, qualitative) ---"
python3 "$PKG/scripts/eval_approved_model.py" \
    "/kaggle/working/models/$BEST" "$YAML" "$TAG" || echo "[warn] eval failed"
python3 "$PKG/scripts/benchmark_inference.py" \
    "/kaggle/working/models/$BEST" "$YAML" "$TAG" || echo "[warn] benchmark failed"
python3 "$PKG/scripts/failure_cases.py" \
    "/kaggle/working/models/$BEST" "$YAML" "$TAG" || echo "[warn] failure cases failed"
python3 "$PKG/scripts/qualitative_approved.py" \
    "/kaggle/working/models/$BEST" "$TAG" "$TARGET" || echo "[warn] qualitative failed"
echo "--- artifacts: /kaggle/working/models + /kaggle/working/outputs + /kaggle/working/runs ---"
