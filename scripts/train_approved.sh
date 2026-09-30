#!/usr/bin/env bash
# Approved M1/M2 training launcher — GPU ONLY (hard guard below).
# Usage: bash scripts/train_approved.sh m1|m2
#
# Approved configuration (user authorization, 2026-10-01):
#   M1: base models/yolo26n.pt, data datasets/production/dfire/dfire_native.yaml,
#       classes 0=smoke 1=fire, leakage-aware split, imgsz=640, epochs=100, GPU.
#       Best checkpoint -> models/m1_fire_smoke_best.pt
#   M2: base models/yolo26n.pt, data datasets/production/hardhats/hardhats_native.yaml,
#       classes 0=helmet 1=no_helmet (native IDs 0=hardhat 1=no-hardhat, same order),
#       leakage-aware split, imgsz=640, epochs=100, GPU.
#       Best checkpoint -> models/m2_helmet_best.pt
# Constraints honored: no dataset merge, no vest/gloves/no_safety_vest training,
# approved splits untouched, models/yolo26n.pt never overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/home/z/my-project/.ultralytics}"
TARGET="${1:?usage: train_approved.sh m1|m2}"

# HARD GPU GUARD — refuses to train on CPU (per approved spec: GPU training)
python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("REFUSING TO TRAIN: no CUDA device available. Approved spec requires GPU training.")
props = torch.cuda.get_device_properties(0)
print(f"GPU OK: {props.name} ({props.total_memory/1e9:.1f} GB)")
PY

case "$TARGET" in
  m1)
    NAME=m1_fire_smoke
    DATA=datasets/production/dfire/dfire_native.yaml
    BEST=models/m1_fire_smoke_best.pt
    ;;
  m2)
    NAME=m2_helmet
    DATA=datasets/production/hardhats/hardhats_native.yaml
    BEST=models/m2_helmet_best.pt
    ;;
  *) echo "unknown target: $TARGET (use m1|m2)"; exit 1 ;;
esac

echo "=== [$TARGET] training START $(date -Iseconds) ==="
python3 - "$NAME" "$DATA" <<'PY'
import sys
from ultralytics import YOLO
name, data = sys.argv[1], sys.argv[2]
model = YOLO('models/yolo26n.pt')   # pretrained base; file itself is never modified
model.train(data=data, epochs=100, imgsz=640, device=0,
            project='runs/detect', name=name)
PY

BEST_RUN="runs/detect/$NAME/weights/best.pt"
test -f "$BEST_RUN" || { echo "ERROR: $BEST_RUN not produced"; exit 1; }
cp "$BEST_RUN" "$BEST"
echo "=== [$TARGET] training DONE $(date -Iseconds); best -> $BEST ==="
