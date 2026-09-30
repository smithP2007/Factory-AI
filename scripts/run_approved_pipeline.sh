#!/usr/bin/env bash
# Approved M1+M2 end-to-end pipeline (GPU ONLY): train -> eval -> benchmark ->
# failure cases -> qualitative, for both models, sequentially.
# Run detached:  nohup bash scripts/run_approved_pipeline.sh > outputs/training/orchestrator.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/training

exec > >(tee -a outputs/training/orchestrator.log) 2>&1

echo "############ APPROVED PIPELINE START $(date -Iseconds) ############"

# Hard GPU guard for the whole pipeline
python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("REFUSING TO RUN: no CUDA device. Approved spec requires GPU training.")
print("GPU:", torch.cuda.get_device_name(0))
PY
[ $? -ne 0 ] && exit 1

run_stage () {  # $1=model tag  $2=best.pt  $3=data.yaml
  local TAG="$1" BEST="$2" DATA="$3"
  local T0=$(date +%s)
  bash scripts/train_approved.sh "$TAG" || { echo "[$TAG] TRAIN FAILED"; return 1; }
  local T1=$(date +%s)
  mkdir -p "outputs/training/$TAG"
  echo "start $(date -Iseconds -d @$T0) end $(date -Iseconds -d @$T1) elapsed_s $((T1-T0))" \
      > "outputs/training/$TAG/duration.txt"
  # preserve training config + curves
  cp "runs/detect/$( [ "$TAG" = m1 ] && echo m1_fire_smoke || echo m2_helmet )/args.yaml" \
     "outputs/training/$TAG/args.yaml" 2>/dev/null || true
  cp "runs/detect/$( [ "$TAG" = m1 ] && echo m1_fire_smoke || echo m2_helmet )/results.csv" \
     "outputs/training/$TAG/results.csv" 2>/dev/null || true
  cp "runs/detect/$( [ "$TAG" = m1 ] && echo m1_fire_smoke || echo m2_helmet )/results.png" \
     "outputs/training/$TAG/results.png" 2>/dev/null || true

  echo "--- [$TAG] val/test evaluation ---"
  python3 scripts/eval_approved_model.py "$BEST" "$DATA" "$TAG" || echo "[$TAG] EVAL FAILED"
  echo "--- [$TAG] inference benchmark ---"
  python3 scripts/benchmark_inference.py "$BEST" "$DATA" "$TAG" || echo "[$TAG] BENCH FAILED"
  echo "--- [$TAG] failure cases ---"
  python3 scripts/failure_cases.py "$BEST" "$DATA" "$TAG" || echo "[$TAG] FAILCASE FAILED"
  echo "--- [$TAG] qualitative unseen ---"
  python3 scripts/qualitative_approved.py "$BEST" "$TAG" "$TAG" || echo "[$TAG] QUAL FAILED"
}

run_stage m1 models/m1_fire_smoke_best.pt datasets/production/dfire/dfire_native.yaml
echo "############ M1 PHASE COMPLETE $(date -Iseconds) ############"
run_stage m2 models/m2_helmet_best.pt datasets/production/hardhats/hardhats_native.yaml
echo "############ M2 PHASE COMPLETE $(date -Iseconds) ############"
echo "############ APPROVED PIPELINE END $(date -Iseconds) ############"
