#!/usr/bin/env python3
"""PS6 core PPE demo: person tracking + PPE detection + compliance.

Usage example:
  python scripts/run_ppe_demo.py --source videos/test/factory.mp4 \
    --person-model models/yolo26n.pt \
    --ppe-model models/ppe_best.pt \
    --output outputs/ppe_demo.mp4
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO

from compliance_engine import Detection, Worker, associate_ppe, PersistentViolationGate


def xyxy_tuple(box) -> tuple[float, float, float, float]:
    a = box.xyxy[0].tolist()
    return float(a[0]), float(a[1]), float(a[2]), float(a[3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--person-model", default="yolo26n.pt")
    ap.add_argument("--ppe-model", required=True)
    ap.add_argument("--output", default="outputs/ppe_demo.mp4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    person_model = YOLO(args.person_model)
    ppe_model = YOLO(args.ppe_model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    gate = PersistentViolationGate(confirm_frames=3, clear_frames=2)

    # Unique, never-reused fallback IDs for person boxes the tracker could not
    # assign a track to. They are negative and strictly decreasing, so they can
    # never collide with real tracker IDs (positive) nor with each other across
    # frames. Because a fallback ID exists for a single frame only, the
    # persistence gate accumulates no history for it: untracked workers still
    # get per-frame compliance status but cannot trigger persistent-violation
    # alerts (no temporal identity -> no temporal evidence).
    fallback_ids = itertools.count(-1, -1)

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1

        person_result = person_model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.35,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        ppe_result = ppe_model.predict(
            frame,
            conf=0.25,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        workers: List[Worker] = []
        if person_result.boxes is not None:
            for b in person_result.boxes:
                track_id = int(b.id.item()) if b.id is not None else next(fallback_ids)
                workers.append(Worker(track_id=track_id, xyxy=xyxy_tuple(b)))

        ppe_dets: List[Detection] = []
        names = ppe_result.names
        if ppe_result.boxes is not None:
            for b in ppe_result.boxes:
                cls_id = int(b.cls.item())
                ppe_dets.append(Detection(cls=str(names[cls_id]).lower(), xyxy=xyxy_tuple(b), conf=float(b.conf.item())))

        compliance = associate_ppe(workers, ppe_dets)

        # Draw PPE detections.
        for det in ppe_dets:
            x1, y1, x2, y2 = [int(v) for v in det.xyxy]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(frame, f"{det.cls} {det.conf:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1, cv2.LINE_AA)

        for worker in workers:
            item = compliance[worker.track_id]
            x1, y1, x2, y2 = [int(v) for v in worker.xyxy]
            status = "COMPLIANT" if item.compliant else "NON-COMPLIANT"
            color = (0, 200, 0) if item.compliant else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"Worker #{worker.track_id}: {status}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            if item.missing:
                cv2.putText(frame, "Missing: " + ", ".join(item.missing), (x1, min(height - 10, y2 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            event = gate.update(worker.track_id, item.missing)
            if event:
                print({"frame": frame_no, **event})

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved demo: {out_path}")


if __name__ == "__main__":
    main()
