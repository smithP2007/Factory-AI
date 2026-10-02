#!/usr/bin/env python3
"""Worker-level PPE association and persistent compliance logic for PS6.

The detector finds PPE objects; this module decides which worker they belong to and
whether the worker satisfies the required PPE rule. Missing PPE is inferred by the
rule engine and is NOT represented as a fake detector box.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math

PPE_CLASSES = ("helmet", "gloves", "goggles", "vest")

# Normalized expected PPE locations inside a person box.
ANCHORS = {
    "helmet": (0.50, 0.14),
    "goggles": (0.50, 0.26),
    "vest": (0.50, 0.52),
    "gloves": (0.50, 0.68),
}


@dataclass(frozen=True)
class Detection:
    cls: str
    xyxy: Tuple[float, float, float, float]
    conf: float


@dataclass(frozen=True)
class Worker:
    track_id: int
    xyxy: Tuple[float, float, float, float]


@dataclass
class WorkerCompliance:
    track_id: int
    present: Dict[str, bool] = field(default_factory=lambda: {c: False for c in PPE_CLASSES})
    associated: Dict[str, List[Detection]] = field(default_factory=lambda: {c: [] for c in PPE_CLASSES})
    missing: List[str] = field(default_factory=list)
    compliant: bool = False


def _center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _expanded_contains(point: Tuple[float, float], box: Sequence[float], margin: float = 0.15) -> bool:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ex1, ey1 = x1 - margin * w, y1 - margin * h
    ex2, ey2 = x2 + margin * w, y2 + margin * h
    x, y = point
    return ex1 <= x <= ex2 and ey1 <= y <= ey2


def _association_score(worker: Worker, det: Detection) -> float:
    px1, py1, px2, py2 = worker.xyxy
    w = max(px2 - px1, 1.0)
    h = max(py2 - py1, 1.0)
    cx, cy = _center(det.xyxy)
    nx = (cx - px1) / w
    ny = (cy - py1) / h
    ax, ay = ANCHORS.get(det.cls, (0.5, 0.5))
    # Gloves are often offset toward either side. Give their x-position less weight.
    x_weight = 0.15 if det.cls == "gloves" else 0.30
    y_weight = 0.70 if det.cls in {"helmet", "goggles"} else 0.55
    dist = math.sqrt(x_weight * (nx - ax) ** 2 + y_weight * (ny - ay) ** 2)
    in_box = _expanded_contains((cx, cy), worker.xyxy, margin=0.20)
    proximity = max(0.0, 1.0 - min(dist / 0.75, 1.0))
    return (0.65 if in_box else 0.0) + 0.35 * proximity + 0.05 * max(0.0, min(det.conf, 1.0))


def associate_ppe(workers: Iterable[Worker], ppe_detections: Iterable[Detection], min_score: float = 0.45) -> Dict[int, WorkerCompliance]:
    workers = list(workers)
    result = {w.track_id: WorkerCompliance(track_id=w.track_id) for w in workers}
    for det in ppe_detections:
        if det.cls not in PPE_CLASSES:
            continue
        scored = [(w, _association_score(w, det)) for w in workers]
        if not scored:
            continue
        best_worker, best_score = max(scored, key=lambda x: x[1])
        if best_score < min_score:
            continue
        result[best_worker.track_id].associated[det.cls].append(det)
        result[best_worker.track_id].present[det.cls] = True

    for item in result.values():
        item.missing = [c for c in PPE_CLASSES if not item.present[c]]
        item.compliant = len(item.missing) == 0
    return result


class PersistentViolationGate:
    """Suppresses one-frame compliance noise.

    `confirm_frames` is the number of consecutive violating observations required
    before an alert is emitted. `clear_frames` is the number of consecutive clear
    observations required to reset the state.
    """

    def __init__(self, confirm_frames: int = 3, clear_frames: int = 2):
        if confirm_frames < 1 or clear_frames < 1:
            raise ValueError("confirm_frames and clear_frames must be >= 1")
        self.confirm_frames = confirm_frames
        self.clear_frames = clear_frames
        self._state: Dict[int, Dict[str, int | bool | tuple]] = {}

    def update(self, track_id: int, missing: Sequence[str]) -> Optional[Dict[str, object]]:
        missing_tuple = tuple(sorted(missing))
        state = self._state.setdefault(track_id, {"bad": 0, "clear": 0, "alerted": False, "missing": ()})
        state["missing"] = missing_tuple
        if missing_tuple:
            state["bad"] = int(state["bad"]) + 1
            state["clear"] = 0
            if int(state["bad"]) >= self.confirm_frames and not bool(state["alerted"]):
                state["alerted"] = True
                return {"track_id": track_id, "event": "ppe_violation", "missing": list(missing_tuple), "confirmed": True}
            return None

        state["clear"] = int(state["clear"]) + 1
        state["bad"] = 0
        if int(state["clear"]) >= self.clear_frames:
            state["alerted"] = False
        return None
