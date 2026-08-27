"""
Runtime face-pipeline counters for the dashboard (quality gate + cross-camera).

Written by main.py / face_analyzer.py; read by dashboard.py.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_DIR = Path(__file__).resolve().parent
STATS_PATH = PROJECT_DIR / "live_preview" / "face_stats.json"
MAX_CROSS_EVENTS = 40

_lock = threading.Lock()

_DEFAULT: Dict[str, Any] = {
    "updated": 0.0,
    "total_scanned": 0,
    "accepted": 0,
    "rejected": 0,
    "by_reason": {},
    "anti_spoof": {
        "checked": 0,
        "rejected": 0,
        "score_sum": 0.0,
    },
    "cross_camera": [],
}


def _empty_stats() -> Dict[str, Any]:
    import copy

    return copy.deepcopy(_DEFAULT)


def _read_unlocked() -> Dict[str, Any]:
    if not STATS_PATH.is_file():
        return _empty_stats()
    try:
        data = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _empty_stats()
        out = _empty_stats()
        out.update({k: data[k] for k in out if k in data})
        if isinstance(data.get("by_reason"), dict):
            out["by_reason"] = dict(data["by_reason"])
        if isinstance(data.get("anti_spoof"), dict):
            out["anti_spoof"].update(data["anti_spoof"])
        if isinstance(data.get("cross_camera"), list):
            out["cross_camera"] = list(data["cross_camera"])
        return out
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return _empty_stats()


def _write_unlocked(payload: Dict[str, Any]) -> None:
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["updated"] = time.time()
    tmp = STATS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=0), encoding="utf-8")
    tmp.replace(STATS_PATH)


def read_face_stats() -> Dict[str, Any]:
    with _lock:
        data = _read_unlocked()
    anti = data.get("anti_spoof") or {}
    checked = int(anti.get("checked", 0))
    score_sum = float(anti.get("score_sum", 0.0))
    data["anti_spoof_avg"] = round(score_sum / checked, 3) if checked else None
    total = int(data.get("total_scanned", 0))
    accepted = int(data.get("accepted", 0))
    data["accept_rate"] = round(accepted / total, 3) if total else None
    return data


def record_face_detection(
    quality_ok: bool,
    reject_reason: Optional[str] = None,
    anti_spoof_score: Optional[float] = None,
    anti_spoof_checked: bool = False,
    anti_spoof_rejected: bool = False,
) -> None:
    with _lock:
        data = _read_unlocked()
        data["total_scanned"] = int(data.get("total_scanned", 0)) + 1
        if quality_ok:
            data["accepted"] = int(data.get("accepted", 0)) + 1
        else:
            data["rejected"] = int(data.get("rejected", 0)) + 1
            reason = "unknown"
            if reject_reason:
                reason = reject_reason.split("(", maxsplit=1)[0]
            by_reason: Dict[str, int] = dict(data.get("by_reason") or {})
            by_reason[reason] = by_reason.get(reason, 0) + 1
            data["by_reason"] = by_reason

        anti = dict(data.get("anti_spoof") or _DEFAULT["anti_spoof"])
        if anti_spoof_checked:
            anti["checked"] = int(anti.get("checked", 0)) + 1
            if anti_spoof_score is not None:
                anti["score_sum"] = float(anti.get("score_sum", 0.0)) + float(
                    anti_spoof_score
                )
            if anti_spoof_rejected:
                anti["rejected"] = int(anti.get("rejected", 0)) + 1
        data["anti_spoof"] = anti
        _write_unlocked(data)


def record_face_detections_batch(dets: list) -> None:
    for det in dets:
        record_face_detection(
            quality_ok=bool(getattr(det, "quality_ok", True)),
            reject_reason=getattr(det, "reject_reason", None),
            anti_spoof_score=getattr(det, "anti_spoof_score", None),
            anti_spoof_checked=getattr(det, "anti_spoof_score", None) is not None,
            anti_spoof_rejected=bool(
                getattr(det, "reject_reason", "") or ""
            ).startswith("spoof"),
        )


def record_cross_camera_match(
    track_id: str,
    camera_id: str,
    matched_track: str,
    matched_camera: str,
    score: float,
) -> None:
    with _lock:
        data = _read_unlocked()
        events: List[Dict[str, Any]] = list(data.get("cross_camera") or [])
        events.insert(
            0,
            {
                "track_id": str(track_id),
                "camera_id": str(camera_id),
                "matched_track": str(matched_track),
                "matched_camera": str(matched_camera),
                "score": round(float(score), 4),
                "ts": time.time(),
            },
        )
        data["cross_camera"] = events[:MAX_CROSS_EVENTS]
        _write_unlocked(data)
