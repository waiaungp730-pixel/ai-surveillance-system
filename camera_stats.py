"""Per-camera runtime counters shared between main.py and run_system.py."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_DIR = Path(__file__).resolve().parent
STATS_DIR = PROJECT_DIR / "live_preview" / "cam_stats"


def _safe_key(location: str) -> str:
    safe = re.sub(r"[^\w\-]+", "_", str(location).strip())
    return safe[:64] if safe else "cam"


def write_faces_detected(location: str, count: int) -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATS_DIR / f"{_safe_key(location)}.json"
    tmp = path.with_suffix(".json.tmp")
    payload = {"faces_detected": int(count), "updated": time.time()}
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def read_faces_detected(location: str) -> int:
    path = STATS_DIR / f"{_safe_key(location)}.json"
    if not path.is_file():
        return 0
    try:
        data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("faces_detected", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0
