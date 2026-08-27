"""
Dashboard queries for SharedFaceGallery and cross-camera journeys.
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from face_analyzer import DEFAULT_GALLERY_DB, _unpack_embedding_history
from face_stats import read_face_stats

PROJECT_DIR = Path(__file__).parent


def _gallery_conn(path: Path) -> Optional[sqlite3.Connection]:
    if not path.is_file():
        return None
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_live_face_gallery(
    gallery_path: Path = DEFAULT_GALLERY_DB,
    limit: int = 48,
) -> Dict[str, Any]:
    conn = _gallery_conn(gallery_path)
    if conn is None:
        return {"entries": [], "total": 0, "cameras": []}

    try:
        rows = conn.execute(
            """
            SELECT track_id, camera_id, embedding, updated_at
            FROM gallery
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) AS c FROM gallery").fetchone()["c"]
    finally:
        conn.close()

    now = time.time()
    entries: List[Dict[str, Any]] = []
    cameras: set = set()
    for row in rows:
        blob = row["embedding"]
        history = _unpack_embedding_history(blob) if blob else []
        if not history and blob:
            vec = __import__("numpy").frombuffer(blob, dtype="float32")
            if vec.size:
                history = [vec]
        cameras.add(row["camera_id"])
        age_sec = max(0.0, now - float(row["updated_at"]))
        entries.append(
            {
                "track_id": row["track_id"],
                "camera_id": row["camera_id"],
                "embedding_count": len(history),
                "updated_at": float(row["updated_at"]),
                "age_sec": round(age_sec, 1),
                "fresh": age_sec < 300,
            }
        )

    return {
        "entries": entries,
        "total": int(total),
        "cameras": sorted(cameras),
    }


def fetch_cross_camera_journeys(
    conn: sqlite3.Connection,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Build per-visitor camera-hop timelines from tracking_logs."""
    rows = conn.execute(
        """
        SELECT
            COALESCE(known_person_id, person_id) AS person_id,
            camera_location,
            timestamp
        FROM tracking_logs
        ORDER BY timestamp ASC
        """
    ).fetchall()

    by_person: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        pid = row["person_id"]
        if pid is None:
            continue
        by_person[int(pid)].append(
            {
                "camera": row["camera_location"],
                "at": row["timestamp"],
            }
        )

    journeys: List[Dict[str, Any]] = []
    for pid, events in by_person.items():
        cameras_seen = {e["camera"] for e in events}
        if len(cameras_seen) < 2:
            continue

        steps: List[Dict[str, str]] = []
        prev_cam: Optional[str] = None
        for event in events:
            cam = event["camera"]
            if cam != prev_cam:
                steps.append({"camera": cam, "at": event["at"]})
                prev_cam = cam

        if len({s["camera"] for s in steps}) < 2:
            continue

        journeys.append(
            {
                "person_id": pid,
                "steps": steps,
                "camera_count": len(cameras_seen),
                "hop_count": len(steps),
                "started_at": steps[0]["at"],
                "last_at": steps[-1]["at"],
            }
        )

    journeys.sort(key=lambda j: j["last_at"], reverse=True)
    return journeys[:limit]


def fetch_face_intel_bundle(
    tracking_conn: sqlite3.Connection,
    gallery_path: Path = DEFAULT_GALLERY_DB,
) -> Dict[str, Any]:
    gallery = fetch_live_face_gallery(gallery_path)
    journeys = fetch_cross_camera_journeys(tracking_conn)
    stats = read_face_stats()
    return {
        "gallery": gallery,
        "journeys": journeys,
        "quality": stats,
        "cross_camera_live": stats.get("cross_camera") or [],
    }


def fetch_extended_metrics(
    tracking_conn: Optional[sqlite3.Connection] = None,
    gallery_path: Path = DEFAULT_GALLERY_DB,
    journey_limit: int = 20,
    target_limit: int = 48,
) -> Dict[str, Any]:
    """
    Unified payload for dashboard extended-metrics JS (Watchlist / Journey / Radar).
    """
    gallery = fetch_live_face_gallery(gallery_path, limit=target_limit)
    stats = read_face_stats()
    anti = stats.get("anti_spoof") or {}

    targets: List[Dict[str, Any]] = []
    for entry in gallery.get("entries") or []:
        targets.append(
            {
                "track_id": entry["track_id"],
                "camera_id": entry["camera_id"],
                "identity_token": f"track_{entry['track_id']}_{entry['camera_id']}",
                "embedding_count": entry.get("embedding_count", 1),
                "updated_at": entry.get("updated_at"),
                "age_sec": entry.get("age_sec"),
                "live": bool(entry.get("fresh")),
            }
        )

    journeys_raw: List[Dict[str, Any]] = []
    if tracking_conn is not None:
        journeys_raw = fetch_cross_camera_journeys(tracking_conn, limit=journey_limit)

    journeys: List[Dict[str, Any]] = []
    for j in journeys_raw:
        token = f"person_{j['person_id']}"
        journeys.append(
            {
                "identity_token": token,
                "person_id": j["person_id"],
                "camera_count": j.get("camera_count", 0),
                "hop_count": j.get("hop_count", 0),
                "hops": j.get("steps") or [],
                "started_at": j.get("started_at"),
                "last_at": j.get("last_at"),
            }
        )

    spoof_blocked = int(anti.get("rejected", 0))
    liveness_score = stats.get("anti_spoof_avg")

    return {
        "targets": targets,
        "target_total": int(gallery.get("total", len(targets))),
        "cameras": gallery.get("cameras") or [],
        "journeys": journeys,
        "cross_camera_live": stats.get("cross_camera") or [],
        "spoof_blocked": spoof_blocked,
        "liveness_score": liveness_score,
        "quality": stats,
        "updated": stats.get("updated"),
    }
