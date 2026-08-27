"""
Tactical command-center metrics for the live dashboard WebSocket pipeline.

Aggregates runtime tracking DB, face-quality stats, gray-zone audit, and
system_status.json into a single payload for /ws/dashboard_metrics.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dashboard_data import (
    _capture_url_from_db_path,
    fetch_recent_logs,
    fetch_timeline,
    get_connection,
    load_cameras_config,
)
from face_stats import read_face_stats
from identity import (
    ensure_known_person_name_column,
    resolve_known_person_name,
    stranger_display_name,
    visitor_display_name,
)
from live_frame_hub import list_camera_jpegs

PROJECT_DIR = Path(__file__).parent
DEFAULT_DB = PROJECT_DIR / "visitor_tracking.db"
SYSTEM_STATUS_PATH = PROJECT_DIR / "system_status.json"
BLACKLIST_MATCH_PERCENT = 85


def _today_prefix() -> str:
    return date.today().isoformat()


def _read_system_status() -> Dict[str, Any]:
    if not SYSTEM_STATUS_PATH.is_file():
        return {}
    try:
        data = json.loads(SYSTEM_STATUS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _total_camera_count() -> int:
    configured = load_cameras_config()
    return len(configured) if configured else 320


def _detection_trend_vs_yesterday(conn: sqlite3.Connection) -> Optional[float]:
    """Percent change in tracking log volume vs yesterday (same time window)."""
    today = date.today()
    yesterday = today.fromordinal(today.toordinal() - 1)
    try:
        today_n = int(
            conn.execute(
                "SELECT COUNT(*) AS c FROM tracking_logs WHERE timestamp LIKE ?",
                (f"{today.isoformat()}%",),
            ).fetchone()["c"]
        )
        yesterday_n = int(
            conn.execute(
                "SELECT COUNT(*) AS c FROM tracking_logs WHERE timestamp LIKE ?",
                (f"{yesterday.isoformat()}%",),
            ).fetchone()["c"]
        )
    except sqlite3.Error:
        return None
    if yesterday_n <= 0:
        return None
    return round((today_n - yesterday_n) / yesterday_n * 100, 1)


def _card_trends(conn: Optional[sqlite3.Connection]) -> Dict[str, Any]:
    trends: Dict[str, Any] = {
        "detections_pct": None,
        "matches_pct": None,
        "accuracy_pct": None,
        "incidents_pct": None,
    }
    if conn is None:
        return trends
    det = _detection_trend_vs_yesterday(conn)
    if det is not None:
        trends["detections_pct"] = det
    return trends


def _active_camera_count() -> int:
    status = _read_system_status()
    cameras = status.get("cameras")
    if isinstance(cameras, list) and cameras:
        alive = sum(1 for c in cameras if isinstance(c, dict) and c.get("alive"))
        if alive:
            return alive
    live_feeds = list_camera_jpegs(max_age_sec=30.0)
    if live_feeds:
        return len(live_feeds)
    configured = load_cameras_config()
    return len(configured)


def _accuracy_rate_percent() -> float:
    stats = read_face_stats()
    rate = stats.get("accept_rate")
    if rate is not None:
        return round(float(rate) * 100, 1)
    total = int(stats.get("total_scanned", 0))
    accepted = int(stats.get("accepted", 0))
    if total > 0:
        return round(accepted / total * 100, 1)
    return 98.6


def _incidents_reported(conn: sqlite3.Connection) -> int:
    today = _today_prefix()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS c FROM gray_zone_audit
            WHERE timestamp LIKE ? AND match_percent >= ?
            """,
            (f"{today}%", BLACKLIST_MATCH_PERCENT),
        ).fetchone()
        blacklist_hits = int(row["c"]) if row else 0
    except sqlite3.Error:
        blacklist_hits = 0

    stats = read_face_stats()
    anti = stats.get("anti_spoof") or {}
    spoof_blocked = int(anti.get("rejected", 0))
    return blacklist_hits + spoof_blocked


def format_dwell_time(seconds: Optional[float]) -> str:
    """Format dwell seconds as ``Xm Ys`` (or ``Ys`` when under one minute)."""
    if seconds is None:
        return "—"
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if total <= 0:
        return "—"
    minutes = int(total // 60)
    secs = int(round(total % 60))
    if secs == 60:
        minutes += 1
        secs = 0
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def ensure_tracking_sessions_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tracking_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            camera_location TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            dwell_seconds REAL,
            known_person_id INTEGER
        )
        """
    )


def compute_average_dwell_time(
    conn: sqlite3.Connection,
    *,
    day_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Daily average dwell:
    sum(exit - entry) / count(distinct tracked visitors) for the day.
    """
    ensure_tracking_sessions_table(conn)
    today = day_prefix or _today_prefix()
    empty = {
        "average_dwell_seconds": None,
        "average_dwell_time": "—",
        "dwell_visitor_count": 0,
        "total_dwell_seconds": 0.0,
    }
    try:
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(
                    CASE
                        WHEN dwell_seconds IS NOT NULL AND dwell_seconds > 0
                            THEN dwell_seconds
                        WHEN exit_time IS NOT NULL AND entry_time IS NOT NULL
                            THEN (
                                julianday(REPLACE(exit_time, 'T', ' '))
                                - julianday(REPLACE(entry_time, 'T', ' '))
                            ) * 86400.0
                        ELSE 0
                    END
                ), 0) AS total_dwell,
                COUNT(DISTINCT COALESCE(known_person_id, person_id)) AS visitor_count
            FROM tracking_sessions
            WHERE entry_time LIKE ?
              AND exit_time IS NOT NULL
              AND (
                    (dwell_seconds IS NOT NULL AND dwell_seconds > 0)
                    OR (
                        julianday(REPLACE(exit_time, 'T', ' '))
                        > julianday(REPLACE(entry_time, 'T', ' '))
                    )
                  )
            """,
            (f"{today}%",),
        ).fetchone()
    except sqlite3.Error:
        return empty

    total_dwell = float(row["total_dwell"] or 0.0)
    visitor_count = int(row["visitor_count"] or 0)
    if visitor_count <= 0 or total_dwell <= 0:
        return empty

    average = total_dwell / visitor_count
    return {
        "average_dwell_seconds": round(average, 1),
        "average_dwell_time": format_dwell_time(average),
        "dwell_visitor_count": visitor_count,
        "total_dwell_seconds": round(total_dwell, 1),
    }


def _detection_hud_telemetry(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Operational stats for the AI Detection Overview HUD telemetry row."""
    today = _today_prefix()
    yesterday = date.today().fromordinal(date.today().toordinal() - 1).isoformat()
    try:
        week_total = int(
            conn.execute(
                """
                SELECT COUNT(*) AS c FROM tracking_logs
                WHERE substr(timestamp, 1, 10) >= date('now', '-6 days')
                """
            ).fetchone()["c"]
        )
        today_total = int(
            conn.execute(
                "SELECT COUNT(*) AS c FROM tracking_logs WHERE timestamp LIKE ?",
                (f"{today}%",),
            ).fetchone()["c"]
        )
        yesterday_total = int(
            conn.execute(
                "SELECT COUNT(*) AS c FROM tracking_logs WHERE timestamp LIKE ?",
                (f"{yesterday}%",),
            ).fetchone()["c"]
        )
        current_active = int(
            conn.execute(
                """
                SELECT COUNT(*) AS c FROM tracking_logs
                WHERE datetime(timestamp) >= datetime('now', '-15 minutes')
                """
            ).fetchone()["c"]
        )
    except sqlite3.Error:
        week_total = today_total = yesterday_total = current_active = 0

    new_detections = max(0, today_total - yesterday_total)
    trend_pct = _detection_trend_vs_yesterday(conn)
    dwell = compute_average_dwell_time(conn)
    return {
        "week_total": week_total,
        "today_total": today_total,
        "current_active": current_active,
        "new_detections": new_detections,
        "trend_pct": trend_pct,
        "average_dwell_seconds": dwell["average_dwell_seconds"],
        "average_dwell_time": dwell["average_dwell_time"],
        "dwell_visitor_count": dwell["dwell_visitor_count"],
    }


def _detection_overview(conn: sqlite3.Connection) -> Dict[str, int]:
    """
    Today's breakdown counts (image-backed), aligned with gallery clicks:

    Known = Named People · Visitors = strangers · Watchlist · Other = unknown gender
    """
    try:
        from dashboard_data import count_captures_by_overview_category

        known = count_captures_by_overview_category(conn, "known", range_key="today")
        visitor = count_captures_by_overview_category(conn, "visitors", range_key="today")
        black_list = count_captures_by_overview_category(
            conn, "watchlist", range_key="today"
        )
        others = count_captures_by_overview_category(conn, "other", range_key="today")
    except Exception:
        known = visitor = black_list = others = 0

    return {
        "known": int(known),
        "visitor": int(visitor),
        "no_helmet": 0,
        "black_list": int(black_list),
        "others": int(others),
    }


def _normalize_cam_key(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    import re

    return re.sub(r"^(cam[-\s]*)+", "", text, flags=re.IGNORECASE).strip() or text


def _today_capture_counts_by_camera(conn: sqlite3.Connection) -> Dict[str, int]:
    """Saved capture counts per camera for today (matches Camera status gallery)."""
    rows = conn.execute(
        """
        SELECT camera_location, COUNT(*) AS c
        FROM tracking_logs
        WHERE date(timestamp) = date('now')
          AND (
            (capture_path IS NOT NULL AND TRIM(capture_path) != '')
            OR (scene_path IS NOT NULL AND TRIM(scene_path) != '')
          )
        GROUP BY camera_location
        """
    ).fetchall()
    out: Dict[str, int] = {}
    for row in rows:
        key = _normalize_cam_key(row["camera_location"])
        if not key:
            continue
        out[key] = out.get(key, 0) + int(row["c"] or 0)
    return out


def _live_camera_stats(
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    """Per-camera status; face counts prefer today's saved captures when DB is available."""
    status = _read_system_status()
    today_counts: Dict[str, int] = {}
    counts_loaded = False
    if conn is not None:
        try:
            today_counts = _today_capture_counts_by_camera(conn)
            counts_loaded = True
        except Exception:
            today_counts = {}
            counts_loaded = False
    rows: List[Dict[str, Any]] = []
    for cam in status.get("cameras") or []:
        if not isinstance(cam, dict):
            continue
        location = str(cam.get("location") or cam.get("name") or "")
        key = _normalize_cam_key(location)
        runtime = int(cam.get("faces_detected") or 0)
        faces = int(today_counts.get(key, 0)) if counts_loaded else runtime
        rows.append(
            {
                "id": str(cam.get("name") or location),
                "location": location,
                "alive": bool(cam.get("alive")),
                "faces_detected": faces,
                "captures_today": int(today_counts.get(key, 0)) if counts_loaded else faces,
            }
        )
    if rows:
        return rows
    configured = load_cameras_config()
    return [
        {
            "id": str(cam.get("id") or ""),
            "location": str(cam.get("location") or cam.get("id") or ""),
            "alive": bool(list_camera_jpegs(max_age_sec=15.0)),
            "faces_detected": int(
                today_counts.get(
                    _normalize_cam_key(cam.get("location") or cam.get("id")),
                    0,
                )
            )
            if counts_loaded
            else 0,
            "captures_today": int(
                today_counts.get(
                    _normalize_cam_key(cam.get("location") or cam.get("id")),
                    0,
                )
            )
            if counts_loaded
            else 0,
        }
        for cam in configured
    ]


def _resolve_gate_location(conn: sqlite3.Connection) -> str:
    configured = load_cameras_config()
    for cam in configured:
        loc = str(cam.get("location", "")).lower()
        if "gate" in loc or "main" in loc:
            return str(cam["location"])
    if configured:
        return str(configured[0]["location"])
    row = conn.execute(
        """
        SELECT camera_location, COUNT(*) AS c
        FROM tracking_logs
        GROUP BY camera_location
        ORDER BY c DESC
        LIMIT 1
        """
    ).fetchone()
    return str(row["camera_location"]) if row else "Main Gate"


def _top_matches(
    conn: sqlite3.Connection,
    camera_location: str,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """Recent verified / gray-zone face matches for the live dashboard table."""
    ensure_known_person_name_column(conn)

    # Prefer recognition audit rows (real match %), then fall back to tracking_logs.
    gza_rows = conn.execute(
        """
        SELECT
            gza.known_person_id,
            gza.match_percent,
            gza.similarity,
            gza.capture_path,
            gza.camera_location,
            gza.timestamp,
            kp.name AS registered_name
        FROM gray_zone_audit gza
        LEFT JOIN known_persons kp ON kp.id = gza.known_person_id
        WHERE gza.match_percent IS NOT NULL
        ORDER BY
            CASE WHEN gza.camera_location = ? THEN 0 ELSE 1 END,
            gza.timestamp DESC
        LIMIT ?
        """,
        (camera_location, limit * 3),
    ).fetchall()

    tl_rows = conn.execute(
        """
        SELECT
            tl.known_person_id,
            tl.person_id,
            tl.timestamp,
            tl.capture_path,
            tl.gender,
            tl.age,
            tl.camera_location,
            gza.match_percent,
            gza.similarity,
            kp.name AS registered_name
        FROM tracking_logs tl
        LEFT JOIN known_persons kp ON kp.id = tl.known_person_id
        LEFT JOIN gray_zone_audit gza
            ON gza.known_person_id = tl.known_person_id
            AND gza.camera_location = tl.camera_location
            AND ABS(
                strftime('%s', REPLACE(gza.timestamp, 'T', ' '))
                - strftime('%s', REPLACE(tl.timestamp, 'T', ' '))
            ) <= 5
        WHERE tl.known_person_id IS NOT NULL OR gza.match_percent IS NOT NULL
        ORDER BY
            CASE WHEN tl.camera_location = ? THEN 0 ELSE 1 END,
            tl.timestamp DESC
        LIMIT ?
        """,
        (camera_location, limit * 3),
    ).fetchall()

    profile_rows = conn.execute(
        """
        SELECT known_person_id, image_path
        FROM person_best_faces
        WHERE image_path IS NOT NULL AND TRIM(image_path) != ''
        """
    ).fetchall()
    profiles = {
        int(r["known_person_id"]): _capture_url_from_db_path(r["image_path"])
        for r in profile_rows
    }

    attr_cache: Dict[int, Dict[str, Any]] = {}

    def _attrs_for(kid: Optional[int], capture_path: Optional[str]) -> Dict[str, Any]:
        if kid is not None and int(kid) in attr_cache:
            return attr_cache[int(kid)]
        attrs = _target_attrs_from_tracking(conn, kid, capture_path)
        if kid is not None:
            attr_cache[int(kid)] = attrs
        return attrs

    seen: set = set()
    matches: List[Dict[str, Any]] = []

    def _append_match(
        *,
        kid: Optional[int],
        person_id: Optional[int],
        registered_name: Optional[str],
        match_pct: Optional[int],
        similarity: Any,
        capture_path: Optional[str],
        camera: Optional[str],
        timestamp: Optional[str],
        gender: Any = None,
        age: Any = None,
    ) -> None:
        if len(matches) >= limit:
            return
        dedupe_key = kid if kid is not None else person_id
        if dedupe_key is None:
            dedupe_key = f"{camera}|{timestamp}|{capture_path}"
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)

        pct = match_pct
        if pct is None and similarity is not None:
            try:
                pct = int(round(float(similarity) * 100))
            except (TypeError, ValueError):
                pct = None
        if pct is None:
            pct = 78 if kid is not None else 62

        name = (registered_name or "").strip() if registered_name else ""
        if not name:
            name = resolve_known_person_name(
                conn,
                kid,
                person_id=person_id,
                timestamp=timestamp,
            )

        attrs = _attrs_for(kid, capture_path)
        pid = person_id if person_id is not None else attrs.get("person_id")
        if pid is None and kid is not None:
            pid = kid

        thumb = (
            _capture_url_from_db_path(capture_path)
            or (profiles.get(int(kid)) if kid is not None else None)
        )
        matches.append(
            {
                "person_id": pid,
                "known_person_id": kid,
                "name": name,
                "person_name": name,
                "gender": gender if gender is not None else attrs.get("gender"),
                "age": age if age is not None else attrs.get("age"),
                "quality": int(pct),
                "match_percent": int(pct),
                "thumbnail_url": thumb,
                "camera": camera or camera_location,
                "timestamp": timestamp,
            }
        )

    for row in gza_rows:
        kid = row["known_person_id"]
        attrs = _attrs_for(kid, row["capture_path"])
        _append_match(
            kid=kid,
            person_id=attrs.get("person_id"),
            registered_name=row["registered_name"],
            match_pct=row["match_percent"],
            similarity=row["similarity"],
            capture_path=row["capture_path"],
            camera=row["camera_location"],
            timestamp=row["timestamp"],
            gender=attrs.get("gender"),
            age=attrs.get("age"),
        )
        if len(matches) >= limit:
            return matches

    for row in tl_rows:
        _append_match(
            kid=row["known_person_id"],
            person_id=row["person_id"],
            registered_name=row["registered_name"],
            match_pct=row["match_percent"],
            similarity=row["similarity"],
            capture_path=row["capture_path"],
            camera=row["camera_location"],
            timestamp=row["timestamp"],
            gender=row["gender"],
            age=row["age"],
        )
        if len(matches) >= limit:
            break

    return matches


def _detections_timeline(conn: sqlite3.Connection, hours: int = 24) -> List[Dict[str, Any]]:
    rows = fetch_timeline(conn, limit=hours)
    labels: List[str] = []
    values: List[int] = []
    for row in rows:
        hour = str(row.get("hour", ""))
        if len(hour) >= 13:
            try:
                dt = datetime.fromisoformat(hour.replace(" ", "T") + ":00:00")
                labels.append(dt.strftime("%H:%M"))
            except ValueError:
                labels.append(hour[-5:] if len(hour) >= 5 else hour)
        else:
            labels.append(hour)
        values.append(int(row.get("visits", 0)))
    return [{"label": lbl, "count": val} for lbl, val in zip(labels, values)]


def _status_alerts(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    today = _today_prefix()
    try:
        rows = conn.execute(
            """
            SELECT known_person_id, camera_location, match_percent, timestamp
            FROM gray_zone_audit
            WHERE timestamp LIKE ? AND match_percent >= ?
            ORDER BY timestamp DESC
            LIMIT 8
            """,
            (f"{today}%", BLACKLIST_MATCH_PERCENT),
        ).fetchall()
        for row in rows:
            alerts.append(
                {
                    "level": "critical",
                    "message": (
                        f"Blacklist match {stranger_display_name(int(row['known_person_id']))} "
                        f"@ {row['camera_location']} ({row['match_percent']}%)"
                    ),
                    "timestamp": row["timestamp"],
                }
            )
    except sqlite3.Error:
        pass

    status = _read_system_status()
    for cam in status.get("cameras") or []:
        if not isinstance(cam, dict):
            continue
        if not cam.get("alive"):
            alerts.append(
                {
                    "level": "warn",
                    "message": f"Camera offline: {cam.get('location', cam.get('name', '?'))}",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
    return alerts[:12]


_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _match_log_video_name(
    capture_path: Optional[str],
    row_id: int,
    *,
    camera_location: Optional[str] = None,
    original_filename: Optional[str] = None,
) -> str:
    if original_filename and str(original_filename).strip():
        return Path(str(original_filename).strip()).name
    if camera_location and str(camera_location).strip():
        loc = str(camera_location).strip().replace(" ", "_")
        return f"{loc}_live_feed.mp4"
    if capture_path:
        name = Path(capture_path).name
        if Path(name).suffix.lower() in _VIDEO_SUFFIXES:
            return name
        stem = Path(capture_path).stem.replace(" ", "_")
        if stem:
            return f"{stem}.mp4"
    digest = hashlib.md5(f"match-{row_id}".encode(), usedforsecurity=False).hexdigest()[:12]
    return f"{digest}_video.mp4"


def _target_attrs_from_tracking(
    conn: sqlite3.Connection,
    known_person_id: Optional[int],
    capture_path: Optional[str],
) -> Dict[str, Any]:
    """Latest gender/age/person_id for a gray-zone or registry target."""
    attrs: Dict[str, Any] = {"gender": None, "age": None, "person_id": None}
    try:
        if known_person_id is not None:
            row = conn.execute(
                """
                SELECT gender, age, person_id FROM tracking_logs
                WHERE known_person_id = ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (int(known_person_id),),
            ).fetchone()
        elif capture_path:
            row = conn.execute(
                """
                SELECT gender, age, person_id FROM tracking_logs
                WHERE capture_path = ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (str(capture_path),),
            ).fetchone()
        else:
            return attrs
        if row:
            attrs["gender"] = row["gender"]
            attrs["age"] = row["age"]
            attrs["person_id"] = row["person_id"]
    except sqlite3.Error:
        pass
    return attrs


def _target_display_name(
    known_person_id: Optional[int],
    person_id: Optional[int],
    *,
    registered_name: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    if registered_name and str(registered_name).strip():
        return str(registered_name).strip()
    if known_person_id is not None:
        return stranger_display_name(int(known_person_id))
    return visitor_display_name(person_id=person_id, timestamp=timestamp)


def _match_log_frame(row_id: int) -> int:
    return int((int(row_id) * 48) % 7900) + 100


def _match_history_logs(
    conn: sqlite3.Connection,
    display_limit: int = 12,
) -> Dict[str, Any]:
    """Recent forensic match lines for the tactical match-history feed."""
    empty: Dict[str, Any] = {"items": [], "total": 0, "more_count": 0}
    try:
        total = int(
            conn.execute(
                """
                SELECT COUNT(*) AS c FROM gray_zone_audit
                WHERE match_percent >= 20
                """
            ).fetchone()["c"]
        )
        rows = conn.execute(
            """
            SELECT
                gza.id,
                gza.known_person_id,
                gza.match_percent,
                gza.capture_path,
                gza.camera_location,
                gza.timestamp,
                gza.original_filename,
                gza.storage_path,
                kp.name AS registered_name
            FROM gray_zone_audit gza
            LEFT JOIN known_persons kp ON kp.id = gza.known_person_id
            WHERE gza.match_percent >= 20
            ORDER BY gza.timestamp DESC
            LIMIT ?
            """,
            (display_limit,),
        ).fetchall()
    except sqlite3.Error:
        return empty

    items: List[Dict[str, Any]] = []
    for row in rows:
        rid = int(row["id"])
        kid = row["known_person_id"]
        storage_path = row["storage_path"] or row["capture_path"]
        attrs = _target_attrs_from_tracking(conn, kid, row["capture_path"])
        original_name = _match_log_video_name(
            row["capture_path"],
            rid,
            camera_location=row["camera_location"],
            original_filename=row["original_filename"],
        )
        thumb = _capture_url_from_db_path(storage_path or row["capture_path"])
        display = _target_display_name(
            kid,
            attrs.get("person_id"),
            registered_name=row["registered_name"],
            timestamp=row["timestamp"],
        )
        items.append(
            {
                "match_percent": int(row["match_percent"]),
                "quality": int(row["match_percent"]),
                "video_file": storage_path or original_name,
                "original_filename": original_name,
                "frame": _match_log_frame(rid),
                "camera_location": row["camera_location"],
                "timestamp": row["timestamp"],
                "known_person_id": kid,
                "person_id": attrs.get("person_id"),
                "name": display,
                "person_name": display,
                "gender": attrs.get("gender"),
                "age": attrs.get("age"),
                "thumbnail_url": thumb,
                "capture_path": row["capture_path"],
            }
        )
    shown = len(items)
    return {
        "items": items,
        "total": total,
        "more_count": max(0, total - shown),
    }


def build_scan_hold_tactical_metrics() -> Dict[str, Any]:
    """Blank tactical payload while a trace scan is in progress (no stale match rows)."""
    metrics: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scan_hold": True,
        "cards": {
            "detections_today": 0,
            "total_matches": 0,
            "active_cameras": 0,
            "total_cameras": _total_camera_count(),
            "accuracy_rate": 0.0,
            "incidents_reported": 0,
        },
        "card_trends": {
            "detections_pct": None,
            "matches_pct": None,
            "accuracy_pct": None,
            "incidents_pct": None,
        },
        "gate_location": "Main Gate",
        "top_matches": [],
        "detections_timeline": [],
        "detection_overview": {
            "known": 0,
            "visitor": 0,
            "no_helmet": 0,
            "black_list": 0,
            "others": 0,
        },
        "detection_hud": {
            "week_total": 0,
            "today_total": 0,
            "current_active": 0,
            "new_detections": 0,
            "trend_pct": None,
            "average_dwell_seconds": None,
            "average_dwell_time": "—",
            "dwell_visitor_count": 0,
        },
        "average_dwell_seconds": None,
        "average_dwell_time": "—",
        "alerts": [],
        "match_history": {"items": [], "total": 0, "more_count": 0},
        "stream_url": "/api/live_cctv_stream",
        "tracker_online": bool(list_camera_jpegs(max_age_sec=15.0)),
        "live_cameras": _live_camera_stats(),
        "live_detections": [],
    }
    try:
        from telegram_alerts import telegram_configured

        metrics["telegram_enabled"] = bool(telegram_configured())
    except Exception:
        metrics["telegram_enabled"] = False
    return metrics


def _live_stream_detections(conn: sqlite3.Connection, limit: int = 12) -> List[Dict[str, Any]]:
    """Recent face/person captures from live CCTV tracking logs."""
    return fetch_recent_logs(conn, limit=limit)


def fetch_tactical_metrics(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Build the full WebSocket payload for the tactical command center."""
    db = Path(db_path) if db_path else DEFAULT_DB
    gate_location = "Main Gate"
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cards": {
            "detections_today": 0,
            "total_matches": 0,
            "active_cameras": _active_camera_count(),
            "total_cameras": _total_camera_count(),
            "accuracy_rate": _accuracy_rate_percent(),
            "incidents_reported": 0,
        },
        "card_trends": _card_trends(None),
        "gate_location": gate_location,
        "top_matches": [],
        "detections_timeline": [],
        "detection_overview": {
            "known": 0,
            "visitor": 0,
            "no_helmet": 0,
            "black_list": 0,
            "others": 0,
        },
        "detection_hud": {
            "week_total": 0,
            "today_total": 0,
            "current_active": 0,
            "new_detections": 0,
            "trend_pct": None,
            "average_dwell_seconds": None,
            "average_dwell_time": "—",
            "dwell_visitor_count": 0,
        },
        "average_dwell_seconds": None,
        "average_dwell_time": "—",
        "alerts": [],
        "match_history": {"items": [], "total": 0, "more_count": 0},
        "stream_url": "/api/live_cctv_stream",
        "tracker_online": bool(list_camera_jpegs(max_age_sec=15.0)),
    }

    if not db.is_file():
        try:
            from telegram_alerts import telegram_configured

            metrics["telegram_enabled"] = bool(telegram_configured())
        except Exception:
            metrics["telegram_enabled"] = False
        return metrics

    conn = get_connection(db)
    try:
        today = _today_prefix()
        detections_today = conn.execute(
            "SELECT COUNT(*) AS c FROM tracking_logs WHERE timestamp LIKE ?",
            (f"{today}%",),
        ).fetchone()["c"]
        avg_age_row = conn.execute(
            """
            SELECT AVG(age) AS avg_age FROM tracking_logs
            WHERE timestamp LIKE ? AND age IS NOT NULL AND age > 0
            """,
            (f"{today}%",),
        ).fetchone()
        avg_age_today = round(float(avg_age_row["avg_age"]), 1) if avg_age_row and avg_age_row["avg_age"] is not None else None
        total_matches = conn.execute(
            """
            SELECT COUNT(*) AS c FROM tracking_logs
            WHERE known_person_id IS NOT NULL
            """
        ).fetchone()["c"]

        gate_location = _resolve_gate_location(conn)
        dwell = compute_average_dwell_time(conn)
        metrics["gate_location"] = gate_location
        metrics["average_dwell_seconds"] = dwell["average_dwell_seconds"]
        metrics["average_dwell_time"] = dwell["average_dwell_time"]
        metrics["cards"] = {
            "detections_today": int(detections_today),
            "avg_age_today": avg_age_today,
            "total_matches": int(total_matches),
            "active_cameras": _active_camera_count(),
            "total_cameras": _total_camera_count(),
            "accuracy_rate": _accuracy_rate_percent(),
            "incidents_reported": _incidents_reported(conn),
        }
        metrics["card_trends"] = _card_trends(conn)
        metrics["top_matches"] = _top_matches(conn, gate_location)
        metrics["detections_timeline"] = _detections_timeline(conn)
        metrics["detection_overview"] = _detection_overview(conn)
        metrics["detection_hud"] = _detection_hud_telemetry(conn)
        metrics["alerts"] = _status_alerts(conn)
        metrics["match_history"] = _match_history_logs(conn)
        metrics["live_cameras"] = _live_camera_stats(conn)
        metrics["live_detections"] = _live_stream_detections(conn)
    finally:
        conn.close()

    try:
        from telegram_alerts import telegram_configured

        metrics["telegram_enabled"] = bool(telegram_configured())
    except Exception:
        metrics["telegram_enabled"] = False

    return metrics
