"""
Database queries for the visitor tracking dashboard.
"""

import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from identity import init_identity_db
from captures import (
    CAPTURES_DIR,
    capture_display_is_body_crop,
    capture_display_is_full_frame,
    capture_display_is_person_box,
    ensure_capture_column,
)
from face_attributes import ensure_attribute_columns

PROJECT_DIR = Path(__file__).parent
DEFAULT_DB = PROJECT_DIR / "visitor_tracking.db"
CAMERAS_CONFIG_PATH = PROJECT_DIR / "cameras.json"


def load_configured_cameras() -> List[str]:
    return [row["location"] for row in load_cameras_config()]


def load_cameras_config() -> List[Dict[str, str]]:
    if not CAMERAS_CONFIG_PATH.is_file():
        return []
    try:
        rows = json.loads(CAMERAS_CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            return []
        cameras: List[Dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("location"):
                continue
            cameras.append(
                {
                    "id": str(row.get("id", "")),
                    "group": str(row.get("group", "")),
                    "location": str(row["location"]),
                }
            )
        return cameras
    except (json.JSONDecodeError, KeyError, TypeError, OSError):
        return []


def get_connection(db_path: Path) -> sqlite3.Connection:
    # Busy timeout + WAL so dashboard enroll and live trackers can share the DB.
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
        pass
    init_identity_db(conn)
    ensure_capture_column(conn)
    ensure_attribute_columns(conn)
    return conn


def uses_known_ids(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM tracking_logs WHERE known_person_id IS NOT NULL"
    ).fetchone()
    return row["c"] > 0


def _person_filter_sql(
    conn: sqlite3.Connection, person_id: Optional[int]
) -> Tuple[str, list]:
    """Return SQL AND-clause + params for optional person filter (use after WHERE 1=1)."""
    if person_id is None:
        return "", []
    known = uses_known_ids(conn)
    id_col = "known_person_id" if known else "person_id"
    return f" AND {id_col} = ?", [person_id]


def fetch_summary(
    conn: sqlite3.Connection, person_id: Optional[int] = None
) -> Dict[str, Any]:
    known = uses_known_ids(conn)
    id_col = "known_person_id" if known else "person_id"
    person_clause, person_params = _person_filter_sql(conn, person_id)

    total_visits = conn.execute(
        f"SELECT COUNT(*) AS c FROM tracking_logs WHERE 1=1{person_clause}",
        person_params,
    ).fetchone()["c"]
    unique_people = conn.execute(
        f"""
        SELECT COUNT(DISTINCT {id_col}) AS c FROM tracking_logs WHERE 1=1{person_clause}
        """,
        person_params,
    ).fetchone()["c"]
    cameras = conn.execute(
        f"""
        SELECT COUNT(DISTINCT camera_location) AS c FROM tracking_logs WHERE 1=1{person_clause}
        """,
        person_params,
    ).fetchone()["c"]
    last_seen = conn.execute(
        f"SELECT MAX(timestamp) AS t FROM tracking_logs WHERE 1=1{person_clause}",
        person_params,
    ).fetchone()["t"]

    registered = conn.execute("SELECT COUNT(*) AS c FROM known_persons").fetchone()["c"]
    configured = load_configured_cameras()
    camera_count = len(configured) if configured else 0
    zones_reporting = cameras

    return {
        "total_visits": total_visits,
        "unique_people": unique_people,
        "cameras": zones_reporting,
        "camera_count": camera_count,
        "cameras_configured": camera_count,
        "configured_locations": configured,
        "last_seen": last_seen,
        "registered_profiles": registered,
        "id_label": "Person ID" if known else "Track ID",
        "filtered_person_id": person_id,
    }


def format_gender(value: Optional[str]) -> str:
    """Dashboard display string for gender."""
    if value is None:
        return "Unknown"
    text = str(value).strip()
    return text if text else "Unknown"


def format_age(value: Optional[Any]) -> str:
    """Dashboard display string for estimated age."""
    if value is None:
        return "Unknown"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "Unknown"


def fetch_latest_attributes_by_person(
    conn: sqlite3.Connection, id_col: str
) -> Dict[int, Dict[str, str]]:
    """
    Latest gender/age per visitor, preferring rows that have attribute data.
    """
    rows = conn.execute(
        f"""
        SELECT person_id, gender, age
        FROM (
            SELECT
                {id_col} AS person_id,
                gender,
                age,
                ROW_NUMBER() OVER (
                    PARTITION BY {id_col}
                    ORDER BY
                        CASE
                            WHEN gender IS NOT NULL AND TRIM(gender) != '' THEN 0
                            ELSE 1
                        END,
                        timestamp DESC
                ) AS rn
            FROM tracking_logs
            WHERE {id_col} IS NOT NULL
        )
        WHERE rn = 1
        """
    ).fetchall()
    return {
        row["person_id"]: {
            "gender": format_gender(row["gender"]),
            "age": format_age(row["age"]),
        }
        for row in rows
    }


def _capture_url_from_db_path(path_value: Optional[str]) -> Optional[str]:
    """Convert DB capture/image path to dashboard /captures URL when file exists."""
    if not path_value:
        return None
    raw = str(path_value).strip().replace("\\", "/")
    if not raw:
        return None
    rel = raw.lstrip("/")
    if rel.startswith("captures/"):
        rel = rel[len("captures/") :]
    candidates = [CAPTURES_DIR / rel]
    if not Path(raw).is_absolute():
        candidates.append(PROJECT_DIR / raw)
    p: Optional[Path] = None
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved.is_file():
            p = resolved
            break
    if p is None:
        return None
    try:
        rel = p.relative_to(CAPTURES_DIR.resolve())
    except Exception:
        return None
    rel_str = str(rel).replace("\\", "/")
    return f"/captures/{rel_str}"


def _face_capture_file_ok(path_value: Optional[str]) -> bool:
    """Skip hand/desk/back-of-head crops in dashboard galleries."""
    if not path_value:
        return True
    try:
        import cv2
        from face_print_filter import is_likely_invalid_face_crop

        p = Path(str(path_value))
        if not p.is_absolute():
            p = CAPTURES_DIR / p
        if not p.is_file():
            return True
        img = cv2.imread(str(p))
        if img is None:
            return True
        if capture_display_is_full_frame() or capture_display_is_person_box():
            return True
        h, w = int(img.shape[0]), int(img.shape[1])
        body_crop = capture_display_is_body_crop()
        if body_crop:
            # Hide leftover tight face squares from older saves.
            return h / float(max(1, w)) >= 1.18
        # Tight-face mode: squares only (skip leftover body/scene strips).
        if h > w * 1.12 or w > h * 1.12:
            return False
        bad, _reason = is_likely_invalid_face_crop(img)
        return not bad
    except Exception:
        return True


def fetch_profile_images_by_person(conn: sqlite3.Connection) -> Dict[int, str]:
    """Best-face profile image URL by known person id."""
    rows = conn.execute(
        """
        SELECT known_person_id, image_path
        FROM person_best_faces
        """
    ).fetchall()
    result: Dict[int, str] = {}
    for row in rows:
        pid = int(row["known_person_id"])
        url = _capture_url_from_db_path(row["image_path"])
        if not url:
            guess = CAPTURES_DIR / "best_faces" / f"person_{pid}_best.jpg"
            if guess.is_file():
                url = f"/captures/best_faces/person_{pid}_best.jpg"
        if url:
            result[pid] = url

    # Fallback: most recent saved face capture for people without a portrait file.
    missing = conn.execute(
        """
        SELECT DISTINCT known_person_id
        FROM person_best_faces
        WHERE known_person_id IS NOT NULL
        """
    ).fetchall()
    need_fallback = {int(r[0]) for r in missing if int(r[0]) not in result}
    if need_fallback:
        log_rows = conn.execute(
            """
            SELECT known_person_id, capture_path, timestamp
            FROM tracking_logs
            WHERE known_person_id IS NOT NULL
              AND capture_path IS NOT NULL
              AND TRIM(capture_path) != ''
            ORDER BY timestamp DESC, id DESC
            """
        ).fetchall()
        for row in log_rows:
            pid = int(row["known_person_id"])
            if pid not in need_fallback or pid in result:
                continue
            url = _capture_url_from_db_path(row["capture_path"])
            if url:
                result[pid] = url
                need_fallback.discard(pid)
            if not need_fallback:
                break

    return result


def fetch_visitors(
    conn: sqlite3.Connection, person_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    known = uses_known_ids(conn)
    id_col = "known_person_id" if known else "person_id"
    person_clause, person_params = _person_filter_sql(conn, person_id)

    rows = conn.execute(
        f"""
        SELECT
            {id_col} AS person_id,
            camera_location,
            COUNT(*) AS visit_count,
            MAX(timestamp) AS last_seen
        FROM tracking_logs
        WHERE 1=1{person_clause}
        GROUP BY {id_col}, camera_location
        ORDER BY {id_col}, camera_location
        """,
        person_params,
    ).fetchall()

    grouped: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        pid = row["person_id"]
        if pid is None:
            continue
        if pid not in grouped:
            grouped[pid] = {
                "person_id": pid,
                "total_visits": 0,
                "last_seen": row["last_seen"],
                "locations": [],
                "gender": "Unknown",
                "age": "Unknown",
            }
        grouped[pid]["total_visits"] += row["visit_count"]
        grouped[pid]["locations"].append(
            {
                "name": row["camera_location"],
                "visits": row["visit_count"],
                "last_seen": row["last_seen"],
            }
        )
        if row["last_seen"] > grouped[pid]["last_seen"]:
            grouped[pid]["last_seen"] = row["last_seen"]

    visitors = sorted(grouped.values(), key=lambda v: v["total_visits"], reverse=True)

    if visitors:
        attrs_by_person = fetch_latest_attributes_by_person(conn, id_col)
        profiles = fetch_profile_images_by_person(conn) if known else {}
        for visitor in visitors:
            attrs = attrs_by_person.get(visitor["person_id"], {})
            visitor["gender"] = attrs.get("gender", "Unknown")
            visitor["age"] = attrs.get("age", "Unknown")
            visitor["profile_image_url"] = profiles.get(visitor["person_id"])

    return visitors


def fetch_recent_logs(
    conn: sqlite3.Connection, limit: int = 30, person_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    known = uses_known_ids(conn)
    person_clause, person_params = _person_filter_sql(conn, person_id)
    rows = conn.execute(
        f"""
        SELECT
            person_id,
            known_person_id,
            camera_location,
            timestamp,
            capture_path,
            gender,
            age
        FROM tracking_logs
        WHERE 1=1{person_clause}
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (*person_params, limit),
    ).fetchall()

    result = []
    for row in rows:
        display_id = row["known_person_id"] if known and row["known_person_id"] else row["person_id"]
        result.append(
            {
                "person_id": display_id,
                "camera_location": row["camera_location"],
                "timestamp": row["timestamp"],
                "capture_url": _capture_url_from_db_path(row["capture_path"]),
                "gender": format_gender(row["gender"]),
                "age": format_age(row["age"]),
            }
        )
    return result


def fetch_by_location(
    conn: sqlite3.Connection, person_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    person_clause, person_params = _person_filter_sql(conn, person_id)
    rows = conn.execute(
        f"""
        SELECT camera_location, COUNT(*) AS visits
        FROM tracking_logs
        WHERE 1=1{person_clause}
        GROUP BY camera_location
        ORDER BY visits DESC
        """,
        person_params,
    ).fetchall()
    return [{"location": r["camera_location"], "visits": r["visits"]} for r in rows]


def fetch_timeline(
    conn: sqlite3.Connection, limit: int = 24, person_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    person_clause, person_params = _person_filter_sql(conn, person_id)
    rows = conn.execute(
        f"""
        SELECT substr(timestamp, 1, 13) AS hour, COUNT(*) AS visits
        FROM tracking_logs
        WHERE 1=1{person_clause}
        GROUP BY hour
        ORDER BY hour DESC
        LIMIT ?
        """,
        (*person_params, limit),
    ).fetchall()
    return [{"hour": r["hour"], "visits": r["visits"]} for r in reversed(rows)]


def fetch_captures(
    conn: sqlite3.Connection, limit: int = 48, person_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    known = uses_known_ids(conn)
    query = """
        SELECT
            id,
            person_id,
            known_person_id,
            camera_location,
            timestamp,
            capture_path,
            gender,
            age
        FROM tracking_logs
        WHERE capture_path IS NOT NULL AND capture_path != ''
    """
    params: list = []
    if person_id is not None:
        if known:
            query += " AND known_person_id = ?"
        else:
            query += " AND person_id = ?"
        params.append(person_id)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    result = []
    for row in rows:
        filename = row["capture_path"]
        file_path = CAPTURES_DIR / filename
        if not file_path.is_file():
            continue
        display_id = (
            row["known_person_id"]
            if known and row["known_person_id"]
            else row["person_id"]
        )
        result.append(
            {
                "id": row["id"],
                "person_id": display_id,
                "camera_location": row["camera_location"],
                "timestamp": row["timestamp"],
                "filename": filename,
                "image_url": f"/captures/{filename}",
                "gender": format_gender(row["gender"]),
                "age": format_age(row["age"]),
            }
        )
    return result


def ensure_device_detection_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS device_detection_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            device_type TEXT NOT NULL,
            source_channel TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'HIGH',
            confidence REAL,
            camera_id TEXT,
            scan_source TEXT DEFAULT 'live',
            snapshot_path TEXT,
            frame_number INTEGER
        )
        """
    )
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(device_detection_events)")
    }
    if "snapshot_path" not in existing:
        conn.execute(
            "ALTER TABLE device_detection_events ADD COLUMN snapshot_path TEXT"
        )
    if "frame_number" not in existing:
        conn.execute(
            "ALTER TABLE device_detection_events ADD COLUMN frame_number INTEGER"
        )
    conn.commit()


def fetch_device_detection_logs(
    conn: sqlite3.Connection, limit: int = 50, since_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    ensure_device_detection_table(conn)
    if since_id is not None and since_id > 0:
        rows = conn.execute(
            """
            SELECT id, created_at, device_type, source_channel, severity,
                   confidence, camera_id, scan_source, snapshot_path, frame_number
            FROM device_detection_events
            WHERE id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (since_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, created_at, device_type, source_channel, severity,
                   confidence, camera_id, scan_source, snapshot_path, frame_number
            FROM device_detection_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_live_face_events_as_logs(
    conn: sqlite3.Connection,
    limit: int = 50,
    *,
    today_only: bool = False,
    range_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Map recent tracking_logs face captures into the Live AI Infraction Logs shape.

    device_detection_events is only filled by smoking/device scans; live CCTV
    face captures live in tracking_logs — surface them so the dashboard panel works.

    When ``today_only`` is True (and no ``range_key``), only rows from the current
    calendar day are returned. Pass ``range_key`` ``today`` / ``7d`` / ``30d`` to
    match the dashboard time-range filter.
    """
    ensure_capture_column(conn)
    ensure_attribute_columns(conn)
    try:
        from identity import ensure_log_display_name_column, ensure_watchlist_flag_column

        ensure_log_display_name_column(conn)
        ensure_watchlist_flag_column(conn)
    except Exception:
        pass
    name_by_id: Dict[int, str] = {}
    try:
        kp_cols = {row[1] for row in conn.execute("PRAGMA table_info(known_persons)")}
        if "is_watchlist" in kp_cols:
            for row in conn.execute(
                "SELECT id, name FROM known_persons WHERE name IS NOT NULL AND TRIM(name) != ''"
            ).fetchall():
                name_by_id[int(row["id"])] = str(row["name"]).strip()
        else:
            for row in conn.execute(
                "SELECT id, name FROM known_persons WHERE name IS NOT NULL AND TRIM(name) != ''"
            ).fetchall():
                name_by_id[int(row["id"])] = str(row["name"]).strip()
    except sqlite3.Error:
        pass

    log_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    display_col = ", display_name" if "display_name" in log_cols else ""
    key = (range_key or "").strip().lower()
    if key in ("today", "7d", "30d", "week", "month"):
        if key == "week":
            key = "7d"
        if key == "month":
            key = "30d"
        day_clause = f"AND {_dashboard_range_date_filter(key)}"
    elif today_only:
        day_clause = "AND date(timestamp) = date('now')"
    else:
        day_clause = ""
    rows = conn.execute(
        f"""
        SELECT id, person_id, known_person_id, camera_location, timestamp,
               capture_path, scene_path, gender, age{display_col}
        FROM tracking_logs
        WHERE capture_path IS NOT NULL AND TRIM(capture_path) != ''
          {day_clause}
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
        """,
        (max(1, int(limit) * 3),),
    ).fetchall()

    events: List[Dict[str, Any]] = []
    for row in rows:
        face_ok = _face_capture_file_ok(row["capture_path"])
        cam = str(row["camera_location"] or "").strip() or "—"
        known_id = row["known_person_id"]
        person_id = row["person_id"]
        display_id = known_id if known_id is not None else person_id
        labeled = ""
        try:
            labeled = str(row["display_name"] or "").strip()
        except (KeyError, IndexError, TypeError):
            labeled = ""
        is_stranger = True
        person_name = "Stranger"
        match_distance: Optional[float] = None
        try:
            from identity import (
                capture_identity_label,
                is_auto_identity_name,
                is_confident_named_match,
            )

            person_name, is_stranger = capture_identity_label(
                conn, known_id, person_id=person_id
            )
            if is_stranger and labeled and not is_auto_identity_name(labeled):
                person_name = labeled
                is_stranger = False
        except Exception:
            if labeled:
                person_name = labeled
                is_stranger = False
        is_watchlist = bool(is_stranger)
        closest_named: Dict[str, Any] = {}

        capture_url = None
        if face_ok:
            capture_url = _capture_url_from_db_path(row["capture_path"])
            # Keep relative path for enrichment even if file was purged.
            if not capture_url and row["capture_path"]:
                rel = str(row["capture_path"]).replace("\\", "/").lstrip("/")
                if rel.startswith("captures/"):
                    rel = rel[len("captures/") :]
                capture_url = f"/captures/{rel}"

        scene_url = None
        try:
            scene_url = _capture_url_from_db_path(row["scene_path"])
        except (KeyError, IndexError, TypeError):
            scene_url = None
        if not scene_url:
            # Fallback: same timestamp/id with _scene_ naming.
            face_name = str(row["capture_path"] or "").replace("\\", "/").split("/")[-1]
            if "_face_" in face_name:
                scene_guess = face_name.replace("_face_", "_scene_", 1)
                if (CAPTURES_DIR / scene_guess).is_file():
                    scene_url = f"/captures/{scene_guess}"

        if not face_ok and not scene_url:
            continue

        if is_stranger and face_ok:
            try:
                import cv2
                from identity import (
                    _embed_scan_reference_face,
                    closest_named_candidate,
                    is_auto_identity_name,
                )

                capture_path = str(row["capture_path"] or "").strip()
                if capture_path:
                    probe = cv2.imread(str((PROJECT_DIR / capture_path).resolve()))
                    if probe is not None and getattr(probe, "size", 0) > 0:
                        embedded = _embed_scan_reference_face(probe)
                        if embedded is not None:
                            feature, _meta = embedded
                            candidate = closest_named_candidate(feature, conn)
                            if candidate and not is_auto_identity_name(candidate.get("name")):
                                closest_named = candidate
                                if candidate.get("within_gate"):
                                    person_name = str(candidate.get("name") or person_name)
                                    is_stranger = False
                                    is_watchlist = False
                                    match_distance = float(candidate.get("distance", 999))
            except Exception:
                closest_named = {}

        gender = format_gender(row["gender"]) if row["gender"] else None
        age = format_age(row["age"]) if row["age"] is not None else None
        detail_bits = [b for b in (gender, age) if b and str(b).lower() != "unknown"]
        image_quality = ""
        if face_ok:
            try:
                import cv2
                from identity import describe_face_sharpness

                capture_path = str(row["capture_path"] or "").strip()
                probe = cv2.imread(str((PROJECT_DIR / capture_path).resolve()))
                if probe is not None and getattr(probe, "size", 0) > 0:
                    image_quality, _ = describe_face_sharpness(probe)
                    if image_quality and image_quality != "Clear":
                        detail_bits.append(image_quality)
            except Exception:
                image_quality = ""
        event_type = "Face detected"
        if person_name:
            event_type = f"Face detected · {person_name}"
        if detail_bits:
            event_type = f"{event_type} · {' · '.join(str(b) for b in detail_bits)}"

        cam_label = cam if cam.upper().startswith("CAM") else f"CAM-{cam}"
        primary_url = scene_url or capture_url
        events.append(
            {
                "id": int(row["id"]),
                "created_at": row["timestamp"],
                "timestamp": row["timestamp"],
                "device_type": event_type,
                "event_type": event_type,
                "source_channel": "Live CCTV",
                "severity": "INFO",
                "confidence": None,
                "camera_id": cam_label,
                "camera": cam_label,
                "camera_location": cam,
                "scan_source": "live",
                "capture_path": row["capture_path"],
                "scene_path": row["scene_path"] if "scene_path" in row.keys() else None,
                "snapshot_path": primary_url,
                "image_url": primary_url,
                "snapshot_url": primary_url,
                "face_url": capture_url,
                "capture_url": capture_url,
                "scene_url": scene_url,
                "full_frame": scene_url,
                "scene_image": scene_url,
                "frame_number": None,
                "person_name": person_name,
                "name": person_name,
                "image_quality": image_quality or None,
                "is_watchlist": is_watchlist,
                "is_named": not is_stranger,
                "is_stranger": is_stranger,
                "known_person_id": known_id,
                "person_id": display_id,
                "closest_named_name": closest_named.get("name"),
                "closest_named_person_id": closest_named.get("known_person_id"),
                "closest_named_match_percent": closest_named.get("match_percent"),
                "closest_named_distance": closest_named.get("distance"),
                "closest_named_within_gate": closest_named.get("within_gate"),
                "closest_named_gate": closest_named.get("gate"),
                "review_status": None,
            }
        )
        if len(events) >= limit:
            break
    return events


def _parse_age_years(raw: Any) -> Optional[int]:
    if raw is None or raw == "" or str(raw).lower() == "unknown":
        return None
    try:
        age = int(raw)
    except (TypeError, ValueError):
        return None
    return age if 0 < age < 120 else None


def age_bucket_label(age: Optional[int]) -> str:
    if age is None:
        return "Unknown"
    if age < 18:
        return "0–17"
    if age < 30:
        return "18–29"
    if age < 45:
        return "30–44"
    if age < 60:
        return "45–59"
    return "60+"


def normalize_age_bucket_param(raw: Any) -> str:
    """Map UI / query age labels onto canonical dashboard buckets."""
    text = str(raw or "").strip()
    if not text:
        return "Unknown"
    compact = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace(" ", "")
        .lower()
    )
    aliases = {
        "0-17": "0–17",
        "18-29": "18–29",
        "30-44": "30–44",
        "45-59": "45–59",
        "60+": "60+",
        "60": "60+",
        "unknown": "Unknown",
    }
    if compact in aliases:
        return aliases[compact]
    for label in ("0–17", "18–29", "30–44", "45–59", "60+", "Unknown"):
        if text == label:
            return label
    return "Unknown"


def fetch_captures_by_age_bucket(
    conn: sqlite3.Connection,
    age_bucket: str,
    *,
    range_key: str = "today",
    limit: int = 48,
) -> List[Dict[str, Any]]:
    """
    Face/scene captures whose estimated age falls in ``age_bucket``.

    Used when clicking an Age distribution bar on the dashboard.
    """
    ensure_capture_column(conn)
    ensure_attribute_columns(conn)
    try:
        from identity import ensure_log_display_name_column

        ensure_log_display_name_column(conn)
    except Exception:
        pass

    bucket = normalize_age_bucket_param(age_bucket)
    range_key = (range_key or "today").strip().lower()
    if range_key not in ("today", "7d", "30d"):
        range_key = "today"
    if range_key == "today":
        date_filter = "date(timestamp) = date('now')"
    elif range_key == "30d":
        date_filter = "date(timestamp) >= date('now', '-29 days')"
    else:
        date_filter = "date(timestamp) >= date('now', '-6 days')"

    # Filter age in SQL so busy CATCH_EVERY_PERSON days (many Unknown plates)
    # cannot push real 30–44 rows out of a post-filter LIMIT window.
    if bucket == "Unknown":
        age_filter = (
            "(age IS NULL OR TRIM(CAST(age AS TEXT)) = '' "
            "OR lower(TRIM(CAST(age AS TEXT))) = 'unknown')"
        )
    elif bucket == "0–17":
        age_filter = "(CAST(age AS INTEGER) BETWEEN 1 AND 17)"
    elif bucket == "18–29":
        age_filter = "(CAST(age AS INTEGER) BETWEEN 18 AND 29)"
    elif bucket == "30–44":
        age_filter = "(CAST(age AS INTEGER) BETWEEN 30 AND 44)"
    elif bucket == "45–59":
        age_filter = "(CAST(age AS INTEGER) BETWEEN 45 AND 59)"
    elif bucket == "60+":
        age_filter = "(CAST(age AS INTEGER) >= 60 AND CAST(age AS INTEGER) < 120)"
    else:
        age_filter = "1=1"

    limit = max(1, min(int(limit), 96))
    log_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    display_col = ", display_name" if "display_name" in log_cols else ""
    rows = conn.execute(
        f"""
        SELECT id, person_id, known_person_id, camera_location, timestamp,
               capture_path, scene_path, gender, age{display_col}
        FROM tracking_logs
        WHERE {date_filter}
          AND {age_filter}
          AND (
            (capture_path IS NOT NULL AND TRIM(capture_path) != '')
            OR (scene_path IS NOT NULL AND TRIM(scene_path) != '')
          )
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    events: List[Dict[str, Any]] = []
    for row in rows:
        years = _parse_age_years(row["age"])
        if age_bucket_label(years) != bucket:
            continue
        # Age-bucket gallery: show the saved plate even if rear/profile heuristics
        # would hide it from generic recent-capture strips.
        capture_url = _capture_url_from_db_path(row["capture_path"])
        if not capture_url and row["capture_path"]:
            rel = str(row["capture_path"]).replace("\\", "/").lstrip("/")
            if rel.startswith("captures/"):
                rel = rel[len("captures/") :]
            if (CAPTURES_DIR / rel).is_file():
                capture_url = f"/captures/{rel}"
        scene_url = None
        try:
            scene_url = _capture_url_from_db_path(row["scene_path"])
        except (KeyError, IndexError, TypeError):
            scene_url = None
        if not capture_url and not scene_url:
            continue

        known_id = row["known_person_id"]
        person_id = row["person_id"]
        labeled = ""
        try:
            labeled = str(row["display_name"] or "").strip()
        except (KeyError, IndexError, TypeError):
            labeled = ""
        person_name = labeled or "Stranger"
        is_stranger = True
        try:
            from identity import capture_identity_label, is_auto_identity_name

            person_name, is_stranger = capture_identity_label(
                conn, known_id, person_id=person_id
            )
            if is_stranger and labeled and not is_auto_identity_name(labeled):
                person_name = labeled
                is_stranger = False
        except Exception:
            if labeled:
                person_name = labeled
                is_stranger = False

        cam = str(row["camera_location"] or "").strip() or "—"
        cam_label = cam if cam.upper().startswith("CAM") else f"CAM-{cam}"
        gender = format_gender(row["gender"]) if row["gender"] else None
        age = format_age(row["age"]) if row["age"] is not None else None
        primary_url = capture_url or scene_url
        events.append(
            {
                "id": int(row["id"]),
                "created_at": row["timestamp"],
                "timestamp": row["timestamp"],
                "device_type": f"Age {bucket}",
                "event_type": f"Age {bucket} · {person_name}",
                "source_channel": "Live CCTV",
                "severity": "INFO",
                "camera_id": cam_label,
                "camera": cam_label,
                "camera_location": cam,
                "capture_path": row["capture_path"],
                "scene_path": row["scene_path"] if "scene_path" in row.keys() else None,
                "image_url": primary_url,
                "snapshot_url": primary_url,
                "face_url": capture_url,
                "capture_url": capture_url,
                "scene_url": scene_url,
                "person_name": person_name,
                "name": person_name,
                "is_stranger": is_stranger,
                "is_named": not is_stranger,
                "known_person_id": known_id,
                "person_id": known_id if known_id is not None else person_id,
                "gender": gender,
                "age": age,
                "age_bucket": bucket,
            }
        )
        if len(events) >= limit:
            break
    return events


def normalize_camera_location_param(raw: Any) -> str:
    """Normalize CAM-3 / cam 3 / 3 to the tracking_logs camera_location token."""
    text = str(raw or "").strip()
    if not text:
        return ""
    cleaned = re.sub(r"^(cam[-\s]*)+", "", text, flags=re.IGNORECASE).strip()
    return cleaned or text


def fetch_captures_by_camera(
    conn: sqlite3.Connection,
    camera_location: str,
    *,
    range_key: str = "today",
    limit: int = 48,
) -> List[Dict[str, Any]]:
    """Face/scene captures for one camera (Camera status tile click)."""
    ensure_capture_column(conn)
    ensure_attribute_columns(conn)
    try:
        from identity import ensure_log_display_name_column

        ensure_log_display_name_column(conn)
    except Exception:
        pass

    cam_key = normalize_camera_location_param(camera_location)
    if not cam_key:
        return []
    range_key = (range_key or "today").strip().lower()
    if range_key not in ("today", "7d", "30d"):
        range_key = "today"
    if range_key == "today":
        date_filter = "date(timestamp) = date('now')"
    elif range_key == "30d":
        date_filter = "date(timestamp) >= date('now', '-29 days')"
    else:
        date_filter = "date(timestamp) >= date('now', '-6 days')"

    limit = max(1, min(int(limit), 120))
    log_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    display_col = ", display_name" if "display_name" in log_cols else ""
    # Match location as stored ("3") or labeled ("CAM-3").
    rows = conn.execute(
        f"""
        SELECT id, person_id, known_person_id, camera_location, timestamp,
               capture_path, scene_path, gender, age{display_col}
        FROM tracking_logs
        WHERE {date_filter}
          AND (
            lower(trim(camera_location)) = lower(?)
            OR lower(trim(camera_location)) = lower(?)
            OR lower(replace(trim(camera_location), ' ', '')) = lower(?)
          )
          AND (
            (capture_path IS NOT NULL AND TRIM(capture_path) != '')
            OR (scene_path IS NOT NULL AND TRIM(scene_path) != '')
          )
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
        """,
        (cam_key, f"CAM-{cam_key}", f"cam-{cam_key}", limit * 3),
    ).fetchall()

    events: List[Dict[str, Any]] = []
    for row in rows:
        face_ok = _face_capture_file_ok(row["capture_path"])
        capture_url = _capture_url_from_db_path(row["capture_path"]) if face_ok else None
        if not capture_url and row["capture_path"]:
            rel = str(row["capture_path"]).replace("\\", "/").lstrip("/")
            if rel.startswith("captures/"):
                rel = rel[len("captures/") :]
            if (CAPTURES_DIR / rel).is_file():
                capture_url = f"/captures/{rel}"
        scene_url = None
        try:
            scene_url = _capture_url_from_db_path(row["scene_path"])
        except (KeyError, IndexError, TypeError):
            scene_url = None
        if not capture_url and not scene_url:
            continue

        known_id = row["known_person_id"]
        person_id = row["person_id"]
        labeled = ""
        try:
            labeled = str(row["display_name"] or "").strip()
        except (KeyError, IndexError, TypeError):
            labeled = ""
        person_name = labeled or "Stranger"
        is_stranger = True
        try:
            from identity import capture_identity_label, is_auto_identity_name

            person_name, is_stranger = capture_identity_label(
                conn, known_id, person_id=person_id
            )
            if is_stranger and labeled and not is_auto_identity_name(labeled):
                person_name = labeled
                is_stranger = False
        except Exception:
            if labeled:
                person_name = labeled
                is_stranger = False

        cam = str(row["camera_location"] or "").strip() or cam_key
        cam_label = cam if cam.upper().startswith("CAM") else f"CAM-{cam}"
        gender = format_gender(row["gender"]) if row["gender"] else None
        age = format_age(row["age"]) if row["age"] is not None else None
        primary_url = capture_url or scene_url
        events.append(
            {
                "id": int(row["id"]),
                "created_at": row["timestamp"],
                "timestamp": row["timestamp"],
                "device_type": f"{cam_label} capture",
                "event_type": f"{cam_label} · {person_name}",
                "source_channel": "Live CCTV",
                "severity": "INFO",
                "camera_id": cam_label,
                "camera": cam_label,
                "camera_location": cam,
                "capture_path": row["capture_path"],
                "scene_path": row["scene_path"] if "scene_path" in row.keys() else None,
                "image_url": primary_url,
                "snapshot_url": primary_url,
                "face_url": capture_url,
                "capture_url": capture_url,
                "scene_url": scene_url,
                "person_name": person_name,
                "name": person_name,
                "is_stranger": is_stranger,
                "is_named": not is_stranger,
                "known_person_id": known_id,
                "person_id": known_id if known_id is not None else person_id,
                "gender": gender,
                "age": age,
            }
        )
        if len(events) >= limit:
            break
    return events


def count_captures_by_camera(
    conn: sqlite3.Connection,
    camera_location: str,
    *,
    range_key: str = "today",
) -> int:
    """Total saved captures for a camera in the dashboard time range."""
    cam_key = normalize_camera_location_param(camera_location)
    if not cam_key:
        return 0
    range_key = (range_key or "today").strip().lower()
    if range_key not in ("today", "7d", "30d"):
        range_key = "today"
    if range_key == "today":
        date_filter = "date(timestamp) = date('now')"
    elif range_key == "30d":
        date_filter = "date(timestamp) >= date('now', '-29 days')"
    else:
        date_filter = "date(timestamp) >= date('now', '-6 days')"
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS c FROM tracking_logs
        WHERE {date_filter}
          AND (
            lower(trim(camera_location)) = lower(?)
            OR lower(trim(camera_location)) = lower(?)
            OR lower(replace(trim(camera_location), ' ', '')) = lower(?)
          )
          AND (
            (capture_path IS NOT NULL AND TRIM(capture_path) != '')
            OR (scene_path IS NOT NULL AND TRIM(scene_path) != '')
          )
        """,
        (cam_key, f"CAM-{cam_key}", f"cam-{cam_key}"),
    ).fetchone()
    try:
        return int(row["c"] if row and "c" in row.keys() else (row[0] if row else 0))
    except Exception:
        return 0


def normalize_gender_param(raw: Any) -> str:
    """Map UI / query gender labels onto Male / Female / Unknown."""
    text = str(raw or "").strip().lower()
    if text in ("man", "male") or text.startswith("m"):
        return "Male"
    if text in ("woman", "female") or text.startswith("f"):
        return "Female"
    return "Unknown"


def _gender_sql_clause(gender: str) -> str:
    """SQL predicate matching normalize_gender_param (Male / Female / Unknown)."""
    bucket = normalize_gender_param(gender)
    token = "LOWER(TRIM(COALESCE(gender, '')))"
    if bucket == "Male":
        return f"{token} LIKE 'm%'"
    if bucket == "Female":
        return f"{token} LIKE 'f%'"
    return f"{token} NOT LIKE 'm%' AND {token} NOT LIKE 'f%'"


def _dashboard_range_date_filter(range_key: str) -> str:
    key = (range_key or "today").strip().lower()
    if key == "today":
        return "date(timestamp) = date('now')"
    if key == "30d":
        return "date(timestamp) >= date('now', '-29 days')"
    return "date(timestamp) >= date('now', '-6 days')"


def fetch_captures_by_gender(
    conn: sqlite3.Connection,
    gender: str,
    *,
    range_key: str = "today",
    limit: int = 48,
) -> List[Dict[str, Any]]:
    """Captures for one Gender split bucket (click Male / Female / Unknown)."""
    ensure_capture_column(conn)
    ensure_attribute_columns(conn)
    try:
        from identity import ensure_log_display_name_column

        ensure_log_display_name_column(conn)
    except Exception:
        pass

    bucket = normalize_gender_param(gender)
    date_filter = _dashboard_range_date_filter(range_key)
    gender_filter = _gender_sql_clause(bucket)
    limit = max(1, min(int(limit), 120))
    log_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    display_col = ", display_name" if "display_name" in log_cols else ""
    rows = conn.execute(
        f"""
        SELECT id, person_id, known_person_id, camera_location, timestamp,
               capture_path, scene_path, gender, age{display_col}
        FROM tracking_logs
        WHERE {date_filter}
          AND ({gender_filter})
          AND (
            (capture_path IS NOT NULL AND TRIM(capture_path) != '')
            OR (scene_path IS NOT NULL AND TRIM(scene_path) != '')
          )
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
        """,
        (max(limit * 3, 150),),
    ).fetchall()

    events: List[Dict[str, Any]] = []
    for row in rows:
        if normalize_gender_param(row["gender"]) != bucket:
            continue
        face_ok = _face_capture_file_ok(row["capture_path"])
        capture_url = _capture_url_from_db_path(row["capture_path"]) if face_ok else None
        if not capture_url and row["capture_path"]:
            rel = str(row["capture_path"]).replace("\\", "/").lstrip("/")
            if rel.startswith("captures/"):
                rel = rel[len("captures/") :]
            if (CAPTURES_DIR / rel).is_file():
                capture_url = f"/captures/{rel}"
        scene_url = None
        try:
            scene_url = _capture_url_from_db_path(row["scene_path"])
        except (KeyError, IndexError, TypeError):
            scene_url = None
        if not capture_url and not scene_url:
            continue

        known_id = row["known_person_id"]
        person_id = row["person_id"]
        labeled = ""
        try:
            labeled = str(row["display_name"] or "").strip()
        except (KeyError, IndexError, TypeError):
            labeled = ""
        person_name = labeled or "Stranger"
        is_stranger = True
        try:
            from identity import capture_identity_label, is_auto_identity_name

            person_name, is_stranger = capture_identity_label(
                conn, known_id, person_id=person_id
            )
            if is_stranger and labeled and not is_auto_identity_name(labeled):
                person_name = labeled
                is_stranger = False
        except Exception:
            if labeled:
                person_name = labeled
                is_stranger = False

        cam = str(row["camera_location"] or "").strip() or "—"
        cam_label = cam if cam.upper().startswith("CAM") else f"CAM-{cam}"
        gender_label = format_gender(row["gender"]) if row["gender"] else "Unknown"
        age = format_age(row["age"]) if row["age"] is not None else None
        primary_url = capture_url or scene_url
        events.append(
            {
                "id": int(row["id"]),
                "created_at": row["timestamp"],
                "timestamp": row["timestamp"],
                "device_type": f"{bucket} capture",
                "event_type": f"{bucket} · {person_name}",
                "source_channel": "Live CCTV",
                "severity": "INFO",
                "camera_id": cam_label,
                "camera": cam_label,
                "camera_location": cam,
                "capture_path": row["capture_path"],
                "scene_path": row["scene_path"] if "scene_path" in row.keys() else None,
                "image_url": primary_url,
                "snapshot_url": primary_url,
                "face_url": capture_url,
                "capture_url": capture_url,
                "scene_url": scene_url,
                "person_name": person_name,
                "name": person_name,
                "is_stranger": is_stranger,
                "is_named": not is_stranger,
                "known_person_id": known_id,
                "person_id": known_id if known_id is not None else person_id,
                "gender": gender_label,
                "age": age,
            }
        )
        if len(events) >= limit:
            break
    return events


def count_captures_by_gender(
    conn: sqlite3.Connection,
    gender: str,
    *,
    range_key: str = "today",
) -> int:
    bucket = normalize_gender_param(gender)
    date_filter = _dashboard_range_date_filter(range_key)
    gender_filter = _gender_sql_clause(bucket)
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS c FROM tracking_logs
        WHERE {date_filter}
          AND ({gender_filter})
          AND (
            (capture_path IS NOT NULL AND TRIM(capture_path) != '')
            OR (scene_path IS NOT NULL AND TRIM(scene_path) != '')
          )
        """
    ).fetchone()
    try:
        return int(row["c"] if row and "c" in row.keys() else (row[0] if row else 0))
    except Exception:
        return 0


# 3-hour detection chart buckets (must match dashboard._fetch_hourly_visit_counts).
DETECTION_HOUR_BUCKETS: Dict[str, Tuple[int, int]] = {
    "6am": (6, 8),
    "9am": (9, 11),
    "12pm": (12, 14),
    "3pm": (15, 17),
    "6pm": (18, 20),
    "9pm": (21, 23),
}


def normalize_hour_bucket_param(raw: Any) -> Optional[str]:
    token = str(raw or "").strip().lower().replace(" ", "")
    aliases = {
        "6": "6am",
        "06": "6am",
        "6:00": "6am",
        "9": "9am",
        "09": "9am",
        "9:00": "9am",
        "12": "12pm",
        "12:00": "12pm",
        "15": "3pm",
        "3": "3pm",
        "3:00": "3pm",
        "15:00": "3pm",
        "18": "6pm",
        "6pm": "6pm",
        "18:00": "6pm",
        "21": "9pm",
        "21:00": "9pm",
        "9pm": "9pm",
    }
    if token in DETECTION_HOUR_BUCKETS:
        return token
    mapped = aliases.get(token)
    if mapped in DETECTION_HOUR_BUCKETS:
        return mapped
    return None


def _capture_events_from_log_rows(
    conn: sqlite3.Connection,
    rows: List[Any],
    *,
    limit: int,
    event_prefix: str,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for row in rows:
        face_ok = _face_capture_file_ok(row["capture_path"])
        capture_url = _capture_url_from_db_path(row["capture_path"]) if face_ok else None
        if not capture_url and row["capture_path"]:
            rel = str(row["capture_path"]).replace("\\", "/").lstrip("/")
            if rel.startswith("captures/"):
                rel = rel[len("captures/") :]
            if (CAPTURES_DIR / rel).is_file():
                capture_url = f"/captures/{rel}"
        scene_url = None
        try:
            scene_url = _capture_url_from_db_path(row["scene_path"])
        except (KeyError, IndexError, TypeError):
            scene_url = None
        if not capture_url and not scene_url:
            continue

        known_id = row["known_person_id"]
        person_id = row["person_id"]
        labeled = ""
        try:
            labeled = str(row["display_name"] or "").strip()
        except (KeyError, IndexError, TypeError):
            labeled = ""
        person_name = labeled or "Stranger"
        is_stranger = True
        try:
            from identity import capture_identity_label, is_auto_identity_name

            person_name, is_stranger = capture_identity_label(
                conn, known_id, person_id=person_id
            )
            if is_stranger and labeled and not is_auto_identity_name(labeled):
                person_name = labeled
                is_stranger = False
        except Exception:
            if labeled:
                person_name = labeled
                is_stranger = False

        cam = str(row["camera_location"] or "").strip() or "—"
        cam_label = cam if cam.upper().startswith("CAM") else f"CAM-{cam}"
        primary_url = capture_url or scene_url
        events.append(
            {
                "id": int(row["id"]),
                "created_at": row["timestamp"],
                "timestamp": row["timestamp"],
                "device_type": f"{event_prefix} capture",
                "event_type": f"{event_prefix} · {person_name}",
                "source_channel": "Live CCTV",
                "severity": "INFO",
                "camera_id": cam_label,
                "camera": cam_label,
                "camera_location": cam,
                "scan_source": "live",
                "capture_path": row["capture_path"],
                "scene_path": row["scene_path"] if "scene_path" in row.keys() else None,
                "snapshot_path": primary_url,
                "image_url": primary_url,
                "snapshot_url": primary_url,
                "face_url": capture_url,
                "capture_url": capture_url,
                "scene_url": scene_url,
                "person_name": person_name,
                "name": person_name,
                "is_stranger": bool(is_stranger),
                "is_named": not is_stranger,
                "known_person_id": known_id,
                "person_id": known_id if known_id is not None else person_id,
                "review_status": None,
            }
        )
        if len(events) >= limit:
            break
    return events


OVERVIEW_CATEGORIES = ("known", "visitors", "watchlist", "other")


def normalize_overview_category(raw: object) -> str:
    token = str(raw or "").strip().lower().replace(" ", "_")
    aliases = {
        "known": "known",
        "named": "known",
        "visitor": "visitors",
        "visitors": "visitors",
        "stranger": "visitors",
        "strangers": "visitors",
        "watchlist": "watchlist",
        "blacklist": "watchlist",
        "black_list": "watchlist",
        "other": "other",
        "others": "other",
        "unknown": "other",
    }
    return aliases.get(token, "other")


def _overview_category_sql(category: str) -> str:
    """
    SQL fragment classifying today's breakdown buckets (mutually exclusive).

    Known = Named People (not watchlist)
    Watchlist = enrolled watchlist
    Visitors = strangers with Male/Female gender
    Other = strangers with Unknown / missing gender
    """
    cat = normalize_overview_category(category)
    named = "COALESCE(kp.is_named, 0) = 1 AND COALESCE(kp.is_watchlist, 0) = 0"
    watch = "COALESCE(kp.is_watchlist, 0) = 1"
    stranger = (
        "(tl.known_person_id IS NULL OR ("
        "COALESCE(kp.is_named, 0) = 0 AND COALESCE(kp.is_watchlist, 0) = 0))"
    )
    known_gender = (
        "tl.gender IS NOT NULL AND TRIM(COALESCE(tl.gender, '')) != '' "
        "AND lower(TRIM(tl.gender)) IN ('male', 'female', 'm', 'f', 'man', 'woman')"
    )
    unknown_gender = (
        "tl.gender IS NULL OR TRIM(COALESCE(tl.gender, '')) = '' "
        "OR lower(TRIM(tl.gender)) IN ('unknown', 'n/a', 'na', 'none')"
    )
    if cat == "known":
        return f"({named})"
    if cat == "watchlist":
        return f"({watch})"
    if cat == "visitors":
        return f"({stranger} AND ({known_gender}))"
    return f"({stranger} AND ({unknown_gender}))"


def _ensure_overview_person_flags(conn: sqlite3.Connection) -> None:
    try:
        from identity import ensure_named_flag_column, ensure_watchlist_flag_column

        ensure_named_flag_column(conn)
        ensure_watchlist_flag_column(conn)
    except Exception:
        pass


def _parse_log_timestamp(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    return None


def _captures_look_identical(path_a: str, path_b: str) -> bool:
    """True when two saved JPEG crops are the same picture (duplicate stranger spam)."""
    if not path_a or not path_b:
        return False
    if path_a == path_b:
        return True
    try:
        import cv2
        import numpy as np

        pa = CAPTURES_DIR / Path(str(path_a).replace("\\", "/")).name
        pb = CAPTURES_DIR / Path(str(path_b).replace("\\", "/")).name
        if not pa.is_file() or not pb.is_file():
            return False
        if pa.stat().st_size == pb.stat().st_size:
            with pa.open("rb") as fa, pb.open("rb") as fb:
                if fa.read(4096) == fb.read(4096):
                    return True
        ia = cv2.imread(str(pa))
        ib = cv2.imread(str(pb))
        if ia is None or ib is None or ia.size == 0 or ib.size == 0:
            return False
        sa = cv2.resize(ia, (48, 48), interpolation=cv2.INTER_AREA)
        sb = cv2.resize(ib, (48, 48), interpolation=cv2.INTER_AREA)
        return float(np.mean(cv2.absdiff(sa, sb))) < 4.0
    except Exception:
        return False


def fetch_capture_encounter_cluster(
    conn: sqlite3.Connection,
    camera_location: str,
    timestamp: str,
    *,
    window_sec: int = 60,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """
    Distinct people captured on one camera within a short window (same hallway pass).

    Collapses duplicate stranger rows and visually identical crops so the review UI
    shows one portrait per person met — not two copies of the same walker.
    """
    ensure_capture_column(conn)
    ensure_attribute_columns(conn)
    center = _parse_log_timestamp(timestamp)
    if center is None:
        return []
    cam = str(camera_location or "").strip()
    if not cam:
        return []
    window_sec = max(10, min(int(window_sec), 180))
    limit = max(1, min(int(limit), 12))
    start = (center - timedelta(seconds=window_sec)).strftime("%Y-%m-%dT%H:%M:%S")
    end = (center + timedelta(seconds=window_sec)).strftime("%Y-%m-%dT%H:%M:%S")
    log_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    display_col = ", tl.display_name" if "display_name" in log_cols else ""
    rows = conn.execute(
        f"""
        SELECT tl.id, tl.person_id, tl.known_person_id, tl.camera_location, tl.timestamp,
               tl.capture_path, tl.scene_path, tl.gender, tl.age{display_col}
        FROM tracking_logs tl
        WHERE tl.camera_location = ?
          AND tl.timestamp >= ?
          AND tl.timestamp <= ?
          AND tl.capture_path IS NOT NULL AND TRIM(tl.capture_path) != ''
        ORDER BY tl.timestamp ASC, tl.id ASC
        LIMIT ?
        """,
        (cam, start, end, max(limit * 4, 24)),
    ).fetchall()
    if not rows:
        return []

    picked: List[Any] = []
    seen_paths: Set[str] = set()
    for row in rows:
        cap = str(row["capture_path"] or "").strip()
        if not cap or cap in seen_paths:
            continue
        duplicate = False
        for kept in picked:
            kept_cap = str(kept["capture_path"] or "").strip()
            if _captures_look_identical(cap, kept_cap):
                duplicate = True
                break
        if duplicate:
            continue
        seen_paths.add(cap)
        picked.append(row)
        if len(picked) >= limit:
            break

    return _capture_events_from_log_rows(
        conn, picked, limit=limit, event_prefix="Encounter"
    )


def fetch_captures_by_overview_category(
    conn: sqlite3.Connection,
    category: str,
    *,
    range_key: str = "today",
    limit: int = 96,
) -> List[Dict[str, Any]]:
    """Captures for one Today's breakdown slice (Known / Visitors / Watchlist / Other)."""
    ensure_capture_column(conn)
    ensure_attribute_columns(conn)
    _ensure_overview_person_flags(conn)
    try:
        from identity import ensure_log_display_name_column

        ensure_log_display_name_column(conn)
    except Exception:
        pass

    cat = normalize_overview_category(category)
    date_filter = _dashboard_range_date_filter(range_key).replace(
        "timestamp", "tl.timestamp"
    )
    cat_filter = _overview_category_sql(cat)
    limit = max(1, min(int(limit), 120))
    log_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    display_col = ", tl.display_name" if "display_name" in log_cols else ""
    rows = conn.execute(
        f"""
        SELECT tl.id, tl.person_id, tl.known_person_id, tl.camera_location, tl.timestamp,
               tl.capture_path, tl.scene_path, tl.gender, tl.age{display_col}
        FROM tracking_logs tl
        LEFT JOIN known_persons kp ON kp.id = tl.known_person_id
        WHERE {date_filter}
          AND ({cat_filter})
          AND (
            (tl.capture_path IS NOT NULL AND TRIM(tl.capture_path) != '')
            OR (tl.scene_path IS NOT NULL AND TRIM(tl.scene_path) != '')
          )
        ORDER BY tl.timestamp DESC, tl.id DESC
        LIMIT ?
        """,
        (max(limit * 3, 150),),
    ).fetchall()
    title = {
        "known": "Known",
        "visitors": "Visitors",
        "watchlist": "Watchlist",
        "other": "Other",
    }.get(cat, "Other")
    return _capture_events_from_log_rows(
        conn, rows, limit=limit, event_prefix=title
    )


def count_captures_by_overview_category(
    conn: sqlite3.Connection,
    category: str,
    *,
    range_key: str = "today",
) -> int:
    ensure_capture_column(conn)
    _ensure_overview_person_flags(conn)
    cat = normalize_overview_category(category)
    date_filter = _dashboard_range_date_filter(range_key).replace(
        "timestamp", "tl.timestamp"
    )
    cat_filter = _overview_category_sql(cat)
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS c
        FROM tracking_logs tl
        LEFT JOIN known_persons kp ON kp.id = tl.known_person_id
        WHERE {date_filter}
          AND ({cat_filter})
          AND (
            (tl.capture_path IS NOT NULL AND TRIM(tl.capture_path) != '')
            OR (tl.scene_path IS NOT NULL AND TRIM(tl.scene_path) != '')
          )
        """
    ).fetchone()
    try:
        return int(row["c"] if row and "c" in row.keys() else (row[0] if row else 0))
    except Exception:
        return 0


def fetch_captures_by_hour_bucket(
    conn: sqlite3.Connection,
    hour_bucket: str,
    *,
    limit: int = 96,
) -> List[Dict[str, Any]]:
    """Captures for one Today (hourly) chart bar (e.g. 6am → 06:00–08:59)."""
    ensure_capture_column(conn)
    ensure_attribute_columns(conn)
    try:
        from identity import ensure_log_display_name_column

        ensure_log_display_name_column(conn)
    except Exception:
        pass

    bucket = normalize_hour_bucket_param(hour_bucket)
    if bucket is None:
        return []
    hour_start, hour_end = DETECTION_HOUR_BUCKETS[bucket]
    limit = max(1, min(int(limit), 120))
    log_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    display_col = ", display_name" if "display_name" in log_cols else ""
    rows = conn.execute(
        f"""
        SELECT id, person_id, known_person_id, camera_location, timestamp,
               capture_path, scene_path, gender, age{display_col}
        FROM tracking_logs
        WHERE date(timestamp) = date('now')
          AND CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN ? AND ?
          AND (
            (capture_path IS NOT NULL AND TRIM(capture_path) != '')
            OR (scene_path IS NOT NULL AND TRIM(scene_path) != '')
          )
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
        """,
        (hour_start, hour_end, max(limit * 3, 150)),
    ).fetchall()
    return _capture_events_from_log_rows(
        conn, rows, limit=limit, event_prefix=bucket
    )


def count_captures_by_hour_bucket(
    conn: sqlite3.Connection,
    hour_bucket: str,
) -> int:
    bucket = normalize_hour_bucket_param(hour_bucket)
    if bucket is None:
        return 0
    hour_start, hour_end = DETECTION_HOUR_BUCKETS[bucket]
    row = conn.execute(
        """
        SELECT COUNT(*) AS c FROM tracking_logs
        WHERE date(timestamp) = date('now')
          AND CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN ? AND ?
          AND (
            (capture_path IS NOT NULL AND TRIM(capture_path) != '')
            OR (scene_path IS NOT NULL AND TRIM(scene_path) != '')
          )
        """,
        (hour_start, hour_end),
    ).fetchone()
    try:
        return int(row["c"] if row and "c" in row.keys() else (row[0] if row else 0))
    except Exception:
        return 0


def fetch_captures_by_day(
    conn: sqlite3.Connection,
    day_iso: str,
    *,
    limit: int = 96,
) -> List[Dict[str, Any]]:
    """Captures for one calendar day (7d / 30d chart bar click)."""
    ensure_capture_column(conn)
    ensure_attribute_columns(conn)
    try:
        from identity import ensure_log_display_name_column

        ensure_log_display_name_column(conn)
    except Exception:
        pass

    day = str(day_iso or "").strip()[:10]
    if len(day) < 10:
        return []
    limit = max(1, min(int(limit), 120))
    log_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    display_col = ", display_name" if "display_name" in log_cols else ""
    rows = conn.execute(
        f"""
        SELECT id, person_id, known_person_id, camera_location, timestamp,
               capture_path, scene_path, gender, age{display_col}
        FROM tracking_logs
        WHERE date(timestamp) = ?
          AND (
            (capture_path IS NOT NULL AND TRIM(capture_path) != '')
            OR (scene_path IS NOT NULL AND TRIM(scene_path) != '')
          )
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
        """,
        (day, max(limit * 3, 150)),
    ).fetchall()
    return _capture_events_from_log_rows(
        conn, rows, limit=limit, event_prefix=day
    )


def count_captures_by_day(conn: sqlite3.Connection, day_iso: str) -> int:
    day = str(day_iso or "").strip()[:10]
    if len(day) < 10:
        return 0
    row = conn.execute(
        """
        SELECT COUNT(*) AS c FROM tracking_logs
        WHERE date(timestamp) = ?
          AND (
            (capture_path IS NOT NULL AND TRIM(capture_path) != '')
            OR (scene_path IS NOT NULL AND TRIM(scene_path) != '')
          )
        """,
        (day,),
    ).fetchone()
    try:
        return int(row["c"] if row and "c" in row.keys() else (row[0] if row else 0))
    except Exception:
        return 0


def fetch_captures_by_window(
    conn: sqlite3.Connection,
    window: str,
    *,
    limit: int = 96,
) -> List[Dict[str, Any]]:
    """Captures for KPI cards: ``today`` or ``active`` (last 15 minutes)."""
    ensure_capture_column(conn)
    ensure_attribute_columns(conn)
    try:
        from identity import ensure_log_display_name_column

        ensure_log_display_name_column(conn)
    except Exception:
        pass

    key = str(window or "today").strip().lower()
    limit = max(1, min(int(limit), 120))
    log_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    display_col = ", display_name" if "display_name" in log_cols else ""

    if key in ("active", "active_now", "recent", "15m"):
        where_sql = "datetime(timestamp) >= datetime('now', '-15 minutes')"
        prefix = "active"
        params: tuple = (max(limit * 3, 150),)
    else:
        where_sql = "date(timestamp) = date('now')"
        prefix = "today"
        params = (max(limit * 3, 150),)

    rows = conn.execute(
        f"""
        SELECT id, person_id, known_person_id, camera_location, timestamp,
               capture_path, scene_path, gender, age{display_col}
        FROM tracking_logs
        WHERE {where_sql}
          AND (
            (capture_path IS NOT NULL AND TRIM(capture_path) != '')
            OR (scene_path IS NOT NULL AND TRIM(scene_path) != '')
          )
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return _capture_events_from_log_rows(
        conn, rows, limit=limit, event_prefix=prefix
    )


def count_captures_by_window(conn: sqlite3.Connection, window: str) -> int:
    key = str(window or "today").strip().lower()
    if key in ("active", "active_now", "recent", "15m"):
        where_sql = "datetime(timestamp) >= datetime('now', '-15 minutes')"
    else:
        where_sql = "date(timestamp) = date('now')"
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS c FROM tracking_logs
        WHERE {where_sql}
          AND (
            (capture_path IS NOT NULL AND TRIM(capture_path) != '')
            OR (scene_path IS NOT NULL AND TRIM(scene_path) != '')
          )
        """
    ).fetchone()
    try:
        return int(row["c"] if row and "c" in row.keys() else (row[0] if row else 0))
    except Exception:
        return 0


def fetch_live_infraction_feed(
    conn: sqlite3.Connection, limit: int = 50
) -> List[Dict[str, Any]]:
    """Device/smoking events plus live face captures, newest first."""
    limit = max(1, min(int(limit), 200))
    device_rows = fetch_device_detection_logs(conn, limit=limit)
    face_rows = fetch_live_face_events_as_logs(conn, limit=limit)

    merged: List[Dict[str, Any]] = []
    for row in device_rows:
        item = dict(row)
        item.setdefault("timestamp", item.get("created_at"))
        item.setdefault("event_type", item.get("device_type"))
        merged.append(item)
    merged.extend(face_rows)

    def _sort_key(item: Dict[str, Any]) -> str:
        return str(item.get("created_at") or item.get("timestamp") or "")

    merged.sort(key=_sort_key, reverse=True)
    return merged[:limit]


def insert_device_detection_event(
    conn: sqlite3.Connection,
    device_type: str,
    source_channel: str,
    severity: str = "HIGH",
    confidence: Optional[float] = None,
    camera_id: Optional[str] = None,
    scan_source: str = "live",
    snapshot_path: Optional[str] = None,
    frame_number: Optional[int] = None,
) -> int:
    ensure_device_detection_table(conn)
    cur = conn.execute(
        """
        INSERT INTO device_detection_events
            (device_type, source_channel, severity, confidence, camera_id,
             scan_source, snapshot_path, frame_number)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            device_type,
            source_channel,
            severity,
            confidence,
            camera_id,
            scan_source,
            snapshot_path,
            frame_number,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def clear_device_detection_events(conn: sqlite3.Connection) -> int:
    ensure_device_detection_table(conn)
    cur = conn.execute("DELETE FROM device_detection_events")
    conn.commit()
    return int(cur.rowcount)
