"""
Face-based identity: match people across visits and photo lookups.

Live CCTV + video scan prefer InsightFace (RetinaFace + ArcFace) on CUDA when
onnxruntime-gpu is available. OpenCV YuNet + SFace remain as CPU fallback for
legacy paths only — photo search uses ArcFace so it matches the gallery.
"""

import numpy_compat  # noqa: F401  — unpickle NumPy 2 gallery blobs on NumPy 1.x

import base64
import glob
import os
import pickle
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from body_reid import BODY_MATCH_THRESHOLD, cosine_distance, extract_body_embedding
from image_geometry import LetterboxTransform, letterbox_resize, square_padded_crop
from face_landmark_verify import verify_yunet_face_for_capture, yunet_landmarks_dict
from tracker_config import (
    FACE_DETECT_MIN_CONFIDENCE,
    FACE_MIN_BOX_PX,
)

PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# Cosine distance: lower = same person (distance = 1 - similarity).
def _env_live_match_distance(default: float = 0.50) -> float:
    """Max cosine distance for live watchlist match (0.50 ≈ 50% similarity)."""
    raw = (
        os.environ.get("WATCHLIST_MATCH_THRESHOLD")
        or os.environ.get("TELEGRAM_ALERT_MIN_PERCENT")
        or os.environ.get("WATCHLIST_ALERT_MIN_PERCENT")
        or ""
    ).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value <= 1.0:
        return max(0.0, min(1.0, 1.0 - value))
    return max(0.0, min(1.0, 1.0 - value / 100.0))


MATCH_THRESHOLD = _env_live_match_distance(0.50)
RELAXED_MATCH_THRESHOLD = min(0.85, MATCH_THRESHOLD + 0.08)
# Named gallery (not watchlist): max cosine distance (lower = stricter). 0.80 ≈ 20% similarity.
try:
    _named_raw = os.environ.get("NAMED_MATCH_THRESHOLD", "0.80").strip()
    NAMED_MATCH_THRESHOLD = min(0.92, max(0.20, float(_named_raw)))
except ValueError:
    NAMED_MATCH_THRESHOLD = 0.80
# Never add live CCTV embeddings onto Named People (enrollment photos only).
NAMED_LIVE_ENCODING_MAX_DISTANCE = 0.0
# Minimum match % before dashboard / Telegram show a named label (not Stranger).
NAMED_LABEL_MIN_PERCENT = int(
    os.environ.get("NAMED_LABEL_MIN_PERCENT", "20")
)
# Side/profile views need a looser gate than frontal enrollment photos.
try:
    NAMED_MATCH_THRESHOLD_PROFILE = min(
        0.92,
        max(0.20, float(os.environ.get("NAMED_MATCH_THRESHOLD_PROFILE", "0.80"))),
    )
except ValueError:
    NAMED_MATCH_THRESHOLD_PROFILE = 0.80
NAMED_LABEL_MIN_PERCENT_PROFILE = int(
    os.environ.get("NAMED_LABEL_MIN_PERCENT_PROFILE", "20")
)
# Re-use a recent confident named sighting on the same camera (front → side).
NAMED_SIGHTING_MEMORY_SEC = float(os.environ.get("NAMED_SIGHTING_MEMORY_SEC", "180"))
NAMED_TRACK_MEMORY_MAX_DIST = float(
    os.environ.get("NAMED_TRACK_MEMORY_MAX_DIST", "0.82")
)
_NAMED_SIGHTING_MEMORY: Dict[str, List[Dict[str, object]]] = {}
_NAMED_SIGHTING_MEMORY_LOCK = threading.Lock()
# Dashboard photo upload — cosine distance (lower = stricter).
# 0.50 ≈ live CCTV gate. 0.68 was letting weak looking-down CCTV faces through
# as false positives when a clear enrollment photo already matched strongly.
try:
    PHOTO_SEARCH_MATCH_THRESHOLD = min(
        0.85,
        max(0.20, float(os.environ.get("PHOTO_SEARCH_MATCH_THRESHOLD", "0.50"))),
    )
except ValueError:
    PHOTO_SEARCH_MATCH_THRESHOLD = 0.50
MULTI_MATCH_THRESHOLD = PHOTO_SEARCH_MATCH_THRESHOLD
# When the top hit is very strong, drop far weaker gallery noise.
PHOTO_SEARCH_STRONG_BEST = 0.25
PHOTO_SEARCH_STRONG_MAX_GAP = 0.20
BLURRY_SHARPNESS_THRESHOLD = 60.0
# Soft CCTV faces (laplacian below this) still try Named People with a looser gate
# and recent sighting memory — so enrolled names show even when the crop is soft.
try:
    NAMED_SOFT_FACE_LAPLACIAN = max(
        5.0,
        float(os.environ.get("NAMED_SOFT_FACE_LAPLACIAN", str(BLURRY_SHARPNESS_THRESHOLD))),
    )
except ValueError:
    NAMED_SOFT_FACE_LAPLACIAN = float(BLURRY_SHARPNESS_THRESHOLD)
try:
    NAMED_SOFT_MATCH_THRESHOLD = min(
        0.92,
        max(
            0.20,
            float(
                os.environ.get(
                    "NAMED_SOFT_MATCH_THRESHOLD",
                    str(NAMED_TRACK_MEMORY_MAX_DIST),
                )
            ),
        ),
    )
except ValueError:
    NAMED_SOFT_MATCH_THRESHOLD = float(NAMED_TRACK_MEMORY_MAX_DIST)
VIDEO_SEARCH_FRAME_STEP = max(
    1,
    int(os.environ.get("VIDEO_SEARCH_FRAME_STEP", "10")),
)
VIDEO_SEARCH_MAX_FRAMES = 600
VIDEO_SEARCH_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FOLDER_SCAN_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
# Target Trace Scan — balanced speed + match quality (override via env).
# Stride 12 @ 30fps ≈ 2.5 inspects/sec of video time; adaptive densifies on faces.
VIDEO_TARGET_SCAN_FRAME_STEP = max(
    1,
    int(os.environ.get("VIDEO_SCAN_FRAME_INTERVAL", "12")),
)
VIDEO_SCAN_BATCH_SIZE = max(
    1,
    int(os.environ.get("VIDEO_SCAN_BATCH_SIZE", "20")),
)  # GPU YOLO / decode batch width
VIDEO_SCAN_DETECT_SCALE = float(os.environ.get("VIDEO_SCAN_DETECT_SCALE", "0.50"))
VIDEO_SCAN_WIDTH = max(
    320,
    int(os.environ.get("VIDEO_SCAN_MAX_WIDTH", "640")),
)  # optional width cap after detect downscale
VIDEO_SCAN_YOLO_IMGSZ = 640
VIDEO_SCAN_ARCFACE_MIN_SIDE = 112  # upscale tiny crops only; native res otherwise
VIDEO_SCAN_LEGACY_LETTERBOX_HEIGHT = 480  # legacy photo lookup letterbox only
VIDEO_SCAN_YOLO_CONF = 0.15
VIDEO_SCAN_RETINAFACE_CONF = 0.28  # slightly lower → more hallway faces for matching
VIDEO_SCAN_RETINAFACE_DET_THRESH = 0.28
VIDEO_SCAN_DETECT_UPSCALE = float(os.environ.get("VIDEO_SCAN_DETECT_UPSCALE", "1.15"))
VIDEO_SCAN_VERBOSE = os.environ.get("VIDEO_SCAN_VERBOSE", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
VIDEO_SCAN_MAX_BLUR = 3.0  # Laplacian min for scan (CCTV overhead is soft)
VIDEO_SCAN_MIN_FACE_PX = 24
VIDEO_SCAN_ARCFACE_INPUT_SIZE = 112
VIDEO_SCAN_ENHANCE_MAX_SIDE = 64  # GFPGAN only for small soft crops
VIDEO_SCAN_ENHANCE_SHARP_MIN = 100.0
VIDEO_SCAN_SCORE_THRESHOLD = 0.45
VIDEO_SCAN_UPSCALE_MIN_SIDE = 480
VIDEO_SCAN_PERSON_UPSCALE_MIN_SIDE = 320
# Named / video target match floors (higher % = stricter / more accurate)
VIDEO_TARGET_MATCH_THRESHOLD = float(os.environ.get("VIDEO_TARGET_MATCH_THRESHOLD", "0.80"))
VIDEO_TARGET_CONFIRM_MATCH_PERCENT = int(os.environ.get("VIDEO_TARGET_CONFIRM_MATCH_PERCENT", "20"))
VIDEO_TARGET_BORDERLINE_MIN_PERCENT = int(os.environ.get("VIDEO_TARGET_BORDERLINE_MIN_PERCENT", "10"))
VIDEO_TARGET_BORDERLINE_MAX_PERCENT = int(os.environ.get("VIDEO_TARGET_BORDERLINE_MAX_PERCENT", "19"))
VIDEO_TARGET_NEAR_MISS_MIN_PERCENT = int(os.environ.get("VIDEO_TARGET_NEAR_MISS_MIN_PERCENT", "10"))
VIDEO_TARGET_NEAR_MISS_MAX_RESULTS = 10000
VIDEO_SCAN_ALERT_MATCH_PERCENT = int(os.environ.get("VIDEO_SCAN_ALERT_MATCH_PERCENT", "50"))
# Richer reference bank → better CCTV match % (studio photo ↔ hallway)
REFERENCE_AUG_ROTATIONS_DEG = (-15, -8, 0, 8, 15)
REFERENCE_AUG_BRIGHTNESS_FACTORS = (0.75, 0.9, 1.1, 1.25)
REFERENCE_AUG_BLUR_SIGMA = 1.0
# Named People enroll: try upright + sideways phone orientations, then small tilts.
NAMED_CARDINAL_ROTATIONS_DEG = (0, 90, 180, 270)
NAMED_TILT_ROTATIONS_DEG = (-15, -8, 8, 15)
NAMED_MAX_ENROLL_VARIANTS = int(os.environ.get("NAMED_MAX_ENROLL_VARIANTS", "12"))
VIDEO_TARGET_MERGE_GAP_SEC = 2.0
FOLDER_SCAN_PARALLEL_WORKERS = max(
    1,
    min(
        8,
        int(os.environ.get("FOLDER_SCAN_PARALLEL_WORKERS", "6")),
    ),
)
# TEMP debug — save sample JPEGs at probe frame indices
VIDEO_SCAN_DEBUG_FRAME_INDICES = frozenset({100, 500, 1000, 2000, 2500})
VIDEO_SCAN_DEBUG_DIR = PROJECT_DIR / "debug_frames"
# TEMP — embedding pipeline diagnostics (save crops + vector checks)
# Off by default — disk I/O was adding many seconds to scan duration.
VIDEO_SCAN_EMBED_DIAG = os.environ.get("VIDEO_SCAN_EMBED_DIAG", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
SCAN_EMBED_DEBUG_DIR = PROJECT_DIR / "debug_faces"
_SCAN_DIAG_REF: Dict[str, object] = {}
_SCAN_DIAG_CANDIDATES: List[Dict[str, object]] = []
_SCAN_DIAG_CANDIDATE_CAP = 25
TARGET_TRACK_BOX_BGR = (255, 120, 0)
TARGET_TRACK_BOX_THICKNESS = 4
TARGET_TRACK_LABEL_THICKNESS = 3
TARGET_TRACK_SNIPPET_PAD = 24
BLURRY_MATCH_MARGIN = 0.02
QUALITY_UPGRADE_RATIO = 1.20

_detector = None
_recognizer = None
_gpu_scan_analyzer_local = threading.local()
BEST_FACES_DIR = PROJECT_DIR / "captures" / "best_faces"
CAPTURES_DIR = PROJECT_DIR / "captures"


def is_auto_identity_name(name: Optional[str]) -> bool:
    """True for generated Stranger_/Visitor_ labels, not a human-assigned name."""
    n = str(name or "").strip()
    if not n:
        return True
    low = n.lower()
    return (
        low.startswith("stranger_")
        or low.startswith("passerby_")
        or low.startswith("visitor_")
        or low.startswith("unknown")
        or n in ("Person of Interest", "Unknown Visitor")
    )


def stranger_display_name(known_person_id: int) -> str:
    """Structured label for an unrecognized identity linked to a gallery id."""
    return f"Stranger_{int(known_person_id)}"


def visitor_display_name(
    *,
    person_id: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Structured label for an unknown / unverified visitor sighting."""
    if timestamp:
        compact = (
            str(timestamp)
            .strip()
            .replace("-", "")
            .replace(":", "")
            .replace("T", "_")
            .replace(" ", "_")
        )
        if compact:
            return f"Visitor_{compact}"
    if person_id is not None:
        return f"Visitor_{int(person_id)}"
    return f"Visitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_known_person_name_column(conn: sqlite3.Connection) -> None:
    """Add known_persons.name and backfill Stranger_{id} for legacy rows."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(known_persons)")}
    if "name" not in cols:
        conn.execute("ALTER TABLE known_persons ADD COLUMN name TEXT")
    # Only write when needed — unconditional UPDATE here locked the DB under
    # dashboard polling + multi-camera startup.
    needs_backfill = conn.execute(
        """
        SELECT 1 FROM known_persons
        WHERE name IS NULL OR TRIM(name) = ''
        LIMIT 1
        """
    ).fetchone()
    if needs_backfill is None:
        return
    conn.execute(
        """
        UPDATE known_persons
        SET name = 'Stranger_' || id
        WHERE name IS NULL OR TRIM(name) = ''
        """
    )


def ensure_watchlist_flag_column(conn: sqlite3.Connection) -> None:
    """
    Watchlist is an explicit flag, not “has a custom name”.

    That lets you label a captured stranger without Telegram / red POI alerts.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(known_persons)")}
    if "is_watchlist" not in cols:
        conn.execute(
            "ALTER TABLE known_persons ADD COLUMN is_watchlist INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute(
            """
            UPDATE known_persons
            SET is_watchlist = 1
            WHERE COALESCE(is_watchlist, 0) = 0
              AND name IS NOT NULL AND TRIM(name) != ''
              AND lower(name) NOT LIKE 'stranger_%'
              AND lower(name) NOT LIKE 'passerby_%'
              AND lower(name) NOT LIKE 'visitor_%'
              AND lower(name) NOT LIKE 'unknown%'
              AND name NOT IN ('Person of Interest', 'Unknown Visitor')
            """
        )


def ensure_blacklist_flag_column(conn: sqlite3.Connection) -> None:
    """Separate Blacklist enroll list (mutually exclusive with Watchlist)."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(known_persons)")}
    if "is_blacklist" not in cols:
        conn.execute(
            "ALTER TABLE known_persons ADD COLUMN is_blacklist INTEGER NOT NULL DEFAULT 0"
        )


def ensure_named_flag_column(conn: sqlite3.Connection) -> None:
    """People enrolled in the Named gallery (or watchlist) — used for live ID."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(known_persons)")}
    if "is_named" not in cols:
        conn.execute(
            "ALTER TABLE known_persons ADD COLUMN is_named INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute(
            """
            UPDATE known_persons
            SET is_named = 1
            WHERE COALESCE(is_named, 0) = 0
              AND name IS NOT NULL AND TRIM(name) != ''
              AND lower(name) NOT LIKE 'stranger_%'
              AND lower(name) NOT LIKE 'passerby_%'
              AND lower(name) NOT LIKE 'visitor_%'
              AND lower(name) NOT LIKE 'unknown%'
              AND name NOT IN ('Person of Interest', 'Unknown Visitor')
            """
        )
        try:
            conn.execute(
                """
                UPDATE known_persons
                SET is_named = 1
                WHERE COALESCE(is_watchlist, 0) = 1
                """
            )
        except sqlite3.Error:
            pass


def ensure_log_display_name_column(conn: sqlite3.Connection) -> None:
    """Optional per-capture label that does not enroll the person on the watchlist."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    if "display_name" not in cols:
        conn.execute("ALTER TABLE tracking_logs ADD COLUMN display_name TEXT")


def mark_person_watchlist(
    conn: sqlite3.Connection,
    known_person_id: int,
    *,
    watchlist: bool = True,
) -> None:
    ensure_watchlist_flag_column(conn)
    ensure_blacklist_flag_column(conn)
    kid = int(known_person_id)
    if watchlist:
        # Watchlist and Blacklist are separate lists.
        conn.execute(
            "UPDATE known_persons SET is_watchlist = 1, is_blacklist = 0 WHERE id = ?",
            (kid,),
        )
    else:
        conn.execute(
            "UPDATE known_persons SET is_watchlist = 0 WHERE id = ?",
            (kid,),
        )


def mark_person_blacklist(
    conn: sqlite3.Connection,
    known_person_id: int,
    *,
    blacklist: bool = True,
) -> None:
    ensure_watchlist_flag_column(conn)
    ensure_blacklist_flag_column(conn)
    kid = int(known_person_id)
    if blacklist:
        conn.execute(
            "UPDATE known_persons SET is_blacklist = 1, is_watchlist = 0 WHERE id = ?",
            (kid,),
        )
    else:
        conn.execute(
            "UPDATE known_persons SET is_blacklist = 0 WHERE id = ?",
            (kid,),
        )


def mark_person_named(
    conn: sqlite3.Connection,
    known_person_id: int,
    *,
    named: bool = True,
) -> None:
    ensure_named_flag_column(conn)
    conn.execute(
        "UPDATE known_persons SET is_named = ? WHERE id = ?",
        (1 if named else 0, int(known_person_id)),
    )


def delete_named_gallery_people(conn: sqlite3.Connection) -> Dict[str, int]:
    """
    Remove everyone enrolled only in the Named people gallery (not watchlist).

    Deletes face encodings, portrait files, and known_persons rows.
    """
    ensure_named_flag_column(conn)
    ensure_watchlist_flag_column(conn)
    rows = conn.execute(
        """
        SELECT id FROM known_persons
        WHERE COALESCE(is_named, 0) = 1
          AND COALESCE(is_watchlist, 0) = 0
        """
    ).fetchall()
    ids = [int(r[0] if not isinstance(r, sqlite3.Row) else r["id"]) for r in rows]
    if not ids:
        return {"deleted_people": 0, "deleted_images": 0}

    images_removed = 0
    for pid in ids:
        for row in conn.execute(
            "SELECT image_path FROM person_best_faces WHERE known_person_id = ?",
            (pid,),
        ).fetchall():
            raw = str(row[0] if not isinstance(row, sqlite3.Row) else row["image_path"] or "").strip()
            if raw:
                p = Path(raw)
                if not p.is_absolute():
                    p = PROJECT_DIR / raw.replace("\\", "/")
                try:
                    if p.is_file():
                        p.unlink()
                        images_removed += 1
                except OSError:
                    pass
        guess = BEST_FACES_DIR / f"person_{pid}_best.jpg"
        try:
            if guess.is_file():
                guess.unlink()
                images_removed += 1
        except OSError:
            pass
        try:
            from named_gallery import remove_person_gallery

            images_removed += remove_person_gallery(pid)
        except Exception:
            pass

    placeholders = ",".join("?" for _ in ids)
    conn.execute(
        f"DELETE FROM person_encodings WHERE known_person_id IN ({placeholders})",
        ids,
    )
    conn.execute(
        f"DELETE FROM person_best_faces WHERE known_person_id IN ({placeholders})",
        ids,
    )
    conn.execute(
        f"DELETE FROM known_persons WHERE id IN ({placeholders})",
        ids,
    )
    conn.commit()
    try:
        refresh_gpu_face_index(conn)
    except Exception:
        pass
    try:
        from watchlist_vector_db import notify_watchlist_changed, sync_watchlist_from_sqlite

        sync_watchlist_from_sqlite(conn, notify=True)
    except Exception:
        try:
            from watchlist_vector_db import notify_watchlist_changed

            notify_watchlist_changed()
        except Exception:
            pass
    return {"deleted_people": len(ids), "deleted_images": images_removed}


def _resolve_capture_image_path(raw_path: Optional[str]) -> Optional[Path]:
    """Map a tracking_logs capture_path to an on-disk JPEG."""
    if not raw_path:
        return None
    rel = str(raw_path).strip().replace("\\", "/").lstrip("/")
    if not rel:
        return None
    if rel.startswith("captures/"):
        rel = rel[len("captures/") :]
    candidates = [
        CAPTURES_DIR / Path(rel).name,
        CAPTURES_DIR / rel,
        PROJECT_DIR / rel,
        PROJECT_DIR / "captures" / Path(rel).name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _enroll_labeled_capture_face(
    conn: sqlite3.Connection,
    known_id: int,
    capture_path: Optional[str],
) -> bool:
    """
    Re-embed the labeled capture into Named People gallery so later CCTV
    sightings reuse the same ``known_person_id``.
    """
    path = _resolve_capture_image_path(capture_path)
    if path is None:
        return False
    img = load_bgr_image_oriented(path)
    if img is None or getattr(img, "size", 0) == 0:
        img = cv2.imread(str(path))
    if img is None or getattr(img, "size", 0) == 0:
        return False

    feature: Optional[np.ndarray] = None
    quality = float(face_sharpness(img))
    try:
        embedded = _embed_scan_reference_face(img)
        if embedded is not None:
            feature, meta = embedded
            try:
                if isinstance(meta, dict) and meta.get("quality") is not None:
                    quality = float(meta["quality"])
            except Exception:
                pass
    except Exception:
        feature = None
    if feature is None:
        try:
            variants = embed_named_enrollment_variants(img)
            if variants:
                feature, quality, _label = variants[0]
        except Exception:
            feature = None
    if feature is None:
        return False

    flat = np.asarray(feature, dtype=np.float32).reshape(-1)
    if int(flat.size) != 512:
        return False

    # Named rematch reads person_best_faces — force-update from this labeled crop.
    upsert_best_face_profile(
        conn,
        int(known_id),
        flat,
        img,
        float(quality) if np.isfinite(quality) else 1.0,
        allow_named=True,
    )
    try:
        add_encoding_for_person(conn, int(known_id), flat, quality)
    except Exception:
        pass
    try:
        from named_gallery import add_gallery_photo_from_array

        add_gallery_photo_from_array(int(known_id), img)
    except Exception:
        pass
    try:
        refresh_gpu_face_index(conn)
    except Exception:
        pass
    return True


def label_captured_person(
    conn: sqlite3.Connection,
    name: str,
    *,
    log_id: Optional[int] = None,
    known_person_id: Optional[int] = None,
) -> Dict[str, object]:
    """
    Save a human name on a capture and enroll that face for rematch.

    Without enrollment, the next CCTV sighting often mints a new stranger ID.
    This path keeps ``known_person_id`` stable across later catches (not watchlist).
    """
    display = str(name or "").strip()[:80]
    if not display:
        raise ValueError("Name is required")
    ensure_known_person_name_column(conn)
    ensure_watchlist_flag_column(conn)
    ensure_named_flag_column(conn)
    ensure_log_display_name_column(conn)

    kid: Optional[int] = None
    if known_person_id is not None and str(known_person_id).strip() != "":
        kid = int(known_person_id)
    lid: Optional[int] = None
    if log_id is not None and str(log_id).strip() != "":
        lid = int(log_id)

    capture_path: Optional[str] = None
    log_row = None
    if lid is not None:
        log_row = conn.execute(
            """
            SELECT id, known_person_id, capture_path, scene_path
            FROM tracking_logs WHERE id = ?
            """,
            (lid,),
        ).fetchone()
        if log_row is None:
            raise ValueError("Capture not found")
        conn.execute(
            "UPDATE tracking_logs SET display_name = ? WHERE id = ?",
            (display, lid),
        )
        if kid is None:
            raw = (
                log_row["known_person_id"]
                if isinstance(log_row, sqlite3.Row)
                else log_row[1]
            )
            if raw is not None and str(raw).strip() != "":
                kid = int(raw)
        capture_path = (
            log_row["capture_path"]
            if isinstance(log_row, sqlite3.Row)
            else log_row[2]
        )
        if not capture_path:
            capture_path = (
                log_row["scene_path"]
                if isinstance(log_row, sqlite3.Row)
                else log_row[3]
            )

    enrolled = False
    created = False

    # No gallery ID yet — create one from this capture so later sightings can reuse it.
    if kid is None:
        if not capture_path:
            raise ValueError("Need a capture with a face image to assign a lasting ID")
        path = _resolve_capture_image_path(str(capture_path))
        if path is None:
            raise ValueError("Capture image not found on disk")
        img = load_bgr_image_oriented(path)
        if img is None or getattr(img, "size", 0) == 0:
            img = cv2.imread(str(path))
        if img is None or getattr(img, "size", 0) == 0:
            raise ValueError("Could not read capture image")
        feature = None
        quality = float(face_sharpness(img))
        try:
            embedded = _embed_scan_reference_face(img)
            if embedded is not None:
                feature, meta = embedded
                try:
                    if isinstance(meta, dict) and meta.get("quality") is not None:
                        quality = float(meta["quality"])
                except Exception:
                    pass
        except Exception:
            feature = None
        if feature is None:
            raise ValueError("Could not read a face from this capture to assign an ID")
        kid = register_person(
            conn,
            feature,
            quality_score=quality,
            name=display,
            watchlist=False,
            named=True,
        )
        created = True
        enrolled = True
        if lid is not None:
            conn.execute(
                "UPDATE tracking_logs SET known_person_id = ?, display_name = ? WHERE id = ?",
                (kid, display, lid),
            )
        try:
            upsert_best_face_profile(
                conn,
                int(kid),
                np.asarray(feature, dtype=np.float32).reshape(-1),
                img,
                float(quality) if np.isfinite(quality) else 1.0,
                allow_named=True,
            )
        except Exception:
            pass
        try:
            from named_gallery import add_gallery_photo_from_array

            add_gallery_photo_from_array(int(kid), img)
        except Exception:
            pass

    if kid is not None and not created:
        exists = conn.execute(
            "SELECT id FROM known_persons WHERE id = ?",
            (kid,),
        ).fetchone()
        if exists is None:
            raise ValueError("Person not found")
        conn.execute(
            """
            UPDATE known_persons
            SET name = ?, is_named = 1, is_watchlist = 0
            WHERE id = ?
            """,
            (display, kid),
        )
        conn.execute(
            "UPDATE tracking_logs SET display_name = ? WHERE known_person_id = ?",
            (display, kid),
        )
        if lid is not None:
            conn.execute(
                "UPDATE tracking_logs SET known_person_id = ? WHERE id = ?",
                (kid, lid),
            )
        if capture_path:
            enrolled = _enroll_labeled_capture_face(conn, int(kid), str(capture_path))

    if kid is None and lid is None:
        raise ValueError("Need a capture id or person id")

    conn.commit()
    try:
        from captures import rename_log_captures_with_name

        rename_log_captures_with_name(
            conn,
            display,
            log_id=lid,
            known_person_id=kid,
        )
        conn.commit()
    except Exception:
        pass
    return {
        "name": display,
        "known_person_id": kid,
        "log_id": lid,
        "is_watchlist": False,
        "enrolled": bool(enrolled or created),
        "created": bool(created),
    }


def resolve_known_person_name(
    conn: sqlite3.Connection,
    known_person_id: Optional[int],
    *,
    person_id: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Registered gallery name, else Stranger_{id} / Visitor_{timestamp}."""
    if known_person_id is not None:
        kid = int(known_person_id)
        try:
            ensure_known_person_name_column(conn)
            row = conn.execute(
                "SELECT name FROM known_persons WHERE id = ?",
                (kid,),
            ).fetchone()
            if row is not None:
                registered = row["name"] if isinstance(row, sqlite3.Row) else row[0]
                if registered and str(registered).strip():
                    return str(registered).strip()
        except sqlite3.Error:
            pass
        return stranger_display_name(kid)
    return visitor_display_name(person_id=person_id, timestamp=timestamp)


def is_named_person(
    conn: sqlite3.Connection, known_person_id: Optional[int]
) -> bool:
    """True when this gallery id was enrolled with a real name (named box or watchlist)."""
    if known_person_id is None:
        return False
    kid = int(known_person_id)
    try:
        ensure_named_flag_column(conn)
        ensure_watchlist_flag_column(conn)
        row = conn.execute(
            "SELECT is_named, is_watchlist, name FROM known_persons WHERE id = ?",
            (kid,),
        ).fetchone()
        if row is None:
            return False
        named = row["is_named"] if isinstance(row, sqlite3.Row) else row[0]
        watchlist = row["is_watchlist"] if isinstance(row, sqlite3.Row) else row[1]
        if named is not None and int(named) == 1:
            return True
        if watchlist is not None and int(watchlist) == 1:
            return True
        name = row["name"] if isinstance(row, sqlite3.Row) else row[2]
        return not is_auto_identity_name(name)
    except Exception:
        try:
            return is_enrolled_watchlist_person(conn, kid)
        except Exception:
            return False


def capture_identity_label(
    conn: sqlite3.Connection,
    known_person_id: Optional[int],
    *,
    person_id: Optional[int] = None,
    match_distance: Optional[float] = None,
    profile: bool = False,
) -> Tuple[str, bool]:
    """
    Live-capture display name.

    Returns ``(name, is_stranger)``. Named / watchlist enrollments keep their
    name only when the face match is strong enough; weak nearest-neighbor hits
    stay ``Stranger``.
    """
    if known_person_id is None:
        return "Stranger", True
    is_named = is_named_person(conn, known_person_id)
    is_watchlist = is_enrolled_watchlist_person(conn, known_person_id)
    if not is_named and not is_watchlist:
        return "Stranger", True
    if match_distance is not None:
        if not np.isfinite(float(match_distance)):
            return "Stranger", True
        dist = float(match_distance)
        gate = (
            float(NAMED_MATCH_THRESHOLD_PROFILE)
            if profile and is_named
            else float(NAMED_MATCH_THRESHOLD)
        )
        min_pct = (
            int(NAMED_LABEL_MIN_PERCENT_PROFILE)
            if profile and is_named
            else int(NAMED_LABEL_MIN_PERCENT)
        )
        if is_named and dist > gate:
            return "Stranger", True
        if confidence_score_from_distance(dist) < min_pct:
            return "Stranger", True
    return resolve_known_person_name(conn, int(known_person_id), person_id=person_id), False


def is_confident_named_match(
    match_distance: Optional[float],
    *,
    profile: bool = False,
) -> bool:
    """True when a Named People gallery hit is strong enough to label a live capture."""
    if match_distance is None or not np.isfinite(float(match_distance)):
        return False
    gate = (
        float(NAMED_MATCH_THRESHOLD_PROFILE)
        if profile
        else float(NAMED_MATCH_THRESHOLD)
    )
    min_pct = (
        int(NAMED_LABEL_MIN_PERCENT_PROFILE)
        if profile
        else int(NAMED_LABEL_MIN_PERCENT)
    )
    dist = float(match_distance)
    if dist > gate:
        return False
    return confidence_score_from_distance(dist) >= min_pct


def _prune_named_sighting_memory(camera_location: str, now: Optional[float] = None) -> None:
    ts = float(now if now is not None else time.time())
    ttl = float(NAMED_SIGHTING_MEMORY_SEC)
    with _NAMED_SIGHTING_MEMORY_LOCK:
        rows = _NAMED_SIGHTING_MEMORY.get(camera_location) or []
        kept = [row for row in rows if ts - float(row.get("time", 0.0)) <= ttl]
        if kept:
            _NAMED_SIGHTING_MEMORY[camera_location] = kept[-8:]
        elif camera_location in _NAMED_SIGHTING_MEMORY:
            del _NAMED_SIGHTING_MEMORY[camera_location]


def remember_confident_named_sighting(
    camera_location: Optional[str],
    named_id: int,
    display_name: str,
    feature: np.ndarray,
    match_distance: float,
) -> None:
    """Cache a recent frontal/named hit so the same person keeps their name in profile."""
    if not camera_location:
        return
    feat = _normalize_face_feature(feature)
    if feat is None:
        return
    now = time.time()
    row = {
        "time": now,
        "named_id": int(named_id),
        "name": str(display_name),
        "feature": feat,
        "distance": float(match_distance),
    }
    with _NAMED_SIGHTING_MEMORY_LOCK:
        rows = list(_NAMED_SIGHTING_MEMORY.get(camera_location) or [])
        rows = [
            item
            for item in rows
            if not (
                int(item.get("named_id", -1)) == int(named_id)
                and now - float(item.get("time", 0.0)) <= float(NAMED_SIGHTING_MEMORY_SEC)
            )
        ]
        rows.append(row)
        _NAMED_SIGHTING_MEMORY[camera_location] = rows[-8:]
    _prune_named_sighting_memory(camera_location, now)


def _lookup_named_from_recent_sighting(
    camera_location: Optional[str],
    feature: np.ndarray,
    conn: sqlite3.Connection,
    *,
    profile: bool,
    soft: bool = False,
) -> Tuple[Optional[int], Optional[str], float]:
    """Reuse a recent named ID on this camera when gallery match is weak (side/soft)."""
    if not camera_location or not (profile or soft):
        return None, None, float("inf")
    feat = _normalize_face_feature(feature)
    if feat is None:
        return None, None, float("inf")
    now = time.time()
    _prune_named_sighting_memory(camera_location, now)
    with _NAMED_SIGHTING_MEMORY_LOCK:
        rows = list(_NAMED_SIGHTING_MEMORY.get(camera_location) or [])
    if not rows:
        return None, None, float("inf")
    best_id: Optional[int] = None
    best_name: Optional[str] = None
    best_dist = float("inf")
    mem_gate = float(NAMED_TRACK_MEMORY_MAX_DIST)
    for row in reversed(rows):
        try:
            sight_id = int(row["named_id"])
            sight_feat = row["feature"]
            if not isinstance(sight_feat, np.ndarray):
                continue
            dist = float(_arcface_embedding_distance(feat, sight_feat))
        except (KeyError, TypeError, ValueError):
            continue
        if not is_named_person(conn, sight_id):
            continue
        if dist <= mem_gate and dist < best_dist:
            best_id = sight_id
            best_name = str(row.get("name") or resolve_known_person_name(conn, sight_id))
            best_dist = dist
    if best_id is None:
        return None, None, float("inf")
    return best_id, best_name, best_dist


def resolve_live_capture_identity(
    conn: sqlite3.Connection,
    feature: np.ndarray,
    *,
    known_person_id: Optional[int] = None,
    person_id: Optional[int] = None,
    face_bgr: Optional[np.ndarray] = None,
    camera_location: Optional[str] = None,
) -> Tuple[str, bool, Optional[int], float]:
    """
    Compare a live face embed against Named People.

    Returns ``(display_name, is_stranger, matched_named_id, match_distance)``.
    Soft / slightly blurry crops still resolve to the enrolled Named People name
    when the gallery (or recent sighting) match clears the soft gate.
    """
    profile = False
    soft = False
    looking_down = False
    if face_bgr is not None and getattr(face_bgr, "size", 0) > 0:
        try:
            from face_print_filter import (
                is_likely_invalid_face_crop,
                is_likely_looking_down,
                is_likely_profile_view,
                is_likely_rear_view,
            )

            rear, _ = is_likely_rear_view(face_bgr)
            if rear:
                return "Stranger", True, None, float("inf")
            bad, _ = is_likely_invalid_face_crop(face_bgr)
            if bad:
                return "Stranger", True, None, float("inf")
            profile, _ = is_likely_profile_view(face_bgr)
            looking_down, _ = is_likely_looking_down(face_bgr)
        except Exception:
            pass
        try:
            soft = float(face_sharpness(face_bgr)) < float(NAMED_SOFT_FACE_LAPLACIAN)
        except Exception:
            soft = False
    gate = float(NAMED_MATCH_THRESHOLD_PROFILE if profile else NAMED_MATCH_THRESHOLD)
    if soft and not profile:
        gate = max(gate, float(NAMED_SOFT_MATCH_THRESHOLD))
    named_id, named_dist = _find_named_gallery_match(feature, conn, max_distance=gate)
    # Looking-down hallway plates look alike — 20% nearest-neighbor jumps
    # between enrolled men. Require a clearly stronger hit to keep a name.
    if looking_down:
        look_ok = (
            named_id is not None
            and np.isfinite(float(named_dist))
            and float(named_dist) <= 0.60
            and confidence_score_from_distance(float(named_dist)) >= 40
        )
        if not look_ok:
            return "Stranger", True, None, float(
                named_dist if np.isfinite(float(named_dist)) else float("inf")
            )
    if named_id is not None and (
        is_confident_named_match(named_dist, profile=profile)
        or (
            soft
            and np.isfinite(float(named_dist))
            and float(named_dist) <= float(NAMED_SOFT_MATCH_THRESHOLD)
            and confidence_score_from_distance(float(named_dist))
            >= int(NAMED_LABEL_MIN_PERCENT)
        )
    ):
        label, is_stranger = capture_identity_label(
            conn,
            int(named_id),
            person_id=person_id,
            match_distance=named_dist,
            profile=profile,
        )
        if not is_stranger:
            remember_confident_named_sighting(
                camera_location,
                int(named_id),
                label,
                feature,
                float(named_dist),
            )
            return label, False, int(named_id), float(named_dist)
    mem_id, mem_name, mem_dist = _lookup_named_from_recent_sighting(
        camera_location,
        feature,
        conn,
        profile=profile,
        soft=soft,
    )
    if mem_id is not None and mem_name:
        label, is_stranger = capture_identity_label(
            conn,
            int(mem_id),
            person_id=person_id,
        )
        if not is_stranger:
            return label, False, int(mem_id), float(mem_dist)
    if known_person_id is not None:
        # Track sticky IDs must still match this crop — otherwise a later
        # stranger/soft face on a reused ByteTrack ID inherits the prior name.
        sticky_id = int(known_person_id)
        sticky_dist = float("inf")
        try:
            for pid, dist in _score_named_person_encodings_only(feature, conn):
                if int(pid) == sticky_id:
                    sticky_dist = float(dist)
                    break
        except Exception:
            sticky_dist = float("inf")
        sticky_ok = (
            np.isfinite(sticky_dist)
            and float(sticky_dist) <= float(NAMED_TRACK_MEMORY_MAX_DIST)
            and confidence_score_from_distance(float(sticky_dist))
            >= int(
                NAMED_LABEL_MIN_PERCENT_PROFILE
                if profile
                else NAMED_LABEL_MIN_PERCENT
            )
        )
        if sticky_ok:
            label, is_stranger = capture_identity_label(
                conn,
                sticky_id,
                person_id=person_id,
                match_distance=sticky_dist,
                profile=profile,
            )
            if not is_stranger:
                return label, False, sticky_id, float(sticky_dist)
    return "Stranger", True, None, float(named_dist if np.isfinite(named_dist) else float("inf"))


def is_enrolled_watchlist_person(
    conn: sqlite3.Connection, known_person_id: Optional[int]
) -> bool:
    """True only for people enrolled onto the Watchlist (not Blacklist / labels)."""
    if known_person_id is None:
        return False
    kid = int(known_person_id)
    try:
        ensure_watchlist_flag_column(conn)
        ensure_blacklist_flag_column(conn)
        row = conn.execute(
            """
            SELECT is_watchlist, is_blacklist, name
            FROM known_persons WHERE id = ?
            """,
            (kid,),
        ).fetchone()
        if row is None:
            return False
        if isinstance(row, sqlite3.Row):
            if int(row["is_blacklist"] or 0) == 1:
                return False
            if row["is_watchlist"] is not None:
                return int(row["is_watchlist"]) == 1
            name = row["name"]
        else:
            if int(row[1] or 0) == 1:
                return False
            if row[0] is not None:
                return int(row[0]) == 1
            name = row[2]
        return not is_auto_identity_name(name)
    except Exception:
        try:
            return not is_auto_identity_name(
                resolve_known_person_name(conn, kid)
            )
        except Exception:
            return False


def is_enrolled_blacklist_person(
    conn: sqlite3.Connection, known_person_id: Optional[int]
) -> bool:
    """True only for people enrolled onto the Blacklist."""
    if known_person_id is None:
        return False
    kid = int(known_person_id)
    try:
        ensure_blacklist_flag_column(conn)
        row = conn.execute(
            "SELECT is_blacklist FROM known_persons WHERE id = ?",
            (kid,),
        ).fetchone()
        if row is None:
            return False
        flag = row["is_blacklist"] if isinstance(row, sqlite3.Row) else row[0]
        return int(flag or 0) == 1
    except Exception:
        return False


def is_enrolled_alert_person(
    conn: sqlite3.Connection, known_person_id: Optional[int]
) -> bool:
    """Watchlist or Blacklist enrollment (either list gets match priority)."""
    return is_enrolled_watchlist_person(conn, known_person_id) or is_enrolled_blacklist_person(
        conn, known_person_id
    )


def _pick_match_preferring_enrolled(
    candidates: Sequence[Tuple[int, float]],
    conn: sqlite3.Connection,
    threshold: float,
) -> Tuple[Optional[int], float]:
    """
    Prefer enrolled Watchlist / Blacklist IDs over auto Stranger_* gallery noise.

    Auto-captures flood FAISS/Qdrant; without this preference, enrolled people
    rarely win and CCTV list alerts never fire.
    """
    under: List[Tuple[int, float]] = []
    for pid, dist in candidates:
        try:
            d = float(dist)
            i = int(pid)
        except (TypeError, ValueError):
            continue
        if d <= float(threshold):
            under.append((i, d))
    if not under:
        if not candidates:
            return None, float("inf")
        try:
            best_id, best_dist = min(candidates, key=lambda x: float(x[1]))
            return int(best_id), float(best_dist)
        except Exception:
            return None, float("inf")

    enrolled = [
        (i, d)
        for i, d in under
        if is_named_person(conn, i) or is_enrolled_alert_person(conn, i)
    ]
    pool = enrolled if enrolled else under
    pool.sort(key=lambda x: x[1])
    return pool[0][0], pool[0][1]


def _pick_best_named_person(
    candidates: Sequence[Tuple[int, float]],
    conn: sqlite3.Connection,
    *,
    max_distance: float,
) -> Tuple[Optional[int], float]:
    """Best named-gallery hit under a relaxed distance (enrollment photo vs CCTV)."""
    hits: List[Tuple[int, float]] = []
    for pid, dist in candidates:
        try:
            d = float(dist)
            i = int(pid)
        except (TypeError, ValueError):
            continue
        if d <= float(max_distance) and is_named_person(conn, i):
            hits.append((i, d))
    if not hits:
        return None, float("inf")
    hits.sort(key=lambda x: x[1])
    return hits[0][0], hits[0][1]


def _finalize_face_match(
    candidates: Sequence[Tuple[int, float]],
    conn: sqlite3.Connection,
    threshold: float,
    *,
    log_candidates: int = 0,
) -> Tuple[Optional[int], float]:
    """Strict watchlist / stranger match (named gallery handled earlier)."""
    best_id, best_dist = _pick_match_preferring_enrolled(candidates, conn, threshold)
    if best_id is not None and best_dist <= threshold:
        return int(best_id), float(best_dist)
    return None, float(best_dist if best_id is not None else float("inf"))


# Per-process: schema init once per DB file (dashboard was re-running DDL+UPDATE
# on every API poll and colliding with camera workers).
_IDENTITY_SCHEMA_READY: Set[str] = set()
_IDENTITY_SCHEMA_LOCK = threading.Lock()


def _sqlite_file_key(conn: sqlite3.Connection) -> str:
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
        if row is None:
            return ""
        path = row["file"] if isinstance(row, sqlite3.Row) else row[2]
        return str(path or "").strip().lower()
    except sqlite3.Error:
        return ""


def init_identity_db(conn: sqlite3.Connection) -> None:
    db_key = _sqlite_file_key(conn)
    if db_key:
        with _IDENTITY_SCHEMA_LOCK:
            if db_key in _IDENTITY_SCHEMA_READY:
                return

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS known_persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            name TEXT
        )
        """
    )
    ensure_known_person_name_column(conn)
    ensure_watchlist_flag_column(conn)
    ensure_blacklist_flag_column(conn)
    ensure_named_flag_column(conn)
    ensure_log_display_name_column(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS person_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            known_person_id INTEGER NOT NULL,
            encoding BLOB NOT NULL,
            FOREIGN KEY (known_person_id) REFERENCES known_persons(id)
        )
        """
    )
    cols_enc = {row[1] for row in conn.execute("PRAGMA table_info(person_encodings)")}
    if "quality_score" not in cols_enc:
        conn.execute("ALTER TABLE person_encodings ADD COLUMN quality_score REAL")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS person_body_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            known_person_id INTEGER NOT NULL,
            encoding BLOB NOT NULL,
            FOREIGN KEY (known_person_id) REFERENCES known_persons(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS person_best_faces (
            known_person_id INTEGER PRIMARY KEY,
            encoding BLOB NOT NULL,
            quality_score REAL NOT NULL,
            image_path TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (known_person_id) REFERENCES known_persons(id)
        )
        """
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    if "known_person_id" not in cols:
        conn.execute(
            "ALTER TABLE tracking_logs ADD COLUMN known_person_id INTEGER"
        )
    # မြန်မာ: gray-zone audit table — Tier 2 ဖြစ်ရပ်များကို နောက်မှ စစ်ဆေးရန်
    from orion_frame_pipeline import init_gray_zone_audit_db

    init_gray_zone_audit_db(conn)
    conn.commit()
    if db_key:
        with _IDENTITY_SCHEMA_LOCK:
            _IDENTITY_SCHEMA_READY.add(db_key)

def models_ready() -> bool:
    return YUNET_PATH.is_file() and SFACE_PATH.is_file()


def gpu_scan_ready() -> bool:
    """True when InsightFace CUDA stack is available for video target scanning."""
    try:
        from inference_device import torch_cuda_active
        from face_analyzer import _try_load_insightface_cuda

        if not torch_cuda_active():
            return False
        return _try_load_insightface_cuda() is not None
    except Exception:
        return False


def scan_models_ready() -> bool:
    """Video scan can run on GPU (InsightFace) or CPU fallback (YuNet+SFace)."""
    return gpu_scan_ready() or models_ready()


def _get_gpu_scan_analyzer():
    """Per-thread FaceAnalyzer for parallel folder scans (own InsightFace ORT sessions)."""
    cached = getattr(_gpu_scan_analyzer_local, "analyzer", None)
    if cached is not None:
        return cached
    if not gpu_scan_ready():
        return None
    from face_analyzer import FaceAnalyzer, SCAN_MIN_FACE_PX

    analyzer = FaceAnalyzer(
        min_size=SCAN_MIN_FACE_PX,
        max_blur=18.0,
        max_yaw=0.55,
        detect_threshold=VIDEO_SCAN_RETINAFACE_CONF,
        isolated_insightface=True,
    )
    _gpu_scan_analyzer_local.analyzer = analyzer
    return analyzer


def _arcface_embedding_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    """Cosine distance for L2-normalized ArcFace vectors (InsightFace path)."""
    left = _normalize_face_feature(f1)
    right = _normalize_face_feature(f2)
    if left is None or right is None:
        return float("inf")
    return cosine_distance(left.reshape(-1), right.reshape(-1))


def _scan_embedding_distance(
    f1: np.ndarray, f2: np.ndarray, *, gpu: bool = False
) -> float:
    """Cosine distance on L2-normalized embeddings (lower = stronger match).

    f1 may be a single (D,) vector or an (N, D) reference bank — uses the
    best (minimum) distance across all stored reference variants.
    """
    ref = np.asarray(f1, dtype=np.float32)
    cmp_feat = np.asarray(f2, dtype=np.float32).reshape(-1)

    if ref.ndim == 2:
        if ref.shape[0] == 0:
            return float("inf")
        best = float("inf")
        for row in ref:
            d = _scan_embedding_distance(row, cmp_feat, gpu=gpu)
            if d < best:
                best = d
        return best

    if gpu:
        return _arcface_embedding_distance(ref, cmp_feat)
    return _distance(ref, cmp_feat)


def _yolo_scan_predict_kwargs(
    frames: List[np.ndarray], *, for_scan: bool = False
) -> dict:
    """Shared ultralytics kwargs for single- or multi-frame person detect."""
    from inference_device import yolo_device_arg

    kwargs: dict = {
        "classes": [0],
        "verbose": False,
        "device": yolo_device_arg(),
    }
    if for_scan:
        kwargs["conf"] = VIDEO_SCAN_YOLO_CONF
        kwargs["imgsz"] = VIDEO_SCAN_YOLO_IMGSZ
    return kwargs


def _yolo_predict_persons(yolo_model, frame: np.ndarray, *, for_scan: bool = False):
    """Run YOLO person detection on the configured torch device."""
    from inference_device import cuda_stream_context

    with cuda_stream_context():
        return yolo_model(
            frame, **_yolo_scan_predict_kwargs([frame], for_scan=for_scan)
        )


def _yolo_predict_persons_batch(
    yolo_model, frames: List[np.ndarray], *, for_scan: bool = False
):
    """Batch YOLO person detection — one GPU kernel for N scan frames."""
    from inference_device import cuda_stream_context

    if not frames:
        return []
    with cuda_stream_context():
        if len(frames) == 1:
            return yolo_model(
                frames[0], **_yolo_scan_predict_kwargs(frames, for_scan=for_scan)
            )
        return yolo_model(
            frames, **_yolo_scan_predict_kwargs(frames, for_scan=for_scan)
        )


def _encoding_from_bgr_gpu(
    img: np.ndarray, yolo_model=None
) -> Optional[np.ndarray]:
    """Extract ArcFace embedding via InsightFace CUDA (RetinaFace detect)."""
    analyzer = _get_gpu_scan_analyzer()
    if analyzer is None or img is None or img.size == 0:
        return None

    dets = analyzer.detect_and_embed_scan(img)
    usable = [d for d in dets if d.quality_ok and d.embedding is not None]
    if usable:
        best = max(
            usable,
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
        )
        return best.embedding

    if yolo_model is None:
        return None

    from orion_frame_pipeline import wrap_yolo_results

    results = _yolo_predict_persons(yolo_model, img)
    first = wrap_yolo_results(results)
    if first is None or first.boxes is None or len(first.boxes) == 0:
        return None

    xyxy = first.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    for bbox in xyxy[areas.argsort()[::-1]]:
        from face_analyzer import extract_face_from_person_crop

        hit = extract_face_from_person_crop(
            img, bbox, analyzer=analyzer, source_frame=img
        )
        if hit is not None:
            return hit[0]
    return None


def load_face_models() -> Tuple[object, object]:
    global _detector, _recognizer
    if _detector is not None and _recognizer is not None:
        return _detector, _recognizer
    if not models_ready():
        raise FileNotFoundError(
            f"Face models missing in {MODELS_DIR}\n"
            "Run: python download_face_models.py --no-proxy"
        )
    _detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320))
    _detector.setScoreThreshold(float(FACE_DETECT_MIN_CONFIDENCE))
    _recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    return _detector, _recognizer


def _yunet_landmarks_dict(face: np.ndarray) -> Dict[str, Tuple[float, float]]:
    return yunet_landmarks_dict(face)


def _yunet_face_structure_ok(
    face: np.ndarray,
    *,
    min_score: float = FACE_DETECT_MIN_CONFIDENCE,
) -> bool:
    """Reject low-confidence, tiny, or non-frontal YuNet boxes (backs, shoulders, hair)."""
    ok, _reason = verify_yunet_face_for_capture(
        face,
        min_confidence=min_score,
    )
    return ok


def _detect_largest_face(
    image: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (aligned_face, raw_feature) or None."""
    faces = _yunet_extract_faces(image)
    if not faces:
        return None
    _aligned, feat, _box = faces[0]
    return _aligned, feat


def _yunet_extract_faces(
    image: np.ndarray,
    *,
    score_threshold: Optional[float] = None,
    upscale_min_side: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]]:
    """
    Detect every YuNet face on a BGR frame.
    Returns (aligned_face, feature, face_box) sorted largest-first.
    """
    if image is None or image.size == 0 or not models_ready():
        return []

    detector, recognizer = load_face_models()
    work = image
    scale_x = scale_y = 1.0
    if upscale_min_side > 0:
        h, w = image.shape[:2]
        if max(h, w) < upscale_min_side:
            scale = upscale_min_side / max(h, w)
            work = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
            )
            scale_x = w / float(work.shape[1])
            scale_y = h / float(work.shape[0])

    prev_threshold: Optional[float] = None
    min_score = (
        float(score_threshold)
        if score_threshold is not None
        else float(FACE_DETECT_MIN_CONFIDENCE)
    )
    if score_threshold is not None:
        prev_threshold = float(detector.getScoreThreshold())
        detector.setScoreThreshold(min_score)

    found: List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]] = []
    try:
        h, w = work.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(work)
        if faces is None or len(faces) == 0:
            return found
        for face in sorted(faces, key=lambda f: f[2] * f[3], reverse=True):
            if not _yunet_face_structure_ok(face, min_score=min_score):
                continue
            aligned = recognizer.alignCrop(work, face)
            feature = recognizer.feature(aligned)
            x, y, fw, fh = [float(v) for v in face[:4]]
            box = (
                int(x * scale_x),
                int(y * scale_y),
                int((x + fw) * scale_x),
                int((y + fh) * scale_y),
            )
            found.append((aligned, feature, box))
    finally:
        if prev_threshold is not None:
            detector.setScoreThreshold(prev_threshold)
    return found


def _scan_arcface_embed(
    frame: np.ndarray,
    face_box: Tuple[int, int, int, int],
    *,
    landmarks: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[Tuple[np.ndarray, Dict[str, object]]]:
    """
    Scan/reference ArcFace path: crop → optional restore → 5-point align → embed.

    Alignment uses RetinaFace landmarks; enhancement runs only on small (<80px) crops.
    """
    analyzer = _get_gpu_scan_analyzer()
    if analyzer is None or frame is None or frame.size == 0:
        return None

    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = face_box
    pad = int(0.15 * max(x2 - x1, y2 - y1))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(fw, x2 + pad), min(fh, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop_h, crop_w = crop.shape[:2]
    local_lm: Optional[Dict[str, Tuple[float, float]]] = None
    if landmarks:
        local_lm = {
            name: (float(pt[0]) - x1, float(pt[1]) - y1)
            for name, pt in landmarks.items()
        }

    from face_restoration import restore_for_scan, should_enhance_for_scan

    embed_crop = crop
    enhanced = False
    if should_enhance_for_scan(
        crop,
        max_side=VIDEO_SCAN_ENHANCE_MAX_SIDE,
        sharp_min=VIDEO_SCAN_ENHANCE_SHARP_MIN,
    ):
        embed_crop, enhanced = restore_for_scan(crop, force=True)

    if enhanced or local_lm is None or len(local_lm) < 5:
        best_det = _best_scan_detection_with_landmarks(analyzer, embed_crop)
        if best_det is not None and best_det.landmarks:
            local_lm = dict(best_det.landmarks)

    embed_debug: Optional[Dict[str, object]] = (
        {} if VIDEO_SCAN_EMBED_DIAG else None
    )
    if embed_debug is not None:
        embed_debug["raw_crop"] = crop.copy()
        embed_debug["enhanced_crop"] = (
            embed_crop.copy() if enhanced else None
        )
        embed_debug["enhanced_applied"] = enhanced

    feat = analyzer.embed_crop_recognition_only(
        embed_crop,
        landmarks=local_lm,
        align_size=VIDEO_SCAN_ARCFACE_INPUT_SIZE,
        debug_out=embed_debug,
    )
    if feat is None:
        return None

    meta: Dict[str, object] = {
        "crop_w": int(crop_w),
        "crop_h": int(crop_h),
        "enhanced": bool(enhanced),
        "aligned": local_lm is not None and len(local_lm) >= 5,
    }
    if embed_debug is not None:
        meta["debug"] = embed_debug
    return feat, meta


def _best_scan_detection_with_landmarks(analyzer, frame_bgr: np.ndarray):
    """Largest RetinaFace hit that has usable 5-point landmarks."""
    best = None
    best_area = 0
    for det in analyzer.detect(frame_bgr):
        if not det.landmarks or len(det.landmarks) < 5:
            continue
        x1, y1, x2, y2 = det.bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = det
    return best


def _embed_scan_reference_face(
    img: np.ndarray,
) -> Optional[Tuple[np.ndarray, Dict[str, object]]]:
    """Reference photo embedding — same align/enhance/ArcFace path as video scan."""
    analyzer = _get_gpu_scan_analyzer()
    if analyzer is None or img is None or img.size == 0:
        return None

    best = _best_scan_detection_with_landmarks(analyzer, img)
    if best is None:
        for det in sorted(
            analyzer.detect(img),
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
            reverse=True,
        ):
            if det.landmarks:
                best = det
                break
    if best is None:
        return None

    return _scan_arcface_embed(img, best.bbox, landmarks=best.landmarks)


def load_bgr_image_oriented(path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load an enrollment photo with EXIF orientation applied.

    ``cv2.imread`` ignores EXIF, so sideways phone photos stay rotated and
    Named People matching fails against upright CCTV faces.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        from PIL import Image, ImageOps

        with Image.open(file_path) as pil_img:
            oriented = ImageOps.exif_transpose(pil_img)
            rgb = np.asarray(oriented.convert("RGB"))
        if rgb.ndim != 3 or rgb.size == 0:
            return None
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(str(file_path))
        if img is None or img.size == 0:
            return None
        return img


def apply_exif_orientation_bgr(img: np.ndarray, path: Optional[Union[str, Path]] = None) -> np.ndarray:
    """Prefer reloading from ``path`` with EXIF; otherwise return ``img`` unchanged."""
    if path is not None:
        oriented = load_bgr_image_oriented(path)
        if oriented is not None:
            return oriented
    return np.ascontiguousarray(img)


def embed_named_enrollment_variants(
    img: np.ndarray,
) -> List[Tuple[np.ndarray, float, str]]:
    """
    Embed one Named People photo under upright + rotated variants.

    Returns ``(feature, quality, label)`` rows. Label ``original`` / ``rot0`` is
    preferred for the roster best-face thumbnail.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return []
    results: List[Tuple[np.ndarray, float, str]] = []
    seen: set = set()

    def _quality(meta: Dict[str, object], crop: np.ndarray) -> float:
        try:
            if isinstance(meta, dict):
                q = meta.get("quality")
                if q is not None and np.isfinite(float(q)):
                    return float(q)
        except Exception:
            pass
        try:
            return float(face_sharpness(crop))
        except Exception:
            return 0.0

    for card in NAMED_CARDINAL_ROTATIONS_DEG:
        base = img if int(card) == 0 else _rotate_bgr_image(img, float(card))
        packed = _embed_scan_reference_face(base)
        if packed is None:
            continue
        feat, meta = packed
        norm = _normalize_face_feature(feat)
        if norm is None:
            continue
        label = "original" if int(card) == 0 else f"rot{int(card)}"
        if label not in seen:
            seen.add(label)
            results.append((norm, _quality(meta if isinstance(meta, dict) else {}, base), label))
        for tilt in NAMED_TILT_ROTATIONS_DEG:
            tilted = _rotate_bgr_image(base, float(tilt))
            packed2 = _embed_scan_reference_face(tilted)
            if packed2 is None:
                continue
            feat2, meta2 = packed2
            norm2 = _normalize_face_feature(feat2)
            if norm2 is None:
                continue
            tlabel = f"rot{int(card)}_tilt{int(tilt):+d}"
            if tlabel in seen:
                continue
            seen.add(tlabel)
            results.append(
                (norm2, _quality(meta2 if isinstance(meta2, dict) else {}, tilted), tlabel)
            )
            if len(results) >= NAMED_MAX_ENROLL_VARIANTS:
                return results
        if len(results) >= NAMED_MAX_ENROLL_VARIANTS:
            break
    return results


def store_named_enrollment_embeddings(
    conn: sqlite3.Connection,
    person_id: int,
    img: np.ndarray,
    *,
    replace_existing: bool = False,
) -> int:
    """
    Write rotation-aware Named People embeddings for one enrollment photo.

    Keeps the upright/original variant as the best-face profile when possible.
    """
    variants = embed_named_enrollment_variants(img)
    if not variants:
        return 0
    pid = int(person_id)
    if replace_existing:
        conn.execute("DELETE FROM person_encodings WHERE known_person_id = ?", (pid,))
    added = 0
    primary = None
    for feat, quality, label in variants:
        if primary is None and label in ("original", "rot0"):
            primary = (feat, quality)
        add_encoding_for_person(conn, pid, feat, quality)
        added += 1
    if primary is None:
        primary = (variants[0][0], variants[0][1])
    try:
        upsert_best_face_profile(
            conn,
            pid,
            primary[0],
            img,
            float(primary[1] or 0.0),
            allow_named=True,
        )
    except TypeError:
        # Older signature during hot reload — fall back.
        upsert_best_face_profile(conn, pid, primary[0], img, float(primary[1] or 0.0))
    except Exception:
        _seed_best_face_if_missing(conn, pid, primary[0], primary[1])
    return added



def _rotate_bgr_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _adjust_bgr_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    scaled = img.astype(np.float32) * float(factor)
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _blur_bgr_image(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _iter_reference_augmentations(
    img: np.ndarray,
) -> List[Tuple[str, np.ndarray]]:
    """Build ordered reference variants for augmented embedding bank."""
    variants: List[Tuple[str, np.ndarray]] = [("original", img)]
    for deg in REFERENCE_AUG_ROTATIONS_DEG:
        variants.append((f"rotate_{deg:+d}", _rotate_bgr_image(img, deg)))
    for factor in REFERENCE_AUG_BRIGHTNESS_FACTORS:
        tag = f"brightness_{factor:.2f}".replace(".", "_")
        variants.append((tag, _adjust_bgr_brightness(img, factor)))
    variants.append(("flip_h", cv2.flip(img, 1)))
    variants.append(
        (f"blur_sigma_{REFERENCE_AUG_BLUR_SIGMA:g}", _blur_bgr_image(img, REFERENCE_AUG_BLUR_SIGMA))
    )
    return variants


def _build_augmented_reference_embeddings(
    target_img: np.ndarray,
) -> Tuple[Optional[np.ndarray], List[str], Optional[Dict[str, object]]]:
    """
    Embed the reference photo under multiple augmentations.

    Returns (embeddings[N,D], labels, primary_meta) where row 0 is always
    the original when that variant embeds successfully.
    """
    embeddings: List[np.ndarray] = []
    labels: List[str] = []
    primary_meta: Optional[Dict[str, object]] = None

    for label, variant in _iter_reference_augmentations(target_img):
        packed = _embed_scan_reference_face(variant)
        if packed is None:
            continue
        feat, meta = packed
        norm = _normalize_face_feature(feat)
        if norm is None:
            continue
        vec = norm.reshape(-1)
        embeddings.append(vec)
        labels.append(label)
        if label == "original":
            primary_meta = meta

    if not embeddings:
        return None, [], None

    # Keep original first when present so diagnostics stay stable.
    if "original" in labels:
        orig_idx = labels.index("original")
        if orig_idx != 0:
            embeddings.insert(0, embeddings.pop(orig_idx))
            labels.insert(0, labels.pop(orig_idx))

    return np.stack(embeddings, axis=0), labels, primary_meta


def _reference_bank_primary_feat(bank: np.ndarray) -> np.ndarray:
    """First (original) row of an augmented reference bank, or the vector itself."""
    arr = np.asarray(bank, dtype=np.float32)
    if arr.ndim == 2:
        return arr[0]
    return arr.reshape(-1)


def _target_reference_valid(feat: Optional[np.ndarray]) -> bool:
    if feat is None:
        return False
    arr = np.asarray(feat, dtype=np.float32)
    if arr.ndim == 2:
        return arr.shape[0] > 0 and arr.shape[1] > 0
    return _normalize_face_feature(arr) is not None


def _embedding_head(vec: Optional[np.ndarray], n: int = 8) -> List[float]:
    if vec is None:
        return []
    flat = np.asarray(vec, dtype=np.float32).reshape(-1)
    return [round(float(v), 6) for v in flat[:n]]


def _print_cosine_similarity_diag(
    ref_feat: np.ndarray,
    cctv_feat: np.ndarray,
    *,
    label: str = "",
) -> None:
    r = np.asarray(ref_feat, dtype=np.float64).reshape(-1)
    c = np.asarray(cctv_feat, dtype=np.float64).reshape(-1)
    ref_norm = float(np.linalg.norm(r))
    cctv_norm = float(np.linalg.norm(c))
    raw_dot = float(np.dot(r, c))
    cosine_sim = raw_dot / (ref_norm * cctv_norm + 1e-12)
    prefix = f"{label} " if label else ""
    print(
        f"{prefix}cosine_sim = {cosine_sim:.6f}, raw_dot = {raw_dot:.6f}, "
        f"ref_norm = {ref_norm:.6f}, cctv_norm = {cctv_norm:.6f}"
    )


def _save_debug_bgr(path: Path, img: Optional[np.ndarray]) -> None:
    if img is None or img.size == 0:
        return
    cv2.imwrite(str(path), img)
    print(f"Saved debug frame: {path}")


def _clear_scan_embed_diagnostics() -> None:
    global _SCAN_DIAG_REF, _SCAN_DIAG_CANDIDATES
    _SCAN_DIAG_REF = {}
    _SCAN_DIAG_CANDIDATES = []


def _record_embed_diag_candidate(entry: Dict[str, object]) -> None:
    """Keep only the highest-similarity embed debug payloads (bounded RAM)."""
    if not VIDEO_SCAN_EMBED_DIAG:
        return
    global _SCAN_DIAG_CANDIDATES
    _SCAN_DIAG_CANDIDATES.append(entry)
    _SCAN_DIAG_CANDIDATES.sort(
        key=lambda e: int(e.get("similarity", 0)), reverse=True
    )
    del _SCAN_DIAG_CANDIDATES[_SCAN_DIAG_CANDIDATE_CAP:]


def finalize_scan_embed_diagnostics(
    target_feat: np.ndarray,
    top_face_matches: List[Dict[str, object]],
) -> None:
    """TEMP: export aligned crops + embedding vector checks after a video scan."""
    if not VIDEO_SCAN_EMBED_DIAG:
        return

    from face_analyzer import describe_arcface_recognition_model

    SCAN_EMBED_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    analyzer = _get_gpu_scan_analyzer()

    print("=== SCAN EMBED DIAGNOSTICS ===")

    ref_info = describe_arcface_recognition_model(analyzer)
    print(f"Reference ArcFace model: {ref_info}")
    print(f"CCTV ArcFace model: {ref_info}")

    ref_pack = _SCAN_DIAG_REF
    ref_feat_stored = ref_pack.get("feat")
    ref_bank = ref_pack.get("feats")
    ref_labels = ref_pack.get("labels")
    ref_debug = ref_pack.get("debug") if isinstance(ref_pack.get("debug"), dict) else {}

    ref_aligned = ref_debug.get("aligned_bgr")
    if ref_aligned is not None:
        _save_debug_bgr(SCAN_EMBED_DEBUG_DIR / "ref_aligned.jpg", ref_aligned)

    if isinstance(ref_bank, np.ndarray) and ref_bank.ndim == 2:
        print(f"ref_embedding_bank: {ref_bank.shape[0]} variants")
        if isinstance(ref_labels, list):
            print(f"ref_augmentation_labels: {ref_labels}")
    ref_vec = (
        ref_feat_stored
        if ref_feat_stored is not None
        else _reference_bank_primary_feat(target_feat)
    )
    print(f"ref_embedding[:8] = {_embedding_head(ref_vec)}")

    ranked = sorted(
        _SCAN_DIAG_CANDIDATES,
        key=lambda e: int(e.get("similarity", 0)),
        reverse=True,
    )
    if not ranked:
        ranked = sorted(
            top_face_matches,
            key=lambda e: int(e.get("similarity", 0)),
            reverse=True,
        )
    if not ranked:
        print("cctv_best_embedding[:8] = [] (no faces compared)")
        print("=== END SCAN EMBED DIAGNOSTICS ===")
        return

    best = ranked[0]
    best_emb = best.get("embedding")
    print(f"cctv_best_embedding[:8] = {_embedding_head(best_emb)}")

    if ref_vec is not None and best_emb is not None:
        _print_cosine_similarity_diag(ref_vec, best_emb)
        if isinstance(ref_bank, np.ndarray) and ref_bank.ndim == 2:
            best_dist = _scan_embedding_distance(ref_bank, best_emb, gpu=True)
            print(
                f"best_augmented_match_distance = {best_dist:.6f} "
                f"({confidence_score_from_distance(best_dist)}%)"
            )

    best_debug = best.get("debug") if isinstance(best.get("debug"), dict) else {}
    if ref_debug.get("arcface_input_shape") is not None:
        print(f"ArcFace input shape (reference): {ref_debug['arcface_input_shape']}")
    if best_debug.get("arcface_input_shape") is not None:
        print(f"ArcFace input shape (cctv best): {best_debug['arcface_input_shape']}")

    seen_frames = set()
    slot = 0
    for entry in ranked:
        frame_n = int(entry.get("frame", -1))
        if frame_n in seen_frames:
            continue
        seen_frames.add(frame_n)
        slot += 1
        if slot > 5:
            break

        dbg = entry.get("debug") if isinstance(entry.get("debug"), dict) else {}
        prefix = f"cctv_{frame_n}"
        _save_debug_bgr(
            SCAN_EMBED_DEBUG_DIR / f"{prefix}_raw.jpg",
            dbg.get("raw_crop"),
        )
        _save_debug_bgr(
            SCAN_EMBED_DEBUG_DIR / f"{prefix}_aligned.jpg",
            dbg.get("aligned_bgr"),
        )
        if dbg.get("enhanced_applied") and dbg.get("enhanced_crop") is not None:
            _save_debug_bgr(
                SCAN_EMBED_DEBUG_DIR / f"{prefix}_enhanced.jpg",
                dbg.get("enhanced_crop"),
            )

        print(
            f"Frame {frame_n}: similarity={entry.get('similarity')}%, "
            f"embedding[:8]={_embedding_head(entry.get('embedding'))}"
        )

    print("=== END SCAN EMBED DIAGNOSTICS ===")


def _feature_from_face_bbox(
    frame: np.ndarray,
    face_box: Tuple[int, int, int, int],
    *,
    gpu: bool = False,
    landmarks: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[np.ndarray]:
    """ArcFace embed from a full-resolution face crop (detection already done at 640w)."""
    if not gpu:
        if not models_ready():
            return None
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = face_box
        pad = int(0.15 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(fw, x2 + pad), min(fh, y2 + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        faces = _yunet_extract_faces(
            crop,
            score_threshold=VIDEO_SCAN_SCORE_THRESHOLD,
            upscale_min_side=VIDEO_SCAN_UPSCALE_MIN_SIDE,
        )
        if not faces:
            return None
        return faces[0][1]

    packed = _scan_arcface_embed(frame, face_box, landmarks=landmarks)
    if packed is None:
        return None
    feat, _meta = packed
    return feat


def _scan_hit_thumbnail(frame: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Downscale a full annotated frame for JSON thumbnails without cropping."""
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _build_scan_hit(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    video_name: str,
    face_box: Tuple[int, int, int, int],
    dist: float,
    match_pct: int,
    *,
    original_filename: Optional[str] = None,
) -> Dict:
    annotated = _draw_target_match_overlay(frame, face_box, dist)
    ts = _format_video_timestamp(frame_idx, fps)
    storage_name = str(video_name or "").strip()
    display_name = resolve_original_video_filename(storage_name, original_filename)
    full_scene = _scan_hit_thumbnail(annotated, max_side=1280)
    grid_scene = _scan_hit_thumbnail(annotated)
    full_frame_url = _crop_to_data_url(full_scene)
    return {
        "timestamp": ts,
        "timestamp_seconds": int(frame_idx / fps),
        "frame": frame_idx,
        "distance": round(float(dist), 4),
        "match_percent": match_pct,
        "confidence_score": match_pct,
        "full_frame": full_frame_url,
        "scene_image": full_frame_url,
        "video_frame": full_frame_url,
        "image": full_frame_url,
        "thumbnail": _crop_to_data_url(grid_scene),
        "video_file": storage_name,
        "original_filename": display_name,
        "location": f"{display_name} at {ts}",
    }


def _finalize_near_miss_hits(hits: List[Dict]) -> List[Dict]:
    """Dedupe and cap low-confidence scan candidates for manual review."""
    if not hits:
        return []
    seen = set()
    unique: List[Dict] = []
    for hit in sorted(
        hits,
        key=lambda h: confidence_score_from_distance(float(h.get("distance", 1.0))),
        reverse=True,
    ):
        key = (
            str(hit.get("video_file", "")),
            int(hit.get("frame", -1)),
        )
        if key in seen:
            continue
        seen.add(key)
        pct = confidence_score_from_distance(float(hit.get("distance", 1.0)))
        if pct < VIDEO_TARGET_NEAR_MISS_MIN_PERCENT:
            continue
        finalized = _finalize_scan_hit(dict(hit))
        finalized["status"] = "near_miss"
        finalized["label"] = f"LOW CONFIDENCE [{pct}%]"
        unique.append(finalized)
        if len(unique) >= VIDEO_TARGET_NEAR_MISS_MAX_RESULTS:
            break
    return unique


def _crop_largest_face_with_padding(
    image: np.ndarray, pad_ratio: float = 0.10
) -> Optional[np.ndarray]:
    """Detect the largest face and return a padded face crop from the original image."""
    detector, _ = load_face_models()
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None

    best = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = [float(v) for v in best[:4]]
    pad_x = int(fw * pad_ratio)
    pad_y = int(fh * pad_ratio)

    x1 = max(0, int(x) - pad_x)
    y1 = max(0, int(y) - pad_y)
    x2 = min(w, int(x + fw) + pad_x)
    y2 = min(h, int(y + fh) + pad_y)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def crop_person(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = frame.shape[:2]
    pad = int(0.1 * max(x2 - x1, y2 - y1))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def _upscale_crop(crop: np.ndarray, min_side: int = 256) -> np.ndarray:
    h, w = crop.shape[:2]
    if max(h, w) >= min_side:
        return crop
    scale = min_side / max(h, w)
    return cv2.resize(
        crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
    )


def _downscale_frame_for_video_scan(
    frame: np.ndarray,
    *,
    width: int = VIDEO_SCAN_WIDTH,
) -> Tuple[np.ndarray, "UniformScaleTransform"]:
    """640w downscale for dashboard video scan (delegates to orion executor)."""
    from image_geometry import UniformScaleTransform
    from orion_frame_pipeline import resize_frame_for_video_scan

    return resize_frame_for_video_scan(frame, width=width)


def _unmap_scan_bbox(
    bbox: Tuple[int, int, int, int],
    transform: LetterboxTransform,
) -> Tuple[int, int, int, int]:
    """Map a face box from the letterboxed scan canvas to full-frame coordinates."""
    return transform.unmap_bbox(bbox)


def _ensure_video_scan_gpu() -> bool:
    """Warm InsightFace CUDA stack once per scan; required for pure-GPU video path."""
    if not gpu_scan_ready():
        return False
    _get_gpu_scan_analyzer()
    return True


def _resize_frame_for_video_scan(
    frame: np.ndarray,
    *,
    width: int = VIDEO_SCAN_WIDTH,
    height: int = VIDEO_SCAN_LEGACY_LETTERBOX_HEIGHT,
) -> Tuple[np.ndarray, LetterboxTransform]:
    """Letterbox to fixed WxH (legacy photo lookup paths only)."""
    return letterbox_resize(frame, width, height)


def encoding_from_frame_bbox(
    frame: np.ndarray, bbox: np.ndarray
) -> Optional[np.ndarray]:
    """Extract face feature from a person bounding box in a BGR frame."""
    result = face_from_person_bbox(frame, bbox)
    if result is None:
        return None
    return result[0]


def face_from_person_bbox(
    frame: np.ndarray,
    bbox: np.ndarray,
    min_sharpness: float = 0.0,
    upscale_min_side: int = 256,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Return (feature, hi-res face_bgr, face_area) or None if no face in bbox."""
    if not models_ready():
        return None
    from captures import crop_face_from_frame, map_bbox_from_person_crop_to_source

    crop = crop_person(frame, bbox)
    if crop is None:
        return None
    detector, recognizer = load_face_models()
    upscaled = _upscale_crop(crop, min_side=upscale_min_side)
    h, w = upscaled.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(upscaled)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: f[2] * f[3])
    ok, _reason = verify_yunet_face_for_capture(best)
    if not ok:
        return None
    area = float(best[2] * best[3])

    bx, by, bw, bh = best[0], best[1], best[2], best[3]
    local_bbox = (int(bx), int(by), int(bx + bw), int(by + bh))
    abs_bbox = map_bbox_from_person_crop_to_source(
        bbox,
        local_bbox,
        frame.shape,
        pad_ratio=0.1,
        upscale_min_side=upscale_min_side,
        crop_shape=crop.shape[:2],
    )
    thumbnail = crop_face_from_frame(frame, abs_bbox)
    if thumbnail is None or thumbnail.size == 0:
        thumbnail = upscaled[int(by) : int(by + bh), int(bx) : int(bx + bw)]

    aligned = recognizer.alignCrop(upscaled, best)
    if min_sharpness > 0 and face_sharpness(aligned) < min_sharpness:
        return None
    feature = recognizer.feature(aligned)
    display_face = thumbnail if thumbnail is not None and thumbnail.size else aligned
    return feature, display_face, area


def body_embedding_from_frame_bbox(
    frame: np.ndarray, bbox: np.ndarray
) -> Optional[np.ndarray]:
    crop = crop_person(frame, bbox)
    if crop is None:
        return None
    return extract_body_embedding(crop)


def encoding_from_image(image_path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    return encoding_from_bgr_image(img, yolo_model=None)


def encoding_from_bgr_image(
    img: np.ndarray, yolo_model=None
) -> Optional[np.ndarray]:
    """Find a face in the full image, or inside YOLO person boxes (CCTV-friendly)."""
    result = _detect_largest_face(img)
    if result is not None:
        return result[1]

    if yolo_model is None:
        return None

    from orion_frame_pipeline import wrap_yolo_results

    results = yolo_model(img, classes=[0], verbose=False)
    first = wrap_yolo_results(results)
    if first is None:
        return None
    boxes = first.boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    for bbox in xyxy[areas.argsort()[::-1]]:
        feature = encoding_from_frame_bbox(img, bbox)
        if feature is not None:
            return feature
    return None


def _largest_person_bbox(img: np.ndarray, yolo_model) -> Optional[np.ndarray]:
    from orion_frame_pipeline import wrap_yolo_results

    results = yolo_model(img, classes=[0], verbose=False)
    first = wrap_yolo_results(results)
    if first is None:
        return None
    boxes = first.boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return xyxy[areas.argmax()]


def _blob_to_feature(blob: bytes) -> np.ndarray:
    """Unpickle a stored embedding (NumPy 1.x / 2.x pickle compatible)."""
    try:
        return pickle.loads(blob)
    except ModuleNotFoundError as exc:
        # Gallery blobs pickled under NumPy 2 reference numpy._core.*
        if "numpy._core" not in str(exc):
            raise
        import numpy_compat as _np_compat

        _np_compat.apply_numpy_core_alias()
        return pickle.loads(blob)


def _feature_to_blob(feature: np.ndarray) -> bytes:
    return pickle.dumps(feature, protocol=pickle.HIGHEST_PROTOCOL)


def face_sharpness(image: Optional[np.ndarray]) -> float:
    """Estimate image sharpness via Laplacian variance."""
    if image is None or image.size == 0:
        return 0.0
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    try:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def describe_face_sharpness(image: Optional[np.ndarray]) -> Tuple[str, float]:
    """
    Human label for capture / Telegram quality.

    Returns ``(label, laplacian)`` where label is Clear, Slightly blurry, or Blurry.
    Tiny crops often score a huge Laplacian from pixel noise — never call those Clear.
    """
    score = face_sharpness(image)
    try:
        h, w = int(image.shape[0]), int(image.shape[1])
        side = min(h, w)
    except Exception:
        side = 0
    if side and side < 64:
        return "Too small", score
    clear_floor = float(BLURRY_SHARPNESS_THRESHOLD)
    soft_floor = max(8.0, clear_floor * 0.28)
    if side and side < 80 and score >= 400.0:
        # Alias mush on small plates — not a truly sharp face.
        return "Slightly blurry", score
    if score >= clear_floor:
        return "Clear", score
    if score >= soft_floor:
        return "Slightly blurry", score
    return "Blurry", score


def _normalize_face_feature(feature: np.ndarray) -> Optional[np.ndarray]:
    """Return a valid 2D float32 face embedding (1, D), or None when invalid."""
    if feature is None:
        return None
    try:
        arr = np.asarray(feature, dtype=np.float32).reshape(1, -1)
    except Exception:
        return None
    if arr.size == 0 or arr.shape[1] == 0:
        return None
    return arr


def _distance(f1: np.ndarray, f2: np.ndarray) -> float:
    left = _normalize_face_feature(f1)
    right = _normalize_face_feature(f2)
    if left is None or right is None:
        return float("inf")
    if int(left.shape[1]) != int(right.shape[1]):
        return float("inf")

    # Gallery is InsightFace ArcFace (512-d). Use cosine — do not send ArcFace
    # vectors through OpenCV SFace matcher (128-d only).
    if int(left.shape[1]) == 512:
        return _arcface_embedding_distance(left, right)

    _, recognizer = load_face_models()
    score = recognizer.match(left, right, cv2.FaceRecognizerSF_FR_COSINE)
    # မြန်မာ: OpenCV ရလဒ် ဗလာ/ပုံစံမမှန်ပါက tuple index out of range မဖြစ်စေရန်
    from orion_frame_pipeline import safe_match_score

    score = safe_match_score(score)
    if not np.isfinite(score):
        return float("inf")

    # FaceRecognizerSF_FR_COSINE returns similarity (higher is better).
    # Convert to cosine distance so our threshold semantics remain:
    # lower = more similar, higher = less similar.
    if -1.0 <= score <= 1.0:
        return 1.0 - score

    # Fallback for unusual OpenCV builds.
    return score


def load_all_encodings(
    conn: sqlite3.Connection, table: str = "person_encodings"
) -> List[Tuple[int, np.ndarray, Optional[float]]]:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    has_quality = "quality_score" in cols
    if has_quality:
        rows = conn.execute(
            f"SELECT known_person_id, encoding, quality_score FROM {table}"
        ).fetchall()
        return [(row[0], _blob_to_feature(row[1]), row[2]) for row in rows]
    rows = conn.execute(
        f"SELECT known_person_id, encoding FROM {table}"
    ).fetchall()
    return [(row[0], _blob_to_feature(row[1]), None) for row in rows]


def _backfill_best_faces_from_encodings(conn: sqlite3.Connection) -> None:
    """Seed person_best_faces from person_encodings when CCTV never wrote a best profile."""
    rows = conn.execute(
        """
        SELECT pe.known_person_id, pe.encoding, pe.quality_score
        FROM person_encodings pe
        LEFT JOIN person_best_faces pf ON pf.known_person_id = pe.known_person_id
        WHERE pf.known_person_id IS NULL
        """
    ).fetchall()
    if not rows:
        return

    best_by_person: Dict[int, Tuple[bytes, Optional[float]]] = {}
    for known_id, encoding, quality in rows:
        q = float(quality) if quality is not None else 0.0
        prev = best_by_person.get(known_id)
        if prev is None or q > (float(prev[1]) if prev[1] is not None else 0.0):
            best_by_person[known_id] = (encoding, quality)

    now = datetime.now().isoformat(timespec="seconds")
    for known_id, (encoding, quality) in best_by_person.items():
        q = float(quality) if quality is not None else 0.0
        conn.execute(
            """
            INSERT INTO person_best_faces
                (known_person_id, encoding, quality_score, image_path, updated_at)
            VALUES (?, ?, ?, NULL, ?)
            """,
            (known_id, encoding, q, now),
        )
    conn.commit()


def _collect_person_face_embeddings(
    conn: sqlite3.Connection,
) -> Dict[int, List[Tuple[np.ndarray, Optional[float], str]]]:
    """
    Gather every face embedding per known person from person_best_faces and
    person_encodings (no early filtering — all rows are kept for min-distance scoring).
    """
    _backfill_best_faces_from_encodings(conn)

    by_person: Dict[int, List[Tuple[np.ndarray, Optional[float], str]]] = {}

    for known_id, blob, quality in conn.execute(
        "SELECT known_person_id, encoding, quality_score FROM person_best_faces"
    ).fetchall():
        feat = _normalize_face_feature(_blob_to_feature(blob))
        if feat is not None:
            by_person.setdefault(int(known_id), []).append(
                (feat, quality, "person_best_faces")
            )

    for known_id, blob, quality in conn.execute(
        "SELECT known_person_id, encoding, quality_score FROM person_encodings"
    ).fetchall():
        feat = _normalize_face_feature(_blob_to_feature(blob))
        if feat is not None:
            by_person.setdefault(int(known_id), []).append(
                (feat, quality, "person_encodings")
            )

    for (known_id,) in conn.execute("SELECT id FROM known_persons ORDER BY id"):
        by_person.setdefault(int(known_id), [])

    return by_person


def load_face_gallery(
    conn: sqlite3.Connection,
) -> List[Tuple[int, np.ndarray, Optional[float], str]]:
    """
    One representative embedding per person (best quality available).
    Used for display/back-compat; search scoring uses all embeddings via
    _collect_person_face_embeddings().
    """
    gallery: Dict[int, Tuple[np.ndarray, Optional[float], str]] = {}
    for known_id, entries in _collect_person_face_embeddings(conn).items():
        if not entries:
            continue
        best = max(
            entries,
            key=lambda item: float(item[1]) if item[1] is not None else 0.0,
        )
        gallery[known_id] = best
    return [
        (known_id, feat, quality, source)
        for known_id, (feat, quality, source) in sorted(gallery.items())
    ]


def _score_all_persons_min_distance(
    feature: np.ndarray,
    conn: sqlite3.Connection,
) -> List[Tuple[int, float, Optional[float], str, int]]:
    """
    For every known_person_id, compare the query against ALL embeddings in
    person_best_faces and person_encodings; keep the minimum distance.
    Returns (person_id, min_distance, quality_at_best, source_at_best, num_embeddings).
    """
    query = _normalize_face_feature(feature)
    if query is None:
        return []

    query_dim = int(query.reshape(-1).shape[0])
    by_person = _collect_person_face_embeddings(conn)
    results: List[Tuple[int, float, Optional[float], str, int]] = []

    for known_id in sorted(by_person.keys()):
        entries = by_person[known_id]
        if not entries:
            continue

        min_dist = float("inf")
        best_quality: Optional[float] = None
        best_source = ""

        for stored, quality, source in entries:
            if int(stored.reshape(-1).shape[0]) != query_dim:
                continue
            dist = _distance(query, stored)
            if not np.isfinite(dist):
                continue
            if dist < min_dist:
                min_dist = dist
                best_quality = quality
                best_source = source

        if np.isfinite(min_dist) and min_dist != float("inf"):
            results.append(
                (known_id, min_dist, best_quality, best_source, len(entries))
            )

    results.sort(key=lambda row: row[1])
    return results


def _score_face_gallery(
    feature: np.ndarray,
    conn: sqlite3.Connection,
) -> List[Tuple[int, float, Optional[float], str]]:
    """Compatibility wrapper — min distance per person."""
    return [
        (person_id, dist, quality, source)
        for person_id, dist, quality, source, _n in _score_all_persons_min_distance(
            feature, conn
        )
    ]


def find_face_matches(
    feature: np.ndarray,
    conn: sqlite3.Connection,
    query_quality: Optional[float] = None,
    threshold: float = MULTI_MATCH_THRESHOLD,
    log_candidates: int = 0,
) -> List[Dict]:
    """
    Return every person whose minimum face distance is <= threshold.
    Uses one fixed threshold for all IDs (no per-person cutoff).
    """
    del query_quality  # reserved; threshold is explicit for photo multi-match

    scored = _score_all_persons_min_distance(feature, conn)
    if not scored:
        if log_candidates:
            print(
                "[FACE MATCH] No gallery embeddings found in "
                "person_best_faces or person_encodings."
            )
        return []

    if log_candidates > 0:
        print(f"[FACE MATCH] Multi-match threshold: {threshold:.2f}")
        for rank, (known_id, dist, _gq, source, n_enc) in enumerate(
            scored[: max(log_candidates, len(scored))], 1
        ):
            print(
                f"[FACE MATCH] #{rank} Person #{known_id}, "
                f"min distance: {dist:.3f} ({n_enc} embedding(s), best from {source})"
            )

    matches: List[Dict] = []
    for known_id, dist, _gallery_quality, source, _n_enc in scored:
        if dist <= threshold:
            matches.append(
                {
                    "person_id": known_id,
                    "known_person_id": known_id,
                    "distance": dist,
                    "source": source,
                    "threshold_used": threshold,
                }
            )
        elif log_candidates > 0:
            print(
                f"[FACE MATCH] Excluded Person #{known_id}: "
                f"min distance {dist:.3f} > {threshold:.2f}"
            )

    # Suppress weak secondary hits when the best match is already decisive
    # (e.g. enrollment selfie vs noisy looking-down stranger crops).
    if (
        matches
        and float(threshold) >= float(PHOTO_SEARCH_MATCH_THRESHOLD) - 1e-9
        and float(matches[0]["distance"]) <= PHOTO_SEARCH_STRONG_BEST
    ):
        best = float(matches[0]["distance"])
        cap = best + PHOTO_SEARCH_STRONG_MAX_GAP
        filtered = [m for m in matches if float(m["distance"]) <= cap]
        if log_candidates > 0 and len(filtered) < len(matches):
            print(
                f"[FACE MATCH] Strong-best filter: kept {len(filtered)}/"
                f"{len(matches)} (best={best:.3f}, cap={cap:.3f})"
            )
        matches = filtered

    if log_candidates > 0 and not matches and scored:
        print(
            f"[FACE MATCH] No match — closest Person #{scored[0][0]} "
            f"at {scored[0][1]:.3f} (threshold {threshold:.2f})"
        )

    return matches


def _score_named_person_encodings_only(
    feature: np.ndarray,
    conn: sqlite3.Connection,
) -> List[Tuple[int, float]]:
    """
    Score live CCTV embeds against Named People enrollment portraits only.

    Uses ``person_best_faces`` (the Named People photo), not live
    ``person_encodings`` rows that can accumulate from hallway captures and
    cause wrong names.
    """
    query = _normalize_face_feature(feature)
    if query is None:
        return []
    hits: List[Tuple[int, float]] = []
    rows = conn.execute(
        """
        SELECT bf.known_person_id, bf.encoding
        FROM person_best_faces bf
        JOIN known_persons kp ON kp.id = bf.known_person_id
        WHERE COALESCE(kp.is_named, 0) = 1
          AND COALESCE(kp.is_watchlist, 0) = 0
        """
    ).fetchall()
    for known_id, blob in rows:
        try:
            if not is_named_person(conn, int(known_id)):
                continue
            if is_enrolled_watchlist_person(conn, int(known_id)):
                continue
            feat = _normalize_face_feature(_blob_to_feature(blob))
            if feat is None:
                continue
            dist = float(_arcface_embedding_distance(query, feat))
            if np.isfinite(dist):
                hits.append((int(known_id), dist))
        except (TypeError, ValueError):
            continue
    # Named people with no best-face row: fall back to any stored encoding.
    covered = {pid for pid, _ in hits}
    for known_id, entries in _collect_person_face_embeddings(conn).items():
        if int(known_id) in covered:
            continue
        if not is_named_person(conn, int(known_id)):
            continue
        if is_enrolled_watchlist_person(conn, int(known_id)):
            continue
        if not entries:
            continue
        best = float("inf")
        for feat, _quality, _source in entries:
            dist = _arcface_embedding_distance(query, feat)
            if dist < best:
                best = float(dist)
        if np.isfinite(best):
            hits.append((int(known_id), best))
    hits.sort(key=lambda x: x[1])
    return hits


def _find_named_gallery_match(
    feature: np.ndarray,
    conn: sqlite3.Connection,
    *,
    max_distance: Optional[float] = None,
    log_candidates: int = 0,
) -> Tuple[Optional[int], float]:
    """Match live CCTV against Named People enrollment embeddings (and gallery JPEGs)."""
    gate = float(max_distance if max_distance is not None else NAMED_MATCH_THRESHOLD)
    by_id: Dict[int, float] = {}
    for pid, dist in _score_named_person_encodings_only(feature, conn):
        try:
            by_id[int(pid)] = float(dist)
        except (TypeError, ValueError):
            continue
    # Only re-embed gallery JPEGs for named people who have no SQLite encodings.
    # Re-detecting enrollment photos every capture used to return a weak score that
    # blocked a stronger encoding match and labeled the person Stranger.
    try:
        from named_gallery import score_query_against_named_galleries

        jpeg_scores = score_query_against_named_galleries(
            feature,
            conn,
            skip_person_ids=set(by_id.keys()),
        )
        for pid, dist in jpeg_scores:
            i = int(pid)
            d = float(dist)
            if d < by_id.get(i, float("inf")):
                by_id[i] = d
    except Exception:
        pass
    enrollment_scores = list(by_id.items())
    if not enrollment_scores:
        return None, float("inf")
    named_id, named_dist = _pick_best_named_person(
        enrollment_scores,
        conn,
        max_distance=gate,
    )
    if named_id is not None and log_candidates > 0:
        print(
            f"[FACE MATCH] Named enrollment hit Person #{named_id}, "
            f"Distance: {named_dist:.3f} (gate {gate:.2f})"
        )
    elif log_candidates > 0 and enrollment_scores:
        best_pid, best_dist = min(enrollment_scores, key=lambda item: item[1])
        print(
            f"[FACE MATCH] Named miss — closest Person #{best_pid} "
            f"at {best_dist:.3f} (gate {gate:.2f})"
        )
    return named_id, named_dist


def closest_named_candidate(
    feature: np.ndarray,
    conn: sqlite3.Connection,
) -> Dict[str, object]:
    """Best named person regardless of threshold, for UI debugging."""
    named_id, named_dist = _find_named_gallery_match(
        feature, conn, max_distance=0.99
    )
    if named_id is None or not np.isfinite(named_dist):
        scored = _score_named_person_encodings_only(feature, conn)
        if not scored:
            return {}
        named_id, named_dist = scored[0]
    if named_id is None or not np.isfinite(named_dist):
        return {}
    return {
        "known_person_id": int(named_id),
        "name": resolve_known_person_name(conn, int(named_id)),
        "distance": round(float(named_dist), 4),
        "match_percent": int(confidence_score_from_distance(named_dist)),
        "within_gate": bool(float(named_dist) <= float(NAMED_MATCH_THRESHOLD)),
        "gate": float(NAMED_MATCH_THRESHOLD),
    }


def find_best_face_match(
    feature: np.ndarray,
    conn: sqlite3.Connection,
    query_quality: Optional[float] = None,
    log_candidates: int = 0,
    *,
    camera_location: Optional[str] = None,
) -> Tuple[Optional[int], float]:
    named_id, named_dist = _find_named_gallery_match(
        feature, conn, log_candidates=log_candidates
    )
    if named_id is not None:
        return int(named_id), float(named_dist)

    query = _normalize_face_feature(feature)
    if query is not None:
        # Fast path: Qdrant / in-memory cosine watchlist (hot-reloaded).
        try:
            from watchlist_vector_db import (
                cosine_similarity_to_distance,
                search_similar_faces,
            )

            threshold = MATCH_THRESHOLD
            if query_quality is not None and query_quality < BLURRY_SHARPNESS_THRESHOLD:
                threshold = RELAXED_MATCH_THRESHOLD
            # Cosine similarity floor corresponding to distance threshold.
            sim_floor = max(0.0, 1.0 - float(threshold))
            qdrant_hits = search_similar_faces(
                query.reshape(-1),
                top_k=24,
                score_threshold=sim_floor,
                prefer_memory=True,
            )
            if qdrant_hits:
                scored_q = [
                    (int(hit.person_id), cosine_similarity_to_distance(hit.score))
                    for hit in qdrant_hits
                    if hit.person_id is not None
                ]
                if log_candidates > 0:
                    for rank, (pid, dist) in enumerate(scored_q[:log_candidates], 1):
                        print(
                            f"[FACE MATCH] #{rank} Candidate Person #{pid}, "
                            f"Distance: {dist:.3f} (source: qdrant)"
                        )
                best_id, best_dist = _finalize_face_match(
                    scored_q, conn, threshold, log_candidates=log_candidates
                )
                if best_id is not None:
                    return int(best_id), float(best_dist)
                # Day-2 / overnight: reuse unlabeled stranger even when slightly
                # over the strict watchlist gate.
                soft_hit = _soft_stranger_rematch(
                    scored_q,
                    conn,
                    camera_location=camera_location,
                    log_candidates=log_candidates,
                )
                if soft_hit is not None:
                    return soft_hit
        except Exception:
            pass

        try:
            from gpu_face_index import search_best

            faiss_hits = search_best(query.reshape(-1), top_k=48)
            if faiss_hits:
                scored = [
                    (int(person_id), (l2 ** 2) / 2.0)
                    for person_id, l2 in faiss_hits
                    if person_id is not None
                ]
                if log_candidates > 0:
                    for rank, (known_id, dist) in enumerate(scored[:log_candidates], 1):
                        print(
                            f"[FACE MATCH] #{rank} Candidate Person #{known_id}, "
                            f"Distance: {dist:.3f} (source: faiss-gpu)"
                        )
                threshold = MATCH_THRESHOLD
                if query_quality is not None and query_quality < BLURRY_SHARPNESS_THRESHOLD:
                    threshold = RELAXED_MATCH_THRESHOLD
                best_id, best_dist = _finalize_face_match(
                    scored, conn, threshold, log_candidates=log_candidates
                )
                if best_id is not None:
                    return int(best_id), float(best_dist)
                soft_hit = _soft_stranger_rematch(
                    scored,
                    conn,
                    camera_location=camera_location,
                    log_candidates=log_candidates,
                )
                if soft_hit is not None:
                    return soft_hit
                if log_candidates > 0:
                    print(
                        f"[FACE MATCH] FAISS miss — best distance {best_dist:.3f} "
                        f"> threshold {threshold:.2f}"
                    )
        except Exception:
            pass

    scored_rows = _score_face_gallery(feature, conn)
    if not scored_rows:
        if log_candidates:
            print(
                "[FACE MATCH] No gallery embeddings found in "
                "person_best_faces or person_encodings."
            )
        return None, float("inf")

    if log_candidates > 0:
        for rank, (known_id, dist, _gq, source) in enumerate(scored_rows[:log_candidates], 1):
            print(
                f"[FACE MATCH] #{rank} Candidate Person #{known_id}, "
                f"Distance: {dist:.3f} (source: {source})"
            )

    threshold = MATCH_THRESHOLD
    gallery_quality = scored_rows[0][2] if scored_rows else None
    if (query_quality is not None and query_quality < BLURRY_SHARPNESS_THRESHOLD) or (
        gallery_quality is not None and gallery_quality < BLURRY_SHARPNESS_THRESHOLD
    ):
        threshold = RELAXED_MATCH_THRESHOLD

    candidates = [(int(kid), float(dist)) for kid, dist, _gq, _src in scored_rows]
    best_id, best_dist = _finalize_face_match(
        candidates, conn, threshold, log_candidates=log_candidates
    )

    if best_id is not None:
        return best_id, best_dist

    soft_hit = _soft_stranger_rematch(
        scored_rows,
        conn,
        camera_location=camera_location,
        log_candidates=log_candidates,
    )
    if soft_hit is not None:
        return soft_hit

    if log_candidates > 0:
        print(
            f"[FACE MATCH] No match — best distance {best_dist:.3f} "
            f"> threshold {threshold:.2f}"
        )
    return None, best_dist


def _recent_camera_stranger_ids(
    conn: sqlite3.Connection,
    camera_location: str,
    *,
    minutes: int = 20,
) -> Set[int]:
    """known_person_ids logged on this camera in the last N minutes."""
    cam = str(camera_location or "").strip()
    if not cam:
        return set()
    cam_key = cam.upper().replace("CAM-", "").replace("CAM", "").strip() or cam
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT known_person_id
            FROM tracking_logs
            WHERE known_person_id IS NOT NULL
              AND datetime(timestamp) >= datetime('now', ?)
              AND (
                    camera_location = ?
                 OR camera_location = ?
                 OR REPLACE(UPPER(camera_location), 'CAM-', '') = ?
              )
            """,
            (f"-{max(1, int(minutes))} minutes", cam, f"CAM-{cam_key}", cam_key),
        ).fetchall()
    except sqlite3.Error:
        return set()
    out: Set[int] = set()
    for row in rows:
        raw = row[0] if not isinstance(row, sqlite3.Row) else row["known_person_id"]
        try:
            out.add(int(raw))
        except (TypeError, ValueError):
            continue
    return out


def _prior_stranger_ids(
    conn: sqlite3.Connection,
    *,
    days: int = 14,
) -> Set[int]:
    """
    Unlabeled stranger IDs seen in the last N days (cross-day rematch pool).

    Named / watchlist people are excluded — they already match via the
    strict named/watchlist path. This set keeps ``Stranger_42`` stable when
    the same face returns tomorrow.
    """
    try:
        lookback = max(1, int(days))
    except (TypeError, ValueError):
        lookback = 14
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT kp.id
            FROM known_persons kp
            WHERE COALESCE(kp.is_named, 0) = 0
              AND COALESCE(kp.is_watchlist, 0) = 0
              AND (
                    datetime(COALESCE(kp.last_seen, kp.created_at))
                        >= datetime('now', ?)
                 OR EXISTS (
                        SELECT 1 FROM tracking_logs tl
                        WHERE tl.known_person_id = kp.id
                          AND datetime(tl.timestamp) >= datetime('now', ?)
                    )
              )
              AND (
                    EXISTS (
                        SELECT 1 FROM person_encodings pe
                        WHERE pe.known_person_id = kp.id
                    )
                 OR EXISTS (
                        SELECT 1 FROM person_best_faces bf
                        WHERE bf.known_person_id = kp.id
                    )
              )
            """,
            (f"-{lookback} days", f"-{lookback} days"),
        ).fetchall()
    except sqlite3.Error:
        return set()
    out: Set[int] = set()
    for row in rows:
        raw = row[0] if not isinstance(row, sqlite3.Row) else row["id"]
        try:
            out.add(int(raw))
        except (TypeError, ValueError):
            continue
    return out


def _soft_stranger_candidates(
    scored_rows: Sequence,
) -> List[Tuple[int, float]]:
    """Normalize scored rows to ``(person_id, distance)`` sorted ascending."""
    out: List[Tuple[int, float]] = []
    for row in scored_rows:
        try:
            kid = int(row[0])
            dist = float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not np.isfinite(dist):
            continue
        out.append((kid, dist))
    out.sort(key=lambda item: item[1])
    return out


def _soft_stranger_rematch(
    scored_rows: Sequence,
    conn: sqlite3.Connection,
    *,
    camera_location: Optional[str] = None,
    log_candidates: int = 0,
) -> Optional[Tuple[int, float]]:
    """
    Reuse an existing unlabeled stranger ID when the face is near the strict gate.

    Priority:
    1. Same-camera sightings in the last ~20 minutes (looser gate)
    2. Any stranger seen in the last N days (cross-day / next-day same ID)
    3. Any unlabeled stranger under the soft distance floor

    A small margin vs the next candidate reduces wrong merges overnight.
    """
    candidates = _soft_stranger_candidates(scored_rows)
    if not candidates:
        return None
    try:
        soft_max = float(os.environ.get("STRANGER_REMATCH_MAX_DIST", "0.60"))
    except (TypeError, ValueError):
        soft_max = 0.60
    soft_max = max(float(MATCH_THRESHOLD), min(soft_max, 0.68))
    try:
        cross_day_max = float(os.environ.get("STRANGER_CROSS_DAY_MAX_DIST", "0.62"))
    except (TypeError, ValueError):
        cross_day_max = 0.62
    cross_day_max = max(soft_max, min(cross_day_max, 0.70))
    try:
        cross_day_days = int(os.environ.get("STRANGER_CROSS_DAY_DAYS", "14"))
    except (TypeError, ValueError):
        cross_day_days = 14
    try:
        min_margin = float(os.environ.get("STRANGER_REMATCH_MARGIN", "0.02"))
    except (TypeError, ValueError):
        min_margin = 0.02

    recent = _recent_camera_stranger_ids(conn, str(camera_location or ""))
    prior = _prior_stranger_ids(conn, days=cross_day_days)
    # Same camera, last ~20 min: allow a slightly looser gate.
    cam_soft = min(0.64, soft_max + 0.04) if recent else soft_max

    def _ok_stranger(pid: int) -> bool:
        try:
            return (
                not is_named_person(conn, int(pid))
                and not is_enrolled_watchlist_person(conn, int(pid))
            )
        except Exception:
            return False

    def _margin_ok(best_dist: float, pool: Sequence[Tuple[int, float]]) -> bool:
        """Require clear separation from the next stranger under the same gate."""
        under = [(pid, d) for pid, d in pool if d <= best_dist + max(min_margin, 0.0) + 0.05]
        if len(under) <= 1:
            return True
        # Next distinct person must be clearly farther.
        second = None
        best_pid = under[0][0]
        for pid, d in under[1:]:
            if int(pid) != int(best_pid):
                second = d
                break
        if second is None:
            return True
        return float(second) - float(best_dist) >= float(min_margin)

    if recent:
        for kid, dist in candidates:
            if dist > cam_soft:
                break
            if int(kid) in recent and _ok_stranger(int(kid)):
                if log_candidates > 0:
                    print(
                        f"[FACE MATCH] Same-cam stranger rematch Person #{int(kid)} "
                        f"at {dist:.3f} (gate {cam_soft:.2f})"
                    )
                return int(kid), dist

    # Cross-day: prefer strangers already in the gallery from prior days.
    if prior:
        prior_hits = [
            (kid, dist)
            for kid, dist in candidates
            if dist <= cross_day_max and int(kid) in prior and _ok_stranger(int(kid))
        ]
        if prior_hits and _margin_ok(prior_hits[0][1], prior_hits):
            kid, dist = prior_hits[0]
            if log_candidates > 0:
                print(
                    f"[FACE MATCH] Cross-day stranger rematch Person #{int(kid)} "
                    f"at {dist:.3f} (gate {cross_day_max:.2f}, lookback {cross_day_days}d)"
                )
            return int(kid), dist

    for kid, dist in candidates:
        if dist > soft_max:
            break
        if _ok_stranger(int(kid)):
            pool = [
                (k, d)
                for k, d in candidates
                if d <= soft_max and _ok_stranger(int(k))
            ]
            if not _margin_ok(dist, pool):
                if log_candidates > 0:
                    print(
                        f"[FACE MATCH] Soft stranger rematch skipped Person #{int(kid)} "
                        f"at {dist:.3f} — ambiguous vs other strangers"
                    )
                return None
            if log_candidates > 0:
                print(
                    f"[FACE MATCH] Soft stranger rematch Person #{int(kid)} "
                    f"at {dist:.3f} (gate {soft_max:.2f})"
                )
            return int(kid), dist
    return None


def find_best_body_match(
    feature: np.ndarray, conn: sqlite3.Connection
) -> Tuple[Optional[int], float]:
    best_id = None
    best_dist = 999.0
    for known_id, stored, _quality in load_all_encodings(conn, "person_body_encodings"):
        dist = cosine_distance(feature, stored)
        if dist < best_dist:
            best_dist = dist
            best_id = known_id
    if best_id is not None and best_dist <= BODY_MATCH_THRESHOLD:
        return best_id, best_dist
    return None, best_dist


def register_person(
    conn: sqlite3.Connection,
    feature: np.ndarray,
    quality_score: Optional[float] = None,
    *,
    name: Optional[str] = None,
    watchlist: Optional[bool] = None,
    named: Optional[bool] = None,
) -> int:
    # Live watchlist / Telegram matching uses InsightFace ArcFace (512-d).
    # Refuse OpenCV SFace (128-d) registrations — they poison the gallery index
    # and can never match enrolled watchlist faces.
    flat = np.asarray(feature, dtype=np.float32).reshape(-1)
    if int(flat.size) != 512:
        raise ValueError(
            f"reject non-ArcFace embedding dim={int(flat.size)} (expected 512)"
        )
    ensure_known_person_name_column(conn)
    ensure_watchlist_flag_column(conn)
    ensure_named_flag_column(conn)
    now = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        """
        INSERT INTO known_persons (created_at, last_seen, name, is_watchlist, is_named)
        VALUES (?, ?, ?, ?, ?)
        """,
        (now, now, None, 0, 0),
    )
    known_id = cur.lastrowid
    display_name = (name or "").strip() or stranger_display_name(known_id)
    has_real = bool(name and not is_auto_identity_name(display_name))
    wl = 1 if (watchlist if watchlist is not None else has_real) else 0
    nm = 1 if (named if named is not None else has_real) else 0
    conn.execute(
        "UPDATE known_persons SET name = ?, is_watchlist = ?, is_named = ? WHERE id = ?",
        (display_name, wl, nm, known_id),
    )
    conn.execute(
        """
        INSERT INTO person_encodings (known_person_id, encoding, quality_score)
        VALUES (?, ?, ?)
        """,
        (known_id, _feature_to_blob(flat), quality_score),
    )
    _seed_best_face_if_missing(conn, known_id, flat, quality_score)
    conn.commit()
    refresh_gpu_face_index(conn)
    return known_id

def refresh_gpu_face_index(conn: sqlite3.Connection) -> int:
    """Rebuild FAISS GPU gallery index after registration or purge."""
    count = 0
    try:
        from gpu_face_index import rebuild_face_index

        count = rebuild_face_index(conn)
    except Exception:
        count = 0
    # Keep Qdrant + in-memory watchlist cosine index in sync (no tracker restart).
    try:
        from watchlist_vector_db import notify_watchlist_changed, sync_watchlist_from_sqlite

        sync_watchlist_from_sqlite(conn, notify=False)
        notify_watchlist_changed()
    except Exception:
        pass
    return count


def touch_person(conn: sqlite3.Connection, known_id: int) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "UPDATE known_persons SET last_seen = ? WHERE id = ?",
        (now, known_id),
    )


def add_encoding_for_person(
    conn: sqlite3.Connection,
    known_id: int,
    feature: np.ndarray,
    quality_score: Optional[float] = None,
) -> None:
    """Store an extra face sample to improve future matching."""
    conn.execute(
        """
        INSERT INTO person_encodings (known_person_id, encoding, quality_score)
        VALUES (?, ?, ?)
        """,
        (known_id, _feature_to_blob(feature), quality_score),
    )
    _seed_best_face_if_missing(conn, known_id, feature, quality_score)


def _seed_best_face_if_missing(
    conn: sqlite3.Connection,
    known_id: int,
    feature: np.ndarray,
    quality_score: Optional[float],
) -> None:
    """Ensure person_best_faces has a profile when CCTV registers a new person."""
    row = conn.execute(
        "SELECT 1 FROM person_best_faces WHERE known_person_id = ?",
        (known_id,),
    ).fetchone()
    if row:
        return
    now = datetime.now().isoformat(timespec="seconds")
    q = float(quality_score) if quality_score is not None else 0.0
    conn.execute(
        """
        INSERT INTO person_best_faces
            (known_person_id, encoding, quality_score, image_path, updated_at)
        VALUES (?, ?, ?, NULL, ?)
        """,
        (known_id, _feature_to_blob(feature), q, now),
    )


def _capture_profile_image_exists(path_value: Optional[str]) -> bool:
    """True when a stored best-face image path points at an on-disk file."""
    if not path_value:
        return False
    raw = str(path_value).strip().replace("\\", "/")
    if not raw:
        return False
    rel = raw.lstrip("/")
    if rel.startswith("captures/"):
        rel = rel[len("captures/") :]
    candidates = [BEST_FACES_DIR / Path(rel).name, PROJECT_DIR / raw, CAPTURES_DIR / rel]
    return any(p.is_file() for p in candidates)


def _save_best_face_image(known_id: int, face_img: np.ndarray) -> Optional[str]:
    if face_img is None or face_img.size == 0:
        return None
    from captures import CAPTURE_JPEG_QUALITY, prepare_face_thumbnail

    BEST_FACES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"person_{known_id}_best.jpg"
    path = BEST_FACES_DIR / filename
    thumbnail = prepare_face_thumbnail(face_img)
    if cv2.imwrite(
        str(path),
        thumbnail,
        [int(cv2.IMWRITE_JPEG_QUALITY), CAPTURE_JPEG_QUALITY],
    ):
        return f"best_faces/{filename}"
    return None


def upsert_best_face_profile(
    conn: sqlite3.Connection,
    known_id: int,
    feature: np.ndarray,
    face_img: Optional[np.ndarray],
    quality_score: Optional[float],
    *,
    allow_named: bool = False,
) -> bool:
    """Overwrite stored face profile when a significantly sharper shot appears.

    Named People keep enrollment photos only — live CCTV must not replace them
    unless ``allow_named=True`` (dashboard enroll / gallery sync).
    """
    if quality_score is None or not np.isfinite(quality_score):
        return False
    if not allow_named:
        try:
            if is_named_person(conn, int(known_id)) and not is_enrolled_watchlist_person(
                conn, int(known_id)
            ):
                return False
        except Exception:
            pass

    row = conn.execute(
        """
        SELECT quality_score, image_path FROM person_best_faces WHERE known_person_id = ?
        """,
        (known_id,),
    ).fetchone()
    existing_quality = float(row[0]) if row and row[0] is not None else None
    existing_image = str(row[1] or "").strip() if row else ""
    has_image = bool(existing_image and _capture_profile_image_exists(existing_image))
    should_upgrade = (
        existing_quality is None
        or quality_score >= (existing_quality * QUALITY_UPGRADE_RATIO)
        or not has_image
    )
    if not should_upgrade:
        return False

    image_path = _save_best_face_image(known_id, face_img) if face_img is not None else None
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO person_best_faces (known_person_id, encoding, quality_score, image_path, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(known_person_id) DO UPDATE SET
            encoding=excluded.encoding,
            quality_score=excluded.quality_score,
            image_path=excluded.image_path,
            updated_at=excluded.updated_at
        """,
        (known_id, _feature_to_blob(feature), float(quality_score), image_path, now),
    )

    # Keep person_encodings in sync by replacing one representative record.
    enc_row = conn.execute(
        "SELECT id FROM person_encodings WHERE known_person_id = ? ORDER BY id ASC LIMIT 1",
        (known_id,),
    ).fetchone()
    if enc_row:
        conn.execute(
            """
            UPDATE person_encodings
            SET encoding = ?, quality_score = ?
            WHERE id = ?
            """,
            (_feature_to_blob(feature), float(quality_score), enc_row[0]),
        )
    else:
        conn.execute(
            """
            INSERT INTO person_encodings (known_person_id, encoding, quality_score)
            VALUES (?, ?, ?)
            """,
            (known_id, _feature_to_blob(feature), float(quality_score)),
        )
    conn.commit()
    return True


def register_person_body(
    conn: sqlite3.Connection,
    feature: np.ndarray,
    *,
    name: Optional[str] = None,
) -> int:
    ensure_known_person_name_column(conn)
    now = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO known_persons (created_at, last_seen, name) VALUES (?, ?, ?)",
        (now, now, None),
    )
    known_id = cur.lastrowid
    display_name = (name or "").strip() or stranger_display_name(known_id)
    conn.execute(
        "UPDATE known_persons SET name = ? WHERE id = ?",
        (display_name, known_id),
    )
    conn.execute(
        """
        INSERT INTO person_body_encodings (known_person_id, encoding)
        VALUES (?, ?)
        """,
        (known_id, _feature_to_blob(feature)),
    )
    conn.commit()
    return known_id

def add_body_encoding(conn: sqlite3.Connection, known_id: int, feature: np.ndarray) -> None:
    conn.execute(
        """
        INSERT INTO person_body_encodings (known_person_id, encoding)
        VALUES (?, ?)
        """,
        (known_id, _feature_to_blob(feature)),
    )


def _assign_identity(
    conn: sqlite3.Connection,
    feature: np.ndarray,
    match_fn,
    register_fn,
    add_fn,
    label: str,
) -> int:
    known_id, dist = match_fn(feature, conn)
    try:
        match_dist = float(dist)
    except (TypeError, ValueError):
        match_dist = float("inf")
    if known_id is not None and is_named_person(conn, int(known_id)):
        if not np.isfinite(match_dist) or match_dist > float(NAMED_MATCH_THRESHOLD):
            known_id = None
    if known_id is None:
        known_id = register_fn(conn, feature)
        print(f"[NEW PERSON] Registered {label} identity #{known_id}")
    else:
        touch_person(conn, known_id)
        skip_encode = is_named_person(conn, int(known_id))
        if not skip_encode:
            add_fn(conn, known_id, feature)
    conn.commit()
    return known_id


def resolve_known_person_from_face(
    conn: sqlite3.Connection,
    face_feature: np.ndarray,
    session_map: Dict[int, int],
    track_id: int,
    *,
    camera_location: Optional[str] = None,
) -> Optional[int]:
    """Assign identity from a face feature only (no body fallback)."""
    if track_id in session_map:
        return session_map[track_id]

    def _match(feature: np.ndarray, db: sqlite3.Connection):
        return find_best_face_match(
            feature,
            db,
            camera_location=camera_location,
        )

    known_id = _assign_identity(
        conn,
        face_feature,
        _match,
        register_person,
        add_encoding_for_person,
        "face",
    )
    if known_id is not None:
        session_map[track_id] = known_id
    return known_id


def resolve_known_person(
    conn: sqlite3.Connection,
    frame: np.ndarray,
    bbox: np.ndarray,
    session_map: Dict[int, int],
    track_id: int,
) -> Optional[int]:
    """
    Map a video track to a persistent known_person_id using face recognition.
    session_map caches track_id -> known_person_id within one run.
    """
    if track_id in session_map:
        return session_map[track_id]

    known_id = None

    if models_ready():
        try:
            face_feat = encoding_from_frame_bbox(frame, bbox)
        except FileNotFoundError:
            face_feat = None
        if face_feat is not None:
            known_id = _assign_identity(
                conn,
                face_feat,
                find_best_face_match,
                register_person,
                add_encoding_for_person,
                "face",
            )

    if known_id is None:
        body_feat = body_embedding_from_frame_bbox(frame, bbox)
        if body_feat is not None:
            known_id = _assign_identity(
                conn,
                body_feat,
                find_best_body_match,
                register_person_body,
                add_body_encoding,
                "body",
            )

    if known_id is not None:
        session_map[track_id] = known_id
    return known_id


def get_visit_stats(
    conn: sqlite3.Connection, known_person_id: int
) -> Tuple[int, Optional[str], List[Tuple[str, int]]]:
    """Return (total_visits, last_seen, [(location, count), ...])."""
    rows = conn.execute(
        """
        SELECT camera_location, COUNT(*) AS cnt, MAX(timestamp) AS last_ts
        FROM tracking_logs
        WHERE known_person_id = ?
        GROUP BY camera_location
        ORDER BY cnt DESC
        """,
        (known_person_id,),
    ).fetchall()
    total = sum(r[1] for r in rows)
    last_seen = None
    if rows:
        last_seen = max(r[2] for r in rows)
    by_location = [(r[0], r[1]) for r in rows]
    return total, last_seen, by_location


def _faces_from_frame_yolo(
    frame: np.ndarray, yolo_model
) -> List[Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]]:
    """
    Detect faces in a BGR frame via YuNet and YOLO person boxes.
    Returns (feature, aligned_face, sharpness, optional_crop) per face found.
    """
    found: List[Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]] = []
    if not models_ready():
        return found

    direct = _detect_largest_face(frame)
    if direct is not None:
        aligned, feat = direct
        crop = _crop_largest_face_with_padding(frame, pad_ratio=0.10)
        found.append((feat, aligned, face_sharpness(aligned), crop))

    if yolo_model is None:
        return found

    from orion_frame_pipeline import wrap_yolo_results

    results = _yolo_predict_persons(yolo_model, frame)
    first = wrap_yolo_results(results)
    if first is None:
        return found
    boxes = first.boxes
    if boxes is None or len(boxes) == 0:
        return found

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    for bbox in xyxy[areas.argsort()[::-1]]:
        face_result = face_from_person_bbox(frame, bbox)
        if face_result is None:
            continue
        feat, aligned, _area = face_result
        person_crop = crop_person(frame, bbox)
        found.append(
            (feat, aligned, face_sharpness(aligned), person_crop)
        )
    return found


def _match_candidates_from_face(
    conn: sqlite3.Connection,
    face_feat: np.ndarray,
    query_quality: float,
    face_crop: Optional[np.ndarray],
    profile_image: np.ndarray,
) -> Tuple[List[dict], bool, str]:
    """Run gallery match and optional profile upgrade; returns (candidates, upgraded, match_type)."""
    raw_matches = find_face_matches(
        face_feat,
        conn,
        query_quality=query_quality,
        threshold=PHOTO_SEARCH_MATCH_THRESHOLD,
        log_candidates=10,
    )
    candidates: List[dict] = []
    for item in raw_matches:
        person_id = item["person_id"]
        total, last_seen, by_location = get_visit_stats(conn, person_id)
        candidates.append(
            {
                "person_id": person_id,
                "known_person_id": person_id,
                "distance": item["distance"],
                "source": item.get("source", ""),
                "total_visits": total,
                "last_seen": last_seen,
                "by_location": by_location,
            }
        )
    profile_upgraded = False
    if candidates:
        profile_upgraded = upsert_best_face_profile(
            conn,
            candidates[0]["person_id"],
            face_feat,
            face_crop if face_crop is not None else profile_image,
            query_quality,
        )
    return candidates, profile_upgraded, "face"


def _format_video_timestamp(frame_idx: int, fps: float) -> str:
    """Format frame index as MM:SS for dashboard display."""
    if fps <= 0:
        fps = 25.0
    total_seconds = int(frame_idx / fps)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def resolve_original_video_filename(
    storage_name: Optional[str],
    original_filename: Optional[str] = None,
) -> str:
    """Human-readable CCTV filename for UI; storage_name may be a hashed cache path."""
    if original_filename and str(original_filename).strip():
        return Path(str(original_filename).strip()).name
    storage = str(storage_name or "").strip()
    if storage:
        return Path(storage).name
    return "unknown_video.mp4"


def _looks_like_hashed_storage_name(name: str) -> bool:
    """True when basename looks like an internal upload/cache id, not a real CCTV name."""
    base = Path(str(name or "")).name
    if not base:
        return False
    lower = base.lower()
    if lower.endswith("_video.mp4") or lower.endswith("_video.avi") or lower.endswith("_video.mov"):
        stem = Path(base).stem
        if stem.endswith("_video"):
            prefix = stem[:-6]
            if len(prefix) == 12 and all(c in "0123456789abcdef" for c in prefix.lower()):
                return True
    return False


def _distance_to_match_percent(distance: float) -> int:
    """Map cosine distance to a 0–100 match score (higher = stronger match)."""
    if not np.isfinite(distance):
        return 0
    similarity = max(0.0, min(1.0, 1.0 - float(distance)))
    return int(round(similarity * 100))


def confidence_score_from_distance(distance: float) -> int:
    """Single source of truth for UI confidence % (matches gallery + borderline log)."""
    return _distance_to_match_percent(distance)


def _finalize_scan_hit(hit: Dict) -> Dict:
    """Keep distance, match_percent, confidence_score, and label in sync after merge."""
    dist = float(hit.get("distance", 1.0))
    pct = confidence_score_from_distance(dist)
    hit["distance"] = round(dist, 4)
    hit["match_percent"] = pct
    hit["confidence_score"] = pct
    if pct >= VIDEO_TARGET_CONFIRM_MATCH_PERCENT:
        hit["label"] = f"TARGET MATCHED [{pct}%]"
    elif VIDEO_TARGET_BORDERLINE_MIN_PERCENT <= pct <= VIDEO_TARGET_BORDERLINE_MAX_PERCENT:
        hit["label"] = f"BORDERLINE [{pct}%]"
    else:
        hit["label"] = f"MATCH [{pct}%]"
    return hit


def normalize_detection_hit(hit: Dict) -> Dict:
    """Ensure API/JSON records expose a consistent confidence_score for the frontend."""
    if not isinstance(hit, dict):
        return hit
    out = dict(hit)
    if out.get("distance") is not None and np.isfinite(float(out["distance"])):
        pct = confidence_score_from_distance(float(out["distance"]))
    elif out.get("match_percent") is not None:
        pct = int(round(float(out["match_percent"])))
    elif out.get("confidence_score") is not None:
        pct = int(round(float(out["confidence_score"])))
    else:
        pct = 0
    out["match_percent"] = pct
    out["confidence_score"] = pct
    storage = out.get("video_file") or out.get("filename") or out.get("file") or ""
    if not out.get("original_filename"):
        if storage and not _looks_like_hashed_storage_name(str(storage)):
            out["original_filename"] = resolve_original_video_filename(storage)
        elif storage:
            out["original_filename"] = resolve_original_video_filename(storage)
    if pct >= VIDEO_TARGET_CONFIRM_MATCH_PERCENT:
        out["label"] = f"TARGET MATCHED [{pct}%]"
    elif VIDEO_TARGET_BORDERLINE_MIN_PERCENT <= pct <= VIDEO_TARGET_BORDERLINE_MAX_PERCENT:
        out["label"] = f"BORDERLINE [{pct}%]"
    return out


def normalize_scan_response(result: Dict) -> Dict:
    """Normalize matches and borderline_suspects before returning to the dashboard."""
    if not isinstance(result, dict):
        return result
    out = dict(result)
    top_original = out.get("original_filename")
    if not top_original:
        vf = out.get("video_file") or ""
        if vf and not _looks_like_hashed_storage_name(str(vf)):
            top_original = resolve_original_video_filename(vf)
            out["original_filename"] = top_original
    for key in ("matches", "borderline_suspects", "near_miss_suspects", "all_matches"):
        items = out.get(key)
        if isinstance(items, list):
            normalized: List[Dict] = []
            for h in items:
                nh = normalize_detection_hit(h)
                if top_original and (
                    not nh.get("original_filename")
                    or _looks_like_hashed_storage_name(str(nh.get("original_filename", "")))
                ):
                    nh["original_filename"] = top_original
                normalized.append(nh)
            out[key] = normalized
    all_hits = out.get("all_matches") or out.get("matches") or []
    if isinstance(all_hits, list):
        out["all_matches"] = sorted(
            all_hits,
            key=lambda h: int(
                h.get("confidence_score", h.get("match_percent", 0)) or 0
            ),
            reverse=True,
        )
        if not out.get("matches"):
            out["matches"] = out["all_matches"]
    out["scan_thresholds"] = {
        "borderline_min": VIDEO_TARGET_BORDERLINE_MIN_PERCENT,
        "confirmed_min": VIDEO_TARGET_CONFIRM_MATCH_PERCENT,
        "alert_min": VIDEO_SCAN_ALERT_MATCH_PERCENT,
    }
    return out


def _is_borderline_score(pct: int) -> bool:
    return VIDEO_TARGET_BORDERLINE_MIN_PERCENT <= pct <= VIDEO_TARGET_BORDERLINE_MAX_PERCENT


def _append_demoted_borderline(hit: Dict, bucket: List[Dict]) -> None:
    """Keep weaker borderline frames that lost a merge to a stronger nearby hit."""
    pct = confidence_score_from_distance(float(hit.get("distance", 1.0)))
    if _is_borderline_score(pct):
        finalized = _finalize_scan_hit(dict(hit))
        finalized["status"] = "borderline_demoted"
        bucket.append(finalized)


def _dedupe_borderline_hits(hits: List[Dict]) -> List[Dict]:
    seen = set()
    unique: List[Dict] = []
    for hit in hits:
        key = (
            str(hit.get("video_file", "")),
            int(hit.get("frame", -1)),
            float(hit.get("distance", 0)),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique


def _split_merged_scan_hits(merged: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Assign each merged cluster winner to confirmed OR borderline — never both."""
    confirmed: List[Dict] = []
    borderline: List[Dict] = []
    for hit in merged:
        finalized = _finalize_scan_hit(hit)
        pct = finalized["confidence_score"]
        if pct >= VIDEO_TARGET_CONFIRM_MATCH_PERCENT:
            confirmed.append(finalized)
        elif _is_borderline_score(pct):
            borderline.append(finalized)
    return confirmed, borderline


def _face_bbox_on_image(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return face bounding box as (x1, y1, x2, y2) on the given BGR image."""
    if not models_ready() or image is None or image.size == 0:
        return None
    detector, _ = load_face_models()
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = [float(v) for v in best[:4]]
    return int(x), int(y), int(x + fw), int(y + fh)


def _padded_person_rect(
    frame: np.ndarray, bbox: np.ndarray
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = frame.shape[:2]
    pad = int(0.1 * max(x2 - x1, y2 - y1))
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)


def _face_bbox_in_person_box(
    frame: np.ndarray, person_bbox: np.ndarray
) -> Optional[Tuple[int, int, int, int]]:
    """Map the largest YuNet face inside a YOLO person box to full-frame coordinates."""
    px1, py1, px2, py2 = _padded_person_rect(frame, person_bbox)
    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return None
    upscaled = _upscale_crop(crop)
    face_box = _face_bbox_on_image(upscaled)
    if face_box is None:
        return None
    ux1, uy1, ux2, uy2 = face_box
    ch, cw = crop.shape[:2]
    uh, uw = upscaled.shape[:2]
    sx = cw / max(float(uw), 1.0)
    sy = ch / max(float(uh), 1.0)
    fh, fw = frame.shape[:2]
    x1 = max(0, int(px1 + ux1 * sx))
    y1 = max(0, int(py1 + uy1 * sy))
    x2 = min(fw, int(px1 + ux2 * sx))
    y2 = min(fh, int(py1 + uy2 * sy))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _draw_target_match_overlay(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    distance: float,
) -> np.ndarray:
    """Draw neon blue target box and match label on a BGR frame copy."""
    annotated = frame.copy()
    x1, y1, x2, y2 = bbox
    color = TARGET_TRACK_BOX_BGR
    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        color,
        TARGET_TRACK_BOX_THICKNESS,
        cv2.LINE_AA,
    )
    pct = _distance_to_match_percent(distance)
    if pct >= VIDEO_TARGET_CONFIRM_MATCH_PERCENT:
        label = f"TARGET MATCHED [{pct}%]"
    elif VIDEO_TARGET_BORDERLINE_MIN_PERCENT <= pct <= VIDEO_TARGET_BORDERLINE_MAX_PERCENT:
        label = f"BORDERLINE [{pct}%]"
    else:
        label = f"MATCH [{pct}%]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = TARGET_TRACK_LABEL_THICKNESS
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    ty = max(th + baseline + 6, y1 - 10)
    tx = max(0, x1)
    cv2.rectangle(
        annotated,
        (tx, ty - th - baseline - 6),
        (tx + tw + 8, ty + baseline + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        annotated,
        label,
        (tx + 4, ty),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def _annotated_match_snippet(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    distance: float,
    pad_ratio: float = 0.25,
) -> np.ndarray:
    """Crop a square annotated region around the matched face for gallery thumbnails."""
    annotated = _draw_target_match_overlay(frame, bbox, distance)
    snippet = square_padded_crop(annotated, bbox, pad_ratio=pad_ratio)
    return snippet if snippet.size else annotated


FaceScanHit = Tuple[
    Tuple[int, int, int, int],
    Optional[Dict[str, Tuple[float, float]]],
]


def _scan_hits_from_detections(
    dets,
    frame: np.ndarray,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    coord_scale: float = 1.0,
) -> List[FaceScanHit]:
    """Convert RetinaFace detections to scan face boxes in work_frame coordinates."""
    from face_analyzer import _is_usable_face_scan

    hits: List[FaceScanHit] = []
    for det in dets:
        ok, _reason = _is_usable_face_scan(
            det,
            frame,
            min_confidence=VIDEO_SCAN_RETINAFACE_CONF,
            max_blur=VIDEO_SCAN_MAX_BLUR,
        )
        if not ok:
            continue
        dx1, dy1, dx2, dy2 = det.bbox
        face_box = (
            int(offset_x + dx1 * coord_scale),
            int(offset_y + dy1 * coord_scale),
            int(offset_x + dx2 * coord_scale),
            int(offset_y + dy2 * coord_scale),
        )
        lm: Optional[Dict[str, Tuple[float, float]]] = None
        if det.landmarks:
            lm = {
                name: (
                    offset_x + float(pt[0]) * coord_scale,
                    offset_y + float(pt[1]) * coord_scale,
                )
                for name, pt in det.landmarks.items()
            }
        hits.append((face_box, lm))
    return hits


def _face_boxes_for_scan_frame(
    work_frame: np.ndarray,
    *,
    scan_frame_idx: Optional[int] = None,
    source_frame: Optional[np.ndarray] = None,
) -> List[FaceScanHit]:
    """
    RetinaFace on downscaled work frame; boxes mapped back to work coords.

    ArcFace embedding uses the full-res source crop via _feature_from_face_bbox.
    """
    analyzer = _get_gpu_scan_analyzer()
    if analyzer is None or work_frame is None or work_frame.size == 0:
        if VIDEO_SCAN_VERBOSE and scan_frame_idx is not None:
            print(f"Frame {scan_frame_idx}: 0 faces found, sizes=[]")
        return []

    scale = float(VIDEO_SCAN_DETECT_UPSCALE)
    if scale > 1.0:
        detect_frame = cv2.resize(
            work_frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )
        coord_scale = 1.0 / scale
    else:
        detect_frame = work_frame
        coord_scale = 1.0

    raw_dets = analyzer.detect(detect_frame)
    if VIDEO_SCAN_VERBOSE and scan_frame_idx is not None:
        sizes = [
            f"{int((det.bbox[2] - det.bbox[0]) * coord_scale)}"
            f"x{int((det.bbox[3] - det.bbox[1]) * coord_scale)}"
            for det in raw_dets
        ]
        print(
            f"Frame {scan_frame_idx}: {len(raw_dets)} faces found, sizes={sizes}"
        )

    hits = _scan_hits_from_detections(
        raw_dets,
        detect_frame,
        coord_scale=coord_scale,
    )

    if (
        scan_frame_idx is not None
        and scan_frame_idx in VIDEO_SCAN_DEBUG_FRAME_INDICES
        and source_frame is not None
        and source_frame.size > 0
    ):
        VIDEO_SCAN_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        out = VIDEO_SCAN_DEBUG_DIR / f"frame_{scan_frame_idx}.jpg"
        cv2.imwrite(str(out), source_frame)
        print(f"[SCAN DEBUG] saved {out}")

    return hits


def _faces_from_frame_gpu_yolo_with_bboxes(
    frame: np.ndarray,
    yolo_model,
    *,
    yolo_first: bool = False,
    yolo_result=None,
    source_frame: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray], Tuple[int, int, int, int]]]:
    """
    InsightFace CUDA path for video scan — RetinaFace detect + ArcFace embed on GPU.
    """
    found: List[
        Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray], Tuple[int, int, int, int]]
    ] = []
    analyzer = _get_gpu_scan_analyzer()
    if analyzer is None:
        return found

    host = source_frame if source_frame is not None else frame

    from orion_frame_pipeline import wrap_yolo_results

    if yolo_model is not None and yolo_first:
        first = yolo_result
        if first is None:
            results = _yolo_predict_persons(yolo_model, frame, for_scan=True)
            first = wrap_yolo_results(results)
        if first is not None and first.boxes is not None and len(first.boxes) > 0:
            xyxy = first.boxes.xyxy.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            for bbox in xyxy[areas.argsort()[::-1]]:
                from face_analyzer import extract_face_from_person_crop

                hit = extract_face_from_person_crop(
                    frame, bbox, analyzer=analyzer, source_frame=host
                )
                if hit is None:
                    continue
                feat, aligned, _area = hit
                person_crop = crop_person(frame, bbox)
                face_box = _face_bbox_in_person_box(frame, bbox)
                if face_box is None:
                    x1, y1, x2, y2 = bbox.astype(int)
                    face_box = (x1, y1, x2, y2)
                found.append(
                    (feat, aligned, face_sharpness(aligned), person_crop, face_box)
                )

    if found:
        return found

    dets = analyzer.detect_and_embed_scan(frame)
    fh, fw = frame.shape[:2]
    for det in dets:
        if not det.quality_ok or det.embedding is None:
            continue
        x1, y1, x2, y2 = det.bbox
        pad = int(0.1 * max(x2 - x1, y2 - y1))
        crop = frame[
            max(0, y1 - pad) : min(fh, y2 + pad),
            max(0, x1 - pad) : min(fw, x2 + pad),
        ]
        aligned = crop if crop.size else frame[y1:y2, x1:x2]
        found.append(
            (
                det.embedding,
                aligned,
                face_sharpness(aligned),
                crop if crop.size else None,
                (x1, y1, x2, y2),
            )
        )
    return found


def _faces_from_frame_yolo_with_bboxes(
    frame: np.ndarray,
    yolo_model,
    *,
    yolo_first: bool = False,
    yolo_result=None,
    gpu_only: bool = False,
    source_frame: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray], Tuple[int, int, int, int]]]:
    """
    Like _faces_from_frame_yolo but includes full-frame face bbox (x1, y1, x2, y2)
    for CCTV-style overlay thumbnails.

    When yolo_first is True (video target scan), try YOLO person boxes first on the
    downscaled frame, then fall back to relaxed full-frame YuNet if no faces are found.

    When gpu_only is True (dashboard video scan), never fall back to CPU YuNet/OpenVINO.
    """
    if gpu_scan_ready():
        gpu_found = _faces_from_frame_gpu_yolo_with_bboxes(
            frame,
            yolo_model,
            yolo_first=yolo_first,
            yolo_result=yolo_result,
            source_frame=source_frame,
        )
        if gpu_found or gpu_only:
            return gpu_found

    found: List[
        Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray], Tuple[int, int, int, int]]
    ] = []
    if not models_ready():
        return found

    if not yolo_first:
        for aligned, feat, face_box in _yunet_extract_faces(frame):
            crop = _crop_largest_face_with_padding(frame, pad_ratio=0.10)
            found.append((feat, aligned, face_sharpness(aligned), crop, face_box))
            break
        if found:
            return found

    if yolo_model is not None:
        from orion_frame_pipeline import wrap_yolo_results

        first = yolo_result
        if first is None:
            results = _yolo_predict_persons(yolo_model, frame, for_scan=yolo_first)
            first = wrap_yolo_results(results)
        if first is None:
            return found
        boxes = first.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            for bbox in xyxy[areas.argsort()[::-1]]:
                face_result = face_from_person_bbox(
                    frame,
                    bbox,
                    upscale_min_side=VIDEO_SCAN_PERSON_UPSCALE_MIN_SIDE,
                )
                if face_result is None:
                    continue
                feat, aligned, _area = face_result
                person_crop = crop_person(frame, bbox)
                face_box = _face_bbox_in_person_box(frame, bbox)
                if face_box is None:
                    continue
                found.append(
                    (feat, aligned, face_sharpness(aligned), person_crop, face_box)
                )

    if found:
        return found

    fh, fw = frame.shape[:2]
    for aligned, feat, face_box in _yunet_extract_faces(
        frame,
        score_threshold=VIDEO_SCAN_SCORE_THRESHOLD,
        upscale_min_side=VIDEO_SCAN_UPSCALE_MIN_SIDE,
    ):
        x1, y1, x2, y2 = face_box
        pad = int(0.1 * max(x2 - x1, y2 - y1))
        crop = frame[
            max(0, y1 - pad) : min(fh, y2 + pad),
            max(0, x1 - pad) : min(fw, x2 + pad),
        ]
        found.append((feat, aligned, face_sharpness(aligned), crop, face_box))
    return found


def _crop_to_data_url(crop: np.ndarray, quality: int = 82) -> str:
    """Encode a BGR crop as a JPEG data URL for JSON responses."""
    if crop is None or crop.size == 0:
        return ""
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _merge_target_scan_matches(
    raw: List[Dict], fps: float, gap_sec: float = VIDEO_TARGET_MERGE_GAP_SEC
) -> Tuple[List[Dict], List[Dict]]:
    """
    Collapse nearby hits; keep the strongest match per cluster.
    Weaker borderline (50–57%) frames replaced during merge are returned separately
    so the dashboard log can show them with their true confidence_score.
    """
    demoted_borderline: List[Dict] = []
    if not raw:
        return [], demoted_borderline
    if fps <= 0:
        return [_finalize_scan_hit(h) for h in raw], demoted_borderline

    gap_frames = max(int(gap_sec * fps), 1)
    sorted_hits = sorted(raw, key=lambda item: item["frame"])
    merged: List[Dict] = []

    for hit in sorted_hits:
        if not merged:
            merged.append(hit)
            continue
        prev = merged[-1]
        if hit["frame"] - prev["frame"] <= gap_frames:
            if hit["distance"] < prev["distance"]:
                _append_demoted_borderline(prev, demoted_borderline)
                merged[-1] = hit
            else:
                _append_demoted_borderline(hit, demoted_borderline)
        else:
            merged.append(hit)

    winners = [_finalize_scan_hit(h) for h in merged]
    return winners, _dedupe_borderline_hits(demoted_borderline)


def _load_target_feature(
    target_image_path: Path, yolo_model=None
) -> Tuple[Optional[np.ndarray], str]:
    """Load target face embedding from a reference image. Returns (feature, error_message)."""
    if not scan_models_ready():
        return None, (
            "Face models not installed. Run: python download_face_models.py "
            "and ensure onnxruntime-gpu + insightface for GPU scan."
        )

    target_img = cv2.imread(str(target_image_path))
    if target_img is None:
        return None, "Could not read target face image."

    use_gpu = gpu_scan_ready()
    if use_gpu:
        _clear_scan_embed_diagnostics()
        bank, labels, primary_meta = _build_augmented_reference_embeddings(target_img)
        if bank is not None and bank.shape[0] > 0:
            target_feat = bank
            if VIDEO_SCAN_EMBED_DIAG:
                global _SCAN_DIAG_REF
                primary = _reference_bank_primary_feat(bank)
                _SCAN_DIAG_REF = {
                    "feat": primary.copy(),
                    "feats": bank.copy(),
                    "labels": list(labels),
                    "debug": (
                        primary_meta.get("debug")
                        if isinstance(primary_meta, dict)
                        else None
                    ),
                    "meta": primary_meta,
                }
            print(
                "Target reference embedded: "
                f"{bank.shape[0]}/{len(_iter_reference_augmentations(target_img))} "
                f"augmented variants"
                + (
                    f", crop_size={primary_meta.get('crop_w')}x{primary_meta.get('crop_h')}"
                    if isinstance(primary_meta, dict)
                    else ""
                )
            )
        else:
            target_feat = None
    else:
        target_feat = None
    if target_feat is None:
        target_feat = encoding_from_bgr_image(target_img, yolo_model=yolo_model)
    if target_feat is None:
        direct = _detect_largest_face(target_img)
        if direct is not None:
            target_feat = direct[1]
    if not _target_reference_valid(target_feat):
        return None, (
            "No face detected in target image. Use a clear photo of the person's face."
        )
    return target_feat, ""


def _video_extension(name: str) -> str:
    """Case-insensitive extension check helper (e.g. '.MP4' -> '.mp4')."""
    return Path(name).suffix.lower()


def normalize_batch_folder_path(
    folder_path: Union[str, Path],
    storage_mode: str = "local",
) -> Path:
    """Normalize local filesystem or network share paths for batch folder scans."""
    raw = str(folder_path).strip().strip('"').strip("'")
    mode = (storage_mode or "local").strip().lower()
    if mode not in ("local", "server"):
        mode = "local"

    # Stream URLs are not filesystem paths — do not apply SCAN_ROOT_ALLOWLIST.
    lower = raw.lower()
    if lower.startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://")):
        return Path(raw)

    if mode == "server":
        text = raw
        if text.lower().startswith("file:"):
            parsed = urlparse(text)
            host = parsed.netloc or ""
            path_part = unquote(parsed.path or "")
            if os.name == "nt" and host:
                text = f"\\\\{host}{path_part.replace('/', os.sep)}"
            elif host:
                text = f"//{host}{path_part}"
            else:
                text = path_part
        if os.name == "nt":
            text = text.replace("/", "\\")
            if text.startswith("//") and not text.startswith("\\\\"):
                text = "\\" + text.lstrip("/").replace("/", "\\")
            while len(text) > 3 and text.endswith("\\"):
                text = text.rstrip("\\")
            from security_bootstrap import assert_scan_path_allowed

            return assert_scan_path_allowed(Path(text), storage_mode=mode)
        from security_bootstrap import assert_scan_path_allowed

        return assert_scan_path_allowed(Path(text.rstrip("/")), storage_mode=mode)

    folder = Path(raw).expanduser()
    try:
        folder = folder.resolve()
    except OSError:
        pass
    try:
        from security_bootstrap import assert_scan_path_allowed

        return assert_scan_path_allowed(folder, storage_mode=mode)
    except PermissionError:
        raise
    except Exception:
        return folder


def _is_folder_scan_video(path: Union[str, Path]) -> bool:
    return _video_extension(str(path)) in FOLDER_SCAN_VIDEO_EXTENSIONS


def collect_videos_in_folder(
    folder_path: Union[str, Path],
    storage_mode: str = "local",
) -> List[Path]:
    """Return sorted video files in a directory (non-recursive)."""
    folder = normalize_batch_folder_path(folder_path, storage_mode)
    if not folder.is_dir():
        return []

    extensions = {ext.lower() for ext in FOLDER_SCAN_VIDEO_EXTENSIONS}
    found: List[Path] = []
    seen = set()

    def _add(candidate: Path) -> None:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        key = str(resolved).lower()
        if key in seen:
            return
        if not resolved.is_file():
            return
        if _video_extension(resolved.name) not in extensions:
            return
        seen.add(key)
        found.append(resolved)

    try:
        for name in os.listdir(folder):
            _add(folder / name)
    except OSError:
        pass

    try:
        for entry in folder.iterdir():
            if entry.is_file():
                _add(entry)
    except OSError:
        pass

    pattern_base = str(folder / "*")
    for ext in FOLDER_SCAN_VIDEO_EXTENSIONS:
        for variant in (ext, ext.upper(), ext.capitalize()):
            for match in glob.glob(pattern_base + variant):
                _add(Path(match))

    return sorted(found, key=lambda p: p.name.lower())


def scan_min_match_percent(
    match_threshold: float = VIDEO_TARGET_MATCH_THRESHOLD,
) -> int:
    """Percent gate for scan hits — minimum similarity to include in results."""
    return VIDEO_TARGET_CONFIRM_MATCH_PERCENT


def build_all_scan_hits_sorted(raw_hits: List[Dict]) -> List[Dict]:
    """All per-frame hits ≥ display minimum (30%), sorted by similarity (highest first)."""
    min_pct = VIDEO_TARGET_CONFIRM_MATCH_PERCENT
    seen = set()
    out: List[Dict] = []
    for hit in raw_hits:
        finalized = _finalize_scan_hit(dict(hit))
        pct = int(finalized.get("confidence_score", 0))
        if pct < min_pct:
            continue
        key = (str(finalized.get("video_file", "")), int(finalized.get("frame", -1)))
        if key in seen:
            continue
        seen.add(key)
        out.append(finalized)
    out.sort(key=lambda h: int(h.get("confidence_score", 0)), reverse=True)
    return out


def tier_scan_hits_by_percent(
    hits: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Split sorted hits into confirmed (≥30%) and borderline (unused — sub-30% dropped upstream)."""
    confirmed: List[Dict] = []
    borderline: List[Dict] = []
    for hit in hits:
        pct = int(hit.get("confidence_score", 0))
        if pct >= VIDEO_TARGET_CONFIRM_MATCH_PERCENT:
            confirmed.append(hit)
    return confirmed, borderline


def scan_video_for_target(
    target_image_path: Path,
    video_path: Path,
    yolo_model=None,
    frame_step: int = VIDEO_TARGET_SCAN_FRAME_STEP,
    match_threshold: float = VIDEO_TARGET_MATCH_THRESHOLD,
    target_feat: Optional[np.ndarray] = None,
    progress_callback=None,
    match_score_callback=None,
    scan_id: Optional[str] = None,
    video_index: int = 0,
    videos_total: int = 1,
    require_cuda: bool = True,
    original_filename: Optional[str] = None,
    face_demographics_callback=None,
    target_hit_callback=None,
) -> dict:
    """
    Dashboard video target scan — delegates to Orion VideoScanExecutor (pure GPU).

    Stride sampling (default every 5th frame) @ 50% detect downscale, RetinaFace + ArcFace on CUDA.
    """
    from orion_frame_pipeline import VideoScanExecutor

    return VideoScanExecutor.scan_for_target(
        target_image_path,
        video_path,
        yolo_model=yolo_model,
        frame_step=frame_step,
        match_threshold=match_threshold,
        target_feat=target_feat,
        progress_callback=progress_callback,
        match_score_callback=match_score_callback,
        scan_id=scan_id,
        video_index=video_index,
        videos_total=videos_total,
        require_cuda=require_cuda,
        original_filename=original_filename,
        face_demographics_callback=face_demographics_callback,
        target_hit_callback=target_hit_callback,
    )


def _thread_safe_match_score_callback(
    callback,
) -> Optional[Callable[[float], None]]:
    if callback is None:
        return None
    lock = threading.Lock()

    def _emit(score: float) -> None:
        with lock:
            callback(score)

    return _emit


def _thread_safe_target_hit_callback(
    callback,
) -> Optional[Callable[[dict], None]]:
    if callback is None:
        return None
    lock = threading.Lock()

    def _emit(hit: dict) -> None:
        with lock:
            callback(hit)

    return _emit


def _thread_safe_video_start_callback(
    callback,
) -> Optional[Callable[[dict], None]]:
    if callback is None:
        return None
    lock = threading.Lock()

    def _emit(meta: dict) -> None:
        with lock:
            callback(meta)

    return _emit


def _thread_safe_progress_callback(
    callback,
) -> Optional[Callable[[dict], None]]:
    if callback is None:
        return None
    lock = threading.Lock()

    def _emit(payload: dict) -> None:
        with lock:
            callback(payload)

    return _emit


def _ingest_folder_video_result(
    result: dict,
    vname: str,
    *,
    skipped: List[str],
    processed_videos: List[str],
) -> Tuple[bool, int, int, int]:
    """
    Merge one per-video scan result into folder accumulators.

    Returns (counted_as_scanned, frames_done, faces_compared, best_match_pct).
    """
    msg = str(result.get("message", "") or "")
    has_hits = bool(result.get("matches")) or bool(
        result.get("borderline_suspects")
    ) or bool(result.get("near_miss_suspects"))
    frames_done = int(result.get("frames_scanned") or 0)
    faces = int(result.get("faces_compared") or 0)
    best_pct = int(result.get("best_match_percent") or 0)
    unreadable = "Could not open" in msg or "unsupported" in msg.lower()

    if unreadable:
        skipped.append(f"{vname} ({msg})")
        return False, frames_done, faces, best_pct

    if not result.get("success") and not has_hits and frames_done <= 0:
        skipped.append(f"{vname} ({msg or 'scan did not complete'})")
        return False, frames_done, faces, best_pct

    processed_videos.append(vname)
    return True, frames_done, faces, best_pct


def _folder_scan_worker(
    video_index: int,
    video_path: Path,
    *,
    target_image_path: Path,
    yolo_model,
    frame_step: int,
    match_threshold: float,
    target_feat: np.ndarray,
    progress_callback,
    match_score_callback=None,
    scan_id: Optional[str],
    videos_total: int,
    face_demographics_callback=None,
    target_hit_callback=None,
    video_start_callback=None,
) -> Tuple[int, str, dict]:
    """Scan one folder video on this thread's CUDA stream."""
    from inference_device import cuda_stream_context, thread_cuda_stream

    vname = video_path.name
    if video_start_callback is not None:
        try:
            video_start_callback(
                {
                    "video_index": video_index,
                    "video_file": vname,
                    "original_filename": vname,
                }
            )
        except Exception:
            pass
    with cuda_stream_context(sync=False):
        result = scan_video_for_target(
            target_image_path,
            video_path,
            yolo_model=yolo_model,
            frame_step=frame_step,
            match_threshold=match_threshold,
            target_feat=target_feat,
            progress_callback=progress_callback,
            match_score_callback=match_score_callback,
            scan_id=scan_id,
            video_index=video_index,
            videos_total=videos_total,
            require_cuda=False,
            original_filename=video_path.name,
            face_demographics_callback=face_demographics_callback,
            target_hit_callback=target_hit_callback,
        )
        stream = thread_cuda_stream()
        if stream is not None:
            stream.synchronize()
    return video_index, vname, result


def scan_folder_for_target(
    target_image_path: Path,
    folder_path: Union[str, Path],
    yolo_model=None,
    frame_step: int = VIDEO_TARGET_SCAN_FRAME_STEP,
    match_threshold: float = VIDEO_TARGET_MATCH_THRESHOLD,
    progress_callback=None,
    match_score_callback=None,
    scan_id: Optional[str] = None,
    storage_mode: str = "local",
    face_demographics_callback=None,
    target_hit_callback=None,
    video_start_callback=None,
) -> dict:
    """
    Batch-scan every video in a folder for appearances of the target face.

    Up to FOLDER_SCAN_PARALLEL_WORKERS videos run in parallel, each on its own
    CUDA stream (ThreadPoolExecutor).
    """
    folder = normalize_batch_folder_path(folder_path, storage_mode)
    empty = {
        "success": False,
        "matches": [],
        "borderline_suspects": [],
        "frames_scanned": 0,
        "videos_scanned": 0,
        "videos_found": 0,
        "folder_path": str(folder),
        "storage_mode": storage_mode,
        "skipped_videos": [],
        "processed_videos": [],
        "message": "",
    }

    if yolo_model is None:
        empty["message"] = "Search model unavailable."
        return empty

    if not folder.is_dir():
        empty["message"] = f"Folder not found: {folder}"
        return empty

    videos = collect_videos_in_folder(folder, storage_mode)
    empty["videos_found"] = len(videos)
    if not videos:
        empty["message"] = (
            f"No video files (.mp4, .avi, .mkv, .mov, .webm) found in {folder}"
        )
        return empty

    target_feat, err = _load_target_feature(target_image_path, yolo_model=yolo_model)
    if target_feat is None:
        empty["message"] = err
        return empty

    from orion_frame_pipeline import require_cuda_video_scan

    require_cuda_video_scan()

    safe_progress = _thread_safe_progress_callback(progress_callback)
    safe_match_score = _thread_safe_match_score_callback(match_score_callback)
    safe_target_hit = _thread_safe_target_hit_callback(target_hit_callback)
    safe_video_start = _thread_safe_video_start_callback(video_start_callback)

    all_matches: List[Dict] = []
    all_borderline: List[Dict] = []
    all_near_miss: List[Dict] = []
    total_frames = 0
    faces_compared = 0
    best_match_pct = 0
    videos_scanned = 0
    skipped: List[str] = []
    processed_videos: List[str] = []
    results_by_index: Dict[int, Tuple[str, dict]] = {}

    workers = min(FOLDER_SCAN_PARALLEL_WORKERS, len(videos))
    if workers <= 1:
        for video_index, video_path in enumerate(videos):
            vname = video_path.name
            try:
                _, _, result = _folder_scan_worker(
                    video_index,
                    video_path,
                    target_image_path=target_image_path,
                    yolo_model=yolo_model,
                    frame_step=frame_step,
                    match_threshold=match_threshold,
                    target_feat=target_feat,
                    progress_callback=safe_progress,
                    match_score_callback=safe_match_score,
                    scan_id=scan_id,
                    videos_total=len(videos),
                    face_demographics_callback=face_demographics_callback,
                    target_hit_callback=safe_target_hit,
                    video_start_callback=safe_video_start,
                )
            except Exception as exc:
                skipped.append(f"{vname} ({exc})")
                continue
            results_by_index[video_index] = (vname, result)
    else:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="folder-scan"
        ) as pool:
            futures = {
                pool.submit(
                    _folder_scan_worker,
                    video_index,
                    video_path,
                    target_image_path=target_image_path,
                    yolo_model=yolo_model,
                    frame_step=frame_step,
                    match_threshold=match_threshold,
                    target_feat=target_feat,
                    progress_callback=safe_progress,
                    match_score_callback=safe_match_score,
                    scan_id=scan_id,
                    videos_total=len(videos),
                    face_demographics_callback=face_demographics_callback,
                    target_hit_callback=safe_target_hit,
                    video_start_callback=safe_video_start,
                ): video_index
                for video_index, video_path in enumerate(videos)
            }
            for future in as_completed(futures):
                video_index = futures[future]
                vname = videos[video_index].name
                try:
                    idx, name, result = future.result()
                    results_by_index[idx] = (name, result)
                except Exception as exc:
                    skipped.append(f"{vname} ({exc})")

    for video_index in sorted(results_by_index):
        vname, result = results_by_index[video_index]
        try:
            counted, frames_done, faces, best_pct = _ingest_folder_video_result(
                result,
                vname,
                skipped=skipped,
                processed_videos=processed_videos,
            )
        except Exception as exc:
            skipped.append(f"{vname} ({exc})")
            continue

        faces_compared += faces
        best_match_pct = max(best_match_pct, best_pct)
        if counted:
            videos_scanned += 1
            total_frames += frames_done
        for match in result.get("matches", []):
            ts = match.get("timestamp", "00:00")
            match["video_file"] = vname
            match["original_filename"] = resolve_original_video_filename(vname, match.get("original_filename"))
            match["location"] = f"{match['original_filename']} at {ts}"
            all_matches.append(match)
        for suspect in result.get("borderline_suspects", []):
            ts = suspect.get("timestamp", "00:00")
            suspect["video_file"] = vname
            suspect["original_filename"] = resolve_original_video_filename(vname, suspect.get("original_filename"))
            suspect["location"] = f"{suspect['original_filename']} at {ts}"
            all_borderline.append(suspect)
        for suspect in result.get("near_miss_suspects", []):
            ts = suspect.get("timestamp", "00:00")
            suspect["video_file"] = vname
            suspect["original_filename"] = resolve_original_video_filename(vname, suspect.get("original_filename"))
            suspect["location"] = f"{suspect['original_filename']} at {ts}"
            all_near_miss.append(suspect)

    all_near_miss = _finalize_near_miss_hits(all_near_miss)

    all_matches.sort(
        key=lambda m: int(
            m.get("confidence_score", m.get("match_percent", 0)) or 0
        ),
        reverse=True,
    )
    all_borderline.sort(
        key=lambda m: int(
            m.get("confidence_score", m.get("match_percent", 0)) or 0
        ),
        reverse=True,
    )

    if not all_matches and not all_borderline and not all_near_miss:
        return {
            "success": True,
            "matches": [],
            "all_matches": [],
            "borderline_suspects": [],
            "near_miss_suspects": [],
            "frames_scanned": total_frames,
            "faces_compared": faces_compared,
            "best_match_percent": best_match_pct,
            "videos_scanned": videos_scanned,
            "videos_found": len(videos),
            "folder_path": str(folder),
            "skipped_videos": skipped,
            "processed_videos": processed_videos,
            "message": (
                f"No matches in {videos_scanned} of {len(videos)} video(s) "
                f"from {folder.name}."
                + (
                    f" Best similarity was {best_match_pct}% "
                    f"(need ≥{scan_min_match_percent()}% for a hit)."
                    if best_match_pct > 0
                    else ""
                )
            ),
        }

    return {
        "success": True,
        "matches": all_matches,
        "all_matches": all_matches,
        "borderline_suspects": all_borderline,
        "near_miss_suspects": all_near_miss,
        "frames_scanned": total_frames,
        "faces_compared": faces_compared,
        "best_match_percent": best_match_pct,
        "videos_scanned": videos_scanned,
        "videos_found": len(videos),
        "folder_path": str(folder),
        "skipped_videos": skipped,
        "processed_videos": processed_videos,
            "message": (
                f"Found {len(all_matches)} confirmed and {len(all_borderline)} "
                f"borderline appearance(s) across {videos_scanned} of "
                f"{len(videos)} video(s) in {folder.name}"
                f" ({workers}-wide parallel scan)."
            ),
    }


def lookup_video(
    conn: sqlite3.Connection, video_path: Path, yolo_model=None
) -> dict:
    """Pick the sharpest face across video frames, then match like photo search."""
    empty = {
        "matched": False,
        "candidates": [],
        "query_quality": None,
        "match_type": "",
        "profile_upgraded": False,
        "message": "",
        "distance": None,
        "frames_scanned": 0,
        "video_frames_total": 0,
    }

    if yolo_model is None:
        empty["message"] = "Search model unavailable."
        return empty
    if not models_ready():
        empty["message"] = (
            "Face models not installed. Run: python download_face_models.py"
        )
        return empty

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        empty["message"] = "Could not open video file."
        return empty

    best_sharp = -1.0
    best_feat: Optional[np.ndarray] = None
    best_aligned: Optional[np.ndarray] = None
    best_crop: Optional[np.ndarray] = None
    frame_idx = 0
    scanned = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    try:
        while scanned < VIDEO_SEARCH_MAX_FRAMES:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % VIDEO_SEARCH_FRAME_STEP == 0:
                for feat, aligned, sharp, crop in _faces_from_frame_yolo(frame, yolo_model):
                    if sharp > best_sharp:
                        best_sharp = sharp
                        best_feat = feat
                        best_aligned = aligned
                        best_crop = crop
                scanned += 1
            frame_idx += 1
    finally:
        cap.release()

    empty["frames_scanned"] = scanned
    empty["video_frames_total"] = total_frames if total_frames > 0 else frame_idx

    if best_feat is None or best_aligned is None:
        empty["message"] = (
            "No face detected in video. Use footage with a clear, visible face."
        )
        return empty

    if _normalize_face_feature(best_feat) is None:
        empty["message"] = (
            "Invalid face embedding from video. Try a clearer clip."
        )
        return empty

    query_quality = best_sharp if best_sharp >= 0 else face_sharpness(best_aligned)
    candidates, profile_upgraded, match_type = _match_candidates_from_face(
        conn,
        best_feat,
        query_quality,
        best_crop,
        best_aligned,
    )

    if not candidates:
        empty["message"] = "No matching person in database."
        empty["query_quality"] = query_quality
        return empty

    best = candidates[0]
    return {
        "matched": True,
        "candidates": candidates,
        "known_person_id": best["person_id"],
        "person_id": best["person_id"],
        "distance": best["distance"],
        "match_type": f"{match_type} (video)",
        "total_visits": best["total_visits"],
        "last_seen": best["last_seen"],
        "by_location": best["by_location"],
        "query_quality": query_quality,
        "profile_upgraded": profile_upgraded,
        "frames_scanned": scanned,
        "video_frames_total": empty["video_frames_total"],
    }


def lookup_photo(
    conn: sqlite3.Connection, image_path: Path, yolo_model=None
) -> dict:
    """Match a photo to known persons (face first, then body for CCTV)."""
    empty = {
        "matched": False,
        "candidates": [],
        "query_quality": None,
        "match_type": "",
        "profile_upgraded": False,
        "message": "",
        "distance": None,
    }

    img = cv2.imread(str(image_path))
    if img is None:
        empty["message"] = "Could not read uploaded image."
        return empty

    if yolo_model is None:
        empty["message"] = "Search model unavailable."
        return empty

    match_type = ""
    query_quality = 0.0
    profile_upgraded = False
    face_feat = None
    face_crop = None
    candidates: List[dict] = []

    # Prefer InsightFace ArcFace (512-d) — same model as the CCTV gallery.
    # OpenCV SFace (128-d) cannot match ArcFace embeddings and always misses.
    packed = _embed_scan_reference_face(img)
    if packed is not None:
        face_feat, meta = packed
        if isinstance(meta, dict):
            debug = meta.get("debug") if isinstance(meta.get("debug"), dict) else {}
            aligned = debug.get("aligned_bgr")
            if aligned is not None and getattr(aligned, "size", 0):
                face_crop = aligned
            query_quality = float(meta.get("quality") or 0.0) or face_sharpness(
                face_crop if face_crop is not None else img
            )
        if face_crop is None:
            face_crop = _crop_largest_face_with_padding(img, pad_ratio=0.10)
            if query_quality <= 0:
                query_quality = face_sharpness(
                    face_crop if face_crop is not None else img
                )

    if face_feat is None and models_ready():
        # CPU / legacy fallback only when ArcFace stack is unavailable.
        face_crop = _crop_largest_face_with_padding(img, pad_ratio=0.10)
        if face_crop is not None:
            direct = _detect_largest_face(face_crop)
            if direct is not None:
                face_feat = direct[1]
                query_quality = face_sharpness(direct[0])
        if face_feat is None:
            face_feat = encoding_from_bgr_image(img, yolo_model=yolo_model)
            query_quality = face_sharpness(face_crop if face_crop is not None else img)

    if face_feat is not None:
        if _normalize_face_feature(face_feat) is None:
            empty["message"] = (
                "Invalid face embedding from uploaded image. Try a clearer photo."
            )
            return empty

        candidates, profile_upgraded, match_type = _match_candidates_from_face(
            conn,
            face_feat,
            query_quality,
            face_crop,
            img,
        )

    if not candidates:
        bbox = _largest_person_bbox(img, yolo_model)
        if bbox is not None:
            try:
                body_feat = body_embedding_from_frame_bbox(img, bbox)
            except (ImportError, FileNotFoundError, ModuleNotFoundError) as exc:
                # Face search must not hard-fail when OSNet/torchreid/numpy extras are missing.
                print(f"[LOOKUP] Body Re-ID skipped: {exc}")
                body_feat = None
            if body_feat is not None:
                known_id, dist = find_best_body_match(body_feat, conn)
                match_type = "body (CCTV appearance)"
                if known_id is not None:
                    total, last_seen, by_location = get_visit_stats(conn, known_id)
                    candidates.append(
                        {
                            "person_id": known_id,
                            "known_person_id": known_id,
                            "distance": dist,
                            "source": "person_body_encodings",
                            "total_visits": total,
                            "last_seen": last_seen,
                            "by_location": by_location,
                        }
                    )

    if not candidates:
        empty["message"] = "No matching person in database."
        empty["query_quality"] = query_quality
        return empty

    best = candidates[0]
    return {
        "matched": True,
        "candidates": candidates,
        "known_person_id": best["person_id"],
        "person_id": best["person_id"],
        "distance": best["distance"],
        "match_type": match_type,
        "total_visits": best["total_visits"],
        "last_seen": best["last_seen"],
        "by_location": best["by_location"],
        "query_quality": query_quality,
        "profile_upgraded": profile_upgraded,
    }
