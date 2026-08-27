"""
AI Person Tracking and Counting System.

Uses YOLOv8 with ByteTrack to track people (COCO class 0), assigns persistent
track IDs, and logs sightings to SQLite with per-person cooldown.
"""

import argparse
import os
import socket
import sqlite3
import threading
import time
from urllib.parse import urlparse

# Serialize only open/release of FFmpeg captures (not per-frame reads) so
# multi-camera decode threads can run in parallel without RTSP backlog lag.
_FFMPEG_OPEN_LOCK = threading.RLock()

try:
    cv2.setNumThreads(1)
except Exception:
    pass
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from captures import (
    choose_display_capture_crop,
    capture_display_is_body_crop,
    crop_face_from_frame,
    ensure_capture_column,
    face_crop_is_photographic,
    fallback_display_capture_crop,
    save_face_capture,
    save_scene_capture,
)
from face_attributes import (
    CONFIDENCE_THRESHOLD,
    active_backend_name,
    attribute_models_ready,
    ensure_attribute_columns,
    predict_face_attributes,
    preload_attribute_models,
)
from camera_stats import write_faces_detected
from face_landmark_verify import (
    face_bbox_in_person_head_band,
    verify_face_detection_for_capture,
)
from face_analyzer import (
    DEFAULT_ANTI_SPOOF_MODEL,
    DEFAULT_GALLERY_DB,
    AdaptiveFrameSkip,
    FaceAnalyzer,
    FaceDetection,
    SharedFaceGallery,
    active_pipeline_name,
    analyzer_ready,
    apply_face_clahe,
    assign_unique_faces_to_persons,
    extract_face_from_person_crop,
    extract_face_from_person_crop_gpu,
    match_face_to_person_detections,
    _face_in_person_bbox,
)
from gpu_pipeline import (
    bgr_numpy_to_gpu,
    crop_xyxy,
    laplacian_variance,
    resize_hwc,
    scale_xyxy_boxes,
    tensor_to_bgr_numpy,
    unmap_letterbox_xyxy_boxes,
    vram_pipeline_active,
    yolo_track_source,
    yolo_track_batch,
)
from image_geometry import LetterboxTransform, letterbox_resize
from identity import (
    confidence_score_from_distance,
    face_sharpness,
    face_from_person_bbox,
    find_best_face_match,
    init_identity_db,
    models_ready,
    resolve_known_person_from_face,
    resolve_known_person_name,
    upsert_best_face_profile,
)
from telegram_alerts import (
    maybe_send_multi_face_alert_async,
    maybe_send_target_alert_async,
)
from orion_frame_pipeline import (
    FrameMemoryManager,
    FrameValidator,
    MEMORY_GC_INTERVAL_FRAMES,
    RTSP_RECONNECT_WAIT_SEC,
    SafeInferenceRunner,
    apply_identity_tiers,
    audit_hardware_dependencies,
    get_super_resolution_pipeline,
    init_face_restoration_stack,
    init_gpu_face_stack,
    release_frame_buffers,
    wrap_yolo_results,
)
from person_attributes import AttributeHUDCache, PersonAttributes, parse_person_attributes
from inference_device import (
    describe_device,
    get_device,
    log_device_info,
    move_model_to_device,
    tensor_to_numpy,
    torch_cuda_active,
    yolo_device_arg,
)
from live_frame_hub import publish_camera_jpeg, publish_jpeg

from yolo_loader import DEFAULT_MODEL, load_yolo_model, resolve_model_path as resolve_yolo_path

PROJECT_DIR = Path(__file__).parent
DB_PATH = PROJECT_DIR / "visitor_tracking.db"
COOLDOWN_SECONDS = 3
PERSON_CLASS_ID = 0
MIN_MODEL_BYTES = 5_000_000
MIN_PERSON_HEIGHT_PX = 36
# Outdoor gate cams: people are far + high-angle → smaller boxes.
OUTDOOR_MIN_PERSON_HEIGHT_PX = 20
OUTDOOR_LIVE_MIN_PERSON_PX = 18
OUTDOOR_PERSON_ASPECT_MAX = 2.6
OUTDOOR_MIN_FACE_LAPLACIAN = 2.0
OUTDOOR_SKIN_MIN_FRACTION = 0.02
_FACE_DETECT_COUNTS: Dict[str, int] = {}

from gpu_inference_worker import FrameJob, GPUInferenceWorker, _process_batch
from gpu_face_index import rebuild_face_index
import tracker_config
from tracker_config import (
    FACE_DETECT_MIN_CONFIDENCE,
    FACE_MIN_BOX_PX,
    TRACK_CONF,
    configure_inference_size,
    frame_skip_for_camera,
    gate_display_min_side,
    gate_face_min_confidence,
    gate_min_face_box_px,
    gate_min_face_laplacian,
    gate_min_person_height_px,
    gate_skin_min_fraction,
    is_outdoor_gate_camera,
    live_face_force_every_for_camera,
    parse_inference_size,
    track_conf_for_camera,
)


def _yolo_track_kwargs() -> dict:
    """YOLO track kwargs including the active inference device."""
    return {**tracker_config.TRACK_KWARGS, "device": yolo_device_arg()}


def _load_yolo_model(model_path: Path):
    """Load YOLOv8 Large on CUDA (delegates to yolo_loader; CPU refused for yolov8l)."""
    return load_yolo_model(model_path, PROJECT_DIR)


def configure_frame_skip(stride: int) -> int:
    """Process AI detection every N frames (1 = every frame). Preview stays smooth."""
    global FRAME_SKIP
    FRAME_SKIP = max(1, int(stride))
    return FRAME_SKIP


FRAME_SKIP = 3
# Dashboard JPEG encode is CPU-heavy with many cameras — keep preview lighter.
PREVIEW_PUBLISH_INTERVAL_SEC = float(os.environ.get("PREVIEW_PUBLISH_INTERVAL_SEC", "0.18"))
MAIN_LOOP_SLEEP_SEC = 0.01
STALE_FRAME_MAX_SEC = 15.0
READ_STALL_RECONNECT_SEC = 8.0
# OpenCV buffer depth for RTSP — 1 keeps latency low; outdoor gates use a slightly
# deeper buffer to reduce choppy/unstable motion when wind shakes the mount.
CAPTURE_BUFFER_SIZE = 1
OUTDOOR_GATE_CAPTURE_BUFFER_SIZE = max(
    1, int(os.environ.get("OUTDOOR_GATE_CAPTURE_BUFFER_SIZE", "3"))
)


def _bootstrap_env() -> None:
    """Load .env before module-level TELEGRAM_* flags are read."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_bootstrap_env()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Laplacian sharpness floor for live CCTV (lower = accept blurrier distant faces).
MIN_FACE_LAPLACIAN = _env_float("LIVE_FACE_MIN_SHARPNESS", 1.5)
# Live face detector confidence (studio default 0.88 is too strict for CCTV).
LIVE_FACE_DETECT_CONFIDENCE = _env_float("LIVE_FACE_DETECT_CONF", 0.28)
# Minimum face box side in pixels for hallway / distant CCTV faces.
LIVE_FACE_MIN_BOX_PX = int(_env_float("LIVE_FACE_MIN_BOX_PX", 18.0))
# Allow more head turn for people walking past cameras (frontal + side/profile).
# Backs of heads are rejected via landmark geometry — not by this yaw alone.
LIVE_FACE_MAX_YAW = _env_float("LIVE_FACE_MAX_YAW", 1.15)
# Native face side (px) required before save/Telegram. Hallway CCTV faces are often
# ~22–40px after pad; keep mild upscale caps so saves stay non-cartoonish.
LIVE_FACE_DISPLAY_MIN_SIDE = int(_env_float("LIVE_FACE_DISPLAY_MIN_SIDE", 36.0))
# Telegram: alert enrolled watchlist and/or non-watchlist CCTV incidents separately.
def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


TELEGRAM_ALERT_NON_WATCHLIST = _env_bool(
    "TELEGRAM_ALERT_NON_WATCHLIST",
    _env_bool("TELEGRAM_ALERT_STRANGERS", True),
)
TELEGRAM_ALERT_WATCHLIST = _env_bool("TELEGRAM_ALERT_WATCHLIST", True)
# Named People (not watchlist): Telegram when CCTV matches enrolled photos.
TELEGRAM_ALERT_NAMED = _env_bool("TELEGRAM_ALERT_NAMED", True)
TELEGRAM_NON_WATCHLIST_COOLDOWN_SEC = max(
    3.0,
    _env_float(
        "TELEGRAM_NON_WATCHLIST_COOLDOWN_SEC",
        _env_float("TELEGRAM_STRANGER_COOLDOWN_SEC", 8.0),
    ),
)
TELEGRAM_FACE_MIN_SIDE = int(_env_float("TELEGRAM_FACE_MIN_SIDE", 28.0))
# Back-compat alias
TELEGRAM_ALERT_STRANGERS = TELEGRAM_ALERT_NON_WATCHLIST
TELEGRAM_STRANGER_COOLDOWN_SEC = TELEGRAM_NON_WATCHLIST_COOLDOWN_SEC
# Landmark confidence for commit (indoor). Outdoor gates use gate_face_min_confidence.
LIVE_FACE_LANDMARK_MIN_CONF = _env_float("LIVE_FACE_LANDMARK_MIN_CONF", 0.40)
# Indoor skin fraction floor — moths/doors fail; real faces usually clear this.
LIVE_FACE_SKIN_MIN_FRACTION = _env_float("LIVE_FACE_SKIN_MIN_FRACTION", 0.12)
# Person-crop RetinaFace fallback stride (1 = check every track every face pass).
FACE_PERSON_CROP_STRIDE = int(_env_float("FACE_PERSON_CROP_STRIDE", 1.0))
# When YOLO sees a person, force RetinaFace at most every N inference ticks.
LIVE_FACE_FORCE_EVERY = max(1, int(_env_float("LIVE_FACE_FORCE_EVERY", 1.0)))
# Zoom into each YOLO person head when full-frame detect misses someone (multi-person halls).
LIVE_CATCH_EVERYONE = _env_bool("LIVE_CATCH_EVERYONE", True)
LIVE_CATCH_EVERYONE_MAX_PERSONS = max(
    1, int(_env_float("LIVE_CATCH_EVERYONE_MAX_PERSONS", 16.0))
)
# Save + Telegram every YOLO person on all cameras (frontal / side when possible).
# Still drops wood / floor / railing false positives.
CATCH_EVERY_PERSON = _env_bool("CATCH_EVERY_PERSON", True)
# Walking-away / back-of-head plates — off by default (user does not want backs).
CATCH_BACKS = _env_bool("CATCH_BACKS", False)
# Faces only: never save a YOLO person plate without a usable face (no back/body alerts).
FACE_ONLY_CATCH = _env_bool("FACE_ONLY_CATCH", True)
# Person-catch / Telegram must be at least this large (px). Tiny mush still
# looks "Clear" to Laplacian because of pixel noise.
PERSON_CATCH_MIN_SIDE = int(_env_float("PERSON_CATCH_MIN_SIDE", "72"))
# Upscale tiny person/head ROIs so RetinaFace can lock far hallway faces.
FACE_PERSON_CROP_TARGET_PX = int(_env_float("FACE_PERSON_CROP_TARGET_PX", 640.0))
# Commit once a usable face is stable — do not wait for the track to leave view.
TRACK_BEST_EARLY_COMMIT_FRAMES = int(_env_float("TRACK_BEST_EARLY_COMMIT_FRAMES", 0.0))
FACE_RESCAN_FRAMES = 20
FACE_SHIFT_RATIO = 0.12
FACE_SHIFT_MIN_PX = 40.0
# Flush walking passers quickly — long idle wait drops hallway faces.
TRACK_BEST_IDLE_FRAMES = int(_env_float("TRACK_BEST_IDLE_FRAMES", 2.0))
TRACK_BEST_MAX_COLLECT_FRAMES = int(_env_float("TRACK_BEST_MAX_COLLECT_FRAMES", 18.0))
ATTRIBUTE_HUD_REFRESH_FRAMES = 20
# Clothing/age HUD is CPU-heavy; keep off for large multi-camera GPU installs.
LIVE_ATTR_HUD = os.environ.get("LIVE_ATTR_HUD", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
PERSON_BOX_COLOR = (0, 229, 255)
PERSON_BOX_THICKNESS = 4
PERSON_HUD_TEXT_THICKNESS = 2
HUD_BG_BGR = (42, 36, 32)
HUD_TEXT_BGR = (240, 230, 220)
HUD_BORDER_BGR = (72, 65, 60)

# ── VRAM hybrid mode ───────────────────────────────────────────────────────────
# Set True to run YOLO on GPU while forcing face ONNX / anti-spoof to CPU.
# Toggle when many camera subprocesses hit VRAM limits.
HYBRID_FACE_ON_CPU = False


def init_db(db_path: Path) -> sqlite3.Connection:
    last_err: Optional[BaseException] = None
    for attempt in range(8):
        conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error:
            pass
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tracking_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    camera_location TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    known_person_id INTEGER,
                    capture_path TEXT,
                    gender TEXT,
                    age INTEGER
                )
                """
            )
            init_identity_db(conn)
            ensure_capture_column(conn)
            ensure_attribute_columns(conn)
            ensure_tracking_sessions_table(conn)
            conn.commit()
            return conn
        except sqlite3.OperationalError as exc:
            last_err = exc
            try:
                conn.close()
            except Exception:
                pass
            if "locked" not in str(exc).lower() or attempt >= 7:
                raise
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"init_db failed: {last_err}")


def ensure_tracking_sessions_table(conn: sqlite3.Connection) -> None:
    """Persist per-track entry/exit wall-clock times for dwell-time metrics."""
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
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_sessions)")}
    if "dwell_seconds" not in cols:
        conn.execute("ALTER TABLE tracking_sessions ADD COLUMN dwell_seconds REAL")
    if "known_person_id" not in cols:
        conn.execute("ALTER TABLE tracking_sessions ADD COLUMN known_person_id INTEGER")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tracking_sessions_entry
        ON tracking_sessions(entry_time)
        """
    )


def log_tracking_session(
    conn: sqlite3.Connection,
    person_id: int,
    camera_location: str,
    entry_time: datetime,
    exit_time: datetime,
    *,
    known_person_id: Optional[int] = None,
) -> Optional[float]:
    """
    Close a track appearance: dwell_time = exit_time - entry_time.
    Returns dwell seconds when a row is written.
    """
    ensure_tracking_sessions_table(conn)
    if exit_time < entry_time:
        entry_time, exit_time = exit_time, entry_time
    dwell_seconds = (exit_time - entry_time).total_seconds()
    if dwell_seconds < 0.05:
        return None
    entry_iso = entry_time.isoformat(timespec="seconds")
    exit_iso = exit_time.isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO tracking_sessions
            (person_id, camera_location, entry_time, exit_time,
             dwell_seconds, known_person_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            int(person_id),
            camera_location,
            entry_iso,
            exit_iso,
            round(dwell_seconds, 3),
            known_person_id,
        ),
    )
    conn.commit()
    return dwell_seconds


@dataclass
class _OpenDwellSession:
    entry_time: datetime
    last_seen_time: datetime
    miss_count: int = 0


class TrackDwellTracker:
    """
    Wall-clock entry/exit logger for ByteTrack person IDs.
    A session opens on first sighting and closes after idle_frames without the track.
    """

    def __init__(self, idle_frames: int = TRACK_BEST_IDLE_FRAMES) -> None:
        self.idle_frames = idle_frames
        self._open: Dict[Tuple[str, int], _OpenDwellSession] = {}

    def observe(
        self,
        camera_location: str,
        track_id: int,
        seen_at: Optional[datetime] = None,
    ) -> None:
        now = seen_at or datetime.now()
        key = (camera_location, int(track_id))
        current = self._open.get(key)
        if current is None:
            self._open[key] = _OpenDwellSession(entry_time=now, last_seen_time=now)
            return
        current.last_seen_time = now
        current.miss_count = 0

    def flush_inactive(
        self,
        camera_location: str,
        active_track_ids: Set[int],
        *,
        seen_at: Optional[datetime] = None,
    ) -> List[Tuple[int, datetime, datetime]]:
        """Return completed (track_id, entry_time, exit_time) for idle tracks."""
        now = seen_at or datetime.now()
        closed: List[Tuple[int, datetime, datetime]] = []
        for key in list(self._open.keys()):
            if key[0] != camera_location:
                continue
            track_id = key[1]
            if track_id in active_track_ids:
                session = self._open[key]
                session.miss_count = 0
                continue

            session = self._open[key]
            session.miss_count += 1
            if session.miss_count < self.idle_frames:
                continue

            closed.append((track_id, session.entry_time, session.last_seen_time or now))
            self._open.pop(key, None)
        return closed

    def close_all(
        self,
        camera_location: Optional[str] = None,
        *,
        seen_at: Optional[datetime] = None,
    ) -> List[Tuple[str, int, datetime, datetime]]:
        """Force-close open sessions (optionally for one camera)."""
        now = seen_at or datetime.now()
        closed: List[Tuple[str, int, datetime, datetime]] = []
        for key in list(self._open.keys()):
            if camera_location is not None and key[0] != camera_location:
                continue
            session = self._open.pop(key)
            closed.append(
                (key[0], key[1], session.entry_time, session.last_seen_time or now)
            )
        return closed


def _persist_closed_dwell_sessions(
    conn: sqlite3.Connection,
    camera_location: str,
    closed: List[Tuple[int, datetime, datetime]],
    track_to_known: Optional[Dict[str, Dict[int, int]]] = None,
) -> None:
    known_map = (track_to_known or {}).get(camera_location) or {}
    for track_id, entry_time, exit_time in closed:
        log_tracking_session(
            conn,
            track_id,
            camera_location,
            entry_time,
            exit_time,
            known_person_id=known_map.get(int(track_id)),
        )


def log_visit(
    conn: sqlite3.Connection,
    person_id: int,
    camera_location: str,
    known_person_id: Optional[int] = None,
    capture_path: Optional[str] = None,
    gender: Optional[str] = None,
    age: Optional[int] = None,
    scene_path: Optional[str] = None,
    display_name: Optional[str] = None,
) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    ensure_capture_column(conn)
    try:
        from identity import (
            ensure_log_display_name_column,
            is_auto_identity_name,
            resolve_known_person_name,
        )

        ensure_log_display_name_column(conn)
        if not display_name:
            display_name = resolve_known_person_name(
                conn,
                known_person_id,
                person_id=person_id,
                timestamp=timestamp,
            )
        if is_auto_identity_name(display_name):
            display_name = None
    except Exception:
        display_name = display_name or None
    try:
        conn.execute(
            """
            INSERT INTO tracking_logs
                (person_id, camera_location, timestamp, known_person_id,
                 capture_path, gender, age, scene_path, display_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                person_id,
                camera_location,
                timestamp,
                known_person_id,
                capture_path,
                gender,
                age,
                scene_path,
                display_name,
            ),
        )
    except sqlite3.OperationalError:
        conn.execute(
            """
            INSERT INTO tracking_logs
                (person_id, camera_location, timestamp, known_person_id,
                 capture_path, gender, age, scene_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                person_id,
                camera_location,
                timestamp,
                known_person_id,
                capture_path,
                gender,
                age,
                scene_path,
            ),
        )
    conn.commit()
    label = resolve_known_person_name(
        conn,
        known_person_id,
        person_id=person_id,
        timestamp=timestamp,
    )
    if known_person_id is not None:
        label = f"{label} (track {person_id})"
    attrs = ""
    if gender or age is not None:
        parts = []
        if gender:
            parts.append(gender)
        if age is not None:
            parts.append(f"~{age}y")
        attrs = f" [{', '.join(parts)}]"
    print(f"[LOG] {label}{attrs} @ {camera_location} — {timestamp}")

def resolve_model_path(model_arg: str) -> Path:
    return resolve_yolo_path(model_arg, PROJECT_DIR)


STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")


def is_stream_url(source: str) -> bool:
    return source.lower().startswith(STREAM_PREFIXES)


def sanitize_rtsp_url(source: str) -> str:
    """
    URL-encode RTSP user/password so passwords with ``@`` / ``:`` / ``#`` work.

    Broken example: ``rtsp://admin:Mstc@2024@192.168.2.230:554/...``
    Fixed:          ``rtsp://admin:Mstc%402024@192.168.2.230:554/...``
    """
    if not isinstance(source, str) or not source.lower().startswith("rtsp://"):
        return source
    from urllib.parse import quote, unquote

    rest = source[7:]  # strip rtsp://
    if "/" in rest:
        authority, path = rest.split("/", 1)
        path = "/" + path
    else:
        authority, path = rest, ""
    if "@" not in authority:
        return source
    creds, hostport = authority.rsplit("@", 1)
    if ":" not in creds:
        return source
    user, password = creds.split(":", 1)
    user_enc = quote(unquote(user), safe="")
    pass_enc = quote(unquote(password), safe="")
    return f"rtsp://{user_enc}:{pass_enc}@{hostport}{path}"


def resolve_video_path(source: str) -> Path:
    """Resolve video path; check project folder and current working directory."""
    path = Path(source)
    candidates = [path]
    if not path.is_absolute():
        candidates = [
            PROJECT_DIR / path,
            Path.cwd() / path,
            path,
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = "\n".join(f"    - {c.resolve()}" for c in candidates)
    raise FileNotFoundError(
        f"Video file not found: {source}\n"
        f"  Looked in:\n{tried}\n"
        f"  Put your .mp4 in: {PROJECT_DIR}\n"
        f"  Then run: python main.py --source {path.name} --camera-location main_entrance"
    )


def parse_source(source: str) -> Union[int, str]:
    if source.isdigit():
        return int(source)
    if is_stream_url(source):
        return source
    return str(resolve_video_path(source))


# Fail unreachable RTSP hosts in ~1–4s instead of OpenCV's default ~30s stall.
RTSP_TCP_PROBE_SEC = float(os.environ.get("RTSP_TCP_PROBE_SEC", "1.2"))
RTSP_OPEN_TIMEOUT_MS = int(os.environ.get("RTSP_OPEN_TIMEOUT_MS", "4000"))
RTSP_FIRST_FRAME_TRIES = int(os.environ.get("RTSP_FIRST_FRAME_TRIES", "15"))


def _ffmpeg_stream_options(rtsp_tcp: bool) -> Optional[str]:
    """Build OpenCV/FFmpeg capture options for low-latency RTSP."""
    # stimeout / rw_timeout are microseconds; keep them aligned with open timeout.
    sock_us = max(1_000_000, RTSP_OPEN_TIMEOUT_MS * 1000)
    opts = [
        "threads;2",
        "fflags;nobuffer",
        "flags;low_delay",
        "max_delay;0",
        f"stimeout;{sock_us}",
        f"rw_timeout;{sock_us}",
    ]
    # Prefer NVIDIA decode when available (falls back silently if unsupported).
    if os.environ.get("RTSP_HWACCEL", "1").strip() not in ("0", "false", "False"):
        opts.extend(
            [
                "hwaccel;cuda",
                "hwaccel_device;0",
                "hwaccel_output_format;nv12",
            ]
        )
    if rtsp_tcp:
        opts.insert(0, "rtsp_transport;tcp")
    return "|".join(opts)


def _stream_host_port(source: str) -> Optional[Tuple[str, int]]:
    """Extract host/port from an RTSP/HTTP(S) URL for a fast TCP probe."""
    try:
        parsed = urlparse(source)
    except Exception:
        return None
    host = parsed.hostname
    if not host:
        return None
    port = parsed.port
    if port is None:
        scheme = (parsed.scheme or "").lower()
        if scheme == "rtsp":
            port = 554
        elif scheme == "https":
            port = 443
        elif scheme in ("http", "rtmp"):
            port = 80 if scheme == "http" else 1935
        else:
            return None
    return host, int(port)


def _ensure_stream_host_reachable(source: str, timeout: float = RTSP_TCP_PROBE_SEC) -> None:
    """
    TCP connect probe before OpenCV open.

    Dead / offline cameras otherwise block VideoCapture for ~30s each while
    holding the FFmpeg open lock, which stalls every other camera.
    """
    endpoint = _stream_host_port(source)
    if endpoint is None:
        return
    host, port = endpoint
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as exc:
        raise RuntimeError(
            f"Could not connect to stream.\n"
            f"  URL: {source}\n"
            f"  Host {host}:{port} unreachable within {timeout:.1f}s ({exc}).\n"
            "  Check the camera IP/power/network, or try --rtsp-tcp"
        ) from exc


def _apply_capture_buffer_size(
    cap: cv2.VideoCapture, location: Optional[str] = None
) -> None:
    """Set OpenCV ring buffer; outdoor gates use a slightly deeper buffer."""
    size = CAPTURE_BUFFER_SIZE
    if location and is_outdoor_gate_camera(location):
        size = OUTDOOR_GATE_CAPTURE_BUFFER_SIZE
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, size)
    except Exception:
        pass


def _apply_capture_timeouts(cap: cv2.VideoCapture, timeout_ms: int = RTSP_OPEN_TIMEOUT_MS) -> None:
    """Cap OpenCV FFmpeg open/read waits (OpenCV 4.5+ / 4.8+ props)."""
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(timeout_ms))
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(timeout_ms))
    except Exception:
        pass


def open_capture(
    source: Union[int, str], rtsp_tcp: bool = False
) -> cv2.VideoCapture:
    """Open webcam, video file, or network stream (RTSP/HTTP)."""
    if isinstance(source, str):
        if is_stream_url(source):
            if source.lower().startswith("rtsp://"):
                source = sanitize_rtsp_url(source)
            print(f"Opening stream: {source.split('@')[-1] if '@' in source else source}")

            # Probe outside the FFmpeg lock so offline cams fail in parallel.
            if source.lower().startswith(("rtsp://", "http://", "https://")):
                _ensure_stream_host_reachable(source)

            with _FFMPEG_OPEN_LOCK:
                if source.lower().startswith("rtsp://"):
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _ffmpeg_stream_options(
                        rtsp_tcp
                    )
                else:
                    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

                # Use FFmpeg backend for all network streams (RTSP / HTTP / HTTPS).
                # Set timeouts before open() so dead streams do not stall ~30s.
                cap = cv2.VideoCapture()
                _apply_capture_timeouts(cap)
                if not cap.open(source, cv2.CAP_FFMPEG):
                    raise RuntimeError(
                        f"Could not connect to stream.\n"
                        f"  URL: {source}\n"
                        "  404 Not Found = wrong stream path (not IP/password).\n"
                        "  Check the camera RTSP URL in run_system.py or try --rtsp-tcp"
                    )

                # Keep only the newest frame — avoids 30s stalls on buffered RTSP data.
                if source.lower().startswith(("rtsp://", "http://", "https://")):
                    _apply_capture_buffer_size(cap)

                if not cap.isOpened():
                    cap.release()
                    raise RuntimeError(
                        f"Could not connect to stream.\n"
                        f"  URL: {source}\n"
                        "  404 Not Found = wrong stream path (not IP/password).\n"
                        "  Check the camera RTSP URL in run_system.py or try --rtsp-tcp"
                    )
                for _ in range(max(1, RTSP_FIRST_FRAME_TRIES)):
                    ok, _ = cap.read()
                    if ok:
                        _apply_capture_buffer_size(cap)
                        return cap
                    time.sleep(0.05)
                cap.release()
            raise RuntimeError(
                "Stream opened but no frames received.\n"
                "  Wait a few seconds and try again, or verify the RTSP path (e.g. stream1)."
            )

        print(f"Opening video: {source}")
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return cap
            cap.release()
        raise RuntimeError(
            f"Could not read frames from video: {source}\n"
            "  The file may be corrupt or use an unsupported codec."
        )

    # Webcam index — try DirectShow first on Windows (more reliable than default)
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
    for backend in backends:
        cap = cv2.VideoCapture(source, backend) if backend else cv2.VideoCapture(source)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                _apply_capture_buffer_size(cap)
                return cap
            cap.release()
        else:
            cap.release()

    available = find_camera_indices()
    hint = (
        f"Could not open webcam index {source}.\n"
        f"  • Cameras detected at index: {available or 'none'}\n"
        "  • Try: python main.py --source 0 --camera-location main_entrance\n"
        "  • Close Zoom/Teams/Camera app if the webcam is in use\n"
        "  • Or use a video file:\n"
        "      python main.py --source C:\\path\\to\\video.mp4 --camera-location main_entrance"
    )
    raise RuntimeError(hint)


def find_camera_indices(max_index: int = 5) -> List[int]:
    """Return webcam indices that return at least one frame."""
    found = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(index)
        cap.release()
    return found


GRID_COLS = 2
GRID_CELL_W = 640
GRID_CELL_H = 360
STABLE_PUBLISH_W = 640
STABLE_PUBLISH_H = 360
PREVIEW_MAX_WIDTH = 1920
PREVIEW_MAX_HEIGHT = 1080
TILE_BORDER_COLOR = (180, 170, 150)  # soft warm edge in BGR
TILE_BORDER_PX = 3
DISCONNECTED_BG_COLOR = (40, 28, 18)  # dark blue-gray
EMPTY_SLOT_BG_COLOR = (28, 22, 16)


def _configure_preview_grid(num_cameras: int) -> Tuple[int, int, int]:
    """Pick column count and tile size so all cameras fill the preview window."""
    global GRID_COLS, GRID_CELL_W, GRID_CELL_H, STABLE_PUBLISH_W, STABLE_PUBLISH_H

    if num_cameras <= 1:
        GRID_COLS = 1
    elif num_cameras == 2:
        GRID_COLS = 1  # stack vertically — each camera uses full width
    elif num_cameras <= 4:
        GRID_COLS = 2
    else:
        GRID_COLS = 4

    rows = max(1, (num_cameras + GRID_COLS - 1) // GRID_COLS)
    if GRID_COLS == 1:
        GRID_CELL_W = max(480, min(960, PREVIEW_MAX_WIDTH))
    else:
        GRID_CELL_W = max(360, min(640, PREVIEW_MAX_WIDTH // GRID_COLS))
    GRID_CELL_H = max(200, int(GRID_CELL_W * 9 / 16))  # 16:9 cells
    STABLE_PUBLISH_W = GRID_COLS * GRID_CELL_W
    STABLE_PUBLISH_H = rows * GRID_CELL_H
    return GRID_COLS, GRID_CELL_W, GRID_CELL_H


class CameraReader(threading.Thread):
    """
    Dedicated RTSP/video decode thread — one per camera.

    Keeps a single-slot latest-frame buffer (size = 1) so AI never processes a
    backlog of stale frames. Decode runs independently of YOLO so the preview
    stays live even when inference is stride-skipped.
    """

    def __init__(
        self,
        slot: int,
        location: str,
        source: Union[int, str],
        rtsp_tcp: bool = False,
    ):
        super().__init__(daemon=True, name=f"cam-{location}")
        self.slot = slot
        self.location = location
        self.source = source
        self.rtsp_tcp = rtsp_tcp
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._connected = False
        self._stop_event = threading.Event()
        self._failures = 0
        self._open_failures = 0
        self._last_open_warn = 0.0
        self._last_frame_time = 0.0
        self.cap: Optional[cv2.VideoCapture] = None

    def _open_backoff_sec(self) -> float:
        """Exponential backoff while the stream stays unreachable."""
        return min(30.0, 2.0 * (2 ** min(self._open_failures, 4)))

    def _release_capture(self) -> None:
        if self.cap is not None:
            with _FFMPEG_OPEN_LOCK:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
        # Let the camera free the RTSP session before we reconnect.
        self._stop_event.wait(0.75)

    def _clear_latest_frame(self) -> None:
        with self._lock:
            old = self._frame
            self._frame = None
            self._connected = False
        if old is not None:
            del old

    def _should_force_reconnect(self) -> bool:
        if self._last_frame_time <= 0:
            return self._failures >= 12
        return (time.time() - self._last_frame_time) >= READ_STALL_RECONNECT_SEC

    def _read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Grab the newest decoded frame without holding a global FFmpeg lock.

        With CAP_PROP_BUFFERSIZE=1, a single grab+retrieve is enough; we still
        drain a few grabs when the backend ignores buffer size.
        """
        assert self.cap is not None
        try:
            grabbed = False
            for _ in range(3):
                if not self.cap.grab():
                    break
                grabbed = True
            if not grabbed:
                return False, None
            return self.cap.retrieve()
        except cv2.error:
            return False, None

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                try:
                    # open_capture probes outside its FFmpeg lock; do not wrap
                    # this call in _FFMPEG_OPEN_LOCK or offline cams serialize again.
                    self.cap = open_capture(self.source, rtsp_tcp=self.rtsp_tcp)
                    _apply_capture_buffer_size(self.cap, self.location)
                    self._failures = 0
                    self._open_failures = 0
                    buf = (
                        OUTDOOR_GATE_CAPTURE_BUFFER_SIZE
                        if is_outdoor_gate_camera(self.location)
                        else CAPTURE_BUFFER_SIZE
                    )
                    print(f"[CAM] {self.location}: stream connected (buffer={buf})")
                except RuntimeError as exc:
                    self._mark_disconnected()
                    now = time.time()
                    if self._open_failures == 0 or now - self._last_open_warn >= 60.0:
                        print(f"[WARN] {self.location}: open failed ({exc})")
                        self._last_open_warn = now
                    self._open_failures += 1
                    self._stop_event.wait(self._open_backoff_sec())
                    continue

            ok, frame = self._read_latest()

            if ok and FrameValidator.is_valid(ok, frame):
                now = time.time()
                with self._lock:
                    old_frame = self._frame
                    # Single-slot buffer: overwrite previous ndarray reference.
                    self._frame = frame
                    self._frame_id += 1
                    self._connected = True
                    self._last_frame_time = now
                if old_frame is not None:
                    del old_frame
                self._failures = 0
            else:
                self._failures += 1
                if self._failures == 1:
                    print(
                        f"[WARN] {self.location}: invalid frame "
                        f"(ret={ok}) — waiting {RTSP_RECONNECT_WAIT_SEC:.0f}s"
                    )
                elif self._failures % 20 == 0:
                    print(f"[WARN] {self.location}: still no frames ({self._failures}x)")
                if self._should_force_reconnect() or self._failures >= 3:
                    self._mark_disconnected()
                    print(f"[WARN] {self.location}: reconnecting after stream drop…")
                    self._release_capture()
                    # မြန်မာ: RTSP ပြတ်တောက်ပါက ၂ စက္ကန့် စောင့်ပြီး ပြန်ချိတ်သည်
                    self._stop_event.wait(RTSP_RECONNECT_WAIT_SEC)
                    continue
                self._stop_event.wait(0.01)

        self._release_capture()
        self._clear_latest_frame()

    def _mark_disconnected(self) -> None:
        with self._lock:
            self._connected = False

    @property
    def is_live(self) -> bool:
        with self._lock:
            if not self._connected or self._frame is None or self._last_frame_time <= 0:
                return False
            return (time.time() - self._last_frame_time) <= STALE_FRAME_MAX_SEC

    def get_frame(self, *, copy: bool = True) -> Tuple[Optional[np.ndarray], bool]:
        """
        Return the latest full-resolution frame for preview / face crops.

        ``copy=True`` (default) is required when the caller may mutate or hold
        the array across the next decode tick.
        """
        with self._lock:
            if self._frame is None or self._last_frame_time <= 0:
                return None, False
            if time.time() - self._last_frame_time > STALE_FRAME_MAX_SEC:
                return None, False
            if copy:
                return self._frame.copy(), True
            return self._frame, True

    def get_preview_frame(
        self,
        out_w: int,
        out_h: int,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Small BGR tile for dashboard MJPEG — copies and downscales inside the
        decode lock so the main loop never duplicates full 2K/4K RTSP frames.
        """
        import cv2

        with self._lock:
            if self._frame is None or self._last_frame_time <= 0:
                return None, False
            if time.time() - self._last_frame_time > STALE_FRAME_MAX_SEC:
                return None, False
            frame = self._frame
            h, w = int(frame.shape[0]), int(frame.shape[1])
            if h < 1 or w < 1:
                return None, False
            tw = max(64, int(out_w))
            th = max(64, int(out_h))
            scale = max(tw / float(w), th / float(h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
            x0 = max(0, (new_w - tw) // 2)
            y0 = max(0, (new_h - th) // 2)
            crop = resized[y0 : y0 + th, x0 : x0 + tw]
            if crop.shape[0] != th or crop.shape[1] != tw:
                canvas = np.full((th, tw, 3), EMPTY_SLOT_BG_COLOR, dtype=np.uint8)
                ch, cw = crop.shape[:2]
                canvas[:ch, :cw] = crop
                return canvas.copy(), True
            return np.ascontiguousarray(crop.copy()), True

    def stop(self) -> None:
        self._stop_event.set()
        self._release_capture()
        self._clear_latest_frame()


def _draw_tile_border(
    frame: np.ndarray, color: Tuple[int, int, int] = TILE_BORDER_COLOR
) -> np.ndarray:
    """Draw a colored frame around a grid tile."""
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, TILE_BORDER_PX)
    return out


def _solid_tile(color: Tuple[int, int, int]) -> np.ndarray:
    tile = np.full((GRID_CELL_H, GRID_CELL_W, 3), color, dtype=np.uint8)
    return _draw_tile_border(tile)


def _annotate_camera_label(frame, label: str) -> None:
    """Draw a subtle HUD zone badge (dashboard composites add its own labels too)."""
    h, w = frame.shape[:2]
    bar_h = max(26, int(h * 0.07))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 14, 22), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.line(frame, (0, bar_h), (w, bar_h), (0, 180, 210), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"ZONE {label}".upper()[:24],
        (10, max(18, bar_h - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 229, 255),
        1,
        cv2.LINE_AA,
    )


def _disconnected_tile(slot: int, location: str) -> np.ndarray:
    """Colored placeholder tile for a grid slot with no live frame."""
    frame = _solid_tile(DISCONNECTED_BG_COLOR)
    cv2.putText(
        frame,
        f"Camera {slot + 1} - Disconnected",
        (16, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        location,
        (16, GRID_CELL_H - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        TILE_BORDER_COLOR,
        2,
        cv2.LINE_AA,
    )
    return frame


def _fit_cell(frame: np.ndarray) -> np.ndarray:
    """Letterbox into the cell — entire frame visible, no stretch or crop."""
    cell = np.full((GRID_CELL_H, GRID_CELL_W, 3), EMPTY_SLOT_BG_COLOR, dtype=np.uint8)
    h, w = frame.shape[:2]
    if h < 1 or w < 1:
        return _draw_tile_border(cell)
    scale = min(GRID_CELL_W / w, GRID_CELL_H / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (GRID_CELL_H - new_h) // 2
    x0 = (GRID_CELL_W - new_w) // 2
    cell[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return _draw_tile_border(cell)


def _build_grid(tiles: List[np.ndarray], cols: int = GRID_COLS) -> np.ndarray:
    """Stitch equal-size tiles into a rigid cols-wide grid."""
    if not tiles:
        return _disconnected_tile(0, "no cameras")

    padded = list(tiles)
    blank = _solid_tile(EMPTY_SLOT_BG_COLOR)
    while len(padded) % cols:
        padded.append(blank)

    rows = []
    for i in range(0, len(padded), cols):
        rows.append(cv2.hconcat(padded[i : i + cols]))
    return cv2.vconcat(rows)


WEB_STREAM_MAX_W = 1280
WEB_STREAM_MAX_H = 960
# Per-camera Fast Live tiles — smaller/faster encode keeps many cameras lighter.
CAMERA_PUBLISH_W = int(os.environ.get("CAMERA_PUBLISH_W", "720"))
CAMERA_PUBLISH_H = int(os.environ.get("CAMERA_PUBLISH_H", "405"))
CAMERA_PUBLISH_JPEG_QUALITY = int(os.environ.get("CAMERA_PUBLISH_JPEG_QUALITY", "68"))
# Clean NVR-style tiles by default (set LIVE_STREAM_DETECT=1 to show DETECT boxes).
LIVE_STREAM_DETECT_ENABLED = os.environ.get("LIVE_STREAM_DETECT", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _letterbox_to_fixed(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Letterbox into a fixed canvas — every published frame has identical dimensions."""
    canvas = np.full((out_h, out_w, 3), EMPTY_SLOT_BG_COLOR, dtype=np.uint8)
    h, w = frame.shape[:2]
    if h < 1 or w < 1:
        return canvas
    scale = min(out_w / w, out_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (out_h - new_h) // 2
    x0 = (out_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _cover_to_fixed(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Center-crop scale-to-fill — full tile, no black bars (NVR Fast Live style)."""
    h, w = frame.shape[:2]
    if h < 1 or w < 1:
        return np.full((out_h, out_w, 3), EMPTY_SLOT_BG_COLOR, dtype=np.uint8)
    scale = max(out_w / float(w), out_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    x0 = max(0, (new_w - out_w) // 2)
    y0 = max(0, (new_h - out_h) // 2)
    crop = resized[y0 : y0 + out_h, x0 : x0 + out_w]
    if crop.shape[0] != out_h or crop.shape[1] != out_w:
        canvas = np.full((out_h, out_w, 3), EMPTY_SLOT_BG_COLOR, dtype=np.uint8)
        ch, cw = crop.shape[:2]
        canvas[:ch, :cw] = crop
        return canvas
    return np.ascontiguousarray(crop)


def _publish_preview_frame(
    frame: np.ndarray, camera_key: Optional[str] = None
) -> None:
    """Share frames with dashboard.py via /api/live_cctv_stream (MJPEG hub)."""
    if frame is None or frame.size == 0:
        return
    if camera_key:
        target_w, target_h = CAMERA_PUBLISH_W, CAMERA_PUBLISH_H
        web_frame = _cover_to_fixed(frame, target_w, target_h)
        jpeg_q = CAMERA_PUBLISH_JPEG_QUALITY
    else:
        target_w = min(STABLE_PUBLISH_W, WEB_STREAM_MAX_W)
        target_h = min(STABLE_PUBLISH_H, WEB_STREAM_MAX_H)
        web_frame = _letterbox_to_fixed(frame, target_w, target_h)
        jpeg_q = 82
    ok, buf = cv2.imencode(".jpg", web_frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_q)])
    if ok:
        try:
            payload = buf.tobytes()
            if camera_key:
                publish_camera_jpeg(camera_key, payload)
            else:
                publish_jpeg(payload)
        except OSError as exc:
            # Never crash the tracker when the dashboard holds frame.jpg open.
            print(f"[WARN] live preview publish failed: {exc}")


def _crop_person_numpy(
    frame: np.ndarray,
    bbox: np.ndarray,
    *,
    pad_ratio: float = 0.04,
) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = (int(v) for v in bbox[:4])
    h, w = frame.shape[:2]
    pad = int(pad_ratio * max(x2 - x1, y2 - y1))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _draw_attribute_hud_card(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    attrs: PersonAttributes,
) -> None:
    """Draw person bbox plus semi-transparent attribute card at top-right of bbox."""
    cv2.rectangle(
        frame, (x1, y1), (x2, y2), PERSON_BOX_COLOR, PERSON_BOX_THICKNESS, cv2.LINE_AA
    )

    lines = attrs.hud_lines()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = PERSON_HUD_TEXT_THICKNESS
    pad_x, pad_y = 7, 6
    line_h = 18

    text_widths = [
        cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines
    ]
    card_w = max(text_widths, default=80) + pad_x * 2
    card_h = len(lines) * line_h + pad_y * 2

    fh, fw = frame.shape[:2]
    card_x = min(fw - card_w - 2, x2 + 5)
    card_y = max(0, y1)
    if card_x + card_w > fw:
        card_x = max(0, x1 - card_w - 5)
    if card_y + card_h > fh:
        card_y = max(0, fh - card_h - 2)

    x0, y0 = card_x, card_y
    x1c, y1c = card_x + card_w, card_y + card_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1c, y1c), HUD_BG_BGR, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1c, y1c), HUD_BORDER_BGR, 1, cv2.LINE_AA)

    for i, line in enumerate(lines):
        ty = y0 + pad_y + (i + 1) * line_h - 5
        if line.startswith("■ Female"):
            color = (210, 120, 255)
        elif line.startswith("■ Male"):
            color = (255, 200, 80)
        else:
            color = HUD_TEXT_BGR
        cv2.putText(
            frame,
            line,
            (x0 + pad_x, ty),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        if i < len(lines) - 1:
            sep_y = y0 + pad_y + (i + 1) * line_h + 1
            cv2.line(
                frame,
                (x0 + 4, sep_y),
                (x1c - 4, sep_y),
                HUD_BORDER_BGR,
                1,
                cv2.LINE_AA,
            )


def _render_detection_hud(
    frame: np.ndarray,
    result,
    camera_location: str,
    frame_index: int,
    attr_cache: AttributeHUDCache,
    *,
    letterbox: Optional[LetterboxTransform] = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """Overlay enterprise attribute HUD cards for each tracked person."""
    if not LIVE_ATTR_HUD:
        return frame
    if result.boxes is None or len(result.boxes) == 0:
        return frame

    out = frame
    boxes_t = result.boxes.xyxy
    if letterbox is not None and not letterbox.is_identity:
        boxes_t = _boxes_to_source_space(boxes_t, letterbox)
    elif scale_x != 1.0 or scale_y != 1.0:
        boxes_t = scale_xyxy_boxes(boxes_t, scale_x, scale_y)
    confs_t = result.boxes.conf
    track_ids_t = result.boxes.id

    for i in range(len(boxes_t)):
        conf = float(confs_t[i].item()) if confs_t is not None else 1.0
        if conf < track_conf_for_camera(camera_location):
            continue
        bbox_host = tensor_to_numpy(boxes_t[i])
        height = float(bbox_host[3] - bbox_host[1])
        min_h = (
            gate_min_person_height_px(camera_location)
            if is_outdoor_gate_camera(camera_location)
            else float(MIN_PERSON_HEIGHT_PX)
        )
        if height < min_h:
            continue

        x1, y1, x2, y2 = (int(v) for v in bbox_host)
        body = _crop_person_numpy(out, bbox_host)
        if body is None:
            continue

        track_id = (
            int(track_ids_t[i].item()) if track_ids_t is not None else None
        )
        head = body[: max(1, int(body.shape[0] * 0.45))]
        if track_id is not None:
            attrs = attr_cache.get(
                camera_location,
                track_id,
                frame_index,
                body,
                head,
            )
        else:
            attrs = parse_person_attributes(body, head)

        _draw_attribute_hud_card(out, x1, y1, x2, y2, attrs)

    return out


def _safe_render_detection_hud(
    frame: np.ndarray,
    result,
    camera_location: str,
    frame_index: int,
    attr_cache: AttributeHUDCache,
    *,
    letterbox: Optional[LetterboxTransform] = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """HUD overlay must never block preview publishing."""
    try:
        return _render_detection_hud(
            frame,
            result,
            camera_location,
            frame_index,
            attr_cache,
            letterbox=letterbox,
            scale_x=scale_x,
            scale_y=scale_y,
        )
    except Exception as exc:
        print(f"[WARN] HUD render failed @ {camera_location}: {exc}")
        return frame


LIVE_STREAM_DETECT_CONF = 0.50
LIVE_STREAM_MIN_PERSON_PX = 48
# Reject square/wide YOLO false-positives (floor tiles, rugs, wall art).
LIVE_STREAM_PERSON_ASPECT_MAX = 0.92  # width / height — people are taller
LIVE_STREAM_PERSON_ASPECT_MIN = 0.22
LIVE_STREAM_DETECT_BGR = (255, 229, 0)


def _snapshot_face_boxes(
    face_dets: Optional[list],
    frame_bgr: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, int, int]]:
    raw: List[Tuple[object, Tuple[int, int, int, int]]] = []
    for det in face_dets or []:
        bbox = getattr(det, "bbox", None)
        if bbox is None:
            continue
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        except (TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            raw.append((det, (x1, y1, x2, y2)))

    all_boxes = [b for _, b in raw]
    boxes: List[Tuple[int, int, int, int]] = []
    for _det, bbox in raw:
        if frame_bgr is not None:
            try:
                from face_print_filter import is_likely_wall_print_or_poster

                bad, _ = is_likely_wall_print_or_poster(
                    frame_bgr, bbox, other_face_bboxes=all_boxes
                )
                if bad:
                    continue
            except Exception:
                pass
        boxes.append(bbox)
    return boxes


def _publish_live_face_boxes(
    camera_location: str,
    frame: np.ndarray,
    face_boxes: Optional[List[Tuple[int, int, int, int]]],
) -> None:
    """Share latest face boxes with the Face Detection MJPEG overlay endpoint."""
    try:
        from live_detection_hub import publish_face_boxes

        h, w = frame.shape[:2]
        publish_face_boxes(
            camera_location,
            face_boxes or [],
            frame_width=int(w),
            frame_height=int(h),
        )
    except Exception:
        pass


def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _point_inside_box(px: float, py: float, box: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def _union_boxes(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _box_area(box: Tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _person_box_looks_human(
    box: Tuple[int, int, int, int],
    *,
    frame_shape: Optional[Tuple[int, ...]] = None,
    outdoor: bool = False,
    frame_bgr: Optional[np.ndarray] = None,
) -> bool:
    """Reject YOLO false positives on floor patterns / square floor tiles / rail posts."""
    x1, y1, x2, y2 = box
    w = float(max(1, x2 - x1))
    h = float(max(1, y2 - y1))
    min_h = float(OUTDOOR_LIVE_MIN_PERSON_PX if outdoor else LIVE_STREAM_MIN_PERSON_PX)
    if CATCH_EVERY_PERSON and not outdoor:
        # Far hallway walkers are often 24–40px tall — still real people.
        min_h = min(min_h, 24.0)
    if h < min_h:
        return False
    aspect = w / h
    aspect_max = (
        float(OUTDOOR_PERSON_ASPECT_MAX)
        if outdoor or CATCH_EVERY_PERSON
        else float(LIVE_STREAM_PERSON_ASPECT_MAX)
    )
    aspect_min = 0.20 if (outdoor or CATCH_EVERY_PERSON) else float(LIVE_STREAM_PERSON_ASPECT_MIN)
    if aspect > aspect_max or aspect < aspect_min:
        return False
    if frame_shape is not None and len(frame_shape) >= 2:
        fh = float(max(1, frame_shape[0]))
        fw = float(max(1, frame_shape[1]))
        area_frac = (w * h) / (fw * fh)
        cy = (y1 + y2) * 0.5
        # Small mid/lower-frame blobs that are nearly square are almost always tiles/rugs.
        # Skip this indoor tile heuristic for outdoor gate cams (squat high-angle people).
        if (
            not outdoor
            and cy > 0.45 * fh
            and area_frac < 0.06
            and aspect > 0.75
        ):
            return False
        # Foreground railing posts sit in the bottom corners and stay tiny.
        if area_frac < 0.035 and cy > 0.62 * fh and aspect < 0.55:
            return False
    if frame_bgr is not None and getattr(frame_bgr, "size", 0) > 0:
        try:
            from face_print_filter import is_likely_wood_post_or_rail

            ih, iw = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
            xx1 = max(0, min(iw - 1, int(x1)))
            yy1 = max(0, min(ih - 1, int(y1)))
            xx2 = max(0, min(iw, int(x2)))
            yy2 = max(0, min(ih, int(y2)))
            if xx2 - xx1 >= 12 and yy2 - yy1 >= 12:
                crop = frame_bgr[yy1:yy2, xx1:xx2]
                wood, _ = is_likely_wood_post_or_rail(crop)
                if wood:
                    return False
        except Exception:
            pass
    return True


def _all_live_detect_boxes(
    result=None,
    *,
    letterbox: Optional[LetterboxTransform] = None,
    face_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    frame_shape: Optional[Tuple[int, ...]] = None,
    camera_location: str = "",
) -> List[Tuple[Tuple[int, int, int, int], int]]:
    """Return every person(+face) cluster for live preview — catch everyone, not just largest."""
    outdoor = is_outdoor_gate_camera(camera_location)
    min_conf = 0.22 if outdoor else min(0.35, LIVE_STREAM_DETECT_CONF)
    person_boxes: List[Tuple[Tuple[int, int, int, int], float]] = []
    if result is not None and getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
        boxes_t = result.boxes.xyxy
        if letterbox is not None and not letterbox.is_identity:
            boxes_t = _boxes_to_source_space(boxes_t, letterbox)
        confs_t = result.boxes.conf
        for i in range(len(boxes_t)):
            conf = float(confs_t[i].item()) if confs_t is not None else 1.0
            if conf < min_conf:
                continue
            bbox_host = tensor_to_numpy(boxes_t[i])
            x1, y1, x2, y2 = (int(v) for v in bbox_host)
            box = (x1, y1, x2, y2)
            if not _person_box_looks_human(
                box, frame_shape=frame_shape, outdoor=outdoor
            ):
                continue
            person_boxes.append((box, conf))

    candidates: List[Tuple[Tuple[int, int, int, int], float]] = []
    used_faces: set[int] = set()
    for person_box, person_conf in person_boxes:
        merged = person_box
        cx, cy = _box_center(person_box)
        for idx, face_box in enumerate(face_boxes or []):
            if idx in used_faces:
                continue
            fx, fy = _box_center(face_box)
            if _point_inside_box(fx, fy, person_box) or _point_inside_box(cx, cy, face_box):
                merged = _union_boxes(merged, face_box)
                used_faces.add(idx)
        candidates.append((merged, person_conf))

    # Orphan faces (no YOLO body) still count — multi-walker frames often miss a body.
    for idx, face_box in enumerate(face_boxes or []):
        if idx in used_faces:
            continue
        candidates.append((face_box, 0.5))

    return [
        (box, int(round(conf * 100)))
        for box, conf in candidates
    ]


def _primary_live_detect_box(
    result=None,
    *,
    letterbox: Optional[LetterboxTransform] = None,
    face_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    frame_shape: Optional[Tuple[int, ...]] = None,
    camera_location: str = "",
) -> Optional[Tuple[Tuple[int, int, int, int], int]]:
    """Largest detect box (compat helper). Prefer ``_all_live_detect_boxes`` for overlays."""
    boxes = _all_live_detect_boxes(
        result,
        letterbox=letterbox,
        face_boxes=face_boxes,
        frame_shape=frame_shape,
        camera_location=camera_location,
    )
    if not boxes:
        return None
    return max(boxes, key=lambda item: _box_area(item[0]) * max(item[1], 1))


def _draw_live_stream_detect_overlay(
    frame: np.ndarray,
    result=None,
    *,
    letterbox: Optional[LetterboxTransform] = None,
    face_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    camera_location: str = "",
) -> np.ndarray:
    """Draw a DETECT box on every person/face — not only the largest."""
    if not LIVE_STREAM_DETECT_ENABLED:
        return frame
    boxes = _all_live_detect_boxes(
        result,
        letterbox=letterbox,
        face_boxes=face_boxes,
        frame_shape=getattr(frame, "shape", None),
        camera_location=camera_location,
    )
    if not boxes:
        return frame

    out = frame
    for (x1, y1, x2, y2), pct in boxes:
        from face_overlay_draw import draw_corner_bracket_box

        draw_corner_bracket_box(
            out,
            (x1, y1, x2, y2),
            color=(255, 255, 255),
            thickness=2,
        )
    return out


def _boxes_to_source_space(
    boxes_t: torch.Tensor,
    letterbox: LetterboxTransform,
) -> torch.Tensor:
    if letterbox.is_identity:
        return boxes_t
    return unmap_letterbox_xyxy_boxes(boxes_t, letterbox)


def _resize_for_inference(
    frame: np.ndarray,
) -> Tuple[np.ndarray, LetterboxTransform]:
    """Letterbox to AI size only — full-res frame stays for preview / face crops."""
    return letterbox_resize(
        frame,
        tracker_config.INFERENCE_WIDTH,
        tracker_config.INFERENCE_HEIGHT,
    )


def _should_run_inference(frame_index: int, camera_location: str = "") -> bool:
    skip = (
        frame_skip_for_camera(camera_location, default_skip=FRAME_SKIP)
        if camera_location
        else FRAME_SKIP
    )
    if skip <= 1:
        return True
    return frame_index % skip == 0


def _camera_inference_due(
    location: str,
    cam_tick: int,
    global_frame_index: int,
) -> bool:
    """Global stride for all cams; gate cams also get extra ticks between globals."""
    if _should_run_inference(global_frame_index - 1):
        return True
    if is_outdoor_gate_camera(location):
        gate_skip = frame_skip_for_camera(location, default_skip=FRAME_SKIP)
        return cam_tick % gate_skip == 0
    return False


def _should_run_face_inference(frame_index: int, skipper: AdaptiveFrameSkip) -> bool:
    if skipper.skip <= 1:
        return True
    return frame_index % skipper.skip == 0


class FaceTrackGate:
    """Per-track frame skip — rescan after N frames or a large bbox shift."""

    def __init__(
        self,
        rescan_frames: int = FACE_RESCAN_FRAMES,
        shift_ratio: float = FACE_SHIFT_RATIO,
        shift_min_px: float = FACE_SHIFT_MIN_PX,
    ) -> None:
        self.rescan_frames = rescan_frames
        self.shift_ratio = shift_ratio
        self.shift_min_px = shift_min_px
        self._state: Dict[Tuple[str, object], Tuple[int, float, float, float]] = {}

    def should_recognize(
        self,
        camera_location: str,
        track_id: Optional[int],
        bbox: np.ndarray,
        frame_index: int,
        visit_key: Tuple[str, object],
    ) -> bool:
        gate_key = (camera_location, int(track_id)) if track_id is not None else visit_key
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        diagonal = max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)

        prev = self._state.get(gate_key)
        if prev is None:
            self._state[gate_key] = (frame_index, cx, cy, diagonal)
            return True

        last_frame, px, py, pdiag = prev
        frame_delta = frame_index - last_frame
        shift = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        shift_limit = max(
            self.shift_min_px,
            self.shift_ratio * max(diagonal, pdiag),
        )
        if frame_delta >= self.rescan_frames or shift >= shift_limit:
            self._state[gate_key] = (frame_index, cx, cy, diagonal)
            return True
        return False


@dataclass
class _BestTrackFrame:
    sharpness: float
    face_crop: np.ndarray
    person_bbox: np.ndarray
    first_seen_index: int
    last_seen_index: int
    face_bbox: Optional[np.ndarray] = None
    # Full hallway frame from the same moment as face_crop (for Review scene).
    scene_frame: Optional[np.ndarray] = None


def _scene_cache_pair(
    frame_bgr: Optional[np.ndarray],
    face_bbox: Optional[np.ndarray],
    *,
    max_side: int = 960,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Downscale a source frame for RAM-safe best-frame caching; scale face box too."""
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return None, None if face_bbox is None else np.asarray(face_bbox, dtype=np.float32).copy()
    scene = np.ascontiguousarray(frame_bgr)
    box = None if face_bbox is None else np.asarray(face_bbox, dtype=np.float32).reshape(-1)[:4].copy()
    h, w = int(scene.shape[0]), int(scene.shape[1])
    side = max(h, w)
    if side > int(max_side) > 0:
        scale = float(max_side) / float(side)
        scene = cv2.resize(
            scene,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        if box is not None and box.size >= 4:
            box = box * float(scale)
    return scene, box


class TrackBestFrameSelector:
    """
    Cache per-track face crops and keep only the sharpest frame per trajectory.
    DeepFace / DB matching runs once when the track goes idle (person leaves view).
    """

    def __init__(
        self,
        idle_frames: int = TRACK_BEST_IDLE_FRAMES,
        max_collect_frames: int = TRACK_BEST_MAX_COLLECT_FRAMES,
        min_sharpness: float = MIN_FACE_LAPLACIAN,
    ) -> None:
        self.idle_frames = idle_frames
        self.max_collect_frames = max_collect_frames
        self.min_sharpness = min_sharpness
        self._best: Dict[Tuple[str, int], _BestTrackFrame] = {}
        self._missing: Dict[Tuple[str, int], int] = {}

    def observe(
        self,
        camera_location: str,
        track_id: int,
        frame_index: int,
        sharpness: float,
        face_crop: np.ndarray,
        person_bbox: np.ndarray,
        face_bbox: Optional[np.ndarray] = None,
        scene_frame: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, _BestTrackFrame]]:
        """Update the sharpest cached crop for a track; may finalize a long trajectory."""
        outdoor = is_outdoor_gate_camera(camera_location)
        sharp_floor = 1.5 if outdoor else max(3.0, self.min_sharpness * 0.25)
        if sharpness < sharp_floor:
            return None

        key = (camera_location, int(track_id))
        self._missing[key] = 0
        current = self._best.get(key)
        cached_scene, face_box_copy = _scene_cache_pair(scene_frame, face_bbox)
        if current is not None:
            current.last_seen_index = frame_index
            if frame_index - current.first_seen_index >= self.max_collect_frames:
                finalized = (track_id, current)
                self._best[key] = _BestTrackFrame(
                    sharpness=sharpness,
                    face_crop=face_crop.copy(),
                    person_bbox=person_bbox.copy(),
                    first_seen_index=frame_index,
                    last_seen_index=frame_index,
                    face_bbox=face_box_copy,
                    scene_frame=cached_scene,
                )
                return finalized
            # Prefer larger native faces — tiny sharp crops look cartoon when saved.
            cur_side = float(
                min(int(current.face_crop.shape[0]), int(current.face_crop.shape[1]))
            )
            new_side = float(min(int(face_crop.shape[0]), int(face_crop.shape[1])))
            better_sharp = sharpness > current.sharpness
            similar_sharp = sharpness >= current.sharpness * 0.85
            much_larger = new_side >= cur_side * 1.25
            if not better_sharp and not (similar_sharp and much_larger):
                return None

        self._best[key] = _BestTrackFrame(
            sharpness=sharpness,
            face_crop=face_crop.copy(),
            person_bbox=person_bbox.copy(),
            first_seen_index=current.first_seen_index if current else frame_index,
            last_seen_index=frame_index,
            face_bbox=face_box_copy,
            scene_frame=cached_scene,
        )
        return None

    def take_ready(
        self,
        camera_location: str,
        track_id: int,
        frame_index: int,
        *,
        min_frames: int = TRACK_BEST_EARLY_COMMIT_FRAMES,
    ) -> Optional[_BestTrackFrame]:
        """
        Pop a usable cached face once it has been stable for a few frames.

        Live Telegram / captures must not wait until the person leaves the hallway.
        """
        key = (camera_location, int(track_id))
        current = self._best.get(key)
        if current is None:
            return None
        outdoor = is_outdoor_gate_camera(camera_location)
        ready_floor = (
            gate_min_face_laplacian(camera_location) * 0.35
            if outdoor
            else max(2.0, self.min_sharpness * 0.35)
        )
        if current.sharpness < ready_floor:
            return None
        if frame_index - current.first_seen_index < max(0, int(min_frames)):
            return None
        self._best.pop(key, None)
        self._missing.pop(key, None)
        return current

    def flush_inactive(
        self,
        camera_location: str,
        active_track_ids: Set[int],
    ) -> List[Tuple[int, _BestTrackFrame]]:
        """Finalize trajectories whose track IDs are no longer visible."""
        finalized: List[Tuple[int, _BestTrackFrame]] = []
        for key in list(self._best.keys()):
            if key[0] != camera_location:
                continue
            track_id = key[1]
            if track_id in active_track_ids:
                self._missing[key] = 0
                continue

            misses = self._missing.get(key, 0) + 1
            self._missing[key] = misses
            if misses < self.idle_frames:
                continue

            entry = self._best.pop(key, None)
            self._missing.pop(key, None)
            if entry is not None:
                finalized.append((track_id, entry))
        return finalized


def _frame_as_bgr_numpy(frame) -> np.ndarray:
    """Full-resolution host frame for high-quality face thumbnail crops."""
    if isinstance(frame, torch.Tensor):
        return tensor_to_bgr_numpy(frame)
    return frame


def _laplacian_on_bgr(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _head_crop_from_person_bbox(
    source_frame: np.ndarray,
    person_bbox,
    *,
    head_height_ratio: float = 0.38,
    pad_ratio: float = 0.12,
) -> Optional[np.ndarray]:
    """
    Approximate head/face crop from a YOLO person box.

    Distant hallway subjects often have a usable body box but no RetinaFace hit;
    the upper body region still makes a useful CCTV catch for Telegram.
    """
    if source_frame is None or getattr(source_frame, "size", 0) == 0:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in np.asarray(person_bbox, dtype=np.float32).reshape(-1)[:4])
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    h = y2 - y1
    w = x2 - x1
    if h < 40 or w < 20:
        return None
    head_h = max(24.0, h * float(head_height_ratio))
    pad_x = w * float(pad_ratio)
    pad_y = head_h * float(pad_ratio)
    fh, fw = source_frame.shape[:2]
    sx1 = int(max(0, np.floor(x1 - pad_x)))
    sy1 = int(max(0, np.floor(y1 - pad_y)))
    sx2 = int(min(fw, np.ceil(x2 + pad_x)))
    sy2 = int(min(fh, np.ceil(y1 + head_h + pad_y)))
    if sx2 - sx1 < 16 or sy2 - sy1 < 16:
        return None
    crop = source_frame[sy1:sy2, sx1:sx2]
    if crop is None or crop.size == 0:
        return None
    return np.ascontiguousarray(crop.copy())


def _head_search_bbox(
    person_bbox: np.ndarray,
    *,
    outdoor: bool = False,
) -> np.ndarray:
    """Prefer the upper body/head region — faces live in the top of the person box."""
    x1, y1, x2, y2 = (float(v) for v in np.asarray(person_bbox, dtype=np.float32).reshape(-1)[:4])
    height = max(1.0, y2 - y1)
    width = max(1.0, x2 - x1)
    # Outdoor high-angle: head occupies more of the person box; search top ~72%.
    top_frac = 0.72 if outdoor else 0.58
    pad_x = (0.12 if outdoor else 0.08) * width
    return np.asarray(
        [x1 - pad_x, y1 - 0.06 * height, x2 + pad_x, y1 + top_frac * height],
        dtype=np.float32,
    )


def _face_passes_person_head_gate(
    face_bbox: Sequence[float],
    person_bbox: np.ndarray,
    *,
    camera_location: str = "",
) -> bool:
    """Drop arm/shoulder RetinaFace hits that sit below the head band."""
    outdoor = is_outdoor_gate_camera(camera_location)
    ok, _reason = face_bbox_in_person_head_band(
        face_bbox, person_bbox, outdoor=outdoor
    )
    return ok


    return ok


def _person_has_head_band_face(
    person_bbox: np.ndarray,
    face_dets: Optional[list],
    *,
    camera_location: str = "",
) -> bool:
    """True when a quality face already overlaps this person's head band."""
    outdoor = is_outdoor_gate_camera(camera_location)
    host = np.asarray(person_bbox, dtype=np.float32).reshape(-1)[:4]
    for det in face_dets or []:
        if not getattr(det, "quality_ok", False):
            continue
        if getattr(det, "embedding", None) is None:
            continue
        bbox = getattr(det, "bbox", None)
        if bbox is None:
            continue
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        except (TypeError, ValueError):
            continue
        if not _face_in_person_bbox((x1, y1, x2, y2), host):
            continue
        ok, _reason = face_bbox_in_person_head_band(
            (x1, y1, x2, y2), host, outdoor=outdoor
        )
        if ok:
            return True
    return False


def _find_face_det_in_person_box(
    source_frame: np.ndarray,
    person_bbox: np.ndarray,
    face_analyzer: Optional[FaceAnalyzer],
    *,
    camera_location: str = "",
) -> Optional[Tuple[FaceDetection, Tuple[int, int, int, int]]]:
    """
    Detect a face inside one YOLO person box (head ROI + upscale).

    Returns (FaceDetection with source-frame bbox, abs_bbox tuple) or None.
    """
    if face_analyzer is None or not analyzer_ready() or source_frame is None:
        return None
    outdoor = is_outdoor_gate_camera(camera_location)
    host_bbox = np.asarray(person_bbox, dtype=np.float32).reshape(-1)[:4]
    try:
        from captures import map_bbox_from_person_crop_to_source
        from face_analyzer import _crop_person
        from dataclasses import replace

        search_bbox = _head_search_bbox(host_bbox, outdoor=outdoor)
        person_crop = _crop_person(source_frame, search_bbox, pad=0.05)
        crop_pad_ratio = 0.05
        used_bbox = search_bbox
        if person_crop is None or person_crop.size == 0:
            person_crop = _crop_person(source_frame, host_bbox, pad=0.12)
            crop_pad_ratio = 0.12
            used_bbox = host_bbox
        if person_crop is None or person_crop.size == 0:
            return None
        ch, cw = person_crop.shape[:2]
        scale = 1.0
        work = person_crop
        target = max(320, int(FACE_PERSON_CROP_TARGET_PX))
        if outdoor:
            target = max(target, 640)
        elif max(ch, cw) < 420:
            # Indoor hallway pairs — boost ROI so the second walker is not missed.
            target = max(target, 720)
        if max(ch, cw) < target:
            scale = min(10.0 if outdoor else 8.0, target / float(max(ch, cw)))
            work = cv2.resize(
                person_crop,
                (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
                interpolation=cv2.INTER_CUBIC,
            )
        local_dets = face_analyzer.detect_and_embed_scan(work)
        usable = [
            d
            for d in local_dets
            if getattr(d, "quality_ok", False) and d.embedding is not None
        ]
        if not usable:
            usable = [d for d in local_dets if d.embedding is not None]
        if not usable:
            return None
        best = max(
            usable,
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
        )
        local_w = float(best.bbox[2] - best.bbox[0])
        local_h = float(best.bbox[3] - best.bbox[1])
        source_face_side = max(local_w, local_h) / max(scale, 1e-6)
        ok_lm, _reason = verify_face_detection_for_capture(
            best,
            min_confidence=gate_face_min_confidence(camera_location) if outdoor else 0.18,
            allow_side_view=True,
            max_yaw=LIVE_FACE_MAX_YAW,
            allow_distant=outdoor or source_face_side < 52.0,
            allow_high_angle=True,
            min_box_px=gate_min_face_box_px(camera_location) if outdoor else FACE_MIN_BOX_PX,
        )
        if not ok_lm:
            return None
        local_bbox = list(best.bbox)
        if scale != 1.0:
            local_bbox = [float(v) / scale for v in local_bbox]
        abs_bbox = map_bbox_from_person_crop_to_source(
            used_bbox,
            local_bbox,
            source_frame.shape,
            pad_ratio=crop_pad_ratio,
            crop_shape=person_crop.shape[:2],
            upscale_min_side=0,
        )
        if not _face_passes_person_head_gate(
            abs_bbox, host_bbox, camera_location=camera_location
        ):
            return None
        x1, y1, x2, y2 = (int(v) for v in abs_bbox[:4])
        if x2 <= x1 or y2 <= y1:
            return None
        crop = crop_face_from_frame(source_frame, abs_bbox)
        if crop is None or crop.size == 0:
            return None
        face_xyxy = np.asarray(abs_bbox, dtype=np.float32).reshape(-1)[:4]
        if _reject_wall_print_face(source_frame, face_xyxy, person_bbox=host_bbox):
            return None
        if outdoor and not _face_crop_has_skin(
            crop, min_fraction=gate_skin_min_fraction(camera_location)
        ):
            return None
        mapped = replace(best, bbox=(x1, y1, x2, y2), quality_ok=True)
        return mapped, (x1, y1, x2, y2)
    except Exception:
        return None


def _supplement_face_dets_for_persons(
    source_frame: np.ndarray,
    person_bboxes: Sequence[np.ndarray],
    face_dets: Optional[list],
    face_analyzer: Optional[FaceAnalyzer],
    *,
    camera_location: str = "",
) -> list:
    """
    Ensure every visible YOLO person gets a face det when full-frame scan missed them.
    """
    merged = list(face_dets or [])
    if not LIVE_CATCH_EVERYONE or not person_bboxes or face_analyzer is None:
        return merged
    added = 0
    for pb in person_bboxes[: int(LIVE_CATCH_EVERYONE_MAX_PERSONS)]:
        if _person_has_head_band_face(pb, merged, camera_location=camera_location):
            continue
        found = _find_face_det_in_person_box(
            source_frame,
            pb,
            face_analyzer,
            camera_location=camera_location,
        )
        if found is None:
            continue
        det, _bbox = found
        merged.append(det)
        added += 1
    if added:
        try:
            from face_print_filter import filter_face_dets_reject_prints

            merged = filter_face_dets_reject_prints(source_frame, merged)
        except Exception:
            pass
    return merged


def _collect_eligible_person_bboxes(
    result,
    *,
    letterbox: Optional[LetterboxTransform] = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    source_shape: Optional[Tuple[int, ...]] = None,
    camera_location: str = "",
) -> List[np.ndarray]:
    """YOLO person boxes that pass human-shape filters."""
    if result is None or getattr(result, "boxes", None) is None or len(result.boxes) == 0:
        return []
    outdoor = is_outdoor_gate_camera(camera_location)
    min_track_conf = track_conf_for_camera(camera_location)
    min_person_h = (
        gate_min_person_height_px(camera_location)
        if outdoor
        else float(MIN_PERSON_HEIGHT_PX)
    )
    if CATCH_EVERY_PERSON and not outdoor:
        min_person_h = min(float(min_person_h), 24.0)
    boxes_t = result.boxes.xyxy
    if letterbox is not None and not letterbox.is_identity:
        boxes_t = _boxes_to_source_space(boxes_t, letterbox)
    elif scale_x != 1.0 or scale_y != 1.0:
        boxes_t = scale_xyxy_boxes(boxes_t, scale_x, scale_y)
    confs_t = result.boxes.conf
    out: List[np.ndarray] = []
    for i in range(len(boxes_t)):
        conf = float(confs_t[i].item()) if confs_t is not None else 1.0
        if conf < min_track_conf:
            continue
        height = float((boxes_t[i][3] - boxes_t[i][1]).item())
        if height < min_person_h:
            continue
        host = tensor_to_numpy(boxes_t[i])
        box = (
            int(host[0]),
            int(host[1]),
            int(host[2]),
            int(host[3]),
        )
        if not _person_box_looks_human(
            box, frame_shape=source_shape, outdoor=outdoor
        ):
            continue
        out.append(host)
    return out


def _score_track_face_crop(
    source_frame: np.ndarray,
    person_bbox,
    face_dets: Optional[list],
    face_analyzer: Optional[FaceAnalyzer] = None,
    *,
    allow_person_crop: bool = True,
    matched_face=None,
    camera_location: str = "",
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Laplacian score + hi-res face crop + face xyxy from the original frame.

    Prefer full-frame RetinaFace hits. If the hallway face is too small for a
    full-frame detect, fall back to RetinaFace inside the YOLO head/person crop.
    Backs are still rejected by landmark pose checks.
    """
    outdoor = is_outdoor_gate_camera(camera_location)
    host_bbox = (
        tensor_to_numpy(person_bbox)
        if isinstance(person_bbox, torch.Tensor)
        else np.asarray(person_bbox, dtype=np.float32)
    )

    matched = matched_face
    if matched is None and face_dets:
        matched = match_face_to_person_detections(None, host_bbox, face_dets)
    if matched is not None:
        face_w = float(matched.bbox[2] - matched.bbox[0])
        face_h = float(matched.bbox[3] - matched.bbox[1])
        distant = outdoor or max(face_w, face_h) < 48.0
        ok_lm, _reason = verify_face_detection_for_capture(
            matched,
            min_confidence=gate_face_min_confidence(camera_location) if outdoor else 0.22,
            allow_side_view=True,
            max_yaw=LIVE_FACE_MAX_YAW,
            allow_distant=distant,
            allow_high_angle=True,
            min_box_px=gate_min_face_box_px(camera_location) if outdoor else FACE_MIN_BOX_PX,
        )
        if ok_lm:
            if not _face_passes_person_head_gate(
                matched.bbox, host_bbox, camera_location=camera_location
            ):
                return 0.0, None, None
            crop = crop_face_from_frame(source_frame, matched.bbox)
            if crop is not None and crop.size > 0:
                face_xyxy = np.asarray(matched.bbox, dtype=np.float32).reshape(-1)[:4]
                if _reject_wall_print_face(
                    source_frame, face_xyxy, person_bbox=host_bbox
                ):
                    return 0.0, None, None
                if outdoor and not _face_crop_has_skin(
                    crop, min_fraction=gate_skin_min_fraction(camera_location)
                ):
                    return 0.0, None, None
                return _laplacian_on_bgr(crop), crop, face_xyxy

    # Full-frame RetinaFace often misses distant hallway faces; zoom into the
    # head/upper-body ROI and detect there (still requires a real face).
    if (
        allow_person_crop
        and face_analyzer is not None
        and analyzer_ready()
        and source_frame is not None
    ):
        found = _find_face_det_in_person_box(
            source_frame,
            host_bbox,
            face_analyzer,
            camera_location=camera_location,
        )
        if found is not None:
            _det, abs_tuple = found
            crop = crop_face_from_frame(source_frame, abs_tuple)
            if crop is not None and crop.size > 0:
                face_xyxy = np.asarray(abs_tuple, dtype=np.float32).reshape(-1)[:4]
                return _laplacian_on_bgr(crop), crop, face_xyxy
    return 0.0, None, None


def _embedding_from_sharpened_crop(
    face_crop: np.ndarray,
    face_analyzer: Optional[FaceAnalyzer],
) -> Optional[np.ndarray]:
    sr = get_super_resolution_pipeline(initialize=True)
    if sr is not None and sr.ready and face_analyzer is not None and analyzer_ready():
        restored = sr.restore_and_embed_face_patch(face_crop, face_analyzer=face_analyzer)
        if restored is not None:
            return restored[0]
    if face_analyzer is not None and analyzer_ready():
        return face_analyzer.embed_crop(face_crop)
    if models_ready():
        enhanced = apply_face_clahe(face_crop)
        result = face_from_person_bbox(
            enhanced,
            np.array([0, 0, enhanced.shape[1], enhanced.shape[0]]),
            min_sharpness=0.0,
        )
        if result is not None:
            return result[0]
    return None


def _face_crop_sharp_enough(
    aligned_face,
    min_score: float = MIN_FACE_LAPLACIAN,
) -> bool:
    if isinstance(aligned_face, torch.Tensor):
        return laplacian_variance(aligned_face) >= min_score
    return face_sharpness(aligned_face) >= min_score


def _face_crop_has_skin(
    face_bgr: Optional[np.ndarray],
    *,
    min_fraction: float = 0.08,
) -> bool:
    """Reject floor tiles / texture FPs that have almost no skin-colored pixels."""
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return False
    try:
        arr = np.asarray(face_bgr)
        if arr.ndim == 2:
            return False
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if min(int(arr.shape[0]), int(arr.shape[1])) < 8:
            return False
        ycrcb = cv2.cvtColor(arr, cv2.COLOR_BGR2YCrCb)
        # Standard Cr/Cb skin band — gray floor tiles fall outside.
        mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        frac = float(cv2.countNonZero(mask)) / float(max(1, mask.size))
        return frac >= float(min_fraction)
    except Exception:
        return False


def _reject_wall_print_face(
    frame_bgr: Optional[np.ndarray],
    face_bbox: Optional[Sequence[float]],
    *,
    person_bbox: Optional[Sequence[float]] = None,
    other_face_bboxes: Optional[Sequence[Sequence[float]]] = None,
) -> bool:
    """True when face should be dropped (framed wall photo / poster)."""
    if frame_bgr is None or face_bbox is None:
        return False
    try:
        from face_print_filter import is_likely_wall_print_or_poster

        bad, _reason = is_likely_wall_print_or_poster(
            frame_bgr,
            face_bbox,
            person_bbox=person_bbox,
            other_face_bboxes=other_face_bboxes,
        )
        return bool(bad)
    except Exception:
        return False


def _face_crop_displayable(
    face_bgr: Optional[np.ndarray],
    *,
    min_side: Optional[int] = None,
    min_sharpness: Optional[float] = None,
    min_skin_fraction: Optional[float] = None,
) -> bool:
    """
    True when a face square is large/sharp enough to show on Telegram / captures.

    Rejects tiny distant head patches that only look like noise after upscaling.
    """
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return False
    try:
        arr = np.asarray(face_bgr)
        if arr.ndim < 2:
            return False
        h, w = int(arr.shape[0]), int(arr.shape[1])
        side_gate = (
            int(min_side)
            if min_side is not None
            else int(LIVE_FACE_DISPLAY_MIN_SIDE)
        )
        if min(h, w) < side_gate:
            return False
        skin_min = (
            float(LIVE_FACE_SKIN_MIN_FRACTION)
            if min_skin_fraction is None
            else float(min_skin_fraction)
        )
        if not _face_crop_has_skin(arr, min_fraction=skin_min):
            return False
        try:
            from face_print_filter import is_likely_non_face_object

            bad_obj, _ = is_likely_non_face_object(arr)
            if bad_obj:
                return False
        except Exception:
            pass
        try:
            from face_print_filter import is_likely_streaked_or_corrupt_capture

            streak, _ = is_likely_streaked_or_corrupt_capture(arr)
            if streak:
                return False
        except Exception:
            pass
        gate = (
            float(min_sharpness)
            if min_sharpness is not None
            else float(MIN_FACE_LAPLACIAN) * 0.5
        )
        return float(face_sharpness(arr)) >= gate
    except Exception:
        return False


def _aligned_face_numpy(aligned_face) -> np.ndarray:
    if isinstance(aligned_face, torch.Tensor):
        return tensor_to_bgr_numpy(aligned_face)
    return aligned_face


def _predict_capture_attributes(
    display_crop: Optional[np.ndarray],
    aligned_face: Optional[np.ndarray] = None,
) -> Tuple[str, Optional[int]]:
    """
    Prefer a face/head crop for gender — full upper-body plates confuse
    InsightFace (men in uniform / profiles → false Female).
    """
    candidates: List[np.ndarray] = []

    def _push(crop: Optional[np.ndarray]) -> None:
        if crop is None:
            return
        try:
            arr = np.ascontiguousarray(crop)
        except Exception:
            return
        if getattr(arr, "size", 0) <= 0:
            return
        candidates.append(arr)

    # 1) Tight face first when available.
    _push(aligned_face)
    # 2) Head band from tall body plates (top ~42%).
    if display_crop is not None:
        try:
            body = np.ascontiguousarray(display_crop)
            bh, bw = int(body.shape[0]), int(body.shape[1])
            if bh >= int(bw * 1.15) and bh >= 48:
                head = body[0 : max(1, int(round(bh * 0.42)))]
                _push(head)
            _push(body)
        except Exception:
            _push(display_crop)

    if not candidates:
        return "Unknown", None

    best_gender = "Unknown"
    best_age: Optional[int] = None
    for arr in candidates:
        gender, age = predict_face_attributes(arr)
        if gender and gender != "Unknown" and age is not None:
            return gender, age
        if best_gender == "Unknown" and gender and gender != "Unknown":
            best_gender = gender
            best_age = age
        elif best_age is None and age is not None:
            best_age = age
    return best_gender, best_age


def _extract_face_for_person(
    frame,
    bbox,
    face_analyzer: Optional[FaceAnalyzer],
    face_dets: Optional[list],
    min_laplacian: float = MIN_FACE_LAPLACIAN,
    *,
    source_frame: Optional[np.ndarray] = None,
    matched_face=None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, Optional[List[float]]]]:
    """Run recognition on person crops; SR restores for embedding only (display stays natural).

    Returns (embedding, natural_face_bgr, face_area, face_xyxy) or None.
    """
    source_np = source_frame if source_frame is not None else _frame_as_bgr_numpy(frame)
    on_gpu = isinstance(frame, torch.Tensor)
    face_data = None
    natural_face: Optional[np.ndarray] = None
    face_xyxy: Optional[List[float]] = None

    sr = get_super_resolution_pipeline(initialize=True)
    if sr is not None and sr.ready and face_analyzer is not None:
        host_bbox = tensor_to_numpy(bbox) if isinstance(bbox, torch.Tensor) else np.asarray(
            bbox, dtype=np.float32
        )
        sr_hit = sr.process_person_bbox(
            source_np,
            host_bbox,
            face_analyzer=face_analyzer,
        )
        if sr_hit is not None:
            src_bbox = getattr(sr_hit, "source_face_bbox", None)
            skin_ok = True
            display = None
            if src_bbox is not None:
                face_xyxy = [float(v) for v in src_bbox[:4]]
                natural = crop_face_from_frame(source_np, src_bbox)
                if natural is not None and natural.size > 0:
                    if not _face_crop_has_skin(natural):
                        skin_ok = False
                    else:
                        display = natural
                        natural_face = natural
            # Never fall back to GFPGAN "restored" pixels for saves/Telegram.
            if skin_ok and display is not None:
                face_data = (
                    sr_hit.embedding,
                    display,
                    sr_hit.face_area,
                )
            else:
                face_xyxy = None
                natural_face = None

    if face_dets is not None and face_analyzer is not None and face_data is None:
        person_bbox = tensor_to_numpy(bbox) if isinstance(bbox, torch.Tensor) else bbox
        matched = matched_face
        if matched is None:
            matched = match_face_to_person_detections(None, person_bbox, face_dets)
        if matched is not None and matched.embedding is not None:
            ok_lm, _reason = verify_face_detection_for_capture(
                matched,
                min_confidence=0.22,
                allow_side_view=True,
                max_yaw=LIVE_FACE_MAX_YAW,
                allow_distant=True,
            )
            if not ok_lm:
                return None
            aligned_face = crop_face_from_frame(source_np, matched.bbox)
            if aligned_face is None or aligned_face.size == 0:
                return None
            if not _face_crop_has_skin(aligned_face):
                return None
            natural_face = aligned_face
            face_xyxy = [float(v) for v in matched.bbox[:4]]
            if not _face_crop_sharp_enough(aligned_face, min_laplacian):
                if sr is not None and sr.ready:
                    restored = sr.restore_and_embed_face_patch(
                        aligned_face, face_analyzer=face_analyzer
                    )
                    if restored is not None:
                        emb, _restored_bgr, _was = restored
                        x1, y1, x2, y2 = matched.bbox
                        face_data = (
                            emb,
                            aligned_face,
                            float((x2 - x1) * (y2 - y1)),
                        )
                if face_data is None:
                    return None
            else:
                x1, y1, x2, y2 = matched.bbox
                face_data = (
                    matched.embedding,
                    aligned_face,
                    float((x2 - x1) * (y2 - y1)),
                )

    if face_data is None and analyzer_ready():
        if on_gpu:
            bbox_t = bbox if isinstance(bbox, torch.Tensor) else torch.tensor(
                bbox, device=get_device(), dtype=torch.float32
            )
            face_data = extract_face_from_person_crop_gpu(
                frame, bbox_t, analyzer=face_analyzer, source_frame=source_np
            )
        else:
            face_data = extract_face_from_person_crop(
                frame, bbox, analyzer=face_analyzer, source_frame=source_np
            )

    if face_data is None and models_ready():
        host_bbox = tensor_to_numpy(bbox) if isinstance(bbox, torch.Tensor) else bbox
        face_data = face_from_person_bbox(
            source_np, host_bbox, min_sharpness=min_laplacian
        )

    if face_data is None:
        return None
    feature, aligned_face, face_area = face_data
    if natural_face is not None and natural_face.size > 0:
        aligned_face = natural_face
    if not _face_crop_sharp_enough(aligned_face, min_laplacian):
        if sr is not None and sr.ready and face_analyzer is not None:
            restored = sr.restore_and_embed_face_patch(
                aligned_face, face_analyzer=face_analyzer
            )
            if restored is not None:
                return restored[0], aligned_face, face_area, face_xyxy
        return None
    if not _face_crop_displayable(
        aligned_face, min_sharpness=max(3.0, float(min_laplacian) * 0.5)
    ):
        return None
    return feature, aligned_face, face_area, face_xyxy


def _make_face_analyzer() -> Optional[FaceAnalyzer]:
    if not analyzer_ready():
        return None
    anti_spoof_path = (
        DEFAULT_ANTI_SPOOF_MODEL if DEFAULT_ANTI_SPOOF_MODEL.is_file() else None
    )
    return FaceAnalyzer(
        min_size=max(12, int(LIVE_FACE_MIN_BOX_PX)),
        max_blur=MIN_FACE_LAPLACIAN,
        max_yaw=LIVE_FACE_MAX_YAW,
        anti_spoof_model=anti_spoof_path,
        detect_threshold=LIVE_FACE_DETECT_CONFIDENCE,
    )


def _run_face_pipeline(
    frame,
    analyzer: FaceAnalyzer,
    skipper: AdaptiveFrameSkip,
    frame_index: int,
    *,
    force: bool = False,
) -> Optional[list]:
    if not force and not _should_run_face_inference(frame_index, skipper):
        return None
    t0 = time.perf_counter()

    def _infer() -> list:
        # Live CCTV must use the relaxed scan gate. The VRAM/GPU path previously
        # used detect_and_embed_gpu (strict blur/landmarks) and blocked hallway
        # captures + Telegram even when YOLO drew a person box.
        frame_np = _frame_as_bgr_numpy(frame)
        return analyzer.detect_and_embed_scan(frame_np)

    # မြန်မာ: မျက်နှာ မတွေ့/ရလဒ်ဗလာ ဖြစ်ပါက crash မဖြစ်စေရန် safe inference
    dets = SafeInferenceRunner.run(
        _infer,
        context=f"face pipeline frame {frame_index}",
    )
    if dets is None:
        return None
    skipper.update((time.perf_counter() - t0) * 1000.0)
    return dets


def _track_via_gpu_worker(
    model,
    slot_frames: List[Tuple["CameraReader", Optional[np.ndarray], bool]],
    frame_index: int,
) -> Dict[str, Tuple[object, object, LetterboxTransform]]:
    """Batch YOLO track on the shared model (single VRAM instance, sync batch)."""
    jobs: List[FrameJob] = []
    pending_frames: Dict[str, np.ndarray] = {}
    for reader, frame, ok in slot_frames:
        if not ok or frame is None:
            continue
        jobs.append(
            FrameJob(
                camera_location=reader.location,
                slot=reader.slot,
                frame_index=frame_index,
                frame=frame,
            )
        )
        pending_frames[reader.location] = frame

    tracked: Dict[str, Tuple[object, object, LetterboxTransform]] = {}
    if not jobs:
        return tracked

    for packet in _process_batch(model, jobs):
        src_frame = pending_frames.get(packet.camera_location)
        if src_frame is None or packet.raw_result is None:
            continue
        if vram_pipeline_active():
            src = bgr_numpy_to_gpu(src_frame)
        else:
            src = src_frame
        tracked[packet.camera_location] = (
            packet.raw_result,
            src,
            packet.letterbox,
        )
    return tracked


def _track_via_gpu_worker_async(
    gpu_worker: GPUInferenceWorker,
    slot_frames: List[Tuple["CameraReader", Optional[np.ndarray], bool]],
    frame_index: int,
    *,
    timeout_sec: float = 3.0,
) -> Dict[str, Tuple[object, object, LetterboxTransform]]:
    """Submit to background GPU worker and wait for this frame's batch to finish."""
    pending_frames: Dict[str, np.ndarray] = {}
    expected: set = set()
    for reader, frame, ok in slot_frames:
        if not ok or frame is None:
            continue
        if gpu_worker.submit(
            FrameJob(
                camera_location=reader.location,
                slot=reader.slot,
                frame_index=frame_index,
                frame=frame,
            )
        ):
            pending_frames[reader.location] = frame
            expected.add(reader.location)

    tracked: Dict[str, Tuple[object, object, LetterboxTransform]] = {}
    if not expected:
        return tracked

    deadline = time.perf_counter() + timeout_sec
    while expected - set(tracked.keys()) and time.perf_counter() < deadline:
        for packet in gpu_worker.poll_results(max_items=len(expected) + 8):
            if packet.frame_index != frame_index:
                continue
            if packet.camera_location not in expected:
                continue
            src_frame = pending_frames.get(packet.camera_location)
            if src_frame is None or packet.raw_result is None:
                continue
            if vram_pipeline_active():
                src = bgr_numpy_to_gpu(src_frame)
            else:
                src = src_frame
            tracked[packet.camera_location] = (
                packet.raw_result,
                src,
                packet.letterbox,
            )
        if expected - set(tracked.keys()):
            time.sleep(0.002)
    return tracked


def _track_all_cameras(
    model,
    slot_frames: List[Tuple["CameraReader", Optional[np.ndarray], bool]],
) -> Dict[str, Tuple[object, object, LetterboxTransform]]:
    """Batch-track live cameras on downscaled frames; bboxes map back to full size."""
    live: List[Tuple["CameraReader", object, object, LetterboxTransform]] = []
    for reader, frame, ok in slot_frames:
        if not ok or frame is None:
            continue
        if vram_pipeline_active():
            frame_gpu = bgr_numpy_to_gpu(frame)
            scaled_gpu, letterbox = resize_hwc(
                frame_gpu,
                tracker_config.INFERENCE_WIDTH,
                tracker_config.INFERENCE_HEIGHT,
            )
            live.append((reader, scaled_gpu, frame_gpu, letterbox))
        else:
            scaled, letterbox = _resize_for_inference(frame)
            live.append((reader, scaled, frame, letterbox))

    if not live:
        return {}

    readers = [item[0] for item in live]
    scaled_frames = [item[1] for item in live]
    full_frames = [item[2] for item in live]
    scales = [item[3] for item in live]

    track_kw = _yolo_track_kwargs()
    track_input = yolo_track_batch(scaled_frames)
    if isinstance(track_input, list):
        if len(track_input) == 1:
            results = [model.track(track_input[0], **track_kw)[0]]
        else:
            results = model.track(track_input, **track_kw)
    else:
        results = model.track(track_input, **track_kw)
    if not isinstance(results, list):
        results = [results]
    if len(results) != len(scaled_frames):
        print(
            f"[WARN] batch track returned {len(results)}/{len(scaled_frames)} "
            "results — tracking each camera separately"
        )
        results = [
            model.track(yolo_track_source(scaled), **track_kw)[0]
            for scaled in scaled_frames
        ]

    tracked = {
        reader.location: (result, full_frame, letterbox)
        for reader, result, full_frame, letterbox in zip(
            readers, results, full_frames, scales
        )
    }
    # Drop AI-sized tensors; callers keep full-res frames only.
    del scaled_frames
    return tracked


def _visit_key(
    camera_location: str,
    track_id: Optional[int],
    bbox: np.ndarray,
) -> Tuple[str, object]:
    if track_id is not None:
        return (camera_location, int(track_id))
    x1, y1, x2, y2 = bbox
    return (
        camera_location,
        "grid",
        int((x1 + x2) / 80),
        int((y1 + y2) / 80),
    )


def _face_grid_visit_key(
    camera_location: str,
    bbox: Sequence[float],
    *,
    grid: int = 80,
) -> Tuple[str, object]:
    """Cooldown key for face-direct watchlist matching (no YOLO track required)."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return (
        camera_location,
        "face",
        int(cx / grid),
        int(cy / grid),
    )


def _standalone_face_allowed(
    det,
    person_bboxes: Optional[Sequence[np.ndarray]],
    camera_location: str,
) -> bool:
    """Reject arm/shoulder false positives for gate cameras."""
    bbox = getattr(det, "bbox", None)
    if bbox is None:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError, IndexError):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    outdoor = is_outdoor_gate_camera(camera_location)
    side = max(x2 - x1, y2 - y1)
    min_side = gate_min_face_box_px(camera_location) if outdoor else FACE_MIN_BOX_PX
    if side < float(min_side):
        return False
    if not person_bboxes:
        return True
    host_boxes = [
        np.asarray(pb, dtype=np.float32).reshape(-1)[:4]
        for pb in person_bboxes
        if pb is not None
    ]
    overlapping = [
        pb
        for pb in host_boxes
        if _face_in_person_bbox((int(x1), int(y1), int(x2), int(y2)), pb)
    ]
    if not overlapping:
        return not outdoor
    return any(
        face_bbox_in_person_head_band((x1, y1, x2, y2), pb, outdoor=outdoor)[0]
        for pb in overlapping
    )


def _process_standalone_face_watchlist(
    conn: sqlite3.Connection,
    camera_location: str,
    source_np: np.ndarray,
    face_dets: Optional[list],
    last_logged: Dict[Tuple[str, object], float],
    track_to_known: Dict[str, Dict[int, int]],
    shared_gallery: Optional[SharedFaceGallery],
    cooldown: float,
    now: float,
    *,
    face_analyzer: Optional[FaceAnalyzer] = None,
    telegram_batch: Optional[List[dict]] = None,
    person_bboxes: Optional[Sequence[np.ndarray]] = None,
) -> int:
    """
    Match RetinaFace detections directly against the watchlist.

    CCTV often shows a face box without a reliable YOLO person-body association;
    the body-track path alone never reaches watchlist / Telegram alerts.
    """
    if not face_dets:
        return 0

    log_count = 0
    own_batch = telegram_batch is None
    batch = [] if own_batch else telegram_batch
    assert batch is not None
    best_by_key: Dict[Tuple[str, object], Tuple[float, object, Tuple[float, float, float, float]]] = {}
    for det in face_dets:
        if not _standalone_face_allowed(det, person_bboxes, camera_location):
            continue
        bbox = getattr(det, "bbox", None)
        if bbox is None:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        key = _face_grid_visit_key(camera_location, (x1, y1, x2, y2))
        prev = best_by_key.get(key)
        if prev is None or area > prev[0]:
            best_by_key[key] = (area, det, (x1, y1, x2, y2))

    for _area, det, bbox_xyxy in best_by_key.values():
        visit_key = _face_grid_visit_key(camera_location, bbox_xyxy)
        if now - last_logged.get(visit_key, 0.0) < cooldown:
            continue

        # Only match faces that passed the full detect_and_embed quality gate.
        # Do not re-embed random RetinaFace boxes (posts, knobs, etc.) via embed_crop.
        if not getattr(det, "quality_ok", False):
            continue
        feature = getattr(det, "embedding", None)
        if feature is None:
            continue

        ok_lm, _lm_reason = verify_face_detection_for_capture(
            det,
            min_confidence=(
                gate_face_min_confidence(camera_location)
                if is_outdoor_gate_camera(camera_location)
                else float(LIVE_FACE_LANDMARK_MIN_CONF)
            ),
            allow_side_view=True,
            max_yaw=LIVE_FACE_MAX_YAW,
            allow_distant=is_outdoor_gate_camera(camera_location),
            allow_high_angle=is_outdoor_gate_camera(camera_location),
            min_box_px=(
                gate_min_face_box_px(camera_location)
                if is_outdoor_gate_camera(camera_location)
                else FACE_MIN_BOX_PX
            ),
        )
        if not ok_lm:
            continue

        aligned_face = crop_face_from_frame(source_np, det.bbox)
        if aligned_face is None or aligned_face.size == 0:
            continue
        outdoor = is_outdoor_gate_camera(camera_location)
        skin_min = (
            gate_skin_min_fraction(camera_location)
            if outdoor
            else float(LIVE_FACE_SKIN_MIN_FRACTION)
        )
        if not _face_crop_has_skin(aligned_face, min_fraction=skin_min):
            continue
        sharp_min = (
            gate_min_face_laplacian(camera_location) * 0.5
            if outdoor
            else MIN_FACE_LAPLACIAN * 0.5
        )
        if not _face_crop_sharp_enough(aligned_face, sharp_min):
            continue
        try:
            from face_print_filter import is_likely_non_face_object

            bad_obj, _ = is_likely_non_face_object(aligned_face)
            if bad_obj:
                continue
        except Exception:
            pass
        sibling_boxes = [b for _, _, b in best_by_key.values()]
        if _reject_wall_print_face(
            source_np, bbox_xyxy, other_face_bboxes=sibling_boxes
        ):
            continue
        try:
            from face_print_filter import is_likely_invalid_face_crop

            bad_crop, _reason = is_likely_invalid_face_crop(aligned_face)
            if bad_crop:
                continue
        except Exception:
            pass

        face_area = float(aligned_face.shape[0] * aligned_face.shape[1])
        if _commit_face_candidate(
            conn,
            camera_location,
            None,
            visit_key,
            np.asarray(feature, dtype=np.float32).reshape(-1),
            aligned_face,
            face_area,
            last_logged,
            track_to_known,
            shared_gallery,
            cooldown,
            now,
            bbox=[float(v) for v in bbox_xyxy],
            frame_bgr=source_np,
            telegram_batch=batch,
        ):
            log_count += 1
    if own_batch:
        _flush_telegram_catch_batch(camera_location, source_np, batch)
    return log_count


def _run_watchlist_face_pass(
    conn: sqlite3.Connection,
    camera_location: str,
    frame,
    face_analyzer: Optional[FaceAnalyzer],
    skipper: AdaptiveFrameSkip,
    frame_index: int,
    last_logged: Dict[Tuple[str, object], float],
    track_to_known: Dict[str, Dict[int, int]],
    shared_gallery: Optional[SharedFaceGallery],
    cooldown: float,
    now: float,
) -> int:
    """Face-only watchlist pass — does not require YOLO person tracks."""
    if face_analyzer is None:
        return 0
    face_dets = _run_face_pipeline(frame, face_analyzer, skipper, frame_index)
    if not face_dets:
        return 0
    source_np = _frame_as_bgr_numpy(frame)
    return _process_standalone_face_watchlist(
        conn,
        camera_location,
        source_np,
        face_dets,
        last_logged,
        track_to_known,
        shared_gallery,
        cooldown,
        now,
        face_analyzer=face_analyzer,
    )


def _should_person_catch(camera_location: str) -> bool:
    return bool(CATCH_EVERY_PERSON) or is_outdoor_gate_camera(camera_location)


def _allow_pose_person_fallback(reason: str) -> bool:
    """Allow person-plate fallback for soft poses — never for backs / faces-only mode."""
    if FACE_ONLY_CATCH or not CATCH_BACKS:
        # Faces-only: never save a body plate as a substitute for a face.
        if FACE_ONLY_CATCH:
            return False
    try:
        from face_print_filter import is_keepable_person_pose_reason

        if not is_keepable_person_pose_reason(reason):
            return False
    except Exception:
        return False
    text = str(reason or "").lower()
    is_rear = any(
        token in text
        for token in (
            "rear_view",
            "back_of_head",
            "back_head",
            "walking_away",
            "nape",
            "crown_neck",
            "ears_neck",
            "neck_only",
            "dull_hair",
            "no_face_bright",
            "no_face(",
            "hand_crown",
        )
    )
    if is_rear and not CATCH_BACKS:
        return False
    return True


def _person_plate_ok_for_catch(
    display_crop: Optional[np.ndarray],
    camera_location: str,
) -> bool:
    """Keep real people. Drop wood / floor / railing junk and (by default) backs."""
    if display_crop is None or getattr(display_crop, "size", 0) == 0:
        return False
    if not face_crop_is_photographic(display_crop):
        return False
    try:
        arr = np.ascontiguousarray(display_crop)
        h, w = int(arr.shape[0]), int(arr.shape[1])
    except Exception:
        return False
    # Tiny hallway blobs (~50px) are not usable — always reject.
    min_side_req = max(48, int(PERSON_CATCH_MIN_SIDE))
    if min(h, w) < min_side_req:
        print(
            f"[FACE] skip tiny plate @ cam {camera_location}: "
            f"{w}x{h} < {min_side_req}px"
        )
        return False
    try:
        from face_print_filter import (
            is_junk_person_plate_reason,
            is_likely_non_face_object,
            is_likely_rear_view,
            is_likely_streaked_or_corrupt_capture,
            is_likely_unusable_face_plate,
            is_likely_wood_post_or_rail,
        )

        if not CATCH_BACKS:
            rear, rear_reason = is_likely_rear_view(arr)
            if rear:
                print(f"[FACE] skip back @ cam {camera_location}: {rear_reason}")
                return False
        mush, mush_reason = is_likely_unusable_face_plate(arr)
        if mush:
            print(f"[FACE] skip unusable @ cam {camera_location}: {mush_reason}")
            return False
        wood, wood_reason = is_likely_wood_post_or_rail(arr)
        if wood:
            print(f"[FACE] skip wood/rail @ cam {camera_location}: {wood_reason}")
            return False
        bad_obj, obj_reason = is_likely_non_face_object(arr)
        if bad_obj and is_junk_person_plate_reason(obj_reason):
            print(f"[FACE] skip non-person @ cam {camera_location}: {obj_reason}")
            return False
        # Tall/narrow YOLO person boxes look like "narrow_strip" to the face-crop
        # streak heuristic — keep them when catching every person.
        if not CATCH_EVERY_PERSON:
            streak, streak_reason = is_likely_streaked_or_corrupt_capture(arr)
            if streak:
                print(
                    f"[FACE] skip streaky/distorted @ cam {camera_location}: "
                    f"{streak_reason}"
                )
                return False
        else:
            streak, streak_reason = is_likely_streaked_or_corrupt_capture(arr)
            if streak and any(
                tok in str(streak_reason).lower()
                for tok in ("row_streak", "edge_replicate", "soft_body")
            ):
                print(
                    f"[FACE] skip streaky/distorted @ cam {camera_location}: "
                    f"{streak_reason}"
                )
                return False
    except Exception:
        pass
    # Always require a minimally displayable plate (CATCH_EVERY_PERSON used to
    # bypass this and flooded Telegram with 50px mush).
    if not _face_crop_has_skin(display_crop, min_fraction=0.05):
        print(f"[FACE] skip no-skin plate @ cam {camera_location}")
        return False
    if not _face_crop_displayable(
        display_crop,
        min_side=min_side_req,
        min_sharpness=max(1.0, gate_min_face_laplacian(camera_location) * 0.25),
        min_skin_fraction=0.05,
    ):
        print(f"[FACE] skip undisplayable gate plate @ cam {camera_location}")
        return False
    return True


def _fallback_person_plate(
    conn: sqlite3.Connection,
    camera_location: str,
    track_id: Optional[int],
    visit_key: Tuple[str, object],
    head_crop: Optional[np.ndarray],
    last_logged: Dict[Tuple[str, object], float],
    cooldown: float,
    now: float,
    *,
    bbox: Optional[Sequence[float]] = None,
    frame_bgr: Optional[np.ndarray] = None,
    telegram_batch: Optional[List[dict]] = None,
    why: str = "",
) -> bool:
    if not _should_person_catch(camera_location):
        return False
    crop = head_crop
    if crop is None or getattr(crop, "size", 0) == 0:
        crop = _crop_person_plate(frame_bgr, bbox)
    if crop is None:
        return False
    if why:
        print(f"[FACE] person-catch @ cam {camera_location}: {why}")
    return _commit_person_catch_without_embed(
        conn,
        camera_location,
        track_id,
        visit_key,
        crop,
        last_logged,
        cooldown,
        now,
        bbox=bbox,
        frame_bgr=frame_bgr,
        telegram_batch=telegram_batch,
    )


def _commit_face_candidate(
    conn: sqlite3.Connection,
    camera_location: str,
    track_id: Optional[int],
    visit_key: Tuple[str, object],
    feature: np.ndarray,
    aligned_face: np.ndarray,
    face_area: float,
    last_logged: Dict[Tuple[str, object], float],
    track_to_known: Dict[str, Dict[int, int]],
    shared_gallery: Optional[SharedFaceGallery],
    cooldown: float,
    now: float,
    bbox: Optional[Sequence[float]] = None,
    frame_bgr: Optional[np.ndarray] = None,
    person_bbox: Optional[Sequence[float]] = None,
    telegram_batch: Optional[List[dict]] = None,
) -> bool:
    """Run gallery matching and persist one finalized face sighting."""
    if now - last_logged.get(visit_key, 0.0) < cooldown:
        return False
    outdoor = is_outdoor_gate_camera(camera_location)
    plate_bbox = person_bbox if person_bbox is not None else bbox
    skin_min = (
        gate_skin_min_fraction(camera_location)
        if outdoor
        else float(LIVE_FACE_SKIN_MIN_FRACTION)
    )

    def _person_fallback(why: str) -> bool:
        return _fallback_person_plate(
            conn,
            camera_location,
            track_id,
            visit_key,
            aligned_face,
            last_logged,
            cooldown,
            now,
            bbox=plate_bbox,
            frame_bgr=frame_bgr,
            telegram_batch=telegram_batch,
            why=why,
        )

    if not _face_crop_has_skin(aligned_face, min_fraction=skin_min):
        return _person_fallback("no-skin face crop")
    if not face_crop_is_photographic(aligned_face):
        return _person_fallback("non-photographic face crop")
    try:
        from face_print_filter import (
            is_likely_non_face_object,
        )

        bad_obj, obj_reason = is_likely_non_face_object(aligned_face)
        if bad_obj:
            if _allow_pose_person_fallback(obj_reason):
                return _person_fallback(obj_reason)
            return False
    except Exception:
        pass
    try:
        from face_print_filter import (
            is_likely_invalid_face_crop,
            is_likely_rear_view,
        )

        bad_crop, reason = is_likely_invalid_face_crop(aligned_face)
        if bad_crop:
            print(f"[FACE] reject invalid crop @ cam {camera_location}: {reason}")
            if _allow_pose_person_fallback(reason):
                return _person_fallback(reason)
            return False
        rear, rear_reason = is_likely_rear_view(aligned_face)
        if rear:
            print(f"[FACE] reject back-of-head @ cam {camera_location}: {rear_reason}")
            if CATCH_BACKS:
                return _person_fallback(rear_reason)
            return False
    except Exception:
        pass
    if _reject_wall_print_face(frame_bgr, bbox):
        return False
    if person_bbox is not None and bbox is not None:
        try:
            if not _face_passes_person_head_gate(
                bbox,
                np.asarray(person_bbox, dtype=np.float32).reshape(-1)[:4],
                camera_location=camera_location,
            ):
                return _person_fallback("face not on person head")
        except Exception:
            pass

    display_crop = choose_display_capture_crop(
        frame_bgr,
        bbox,
        aligned_face,
        person_bbox,
    )
    # Walking-away / back plates are skipped unless CATCH_BACKS=1.
    try:
        from face_print_filter import is_likely_rear_view

        rear_early, rear_early_reason = is_likely_rear_view(display_crop)
        if rear_early:
            print(
                f"[FACE] reject walking-away crop @ cam {camera_location}: "
                f"{rear_early_reason}"
            )
            if CATCH_BACKS:
                return _person_fallback(rear_early_reason)
            return False
    except Exception:
        pass
    if capture_display_is_body_crop():
        try:
            from captures import display_body_crop_is_valid, display_image_is_portrait_body

            if (
                not display_image_is_portrait_body(display_crop)
                or not display_body_crop_is_valid(display_crop)
            ):
                display_crop = fallback_display_capture_crop(
                    frame_bgr,
                    bbox,
                    aligned_face,
                    person_bbox,
                )
                print(
                    f"[FACE] body crop fallback @ cam {camera_location} "
                    f"(saved face/person plate — head-to-waist unavailable)"
                )
        except Exception:
            pass
    try:
        from face_print_filter import (
            is_likely_non_face_object,
            is_likely_rear_view,
            is_likely_streaked_or_corrupt_capture,
            is_likely_unusable_face_plate,
            is_likely_wood_post_or_rail,
        )

        wood, wood_reason = is_likely_wood_post_or_rail(display_crop)
        if wood:
            print(f"[FACE] reject wood/rail display @ cam {camera_location}: {wood_reason}")
            return False
        mush, mush_reason = is_likely_unusable_face_plate(display_crop)
        if mush:
            print(f"[FACE] reject unusable display @ cam {camera_location}: {mush_reason}")
            return False
        bad_obj, obj_reason = is_likely_non_face_object(display_crop)
        if bad_obj:
            print(f"[FACE] reject non-face display @ cam {camera_location}: {obj_reason}")
            if _allow_pose_person_fallback(obj_reason):
                return _person_fallback(obj_reason)
            return False
        rear, rear_reason = is_likely_rear_view(display_crop)
        if rear:
            print(f"[FACE] reject walking-away crop @ cam {camera_location}: {rear_reason}")
            if CATCH_BACKS:
                return _person_fallback(rear_reason)
            return False
        streak, streak_reason = is_likely_streaked_or_corrupt_capture(display_crop)
        if streak:
            print(
                f"[FACE] reject streaky/distorted crop @ cam {camera_location}: "
                f"{streak_reason}"
            )
            return False
    except Exception:
        pass

    # Never save mush / floor / non-face plates to the gallery (Telegram already
    # gated this; dashboard review was still showing false "Stranger" faces).
    disp_min = max(int(PERSON_CATCH_MIN_SIDE), int(LIVE_FACE_DISPLAY_MIN_SIDE))
    if outdoor:
        disp_min = max(56, int(PERSON_CATCH_MIN_SIDE) - 8)
    if not (
        _face_crop_displayable(display_crop, min_side=disp_min)
        or _face_crop_displayable(aligned_face, min_side=disp_min)
    ):
        print(
            f"[FACE] reject undisplayable crop @ cam {camera_location} "
            f"(min_side={disp_min})"
        )
        return _person_fallback(f"undisplayable(min_side={disp_min})")

    # Watchlist / Telegram path is ArcFace 512-d only. Skip OpenCV SFace (128-d)
    # fallbacks so they cannot register strangers or miss watchlist matches.
    try:
        feat_dim = int(np.asarray(feature, dtype=np.float32).reshape(-1).size)
    except Exception:
        feat_dim = 0
    if feat_dim != 512:
        return _person_fallback(f"embed_dim={feat_dim}")

    known_maps = track_to_known.setdefault(camera_location, {})
    if shared_gallery is not None and track_id is not None:
        shared_gallery.add(str(int(track_id)), camera_location, feature)
        cross_id, cross_score = shared_gallery.match(
            feature, exclude_camera=camera_location
        )
        if cross_id is not None:
            print(
                f"[CROSS-CAM] track {track_id} @ {camera_location} "
                f"matches gallery {cross_id} ({cross_score:.0%})"
            )
            try:
                from face_stats import record_cross_camera_match

                record_cross_camera_match(
                    str(int(track_id)),
                    camera_location,
                    cross_id,
                    "gallery",
                    cross_score,
                )
            except Exception:
                pass

    db_track_id = (
        int(track_id)
        if track_id is not None
        else abs(hash(visit_key)) % 900_000 + 100_000
    )

    matched_id, match_distance = find_best_face_match(feature, conn)
    alert_name = None
    if matched_id is not None:
        try:
            alert_name = resolve_known_person_name(conn, matched_id)
        except Exception:
            alert_name = None
    # Tier routing for dashboard / gray-zone audit (Telegram is enrolled-only below).
    apply_identity_tiers(
        conn,
        matched_id,
        match_distance,
        camera_location,
        aligned_face,
        send_alert_fn=None,
        frame_bgr=frame_bgr,
        bbox=bbox,
        person_name=alert_name,
        camera_name=f"CAM-{camera_location}",
    )

    # Explicit CCTV watchlist alert for enrolled names (in case tier path is skipped).
    try:
        from identity import is_enrolled_watchlist_person
        from live_detection_hub import publish_person_alert
        from orion_frame_pipeline import IdentityTierRouter, MatchTier

        if matched_id is not None and is_enrolled_watchlist_person(conn, matched_id):
            decision = IdentityTierRouter.evaluate(matched_id, match_distance)
            if decision.tier == MatchTier.BLACKLIST_ALERT:
                conf = float(decision.match_percent or 0)
                if conf <= 0 and match_distance < 900:
                    conf = float(confidence_score_from_distance(match_distance))
                publish_person_alert(
                    camera=str(camera_location),
                    name=str(alert_name or f"Person #{matched_id}"),
                    confidence=max(1.0, conf),
                    face_bgr=display_crop,
                    person_id=int(matched_id),
                    known_person_id=int(matched_id),
                    bbox=bbox,
                    alert_type="blacklist",
                    force=True,
                )
    except Exception:
        pass

    known_id = resolve_known_person_from_face(
        conn, feature, known_maps, db_track_id
    )
    if known_id is not None:
        upsert_best_face_profile(
            conn,
            known_id,
            feature,
            aligned_face,
            face_sharpness(aligned_face),
        )
    person_label = None
    is_stranger = True
    try:
        from identity import (
            capture_identity_label,
            is_confident_named_match,
            is_named_person,
            resolve_live_capture_identity,
        )

        display_name, is_stranger, matched_named_id, match_distance = (
            resolve_live_capture_identity(
                conn,
                feature,
                known_person_id=known_id,
                person_id=db_track_id,
                face_bgr=aligned_face,
                camera_location=camera_location,
            )
        )
        if is_stranger:
            # Soft / looking-down crops must not keep a prior Named sticky ID.
            if known_id is not None and is_named_person(conn, int(known_id)):
                if track_id is not None and known_maps.get(int(track_id)) == int(
                    known_id
                ):
                    known_maps.pop(int(track_id), None)
                known_id = None
            person_label = None
        elif matched_named_id is not None:
            matched_id = matched_named_id
            # Bind this CCTV catch to the Named People record so the enrolled
            # name is stored with the live camera crop (not a new stranger id).
            known_id = int(matched_named_id)
            if track_id is not None:
                known_maps[int(track_id)] = int(matched_named_id)
            try:
                from identity import touch_person

                touch_person(conn, int(matched_named_id))
            except Exception:
                pass
            person_label = display_name
        elif (
            matched_id is not None
            and is_named_person(conn, int(matched_id))
            and is_confident_named_match(match_distance)
        ):
            raw_label, still_stranger = capture_identity_label(
                conn,
                int(matched_id),
                person_id=db_track_id,
                match_distance=match_distance,
            )
            if raw_label and not still_stranger:
                is_stranger = False
                person_label = raw_label
                display_name = raw_label
                known_id = int(matched_id)
                if track_id is not None:
                    known_maps[int(track_id)] = int(matched_id)
                try:
                    from identity import touch_person

                    touch_person(conn, int(matched_id))
                except Exception:
                    pass
    except Exception:
        person_label = None
        is_stranger = True
        display_name = "Stranger"
    capture_file = save_face_capture(
        display_crop,
        known_id,
        db_track_id,
        person_name=person_label,
    )
    scene_file = None
    if frame_bgr is not None:
        scene_file = save_scene_capture(
            frame_bgr,
            known_id,
            db_track_id,
            face_bbox=bbox,
            face_filename=capture_file,
            person_name=person_label,
        )
        if scene_file:
            print(f"[CAPTURE] scene saved {scene_file}")
        else:
            print("[WARN] scene capture failed (frame present but write returned None)")
    else:
        print("[WARN] scene capture skipped — no frame_bgr on commit")
    gender, age = _predict_capture_attributes(display_crop, aligned_face)
    log_visit(
        conn,
        db_track_id,
        camera_location,
        known_person_id=known_id,
        capture_path=capture_file,
        gender=gender,
        age=age,
        scene_path=scene_file,
        display_name=person_label,
    )
    last_logged[visit_key] = now
    _FACE_DETECT_COUNTS[camera_location] = (
        _FACE_DETECT_COUNTS.get(camera_location, 0) + 1
    )
    write_faces_detected(camera_location, _FACE_DETECT_COUNTS[camera_location])

    # Telegram: non-watchlist CCTV incidents (and optionally enrolled watchlist).
    try:
        from identity import is_enrolled_watchlist_person

        on_watchlist = (
            known_id is not None and is_enrolled_watchlist_person(conn, int(known_id))
        ) or (
            matched_id is not None and is_enrolled_watchlist_person(conn, int(matched_id))
        )
        named_hit = bool(person_label and not is_stranger and not on_watchlist)
        named_tg = bool(TELEGRAM_ALERT_NAMED and named_hit)
        watchlist_tg = TELEGRAM_ALERT_WATCHLIST and on_watchlist
        non_watchlist_tg = (
            TELEGRAM_ALERT_NON_WATCHLIST and not on_watchlist and not named_hit
        )
        is_incident = bool(non_watchlist_tg and not watchlist_tg and not named_hit)
        tg_min_side = (
            TELEGRAM_FACE_MIN_SIDE if (non_watchlist_tg or named_tg) else LIVE_FACE_DISPLAY_MIN_SIDE
        )
        # Distant hallway faces are often 56–70px — do not force PERSON_CATCH_MIN_SIDE.
        tg_min_side = max(48, int(tg_min_side))
        tg_skin = None
        if is_outdoor_gate_camera(str(camera_location)):
            tg_skin = 0.06
        # Named People match or stranger incident: still require a usable face
        # plate — tiny mush / letterboxed black-bar panels must not Telegram.
        tg_eligible = False
        if (is_incident or named_tg) and capture_file:
            tg_eligible = _face_crop_displayable(
                display_crop, min_side=tg_min_side, min_skin_fraction=tg_skin
            ) or _face_crop_displayable(
                aligned_face, min_side=tg_min_side, min_skin_fraction=tg_skin
            )
        elif watchlist_tg or non_watchlist_tg:
            tg_eligible = _face_crop_displayable(
                aligned_face, min_side=tg_min_side, min_skin_fraction=tg_skin
            ) or _face_crop_displayable(
                display_crop, min_side=tg_min_side, min_skin_fraction=tg_skin
            )
        if CATCH_EVERY_PERSON and not tg_eligible and capture_file and (
            is_incident or named_tg or non_watchlist_tg
        ):
            try:
                side = int(min(display_crop.shape[0], display_crop.shape[1]))
            except Exception:
                side = 0
            if side >= 20:
                tg_eligible = True
        if tg_eligible:
            tg_id = int(
                known_id
                if known_id is not None
                else matched_id
                if matched_id is not None
                else db_track_id
            )
            tg_name = person_label or "Stranger"
            if matched_id is not None and match_distance < 900:
                tg_conf = int(round(float(confidence_score_from_distance(match_distance))))
            else:
                tg_conf = 100
            if named_tg:
                alert_kind = "named"
            elif is_incident:
                alert_kind = "incident"
            else:
                alert_kind = "watchlist"
            try:
                from identity import describe_face_sharpness

                image_quality, _ = describe_face_sharpness(display_crop)
            except Exception:
                image_quality = ""
            payload = {
                "person_id": tg_id,
                "person_name": tg_name,
                "match_percent": max(1, tg_conf),
                "face_bgr": np.ascontiguousarray(display_crop),
                "bbox": [float(v) for v in bbox[:4]] if bbox is not None and len(bbox) >= 4 else None,
                "person_bbox": (
                    [float(v) for v in person_bbox[:4]]
                    if person_bbox is not None and len(person_bbox) >= 4
                    else None
                ),
                "is_stranger": bool(is_stranger),
                "is_named": bool(named_hit),
                "alert_kind": alert_kind,
                "image_quality": image_quality,
            }
            tg_cooldown = (
                TELEGRAM_NON_WATCHLIST_COOLDOWN_SEC
                if (is_incident or named_tg)
                else None
            )
            tg_min_pct = 1 if (is_incident or named_tg) else None
            if telegram_batch is not None:
                telegram_batch.append(payload)
            else:
                maybe_send_target_alert_async(
                    display_crop,
                    tg_id,
                    camera_location,
                    max(1, tg_conf),
                    frame_bgr=frame_bgr,
                    person_name=tg_name,
                    camera_name=f"CAM-{camera_location}",
                    location=str(camera_location),
                    bbox=bbox,
                    person_bbox=person_bbox,
                    min_percent=tg_min_pct,
                    cooldown_sec=tg_cooldown,
                    alert_kind=payload["alert_kind"],
                    image_quality=image_quality or None,
                )
        elif not watchlist_tg and not non_watchlist_tg:
            pass
        else:
            h = int(getattr(display_crop, "shape", [0, 0])[0] or 0)
            w = int(getattr(display_crop, "shape", [0, 0])[1] or 0)
            print(
                f"[Telegram] skip displayable gate @ cam {camera_location} "
                f"display={w}x{h} min_side={tg_min_side}"
            )
    except Exception as exc:
        print(f"[WARN] Telegram catch alert failed: {exc}")

    # Push a live SSE event for the dashboard "Live stream detections" panel.
    try:
        from live_detection_hub import publish_detection_event
        from identity import is_confident_named_match

        if person_label:
            sse_name = person_label
        else:
            sse_name = "Stranger"
        if matched_id is not None and match_distance < 900:
            confidence = float(confidence_score_from_distance(match_distance))
        else:
            confidence = 58.0
        # Gallery "100% match" on Stranger_* is self-match noise — cap display score.
        if known_id is not None:
            try:
                _dn = resolve_known_person_name(conn, known_id, person_id=db_track_id)
                if str(_dn or "").lower().startswith(("stranger_", "passerby_", "unknown")):
                    confidence = min(confidence, 72.0)
            except Exception:
                pass
        publish_detection_event(
            camera=camera_location,
            name=sse_name,
            confidence=confidence,
            face_bgr=display_crop,
            person_id=db_track_id,
            known_person_id=known_id,
            bbox=bbox,
            is_stranger=is_stranger,
            is_named=bool(person_label and not is_stranger),
            matched_named_id=(
                int(matched_id)
                if matched_id is not None and is_confident_named_match(match_distance)
                else None
            ),
        )
    except Exception as exc:
        print(f"[WARN] live detection publish failed: {exc}")
    return True


def _payload_flag(item: object, key: str) -> bool:
    """Safe bool for Telegram payload flags (numpy arrays must not be used as truth)."""
    if not isinstance(item, dict):
        return False
    value = item.get(key)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        try:
            return bool(value.reshape(-1)[0])
        except Exception:
            return False
    return bool(value)


def _payload_percent(item: object, default: int = 1) -> int:
    if not isinstance(item, dict):
        return default
    value = item.get("match_percent", default)
    try:
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0] if value.size else default
        return max(1, int(value))
    except (TypeError, ValueError, IndexError):
        return default


def _flush_telegram_catch_batch(
    camera_location: str,
    frame_bgr: Optional[np.ndarray],
    telegram_batch: List[dict],
) -> None:
    """Send one multi-catch Telegram when 2+ faces commit together."""
    if telegram_batch is None or isinstance(telegram_batch, np.ndarray):
        return
    try:
        items = list(telegram_batch)
    except TypeError:
        return
    if not items:
        return
    # Multi collage for enrolled watchlist / named matches; strangers one-by-one.
    enrolled = [
        item
        for item in items
        if isinstance(item, dict)
        and (
            not _payload_flag(item, "is_stranger")
            or _payload_flag(item, "is_named")
        )
        and not (
            _payload_flag(item, "is_named") and not TELEGRAM_ALERT_NAMED
        )
    ]
    strangers = [
        item
        for item in items
        if isinstance(item, dict)
        and _payload_flag(item, "is_stranger")
        and not _payload_flag(item, "is_named")
    ]
    try:
        if len(enrolled) >= 2:
            maybe_send_multi_face_alert_async(
                camera_zone=camera_location,
                frame_bgr=frame_bgr,
                faces=enrolled,
                camera_name=f"CAM-{camera_location}",
                location=str(camera_location),
                min_percent=None,
            )
        elif len(enrolled) == 1:
            item = enrolled[0]
            named_hit = _payload_flag(item, "is_named")
            maybe_send_target_alert_async(
                item["face_bgr"],
                int(item["person_id"]),
                camera_location,
                _payload_percent(item),
                frame_bgr=frame_bgr,
                person_name=item.get("person_name"),
                camera_name=f"CAM-{camera_location}",
                location=str(camera_location),
                bbox=item.get("bbox"),
                person_bbox=item.get("person_bbox"),
                min_percent=1 if named_hit else None,
                cooldown_sec=(
                    TELEGRAM_NON_WATCHLIST_COOLDOWN_SEC if named_hit else None
                ),
                alert_kind=item.get("alert_kind") or ("named" if named_hit else "watchlist"),
                image_quality=item.get("image_quality"),
            )
        for item in strangers:
            maybe_send_target_alert_async(
                item["face_bgr"],
                int(item["person_id"]),
                camera_location,
                _payload_percent(item),
                frame_bgr=frame_bgr,
                person_name=item.get("person_name"),
                camera_name=f"CAM-{camera_location}",
                location=str(camera_location),
                bbox=item.get("bbox"),
                person_bbox=item.get("person_bbox"),
                min_percent=1,
                cooldown_sec=TELEGRAM_NON_WATCHLIST_COOLDOWN_SEC,
                alert_kind=item.get("alert_kind", "incident"),
                image_quality=item.get("image_quality"),
            )
    except Exception as exc:
        print(f"[WARN] Telegram multi-catch flush failed: {exc}")
        import traceback
        traceback.print_exc()
        # Fall back to individual alerts so nobody is silently dropped.
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                is_stranger = _payload_flag(item, "is_stranger")
                maybe_send_target_alert_async(
                    item["face_bgr"],
                    int(item["person_id"]),
                    camera_location,
                    _payload_percent(item),
                    frame_bgr=frame_bgr,
                    person_name=item.get("person_name"),
                    camera_name=f"CAM-{camera_location}",
                    location=str(camera_location),
                    bbox=item.get("bbox"),
                    person_bbox=item.get("person_bbox"),
                    min_percent=1 if is_stranger else None,
                    cooldown_sec=TELEGRAM_NON_WATCHLIST_COOLDOWN_SEC if is_stranger else None,
                    alert_kind=item.get("alert_kind", "incident" if is_stranger else "watchlist"),
                )
            except Exception as inner:
                print(f"[WARN] Telegram fallback alert failed: {inner}")


def _crop_person_plate(
    frame_bgr: Optional[np.ndarray],
    bbox: Optional[Sequence[float]],
    *,
    pad_ratio: float = 0.10,
) -> Optional[np.ndarray]:
    """Tight padded crop of a YOLO person box for gate captures without a face embed."""
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0 or bbox is None:
        return None
    try:
        h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        x1, y1, x2, y2 = (float(v) for v in list(bbox)[:4])
    except (TypeError, ValueError, IndexError, AttributeError):
        return None
    if x2 <= x1 or y2 <= y1 or h < 2 or w < 2:
        return None
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad_ratio * bw))
    y1 = max(0, int(y1 - pad_ratio * bh))
    x2 = min(w, int(x2 + pad_ratio * bw))
    y2 = min(h, int(y2 + pad_ratio * bh))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    return np.ascontiguousarray(frame_bgr[y1:y2, x1:x2])


def _commit_person_catch_without_embed(
    conn: sqlite3.Connection,
    camera_location: str,
    track_id: Optional[int],
    visit_key: Tuple[str, object],
    head_crop: np.ndarray,
    last_logged: Dict[Tuple[str, object], float],
    cooldown: float,
    now: float,
    *,
    bbox: Optional[Sequence[float]] = None,
    frame_bgr: Optional[np.ndarray] = None,
    telegram_batch: Optional[List[dict]] = None,
) -> bool:
    """
    Save and Telegram a YOLO person even without a usable frontal face embed.

    Disabled when FACE_ONLY_CATCH=1 (user wants faces only — no back/body plates).
    """
    if FACE_ONLY_CATCH:
        print(f"[FACE] skip person-catch (faces-only) @ cam {camera_location}")
        return False
    if not _should_person_catch(camera_location):
        return False
    if now - last_logged.get(visit_key, 0.0) < cooldown:
        return False

    display_crop = _crop_person_plate(frame_bgr, bbox)
    if display_crop is None or not face_crop_is_photographic(display_crop):
        if head_crop is not None and getattr(head_crop, "size", 0) > 0:
            display_crop = np.ascontiguousarray(head_crop)
    if not _person_plate_ok_for_catch(display_crop, camera_location):
        return False

    db_track_id = (
        int(track_id)
        if track_id is not None
        else abs(hash(visit_key)) % 900_000 + 100_000
    )
    person_label = "Stranger"
    known_id = None
    try:
        feature = _embedding_from_sharpened_crop(display_crop, None)
        if feature is None:
            from identity import _embed_scan_reference_face, analyzer_ready

            if analyzer_ready():
                packed = _embed_scan_reference_face(display_crop)
                if packed is not None:
                    feature, _meta = packed
        if feature is not None:
            from identity import resolve_live_capture_identity

            label, is_stranger, matched_named_id, _dist = resolve_live_capture_identity(
                conn,
                feature,
                face_bgr=display_crop,
                camera_location=camera_location,
            )
            if not is_stranger and label:
                person_label = label
                known_id = matched_named_id
    except Exception:
        person_label = "Stranger"
        known_id = None
    capture_file = save_face_capture(
        display_crop,
        known_id,
        db_track_id,
        person_name=person_label,
    )
    if not capture_file:
        return False
    scene_file = None
    if frame_bgr is not None:
        scene_file = save_scene_capture(
            frame_bgr,
            known_id,
            db_track_id,
            face_bbox=bbox,
            face_filename=capture_file,
            person_name=person_label,
        )
        print(
            f"[CAPTURE] person saved {capture_file}"
            + (f" scene {scene_file}" if scene_file else "")
            + f" @ cam {camera_location}"
        )
    gender, age = _predict_capture_attributes(display_crop, head_crop)
    log_visit(
        conn,
        db_track_id,
        camera_location,
        known_person_id=known_id,
        capture_path=capture_file,
        gender=gender,
        age=age,
        scene_path=scene_file,
        display_name=person_label,
    )
    last_logged[visit_key] = now
    _FACE_DETECT_COUNTS[camera_location] = (
        _FACE_DETECT_COUNTS.get(camera_location, 0) + 1
    )
    write_faces_detected(camera_location, _FACE_DETECT_COUNTS[camera_location])

    named_hit = bool(known_id) and str(person_label or "") not in ("", "Stranger")
    payload = {
        "person_id": db_track_id,
        "person_name": person_label,
        "match_percent": 1,
        "face_bgr": np.ascontiguousarray(display_crop),
        "bbox": [float(v) for v in bbox[:4]] if bbox is not None and len(bbox) >= 4 else None,
        "person_bbox": (
            [float(v) for v in bbox[:4]] if bbox is not None and len(bbox) >= 4 else None
        ),
        "is_stranger": not named_hit,
        "is_named": named_hit,
        "alert_kind": "named" if named_hit else "incident",
    }
    send_named = bool(named_hit and TELEGRAM_ALERT_NAMED)
    send_incident = bool((not named_hit) and TELEGRAM_ALERT_NON_WATCHLIST)
    if send_named or send_incident:
        if telegram_batch is not None:
            telegram_batch.append(payload)
        else:
            maybe_send_target_alert_async(
                display_crop,
                db_track_id,
                camera_location,
                1,
                frame_bgr=frame_bgr,
                person_name=person_label,
                camera_name=f"CAM-{camera_location}",
                location=str(camera_location),
                bbox=bbox,
                person_bbox=bbox,
                min_percent=1,
                cooldown_sec=TELEGRAM_NON_WATCHLIST_COOLDOWN_SEC,
                alert_kind=payload["alert_kind"],
            )
    try:
        from live_detection_hub import publish_detection_event

        publish_detection_event(
            camera=camera_location,
            name=person_label,
            confidence=1.0,
            face_bgr=display_crop,
            person_id=db_track_id,
            known_person_id=known_id,
            bbox=bbox,
            is_stranger=person_label == "Stranger",
            is_named=bool(known_id) and person_label != "Stranger",
            matched_named_id=known_id,
        )
    except Exception:
        pass
    return True


def _finalize_best_track_frame(
    conn: sqlite3.Connection,
    camera_location: str,
    track_id: int,
    best: _BestTrackFrame,
    face_analyzer: Optional[FaceAnalyzer],
    last_logged: Dict[Tuple[str, object], float],
    track_to_known: Dict[str, Dict[int, int]],
    shared_gallery: Optional[SharedFaceGallery],
    cooldown: float,
    now: float,
    frame_bgr: Optional[np.ndarray] = None,
    telegram_batch: Optional[List[dict]] = None,
) -> bool:
    """Embed the sharpest cached crop (restoration + ArcFace) and log one trajectory."""
    visit_key = (camera_location, int(track_id))
    if now - last_logged.get(visit_key, 0.0) < cooldown:
        return False

    # Keep the natural-color square crop for saves / Telegram. CLAHE + GFPGAN
    # are embedding aids only — they look charcoal / painted when shown as the alert face.
    display_face = np.ascontiguousarray(best.face_crop.copy())
    # Skip tiny native heads (forced enlarge → flat cartoon look).
    if not face_crop_is_photographic(display_face):
        if _should_person_catch(camera_location):
            return _commit_person_catch_without_embed(
                conn,
                camera_location,
                track_id,
                visit_key,
                display_face,
                last_logged,
                cooldown,
                now,
                bbox=list(np.asarray(best.person_bbox, dtype=np.float32).reshape(-1)[:4])
                if getattr(best, "person_bbox", None) is not None
                else None,
                frame_bgr=best.scene_frame if best.scene_frame is not None else frame_bgr,
                telegram_batch=telegram_batch,
            )
        return False
    person_bbox: Optional[List[float]] = None
    try:
        pb = np.asarray(best.person_bbox, dtype=np.float32).reshape(-1)
        if pb.size >= 4:
            person_bbox = [float(v) for v in pb[:4]]
    except Exception:
        person_bbox = None

    feature: Optional[np.ndarray] = None
    sr = get_super_resolution_pipeline(initialize=True)
    if sr is not None and sr.ready and face_analyzer is not None and analyzer_ready():
        restored = sr.restore_and_embed_face_patch(
            display_face, face_analyzer=face_analyzer
        )
        if restored is not None:
            feature, _restored_bgr, _was = restored
        else:
            feature = face_analyzer.embed_crop(display_face, apply_clahe=True)
    elif face_analyzer is not None and analyzer_ready():
        feature = face_analyzer.embed_crop(display_face, apply_clahe=True)
    else:
        feature = _embedding_from_sharpened_crop(display_face, face_analyzer)

    if feature is None:
        if _should_person_catch(camera_location):
            plate = display_face
            if frame_bgr is not None and person_bbox is not None:
                plate = _crop_person_plate(frame_bgr, person_bbox) or display_face
            return _commit_person_catch_without_embed(
                conn,
                camera_location,
                track_id,
                visit_key,
                plate,
                last_logged,
                cooldown,
                now,
                bbox=person_bbox,
                frame_bgr=best.scene_frame if best.scene_frame is not None else frame_bgr,
                telegram_batch=telegram_batch,
            )
        # No ArcFace embed ⇒ skip indoor YOLO head/back crops.
        return False

    outdoor = is_outdoor_gate_camera(camera_location)
    if outdoor or CATCH_EVERY_PERSON:
        if not _face_crop_displayable(
            display_face,
            min_side=max(8, gate_display_min_side(camera_location, default_side=LIVE_FACE_DISPLAY_MIN_SIDE) - 4),
            min_sharpness=gate_min_face_laplacian(camera_location) * 0.35,
            min_skin_fraction=gate_skin_min_fraction(camera_location),
        ):
            plate = display_face
            if frame_bgr is not None and person_bbox is not None:
                plate = _crop_person_plate(frame_bgr, person_bbox) or display_face
            return _commit_person_catch_without_embed(
                conn,
                camera_location,
                track_id,
                visit_key,
                plate,
                last_logged,
                cooldown,
                now,
                bbox=person_bbox,
                frame_bgr=best.scene_frame if best.scene_frame is not None else frame_bgr,
                telegram_batch=telegram_batch,
            )
    elif not _face_crop_displayable(display_face):
        return False

    face_area = float(display_face.shape[0] * display_face.shape[1])
    face_box: Optional[List[float]] = None
    if best.face_bbox is not None:
        try:
            fb = np.asarray(best.face_bbox, dtype=np.float32).reshape(-1)
            if fb.size >= 4:
                face_box = [float(v) for v in fb[:4]]
        except Exception:
            face_box = None
    # Prefer the hallway frame frozen with the best face (person + box), not a
    # later empty frame after the track went idle.
    scene_bgr = best.scene_frame if best.scene_frame is not None else frame_bgr
    return _commit_face_candidate(
        conn,
        camera_location,
        track_id,
        visit_key,
        feature,
        display_face,
        face_area,
        last_logged,
        track_to_known,
        shared_gallery,
        cooldown,
        now,
        bbox=face_box,
        frame_bgr=scene_bgr,
        person_bbox=person_bbox,
        telegram_batch=telegram_batch,
    )


def _process_detections(
    conn: sqlite3.Connection,
    frame,
    result,
    camera_location: str,
    last_logged: Dict[Tuple[str, object], float],
    track_to_known: Dict[str, Dict[int, int]],
    cooldown: float,
    now: float,
    letterbox: Optional[LetterboxTransform] = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    face_analyzer: Optional[FaceAnalyzer] = None,
    face_dets: Optional[list] = None,
    shared_gallery: Optional[SharedFaceGallery] = None,
    face_track_gate: Optional[FaceTrackGate] = None,
    best_frame_selector: Optional[TrackBestFrameSelector] = None,
    dwell_tracker: Optional[TrackDwellTracker] = None,
    frame_index: int = 0,
) -> int:
    """
    Score tracked persons cheaply each frame; run DeepFace + DB matching only on
    the sharpest crop when a ByteTrack trajectory ends.
    """
    selector = best_frame_selector or TrackBestFrameSelector()
    dwell = dwell_tracker or TrackDwellTracker()
    session_now = datetime.now()
    source_np = _frame_as_bgr_numpy(frame)
    telegram_batch: List[dict] = []
    person_bboxes_for_faces: List[np.ndarray] = []

    def _run_face_direct_watchlist() -> int:
        return _process_standalone_face_watchlist(
            conn,
            camera_location,
            source_np,
            face_dets,
            last_logged,
            track_to_known,
            shared_gallery,
            cooldown,
            now,
            face_analyzer=face_analyzer,
            telegram_batch=telegram_batch,
            person_bboxes=person_bboxes_for_faces or None,
        )

    if result.boxes is None or len(result.boxes) == 0:
        log_count = 0
        _persist_closed_dwell_sessions(
            conn,
            camera_location,
            dwell.flush_inactive(camera_location, set(), seen_at=session_now),
            track_to_known,
        )
        for track_id, best in selector.flush_inactive(camera_location, set()):
            if _finalize_best_track_frame(
                conn,
                camera_location,
                track_id,
                best,
                face_analyzer,
                last_logged,
                track_to_known,
                shared_gallery,
                cooldown,
                now,
                frame_bgr=source_np,
                telegram_batch=telegram_batch,
            ):
                log_count += 1
        log_count += _run_face_direct_watchlist()
        _flush_telegram_catch_batch(camera_location, source_np, telegram_batch)
        return log_count
    if not models_ready() and not analyzer_ready():
        # Still record dwell even when face models are offline.
        active_track_ids: Set[int] = set()
        boxes_t = result.boxes.xyxy
        confs_t = result.boxes.conf
        track_ids_t = result.boxes.id
        for i in range(len(boxes_t)):
            conf = float(confs_t[i].item()) if confs_t is not None else 1.0
            if conf < track_conf_for_camera(camera_location):
                continue
            if track_ids_t is None:
                continue
            track_id = int(track_ids_t[i].item())
            active_track_ids.add(track_id)
            dwell.observe(camera_location, track_id, seen_at=session_now)
        _persist_closed_dwell_sessions(
            conn,
            camera_location,
            dwell.flush_inactive(
                camera_location, active_track_ids, seen_at=session_now
            ),
            track_to_known,
        )
        log_count = _run_face_direct_watchlist()
        return log_count

    boxes_t = result.boxes.xyxy
    if letterbox is not None and not letterbox.is_identity:
        boxes_t = _boxes_to_source_space(boxes_t, letterbox)
    elif scale_x != 1.0 or scale_y != 1.0:
        boxes_t = scale_xyxy_boxes(boxes_t, scale_x, scale_y)
    confs_t = result.boxes.conf
    track_ids_t = result.boxes.id

    gate = face_track_gate or FaceTrackGate()
    active_track_ids: Set[int] = set()
    immediate_candidates = []
    log_count = 0
    outdoor = is_outdoor_gate_camera(camera_location)
    min_track_conf = track_conf_for_camera(camera_location)
    min_person_h = (
        gate_min_person_height_px(camera_location)
        if outdoor
        else float(MIN_PERSON_HEIGHT_PX)
    )
    if CATCH_EVERY_PERSON and not outdoor:
        min_person_h = min(float(min_person_h), 24.0)
    early_commit_frames = 0 if outdoor else TRACK_BEST_EARLY_COMMIT_FRAMES

    # Pre-assign faces 1:1 so two people in-frame each keep their own face.
    eligible_indices: List[int] = []
    eligible_bboxes: List[np.ndarray] = []
    for i in range(len(boxes_t)):
        conf = float(confs_t[i].item()) if confs_t is not None else 1.0
        if conf < min_track_conf:
            continue
        height = float((boxes_t[i][3] - boxes_t[i][1]).item())
        if height < min_person_h:
            continue
        # Outdoor high-angle people can look squatter — skip indoor tile aspect kill.
        host = tensor_to_numpy(boxes_t[i])
        box = (int(host[0]), int(host[1]), int(host[2]), int(host[3]))
        if not _person_box_looks_human(
            box,
            frame_shape=getattr(source_np, "shape", None),
            outdoor=outdoor,
            frame_bgr=source_np,
        ):
            continue
        eligible_indices.append(i)
        eligible_bboxes.append(host)
    person_bboxes_for_faces = list(eligible_bboxes)
    face_dets = _supplement_face_dets_for_persons(
        source_np,
        eligible_bboxes,
        face_dets,
        face_analyzer,
        camera_location=camera_location,
    )
    face_by_person_idx = assign_unique_faces_to_persons(eligible_bboxes, face_dets or [])
    face_assignment_by_box_i = {
        eligible_indices[pi]: det for pi, det in face_by_person_idx.items()
    }

    for i in range(len(boxes_t)):
        bbox_t = boxes_t[i]
        conf = float(confs_t[i].item()) if confs_t is not None else 1.0
        if conf < min_track_conf:
            continue
        height = float((bbox_t[3] - bbox_t[1]).item())
        if height < min_person_h:
            continue

        track_id = (
            int(track_ids_t[i].item())
            if track_ids_t is not None
            else None
        )
        bbox_host = tensor_to_numpy(bbox_t)
        box = (
            int(bbox_host[0]),
            int(bbox_host[1]),
            int(bbox_host[2]),
            int(bbox_host[3]),
        )
        if not _person_box_looks_human(
            box,
            frame_shape=getattr(source_np, "shape", None),
            outdoor=outdoor,
            frame_bgr=source_np,
        ):
            continue
        key = _visit_key(camera_location, track_id, bbox_host)

        if track_id is not None:
            active_track_ids.add(track_id)
            dwell.observe(camera_location, track_id, seen_at=session_now)

        if now - last_logged.get(key, 0.0) < cooldown:
            continue

        if track_id is not None:
            allow_person_crop = (
                FACE_PERSON_CROP_STRIDE <= 1
                or (frame_index + int(track_id)) % FACE_PERSON_CROP_STRIDE == 0
                or i not in face_assignment_by_box_i
            )
            sharpness, face_crop, face_xyxy = _score_track_face_crop(
                source_np,
                bbox_t,
                face_dets,
                face_analyzer=face_analyzer,
                allow_person_crop=allow_person_crop,
                matched_face=face_assignment_by_box_i.get(i),
                camera_location=camera_location,
            )
            # Soft poses: person-plate catch when enabled — never when faces-only.
            if face_crop is not None and CATCH_EVERY_PERSON and not FACE_ONLY_CATCH:
                try:
                    from face_print_filter import (
                        is_likely_non_face_object,
                        is_likely_rear_view,
                    )

                    rear, rear_why = is_likely_rear_view(face_crop)
                    if rear and not CATCH_BACKS:
                        print(
                            f"[FACE] skip back track @ cam {camera_location}: "
                            f"{rear_why}"
                        )
                        continue
                    bad, bad_why = is_likely_non_face_object(face_crop)
                    if rear or (bad and _allow_pose_person_fallback(bad_why)):
                        pose_why = rear_why if rear else bad_why
                        if _commit_person_catch_without_embed(
                            conn,
                            camera_location,
                            track_id,
                            key,
                            face_crop,
                            last_logged,
                            cooldown,
                            now,
                            bbox=bbox_host,
                            frame_bgr=source_np,
                            telegram_batch=telegram_batch,
                        ):
                            log_count += 1
                        else:
                            print(
                                f"[FACE] person-catch failed @ cam {camera_location}: "
                                f"{pose_why or 'soft-pose'}"
                            )
                        continue
                except Exception:
                    pass
            elif face_crop is not None and FACE_ONLY_CATCH:
                try:
                    from face_print_filter import is_likely_rear_view

                    rear, rear_why = is_likely_rear_view(face_crop)
                    if rear and not CATCH_BACKS:
                        print(
                            f"[FACE] skip back track @ cam {camera_location}: "
                            f"{rear_why}"
                        )
                        continue
                except Exception:
                    pass
            if face_crop is None:
                face_data = _extract_face_for_person(
                    frame,
                    bbox_t,
                    face_analyzer,
                    face_dets,
                    source_frame=source_np,
                    matched_face=face_assignment_by_box_i.get(i),
                )
                if face_data is None:
                    if _should_person_catch(camera_location) and not FACE_ONLY_CATCH:
                        plate = _crop_person_plate(source_np, bbox_host)
                        if plate is not None and _commit_person_catch_without_embed(
                            conn,
                            camera_location,
                            track_id,
                            key,
                            plate,
                            last_logged,
                            cooldown,
                            now,
                            bbox=bbox_host,
                            frame_bgr=source_np,
                            telegram_batch=telegram_batch,
                        ):
                            log_count += 1
                    continue
                _feat, face_crop, _face_area, face_xyxy_raw = face_data
                face_xyxy = (
                    np.asarray(face_xyxy_raw, dtype=np.float32).reshape(-1)[:4]
                    if face_xyxy_raw is not None
                    else None
                )
                sharpness = _laplacian_on_bgr(face_crop)

            rotated = selector.observe(
                camera_location,
                track_id,
                frame_index,
                sharpness,
                face_crop,
                bbox_host,
                face_bbox=face_xyxy,
                scene_frame=source_np,
            )
            if rotated is not None:
                prev_id, prev_best = rotated
                if _finalize_best_track_frame(
                    conn,
                    camera_location,
                    prev_id,
                    prev_best,
                    face_analyzer,
                    last_logged,
                    track_to_known,
                    shared_gallery,
                    cooldown,
                    now,
                    frame_bgr=source_np,
                    telegram_batch=telegram_batch,
                ):
                    log_count += 1
            ready = selector.take_ready(
                camera_location,
                track_id,
                frame_index,
                min_frames=early_commit_frames,
            )
            if ready is not None:
                if _finalize_best_track_frame(
                    conn,
                    camera_location,
                    track_id,
                    ready,
                    face_analyzer,
                    last_logged,
                    track_to_known,
                    shared_gallery,
                    cooldown,
                    now,
                    frame_bgr=source_np,
                    telegram_batch=telegram_batch,
                ):
                    log_count += 1
            continue

        if not gate.should_recognize(
            camera_location, track_id, bbox_host, frame_index, key
        ):
            continue

        face_data = _extract_face_for_person(
            frame,
            bbox_t,
            face_analyzer,
            face_dets,
            source_frame=source_np,
            matched_face=face_assignment_by_box_i.get(i),
        )
        if face_data is None:
            continue

        feature, aligned_face, face_area, face_xyxy = face_data
        immediate_candidates.append(
            {
                "key": key,
                "track_id": track_id,
                "feature": feature,
                "aligned_face": aligned_face,
                "face_area": face_area,
                "bbox": face_xyxy
                or [float(v) for v in np.asarray(bbox_host).reshape(-1)[:4]],
            }
        )

    _persist_closed_dwell_sessions(
        conn,
        camera_location,
        dwell.flush_inactive(camera_location, active_track_ids, seen_at=session_now),
        track_to_known,
    )

    for track_id, best in selector.flush_inactive(camera_location, active_track_ids):
        if _finalize_best_track_frame(
            conn,
            camera_location,
            track_id,
            best,
            face_analyzer,
            last_logged,
            track_to_known,
            shared_gallery,
            cooldown,
            now,
            frame_bgr=source_np,
            telegram_batch=telegram_batch,
        ):
            log_count += 1

    # Standalone path has its own Telegram flush — keep body-track batch separate.
    log_count += _run_face_direct_watchlist()

    # Commit every person with a face in this frame (not only the largest).
    for cand in immediate_candidates:
        if _commit_face_candidate(
            conn,
            camera_location,
            cand["track_id"],
            cand["key"],
            cand["feature"],
            cand["aligned_face"],
            cand["face_area"],
            last_logged,
            track_to_known,
            shared_gallery,
            cooldown,
            now,
            bbox=cand.get("bbox"),
            frame_bgr=source_np,
            telegram_batch=telegram_batch,
        ):
            log_count += 1
    _flush_telegram_catch_batch(camera_location, source_np, telegram_batch)
    return log_count


def _print_face_pipeline_status() -> None:
    if analyzer_ready():
        print(f"Face identity pipeline: {active_pipeline_name()}")
    elif models_ready():
        print("Face identity pipeline: OpenCV YuNet + SFace (fallback)")
    else:
        print("Face models not ready — run: python download_face_models.py --no-proxy")


CAMERA_READER_STAGGER_SEC = 0.05


def _publish_multi_camera_previews(
    slot_frames: List[Tuple["CameraReader", Optional[np.ndarray], bool]],
    last_track_results: Dict[str, Tuple[object, LetterboxTransform]],
    attr_hud_caches: Dict[str, AttributeHUDCache],
    frame_index: int,
    grid_cols: int,
    last_good_tiles: Dict[int, np.ndarray],
    last_good_combined: Optional[np.ndarray],
    last_face_boxes: Optional[Dict[str, List[Tuple[int, int, int, int]]]] = None,
) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray], int]:
    """Fast path: push live JPEGs to the dashboard before heavy YOLO/face work."""
    face_boxes_map = last_face_boxes or {}
    tiles: List[np.ndarray] = []
    live_count = 0
    for reader, frame, ok in slot_frames:
        location = reader.location
        if not ok or frame is None:
            cached = last_good_tiles.get(reader.slot)
            if cached is not None:
                tiles.append(cached)
                live_count += 1
            else:
                tiles.append(_disconnected_tile(reader.slot, location))
            continue

        live_count += 1
        labeled = frame if frame.shape[0] <= CAMERA_PUBLISH_H + 8 and frame.shape[1] <= CAMERA_PUBLISH_W + 8 else frame.copy()
        hud_result = None
        hud_letterbox = None
        if location in last_track_results:
            hud_result, hud_letterbox = last_track_results[location]
        labeled = _draw_live_stream_detect_overlay(
            labeled,
            hud_result,
            letterbox=hud_letterbox,
            face_boxes=face_boxes_map.get(location),
            camera_location=location,
        )
        _annotate_camera_label(labeled, location)
        _publish_preview_frame(labeled, camera_key=location)
        preview_tile = _fit_cell(labeled)
        last_good_tiles[reader.slot] = preview_tile
        tiles.append(preview_tile)

    if live_count > 0 and tiles:
        combined = _build_grid(tiles, cols=grid_cols)
        last_good_combined = combined
        if len(slot_frames) > 1:
            _publish_preview_frame(combined)
    return last_good_tiles, last_good_combined, live_count


def _run_one_camera_ai_tick(
    job,
    *,
    conn,
    face_analyzer: Optional[FaceAnalyzer],
    face_skippers: Dict[str, AdaptiveFrameSkip],
    last_face_dets_by_location: Dict[str, list],
    last_face_boxes_by_location: Dict[str, List[Tuple[int, int, int, int]]],
    face_boxes_lock: threading.Lock,
    last_logged: Dict[Tuple[str, object], float],
    track_to_known: Dict[str, Dict[int, int]],
    shared_gallery: Optional[SharedFaceGallery],
    cooldown: float,
    face_track_gates: Dict[str, FaceTrackGate],
    best_frame_selectors: Dict[str, TrackBestFrameSelector],
    dwell_trackers: Dict[str, TrackDwellTracker],
) -> int:
    """One camera inference tick (face + capture) — runs off the preview thread."""
    location = job.location
    frame = job.frame_bgr
    frame_index = job.frame_index
    now = job.now

    if frame is None or not job.tracked or job.result is None:
        return 0

    skipper = face_skippers.setdefault(location, AdaptiveFrameSkip())
    face_dets = None
    if face_analyzer is not None:
        force_faces = False
        try:
            n_boxes = (
                len(job.result.boxes)
                if job.result is not None and getattr(job.result, "boxes", None) is not None
                else 0
            )
            force_faces = n_boxes >= 1 and (
                frame_index
                % live_face_force_every_for_camera(
                    location, default_every=LIVE_FACE_FORCE_EVERY
                )
                == 0
            )
            if n_boxes >= 2:
                force_faces = True
        except Exception:
            force_faces = False
        src_for_face = job.src_frame if job.src_frame is not None else frame
        fresh_dets = _run_face_pipeline(
            src_for_face,
            face_analyzer,
            skipper,
            frame_index,
            force=force_faces,
        )
        if fresh_dets is not None:
            last_face_dets_by_location[location] = fresh_dets
            face_dets = fresh_dets
        else:
            face_dets = last_face_dets_by_location.get(location)

    source_np = _frame_as_bgr_numpy(
        job.src_frame if job.src_frame is not None else frame
    )
    if face_analyzer is not None and job.result is not None:
        person_boxes = _collect_eligible_person_bboxes(
            job.result,
            letterbox=job.letterbox,
            source_shape=getattr(source_np, "shape", None),
            camera_location=location,
        )
        if face_dets is not None and person_boxes:
            face_dets = _supplement_face_dets_for_persons(
                source_np,
                person_boxes,
                face_dets,
                face_analyzer,
                camera_location=location,
            )
            last_face_dets_by_location[location] = face_dets

    if face_dets:
        with face_boxes_lock:
            last_face_boxes_by_location[location] = _snapshot_face_boxes(
                face_dets, source_np
            )

    src_frame = job.src_frame if job.src_frame is not None else frame
    track_gate = face_track_gates.setdefault(location, FaceTrackGate())
    best_selector = best_frame_selectors.setdefault(
        location, TrackBestFrameSelector()
    )
    dwell_selector = dwell_trackers.setdefault(location, TrackDwellTracker())
    return _process_detections(
        conn,
        src_frame,
        job.result,
        location,
        last_logged,
        track_to_known,
        cooldown,
        now,
        letterbox=job.letterbox,
        face_analyzer=face_analyzer,
        face_dets=face_dets,
        shared_gallery=shared_gallery,
        face_track_gate=track_gate,
        best_frame_selector=best_selector,
        dwell_tracker=dwell_selector,
        frame_index=frame_index,
    )


def run_multi_tracking(
    cameras: List[Tuple[str, Union[int, str]]],
    model_path: Path,
    db_path: Path,
    cooldown: float,
    show_window: bool,
    rtsp_tcp: bool = False,
    device: str = "auto",
    hybrid_face_on_cpu: bool = HYBRID_FACE_ON_CPU,
    gpu_unified: bool = False,
) -> None:
    resolved = audit_hardware_dependencies(
        device_choice=device,
        hybrid_face_on_cpu=hybrid_face_on_cpu,
        pure_gpu=gpu_unified,
    )
    init_gpu_face_stack()
    init_face_restoration_stack()
    conn = init_db(db_path)
    try:
        index_count = rebuild_face_index(conn)
    except Exception as exc:
        print(f"[WARN] FAISS face index build failed: {exc}")
        index_count = 0
    # Rebuild Named People embeds with EXIF + rotation variants (upright & sideways photos).
    try:
        from named_gallery import sync_named_encodings_from_gallery

        named_rows = conn.execute(
            """
            SELECT id, name FROM known_persons
            WHERE COALESCE(is_named, 0) = 1
              AND COALESCE(is_watchlist, 0) = 0
            """
        ).fetchall()
        rebuilt = 0
        for row in named_rows:
            pid = int(row[0] if not hasattr(row, "keys") else row["id"])
            n = int(sync_named_encodings_from_gallery(conn, pid) or 0)
            if n > 0:
                rebuilt += 1
                print(f"[NAMED] Rotation embeds refreshed for #{pid} ({n} variant(s))")
        if named_rows:
            print(f"[NAMED] Enrollment sync done for {rebuilt}/{len(named_rows)} people")
    except Exception as exc:
        print(f"[WARN] Named enrollment rotation sync skipped: {exc}")
    # Qdrant + in-memory cosine watchlist — hot-reloads when gallery changes.
    try:
        from watchlist_vector_db import (
            start_watchlist_sync_daemon,
            sync_watchlist_from_sqlite,
        )

        try:
            qdrant_n = sync_watchlist_from_sqlite(conn, notify=False)
            print(f"Watchlist memory index: {qdrant_n} embedding(s) (hot-reload on)")
        except Exception as exc:
            print(f"[WARN] Watchlist index unavailable: {exc}")
        # Always start the daemon so a later gallery fix can recover without restart.
        start_watchlist_sync_daemon(db_path=db_path)
        try:
            from telegram_alerts import TELEGRAM_ALERT_MIN_PERCENT, telegram_configured

            print(
                f"Telegram alerts: "
                f"{'ON' if telegram_configured() else 'OFF'} "
                f"(watchlist={'ON' if TELEGRAM_ALERT_WATCHLIST else 'OFF'}, "
                f"named={'ON' if TELEGRAM_ALERT_NAMED else 'OFF'}, "
                f"non-watchlist={'ON' if TELEGRAM_ALERT_NON_WATCHLIST else 'OFF'}, "
                f"incident cooldown {int(TELEGRAM_NON_WATCHLIST_COOLDOWN_SEC)}s, "
                f"min ≥{int(TELEGRAM_ALERT_MIN_PERCENT)}%)"
            )
        except Exception:
            pass
    except Exception as exc:
        print(f"[WARN] Watchlist sync bootstrap failed: {exc}")
    print(f"Loading model: {model_path}")
    print(f"Inference devices: {describe_device()} (resolved torch: {resolved})")
    if index_count:
        print(f"FAISS face index: {index_count} embedding(s) on GPU search path")
    model = _load_yolo_model(model_path)
    gpu_worker: Optional[GPUInferenceWorker] = None
    if gpu_unified:
        print("Unified GPU batch tracking enabled (single YOLO model)")
    if attribute_models_ready():
        try:
            preload_attribute_models()
            print("Age/gender attribute models preloaded.")
        except Exception as exc:
            print(f"[WARN] Age/gender models failed to preload: {exc}")
    last_logged: Dict[Tuple[str, object], float] = {}
    track_to_known: Dict[str, Dict[int, int]] = {}
    log_count = 0
    face_analyzer = _make_face_analyzer()
    shared_gallery: Optional[SharedFaceGallery] = None
    if face_analyzer is not None:
        shared_gallery = SharedFaceGallery(DEFAULT_GALLERY_DB)
    face_skippers: Dict[str, AdaptiveFrameSkip] = {}
    face_track_gates: Dict[str, FaceTrackGate] = {}
    best_frame_selectors: Dict[str, TrackBestFrameSelector] = {}
    dwell_trackers: Dict[str, TrackDwellTracker] = {}
    attr_hud_caches: Dict[str, AttributeHUDCache] = {}
    last_track_results: Dict[str, Tuple[object, LetterboxTransform]] = {}
    last_face_dets_by_location: Dict[str, list] = {}

    readers: List[CameraReader] = []
    for slot, (location, source) in enumerate(cameras):
        reader = CameraReader(
            slot,
            location,
            source,
            rtsp_tcp=rtsp_tcp or is_outdoor_gate_camera(location),
        )
        readers.append(reader)
        reader.start()
        time.sleep(CAMERA_READER_STAGGER_SEC)

    if not readers:
        conn.close()
        raise RuntimeError("No cameras configured.")

    grid_cols, cell_w, cell_h = _configure_preview_grid(len(readers))
    rows = max(1, (len(readers) + grid_cols - 1) // grid_cols)

    window_name = "AI Visitor Tracking"
    labels = ", ".join(r.location for r in readers)
    print(f"Tracking started — cameras: {labels}, cooldown: {cooldown}s")
    _print_face_pipeline_status()
    if models_ready() or analyzer_ready():
        print(
            "Capture mode: frontal + side-view faces only "
            "(backs of heads are ignored)"
        )
        if attribute_models_ready():
            backend = active_backend_name()
            print(
                f"Face attributes ON — backend: {backend}, "
                f"min confidence: {CONFIDENCE_THRESHOLD:.0%}, "
                f"face padding: 20%"
            )
            if backend == "opencv":
                print(
                    "  Tip: pip install deepface for stronger age/gender accuracy"
                )
        else:
            print("Age/gender models optional — run: python download_face_attributes.py --no-proxy")
            print("  Or: pip install deepface")
    else:
        print("Face models required for capture — install with:")
        print("  python download_face_models.py --no-proxy")
    print(
        f"Grid preview — {len(readers)} camera(s), "
        f"{grid_cols}x{rows} layout, {cell_w}x{cell_h} per tile."
    )
    face_skip_note = (
        f"face adaptive skip (target {AdaptiveFrameSkip().target_ms:.0f} ms)"
        if face_analyzer is not None
        else "face pipeline off"
    )
    vram_note = "VRAM zero-copy pipeline ON" if vram_pipeline_active() else "CPU frame pipeline"
    gpu_note = "unified GPU batch" if gpu_unified else "inline YOLO"
    print(
        f"Performance: inference {tracker_config.INFERENCE_WIDTH}x"
        f"{tracker_config.INFERENCE_HEIGHT} (display full-res), "
        f"YOLO every {FRAME_SKIP} frame(s), {face_skip_note}, {vram_note}, {gpu_note}"
    )
    print(
        f"Face quality gate: conf >= {LIVE_FACE_DETECT_CONFIDENCE:.2f}, "
        f"box >= {LIVE_FACE_MIN_BOX_PX}px, Laplacian >= {MIN_FACE_LAPLACIAN:.0f}, "
        f"yaw <= {LIVE_FACE_MAX_YAW:.2f} (frontal+side; no backs), "
        f"rescan every {FACE_RESCAN_FRAMES} frames or {FACE_SHIFT_RATIO:.0%} shift"
    )
    print(
        f"Best-frame matching: CLAHE pre-process, sharpest crop per track "
        f"(early commit {TRACK_BEST_EARLY_COMMIT_FRAMES} frames, "
        f"idle {TRACK_BEST_IDLE_FRAMES}, max collect {TRACK_BEST_MAX_COLLECT_FRAMES})"
    )
    print(
        f"Live HUD: person attribute cards (gender/age/clothing, refresh every "
        f"{ATTRIBUTE_HUD_REFRESH_FRAMES} frames per track)"
    )
    print("Live preview streams to the dashboard (no desktop window). Stop via run_system.py Ctrl+C.")

    frame_index = 0
    cam_ticks: Dict[str, int] = {reader.location: 0 for reader in readers}
    last_publish = 0.0
    last_good_tiles: Dict[int, np.ndarray] = {}
    last_good_combined: Optional[np.ndarray] = None
    last_face_boxes_by_location: Dict[str, List[Tuple[int, int, int, int]]] = {}
    memory_mgr = FrameMemoryManager(gc_interval=MEMORY_GC_INTERVAL_FRAMES)
    face_boxes_lock = threading.Lock()
    from face_process_worker import CameraAIJob, MultiCamFaceWorker

    def _face_worker_process(job: CameraAIJob) -> int:
        return _run_one_camera_ai_tick(
            job,
            conn=conn,
            face_analyzer=face_analyzer,
            face_skippers=face_skippers,
            last_face_dets_by_location=last_face_dets_by_location,
            last_face_boxes_by_location=last_face_boxes_by_location,
            face_boxes_lock=face_boxes_lock,
            last_logged=last_logged,
            track_to_known=track_to_known,
            shared_gallery=shared_gallery,
            cooldown=cooldown,
            face_track_gates=face_track_gates,
            best_frame_selectors=best_frame_selectors,
            dwell_trackers=dwell_trackers,
        )

    face_worker = MultiCamFaceWorker(_face_worker_process, max_queue=48)
    face_worker.start()
    gate_list = ", ".join(sorted(tracker_config.OUTDOOR_GATE_CAMERAS))
    print(
        f"Preview decoupled: lightweight MJPEG tiles ({CAMERA_PUBLISH_W}x{CAMERA_PUBLISH_H}) "
        f"+ background face worker (RetinaFace force every {LIVE_FACE_FORCE_EVERY} YOLO ticks)"
    )
    print(
        f"Gate tuning ({gate_list}): YOLO every {tracker_config.OUTDOOR_GATE_FRAME_SKIP} tick(s), "
        f"RTSP/TCP on, relaxed rain/angle face gates"
    )
    try:
        while True:
            frame_index += 1
            for reader in readers:
                cam_ticks[reader.location] = cam_ticks.get(reader.location, 0) + 1
            run_inference = any(
                _camera_inference_due(
                    reader.location, cam_ticks[reader.location], frame_index
                )
                for reader in readers
            )
            now_publish = time.time()
            publish_due = now_publish - last_publish >= PREVIEW_PUBLISH_INTERVAL_SEC

            # Phase 1 — fast preview (small downscaled copies, never blocked by face AI).
            if publish_due:
                preview_slots: List[Tuple[CameraReader, Optional[np.ndarray], bool]] = []
                for reader in readers:
                    pf, ok = reader.get_preview_frame(
                        CAMERA_PUBLISH_W, CAMERA_PUBLISH_H
                    )
                    preview_slots.append((reader, pf, ok))
                last_good_tiles, last_good_combined, _live = _publish_multi_camera_previews(
                    preview_slots,
                    last_track_results,
                    attr_hud_caches,
                    frame_index,
                    grid_cols,
                    last_good_tiles,
                    last_good_combined,
                    last_face_boxes_by_location,
                )
                for _r, pf, _ok in preview_slots:
                    if pf is not None:
                        del pf
                last_publish = now_publish

            if not run_inference:
                memory_mgr.tick(frame_index)
                time.sleep(MAIN_LOOP_SLEEP_SEC)
                continue

            # Phase 2 — YOLO batch (all cams on global tick; gate cams also on gate ticks).
            slot_frames: List[Tuple[CameraReader, Optional[np.ndarray], bool]] = []
            for reader in readers:
                if not _camera_inference_due(
                    reader.location, cam_ticks[reader.location], frame_index
                ):
                    continue
                frame, ok = reader.get_frame(copy=True)
                slot_frames.append((reader, frame, ok))

            if not slot_frames:
                memory_mgr.tick(frame_index)
                time.sleep(MAIN_LOOP_SLEEP_SEC)
                continue

            tracked: Dict[str, Tuple[object, object, LetterboxTransform]] = {}
            if gpu_unified:
                tracked = _track_via_gpu_worker(model, slot_frames, frame_index)
            elif gpu_worker is not None:
                tracked = _track_via_gpu_worker_async(
                    gpu_worker, slot_frames, frame_index
                )
            else:
                tracked = _track_all_cameras(model, slot_frames)
            for loc, (result, _src, letterbox) in tracked.items():
                last_track_results[loc] = (result, letterbox)

            now = time.time()

            # Phase 3 — queue face/capture work (non-blocking for preview loop).
            for reader, frame, ok in slot_frames:
                if not ok or frame is None:
                    continue
                location = reader.location
                if location not in tracked:
                    continue
                result, src_frame, letterbox = tracked[location]
                face_worker.submit(
                    CameraAIJob(
                        frame_index=frame_index,
                        location=location,
                        frame_bgr=frame,
                        now=now,
                        tracked=True,
                        result=result,
                        src_frame=src_frame,
                        letterbox=letterbox,
                    )
                )

            for _reader, cam_frame, _ok in slot_frames:
                if cam_frame is not None:
                    del cam_frame
            slot_frames.clear()
            tracked.clear()
            memory_mgr.tick(frame_index)

            time.sleep(MAIN_LOOP_SLEEP_SEC)
            # Desktop OpenCV window disabled — frames go to dashboard /api/live_cctv_stream only.
            # if show_window:
            #     cv2.imshow(window_name, combined)
            #     if cv2.waitKey(1) & 0xFF == ord("q"):
            #         break
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
    finally:
        face_worker.stop()
        log_count += face_worker.log_count
        if gpu_worker is not None:
            gpu_worker.stop()
        for reader in readers:
            reader.stop()
        for reader in readers:
            reader.join(timeout=2.0)
        if shared_gallery is not None:
            shared_gallery.close()
        registered = sum(len(m) for m in track_to_known.values())
        print(f"Tracking stopped. {log_count} visit(s) logged, {registered} person(s) registered.")
        if log_count == 0:
            print("No visits yet — wait until you see [LOG] lines before quitting.")
            print("  First detection can take 10–20 seconds after the stream opens.")
        conn.close()


def run_tracking(
    source: Union[int, str],
    camera_location: str,
    model_path: Path,
    db_path: Path,
    cooldown: float,
    show_window: bool,
    rtsp_tcp: bool = False,
    device: str = "auto",
    hybrid_face_on_cpu: bool = HYBRID_FACE_ON_CPU,
) -> None:
    resolved = audit_hardware_dependencies(
        device_choice=device,
        hybrid_face_on_cpu=hybrid_face_on_cpu,
    )
    init_gpu_face_stack()
    init_face_restoration_stack()
    conn = init_db(db_path)
    try:
        from watchlist_vector_db import start_watchlist_sync_daemon, sync_watchlist_from_sqlite

        try:
            n = sync_watchlist_from_sqlite(conn, notify=False)
            print(f"Watchlist memory index: {n} embedding(s) (hot-reload on)")
        except Exception as exc:
            print(f"[WARN] Watchlist index unavailable: {exc}")
        # Always start the daemon so a later gallery fix can recover without restart.
        start_watchlist_sync_daemon(db_path=db_path)
        try:
            from telegram_alerts import TELEGRAM_ALERT_MIN_PERCENT, telegram_configured

            print(
                f"Telegram alerts: "
                f"{'ON' if telegram_configured() else 'OFF'} "
                f"(watchlist={'ON' if TELEGRAM_ALERT_WATCHLIST else 'OFF'}, "
                f"named={'ON' if TELEGRAM_ALERT_NAMED else 'OFF'}, "
                f"non-watchlist={'ON' if TELEGRAM_ALERT_NON_WATCHLIST else 'OFF'}, "
                f"incident cooldown {int(TELEGRAM_NON_WATCHLIST_COOLDOWN_SEC)}s, "
                f"min ≥{int(TELEGRAM_ALERT_MIN_PERCENT)}%)"
            )
        except Exception:
            pass
    except Exception as exc:
        print(f"[WARN] Watchlist sync bootstrap failed: {exc}")
    print(f"Loading model: {model_path}")
    print(f"Inference devices: {describe_device()} (resolved torch: {resolved})")
    model = _load_yolo_model(model_path)
    if attribute_models_ready():
        try:
            preload_attribute_models()
            print("Age/gender attribute models preloaded.")
        except Exception as exc:
            print(f"[WARN] Age/gender models failed to preload: {exc}")
    last_logged: Dict[Tuple[str, object], float] = {}
    track_to_known: Dict[str, Dict[int, int]] = {}
    log_count = 0
    face_analyzer = _make_face_analyzer()
    shared_gallery: Optional[SharedFaceGallery] = None
    if face_analyzer is not None:
        shared_gallery = SharedFaceGallery(DEFAULT_GALLERY_DB)
    face_skipper = AdaptiveFrameSkip()
    face_track_gate = FaceTrackGate()
    best_frame_selector = TrackBestFrameSelector()
    dwell_tracker = TrackDwellTracker()
    attr_hud_cache = AttributeHUDCache(refresh_frames=ATTRIBUTE_HUD_REFRESH_FRAMES)
    last_track_result: Optional[Tuple[object, LetterboxTransform]] = None

    # Dedicated decode thread (buffer=1) — same path as multi-camera for RTSP latency.
    reader = CameraReader(
        0,
        camera_location,
        source,
        rtsp_tcp=rtsp_tcp or is_outdoor_gate_camera(camera_location),
    )
    reader.start()

    window_name = "AI Visitor Tracking"
    print(f"Tracking started — camera: {camera_location}, cooldown: {cooldown}s")
    _print_face_pipeline_status()
    if models_ready() or analyzer_ready():
        print(
            "Capture mode: frontal + side-view faces only "
            "(backs of heads are ignored)"
        )
        if attribute_models_ready():
            backend = active_backend_name()
            print(
                f"Face attributes ON — backend: {backend}, "
                f"min confidence: {CONFIDENCE_THRESHOLD:.0%}, "
                f"face padding: 20%"
            )
            if backend == "opencv":
                print(
                    "  Tip: pip install deepface for stronger age/gender accuracy"
                )
        else:
            print("Age/gender models optional — run: python download_face_attributes.py --no-proxy")
            print("  Or: pip install deepface")
    else:
        print("Face models required for capture — install with:")
        print("  python download_face_models.py --no-proxy")
    print("Stand in view (or play a video with people).")
    face_skip_note = (
        f"face adaptive skip (target {face_skipper.target_ms:.0f} ms)"
        if face_analyzer is not None
        else "face pipeline off"
    )
    vram_note = "VRAM zero-copy pipeline ON" if vram_pipeline_active() else "CPU frame pipeline"
    print(
        f"Performance: inference {tracker_config.INFERENCE_WIDTH}x"
        f"{tracker_config.INFERENCE_HEIGHT} (display full-res), "
        f"YOLO every {FRAME_SKIP} frame(s), {face_skip_note}, {vram_note}"
    )
    print(
        f"Best-frame matching: CLAHE pre-process, sharpest crop per track "
        f"(early commit {TRACK_BEST_EARLY_COMMIT_FRAMES} frames, "
        f"idle {TRACK_BEST_IDLE_FRAMES}, max collect {TRACK_BEST_MAX_COLLECT_FRAMES})"
    )
    print(
        f"Live HUD: person attribute cards (gender/age/clothing, refresh every "
        f"{ATTRIBUTE_HUD_REFRESH_FRAMES} frames per track)"
    )
    print("Live preview streams to the dashboard (no desktop window). Stop via run_system.py Ctrl+C.")

    frame_index = 0
    last_publish = 0.0
    last_face_boxes: List[Tuple[int, int, int, int]] = []
    last_face_dets: Optional[list] = None
    memory_mgr = FrameMemoryManager(gc_interval=MEMORY_GC_INTERVAL_FRAMES)
    try:
        while True:
            frame_index += 1
            now_publish = time.time()
            publish_due = now_publish - last_publish >= PREVIEW_PUBLISH_INTERVAL_SEC
            run_inference = _should_run_inference(frame_index - 1, camera_location)
            need_frame = run_inference or publish_due

            if need_frame:
                frame, success = reader.get_frame(copy=True)
            else:
                frame, success = None, reader.is_live

            if need_frame and (not success or frame is None):
                time.sleep(MAIN_LOOP_SLEEP_SEC)
                memory_mgr.tick(frame_index)
                continue

            if not run_inference:
                if publish_due and frame is not None:
                    preview = frame.copy()
                    hud_result = None
                    hud_letterbox = None
                    if last_track_result is not None:
                        hud_result, hud_letterbox = last_track_result
                    preview = _draw_live_stream_detect_overlay(
                        preview,
                        hud_result,
                        letterbox=hud_letterbox,
                        face_boxes=last_face_boxes,
                        camera_location=camera_location,
                    )
                    _annotate_camera_label(preview, camera_location)
                    _publish_preview_frame(preview, camera_key=camera_location)
                    _publish_live_face_boxes(camera_location, frame, last_face_boxes)
                    last_publish = now_publish
                    del preview
                if frame is not None:
                    del frame
                memory_mgr.tick(frame_index)
                time.sleep(MAIN_LOOP_SLEEP_SEC)
                continue

            assert frame is not None

            def _run_tracking_inference() -> Tuple[object, object, object]:
                if vram_pipeline_active():
                    frame_gpu = bgr_numpy_to_gpu(frame)
                    scaled_gpu, letterbox = resize_hwc(
                        frame_gpu,
                        tracker_config.INFERENCE_WIDTH,
                        tracker_config.INFERENCE_HEIGHT,
                    )
                    results = model.track(
                        yolo_track_source(scaled_gpu), **_yolo_track_kwargs()
                    )
                    return results, frame_gpu, letterbox
                scaled, letterbox = _resize_for_inference(frame)
                results = model.track(scaled, **_yolo_track_kwargs())
                return results, frame, letterbox

            # မြန်မာ: YOLO + face inference ကို empty detection crash မဖြစ်စေရန် wrap လုပ်သည်
            inference_out = SafeInferenceRunner.run(
                _run_tracking_inference,
                context=f"{camera_location} tracking frame {frame_index}",
            )
            if inference_out is None:
                del frame
                memory_mgr.tick(frame_index)
                time.sleep(MAIN_LOOP_SLEEP_SEC)
                continue

            results, src_frame, letterbox = inference_out
            result = wrap_yolo_results(results)
            if result is None:
                del frame
                del src_frame
                del results
                memory_mgr.tick(frame_index)
                time.sleep(MAIN_LOOP_SLEEP_SEC)
                continue

            last_track_result = (result, letterbox)
            now = time.time()

            face_dets = None
            if face_analyzer is not None:
                force_faces = False
                try:
                    n_boxes = (
                        len(result.boxes)
                        if result is not None and getattr(result, "boxes", None) is not None
                        else 0
                    )
                    force_faces = n_boxes >= 1
                    if n_boxes >= 2:
                        force_faces = True
                except Exception:
                    force_faces = False
                fresh_dets = _run_face_pipeline(
                    src_frame,
                    face_analyzer,
                    face_skipper,
                    frame_index,
                    force=force_faces,
                )
                if fresh_dets is not None:
                    last_face_dets = fresh_dets
                    face_dets = fresh_dets
                else:
                    face_dets = last_face_dets
                src_np_overlay = _frame_as_bgr_numpy(src_frame)
                person_boxes = _collect_eligible_person_bboxes(
                    result,
                    letterbox=letterbox,
                    source_shape=getattr(src_np_overlay, "shape", None),
                    camera_location=camera_location,
                )
                if face_dets is not None and person_boxes and face_analyzer is not None:
                    face_dets = _supplement_face_dets_for_persons(
                        src_np_overlay,
                        person_boxes,
                        face_dets,
                        face_analyzer,
                        camera_location=camera_location,
                    )
                    last_face_dets = face_dets
                # Standalone watchlist runs inside `_process_detections` so multi-person
                # catches share one Telegram collage with body-track commits.
            if face_dets:
                last_face_boxes = _snapshot_face_boxes(
                    face_dets, _frame_as_bgr_numpy(src_frame)
                )
                _publish_live_face_boxes(camera_location, frame, last_face_boxes)
            log_count += _process_detections(
                conn,
                src_frame,
                result,
                camera_location,
                last_logged,
                track_to_known,
                cooldown,
                now,
                letterbox=letterbox,
                face_analyzer=face_analyzer,
                face_dets=face_dets,
                shared_gallery=shared_gallery,
                face_track_gate=face_track_gate,
                best_frame_selector=best_frame_selector,
                dwell_tracker=dwell_tracker,
                frame_index=frame_index,
            )

            if publish_due:
                preview = frame.copy()
                hud_result = None
                hud_letterbox = None
                if last_track_result is not None:
                    hud_result, hud_letterbox = last_track_result
                preview = _draw_live_stream_detect_overlay(
                    preview,
                    hud_result,
                    letterbox=hud_letterbox,
                    face_boxes=last_face_boxes,
                    camera_location=camera_location,
                )
                _annotate_camera_label(preview, camera_location)
                _publish_preview_frame(preview, camera_key=camera_location)
                _publish_live_face_boxes(camera_location, frame, last_face_boxes)
                last_publish = now_publish
                del preview

            del frame
            del src_frame
            del results
            del face_dets
            memory_mgr.tick(frame_index)
            time.sleep(MAIN_LOOP_SLEEP_SEC)
            # Desktop OpenCV window disabled — frames go to dashboard /api/live_cctv_stream only.
            # if show_window:
            #     cv2.imshow(window_name, annotated)
            #     if cv2.waitKey(1) & 0xFF == ord("q"):
            #         break
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
    finally:
        reader.stop()
        reader.join(timeout=2.0)
        if shared_gallery is not None:
            shared_gallery.close()
        registered = sum(len(m) for m in track_to_known.values())
        print(f"Tracking stopped. {log_count} visit(s) logged, {registered} person(s) registered.")
        if log_count == 0:
            print("No visits yet — wait until you see [LOG] lines before quitting.")
            print("  First detection can take 10–20 seconds after the stream opens.")
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track people in a video stream and log visits to SQLite."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (0), video file, or RTSP/HTTP stream URL",
    )
    parser.add_argument(
        "--camera-location",
        default="main_entrance",
        help="Label for this camera (stored in tracking_logs)",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to yolov8l.pt (run: py -3.12 download_model.py --no-proxy --size l)",
    )
    parser.add_argument(
        "--db",
        default=str(DB_PATH),
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=COOLDOWN_SECONDS,
        help="Seconds between log entries for the same person_id",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening a preview window",
    )
    parser.add_argument(
        "--rtsp-tcp",
        action="store_true",
        help="Use RTSP over TCP (more reliable on some networks/cameras)",
    )
    parser.add_argument(
        "--camera",
        action="append",
        nargs=2,
        metavar=("LOCATION", "SOURCE"),
        dest="cameras",
        help="Add a camera (repeatable): --camera main_entrance rtsp://...",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "cuda:0"],
        default="auto",
        help="Inference device (cuda is shorthand for cuda:0; auto picks CUDA when available)",
    )
    parser.add_argument(
        "--gpu-unified",
        action="store_true",
        help="Pure GPU mode: centralized YOLO worker, VRAM pipeline, CUDA-only ONNX, FAISS GPU",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=FRAME_SKIP,
        metavar="N",
        help="Run YOLO every N frames (e.g. 3 or 5); preview stays smooth (default: %(default)s)",
    )
    parser.add_argument(
        "--inference-size",
        default=f"{tracker_config.INFERENCE_WIDTH}x{tracker_config.INFERENCE_HEIGHT}",
        metavar="WxH",
        help="AI letterbox size only (display stays full-res). Example: 640x360",
    )
    args = parser.parse_args()

    try:
        inf_w, inf_h = parse_inference_size(args.inference_size)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    eff_w, eff_h = configure_inference_size(inf_w, inf_h)
    if (eff_w, eff_h) != (inf_w, inf_h):
        print(
            f"[INFO] Inference size aligned to YOLO stride "
            f"{tracker_config.YOLO_STRIDE}: {inf_w}x{inf_h} → {eff_w}x{eff_h}"
        )
    configure_frame_skip(args.frame_skip)

    model_path = resolve_model_path(args.model)
    common = dict(
        model_path=model_path,
        db_path=Path(args.db),
        cooldown=args.cooldown,
        show_window=False,
        rtsp_tcp=args.rtsp_tcp,
        device=args.device,
        hybrid_face_on_cpu=HYBRID_FACE_ON_CPU,
        gpu_unified=args.gpu_unified,
    )

    if args.cameras:
        camera_list = [
            (location, parse_source(source)) for location, source in args.cameras
        ]
        run_multi_tracking(cameras=camera_list, **common)
    else:
        # run_tracking does not take gpu_unified
        common.pop("gpu_unified", None)
        run_tracking(
            source=parse_source(args.source),
            camera_location=args.camera_location,
            **common,
        )


if __name__ == "__main__":
    main()
