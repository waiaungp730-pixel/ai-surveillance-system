"""
Web dashboard for AI Visitor Tracking.

  python dashboard.py
  Open http://127.0.0.1:5000
"""

import numpy_compat  # noqa: F401  — NumPy 1.x / 2.x import bridge (must be first)

import argparse
import base64
import gc
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
import urllib.error
import urllib.request
from collections import Counter, deque
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.datastructures import FileStorage

try:
    from deepface import DeepFace

    _DEEPFACE_AVAILABLE = True
except ImportError:
    DeepFace = None  # type: ignore[misc, assignment]
    _DEEPFACE_AVAILABLE = False

from captures import CAPTURES_DIR
from dashboard_data import (
    DEFAULT_DB,
    clear_device_detection_events,
    fetch_by_location,
    fetch_captures,
    fetch_device_detection_logs,
    fetch_live_infraction_feed,
    fetch_recent_logs,
    fetch_summary,
    fetch_timeline,
    fetch_visitors,
    fetch_profile_images_by_person,
    get_connection,
    insert_device_detection_event,
    load_cameras_config,
    _face_capture_file_ok,
)
from dashboard_face_data import (
    fetch_cross_camera_journeys,
    fetch_extended_metrics,
    fetch_face_intel_bundle,
    fetch_live_face_gallery,
)
from face_stats import read_face_stats
from identity import (
    VIDEO_SEARCH_EXTENSIONS,
    VIDEO_TARGET_CONFIRM_MATCH_PERCENT,
    MATCH_THRESHOLD,
    add_encoding_for_person,
    find_best_face_match,
    init_identity_db,
    lookup_photo,
    lookup_video,
    normalize_batch_folder_path,
    normalize_scan_response,
    collect_videos_in_folder,
    register_person,
    scan_folder_for_target,
    scan_video_for_target,
)
from dashboard_metrics import build_scan_hold_tactical_metrics, fetch_tactical_metrics
from live_frame_hub import (
    get_camera_jpeg,
    get_latest_jpeg,
    is_feed_available,
    list_camera_jpegs,
)

PROJECT_DIR = Path(__file__).parent
ENV_PATH = PROJECT_DIR / ".env"


def _load_dotenv(path: Path = ENV_PATH) -> None:
    """Load .env into os.environ without overwriting existing variables."""
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv()
CAMERAS_CONFIG_PATH = PROJECT_DIR / "cameras.json"
DASHBOARD_UI_VERSION = "27.3"
METRICS_WS_NAMESPACE = "/ws/dashboard_metrics"
LIVE_VIDEO_WS_NAMESPACE = "/ws/live_video"
ALERTS_WS_NAMESPACE = "/ws/alerts"
WEB_STREAM_SIZE: Tuple[int, int] = (800, 450)  # 16:9 widescreen — fills dashboard card edge-to-edge
MJPEG_JPEG_QUALITY = 80
LIVE_VIDEO_FPS = 20.0
LIVE_VIDEO_LOW_SIZE: Tuple[int, int] = (480, 270)
LIVE_VIDEO_LOW_JPEG_QUALITY = 52

_fallback_capture: Optional[cv2.VideoCapture] = None
_fallback_source: Optional[str] = None
UPLOAD_DIR = PROJECT_DIR / "uploads"
SCAN_CACHE_DIR = UPLOAD_DIR / "scan_cache"
TARGET_UPLOAD_DIR = PROJECT_DIR / "static" / "uploads"
TARGET_UPLOAD_PATH = TARGET_UPLOAD_DIR / "target.jpg"
app = Flask(
    __name__,
    template_folder=str(PROJECT_DIR / "templates"),
    static_folder=str(PROJECT_DIR / "static"),
    static_url_path="/static",
)
from security_bootstrap import (
    assert_scan_path_allowed,
    passwords_match,
    redact_media_url,
    resolve_dashboard_secrets,
)

_FLASK_SECRET, ADMIN_USERNAME, ADMIN_PASSWORD, INTERNAL_ALERT_TOKEN = (
    resolve_dashboard_secrets()
)
app.secret_key = _FLASK_SECRET
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=12)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Set SESSION_COOKIE_SECURE=1 behind HTTPS.
app.config["SESSION_COOKIE_SECURE"] = os.environ.get(
    "SESSION_COOKIE_SECURE", ""
).strip().lower() in ("1", "true", "yes", "on")
app.config["TEMPLATES_AUTO_RELOAD"] = True
_cors = (os.environ.get("SOCKETIO_CORS_ORIGINS") or "").strip()
socketio = SocketIO(
    app,
    cors_allowed_origins=_cors.split(",") if _cors else ["http://127.0.0.1:5000", "http://localhost:5000"],
    async_mode="threading",
)

_AUTH_EXEMPT_ENDPOINTS = frozenset(
    {
        "login",
        "logout",
        "static",
        "internal_database_purged",
        "internal_person_alert",
    }
)


def is_logged_in() -> bool:
    return session.get("logged_in") is True


def dashboard_home_url() -> str:
    """Post-login landing URL — opens the dashboard panel in the SPA."""
    return f"{url_for('index')}#view-dashboard"


def login_required(view):
    """Decorator — redirect HTML clients to /login; return 401 for API routes."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if is_logged_in():
            return view(*args, **kwargs)
        if request.path.startswith("/api/") or (
            request.accept_mimetypes.best_match(["application/json", "text/html"])
            == "application/json"
            and request.method != "GET"
        ):
            return jsonify({"error": "Authentication required"}), 401
        return redirect(url_for("login", next=request.path))

    return wrapped


# Login page needs CSS/JS only. Captures / live_preview / crops require a session.
_PUBLIC_ASSET_PREFIXES = (
    "/static/css/",
    "/static/js/",
    "/static/fonts/",
    "/static/img/",
    "/static/images/",
    "/static/vendor/",
    "/static/dashboard",
)


@app.before_request
def enforce_dashboard_auth():
    """Session gate — block unauthorized access to pages, APIs, and assets."""
    path = request.path or ""
    # Allow bare /static/* for login chrome, but not /static/crops or uploads.
    if path.startswith("/static/"):
        if path.startswith("/static/crops/") or path.startswith("/static/uploads/"):
            if not is_logged_in():
                return jsonify({"error": "Authentication required"}), 401
            return None
        return None
    if any(path.startswith(prefix) for prefix in _PUBLIC_ASSET_PREFIXES):
        return None
    endpoint = request.endpoint or ""
    if endpoint in _AUTH_EXEMPT_ENDPOINTS:
        return None
    if endpoint.startswith("static"):
        return None
    if is_logged_in():
        return None
    if path.startswith("/api/") or path.startswith("/internal/"):
        return jsonify({"error": "Authentication required"}), 401
    if path.startswith(("/captures/", "/plate_captures/", "/live_preview/", "/stills/", "/runs/")):
        return jsonify({"error": "Authentication required"}), 401
    return redirect(url_for("login", next=request.path))

# Per-section dashboard colors (injected into dashboard.html).
PANEL_THEMES = {
    "overview": {
        "bg": "#0a1020",
        "border": "#00e5ff",
        "accent": "#00e5ff",
        "text": "#e8f4ff",
        "muted": "#6b7fa8",
    },
    "search": {
        "bg": "#090e1c",
        "border": "#00e5ff",
        "accent": "#00ff9f",
        "text": "#e8f4ff",
        "muted": "#6b7fa8",
    },
    "captures": {
        "bg": "#090e1c",
        "border": "#c026d3",
        "accent": "#c026d3",
        "text": "#e8f4ff",
        "muted": "#6b7fa8",
    },
    "analytics": {
        "bg": "#0a1020",
        "border": "#00e5ff",
        "accent": "#00e5ff",
        "text": "#e8f4ff",
        "muted": "#6b7fa8",
    },
}

_yolo_model = None
_inference_initialized = False
_scan_progress: Dict[str, dict] = {}
_scan_progress_lock = threading.Lock()
_active_scan_ids: Set[str] = set()
_device_scan_results: Dict[str, dict] = {}
_device_scan_results_lock = threading.Lock()
_active_device_probe_scans: Set[str] = set()
_target_scan_results: Dict[str, dict] = {}
_target_scan_results_lock = threading.Lock()
_active_target_scans: Set[str] = set()
_layout_hold_lock = threading.Lock()
_layout_hold_active = False
_scan_job_lock = threading.Lock()
_scan_job_state: Dict[str, object] = {
    "active": False,
    "mode": "live",
    "videos_total": 0,
    "videos_processed": 0,
    "duration": "—",
    "avg_match": "—",
    "scan_id": None,
}
_scan_processing_timer_lock = threading.Lock()
_scan_processing_started: Dict[str, float] = {}
_scan_processing_locked: Dict[str, str] = {}
_scan_processing_locked_sec: Dict[str, float] = {}
_scan_match_scores_lock = threading.Lock()
_scan_match_scores: Dict[str, List[float]] = {}
_scan_match_locked: Dict[str, str] = {}
_detection_counters_lock = threading.Lock()
_detections_today_count = 0
_total_matches_count = 0
_incidents_reported_count = 0
_detection_counters_date: Optional[str] = None
_detection_counters_bootstrapped = False
_scan_counted_match_keys_lock = threading.Lock()
_scan_counted_match_keys: Dict[str, Set[Tuple[int, str]]] = {}
_session_videos_lock = threading.Lock()
_session_videos_processed_count = 0
_scan_ground_truth_lock = threading.Lock()
_scan_ground_truth_by_id: Dict[str, Dict[str, object]] = {}
TARGET_DEMOGRAPHICS_SMOOTHING_WINDOW = 7
_scan_demo_windows_lock = threading.Lock()
_scan_demo_windows: Dict[str, Dict[str, deque]] = {}
SCAN_DISPLAY_MIN_MATCH_PERCENT = VIDEO_TARGET_CONFIRM_MATCH_PERCENT
_DONE_SCAN_PHASES = frozenset({"done", "complete", "error", "failed", "cancelled"})
_FINALIZING_SCAN_PHASES = frozenset({"finalizing", "postprocess", "enriching"})
_metrics_emitter_started = False
_metrics_emitter_lock = threading.Lock()

_SCAN_HOLD_SUMMARY: Dict[str, object] = {
    "total_visits": 0,
    "unique_people": 0,
    "cameras": 0,
    "camera_count": 0,
    "cameras_configured": 0,
    "configured_locations": [],
    "last_seen": None,
    "registered_profiles": 0,
    "id_label": "Person ID",
    "filtered_person_id": None,
    "scan_hold": True,
}

HEIC_SUFFIXES = {".heic", ".heif"}
STANDARD_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_UPLOAD_SUFFIXES = STANDARD_IMAGE_SUFFIXES | HEIC_SUFFIXES

VIDEO_CODEC_USER_MESSAGE = (
    "This video uses an unsupported codec (often Apple HEVC/H.265 from iPhone). "
    "Please export or convert it to standard H.264 MP4 and try again."
)


class VideoCodecError(Exception):
    """Raised when OpenCV cannot decode an uploaded video (e.g. iPhone HEVC)."""


def _register_heif_opener() -> None:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass


_register_heif_opener()


def _hex_rgba(color: str, alpha: float) -> str:
    """Convert #rrggbb to rgba(); pass rgba()/rgb() through unchanged."""
    color = (color or "").strip()
    if color.lower().startswith(("rgba(", "rgb(")):
        return color
    h = color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return f"rgba(148, 154, 171, {alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def build_panel_theme_css(themes: Optional[Dict] = None) -> str:
    """Generate section-specific panel CSS from PANEL_THEMES."""
    t = themes or PANEL_THEMES
    ov, sr, cp, an = t["overview"], t["search"], t["captures"], t["analytics"]

    return f"""
    .theme-zone-overview .section-title {{
      color: {ov["accent"]};
      text-shadow: 0 0 8px {_hex_rgba(ov["accent"], 0.4)};
    }}
    .stat-card.theme-overview {{
      background: {ov["bg"]};
      border: 1px solid {ov["border"]};
      box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4), inset 0 1px 0 {_hex_rgba(ov["accent"], 0.12)};
    }}
    .stat-card.theme-overview .value {{
      color: {ov["accent"]};
      text-shadow: 0 0 14px {_hex_rgba(ov["accent"], 0.5)};
    }}
    .stat-card.theme-overview .label,
    .stat-card.theme-overview .hint {{ color: {ov["muted"]}; }}
    .stat-card.theme-overview.clickable:hover,
    .stat-card.theme-overview.clickable.active {{
      box-shadow: 0 0 0 1px {ov["border"]}, 0 0 24px {_hex_rgba(ov["accent"], 0.25)};
    }}
    .panel.theme-overview {{
      background: {ov["bg"]};
      border: 1px solid {ov["border"]};
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }}
    .panel.theme-overview:hover {{
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
      transform: none;
    }}
    .panel.theme-overview .panel-head {{
      color: {ov["accent"]};
      border-bottom: 1px solid {_hex_rgba(ov["border"], 0.45)};
      background: linear-gradient(90deg, {_hex_rgba(ov["accent"], 0.08)}, transparent);
    }}
    .panel.theme-overview .panel-body {{ color: {ov["text"]}; }}

    .panel.theme-search {{
      background: {sr["bg"]};
      border: 1px solid {sr["border"]};
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }}
    .panel.theme-search:hover {{
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
      transform: none;
    }}
    .panel.theme-search .panel-head {{
      color: {sr["accent"]};
      border-bottom: 1px solid {_hex_rgba(sr["border"], 0.5)};
      background: linear-gradient(90deg, {_hex_rgba(sr["accent"], 0.1)}, transparent);
    }}
    .panel.theme-search .panel-body {{ color: {sr["text"]}; }}
    .theme-search .preview-box {{
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid {_hex_rgba(sr["border"], 0.65)};
      box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.25);
    }}
    .theme-search .preview-placeholder {{ color: {sr["muted"]}; }}
    .theme-search .btn:not(.btn-search):not(.btn-clear) {{
      border-color: {sr["border"]};
      color: {sr["accent"]};
      background: {_hex_rgba(sr["border"], 0.15)};
    }}
    .theme-search #searchHint {{ color: {sr["accent"]}; }}

    .panel.theme-captures {{
      background: {cp["bg"]};
      border: 1px solid {cp["border"]};
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }}
    .panel.theme-captures:hover {{
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
      transform: none;
    }}
    .panel.theme-captures .panel-head {{
      color: {cp["accent"]};
      border-bottom: 1px solid {_hex_rgba(cp["border"], 0.6)};
      background: linear-gradient(90deg, {_hex_rgba(cp["accent"], 0.08)}, transparent);
    }}
    .panel.theme-captures .panel-body {{ color: {cp["text"]}; }}
    .theme-captures .filter-row select {{
      border: 1px solid {cp["border"]};
      color: {cp["text"]};
      background: rgba(0, 0, 0, 0.35);
      box-shadow: none;
    }}
    .theme-captures .filter-row select:hover,
    .theme-captures .filter-row select:focus {{
      border-color: {cp["accent"]};
      outline: none;
      box-shadow: none;
    }}
    .theme-captures .capture-meta strong {{ color: {cp["text"]}; }}
    .theme-captures #personFilterHint {{ color: {cp["accent"]}; }}

    .theme-zone-analytics .section-title {{
      color: {an["accent"]};
      text-shadow: 0 0 8px {_hex_rgba(an["accent"], 0.35)};
    }}
    .panel.theme-analytics {{
      background: {an["bg"]};
      border: 1px solid {an["border"]};
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
    }}
    .panel.theme-analytics:hover {{
      box-shadow: 0 0 0 1px {an["border"]}, 0 0 20px {_hex_rgba(an["accent"], 0.15)};
    }}
    .panel.theme-analytics .panel-head {{
      color: {an["accent"]};
      border-bottom: 1px solid {_hex_rgba(an["border"], 0.45)};
      background: linear-gradient(90deg, {_hex_rgba(an["accent"], 0.08)}, transparent);
    }}
    .panel.theme-analytics th {{
      color: {an["accent"]};
      background: {_hex_rgba(an["accent"], 0.06)};
    }}
    .panel.theme-analytics .viz-hbars .bar-fill {{
      background: linear-gradient(90deg, {an["border"]}, {an["accent"]} 55%, {an["muted"]});
      box-shadow: 0 0 12px {_hex_rgba(an["accent"], 0.35)};
    }}
    .panel.theme-analytics .viz-hbars .bar-count {{ color: {an["accent"]}; }}
    .panel.theme-analytics .timeline-chart .chart-line {{ stroke: {an["accent"]}; }}
    .panel.theme-analytics .timeline-chart .chart-dot {{
      fill: {an["muted"]};
      stroke: {an["border"]};
    }}
    .panel.theme-analytics .timeline-chart .chart-value {{ fill: {an["muted"]}; }}
    """


def _ensure_inference_device() -> None:
    """Configure CUDA/ONNX providers once for dashboard photo + video scan."""
    global _inference_initialized
    if _inference_initialized:
        return
    from inference_device import configure, resolve_torch_device
    from orion_frame_pipeline import audit_hardware_dependencies

    device = os.environ.get("DEVICE", "cuda")
    pure = os.environ.get("ORION_PURE_GPU", "1") == "1" or resolve_torch_device(device).startswith("cuda")
    configure(device, pure_gpu=pure)
    audit_hardware_dependencies(device_choice=device, pure_gpu=pure)
    _inference_initialized = True


def _require_cuda_video_scan() -> None:
    """
    Hard gate before dashboard video target scan.

    YOLOv8 Large + InsightFace must be on CUDA — raises RuntimeError on CPU fallback.
    """
    _ensure_inference_device()
    from orion_frame_pipeline import require_cuda_video_scan

    device = os.environ.get("DEVICE", "cuda")
    require_cuda_video_scan(device_choice=device)
    get_yolo()


def _set_layout_hold(active: bool) -> None:
    global _layout_hold_active
    with _layout_hold_lock:
        _layout_hold_active = bool(active)
    if not active and not is_dashboard_scan_hold_active():
        _release_scan_job()


def _arm_scan_job(
    mode: str,
    videos_total: int,
    duration: str,
    *,
    scan_id: Optional[str] = None,
) -> None:
    """Track batch/clip scan context for the SYSTEM STATUS telemetry panel."""
    with _scan_job_lock:
        _scan_job_state.update(
            {
                "active": True,
                "mode": str(mode or "scan"),
                "videos_total": 0,
                "videos_processed": get_session_videos_processed_count(),
                "duration": duration if duration and duration != "—" else "—",
                "avg_match": "—",
                "scan_id": str(scan_id).strip() if scan_id else None,
            }
        )


def _update_scan_job_progress(
    *,
    scan_id: Optional[str] = None,
    videos_processed: Optional[int] = None,
    videos_total: Optional[int] = None,
    duration: Optional[str] = None,
    avg_match: Optional[str] = None,
) -> None:
    with _scan_job_lock:
        if scan_id and _scan_job_state.get("scan_id") not in (None, str(scan_id)):
            return
        if videos_processed is not None:
            _scan_job_state["videos_processed"] = max(0, int(videos_processed))
        if videos_total is not None:
            _scan_job_state["videos_total"] = max(0, int(videos_total))
        if duration is not None and duration != "—":
            _scan_job_state["duration"] = duration
        if avg_match is not None and avg_match != "—":
            _scan_job_state["avg_match"] = avg_match


def _release_scan_job(scan_id: Optional[str] = None) -> None:
    with _scan_job_lock:
        current_id = _scan_job_state.get("scan_id")
        if scan_id and current_id not in (None, str(scan_id)):
            return
        locked_duration = None
        locked_avg_match = None
        token = str(scan_id or current_id or "").strip()
        if token:
            with _scan_processing_timer_lock:
                locked_duration = _scan_processing_locked.get(token)
            with _scan_match_scores_lock:
                locked_avg_match = _scan_match_locked.get(token)
        _scan_job_state.update(
            {
                "active": False,
                "mode": "live",
                "videos_total": 0,
                "videos_processed": 0,
                "duration": locked_duration or "—",
                "avg_match": locked_avg_match or "—",
                "scan_id": None,
            }
        )


def _is_scan_job_active() -> bool:
    with _scan_job_lock:
        return bool(_scan_job_state.get("active"))


def _build_system_status_payload() -> Dict[str, object]:
    """Live surveillance vs batch file-processing context for the command center."""
    videos_display = get_session_videos_processed_count()
    scanning = is_dashboard_scan_hold_active() or _is_scan_job_active()
    if scanning:
        with _scan_job_lock:
            job = dict(_scan_job_state)
        job_processed = int(job.get("videos_processed") or 0)
        videos_display = max(videos_display, job_processed)
        duration = str(job.get("duration") or "—")
        avg_match = str(job.get("avg_match") or "—")
        mode = str(job.get("mode") or "scan")
        return {
            "is_live": False,
            "is_scanning": True,
            "mode": mode,
            "label": "SCANNING",
            "videos": videos_display,
            "videos_total": videos_display,
            "videos_processed": videos_display,
            "duration": duration,
            "avg_match": avg_match if avg_match != "—" else None,
        }

    return {
        "is_live": True,
        "is_scanning": False,
        "mode": "live",
        "label": "ON LIVE",
        "videos": videos_display,
        "videos_total": videos_display,
        "videos_processed": videos_display,
        "duration": "—",
        "avg_match": None,
    }


def get_session_videos_processed_count() -> int:
    """Number of videos that have started inference during this server session."""
    with _session_videos_lock:
        return int(_session_videos_processed_count)


def _record_video_processing_started(*, scan_id: Optional[str] = None) -> int:
    """
    Increment the session VIDEOS counter when a clip enters the inference loop.

    Called once per video at processing start — never from on-disk folder pre-scan.
    """
    global _session_videos_processed_count
    with _session_videos_lock:
        _session_videos_processed_count += 1
        count = int(_session_videos_processed_count)
    _update_scan_job_progress(videos_processed=count, scan_id=scan_id)
    try:
        socketio.emit(
            "video_processing_started",
            {
                "status": "processing",
                "videos_processed": count,
                "scan_id": str(scan_id) if scan_id else None,
                "system_status": _build_system_status_payload(),
            },
        )
        socketio.emit(
            "video_processing_started",
            {
                "status": "processing",
                "videos_processed": count,
                "scan_id": str(scan_id) if scan_id else None,
            },
            namespace=METRICS_WS_NAMESPACE,
        )
    except Exception:
        pass
    try:
        socketio.emit(
            "dashboard_metrics",
            _metrics_payload(),
            namespace=METRICS_WS_NAMESPACE,
        )
    except Exception:
        pass
    if scan_id:
        try:
            _publish_scan_progress(
                {
                    "scan_id": str(scan_id),
                    "videos_processed": count,
                    "system_status": _build_system_status_payload(),
                }
            )
        except Exception:
            pass
    return count


def _video_processing_start_callback(scan_id: Optional[str] = None):
    """Folder / batch worker hook — count a video when inference begins."""

    def _on_start(_meta: Optional[dict] = None) -> None:
        _record_video_processing_started(scan_id=scan_id)

    return _on_start


def _reset_session_tracking_metrics() -> None:
    """
    Reset live tactical session counters when a browser client connects or refreshes.

    Ensures DETECTIONS TODAY, TOTAL MATCHES, and INCIDENTS cards start at zero for the
    new layout session instead of carrying over backend process memory.
    """
    global _detections_today_count, _total_matches_count, _incidents_reported_count
    global _detection_counters_bootstrapped, _detection_counters_date
    global _session_videos_processed_count
    today = date.today().isoformat()
    with _detection_counters_lock:
        _detections_today_count = 0
        _total_matches_count = 0
        _incidents_reported_count = 0
        _detection_counters_date = today
        _detection_counters_bootstrapped = True
    with _session_videos_lock:
        _session_videos_processed_count = 0
    with _scan_counted_match_keys_lock:
        _scan_counted_match_keys.clear()
    with _scan_job_lock:
        _scan_job_state["videos_processed"] = 0


def _ensure_detection_counters_bootstrap() -> None:
    """Seed live tactical counters from the database once per process."""
    global _detections_today_count, _total_matches_count, _detection_counters_date
    global _detection_counters_bootstrapped
    if _detection_counters_bootstrapped:
        return
    with _detection_counters_lock:
        if _detection_counters_bootstrapped:
            return
        _detection_counters_date = date.today().isoformat()
        try:
            cards = (fetch_tactical_metrics(db_path=get_db_path()).get("cards") or {})
            _detections_today_count = int(cards.get("detections_today", 0))
            _total_matches_count = int(cards.get("new_strangers_today", 0))
        except Exception:
            _detections_today_count = 0
            _total_matches_count = 0
        _detection_counters_bootstrapped = True


def _roll_detection_counters_if_new_day() -> None:
    """Reset today's detection tally after midnight; total matches persist."""
    global _detections_today_count, _detection_counters_date
    today = date.today().isoformat()
    if _detection_counters_date != today:
        _detection_counters_date = today
        _detections_today_count = 0


def get_incidents_reported_count() -> int:
    """Return the live session incidents counter for tactical metric cards."""
    _ensure_detection_counters_bootstrap()
    with _detection_counters_lock:
        return int(_incidents_reported_count)


def get_detection_counter_snapshot() -> Tuple[int, int]:
    """Return (detections_today, total_matches) for metrics cards."""
    _ensure_detection_counters_bootstrap()
    with _detection_counters_lock:
        _roll_detection_counters_if_new_day()
        return int(_detections_today_count), int(_total_matches_count)


def increment_detection_counters(count: int = 1) -> Tuple[int, int]:
    """Increment session counters and broadcast updated metrics to the UI."""
    if count <= 0:
        return get_detection_counter_snapshot()
    _ensure_detection_counters_bootstrap()
    with _detection_counters_lock:
        _roll_detection_counters_if_new_day()
        global _detections_today_count, _total_matches_count
        _detections_today_count += int(count)
        _total_matches_count += int(count)
        today_val = int(_detections_today_count)
        total_val = int(_total_matches_count)
    _broadcast_detection_counter_metrics(today_val, total_val)
    return today_val, total_val


def _merge_detection_counters_into_metrics(payload: dict) -> dict:
    """Attach live session counters without clobbering DB-backed daily totals."""
    if not isinstance(payload, dict):
        return payload
    session_today, session_total = get_detection_counter_snapshot()
    incidents_val = get_incidents_reported_count()
    out = dict(payload)
    cards = dict(out.get("cards") or {})
    cards["incidents_reported"] = incidents_val
    out["cards"] = cards
    out["live_detection_counters"] = {
        "detections_today": session_today,
        "total_matches": session_total,
        "incidents_reported": incidents_val,
    }
    return out


def _broadcast_detection_counter_metrics(
    detections_today: Optional[int] = None,
    total_matches: Optional[int] = None,
) -> None:
    """Push counter updates instantly over Socket.IO (metrics + scan progress)."""
    if detections_today is None or total_matches is None:
        detections_today, total_matches = get_detection_counter_snapshot()
    metric_cards = {
        "detections_today": int(detections_today),
        "total_matches": int(total_matches),
    }
    try:
        _publish_scan_progress(
            {
                "metric_cards": metric_cards,
                "detections_today": metric_cards["detections_today"],
                "total_matches": metric_cards["total_matches"],
            }
        )
    except Exception:
        pass
    try:
        socketio.emit(
            "dashboard_metrics",
            _metrics_payload(),
            namespace=METRICS_WS_NAMESPACE,
        )
    except Exception:
        pass


def _clear_scan_counted_match_keys(scan_id: Optional[str]) -> None:
    token = str(scan_id or "").strip()
    if not token:
        return
    with _scan_counted_match_keys_lock:
        _scan_counted_match_keys.pop(token, None)


def _claim_scan_match_counter_key(
    scan_id: Optional[str],
    *,
    frame: Optional[object],
    video_file: Optional[object],
) -> bool:
    """Return True when this (scan, frame, video) hit has not been counted yet."""
    token = str(scan_id or "").strip()
    if not token:
        return True
    try:
        frame_idx = int(frame) if frame is not None else -1
    except (TypeError, ValueError):
        frame_idx = -1
    video_key = str(video_file or "")
    dedupe_key = (frame_idx, video_key)
    with _scan_counted_match_keys_lock:
        bucket = _scan_counted_match_keys.setdefault(token, set())
        if dedupe_key in bucket:
            return False
        bucket.add(dedupe_key)
        return True


def _record_tactical_target_match(
    *,
    scan_id: Optional[str] = None,
    frame: Optional[object] = None,
    video_file: Optional[object] = None,
) -> bool:
    """Increment global detection counters for a confirmed target-face match."""
    if not _claim_scan_match_counter_key(scan_id, frame=frame, video_file=video_file):
        return False
    increment_detection_counters(1)
    return True


def _target_match_hit_callback(scan_id: str):
    """Orion pipeline hook — count each ≥30% target hit as it is discovered."""

    def _on_hit(hit: dict) -> None:
        if not isinstance(hit, dict):
            return
        _record_tactical_target_match(
            scan_id=scan_id,
            frame=hit.get("frame"),
            video_file=hit.get("video_file") or hit.get("original_filename"),
        )

    return _on_hit


def _inject_system_status_into_metrics(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    system_status = _build_system_status_payload()
    out["system_status"] = system_status
    out["is_live"] = bool(system_status.get("is_live", True))
    out["is_scanning"] = bool(system_status.get("is_scanning", False))
    if system_status.get("is_scanning"):
        out["tracker_online"] = True
    if "telegram_enabled" not in out:
        try:
            from telegram_alerts import telegram_configured

            out["telegram_enabled"] = bool(telegram_configured())
        except Exception:
            out["telegram_enabled"] = False
    return _merge_detection_counters_into_metrics(out)


def _arm_scan_job_from_hold_payload(payload: dict) -> None:
    """Optional early job arming when the UI posts folder/clip hints with layout hold."""
    if not isinstance(payload, dict):
        return
    mode = str(payload.get("mode") or payload.get("scan_mode") or "scan").strip().lower()
    duration = str(payload.get("duration") or payload.get("video_duration") or "—")
    folder_raw = (payload.get("folder_path") or "").strip().strip('"').strip("'")
    storage_mode = str(payload.get("storage_mode") or "local").strip().lower()

    if folder_raw:
        try:
            folder = normalize_batch_folder_path(folder_raw, storage_mode)
        except PermissionError:
            return
        if folder.is_dir():
            _arm_scan_job("folder", 0, duration)
            return

    _arm_scan_job(
        mode if mode in ("clip", "folder", "scan") else "clip",
        0,
        duration,
    )


def is_dashboard_scan_hold_active() -> bool:
    """True while a target trace scan is running or the UI requested a layout hold."""
    with _layout_hold_lock:
        if _layout_hold_active:
            return True
    with _scan_progress_lock:
        return bool(_active_scan_ids)


def _query_requests_scan_hold() -> bool:
    flag = (
        request.args.get("scan_hold")
        or request.args.get("layout_hold")
        or ""
    ).strip().lower()
    return flag in ("1", "true", "yes", "on")


def _should_hold_dashboard_data() -> bool:
    if _query_requests_scan_hold():
        return True
    return is_dashboard_scan_hold_active()


def _scan_hold_json_response(payload):
    response = jsonify(_inject_system_status_into_metrics(payload))
    response.headers["X-Scan-Hold"] = "1"
    response.headers["Cache-Control"] = "no-store"
    return response


def _format_elapsed_hms(seconds: float) -> str:
    """Format wall-clock elapsed seconds as HH:MM:SS for the processing timer."""
    return _format_duration_hms(int(max(0.0, seconds)))


def _start_scan_processing_timer(scan_id: str) -> None:
    """Register wall-clock start epoch for an active video scan."""
    token = str(scan_id or "").strip()
    if not token:
        return
    with _scan_processing_timer_lock:
        if token not in _scan_processing_started and token not in _scan_processing_locked:
            _scan_processing_started[token] = time.time()


def _clear_scan_processing_timer(scan_id: str) -> None:
    """Drop an in-flight processing timer after scan failure."""
    token = str(scan_id or "").strip()
    if not token:
        return
    with _scan_processing_timer_lock:
        _scan_processing_started.pop(token, None)


def _get_processing_duration_snapshot(scan_id: str) -> Tuple[str, float]:
    """Return current or locked elapsed processing duration for a scan."""
    token = str(scan_id or "").strip()
    if not token:
        return "00:00:00", 0.0
    with _scan_processing_timer_lock:
        if token in _scan_processing_locked:
            return (
                _scan_processing_locked[token],
                float(_scan_processing_locked_sec.get(token, 0.0)),
            )
        started = _scan_processing_started.get(token)
        if started is not None:
            elapsed = max(0.0, time.time() - started)
            return _format_elapsed_hms(elapsed), elapsed
    return "00:00:00", 0.0


def _finalize_scan_processing_timer(scan_id: str) -> Tuple[str, float]:
    """Lock final wall-clock processing duration when a scan completes."""
    token = str(scan_id or "").strip()
    if not token:
        return "00:00:00", 0.0
    with _scan_processing_timer_lock:
        if token in _scan_processing_locked:
            return (
                _scan_processing_locked[token],
                float(_scan_processing_locked_sec.get(token, 0.0)),
            )
        started = _scan_processing_started.pop(token, None)
        if started is None:
            return "00:00:00", 0.0
        elapsed = max(0.0, time.time() - started)
        label = _format_elapsed_hms(elapsed)
        _scan_processing_locked[token] = label
        _scan_processing_locked_sec[token] = elapsed
        return label, elapsed


def _format_avg_match_str(scores: List[float]) -> str:
    """Format mean face-match confidence as an integer percentage label."""
    if not scores:
        return "—"
    avg_match = sum(scores) / len(scores)
    return f"{int(avg_match)}%"


def _start_scan_match_tracking(scan_id: str) -> None:
    """Reset per-scan match score accumulator at scan start."""
    token = str(scan_id or "").strip()
    if not token:
        return
    with _scan_match_scores_lock:
        _scan_match_scores[token] = []
        _scan_match_locked.pop(token, None)


def _clear_scan_match_tracking(scan_id: str) -> None:
    """Drop in-flight match scores after scan failure."""
    token = str(scan_id or "").strip()
    if not token:
        return
    with _scan_match_scores_lock:
        _scan_match_scores.pop(token, None)
        _scan_match_locked.pop(token, None)


def _record_scan_match_score(scan_id: str, confidence_score: float) -> str:
    """Append a face comparison score and return the live running average label."""
    token = str(scan_id or "").strip()
    if not token:
        return "—"
    try:
        score = float(confidence_score)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(score):
        return "—"
    with _scan_match_scores_lock:
        bucket = _scan_match_scores.setdefault(token, [])
        bucket.append(score)
        avg_match_str = _format_avg_match_str(bucket)
    _update_scan_job_progress(scan_id=token, avg_match=avg_match_str)
    return avg_match_str


def _get_scan_avg_match_label(scan_id: str) -> str:
    """Return locked or live running average match label for a scan."""
    token = str(scan_id or "").strip()
    if not token:
        return "—"
    with _scan_match_scores_lock:
        if token in _scan_match_locked:
            return _scan_match_locked[token]
        scores = _scan_match_scores.get(token) or []
        return _format_avg_match_str(scores)


def _finalize_scan_match_tracking(scan_id: str) -> str:
    """Lock final average match percentage when a scan completes."""
    token = str(scan_id or "").strip()
    if not token:
        return "—"
    with _scan_match_scores_lock:
        if token in _scan_match_locked:
            return _scan_match_locked[token]
        scores = _scan_match_scores.pop(token, [])
        avg_match_str = _format_avg_match_str(scores)
        if avg_match_str != "—":
            _scan_match_locked[token] = avg_match_str
        return avg_match_str


def _match_score_callback(scan_id: str):
    """Orion pipeline hook — record each face comparison confidence score."""

    def _record(confidence_score: float) -> None:
        _record_scan_match_score(scan_id, confidence_score)

    return _record


def _enrich_progress_with_avg_match(payload: dict, scan_id: str) -> dict:
    """Attach live running average match confidence to scan progress payloads."""
    out = dict(payload)
    avg_match_str = _get_scan_avg_match_label(scan_id)
    out["avg_match"] = avg_match_str if avg_match_str != "—" else None
    if avg_match_str != "—":
        try:
            out["avg_match_percent"] = int(avg_match_str.rstrip("%"))
        except ValueError:
            pass
    return out


def _enrich_progress_with_processing_duration(payload: dict, scan_id: str) -> dict:
    """Attach live elapsed processing duration to scan progress payloads."""
    out = dict(payload)
    label, elapsed_sec = _get_processing_duration_snapshot(scan_id)
    out["processing_duration"] = label
    out["duration"] = label
    out["elapsed_seconds"] = round(elapsed_sec, 3)
    out["scan_duration_ms"] = int(elapsed_sec * 1000)
    return out


def _nocache_jsonify(payload: dict, status: int = 200):
    """JSON response with cache-busting headers for live progress polling."""
    resp = jsonify(payload)
    resp.status_code = status
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def _scan_progress_payload(scan_id: str) -> Optional[dict]:
    with _scan_progress_lock:
        data = _scan_progress.get(scan_id)
    if not data:
        return None
    out = dict(data)
    phase = str(out.get("phase") or "").strip().lower()
    raw = out.get("percent") or out.get("progress") or out.get("progress_percent") or 0
    out["percent"] = int(round(float(raw)))
    out["progress"] = out["percent"]
    out["is_scanning"] = bool(phase) and phase not in _DONE_SCAN_PHASES
    out["is_finalizing"] = phase in _FINALIZING_SCAN_PHASES
    return out


def _publish_scan_progress(payload: dict) -> None:
    """Push live scan progress to Socket.IO clients and in-memory poll cache."""
    payload = dict(payload)
    scan_id = str(payload.get("scan_id") or "")
    phase = str(payload.get("phase") or "").strip().lower()
    if not scan_id:
        return
    raw = payload.get("percent") or payload.get("progress") or payload.get("progress_percent") or 0
    payload["percent"] = int(round(float(raw)))
    payload["progress"] = payload["percent"]
    payload = _enrich_progress_with_processing_duration(payload, scan_id)
    payload = _enrich_progress_with_avg_match(payload, scan_id)
    with _scan_progress_lock:
        _scan_progress[scan_id] = payload
        if phase in _DONE_SCAN_PHASES:
            _active_scan_ids.discard(scan_id)
        elif phase:
            _active_scan_ids.add(scan_id)

    video_index = payload.get("video_index")
    progress_kwargs: Dict[str, object] = {
        "scan_id": scan_id,
        "duration": payload.get("processing_duration"),
        "avg_match": payload.get("avg_match"),
    }
    if payload.get("videos_processed") is not None:
        progress_kwargs["videos_processed"] = int(payload["videos_processed"])
    elif video_index is not None:
        progress_kwargs["videos_processed"] = get_session_videos_processed_count()
    _update_scan_job_progress(**progress_kwargs)

    socketio.emit("video_scan_progress", payload)
    socketio.emit("video_scan_progress", payload, namespace=METRICS_WS_NAMESPACE)


def _arm_scan_session(scan_id: str) -> None:
    """Mark a scan as active before inference begins."""
    token = str(scan_id or "").strip()
    if not token:
        return
    _start_scan_processing_timer(token)
    _start_scan_match_tracking(token)
    _init_scan_demo_windows(token)
    _publish_scan_progress(
        {
            "scan_id": token,
            "phase": "starting",
            "percent": 0,
            "frame_current": 0,
            "frame_total": 0,
        }
    )


def _release_scan_session(scan_id: str, phase: str = "error") -> None:
    token = str(scan_id or "").strip()
    if not token:
        return
    _clear_scan_processing_timer(token)
    _clear_scan_match_tracking(token)
    _publish_scan_progress({"scan_id": token, "phase": phase, "percent": 0})
    with _scan_progress_lock:
        still_active = bool(_active_scan_ids)
    with _layout_hold_lock:
        hold_active = _layout_hold_active
    if not still_active and not hold_active:
        _release_scan_job(token)


def _finalize_scan_session(scan_id: str) -> None:
    """Guarantee scan ids leave the active set once inference completes."""
    token = str(scan_id or "").strip()
    if not token:
        return
    final_label, elapsed_sec = _finalize_scan_processing_timer(token)
    final_avg_match = _finalize_scan_match_tracking(token)
    with _scan_progress_lock:
        _active_scan_ids.discard(token)
        still_active = bool(_active_scan_ids)
    _publish_scan_progress(
        {
            "scan_id": token,
            "phase": "done",
            "percent": 100,
            "processing_duration": final_label,
            "duration": final_label,
            "elapsed_seconds": round(elapsed_sec, 3),
            "scan_duration_ms": int(elapsed_sec * 1000),
            "avg_match": final_avg_match if final_avg_match != "—" else None,
        }
    )
    if not still_active and not _layout_hold_active:
        _release_scan_job(token)


def _scan_progress_callback(scan_id: str):
    def _emit(payload: dict) -> None:
        payload = dict(payload)
        payload["scan_id"] = scan_id
        _publish_scan_progress(payload)

    return _emit


def run_video_target_scan(
    target_image_path: Path,
    video_path: Path,
    yolo_model=None,
    *,
    scan_id: Optional[str] = None,
    video_index: int = 0,
    videos_total: int = 1,
    original_filename: Optional[str] = None,
    frame_step: Optional[int] = None,
):
    """CUDA-only dashboard clip scan — stride-5 default, optional sandbox stride override."""
    _require_cuda_video_scan()
    if yolo_model is None:
        yolo_model = get_yolo()
    token = scan_id or uuid.uuid4().hex[:12]
    _start_scan_processing_timer(token)
    _start_scan_match_tracking(token)
    _init_scan_demo_windows(token)
    _record_video_processing_started(scan_id=token)
    scan_kwargs: Dict = {
        "progress_callback": _scan_progress_callback(token),
        "match_score_callback": _match_score_callback(token),
        "scan_id": token,
        "video_index": video_index,
        "videos_total": videos_total,
        "original_filename": original_filename,
    }
    if frame_step is not None:
        scan_kwargs["frame_step"] = max(1, int(frame_step))
    ground_truth = _get_scan_ground_truth(token)
    if ground_truth is None and target_image_path.is_file():
        ground_truth = extract_target_profile_ground_truth(target_image_path)
        _register_scan_ground_truth(token, ground_truth)
    # Age/gender runs in enrich_scan_face_attributes after the scan (much faster mid-scan).
    scan_kwargs["face_demographics_callback"] = None
    scan_kwargs["target_hit_callback"] = _target_match_hit_callback(token)
    # Video-start telemetry is emitted above; Orion scan_for_target has no video_start_callback kwarg.
    result = scan_video_for_target(
        target_image_path,
        video_path,
        yolo_model=yolo_model,
        **scan_kwargs,
    )
    return enrich_scan_face_attributes(
        _filter_scan_results_for_display(result),
        target_image_path=target_image_path,
        video_path=video_path,
        ground_truth=ground_truth,
        scan_id=token,
    )


def _emit_scan_metrics_update(
    *,
    scan_id: Optional[str] = None,
) -> None:
    """Push live session VIDEOS count to job state, scan progress, and metrics WS."""
    videos_processed = get_session_videos_processed_count()
    progress_kwargs: Dict[str, object] = {"videos_processed": videos_processed}
    if scan_id:
        progress_kwargs["scan_id"] = str(scan_id)
    _update_scan_job_progress(**progress_kwargs)

    if scan_id:
        _publish_scan_progress(
            {
                "scan_id": str(scan_id),
                "videos_processed": videos_processed,
            }
        )

    try:
        socketio.emit(
            "dashboard_metrics",
            _metrics_payload(),
            namespace=METRICS_WS_NAMESPACE,
        )
    except Exception:
        pass


def process_single_video_stream(
    target_face_path: Union[str, Path],
    video_path: Union[str, Path],
    *,
    scan_id: Optional[str] = None,
    video_index: int = 0,
    videos_total: int = 1,
    yolo_model=None,
    frame_step: Optional[int] = None,
    original_filename: Optional[str] = None,
) -> dict:
    """Run one CUDA target-face scan against a single video file or RTSP stream."""
    target_path = Path(target_face_path)
    source = str(video_path).strip()
    video_file = Path(source)
    display_name = original_filename or (
        video_file.name if video_file.name and not source.startswith("rtsp://") else None
    )
    return run_video_target_scan(
        target_path,
        video_file,
        yolo_model=yolo_model,
        scan_id=scan_id,
        video_index=video_index,
        videos_total=videos_total,
        original_filename=display_name,
        frame_step=frame_step,
    )


def execute_tactical_ai_scan(
    target_face_path: Union[str, Path],
    video_source_input: str,
    *,
    scan_id: Optional[str] = None,
    yolo_model=None,
    frame_step: Optional[int] = None,
    storage_mode: str = "local",
    original_filename: Optional[str] = None,
) -> dict:
    """
    Unified scan orchestrator — auto-detects batch folder vs single clip/RTSP source.

    Accepts a target face image path plus either:
      - a local folder directory (batch mode), or
      - a single video file path / RTSP URL (clip mode).
    """
    target_path = Path(target_face_path)
    source = str(video_source_input or "").strip().strip('"').strip("'")
    error_payload = {
        "success": False,
        "matches": [],
        "borderline_suspects": [],
        "message": "",
    }

    if not target_path.is_file() or not source:
        print("[ERROR] Missing required Target Face or Video Source input values.")
        error_payload["message"] = (
            "Missing required Target Face or Video Source input values."
        )
        return error_payload

    token = str(scan_id or "").strip() or uuid.uuid4().hex[:12]
    ground_truth = extract_target_profile_ground_truth(target_path)
    _register_scan_ground_truth(token, ground_truth)

    try:
        folder = normalize_batch_folder_path(source, storage_mode)
    except PermissionError as exc:
        error_payload["message"] = str(exc)
        return error_payload
    if folder.is_dir():
        print(
            f"[BATCH MODE] Directory detected: {folder}. "
            "Scanning all valid video items..."
        )
        video_files = collect_videos_in_folder(folder, storage_mode)
        if not video_files:
            print("[WARN] No valid video formats found in the provided directory.")
            error_payload["message"] = (
                f"No video files ({', '.join(sorted(VIDEO_SEARCH_EXTENSIONS))}) "
                f"found in {folder}"
            )
            error_payload["folder_path"] = str(folder)
            return error_payload

        total_videos = len(video_files)
        _arm_scan_job("folder", 0, "00:00:00", scan_id=scan_id)

        _require_cuda_video_scan()
        if yolo_model is None:
            yolo_model = get_yolo()

        progress_cb = _scan_progress_callback(scan_id) if scan_id else None
        match_cb = _match_score_callback(scan_id) if scan_id else None
        # Age/gender is applied in post-scan enrichment — mid-scan demos were a major slowdown.
        demo_cb = None
        # Per-video start events fire inside scan_folder_for_target workers — not Orion.
        video_start_cb = _video_processing_start_callback(scan_id or token)

        folder_kwargs: Dict = {
            "target_image_path": target_path,
            "folder_path": folder,
            "yolo_model": yolo_model,
            "progress_callback": progress_cb,
            "match_score_callback": match_cb,
            "scan_id": scan_id or token,
            "storage_mode": storage_mode,
            "face_demographics_callback": demo_cb,
            "target_hit_callback": _target_match_hit_callback(scan_id or token),
            "video_start_callback": video_start_cb,
        }
        if frame_step is not None:
            folder_kwargs["frame_step"] = max(1, int(frame_step))
        result = scan_folder_for_target(**folder_kwargs)
        _emit_scan_metrics_update(scan_id=scan_id)
        return result

    if source.startswith("rtsp://") or os.path.isfile(source):
        print(f"[SINGLE MODE] Explicit source path detected: {redact_media_url(source)}")
        _arm_scan_job("clip", 0, "00:00:00", scan_id=scan_id)
        return process_single_video_stream(
            target_path,
            source,
            scan_id=scan_id,
            video_index=0,
            videos_total=1,
            yolo_model=yolo_model,
            frame_step=frame_step,
            original_filename=original_filename,
        )

    print(
        "[ERROR] Provided input string matches neither an existing directory "
        "folder nor a valid video source link."
    )
    error_payload["message"] = (
        "Provided input matches neither an existing folder nor a valid "
        "video file or RTSP URL."
    )
    return error_payload


def _resolve_tactical_scan_enrichment_paths(
    video_source_input: str,
    storage_mode: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (video_path, folder_path) for face-attribute enrichment."""
    source = str(video_source_input or "").strip().strip('"').strip("'")
    if source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return Path(source), None
    try:
        folder = normalize_batch_folder_path(source, storage_mode)
    except PermissionError:
        return None, None
    if folder.is_dir():
        return None, folder
    video_path = Path(source)
    if video_path.is_file():
        return video_path, None
    return None, None


def _run_tactical_target_scan_api(
    target_image_path: Path,
    video_source_input: str,
    *,
    scan_id: str,
    storage_mode: str = "local",
    frame_step: Optional[int] = None,
    original_filename: Optional[str] = None,
) -> Tuple[dict, int]:
    """Shared scan API handler — returns (json_payload, http_status)."""
    _arm_scan_session(scan_id)
    video_path, folder_path = _resolve_tactical_scan_enrichment_paths(
        video_source_input,
        storage_mode,
    )
    try:
        raw = execute_tactical_ai_scan(
            target_image_path,
            video_source_input,
            scan_id=scan_id,
            frame_step=frame_step,
            storage_mode=storage_mode,
            original_filename=original_filename,
        )
        # Frame pass is finished — tell the UI we are packaging results (age/gender, etc.)
        # so it does not look stuck at "Scanning … 100%".
        _publish_scan_progress(
            {
                "scan_id": scan_id,
                "phase": "finalizing",
                "percent": 100,
                "message": "Packaging match results…",
            }
        )
        if (
            not raw.get("success")
            and not raw.get("matches")
            and not raw.get("borderline_suspects")
            and raw.get("message")
        ):
            msg = str(raw["message"])
            if "Missing required" in msg or "matches neither" in msg:
                _release_scan_session(scan_id)
                return {"error": msg, "success": False, "matches": []}, 400
            if "No video files" in msg or "Folder not found" in msg:
                _release_scan_session(scan_id)
                return {"error": msg, "success": False, "matches": []}, 400

        ground_truth = _get_scan_ground_truth(scan_id)
        result = prepare_scan_api_payload(
            enrich_scan_face_attributes(
                _filter_scan_results_for_display(
                    normalize_scan_response(raw),
                ),
                target_image_path=target_image_path,
                video_path=video_path,
                folder_path=folder_path,
                ground_truth=ground_truth,
                scan_id=scan_id,
            ),
            scan_id,
        )
        result["scan_id"] = scan_id
        if folder_path is not None:
            result["storage_mode"] = storage_mode
        if original_filename:
            result["original_filename"] = original_filename
        if video_path is not None and video_path.is_file():
            file_runtime = calculate_video_duration(str(video_path))
            result = _attach_scan_processing_duration_fields(
                result,
                scan_id,
                video_file_runtime=file_runtime,
            )
        else:
            result = _attach_scan_processing_duration_fields(result, scan_id)
        _finalize_scan_session(scan_id)
        return result, 200
    except RuntimeError as exc:
        _release_scan_session(scan_id)
        return {"error": str(exc), "success": False, "matches": []}, 503
    except VideoCodecError:
        _release_scan_session(scan_id)
        raise
    except FileNotFoundError as exc:
        _release_scan_session(scan_id)
        return {"error": str(exc), "success": False, "matches": []}, 500
    except Exception as exc:
        _release_scan_session(scan_id)
        import traceback

        print(f"[SCAN ERROR] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        msg = str(exc).lower()
        if any(
            token in msg
            for token in ("codec", "hevc", "h265", "h.265", "encoding", "unsupported", "ffmpeg")
        ):
            return {"error": VIDEO_CODEC_USER_MESSAGE, "success": False, "matches": []}, 400
        return {
            "error": f"Video scan failed: {exc}",
            "success": False,
            "matches": [],
        }, 500
    finally:
        _clear_scan_ground_truth(scan_id)
        _clear_scan_demo_windows(scan_id)
        _clear_scan_counted_match_keys(scan_id)


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _ensure_inference_device()
        from inference_device import torch_cuda_active
        from main import PROJECT_DIR
        from yolo_loader import DEFAULT_MODEL, YOLO_LARGE_LOAD_STATUS, load_yolo_model

        if not torch_cuda_active():
            raise RuntimeError(
                "CRITICAL: yolov8l.pt requires NVIDIA CUDA "
                "(torch.cuda.is_available() == False).\n"
                "  Run: py -3.12 run_system.py --device cuda"
            )
        print("Loading YOLO Large for photo/video search (one-time)...")
        _yolo_model = load_yolo_model(DEFAULT_MODEL, PROJECT_DIR)
        if not torch_cuda_active():
            raise RuntimeError(
                "CRITICAL: YOLO Large loaded but CUDA is inactive — refusing CPU execution."
            )
        print(YOLO_LARGE_LOAD_STATUS)
    return _yolo_model


def get_db_path() -> Path:
    return Path(app.config.get("DB_PATH", DEFAULT_DB))


def _is_heic_path(path: Path) -> bool:
    return path.suffix.lower() in HEIC_SUFFIXES


def _convert_heic_to_jpeg(source_path: Path) -> Path:
    """Convert HEIC/HEIF to RGB JPEG for OpenCV / YOLO / FaceRecognizerSF."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError(
            "HEIC images require Pillow. Install: pip install pillow pillow-heif"
        ) from exc

    jpeg_path = source_path.with_suffix(".jpg")
    try:
        with Image.open(source_path) as img:
            rgb = img.convert("RGB")
            rgb.save(jpeg_path, "JPEG", quality=92)
    except Exception as exc:
        raise ValueError(
            f"Could not convert HEIC image ({source_path.name}): {exc}. "
            "Install pillow-heif: pip install pillow-heif"
        ) from exc

    try:
        source_path.unlink()
    except OSError:
        pass
    return jpeg_path


def _normalize_image_upload_path(image_path: Path) -> Path:
    """Ensure the saved upload is readable as BGR by OpenCV (JPEG for HEIC)."""
    if _is_heic_path(image_path):
        return _convert_heic_to_jpeg(image_path)

    if cv2.imread(str(image_path)) is not None:
        return image_path

    try:
        from PIL import Image
    except ImportError:
        return image_path

    jpeg_path = image_path.with_suffix(".jpg")
    try:
        with Image.open(image_path) as img:
            img.convert("RGB").save(jpeg_path, "JPEG", quality=92)
        if jpeg_path != image_path and image_path.is_file():
            try:
                image_path.unlink()
            except OSError:
                pass
        return jpeg_path
    except Exception:
        return image_path


def save_uploaded_image(
    upload_file: FileStorage, dest_path: Path
) -> Path:
    """Save an uploaded image and return a path OpenCV can read."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    upload_file.save(str(dest_path))
    return _normalize_image_upload_path(dest_path)


def _image_suffix_from_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower() or ".jpg"
    if suffix not in IMAGE_UPLOAD_SUFFIXES:
        suffix = ".jpg"
    return suffix


def _format_duration_hms(total_seconds: int) -> str:
    """Format integer seconds as HH:MM:SS."""
    seconds = max(0, int(total_seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def calculate_video_duration(video_path: str) -> str:
    """Return HH:MM:SS duration for a local video file, or em dash when unknown."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "—"
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 0 and frame_count > 0:
            return _format_duration_hms(int(frame_count / fps))
        return "—"
    finally:
        cap.release()


def calculate_folder_videos_duration(
    folder_path: Union[str, Path],
    storage_mode: str = "local",
) -> str:
    """Sum HH:MM:SS duration across decodable video files in a folder."""
    folder = (
        normalize_batch_folder_path(folder_path, storage_mode)
        if isinstance(folder_path, str)
        else folder_path
    )
    if not folder.is_dir():
        return "—"
    videos = collect_videos_in_folder(folder, storage_mode)
    if not videos:
        return "—"
    total_seconds = 0
    found = False
    for entry in videos:
        cap = cv2.VideoCapture(str(entry))
        if not cap.isOpened():
            continue
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0 and frame_count > 0:
                total_seconds += int(frame_count / fps)
                found = True
        finally:
            cap.release()
    if not found:
        return "—"
    return _format_duration_hms(total_seconds)


def _attach_scan_processing_duration_fields(
    payload: dict,
    scan_id: str,
    *,
    video_file_runtime: Optional[str] = None,
) -> dict:
    """Inject wall-clock scan processing duration into API payloads for the UI."""
    if not isinstance(payload, dict):
        return payload
    label, elapsed_sec = _get_processing_duration_snapshot(scan_id)
    out = dict(payload)
    out["processing_duration"] = label
    out["duration"] = label
    out["elapsed_seconds"] = round(elapsed_sec, 3)
    out["scan_duration_ms"] = int(elapsed_sec * 1000)
    out["execution_duration_ms"] = out["scan_duration_ms"]
    if video_file_runtime and video_file_runtime != "—":
        out["video_file_runtime"] = video_file_runtime
    avg_match_str = _get_scan_avg_match_label(scan_id)
    if avg_match_str == "—" and out.get("best_match_percent"):
        avg_match_str = f"{int(out['best_match_percent'])}%"
    scan_status = {
        "is_live": False,
        "is_scanning": False,
        "mode": "complete",
        "label": "SCAN COMPLETE",
        "videos": get_session_videos_processed_count(),
        "videos_total": get_session_videos_processed_count(),
        "videos_processed": get_session_videos_processed_count(),
        "duration": label,
        "avg_match": avg_match_str if avg_match_str != "—" else None,
    }
    out["avg_match"] = scan_status["avg_match"]
    out.setdefault("system_status", scan_status)
    if isinstance(out.get("system_status"), dict):
        out["system_status"] = {
            **out["system_status"],
            "duration": label,
            "processing_duration": label,
            "avg_match": scan_status.get("avg_match"),
        }
    return out


# ---------------------------------------------------------------------------
# Age / gender classification (DeepFace primary, OpenCV Caffe fallback)
# ---------------------------------------------------------------------------

FACE_ATTR_MODELS_DIR = PROJECT_DIR / "models"
FACE_ATTR_AGE_PROTO = FACE_ATTR_MODELS_DIR / "deploy_age.prototxt"
FACE_ATTR_AGE_MODEL = FACE_ATTR_MODELS_DIR / "age_net.caffemodel"
FACE_ATTR_GENDER_PROTO = FACE_ATTR_MODELS_DIR / "deploy_gender.prototxt"
FACE_ATTR_GENDER_MODEL = FACE_ATTR_MODELS_DIR / "gender_net.caffemodel"
FACE_ATTR_MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
FACE_ATTR_GENDER_LABELS = ("Male", "Female")
FACE_ATTR_AGE_BUCKETS = (
    "0-2",
    "4-6",
    "8-12",
    "15-20",
    "25-32",
    "38-43",
    "48-53",
    "60-100",
)
FACE_ATTR_MIN_CONFIDENCE = 0.55
FACE_ATTR_MAX_ENRICH_HITS = 12
FACE_ATTR_UNKNOWN = "N/A"

_dashboard_age_net: Optional[cv2.dnn_Net] = None
_dashboard_gender_net: Optional[cv2.dnn_Net] = None
_face_attr_model_lock = threading.Lock()
_face_attr_backend: Optional[str] = None
_deepface_initialized = False


def _deepface_ready() -> bool:
    return _DEEPFACE_AVAILABLE and DeepFace is not None


def _map_deepface_gender(label: object) -> Optional[str]:
    text = str(label or "").strip().lower()
    if text in ("man", "male"):
        return "Male"
    if text in ("woman", "female"):
        return "Female"
    return None


def _initialize_deepface_models() -> bool:
    """Preload DeepFace age/gender analyzers once for dashboard face crops."""
    global _deepface_initialized
    if not _deepface_ready():
        return False
    with _face_attr_model_lock:
        if _deepface_initialized:
            return True
        try:
            for task in ("Age", "Gender"):
                try:
                    DeepFace.build_model(task)
                except Exception:
                    pass
            stub = np.zeros((64, 64, 3), dtype=np.uint8)
            DeepFace.analyze(
                img_path=stub,
                actions=["age", "gender"],
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )
            _deepface_initialized = True
            return True
        except Exception as exc:
            print(f"DeepFace age/gender warmup skipped ({exc})")
            _deepface_initialized = True
            return False


def _opencv_age_gender_models_ready() -> bool:
    paths = (
        FACE_ATTR_AGE_PROTO,
        FACE_ATTR_AGE_MODEL,
        FACE_ATTR_GENDER_PROTO,
        FACE_ATTR_GENDER_MODEL,
    )
    return all(p.is_file() and p.stat().st_size > 1000 for p in paths)


def _resolve_face_attribute_backend() -> str:
    """Pick face_attributes (InsightFace/DeepFace/OpenCV), or legacy fallbacks."""
    global _face_attr_backend
    if _face_attr_backend is not None:
        return _face_attr_backend
    pref = os.environ.get("FACE_ATTR_BACKEND", "auto").strip().lower()
    try:
        from face_attributes import active_backend_name, attribute_models_ready

        if pref == "insightface" and attribute_models_ready():
            _face_attr_backend = "face_attributes"
        elif pref == "opencv" and _opencv_age_gender_models_ready():
            _face_attr_backend = "opencv"
        elif pref == "deepface" and _deepface_ready():
            _face_attr_backend = "deepface"
        elif attribute_models_ready():
            _face_attr_backend = "face_attributes"
        elif _deepface_ready():
            _face_attr_backend = "deepface"
        elif _opencv_age_gender_models_ready():
            _face_attr_backend = "opencv"
        else:
            _face_attr_backend = "none"
        if _face_attr_backend == "face_attributes":
            backend = active_backend_name()
            if backend and backend != "none":
                print(f"[DASHBOARD] Age/gender backend: {backend}")
    except Exception:
        if pref == "opencv" and _opencv_age_gender_models_ready():
            _face_attr_backend = "opencv"
        elif pref == "deepface" and _deepface_ready():
            _face_attr_backend = "deepface"
        elif _deepface_ready():
            _face_attr_backend = "deepface"
        elif _opencv_age_gender_models_ready():
            _face_attr_backend = "opencv"
        else:
            _face_attr_backend = "none"
    return _face_attr_backend


def initialize_age_gender_models() -> str:
    """Load age/gender models once. Returns active backend name."""
    backend = _resolve_face_attribute_backend()
    if backend == "deepface":
        _initialize_deepface_models()
    elif backend == "opencv":
        _load_dashboard_age_gender_nets()
    elif backend == "face_attributes":
        try:
            from face_attributes import preload_attribute_models

            preload_attribute_models()
        except Exception:
            pass
    return backend


def _load_dashboard_age_gender_nets() -> Tuple[cv2.dnn_Net, cv2.dnn_Net]:
    global _dashboard_age_net, _dashboard_gender_net
    with _face_attr_model_lock:
        if _dashboard_age_net is not None and _dashboard_gender_net is not None:
            return _dashboard_age_net, _dashboard_gender_net
        if not _opencv_age_gender_models_ready():
            raise FileNotFoundError(
                f"Age/gender Caffe models missing in {FACE_ATTR_MODELS_DIR}. "
                "Run: python download_face_attributes.py --no-proxy"
            )
        _dashboard_age_net = cv2.dnn.readNet(
            str(FACE_ATTR_AGE_MODEL), str(FACE_ATTR_AGE_PROTO)
        )
        _dashboard_gender_net = cv2.dnn.readNet(
            str(FACE_ATTR_GENDER_MODEL), str(FACE_ATTR_GENDER_PROTO)
        )
        return _dashboard_age_net, _dashboard_gender_net


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64).flatten()
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / max(float(exp_x.sum()), 1e-9)


def _int_age_to_bucket(age: int) -> str:
    if age <= 2:
        return "0-2"
    if age <= 6:
        return "4-6"
    if age <= 12:
        return "8-12"
    if age <= 20:
        return "15-20"
    if age <= 32:
        return "25-32"
    if age <= 43:
        return "38-43"
    if age <= 53:
        return "48-53"
    return "60-100"


def _pad_face_crop(face_bgr: np.ndarray, margin: float = 0.20) -> np.ndarray:
    h, w = face_bgr.shape[:2]
    pad_y = max(1, int(h * margin))
    pad_x = max(1, int(w * margin))
    return cv2.copyMakeBorder(
        face_bgr, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
    )


def _classify_face_crop_opencv(face_bgr: np.ndarray) -> Tuple[str, str]:
    """Run Caffe age/gender nets on a BGR face crop."""
    try:
        age_net, gender_net = _load_dashboard_age_gender_nets()
    except FileNotFoundError:
        return FACE_ATTR_UNKNOWN, FACE_ATTR_UNKNOWN

    face = _pad_face_crop(face_bgr)
    if max(face.shape[:2]) < 64:
        face = cv2.resize(face, (227, 227), interpolation=cv2.INTER_LINEAR)

    blob = cv2.dnn.blobFromImage(
        face,
        scalefactor=1.0,
        size=(227, 227),
        mean=FACE_ATTR_MODEL_MEAN,
        swapRB=False,
    )

    gender_net.setInput(blob)
    gender_probs = _softmax_logits(gender_net.forward())
    gender_idx = int(gender_probs.argmax())
    gender_conf = float(gender_probs[gender_idx])
    gender = (
        FACE_ATTR_GENDER_LABELS[gender_idx]
        if 0 <= gender_idx < len(FACE_ATTR_GENDER_LABELS)
        else FACE_ATTR_UNKNOWN
    )

    age_net.setInput(blob)
    age_probs = _softmax_logits(age_net.forward())
    age_idx = int(age_probs.argmax())
    age_conf = float(age_probs[age_idx])
    age_bucket = (
        FACE_ATTR_AGE_BUCKETS[age_idx]
        if 0 <= age_idx < len(FACE_ATTR_AGE_BUCKETS)
        else FACE_ATTR_UNKNOWN
    )

    if gender_conf < FACE_ATTR_MIN_CONFIDENCE:
        gender = FACE_ATTR_UNKNOWN
    if age_conf < FACE_ATTR_MIN_CONFIDENCE:
        age_bucket = FACE_ATTR_UNKNOWN

    return age_bucket, gender


def _classify_face_crop_deepface(face_bgr: np.ndarray) -> Tuple[str, str, Optional[int]]:
    """Run DeepFace age/gender analysis on a BGR face crop."""
    if not _deepface_ready():
        return FACE_ATTR_UNKNOWN, FACE_ATTR_UNKNOWN, None

    face = _pad_face_crop(face_bgr)
    if max(face.shape[:2]) < 48:
        face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    try:
        results = DeepFace.analyze(
            img_path=rgb,
            actions=["age", "gender"],
            enforce_detection=False,
            detector_backend="skip",
            silent=True,
        )
    except Exception:
        return FACE_ATTR_UNKNOWN, FACE_ATTR_UNKNOWN, None

    row = results[0] if isinstance(results, list) else results
    if not isinstance(row, dict):
        return FACE_ATTR_UNKNOWN, FACE_ATTR_UNKNOWN, None

    gender_raw = row.get("dominant_gender") or row.get("gender")
    gender_conf = 0.0
    gender: Optional[str] = None

    if isinstance(gender_raw, dict) and gender_raw:
        label = max(gender_raw, key=gender_raw.get)
        gender_conf = float(gender_raw[label]) / 100.0
        gender = _map_deepface_gender(label)
    elif isinstance(gender_raw, str):
        gender = _map_deepface_gender(gender_raw)
        gender_conf = 0.85 if gender else 0.0

    age_years: Optional[int] = None
    try:
        age_val = row.get("age")
        if age_val is not None:
            age_years = int(round(float(age_val)))
    except (TypeError, ValueError):
        age_years = None

    gender_out = gender if gender and gender_conf >= FACE_ATTR_MIN_CONFIDENCE else FACE_ATTR_UNKNOWN
    age_bucket = FACE_ATTR_UNKNOWN
    if age_years is not None and age_years > 0:
        age_bucket = _int_age_to_bucket(age_years)

    return age_bucket, gender_out, age_years


def _classify_face_crop_face_attributes(face_bgr: np.ndarray) -> Tuple[str, str, Optional[int]]:
    try:
        from face_attributes import predict_face_attributes_verbose

        pred = predict_face_attributes_verbose(face_bgr)
    except Exception:
        return FACE_ATTR_UNKNOWN, FACE_ATTR_UNKNOWN, None

    gender = pred.gender if pred.gender and pred.gender != "Unknown" else FACE_ATTR_UNKNOWN
    if pred.gender_conf < FACE_ATTR_MIN_CONFIDENCE:
        gender = FACE_ATTR_UNKNOWN

    age_bucket = FACE_ATTR_UNKNOWN
    age_years: Optional[int] = None
    if pred.age is not None and pred.age_conf >= FACE_ATTR_MIN_CONFIDENCE:
        age_years = int(pred.age)
        age_bucket = _int_age_to_bucket(age_years)
    return age_bucket, gender, age_years


def classify_face_attributes_detailed(
    face_bgr: np.ndarray,
) -> Tuple[str, str, Optional[int]]:
    """
    Predict age bucket, gender, and optional numeric age from a face crop (BGR).
    Returns ('N/A', 'N/A', None) when models are unavailable or confidence is low.
    """
    return _analyze_face_demographics(face_bgr)


def classify_face_attributes(face_bgr: np.ndarray) -> Tuple[str, str]:
    """
    Predict age bucket and gender from a cropped face region (BGR).
    Returns ('N/A', 'N/A') when models are unavailable or confidence is low.
    """
    age_bucket, gender, _age_years = classify_face_attributes_detailed(face_bgr)
    return age_bucket, gender


def _decode_data_url_image(data_url: Optional[str]) -> Optional[np.ndarray]:
    """Decode a data:image/... URL into a BGR ndarray for OpenCV / attribute models."""
    if not data_url or not str(data_url).startswith("data:image"):
        return None
    try:
        raw = base64.b64decode(str(data_url).split(",", 1)[-1])
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None and img.size > 0:
            return img
    except Exception:
        pass
    return None


def _decode_data_url_bytes(data_url: str) -> Optional[bytes]:
    """Decode a data: URL to raw JPEG/PNG bytes (thumbnail persistence)."""
    if not data_url or not isinstance(data_url, str) or not data_url.startswith("data:"):
        return None
    try:
        _header, b64 = data_url.split(",", 1)
        return base64.b64decode(b64)
    except (ValueError, TypeError):
        return None


def _crop_face_from_bbox(
    frame: np.ndarray,
    face_box: Tuple[int, int, int, int],
    *,
    pad_ratio: float = 0.12,
) -> Optional[np.ndarray]:
    """Crop a padded face region from a full BGR frame using a detector bounding box."""
    if frame is None or frame.size == 0:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in face_box)
    except (TypeError, ValueError):
        return None
    h, w = frame.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    return crop if crop is not None and crop.size > 0 else None


def _analyze_face_demographics(face_crop: np.ndarray) -> Tuple[str, str, Optional[int]]:
    """
    Run age/gender attribute analysis on a detected face crop (BGR).

    Uses DeepFace when available, then OpenCV Caffe / face_attributes fallbacks.
    """
    if face_crop is None or face_crop.size == 0:
        return FACE_ATTR_UNKNOWN, FACE_ATTR_UNKNOWN, None

    initialize_age_gender_models()
    backends: List[str] = []
    primary = _resolve_face_attribute_backend()
    if primary and primary != "none":
        backends.append(primary)
    for fallback in ("deepface", "opencv", "face_attributes"):
        if fallback not in backends:
            backends.append(fallback)

    last_exc: Optional[Exception] = None
    for backend in backends:
        try:
            if backend == "deepface" and _deepface_ready():
                return _classify_face_crop_deepface(face_crop)
            if backend == "opencv" and _opencv_age_gender_models_ready():
                age_bucket, gender = _classify_face_crop_opencv(face_crop)
                return age_bucket, gender, None
            if backend == "face_attributes":
                return _classify_face_crop_face_attributes(face_crop)
        except Exception as exc:
            last_exc = exc
            print(f"[WARN] Demographics extraction failed ({backend}): {exc}")

    if _deepface_ready():
        try:
            face = _pad_face_crop(face_crop)
            if max(face.shape[:2]) < 48:
                face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            results = DeepFace.analyze(
                img_path=rgb,
                actions=["age", "gender"],
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )
            row = results[0] if isinstance(results, list) else results
            if isinstance(row, dict):
                gender_raw = row.get("dominant_gender") or row.get("gender")
                gender = _map_deepface_gender(gender_raw)
                age_years: Optional[int] = None
                try:
                    if row.get("age") is not None:
                        age_years = int(round(float(row["age"])))
                except (TypeError, ValueError):
                    age_years = None
                age_bucket = (
                    _int_age_to_bucket(age_years)
                    if age_years is not None and age_years > 0
                    else FACE_ATTR_UNKNOWN
                )
                gender_out = gender or FACE_ATTR_UNKNOWN
                if gender_out != FACE_ATTR_UNKNOWN or age_bucket != FACE_ATTR_UNKNOWN:
                    return age_bucket, gender_out, age_years
        except Exception as exc:
            last_exc = exc
            print(f"[WARN] Demographics extraction failed on this frame: {exc}")

    if last_exc is not None:
        print(f"[WARN] Demographics extraction failed on this frame: {last_exc}")
    return FACE_ATTR_UNKNOWN, FACE_ATTR_UNKNOWN, None


def _init_scan_demo_windows(scan_id: str) -> None:
    """Create rolling majority-vote buffers for a target scan session."""
    token = str(scan_id or "").strip()
    if not token:
        return
    with _scan_demo_windows_lock:
        _scan_demo_windows[token] = {
            "gender_window": deque(maxlen=TARGET_DEMOGRAPHICS_SMOOTHING_WINDOW),
            "age_window": deque(maxlen=TARGET_DEMOGRAPHICS_SMOOTHING_WINDOW),
        }


def _clear_scan_demo_windows(scan_id: Optional[str]) -> None:
    token = str(scan_id or "").strip()
    if not token:
        return
    with _scan_demo_windows_lock:
        _scan_demo_windows.pop(token, None)


def _demographic_value_known(value: Optional[object]) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in ("unknown", "n/a", "")


def _majority_vote(window: deque) -> Optional[str]:
    """Return the most common non-empty demographic label in the rolling window."""
    valid = [str(v).strip() for v in window if _demographic_value_known(v)]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def _demographics_payload_from_smoothed(
    gender: Optional[str],
    age: Optional[str],
    *,
    is_target_matched: bool = True,
) -> Dict[str, object]:
    """Build a hit demographics dict from majority-voted model outputs."""
    out: Dict[str, object] = {
        "gender_source": "temporal_smoothing",
        "age_source": "temporal_smoothing",
    }
    if is_target_matched:
        out["is_target_matched"] = True
    if _demographic_value_known(gender):
        gender_text = str(gender).strip()
        out["gender"] = gender_text
        out["predicted_gender"] = gender_text
    if _demographic_value_known(age):
        age_text = str(age).strip()
        out["age"] = age_text
        out["predicted_age"] = age_text
        out["age_bucket"] = age_text
        out["age_group"] = age_text
    return out


def _push_and_smooth_demographics(
    scan_id: Optional[str],
    *,
    raw_gender: str,
    raw_age: str,
) -> Dict[str, object]:
    """
    Push raw per-frame model predictions into rolling windows and return
    majority-voted stabilized demographics for FACE_TARGET_FOUND rows.
    """
    token = str(scan_id or "").strip() or "default"
    with _scan_demo_windows_lock:
        if token not in _scan_demo_windows:
            _scan_demo_windows[token] = {
                "gender_window": deque(maxlen=TARGET_DEMOGRAPHICS_SMOOTHING_WINDOW),
                "age_window": deque(maxlen=TARGET_DEMOGRAPHICS_SMOOTHING_WINDOW),
            }
        windows = _scan_demo_windows[token]
        if _demographic_value_known(raw_gender):
            windows["gender_window"].append(str(raw_gender).strip())
        if _demographic_value_known(raw_age):
            windows["age_window"].append(str(raw_age).strip())
        smooth_gender = _majority_vote(windows["gender_window"])
        smooth_age = _majority_vote(windows["age_window"])
    return _demographics_payload_from_smoothed(smooth_gender, smooth_age)


def _peek_smoothed_demographics(scan_id: Optional[str]) -> Dict[str, object]:
    """Read current majority-vote demographics without pushing a new frame."""
    token = str(scan_id or "").strip()
    if not token:
        return {}
    with _scan_demo_windows_lock:
        windows = _scan_demo_windows.get(token)
        if not windows:
            return {}
        smooth_gender = _majority_vote(windows["gender_window"])
        smooth_age = _majority_vote(windows["age_window"])
    return _demographics_payload_from_smoothed(
        smooth_gender,
        smooth_age,
        is_target_matched=False,
    )


def _stabilize_target_match_demographics(
    hit: Dict,
    *,
    scan_id: Optional[str] = None,
    raw_gender: Optional[str] = None,
    raw_age: Optional[str] = None,
) -> Dict:
    """Apply temporal majority voting to a target-match hit's raw model outputs."""
    if not isinstance(hit, dict):
        return hit
    gender = raw_gender
    age = raw_age
    if gender is None:
        gender = str(
            hit.get("gender") or hit.get("predicted_gender") or FACE_ATTR_UNKNOWN
        )
    if age is None:
        age = str(
            hit.get("age_bucket")
            or hit.get("age_group")
            or hit.get("age")
            or hit.get("predicted_age")
            or FACE_ATTR_UNKNOWN
        )
    out = dict(hit)
    out.update(_push_and_smooth_demographics(scan_id, raw_gender=gender, raw_age=age))
    return out


def _target_match_confidence_percent(hit: Dict) -> int:
    try:
        return int(
            hit.get("match_percent")
            or hit.get("confidence_score")
            or hit.get("confidence")
            or 0
        )
    except (TypeError, ValueError):
        return 0


def _is_target_match_hit(hit: Dict) -> bool:
    """True when a detection row is a target-face match above the borderline threshold."""
    if not isinstance(hit, dict):
        return False
    if hit.get("is_target_matched") is True:
        return True
    return _target_match_confidence_percent(hit) >= SCAN_DISPLAY_MIN_MATCH_PERCENT


def _extract_hit_match_percentage(hit: Dict) -> int:
    """Resolve integer similarity % from a scan hit or registry row."""
    if not isinstance(hit, dict):
        return 0
    for key in ("match_percent", "confidence_score", "confidence", "percentage"):
        raw = hit.get(key)
        if raw is None or raw == "":
            continue
        try:
            text = str(raw).strip().rstrip("%")
            if text:
                return int(round(float(text)))
        except (TypeError, ValueError):
            continue
    distance = hit.get("distance")
    if distance is not None:
        try:
            from identity import confidence_score_from_distance

            return int(confidence_score_from_distance(float(distance)))
        except (TypeError, ValueError):
            pass
    return 0


def _passes_scan_display_threshold(
    hit: Dict,
    *,
    min_percent: int = SCAN_DISPLAY_MIN_MATCH_PERCENT,
) -> bool:
    """Strict floor — drop weak borderline frames below the dashboard minimum."""
    return _extract_hit_match_percentage(hit) >= int(min_percent)


def _filter_scan_hit_list(
    hits: Optional[List],
    *,
    min_percent: int = SCAN_DISPLAY_MIN_MATCH_PERCENT,
) -> List:
    if not isinstance(hits, list):
        return []
    return [h for h in hits if isinstance(h, dict) and _passes_scan_display_threshold(h, min_percent=min_percent)]


def _filter_scan_results_for_display(
    result: Dict,
    *,
    min_percent: int = SCAN_DISPLAY_MIN_MATCH_PERCENT,
) -> Dict:
    """
    Remove sub-threshold similarity hits before API / Socket.IO emission.

    Enforces a hard 30% floor so weak BORDERLINE rows never reach the UI grid.
    """
    if not isinstance(result, dict):
        return result
    out = dict(result)
    for key in (
        "matches",
        "borderline_suspects",
        "near_miss_suspects",
        "all_matches",
        "confirmed_matches",
        "detections",
        "scan_results",
    ):
        items = out.get(key)
        if isinstance(items, list):
            out[key] = _filter_scan_hit_list(items, min_percent=min_percent)

    thresholds = dict(out.get("scan_thresholds") or {})
    thresholds["display_min"] = int(min_percent)
    thresholds["confirmed_min"] = int(min_percent)
    thresholds["borderline_min"] = int(min_percent)
    out["scan_thresholds"] = thresholds
    return out


def _register_scan_ground_truth(scan_id: str, ground_truth: Dict[str, object]) -> None:
    token = str(scan_id or "").strip()
    if not token:
        return
    with _scan_ground_truth_lock:
        _scan_ground_truth_by_id[token] = dict(ground_truth)


def _get_scan_ground_truth(scan_id: Optional[str]) -> Optional[Dict[str, object]]:
    token = str(scan_id or "").strip()
    if not token:
        return None
    with _scan_ground_truth_lock:
        cached = _scan_ground_truth_by_id.get(token)
    return dict(cached) if cached else None


def _clear_scan_ground_truth(scan_id: Optional[str]) -> None:
    token = str(scan_id or "").strip()
    if not token:
        return
    with _scan_ground_truth_lock:
        _scan_ground_truth_by_id.pop(token, None)


def extract_target_profile_ground_truth(
    target_face_path: Union[str, Path],
) -> Dict[str, object]:
    """
    Analyze the uploaded target reference photo for optional baseline metadata.

    Per-frame CCTV demographics use temporal smoothing — not hardcoded overrides.
    """
    path = Path(target_face_path)
    profile: Dict[str, object] = {"source": "target_reference_model"}
    if not path.is_file():
        return profile

    initialize_age_gender_models()
    age, gender = _classify_target_reference_image(path)
    if _demographic_value_known(gender):
        profile["gender"] = gender
        profile["target_gender_ground_truth"] = gender
        print(f"[TACTICAL AI] Target reference gender (model): {gender}")
    if _demographic_value_known(age):
        profile["age"] = age
        profile["target_age_ground_truth"] = age
        profile["age_bucket"] = age
        profile["age_group"] = age
        print(f"[TACTICAL AI] Target reference age (model): {age}")
    return profile


def _apply_target_ground_truth_to_demographics(
    payload: Dict[str, object],
    ground_truth: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Apply optional reference-photo demographics to non-target rows only."""
    if not isinstance(payload, dict):
        return payload
    if _is_target_match_hit(payload):
        return payload
    if ground_truth:
        out = dict(payload)
        gt_gender = ground_truth.get("gender") or ground_truth.get(
            "target_gender_ground_truth"
        )
        if _demographic_value_known(gt_gender):
            out["gender"] = str(gt_gender)
            out["predicted_gender"] = str(gt_gender)
        gt_age = ground_truth.get("age") or ground_truth.get("target_age_ground_truth")
        if _demographic_value_known(gt_age):
            out["age"] = str(gt_age)
            out["predicted_age"] = str(gt_age)
            out["age_bucket"] = str(
                ground_truth.get("age_bucket") or ground_truth.get("age_group") or gt_age
            )
            out["age_group"] = out["age_bucket"]
        return out
    return payload


def _demographics_dict_from_crop(
    face_crop: np.ndarray,
    *,
    scan_id: Optional[str] = None,
) -> Dict[str, object]:
    """Run native model inference on a face crop and return temporally smoothed demographics."""
    age_bucket, gender, age_years = _analyze_face_demographics(face_crop)
    if age_years is not None and age_years > 0 and _demographic_value_known(age_bucket):
        raw_age = age_bucket
    elif age_years is not None and age_years > 0:
        raw_age = str(age_years)
    else:
        raw_age = age_bucket
    return _push_and_smooth_demographics(
        scan_id,
        raw_gender=gender,
        raw_age=raw_age,
    )


def _scan_frame_demographics_callback(
    *,
    scan_id: Optional[str] = None,
):
    """Orion scan hook — native per-frame demographics stabilized by majority vote."""

    def _apply(face_crop: np.ndarray) -> Dict[str, object]:
        return _demographics_dict_from_crop(face_crop, scan_id=scan_id)

    return _apply


def _extract_face_crop_from_frame(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0:
        return None
    try:
        from identity import _crop_largest_face_with_padding

        crop = _crop_largest_face_with_padding(frame)
        if crop is not None and crop.size > 0:
            return crop
    except Exception:
        pass
    h, w = frame.shape[:2]
    side = min(h, w)
    if side < 32:
        return None
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return frame[y0 : y0 + side, x0 : x0 + side]


def _read_video_frame_bgr(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ok, frame = cap.read()
        return frame if ok and frame is not None else None
    finally:
        cap.release()


def _resolve_hit_source_frame(
    hit: Dict,
    *,
    video_path: Optional[Path] = None,
    folder_path: Optional[Path] = None,
) -> Optional[np.ndarray]:
    for key in ("full_frame", "scene_image", "video_frame", "image", "thumbnail"):
        frame = _decode_data_url_image(hit.get(key))
        if frame is not None:
            return frame

    for key in ("thumbnail_url", "capture_url", "image_url", "image_path"):
        url = hit.get(key)
        if not isinstance(url, str) or not url.strip():
            continue
        local = _local_capture_path(url)
        if local is not None and local.is_file():
            img = cv2.imread(str(local))
            if img is not None and img.size > 0:
                return img

    frame_idx = hit.get("frame")
    if frame_idx is None:
        return None

    candidates: List[Path] = []
    if video_path and video_path.is_file():
        candidates.append(video_path)
    if folder_path and folder_path.is_dir():
        vf = hit.get("video_file") or hit.get("original_filename")
        if vf:
            candidates.append(folder_path / Path(str(vf)).name)

    for path in candidates:
        frame = _read_video_frame_bgr(path, int(frame_idx))
        if frame is not None:
            return frame
    return None


def _apply_age_gender_to_hit(
    hit: Dict,
    age: str,
    gender: str,
    *,
    age_years: Optional[int] = None,
) -> Dict:
    out = dict(hit)
    if age and age != FACE_ATTR_UNKNOWN:
        out["age_bucket"] = age
        out["age_group"] = age
        if age_years is not None and age_years > 0:
            out["age"] = str(age_years)
        else:
            out["age"] = age
    if gender and gender != FACE_ATTR_UNKNOWN:
        out["gender"] = gender
    return out


def _classify_target_reference_image(image_path: Path) -> Tuple[str, str]:
    img = cv2.imread(str(image_path))
    if img is None or img.size == 0:
        return FACE_ATTR_UNKNOWN, FACE_ATTR_UNKNOWN
    crop = _extract_face_crop_from_frame(img)
    if crop is None:
        return FACE_ATTR_UNKNOWN, FACE_ATTR_UNKNOWN
    age_bucket, gender, age_years = classify_face_attributes_detailed(crop)
    if age_years is not None and age_years > 0 and age_bucket != FACE_ATTR_UNKNOWN:
        return str(age_years), gender
    return age_bucket, gender


def _hit_needs_demographic_enrichment(hit: Dict) -> bool:
    age_text = str(hit.get("age") or "").strip()
    gender_text = str(hit.get("gender") or "").strip()
    age_missing = not age_text or age_text.lower() in ("unknown", "n/a")
    gender_missing = not gender_text or gender_text.lower() in ("unknown", "n/a")
    return age_missing or gender_missing


def _enrich_detection_hits_with_attributes(
    hits: List,
    *,
    video_path: Optional[Path] = None,
    folder_path: Optional[Path] = None,
    frame_cache: Optional[Dict] = None,
    ground_truth: Optional[Dict[str, object]] = None,
    scan_id: Optional[str] = None,
) -> List:
    if not hits:
        return hits

    cache = frame_cache if frame_cache is not None else {}
    enriched: List = []
    processed = 0
    work_hits = list(hits)
    # Enrich strongest matches first; remaining hits skip slow DeepFace re-analysis.
    work_hits.sort(
        key=lambda h: (
            -_extract_hit_match_percentage(h) if isinstance(h, dict) else 0,
            str(h.get("video_file") or h.get("original_filename") or "")
            if isinstance(h, dict)
            else "",
            int(h.get("frame") or -1) if isinstance(h, dict) else -1,
        )
    )

    for hit in work_hits:
        if not isinstance(hit, dict):
            enriched.append(hit)
            continue

        match_percentage = _extract_hit_match_percentage(hit)
        if match_percentage < SCAN_DISPLAY_MIN_MATCH_PERCENT:
            continue

        if _is_target_match_hit(hit) and not _hit_needs_demographic_enrichment(hit):
            enriched.append(hit)
            continue

        if not _is_target_match_hit(hit) and not _hit_needs_demographic_enrichment(hit):
            enriched.append(_apply_target_ground_truth_to_demographics(hit, ground_truth))
            continue

        if processed >= FACE_ATTR_MAX_ENRICH_HITS:
            if _is_target_match_hit(hit):
                enriched.append(hit)
            else:
                enriched.append(_apply_target_ground_truth_to_demographics(hit, ground_truth))
            continue

        frame_idx = hit.get("frame")
        video_key = str(hit.get("video_file") or hit.get("original_filename") or "")
        cache_key = (video_key, int(frame_idx) if frame_idx is not None else -1)

        age = FACE_ATTR_UNKNOWN
        gender = FACE_ATTR_UNKNOWN
        age_years: Optional[int] = None
        if cache_key in cache:
            cached = cache[cache_key]
            if len(cached) == 3:
                age, gender, age_years = cached
            else:
                age, gender = cached[0], cached[1]
        else:
            try:
                face_box = hit.get("face_box") or hit.get("bbox")
                source = _resolve_hit_source_frame(
                    hit, video_path=video_path, folder_path=folder_path
                )
                crop = None
                if source is not None and face_box:
                    crop = _crop_face_from_bbox(source, face_box)
                if crop is None and source is not None:
                    crop = _extract_face_crop_from_frame(source)
                if crop is not None:
                    age, gender, age_years = _analyze_face_demographics(crop)
                    cache[cache_key] = (age, gender, age_years)
                    processed += 1
            except Exception as exc:
                print(f"[WARN] Demographics enrichment skipped for hit frame {frame_idx}: {exc}")

        if _is_target_match_hit(hit):
            if age_years is not None and age_years > 0 and _demographic_value_known(age):
                raw_age = age
            elif age_years is not None and age_years > 0:
                raw_age = str(age_years)
            else:
                raw_age = age
            enriched.append(
                _stabilize_target_match_demographics(
                    hit,
                    scan_id=scan_id,
                    raw_gender=gender,
                    raw_age=raw_age,
                )
            )
            continue

        merged = _apply_age_gender_to_hit(hit, age, gender, age_years=age_years)
        enriched.append(_apply_target_ground_truth_to_demographics(merged, ground_truth))

    return enriched


def _apply_ground_truth_to_result_hits(
    result: dict,
    ground_truth: Optional[Dict[str, object]] = None,
) -> dict:
    """Apply optional reference-photo demographics to scan hit lists (non-target rows)."""
    if not isinstance(result, dict):
        return result
    out = dict(result)
    for key in ("matches", "borderline_suspects", "near_miss_suspects", "all_matches"):
        items = out.get(key)
        if not isinstance(items, list) or not items:
            continue
        out[key] = [
            _apply_target_ground_truth_to_demographics(item, ground_truth)
            if isinstance(item, dict)
            else item
            for item in items
        ]
    return out


def enrich_scan_face_attributes(
    result: dict,
    *,
    target_image_path: Optional[Path] = None,
    video_path: Optional[Path] = None,
    folder_path: Optional[Path] = None,
    ground_truth: Optional[Dict[str, object]] = None,
    scan_id: Optional[str] = None,
) -> dict:
    """Attach age/gender buckets to scan hits and top-level scan metadata."""
    if not isinstance(result, dict):
        return result

    if ground_truth is None and target_image_path and target_image_path.is_file():
        ground_truth = extract_target_profile_ground_truth(target_image_path)

    initialize_age_gender_models()
    out = dict(result)
    frame_cache: Dict = {}
    scan_id = str(scan_id or out.get("scan_id") or "").strip() or None

    for key in ("matches", "borderline_suspects", "near_miss_suspects", "all_matches"):
        items = out.get(key)
        if isinstance(items, list) and items:
            out[key] = _enrich_detection_hits_with_attributes(
                items,
                video_path=video_path,
                folder_path=folder_path,
                frame_cache=frame_cache,
                ground_truth=ground_truth,
                scan_id=scan_id,
            )

    smoothed = _peek_smoothed_demographics(scan_id)
    target_age = str(smoothed.get("age") or FACE_ATTR_UNKNOWN)
    target_gender = str(smoothed.get("gender") or FACE_ATTR_UNKNOWN)
    if ground_truth:
        if target_age == FACE_ATTR_UNKNOWN and ground_truth.get("age"):
            target_age = str(ground_truth["age"])
        if target_gender == FACE_ATTR_UNKNOWN and ground_truth.get("gender"):
            target_gender = str(ground_truth["gender"])

    best_hit = None
    for key in ("matches", "borderline_suspects", "all_matches"):
        items = out.get(key) or []
        if items:
            best_hit = items[0]
            break

    if best_hit:
        if target_age == FACE_ATTR_UNKNOWN and best_hit.get("age"):
            target_age = str(best_hit.get("age"))
        if target_gender == FACE_ATTR_UNKNOWN and best_hit.get("gender"):
            target_gender = str(best_hit.get("gender"))

    if target_age != FACE_ATTR_UNKNOWN:
        out["age"] = target_age
        out["predicted_age"] = target_age
        out["target_age"] = target_age
        age_bucket = str(smoothed.get("age_bucket") or target_age)
        out["age_bucket"] = age_bucket
        out["age_group"] = age_bucket
    if target_gender != FACE_ATTR_UNKNOWN:
        out["gender"] = target_gender
        out["predicted_gender"] = target_gender
        out["target_gender"] = target_gender

    out = _apply_ground_truth_to_result_hits(out, ground_truth)

    out["face_attributes_backend"] = _resolve_face_attribute_backend()
    return out


def enrich_target_detail_payload(hit: dict, **kwargs) -> dict:
    """Enrich a single detection record for the target detail modal / logs."""
    if not isinstance(hit, dict):
        return hit
    items = _enrich_detection_hits_with_attributes([hit], **kwargs)
    return items[0] if items else hit


def _enrich_recent_logs_face_attributes(
    conn,
    rows: List[Dict],
    *,
    max_rows: int = 8,
) -> List[Dict]:
    """Fill missing age/gender on recent tracking logs from capture thumbnails."""
    if not rows:
        return rows

    initialize_age_gender_models()
    enriched: List[Dict] = []
    updated = 0

    for row in rows:
        item = dict(row)
        age_text = str(item.get("age") or "")
        gender_text = str(item.get("gender") or "")
        needs_attrs = (
            not age_text
            or age_text.lower() in ("unknown", "n/a")
            or not gender_text
            or gender_text.lower() in ("unknown", "n/a")
        )
        if needs_attrs and updated < max_rows:
            capture_url = item.get("capture_url") or ""
            capture_name = Path(str(capture_url).split("/captures/")[-1]).name
            capture_path = CAPTURES_DIR / capture_name
            if capture_path.is_file():
                try:
                    frame = cv2.imread(str(capture_path))
                    crop = _extract_face_crop_from_frame(frame) if frame is not None else None
                    if crop is not None:
                        age_bucket, gender, age_years = classify_face_attributes_detailed(crop)
                        if age_years is not None and age_years > 0:
                            item["age"] = str(age_years)
                            if age_bucket != FACE_ATTR_UNKNOWN:
                                item["age_bucket"] = age_bucket
                                item["age_group"] = age_bucket
                        elif age_bucket != FACE_ATTR_UNKNOWN:
                            item["age"] = age_bucket
                            item["age_bucket"] = age_bucket
                            item["age_group"] = age_bucket
                        if gender != FACE_ATTR_UNKNOWN:
                            item["gender"] = gender
                        updated += 1
                except Exception:
                    pass
        enriched.append(item)

    return enriched


def _assert_video_readable(video_path: Path) -> None:
    """Verify OpenCV can open the file and decode at least one frame."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise VideoCodecError("Could not open video file.")

        ok, frame = cap.read()
        if ok and frame is not None:
            return

        raise VideoCodecError(
            "Video opened but no frames could be decoded (unsupported codec)."
        )
    finally:
        cap.release()


def save_uploaded_video(upload_file: FileStorage, dest_path: Path) -> None:
    """Save an uploaded video and ensure the codec is decodable by OpenCV."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    upload_file.save(str(dest_path))
    try:
        _assert_video_readable(dest_path)
    except VideoCodecError:
        if dest_path.is_file():
            try:
                dest_path.unlink()
            except OSError:
                pass
        raise


SCAN_THUMBS_DIR = CAPTURES_DIR / "scan_hits"
DEVICE_PROBE_DIR = CAPTURES_DIR / "device_probe"
SNAPSHOTS_STATIC_DIR = PROJECT_DIR / "static" / "snapshots"
INFRACTIONS_STATIC_DIR = PROJECT_DIR / "static" / "infractions"
LIVE_PREVIEW_DIR = PROJECT_DIR / "live_preview"
STILLS_DIR = PROJECT_DIR / "stills"
RUNS_DIR = PROJECT_DIR / "runs"

SNAPSHOTS_STATIC_DIR.mkdir(parents=True, exist_ok=True)
INFRACTIONS_STATIC_DIR.mkdir(parents=True, exist_ok=True)
LIVE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
SCAN_API_MAX_HITS_PER_LIST = 64
_SCAN_HIT_BLOB_KEYS = frozenset(
    {"thumbnail", "image", "full_frame", "scene_image", "video_frame", "embedding"}
)


def _capture_public_url(relative_path: str) -> str:
    """Public URL for a file under CAPTURES_DIR (served by /captures/<path>)."""
    rel = str(relative_path or "").replace("\\", "/").strip().lstrip("/")
    if rel.startswith("captures/"):
        rel = rel[len("captures/") :]
    if rel.startswith("static/captures/"):
        rel = rel[len("static/captures/") :]
    return f"/captures/{rel}"


def _local_capture_path(url_or_rel: str) -> Optional[Path]:
    """Map /captures/... or scan_hits/foo.jpg to a path under CAPTURES_DIR."""
    raw = str(url_or_rel or "").strip()
    if not raw or raw.startswith("data:") or raw.startswith("http"):
        return None
    for prefix in ("/static/captures/", "/captures/", "static/captures/", "captures/"):
        if raw.startswith(prefix):
            rel = raw[len(prefix) :].lstrip("/\\")
            return CAPTURES_DIR / rel.replace("/", os.sep)
    if not raw.startswith("/"):
        return CAPTURES_DIR / raw.replace("/", os.sep)
    return None


def _json_safe(value):
    """Strip numpy / non-finite values so Flask jsonify cannot break the client parser."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if not np.isfinite(v):
            return None
        return v
    return value


def _persist_scan_hit_thumbnail(hit: Dict, session_id: str, idx: int) -> Optional[str]:
    raw: Optional[bytes] = None
    for key in ("thumbnail", "image", "full_frame", "scene_image", "video_frame"):
        thumb = hit.get(key)
        if isinstance(thumb, str) and thumb.startswith("data:"):
            raw = _decode_data_url_bytes(thumb)
            if raw:
                break
    if raw is None:
        for key in ("thumbnail_url", "capture_url", "image_url", "image_path", "thumbnail"):
            existing = hit.get(key)
            if not isinstance(existing, str) or not existing.strip():
                continue
            local = _local_capture_path(existing)
            if local is not None and local.is_file():
                try:
                    return str(local.relative_to(CAPTURES_DIR)).replace("\\", "/")
                except ValueError:
                    pass
            if existing.startswith("data:"):
                raw = _decode_data_url_image(existing)
                if raw:
                    break
    if raw is None:
        return None

    SCAN_THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    frame = hit.get("frame", idx)
    video_slug = Path(str(hit.get("video_file") or "clip")).stem.replace(" ", "_")[:40]
    filename = f"{session_id}_{video_slug}_f{frame}_{idx}.jpg"
    out_path = SCAN_THUMBS_DIR / filename
    try:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None and img.size > 0:
            if not cv2.imwrite(str(out_path), img):
                out_path.write_bytes(raw)
        else:
            out_path.write_bytes(raw)
    except OSError:
        return None
    return f"scan_hits/{filename}"


def _resolve_hit_thumbnail_url(hit: Dict) -> str:
    """Normalize thumbnail fields on a scan hit to a working /captures/ URL."""
    if not isinstance(hit, dict):
        return ""
    for key in (
        "thumbnail_url",
        "capture_url",
        "image_url",
        "preview_url",
        "thumb_url",
        "image_path",
        "face_crop_path",
    ):
        val = hit.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        trimmed = val.strip()
        if trimmed.startswith("data:"):
            continue
        if trimmed.startswith("/static/captures/"):
            return _capture_public_url(trimmed[len("/static/captures/") :])
        if trimmed.startswith("/captures/"):
            return trimmed
        if trimmed.startswith("captures/"):
            return f"/{trimmed}"
        if trimmed.startswith("/"):
            return trimmed
        return _capture_public_url(trimmed)
    for key in ("thumbnail", "image"):
        val = hit.get(key)
        if isinstance(val, str) and val.strip() and not val.startswith("data:"):
            return _capture_public_url(val)
    return ""


def _slim_scan_hit_for_api(hit: Dict, session_id: str, idx: int) -> Dict:
    if not isinstance(hit, dict):
        return {}
    saved = _persist_scan_hit_thumbnail(hit, session_id, idx)
    out = {k: v for k, v in hit.items() if k not in _SCAN_HIT_BLOB_KEYS}
    thumb_url = _resolve_hit_thumbnail_url(hit)
    if saved:
        thumb_url = _capture_public_url(saved)
    if thumb_url:
        out["thumbnail_url"] = thumb_url
        out["capture_url"] = thumb_url
        out["image_url"] = thumb_url
    return out


def _pack_scan_result_row(hit: Dict, idx: int) -> Optional[Dict]:
    if not _passes_scan_display_threshold(hit):
        return None
    pct = int(
        hit.get("match_percent")
        or hit.get("confidence_score")
        or hit.get("confidence")
        or 0
    )
    video_file = str(
        hit.get("original_filename")
        or hit.get("video_file")
        or hit.get("filename")
        or "Unknown"
    )
    thumb = _resolve_hit_thumbnail_url(hit)
    row = {
        "thumbnail_url": thumb,
        "capture_url": thumb,
        "image_url": thumb,
        "confidence": pct,
        "match_percent": pct,
        "confidence_score": pct,
        "video_file": video_file,
        "original_filename": video_file,
        "status": "Confirmed" if pct >= VIDEO_TARGET_CONFIRM_MATCH_PERCENT else "Borderline",
        "timestamp": str(hit.get("timestamp") or "00:00"),
        "frame": hit.get("frame"),
        "scan_index": idx,
    }
    for attr_key in (
        "age",
        "gender",
        "predicted_age",
        "predicted_gender",
        "age_bucket",
        "age_group",
        "target_age",
        "target_gender",
    ):
        if hit.get(attr_key) not in (None, "", FACE_ATTR_UNKNOWN):
            row[attr_key] = hit.get(attr_key)
    if hit.get("distance") is not None:
        row["distance"] = hit.get("distance")
    return row


def prepare_scan_api_payload(result: Dict, session_id: str) -> Dict:
    """
    Persist thumbnails to disk and strip multi-MB base64 blobs before JSON.
    Large inline scene JPEGs per hit were breaking response.json() in the browser.
    """
    if not isinstance(result, dict):
        return {}
    out = _filter_scan_results_for_display(dict(result))
    idx = 0
    for key in (
        "matches",
        "borderline_suspects",
        "near_miss_suspects",
        "all_matches",
        "confirmed_matches",
    ):
        items = out.get(key)
        if not isinstance(items, list) or not items:
            continue
        items = _filter_scan_hit_list(items)
        out[key] = items
        total = len(items)
        slim = [
            _slim_scan_hit_for_api(h, session_id, idx + i)
            for i, h in enumerate(items[:SCAN_API_MAX_HITS_PER_LIST])
        ]
        idx += len(slim)
        out[key] = slim
        if total > SCAN_API_MAX_HITS_PER_LIST:
            out[f"{key}_total"] = total

    scan_results: List[Dict] = []
    seen = set()
    row_idx = 0
    for key in ("matches", "borderline_suspects", "near_miss_suspects"):
        for hit in out.get(key) or []:
            row = _pack_scan_result_row(hit, row_idx)
            if row is None:
                continue
            dedupe = (row["video_file"], row["timestamp"], row["confidence"], row.get("frame"))
            if dedupe in seen:
                continue
            seen.add(dedupe)
            scan_results.append(row)
            row_idx += 1
    out["scan_results"] = scan_results
    if scan_results:
        # Keep registry / visual panels in sync — same rows, full thumbnail URLs.
        out["detections"] = scan_results
        if not out.get("matches"):
            out["matches"] = list(scan_results)
        if not out.get("all_matches"):
            out["all_matches"] = list(scan_results)
    out["registry_synced"] = bool(scan_results)
    for row in scan_results:
        _record_tactical_target_match(
            scan_id=session_id,
            frame=row.get("frame"),
            video_file=row.get("video_file") or row.get("original_filename"),
        )
    return _json_safe(out)


def _video_upload_error_response(exc: Exception) -> tuple:
    if isinstance(exc, VideoCodecError):
        return jsonify({"error": VIDEO_CODEC_USER_MESSAGE}), 400

    msg = str(exc).lower()
    if any(
        token in msg
        for token in (
            "codec",
            "hevc",
            "h265",
            "h.265",
            "encoding",
            "unsupported",
            "ffmpeg",
        )
    ):
        return jsonify({"error": VIDEO_CODEC_USER_MESSAGE}), 400

    return jsonify({"error": f"Video processing failed: {exc}"}), 500


@app.after_request
def no_cache_html(response):
    if response.content_type and "text/html" in response.content_type:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.route("/login", methods=["GET", "POST"])
def login():
    if is_logged_in():
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        if username == ADMIN_USERNAME and passwords_match(password, ADMIN_PASSWORD):
            session.clear()
            session.permanent = True
            session["logged_in"] = True
            session["username"] = username
            next_url = (
                request.args.get("next")
                or request.form.get("next")
                or url_for("dashboard")
            )
            if not next_url.startswith("/") or next_url.startswith("//"):
                next_url = url_for("dashboard")
            return redirect(next_url)
        error = "Invalid credentials. Access denied."
    return render_template(
        "login.html",
        error=error,
        dashboard_version=DASHBOARD_UI_VERSION,
        next_url=request.args.get("next") or "",
    )


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    try:
        from announcement_settings import (
            announcement_schedule_status,
            load_announcement_settings,
        )

        announcement_bootstrap = {
            "settings": load_announcement_settings(use_cache=False),
            "status": announcement_schedule_status(),
        }
    except Exception:
        announcement_bootstrap = {
            "settings": {
                "schedule_enabled": True,
                "start_time": "08:00",
                "end_time": "18:00",
            },
            "status": {"label": "8 AM – 6 PM", "active_now": True, "schedule_enabled": True},
        }
    return render_template(
        "index.html",
        dashboard_version=DASHBOARD_UI_VERSION,
        announcement_bootstrap=announcement_bootstrap,
    )


@app.route("/dashboard")
def dashboard():
    """Redirect to the main SPA with the dashboard panel active."""
    return redirect(dashboard_home_url())


@app.route("/classic")
def classic_dashboard():
    """Legacy multi-panel dashboard (search, video registry, zone matrix)."""
    return render_template(
        "dashboard.html",
        panel_theme_css=build_panel_theme_css(),
        dashboard_version=DASHBOARD_UI_VERSION,
    )


def _metrics_payload() -> dict:
    if is_dashboard_scan_hold_active():
        payload = build_scan_hold_tactical_metrics()
    else:
        payload = fetch_tactical_metrics(db_path=get_db_path())
    return _inject_system_status_into_metrics(payload)


def _ensure_metrics_emitter() -> None:
    """Background task: push tactical metrics to all /ws/dashboard_metrics clients."""
    global _metrics_emitter_started
    _ensure_detection_counters_bootstrap()
    with _metrics_emitter_lock:
        if _metrics_emitter_started:
            return
        _metrics_emitter_started = True

    def _emit_loop() -> None:
        while True:
            try:
                socketio.emit(
                    "dashboard_metrics",
                    _metrics_payload(),
                    namespace=METRICS_WS_NAMESPACE,
                )
            except Exception:
                pass
            socketio.sleep(2.0)

    socketio.start_background_task(_emit_loop)


@socketio.on("connect", namespace=METRICS_WS_NAMESPACE)
def metrics_ws_connect() -> None:
    if not is_logged_in():
        return False
    _reset_session_tracking_metrics()
    _ensure_metrics_emitter()
    _ensure_person_alert_emitter()
    emit("dashboard_metrics", _metrics_payload())


@socketio.on("trigger_system_clear_reset", namespace=METRICS_WS_NAMESPACE)
def trigger_system_clear_reset(_data=None) -> None:
    """Manual session wipe — triggered by the RESET SESSION button on the command center."""
    if not is_logged_in():
        return
    _reset_session_tracking_metrics()
    payload = _metrics_payload()
    emit("dashboard_metrics", payload)
    emit(
        "system_reset_complete",
        {
            "ok": True,
            "message": "Session metrics and dynamic lists cleared.",
            "system_status": _build_system_status_payload(),
            "cards": (payload.get("cards") or {}),
        },
    )


@socketio.on("disconnect", namespace=METRICS_WS_NAMESPACE)
def metrics_ws_disconnect() -> None:
    pass


# ── Instant Person-of-Interest / watchlist alerts (WebSocket) ────────────────
_person_alert_emitter_started = False
_person_alert_emitter_lock = threading.Lock()
_recent_alert_ids: Set[str] = set()
_RECENT_ALERT_IDS_MAX = 200


def _normalize_person_alert_payload(data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    crop = str(data.get("face_crop_url") or data.get("crop_url") or "").strip()
    camera = str(data.get("camera") or "").strip()
    if not crop or not camera:
        return None
    if data.get("is_named") in (True, 1, "1"):
        return None
    try:
        confidence = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    alert_type = str(data.get("alert_type") or "").strip().lower()
    is_stranger = data.get("is_stranger") in (True, 1, "1") or alert_type == "stranger"
    if is_stranger:
        name = "Stranger"
        alert_id = str(data.get("id") or uuid.uuid4().hex[:12])
        return {
            "event": "new_person_alert",
            "id": alert_id,
            "camera": camera,
            "name": name,
            "confidence": round(confidence, 1),
            "timestamp": str(data.get("timestamp") or ""),
            "iso_timestamp": data.get("iso_timestamp"),
            "face_crop_url": crop,
            "person_id": data.get("person_id"),
            "known_person_id": data.get("known_person_id"),
            "bbox": data.get("bbox"),
            "is_poi": False,
            "is_stranger": True,
            "is_named": False,
            "is_watchlist": True,
            "alert_type": "stranger",
            "blacklist": False,
        }
    if data.get("is_watchlist") in (False, 0, "0"):
        return None
    name = str(data.get("name") or "Person of Interest").strip()
    if (
        name.startswith("Passerby_")
        or name.startswith("Stranger_")
        or name.startswith("Unknown")
        or name in ("Person of Interest", "Unknown Visitor", "Stranger")
    ):
        return None
    if data.get("known_person_id") is None:
        return None
    blacklist = bool(data.get("blacklist")) or alert_type == "blacklist" or confidence >= 85.0
    if not alert_type:
        alert_type = "blacklist" if blacklist else "poi"
    alert_id = str(data.get("id") or uuid.uuid4().hex[:12])
    return {
        "event": "new_person_alert",
        "id": alert_id,
        "camera": camera,
        "name": name,
        "confidence": round(confidence, 1),
        "timestamp": str(data.get("timestamp") or ""),
        "iso_timestamp": data.get("iso_timestamp"),
        "face_crop_url": crop,
        "person_id": data.get("person_id"),
        "known_person_id": data.get("known_person_id"),
        "bbox": data.get("bbox"),
        "is_poi": True,
        "is_stranger": False,
        "is_named": False,
        "is_watchlist": True,
        "alert_type": alert_type,
        "blacklist": blacklist,
    }


def emit_new_person_alert(payload: Dict[str, Any]) -> bool:
    """Broadcast ``new_person_alert`` to all connected dashboard clients."""
    event = _normalize_person_alert_payload(payload)
    if not event:
        return False
    alert_id = str(event["id"])
    if alert_id in _recent_alert_ids:
        return False
    _recent_alert_ids.add(alert_id)
    if len(_recent_alert_ids) > _RECENT_ALERT_IDS_MAX:
        # Drop arbitrary old ids (set has no order); rebuild from a slice via list.
        trimmed = list(_recent_alert_ids)[-(_RECENT_ALERT_IDS_MAX // 2) :]
        _recent_alert_ids.clear()
        _recent_alert_ids.update(trimmed)
    try:
        socketio.emit("new_person_alert", event, namespace=ALERTS_WS_NAMESPACE)
        # Also fan out on metrics namespace for clients already connected there.
        socketio.emit("new_person_alert", event, namespace=METRICS_WS_NAMESPACE)
        return True
    except Exception:
        return False


def _ensure_person_alert_emitter() -> None:
    """Fallback tailer: emit alerts written by tracker if HTTP push is missed."""
    global _person_alert_emitter_started
    with _person_alert_emitter_lock:
        if _person_alert_emitter_started:
            return
        _person_alert_emitter_started = True

    def _emit_loop() -> None:
        from live_detection_hub import ALERTS_PATH, ensure_live_detection_dirs, read_alerts_since

        ensure_live_detection_dirs()
        try:
            offset = ALERTS_PATH.stat().st_size if ALERTS_PATH.is_file() else 0
        except OSError:
            offset = 0
        while True:
            try:
                events, offset = read_alerts_since(offset, limit=40)
                for event in events:
                    emit_new_person_alert(event)
            except Exception:
                pass
            socketio.sleep(0.05)

    socketio.start_background_task(_emit_loop)


@socketio.on("connect", namespace=ALERTS_WS_NAMESPACE)
def alerts_ws_connect() -> None:
    if not is_logged_in():
        return False
    _ensure_person_alert_emitter()
    emit("alerts_ready", {"ok": True, "event": "new_person_alert"})


@socketio.on("disconnect", namespace=ALERTS_WS_NAMESPACE)
def alerts_ws_disconnect() -> None:
    pass


@app.route("/api/internal/person_alert", methods=["POST"])
def internal_person_alert():
    """
    Localhost push from tracker → instant Socket.IO ``new_person_alert``.
    Auth: loopback remote addr + shared internal token.
    """
    remote = (request.remote_addr or "").strip()
    if remote not in ("127.0.0.1", "::1", "localhost"):
        return jsonify({"error": "forbidden"}), 403
    expected = str(
        os.environ.get("INTERNAL_ALERT_TOKEN") or INTERNAL_ALERT_TOKEN or ""
    ).strip()
    provided = str(request.headers.get("X-Internal-Alert-Token") or "").strip()
    if not expected or not passwords_match(provided, expected):
        return jsonify({"error": "forbidden"}), 403
    payload = request.get_json(silent=True) or {}
    ok = emit_new_person_alert(payload)
    _ensure_person_alert_emitter()
    return jsonify({"ok": ok}), (200 if ok else 400)


# ── Low-latency live video (JPEG over Socket.IO → browser MediaStream) ───────
_live_video_rooms: Dict[str, int] = {}
_live_video_lock = threading.Lock()
_live_video_emitter_started = False


def _live_video_room(mode: str, camera_key: str, quality: str = "full") -> str:
    mode_key = "face" if str(mode).strip().lower() in ("face", "face_detect", "detect") else "raw"
    cam = (camera_key or "1").strip() or "1"
    q = "low" if str(quality or "").strip().lower() in ("low", "lq", "preview") else "full"
    return f"{mode_key}:{cam}:{q}"


def _parse_live_video_room(room: str) -> Tuple[str, str, str]:
    parts = str(room or "").split(":")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], "full"
    return "raw", "1", "full"


def _encode_live_camera_jpeg(
    camera_key: str,
    *,
    face_boxes: bool = False,
    quality: str = "full",
) -> Optional[bytes]:
    """Latest hub JPEG for one camera; optionally draw green face overlays / downscale."""
    jpeg = get_camera_jpeg(camera_key)
    if not jpeg:
        return None
    want_low = str(quality or "").strip().lower() == "low"
    if not face_boxes and not want_low:
        return jpeg
    try:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jpeg
        if face_boxes:
            frame = _draw_green_face_boxes(frame, camera_key)
        if want_low:
            tw, th = LIVE_VIDEO_LOW_SIZE
            frame = _crop_resize_cover(frame, tw, th)
        jq = LIVE_VIDEO_LOW_JPEG_QUALITY if want_low else MJPEG_JPEG_QUALITY
        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), jq],
        )
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    return jpeg


def _ensure_live_video_emitter() -> None:
    global _live_video_emitter_started
    with _live_video_lock:
        if _live_video_emitter_started:
            return
        _live_video_emitter_started = True

    interval = 1.0 / max(5.0, float(LIVE_VIDEO_FPS))

    def _emit_loop() -> None:
        while True:
            try:
                with _live_video_lock:
                    rooms = [room for room, n in _live_video_rooms.items() if n > 0]
                for room in rooms:
                    mode, camera_key, quality = _parse_live_video_room(room)
                    jpeg = _encode_live_camera_jpeg(
                        camera_key,
                        face_boxes=(mode == "face"),
                        quality=quality,
                    )
                    if not jpeg:
                        jpeg = _placeholder_jpeg(f"{camera_key} - waiting for signal...")
                    if jpeg:
                        socketio.emit(
                            "frame",
                            jpeg,
                            room=room,
                            namespace=LIVE_VIDEO_WS_NAMESPACE,
                        )
            except Exception:
                pass
            socketio.sleep(interval)

    socketio.start_background_task(_emit_loop)


@socketio.on("connect", namespace=LIVE_VIDEO_WS_NAMESPACE)
def live_video_connect() -> None:
    if not is_logged_in():
        return False
    _ensure_live_video_emitter()


@socketio.on("subscribe", namespace=LIVE_VIDEO_WS_NAMESPACE)
def live_video_subscribe(data=None) -> None:
    if not is_logged_in():
        return
    payload = data if isinstance(data, dict) else {}
    room = _live_video_room(
        payload.get("mode", "raw"),
        payload.get("camera", "1"),
        payload.get("quality", "full"),
    )
    join_room(room)
    with _live_video_lock:
        _live_video_rooms[room] = _live_video_rooms.get(room, 0) + 1
    # Immediate first frame so the player does not wait a full tick.
    try:
        mode, camera_key, quality = _parse_live_video_room(room)
        jpeg = _encode_live_camera_jpeg(
            camera_key,
            face_boxes=(mode == "face"),
            quality=quality,
        )
        if jpeg:
            emit("frame", jpeg)
    except Exception:
        pass
    emit("subscribed", {"room": room, "ok": True})


@socketio.on("unsubscribe", namespace=LIVE_VIDEO_WS_NAMESPACE)
def live_video_unsubscribe(data=None) -> None:
    payload = data if isinstance(data, dict) else {}
    room = payload.get("room") or _live_video_room(
        payload.get("mode", "raw"),
        payload.get("camera", "1"),
        payload.get("quality", "full"),
    )
    leave_room(room)
    with _live_video_lock:
        count = max(0, _live_video_rooms.get(room, 1) - 1)
        if count:
            _live_video_rooms[room] = count
        else:
            _live_video_rooms.pop(room, None)


@socketio.on("disconnect", namespace=LIVE_VIDEO_WS_NAMESPACE)
def live_video_disconnect() -> None:
    # Room membership is cleared by Socket.IO; counts are best-effort.
    pass


@app.route("/api/tactical_metrics")
def api_tactical_metrics():
    """REST bootstrap for tactical dashboard (WebSocket takes over live updates)."""
    if _should_hold_dashboard_data():
        return _scan_hold_json_response(build_scan_hold_tactical_metrics())
    return jsonify(_metrics_payload())


@app.route("/api/scan_layout_hold", methods=["POST", "GET"])
def api_scan_layout_hold():
    """
    Frontend layout gate — set active=true while the trace-scan loader is visible
    so summary/visitor/capture feeds return blank payloads instead of stale rows.
    """
    if request.method == "GET":
        with _layout_hold_lock:
            layout_active = _layout_hold_active
        with _scan_progress_lock:
            active_scans = sorted(_active_scan_ids)
        return jsonify({
            "ok": True,
            "active": layout_active,
            "scan_hold": is_dashboard_scan_hold_active(),
            "active_scan_ids": active_scans,
        })

    payload = request.get_json(silent=True) or {}
    active = payload.get("active")
    if active is None:
        active = request.form.get("active", "false")
    hold_active = str(active).strip().lower() in ("1", "true", "yes", "on")
    _set_layout_hold(hold_active)
    if hold_active:
        _arm_scan_job_from_hold_payload(payload)
    with _layout_hold_lock:
        layout_active = _layout_hold_active
    return jsonify({
        "ok": True,
        "active": layout_active,
        "scan_hold": is_dashboard_scan_hold_active(),
        "system_status": _build_system_status_payload(),
    })


def _mjpeg_chunk(jpeg_bytes: bytes) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + jpeg_bytes
        + b"\r\n"
    )


def _crop_resize_cover(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop to target aspect ratio, then resize (no letterbox padding)."""
    h, w = frame.shape[:2]
    if h < 1 or w < 1:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    target_aspect = target_w / target_h
    src_aspect = w / h
    if src_aspect > target_aspect:
        crop_w = max(1, int(round(h * target_aspect)))
        x0 = max(0, (w - crop_w) // 2)
        cropped = frame[:, x0 : x0 + crop_w]
    else:
        crop_h = max(1, int(round(w / target_aspect)))
        y0 = max(0, (h - crop_h) // 2)
        cropped = frame[y0 : y0 + crop_h, :]
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _encode_frame_jpeg(frame: np.ndarray, message: Optional[str] = None) -> bytes:
    """Resize to web dimensions and encode as JPEG for MJPEG streaming."""
    w, h = WEB_STREAM_SIZE
    if frame is None or frame.size == 0:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (8, 12, 20)
        if message:
            for i, line in enumerate(message.split("\n")[:3]):
                cv2.putText(
                    frame,
                    line[:48],
                    (24, 200 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (160, 200, 220),
                    1,
                    cv2.LINE_AA,
                )
    else:
        frame = _crop_resize_cover(frame, w, h)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def _placeholder_jpeg(message: str = "Waiting for tracker...") -> bytes:
    return _encode_frame_jpeg(None, message)


def _normalize_hub_jpeg(jpeg_bytes: bytes) -> bytes:
    """Re-encode hub frames at a consistent 16:9 size for the dashboard."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jpeg_bytes
    return _encode_frame_jpeg(frame)


def _fit_grid_tile_cover(frame: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    """Center-crop one camera frame to fill a grid cell edge-to-edge."""
    cell = np.full((cell_h, cell_w, 3), (14, 18, 28), dtype=np.uint8)
    h, w = frame.shape[:2]
    if h < 1 or w < 1:
        return cell
    target_aspect = cell_w / cell_h
    src_aspect = w / h
    if src_aspect > target_aspect:
        crop_w = max(1, int(round(h * target_aspect)))
        x0 = max(0, (w - crop_w) // 2)
        cropped = frame[:, x0 : x0 + crop_w]
    else:
        crop_h = max(1, int(round(w / target_aspect)))
        y0 = max(0, (h - crop_h) // 2)
        cropped = frame[y0 : y0 + crop_h, :]
    return cv2.resize(cropped, (cell_w, cell_h), interpolation=cv2.INTER_AREA)


def _draw_hud_tile_label(tile: np.ndarray, label: str) -> np.ndarray:
    """Overlay a cyan HUD zone badge on a camera tile."""
    out = tile.copy()
    h, w = out.shape[:2]
    bar_h = max(22, int(h * 0.08))
    cv2.rectangle(out, (0, h - bar_h), (w, h), (10, 14, 22), -1)
    cv2.line(out, (0, h - bar_h), (w, h - bar_h), (0, 180, 210), 1, cv2.LINE_AA)
    text = f"ZONE {label}".upper()[:24]
    cv2.putText(
        out,
        text,
        (8, h - max(6, bar_h // 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 229, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _composite_camera_feeds(feeds: list) -> Optional[bytes]:
    """Stitch per-camera JPEGs into one polished 16:9 dashboard frame."""
    if not feeds:
        return None
    if len(feeds) == 1:
        return _normalize_hub_jpeg(feeds[0][1])

    out_w, out_h = WEB_STREAM_SIZE
    n = len(feeds)
    cols = 2 if n >= 2 else 1
    rows = max(1, (n + cols - 1) // cols)
    gap = 5
    pad = 6
    inner_w = out_w - pad * 2
    inner_h = out_h - pad * 2
    cell_w = max(1, (inner_w - gap * (cols - 1)) // cols)
    cell_h = max(1, (inner_h - gap * (rows - 1)) // rows)

    canvas = np.full((out_h, out_w, 3), (10, 14, 22), dtype=np.uint8)
    for idx, (name, jpeg_bytes) in enumerate(feeds):
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        tile = _draw_hud_tile_label(_fit_grid_tile_cover(frame, cell_w, cell_h), name)
        row, col = divmod(idx, cols)
        y = pad + row * (cell_h + gap)
        x = pad + col * (cell_w + gap)
        canvas[y : y + cell_h, x : x + cell_w] = tile

    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_JPEG_QUALITY])
    return buf.tobytes() if ok else None


def _latest_tracker_jpeg() -> Optional[bytes]:
    """Prefer a multi-camera composite; fall back to legacy combined hub frame."""
    feeds = list_camera_jpegs()
    if len(feeds) >= 2:
        composite = _composite_camera_feeds(feeds)
        if composite:
            return composite
    if len(feeds) == 1:
        return _normalize_hub_jpeg(feeds[0][1])
    legacy = get_latest_jpeg()
    if legacy:
        return _normalize_hub_jpeg(legacy)
    return None


def _load_cameras_config() -> List[dict]:
    if not CAMERAS_CONFIG_PATH.is_file():
        return []
    try:
        data = json.loads(CAMERAS_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [entry for entry in data if isinstance(entry, dict)]
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return []


def _primary_camera_source() -> str:
    cameras = _load_cameras_config()
    if cameras and cameras[0].get("source"):
        return str(cameras[0]["source"])


def _tracker_rtsp_configured() -> bool:
    source = _primary_camera_source()
    return source.lower().startswith(("rtsp://", "rtmp://"))


def _open_fallback_capture() -> Optional[cv2.VideoCapture]:
    """
    Open camera only when the tracker hub is idle (dashboard-only mode).
    Avoids a second RTSP session while run_system.py / main.py is publishing.
    """
    global _fallback_capture, _fallback_source
    if is_feed_available(max_age_sec=5.0):
        if _fallback_capture is not None:
            _fallback_capture.release()
            _fallback_capture = None
        return None

    source = _primary_camera_source()
    # Never open a parallel RTSP session — Dahua cameras allow very few clients.
    if source.lower().startswith(("rtsp://", "rtmp://")):
        return None
    if _fallback_capture is not None and _fallback_source == source:
        return _fallback_capture

    if _fallback_capture is not None:
        _fallback_capture.release()
        _fallback_capture = None

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _fallback_capture = cap
    _fallback_source = source
    return cap


def _read_fallback_frame() -> Optional[np.ndarray]:
    cap = _open_fallback_capture()
    if cap is None:
        return None
    success, frame = cap.read()
    if not success or frame is None:
        return None
    return frame


def gen_frames() -> Generator[bytes, None, None]:
    """
    Continuous MJPEG generator for /api/live_cctv_stream.
    Prefers frames from main.py (live_frame_hub); falls back to direct capture
    only when the tracker is not publishing (no desktop OpenCV window).
    """
    while True:
        try:
            jpeg = _latest_tracker_jpeg()
            if jpeg:
                yield _mjpeg_chunk(jpeg)
                time.sleep(0.033)
                continue

            frame = _read_fallback_frame()
            if frame is not None:
                payload = _encode_frame_jpeg(frame)
                if payload:
                    yield _mjpeg_chunk(payload)
                time.sleep(0.033)
                continue

            offline_msg = (
                "CCTV connecting — waiting for tracker stream…"
                if _tracker_rtsp_configured()
                else "CCTV offline — waiting for tracker…"
            )
            placeholder = _placeholder_jpeg(offline_msg)
            if placeholder:
                yield _mjpeg_chunk(placeholder)
            time.sleep(0.2)

        except GeneratorExit:
            global _fallback_capture
            if _fallback_capture is not None:
                _fallback_capture.release()
                _fallback_capture = None
            raise
        except Exception:
            placeholder = _placeholder_jpeg("Stream error — retrying…")
            if placeholder:
                yield _mjpeg_chunk(placeholder)
            time.sleep(0.5)


def gen_camera_frames(camera_key: str) -> Generator[bytes, None, None]:
    """MJPEG generator for a single per-camera hub JPEG."""
    label = camera_key or "camera"
    while True:
        try:
            jpeg = get_camera_jpeg(camera_key)
            if jpeg:
                yield _mjpeg_chunk(jpeg)
                time.sleep(0.033)
                continue
            placeholder = _placeholder_jpeg(f"{label} - waiting for signal...")
            if placeholder:
                yield _mjpeg_chunk(placeholder)
            time.sleep(0.12)
        except GeneratorExit:
            raise
        except Exception:
            placeholder = _placeholder_jpeg(f"{label} — stream error")
            if placeholder:
                yield _mjpeg_chunk(placeholder)
            time.sleep(0.5)


@app.route("/api/live_cctv_stream")
@app.route("/video_feed")
def live_cctv_stream():
    """Multipart JPEG stream for the dashboard CCTV card (800×450, 16:9)."""
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


def _camera_stream_response(camera_key: str) -> Response:
    return Response(
        gen_camera_frames(camera_key),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/live_camera_stream/<path:camera_key>")
@app.route("/video_feed/<path:camera_key>")
def live_camera_stream(camera_key: str):
    """Per-camera MJPEG for the zone-filtered dashboard matrix."""
    src = request.args.get("src")
    key = (camera_key or src or "").strip()
    if not key:
        return jsonify({"error": "camera key required"}), 400
    return _camera_stream_response(key)


@app.route("/api/cameras")
def api_cameras():
    """
    All cameras from cameras.json, enriched with run_system status + face counts.
    Used by the Live Cameras matrix grid.
    """
    from camera_stats import read_faces_detected

    cameras = load_cameras_config()
    status_by_key: Dict[str, dict] = {}
    status_path = PROJECT_DIR / "system_status.json"
    try:
        if status_path.is_file():
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            for row in payload.get("cameras") or []:
                if not isinstance(row, dict):
                    continue
                for key in (
                    str(row.get("location") or "").strip(),
                    str(row.get("name") or "").strip(),
                ):
                    if key:
                        status_by_key[key] = row
    except (OSError, json.JSONDecodeError, TypeError):
        pass

    enriched = []
    preview_dir = PROJECT_DIR / "live_preview" / "cameras"
    preview_max_age_sec = 45.0
    now = time.time()
    for cam in cameras:
        cam_id = str(cam.get("id") or "").strip()
        location = str(cam.get("location") or cam_id).strip()
        st = status_by_key.get(location) or status_by_key.get(cam_id) or {}
        faces = read_faces_detected(location)
        try:
            faces = max(faces, int(st.get("faces_detected") or 0))
        except (TypeError, ValueError):
            pass
        # Prefer live JPEG freshness over process-level alive flags (unified tracker
        # reports all cams alive even when RTSP open failed for some).
        preview_alive = False
        try:
            from live_frame_hub import _safe_camera_key
        except Exception:
            _safe_camera_key = lambda k: str(k).strip()  # type: ignore[misc,assignment]
        for key in (cam_id, location):
            if not key:
                continue
            for candidate in (str(key), _safe_camera_key(str(key))):
                if not candidate:
                    continue
                path = preview_dir / f"{candidate}.jpg"
                try:
                    if path.is_file() and (now - path.stat().st_mtime) <= preview_max_age_sec:
                        preview_alive = True
                        break
                except OSError:
                    continue
            if preview_alive:
                break
        process_alive = bool(st.get("alive", True)) if st else True
        enriched.append(
            {
                "id": cam_id or location,
                "group": str(cam.get("group") or ""),
                "location": location,
                "alive": bool(preview_alive and process_alive),
                "faces_detected": int(faces),
                "restart_count": int(st.get("restart_count") or 0),
            }
        )
    return jsonify({"cameras": enriched, "count": len(enriched)})


def _person_id_from_request():
    return request.args.get("person_id", type=int)


@app.route("/api/summary")
def api_summary():
    if _should_hold_dashboard_data():
        hold = dict(_SCAN_HOLD_SUMMARY)
        hold["filtered_person_id"] = _person_id_from_request()
        return _scan_hold_json_response(hold)
    db = get_db_path()
    if not db.exists():
        return jsonify({"error": "Database not found", "path": str(db)}), 404
    conn = get_connection(db)
    try:
        return jsonify(fetch_summary(conn, person_id=_person_id_from_request()))
    finally:
        conn.close()


def _last_n_day_labels(days: int = 7) -> List[str]:
    labels: List[str] = []
    today = date.today()
    for offset in range(days - 1, -1, -1):
        labels.append((today - timedelta(days=offset)).strftime("%a"))
    return labels


def _parse_dashboard_age(raw: Any) -> Optional[int]:
    if raw is None or raw == "" or str(raw).lower() == "unknown":
        return None
    try:
        age = int(raw)
    except (TypeError, ValueError):
        return None
    return age if 0 < age < 120 else None


def _dashboard_age_bucket(age: Optional[int]) -> str:
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


def _normalize_dashboard_gender(raw: Any) -> str:
    gender = str(raw or "").strip().lower()
    if gender.startswith("m"):
        return "Male"
    if gender.startswith("f"):
        return "Female"
    return "Unknown"


def _empty_dashboard_stats(range_key: str = "7d") -> Dict[str, Any]:
    """Zeroed chart payload — never return demo/fake visitor bars."""
    range_key = (range_key or "7d").strip().lower()
    if range_key == "today":
        labels = ["6am", "9am", "12pm", "3pm", "6pm", "9pm"]
        caption = "Today (hourly)"
    elif range_key == "30d":
        labels = _last_n_day_labels(30)
        caption = "Past 30 days"
    else:
        labels = _last_n_day_labels(7)
        caption = "Past 7 days"
    zeros = [0] * len(labels)
    return {
        "age": {
            "labels": ["0–17", "18–29", "30–44", "45–59", "60+", "Unknown"],
            "values": [0, 0, 0, 0, 0, 0],
        },
        "gender": {
            "labels": ["Male", "Female", "Unknown"],
            "values": [0, 0, 0],
        },
        "total_visitors": {
            "labels": labels,
            "values": list(zeros),
        },
        "repeat_visitors": {
            "labels": labels,
            "values": list(zeros),
        },
        "metrics": {"total": 0, "avgAge": None},
        "chart_caption": caption,
    }


def _mock_dashboard_stats() -> Dict[str, Any]:
    """Backward-compatible alias — empty stats only (no demo numbers)."""
    return _empty_dashboard_stats("7d")


def _fetch_daily_visit_counts(conn, days: int = 7) -> Tuple[List[str], List[int]]:
    labels = _last_n_day_labels(days)
    values: List[int] = []
    today = date.today()
    for offset in range(days - 1, -1, -1):
        day = today - timedelta(days=offset)
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM tracking_logs WHERE date(timestamp) = ?",
            (day.isoformat(),),
        ).fetchone()
        values.append(int(row["c"]) if row else 0)
    return labels, values


def _fetch_hourly_visit_counts(conn) -> Tuple[List[str], List[int]]:
    """Detection counts for today in 3-hour windows (6am, 9am, 12pm, …)."""
    buckets = [
        (6, 8, "6am"),
        (9, 11, "9am"),
        (12, 14, "12pm"),
        (15, 17, "3pm"),
        (18, 20, "6pm"),
        (21, 23, "9pm"),
    ]
    labels = [label for _, _, label in buckets]
    values: List[int] = []
    for hour_start, hour_end, _ in buckets:
        row = conn.execute(
            """
            SELECT COUNT(*) AS c FROM tracking_logs
            WHERE date(timestamp) = date('now')
              AND CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN ? AND ?
            """,
            (hour_start, hour_end),
        ).fetchone()
        values.append(int(row["c"]) if row else 0)
    return labels, values


def _count_logs_for_range(conn, range_key: str) -> int:
    """Total tracking log rows for a dashboard time range."""
    if range_key == "today":
        sql = "SELECT COUNT(*) AS c FROM tracking_logs WHERE date(timestamp) = date('now')"
        params: tuple = ()
    elif range_key == "30d":
        sql = "SELECT COUNT(*) AS c FROM tracking_logs WHERE date(timestamp) >= date('now', '-29 days')"
        params = ()
    else:
        sql = "SELECT COUNT(*) AS c FROM tracking_logs WHERE date(timestamp) >= date('now', '-6 days')"
        params = ()
    row = conn.execute(sql, params).fetchone()
    return int(row["c"]) if row else 0


def _fetch_demographics_for_range(conn, range_key: str) -> Tuple[Dict[str, int], Dict[str, int], Optional[int]]:
    """Age/gender bucket counts + average age for a dashboard time range."""
    age_labels = ["0–17", "18–29", "30–44", "45–59", "60+", "Unknown"]
    age_counts = {label: 0 for label in age_labels}
    gender_counts = {"Male": 0, "Female": 0, "Unknown": 0}
    age_sum = 0
    age_n = 0

    if range_key == "today":
        date_filter = "date(timestamp) = date('now')"
        params: tuple = ()
    elif range_key == "30d":
        date_filter = "date(timestamp) >= date('now', '-29 days')"
        params = ()
    else:
        date_filter = "date(timestamp) >= date('now', '-6 days')"
        params = ()

    rows = conn.execute(
        f"SELECT gender, age FROM tracking_logs WHERE {date_filter}",
        params,
    ).fetchall()
    for row in rows:
        gender_counts[_normalize_dashboard_gender(row["gender"])] = (
            gender_counts.get(_normalize_dashboard_gender(row["gender"]), 0) + 1
        )
        age = _parse_dashboard_age(row["age"])
        bucket = _dashboard_age_bucket(age)
        age_counts[bucket] = age_counts.get(bucket, 0) + 1
        if age is not None:
            age_sum += age
            age_n += 1

    avg_age = round(age_sum / age_n) if age_n > 0 else None
    return age_counts, gender_counts, avg_age


def _build_dashboard_stats(conn, range_key: str = "7d") -> Dict[str, Any]:
    range_key = (range_key or "7d").strip().lower()
    if range_key not in ("today", "7d", "30d"):
        range_key = "7d"

    # Real DB counts only — never leave demo visitor bars when the range is empty.
    stats = _empty_dashboard_stats(range_key)
    age_labels = ["0–17", "18–29", "30–44", "45–59", "60+", "Unknown"]
    age_counts, gender_counts, avg_age = _fetch_demographics_for_range(conn, range_key)

    stats["gender"] = {
        "labels": ["Male", "Female", "Unknown"],
        "values": [
            gender_counts["Male"],
            gender_counts["Female"],
            gender_counts["Unknown"],
        ],
    }
    stats["age"] = {
        "labels": age_labels,
        "values": [age_counts[label] for label in age_labels],
    }

    if range_key == "today":
        day_labels, total_values = _fetch_hourly_visit_counts(conn)
        caption = "Today (hourly)"
    elif range_key == "30d":
        day_labels, total_values = _fetch_daily_visit_counts(conn, days=30)
        caption = "Past 30 days"
    else:
        day_labels, total_values = _fetch_daily_visit_counts(conn, days=7)
        caption = "Past 7 days"

    range_total = _count_logs_for_range(conn, range_key)
    stats["chart_caption"] = caption
    stats["metrics"] = {
        "total": range_total,
        "avgAge": avg_age,
    }

    # Always replace demo bars with real series (zeros included).
    stats["total_visitors"] = {"labels": day_labels, "values": total_values}
    chart_total = sum(total_values)
    if chart_total > 0:
        try:
            from flask import has_request_context

            person_id = _person_id_from_request() if has_request_context() else None
            summary = fetch_summary(conn, person_id=person_id)
        except Exception:
            summary = {}
        unique_people = int(summary.get("unique_people") or 0)
        total_visits = int(summary.get("total_visits") or chart_total)
        repeat_ratio = max(
            0.0,
            min(1.0, (total_visits - unique_people) / max(total_visits, 1)),
        )
        repeat_values = [
            max(0, round(value * repeat_ratio)) for value in total_values
        ]
        stats["repeat_visitors"] = {"labels": day_labels, "values": repeat_values}
    else:
        stats["repeat_visitors"] = {
            "labels": day_labels,
            "values": [0] * len(day_labels),
        }

    return stats


@app.route("/api/dashboard_stats")
def api_dashboard_stats():
    """Chart payloads for dashboard visitor analytics (Chart.js)."""
    range_key = (request.args.get("range") or "7d").strip().lower()
    db = get_db_path()
    if not db.exists():
        return jsonify(_empty_dashboard_stats(range_key))
    conn = get_connection(db)
    try:
        return jsonify(_build_dashboard_stats(conn, range_key=range_key))
    except Exception as exc:
        app.logger.warning("dashboard_stats failed: %s", exc)
        try:
            range_total = _count_logs_for_range(conn, range_key)
            _, _, avg_age = _fetch_demographics_for_range(conn, range_key)
            empty = _empty_dashboard_stats(range_key)
            empty["metrics"] = {"total": range_total, "avgAge": avg_age}
            return jsonify(empty)
        except Exception:
            return jsonify(_empty_dashboard_stats(range_key))
    finally:
        conn.close()


@app.route("/api/settings/announcements", methods=["GET", "POST"])
def api_announcement_settings():
    """Read or update Telegram announcement time window (dashboard Settings)."""
    from announcement_settings import (
        announcement_schedule_status,
        load_announcement_settings,
        save_announcement_settings,
    )

    if request.method == "GET":
        settings = load_announcement_settings(use_cache=False)
        status = announcement_schedule_status()
        resp = jsonify({"ok": True, "settings": settings, "status": status})
    else:
        payload = request.get_json(silent=True) or {}
        try:
            saved = save_announcement_settings(payload)
            status = announcement_schedule_status()
            resp = jsonify({"ok": True, "settings": saved, "status": status})
        except Exception as exc:
            app.logger.warning("announcement settings save failed: %s", exc)
            return jsonify({"ok": False, "error": str(exc)}), 400
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/recent_captures")
def api_recent_captures():
    """Face/scene captures for the dashboard Recent Captures gallery.

    Supports ``range=today|7d|30d`` (dashboard time filter). Legacy ``today=1``
    still means today-only when ``range`` is omitted.
    """
    if _should_hold_dashboard_data():
        return _scan_hold_json_response([])
    db = get_db_path()
    if not db.exists():
        return jsonify([])
    limit = max(1, min(request.args.get("limit", 12, type=int), 500))
    range_key = (request.args.get("range") or request.args.get("time_range") or "").strip().lower()
    if range_key in ("week",):
        range_key = "7d"
    if range_key in ("month",):
        range_key = "30d"
    today_raw = (request.args.get("today") or "").strip().lower()
    today_only = False
    if range_key not in ("today", "7d", "30d"):
        # Legacy: default today-only unless today=0
        if today_raw in ("0", "false", "no", "all"):
            range_key = None
            today_only = False
        else:
            range_key = "today"
            today_only = True
    conn = get_connection(db)
    try:
        from dashboard_data import fetch_live_face_events_as_logs

        return jsonify(
            fetch_live_face_events_as_logs(
                conn,
                limit=limit,
                today_only=today_only,
                range_key=range_key,
            )
        )
    finally:
        conn.close()


@app.route("/api/recent_plates")
def api_recent_plates():
    """Recent license-plate OCR hits from CCTV cameras."""
    limit = max(1, min(request.args.get("limit", 48, type=int), 200))
    range_key = (request.args.get("range") or request.args.get("time_range") or "today").strip().lower()
    if range_key not in ("today", "7d", "30d", "week", "month"):
        range_key = "today"
    if range_key == "week":
        range_key = "7d"
    if range_key == "month":
        range_key = "30d"
    try:
        from license_plate_ocr import list_recent_plates

        rows = list_recent_plates(limit=limit, range_key=range_key)
    except Exception as exc:
        app.logger.warning("recent_plates failed: %s", exc)
        rows = []
    resp = jsonify(
        {
            "ok": True,
            "plates": rows,
            "count": len(rows),
            "range": range_key,
        }
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/plates/label", methods=["POST"])
def api_plates_label():
    """Correct OCR text or mark a plate event resolved/dismissed."""
    payload = request.get_json(silent=True) or {}
    filename = str(payload.get("filename") or "").strip()
    text = str(payload.get("text") or payload.get("plate") or "").strip()
    review_status = str(payload.get("review_status") or "").strip() or None
    try:
        from license_plate_ocr import update_plate_label

        result = update_plate_label(
            filename=filename,
            text=text,
            review_status=review_status,
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    if not result.get("ok"):
        return jsonify(result), 400
    return jsonify(result)


@app.route("/plate_captures/<path:filename>")
def serve_plate_capture(filename):
    plate_dir = PROJECT_DIR / "plate_captures"
    plate_dir.mkdir(parents=True, exist_ok=True)
    return send_from_directory(plate_dir, filename)


@app.route("/api/captures/encounter_cluster")
def api_capture_encounter_cluster():
    """Distinct people captured on one camera within a short time window."""
    if _should_hold_dashboard_data():
        return _scan_hold_json_response({"ok": True, "entries": [], "count": 0})
    db = get_db_path()
    camera = (request.args.get("camera") or request.args.get("camera_location") or "").strip()
    timestamp = (request.args.get("timestamp") or request.args.get("created_at") or "").strip()
    window_sec = max(10, min(request.args.get("window_sec", 60, type=int), 180))
    limit = max(1, min(request.args.get("limit", 8, type=int), 12))
    if not camera or not timestamp or not db.exists():
        return jsonify({"ok": True, "entries": [], "count": 0})
    conn = get_connection(db)
    try:
        from dashboard_data import fetch_capture_encounter_cluster

        entries = fetch_capture_encounter_cluster(
            conn,
            camera,
            timestamp,
            window_sec=window_sec,
            limit=limit,
        )
        return jsonify({"ok": True, "entries": entries, "count": len(entries)})
    finally:
        conn.close()


@app.route("/api/captures/by_age")
def api_captures_by_age():
    """Captures for one Age distribution bucket (click a bar on the dashboard)."""
    if _should_hold_dashboard_data():
        return _scan_hold_json_response({"ok": True, "bucket": "Unknown", "entries": [], "count": 0})
    db = get_db_path()
    bucket_raw = (
        request.args.get("bucket")
        or request.args.get("age")
        or request.args.get("age_bucket")
        or "Unknown"
    )
    range_key = (request.args.get("range") or request.args.get("time_range") or "today").strip().lower()
    if range_key not in ("today", "7d", "30d"):
        range_key = "today"
    limit = max(1, min(request.args.get("limit", 48, type=int), 96))
    if not db.exists():
        return jsonify({"ok": True, "bucket": str(bucket_raw), "range": range_key, "entries": [], "count": 0})
    conn = get_connection(db)
    try:
        from dashboard_data import fetch_captures_by_age_bucket, normalize_age_bucket_param

        bucket = normalize_age_bucket_param(bucket_raw)
        entries = fetch_captures_by_age_bucket(
            conn,
            bucket,
            range_key=range_key,
            limit=limit,
        )
        return jsonify(
            {
                "ok": True,
                "bucket": bucket,
                "range": range_key,
                "entries": entries,
                "count": len(entries),
            }
        )
    finally:
        conn.close()


@app.route("/api/captures/by_camera")
def api_captures_by_camera():
    """Captures for one camera (click a Camera status tile)."""
    if _should_hold_dashboard_data():
        return _scan_hold_json_response({"ok": True, "camera": "", "entries": [], "count": 0})
    db = get_db_path()
    camera_raw = (
        request.args.get("camera")
        or request.args.get("camera_location")
        or request.args.get("location")
        or ""
    )
    range_key = (request.args.get("range") or request.args.get("time_range") or "today").strip().lower()
    if range_key not in ("today", "7d", "30d"):
        range_key = "today"
    limit = max(1, min(request.args.get("limit", 48, type=int), 96))
    if not db.exists():
        return jsonify({"ok": True, "camera": str(camera_raw), "range": range_key, "entries": [], "count": 0})
    conn = get_connection(db)
    try:
        from dashboard_data import (
            count_captures_by_camera,
            fetch_captures_by_camera,
            normalize_camera_location_param,
        )

        camera = normalize_camera_location_param(camera_raw)
        entries = fetch_captures_by_camera(
            conn,
            camera,
            range_key=range_key,
            limit=limit,
        )
        total = count_captures_by_camera(conn, camera, range_key=range_key)
        return jsonify(
            {
                "ok": True,
                "camera": camera,
                "range": range_key,
                "entries": entries,
                "count": len(entries),
                "total": total,
            }
        )
    finally:
        conn.close()


@app.route("/api/captures/by_gender")
def api_captures_by_gender():
    """Captures for one Gender split bucket (click Male / Female / Unknown)."""
    if _should_hold_dashboard_data():
        return _scan_hold_json_response({"ok": True, "gender": "Unknown", "entries": [], "count": 0})
    db = get_db_path()
    gender_raw = request.args.get("gender") or request.args.get("bucket") or "Unknown"
    range_key = (request.args.get("range") or request.args.get("time_range") or "today").strip().lower()
    if range_key not in ("today", "7d", "30d"):
        range_key = "today"
    limit = max(1, min(request.args.get("limit", 48, type=int), 96))
    if not db.exists():
        return jsonify({"ok": True, "gender": str(gender_raw), "range": range_key, "entries": [], "count": 0})
    conn = get_connection(db)
    try:
        from dashboard_data import (
            count_captures_by_gender,
            fetch_captures_by_gender,
            normalize_gender_param,
        )

        gender = normalize_gender_param(gender_raw)
        entries = fetch_captures_by_gender(
            conn,
            gender,
            range_key=range_key,
            limit=limit,
        )
        total = count_captures_by_gender(conn, gender, range_key=range_key)
        return jsonify(
            {
                "ok": True,
                "gender": gender,
                "range": range_key,
                "entries": entries,
                "count": len(entries),
                "total": total,
            }
        )
    finally:
        conn.close()


@app.route("/api/captures/by_overview")
def api_captures_by_overview():
    """Captures for one Today's breakdown slice (Known / Visitors / Watchlist / Other)."""
    if _should_hold_dashboard_data():
        return _scan_hold_json_response(
            {"ok": True, "category": "known", "entries": [], "count": 0, "total": 0}
        )
    db = get_db_path()
    category_raw = (
        request.args.get("category")
        or request.args.get("bucket")
        or request.args.get("label")
        or "known"
    )
    range_key = (request.args.get("range") or request.args.get("time_range") or "today").strip().lower()
    if range_key not in ("today", "7d", "30d"):
        range_key = "today"
    limit = max(1, min(request.args.get("limit", 96, type=int), 120))
    if not db.exists():
        return jsonify(
            {
                "ok": True,
                "category": str(category_raw),
                "range": range_key,
                "entries": [],
                "count": 0,
                "total": 0,
            }
        )
    conn = get_connection(db)
    try:
        from dashboard_data import (
            count_captures_by_overview_category,
            fetch_captures_by_overview_category,
            normalize_overview_category,
        )

        category = normalize_overview_category(category_raw)
        entries = fetch_captures_by_overview_category(
            conn,
            category,
            range_key=range_key,
            limit=limit,
        )
        total = count_captures_by_overview_category(
            conn, category, range_key=range_key
        )
        label = {
            "known": "Known",
            "visitors": "Visitors",
            "watchlist": "Watchlist",
            "other": "Other",
        }.get(category, "Other")
        return jsonify(
            {
                "ok": True,
                "category": category,
                "label": label,
                "range": range_key,
                "entries": entries,
                "count": len(entries),
                "total": total,
            }
        )
    finally:
        conn.close()


@app.route("/api/captures/by_time")
def api_captures_by_time():
    """Captures for one Detections chart bar (hourly today, or one day in 7d/30d)."""
    if _should_hold_dashboard_data():
        return _scan_hold_json_response({"ok": True, "entries": [], "count": 0})
    db = get_db_path()
    range_key = (request.args.get("range") or request.args.get("time_range") or "today").strip().lower()
    if range_key not in ("today", "7d", "30d"):
        range_key = "today"
    limit = max(1, min(request.args.get("limit", 96, type=int), 120))
    bucket_raw = request.args.get("bucket") or request.args.get("hour") or ""
    day_iso = (request.args.get("day") or "").strip()[:10]
    day_index = request.args.get("day_index", type=int)

    if not db.exists():
        return jsonify({"ok": True, "range": range_key, "entries": [], "count": 0, "total": 0})

    conn = get_connection(db)
    try:
        from dashboard_data import (
            count_captures_by_day,
            count_captures_by_hour_bucket,
            fetch_captures_by_day,
            fetch_captures_by_hour_bucket,
            normalize_hour_bucket_param,
        )

        if range_key == "today":
            bucket = normalize_hour_bucket_param(bucket_raw)
            if bucket is None:
                return jsonify(
                    {
                        "ok": False,
                        "error": "Unknown hour bucket",
                        "entries": [],
                        "count": 0,
                        "total": 0,
                    }
                ), 400
            entries = fetch_captures_by_hour_bucket(conn, bucket, limit=limit)
            total = count_captures_by_hour_bucket(conn, bucket)
            return jsonify(
                {
                    "ok": True,
                    "range": range_key,
                    "bucket": bucket,
                    "label": bucket,
                    "entries": entries,
                    "count": len(entries),
                    "total": total,
                }
            )

        # 7d / 30d: resolve weekday label index → calendar date.
        days = 30 if range_key == "30d" else 7
        if not day_iso:
            if day_index is None:
                return jsonify(
                    {
                        "ok": False,
                        "error": "day or day_index required",
                        "entries": [],
                        "count": 0,
                        "total": 0,
                    }
                ), 400
            idx = max(0, min(int(day_index), days - 1))
            day_iso = (date.today() - timedelta(days=(days - 1 - idx))).isoformat()
        entries = fetch_captures_by_day(conn, day_iso, limit=limit)
        total = count_captures_by_day(conn, day_iso)
        return jsonify(
            {
                "ok": True,
                "range": range_key,
                "day": day_iso,
                "label": day_iso,
                "entries": entries,
                "count": len(entries),
                "total": total,
            }
        )
    finally:
        conn.close()


@app.route("/api/captures/by_window")
def api_captures_by_window():
    """Captures for dashboard KPI clicks: today or active (last 15 minutes)."""
    if _should_hold_dashboard_data():
        return _scan_hold_json_response(
            {"ok": True, "entries": [], "count": 0, "total": 0}
        )
    db = get_db_path()
    window = (request.args.get("window") or "today").strip().lower()
    if window in ("active_now", "recent", "15m"):
        window = "active"
    if window not in ("today", "active"):
        window = "today"
    limit = max(1, min(request.args.get("limit", 96, type=int), 120))

    if not db.exists():
        return jsonify(
            {
                "ok": True,
                "window": window,
                "entries": [],
                "count": 0,
                "total": 0,
            }
        )

    conn = get_connection(db)
    try:
        from dashboard_data import count_captures_by_window, fetch_captures_by_window

        entries = fetch_captures_by_window(conn, window, limit=limit)
        total = count_captures_by_window(conn, window)
        return jsonify(
            {
                "ok": True,
                "window": window,
                "label": "Active now" if window == "active" else "Detections today",
                "entries": entries,
                "count": len(entries),
                "total": total,
            }
        )
    finally:
        conn.close()


@app.route("/api/captures/label", methods=["POST"])
def api_captures_label():
    """Name a captured person without adding them to the watchlist."""
    payload = request.get_json(silent=True) or {}
    name = str(
        payload.get("name")
        or payload.get("person_name")
        or request.form.get("name")
        or ""
    ).strip()
    if not name:
        return jsonify({"ok": False, "error": "Name is required."}), 400
    log_id = payload.get("log_id") or payload.get("id") or request.form.get("log_id")
    known_id = (
        payload.get("known_person_id")
        or payload.get("person_id")
        or request.form.get("known_person_id")
    )
    db = get_db_path()
    if not db.exists():
        return jsonify({"ok": False, "error": "Database not found."}), 404
    conn = get_connection(db)
    try:
        from identity import label_captured_person

        result = label_captured_person(
            conn,
            name,
            log_id=log_id,
            known_person_id=known_id,
        )
        return jsonify({"ok": True, **result})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Could not save name: {exc}"}), 500
    finally:
        conn.close()


@app.route("/api/visitors")
def api_visitors():
    """All visitors with gender/age from tracking_logs (Unknown when not set)."""
    if _should_hold_dashboard_data():
        return _scan_hold_json_response([])
    db = get_db_path()
    if not db.exists():
        return jsonify([])
    conn = get_connection(db)
    try:
        return jsonify(fetch_visitors(conn, person_id=_person_id_from_request()))
    finally:
        conn.close()


@app.route("/static/crops/<path:filename>")
def serve_live_detection_crops(filename):
    """Serve face crops published for the live detection SSE feed."""
    from live_detection_hub import CROPS_DIR, ensure_live_detection_dirs

    ensure_live_detection_dirs()
    return send_from_directory(CROPS_DIR, filename)


@app.route("/api/detection-stream")
def api_detection_stream():
    """
    Server-Sent Events stream of live face detections from tracker processes.
    Emits one JSON object per line as ``data: {...}\\n\\n``.
    """
    from live_detection_hub import ensure_live_detection_dirs, read_events_since, recent_events

    ensure_live_detection_dirs()
    backlog = max(0, min(request.args.get("backlog", 12, type=int), 40))

    def generate():
        # Seed the client with recent events so the panel isn't empty on connect.
        try:
            for event in reversed(recent_events(limit=backlog)):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception:
            pass

        # Follow newly appended JSONL bytes from EOF (avoid replaying backlog).
        from live_detection_hub import EVENTS_PATH

        try:
            offset = EVENTS_PATH.stat().st_size if EVENTS_PATH.is_file() else 0
        except OSError:
            offset = 0
        last_ping = time.time()
        while True:
            try:
                events, offset = read_events_since(offset, limit=40)
                for event in events:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    last_ping = time.time()
                if time.time() - last_ping >= 12.0:
                    yield ": keepalive\n\n"
                    last_ping = time.time()
            except GeneratorExit:
                break
            except Exception:
                yield ": error\n\n"
                time.sleep(1.0)
                continue
            time.sleep(0.35)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(generate(), mimetype="text/event-stream", headers=headers)


@app.route("/api/live_detections")
def api_live_detections():
    """JSON snapshot of recent SSE detection events (for fallback polling)."""
    from live_detection_hub import recent_events

    limit = max(1, min(request.args.get("limit", 24, type=int), 80))
    return jsonify({"ok": True, "entries": recent_events(limit=limit), "count": limit})


@app.route("/api/captured_faces")
def api_captured_faces():
    """
    Face Detection side-panel gallery: high-confidence cropped face images
    with camera ID + timestamp metadata.
    """
    from live_detection_hub import HIGH_CONFIDENCE_THRESHOLD, list_captured_faces

    limit = max(1, min(request.args.get("limit", 24, type=int), 80))
    min_conf = request.args.get("min_confidence", default=HIGH_CONFIDENCE_THRESHOLD, type=float)
    if min_conf is None:
        min_conf = HIGH_CONFIDENCE_THRESHOLD
    entries = list_captured_faces(limit=limit, min_confidence=float(min_conf))
    return jsonify(
        {
            "ok": True,
            "entries": entries,
            "count": len(entries),
            "min_confidence": float(min_conf),
        }
    )


@app.route("/api/face_boxes/<path:camera_key>")
def api_face_boxes(camera_key: str):
    """Latest face bounding boxes for a camera (pixel xyxy + source dimensions)."""
    from live_detection_hub import get_face_boxes

    payload = get_face_boxes(camera_key or "")
    return jsonify({"ok": True, **payload})


FACE_DETECT_BOX_BGR = (0, 220, 0)  # green (BGR)


def _draw_green_face_boxes(frame: np.ndarray, camera_key: str) -> np.ndarray:
    """Overlay white corner-bracket face targets (FR surveillance style)."""
    from live_detection_hub import get_face_boxes
    from face_overlay_draw import draw_corner_bracket_boxes

    payload = get_face_boxes(camera_key)
    boxes = payload.get("boxes") or []
    if not boxes:
        return frame
    out = frame
    src_w = int(payload.get("width") or 0)
    src_h = int(payload.get("height") or 0)
    h, w = out.shape[:2]
    sx = (w / src_w) if src_w > 0 else 1.0
    sy = (h / src_h) if src_h > 0 else 1.0
    # Drop stale boxes (>2.5s) so the overlay does not linger.
    updated = float(payload.get("updated") or 0)
    if updated and (time.time() - updated) > 2.5:
        return out
    scaled: list = []
    for box in boxes:
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
        except (TypeError, ValueError, IndexError):
            continue
        x1 = int(max(0, min(w - 1, x1 * sx)))
        y1 = int(max(0, min(h - 1, y1 * sy)))
        x2 = int(max(0, min(w - 1, x2 * sx)))
        y2 = int(max(0, min(h - 1, y2 * sy)))
        if x2 > x1 and y2 > y1:
            scaled.append((x1, y1, x2, y2))
    return draw_corner_bracket_boxes(out, scaled, color=(255, 255, 255), thickness=2)


def gen_face_detection_frames(camera_key: str):
    """MJPEG generator: live camera frame + green face boxes (Face Detection page only)."""
    label = camera_key or "camera"
    while True:
        try:
            jpeg = _encode_live_camera_jpeg(camera_key, face_boxes=True)
            if jpeg:
                yield _mjpeg_chunk(jpeg)
                time.sleep(0.066)
                continue
            placeholder = _placeholder_jpeg(f"{label} - waiting for face stream...")
            if placeholder:
                yield _mjpeg_chunk(placeholder)
            time.sleep(0.25)
        except GeneratorExit:
            raise
        except Exception:
            placeholder = _placeholder_jpeg(f"{label} — face stream error")
            if placeholder:
                yield _mjpeg_chunk(placeholder)
            time.sleep(0.5)


@app.route("/api/face_detection_stream/<path:camera_key>")
def face_detection_stream(camera_key: str):
    """Per-camera MJPEG for Face Detection view (green face bounding boxes)."""
    key = (camera_key or "").strip()
    if not key:
        return jsonify({"error": "camera key required"}), 400
    return Response(
        gen_face_detection_frames(key),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/recent")
def api_recent():
    db = get_db_path()
    if not db.exists():
        return jsonify([])
    conn = get_connection(db)
    try:
        rows = fetch_recent_logs(conn, person_id=_person_id_from_request())
        return jsonify(_enrich_recent_logs_face_attributes(conn, rows))
    finally:
        conn.close()


@app.route("/api/locations")
def api_locations():
    db = get_db_path()
    if not db.exists():
        return jsonify([])
    conn = get_connection(db)
    try:
        return jsonify(fetch_by_location(conn, person_id=_person_id_from_request()))
    finally:
        conn.close()


@app.route("/captures/<path:filename>")
def serve_capture(filename):
    return send_from_directory(CAPTURES_DIR, filename)


@app.route("/static/captures/<path:filename>")
def serve_static_captures_alias(filename):
    """Backward-compatible alias — scan payloads historically used /static/captures/."""
    return send_from_directory(CAPTURES_DIR, filename)


@app.route("/static/snapshots/<path:filename>")
def serve_infraction_snapshots(filename):
    """Serve legacy per-incident snapshots under static/snapshots/."""
    SNAPSHOTS_STATIC_DIR.mkdir(parents=True, exist_ok=True)
    return send_from_directory(SNAPSHOTS_STATIC_DIR, filename)


@app.route("/static/infractions/<path:filename>")
def serve_infraction_captures(filename):
    """Serve unique live-infraction JPEGs used by the review modal."""
    INFRACTIONS_STATIC_DIR.mkdir(parents=True, exist_ok=True)
    return send_from_directory(INFRACTIONS_STATIC_DIR, filename)


@app.route("/go2rtc/", defaults={"subpath": ""}, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"])
@app.route("/go2rtc/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"])
def go2rtc_proxy(subpath: str):
    """Same-origin proxy to local go2rtc (WebRTC SDP + stream API)."""
    if not is_logged_in():
        return jsonify({"error": "Authentication required"}), 401
    upstream = (os.environ.get("GO2RTC_URL") or "http://127.0.0.1:1986").rstrip("/")
    target = f"{upstream}/{subpath}" if subpath else f"{upstream}/"
    qs = request.query_string.decode("latin-1") if request.query_string else ""
    if qs:
        target = f"{target}?{qs}"
    headers = {
        key: value
        for key, value in request.headers
        if key.lower() not in ("host", "content-length", "connection", "cookie")
    }
    body = request.get_data() if request.method not in ("GET", "HEAD") else None
    req = urllib.request.Request(target, data=body, method=request.method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = resp.read()
            content_type = resp.headers.get("Content-Type", "application/octet-stream")
            return Response(payload, status=int(getattr(resp, "status", 200)), content_type=content_type)
    except urllib.error.HTTPError as exc:
        return Response(exc.read(), status=int(exc.code), content_type=exc.headers.get("Content-Type", "text/plain"))
    except urllib.error.URLError:
        return jsonify({"error": "go2rtc offline"}), 502


@app.route("/live_preview/<path:filename>")
def serve_live_preview(filename):
    """Serve live preview stills written under project/live_preview/."""
    LIVE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    return send_from_directory(LIVE_PREVIEW_DIR, filename)


@app.route("/stills/<path:filename>")
def serve_stills(filename):
    """Serve optional debug still frames written under project/stills/."""
    if not STILLS_DIR.is_dir():
        return jsonify({"error": "stills directory not found"}), 404
    return send_from_directory(STILLS_DIR, filename)


@app.route("/runs/<path:filename>")
def serve_runs(filename):
    """Serve optional YOLO run artifacts under project/runs/."""
    if not RUNS_DIR.is_dir():
        return jsonify({"error": "runs directory not found"}), 404
    return send_from_directory(RUNS_DIR, filename)


@app.route("/api/captures")
def api_captures():
    """Gallery images only when a specific person is selected; blank for All people."""
    if _should_hold_dashboard_data():
        return _scan_hold_json_response([])
    db = get_db_path()
    if not db.exists():
        return jsonify([])

    person_id = request.args.get("person_id", type=int)
    if person_id is None:
        return jsonify([])

    limit = request.args.get("limit", 48, type=int)
    conn = get_connection(db)
    try:
        return jsonify(fetch_captures(conn, limit=limit, person_id=person_id))
    finally:
        conn.close()


@app.route("/api/timeline")
def api_timeline():
    db = get_db_path()
    if not db.exists():
        return jsonify([])
    conn = get_connection(db)
    try:
        return jsonify(fetch_timeline(conn, person_id=_person_id_from_request()))
    finally:
        conn.close()


@app.route("/api/face_gallery")
def api_face_gallery():
    limit = request.args.get("limit", 48, type=int)
    return jsonify(fetch_live_face_gallery(limit=limit))


@app.route("/api/cross_camera_journey")
def api_cross_camera_journey():
    db = get_db_path()
    if not db.exists():
        return jsonify([])
    limit = request.args.get("limit", 20, type=int)
    conn = get_connection(db)
    try:
        return jsonify(fetch_cross_camera_journeys(conn, limit=limit))
    finally:
        conn.close()


@app.route("/api/face_quality_stats")
def api_face_quality_stats():
    return jsonify(read_face_stats())


@app.route("/api/get_extended_metrics")
def api_get_extended_metrics():
    """Real-time Watchlist + Journey + Rejection Radar payload."""
    db = get_db_path()
    if not db.exists():
        return jsonify(fetch_extended_metrics(tracking_conn=None))
    conn = get_connection(db)
    try:
        return jsonify(fetch_extended_metrics(tracking_conn=conn))
    finally:
        conn.close()


@app.route("/api/face_intel")
def api_face_intel():
    """Combined live gallery + journeys + quality stats (single poll)."""
    db = get_db_path()
    if not db.exists():
        return jsonify(
            {
                "gallery": {"entries": [], "total": 0, "cameras": []},
                "journeys": [],
                "quality": read_face_stats(),
                "cross_camera_live": [],
            }
        )
    conn = get_connection(db)
    try:
        return jsonify(fetch_face_intel_bundle(conn))
    finally:
        conn.close()


def _person_display_name(conn: sqlite3.Connection, person_id: int) -> str:
    try:
        row = conn.execute(
            "SELECT name FROM known_persons WHERE id = ?",
            (int(person_id),),
        ).fetchone()
        if row is not None:
            name = str(row[0] or "").strip()
            if name:
                return name
    except (TypeError, ValueError, sqlite3.Error):
        pass
    return f"Person #{int(person_id)}"


def _enrich_lookup_candidates(
    conn: sqlite3.Connection,
    candidates: List[Dict[str, Any]],
    *,
    capture_limit: int = 24,
) -> List[Dict[str, Any]]:
    """Attach name, profile portrait, and capture thumbnails to lookup hits."""
    profiles = fetch_profile_images_by_person(conn)
    enriched: List[Dict[str, Any]] = []
    for item in candidates:
        try:
            pid = int(item["person_id"])
        except (KeyError, TypeError, ValueError):
            continue
        by_loc = [
            {"name": name, "visits": count}
            for name, count in item.get("by_location", [])
        ]
        captures = fetch_captures(conn, limit=capture_limit, person_id=pid)
        profile_url = profiles.get(pid)
        if profile_url and not str(profile_url).startswith("/"):
            name = Path(str(profile_url)).name
            if name:
                guess = CAPTURES_DIR / "best_faces" / name
                profile_url = (
                    f"/captures/best_faces/{name}" if guess.is_file() else None
                )
        enriched.append(
            {
                "person_id": pid,
                "name": _person_display_name(conn, pid),
                "distance": item.get("distance"),
                "total_visits": item.get("total_visits"),
                "last_seen": item.get("last_seen"),
                "source": item.get("source", ""),
                "by_location": by_loc,
                "profile_url": profile_url,
                "captures": captures,
            }
        )
    return enriched


def _format_lookup_response(
    result: dict,
    no_match_message: str,
    *,
    video_duration: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> tuple:
    """Build JSON body and HTTP status from lookup_photo / lookup_video result."""
    duration_fields: Dict[str, object] = {}
    if video_duration and video_duration != "—":
        duration_fields = {
            "video_duration": video_duration,
            "duration": video_duration,
            "system_status": {"duration": video_duration},
        }
    if not result.get("matched"):
        return (
            jsonify(
                {
                    "matched": False,
                    "candidates": [],
                    "distance": result.get("distance"),
                    "message": result.get("message", no_match_message),
                    "frames_scanned": result.get("frames_scanned"),
                    "video_frames_total": result.get("video_frames_total"),
                    **duration_fields,
                }
            ),
            200,
        )

    raw_candidates = list(result.get("candidates") or [])
    if conn is not None:
        candidates = _enrich_lookup_candidates(conn, raw_candidates)
    else:
        candidates = []
        for item in raw_candidates:
            by_loc = [
                {"name": name, "visits": count}
                for name, count in item.get("by_location", [])
            ]
            candidates.append(
                {
                    "person_id": item["person_id"],
                    "name": f"Person #{item['person_id']}",
                    "distance": item["distance"],
                    "total_visits": item["total_visits"],
                    "last_seen": item.get("last_seen"),
                    "source": item.get("source", ""),
                    "by_location": by_loc,
                    "profile_url": None,
                    "captures": [],
                }
            )

    top = candidates[0] if candidates else {}
    return (
        jsonify(
            {
                "matched": True,
                "candidates": candidates,
                "person_id": top.get("person_id"),
                "name": top.get("name"),
                "total_visits": top.get("total_visits"),
                "last_seen": top.get("last_seen"),
                "match_type": result.get("match_type", ""),
                "distance": top.get("distance"),
                "query_quality": result.get("query_quality"),
                "profile_upgraded": result.get("profile_upgraded", False),
                "by_location": top.get("by_location", []),
                "profile_url": top.get("profile_url"),
                "captures": top.get("captures") or [],
                "frames_scanned": result.get("frames_scanned"),
                "video_frames_total": result.get("video_frames_total"),
                **duration_fields,
            }
        ),
        200,
    )


@app.route("/api/lookup", methods=["POST"])
def api_lookup():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    db = get_db_path()
    if not db.exists():
        return jsonify({"error": "Database not found. Run main.py on CCTV first."}), 404

    UPLOAD_DIR.mkdir(exist_ok=True)
    suffix = _image_suffix_from_filename(file.filename)
    save_path = UPLOAD_DIR / f"search{suffix}"

    try:
        save_path = save_uploaded_image(file, save_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        conn = get_connection(db)
        try:
            yolo = get_yolo()
            result = lookup_photo(conn, save_path, yolo_model=yolo)
            return _format_lookup_response(
                result,
                "No person detected. Use a photo with a visible person.",
                conn=conn,
            )
        finally:
            conn.close()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Search failed: {exc}"}), 500


@app.route("/search_by_video", methods=["POST"])
def search_by_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in VIDEO_SEARCH_EXTENSIONS:
        return jsonify(
            {
                "error": "Unsupported video format. Use .mp4, .avi, or .mov.",
            }
        ), 400

    db = get_db_path()
    if not db.exists():
        return jsonify({"error": "Database not found. Run main.py on CCTV first."}), 404

    UPLOAD_DIR.mkdir(exist_ok=True)
    save_path = UPLOAD_DIR / f"search_video{suffix}"

    try:
        save_uploaded_video(file, save_path)
    except VideoCodecError as exc:
        return _video_upload_error_response(exc)

    video_duration = calculate_video_duration(str(save_path))

    try:
        conn = get_connection(db)
        try:
            yolo = get_yolo()
            result = lookup_video(conn, save_path, yolo_model=yolo)
            return _format_lookup_response(
                result,
                "No face detected in video or no database match.",
                video_duration=video_duration,
                conn=conn,
            )
        finally:
            conn.close()
    except VideoCodecError as exc:
        return _video_upload_error_response(exc)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return _video_upload_error_response(exc)
    finally:
        if save_path.is_file():
            try:
                save_path.unlink()
            except OSError:
                pass


@app.route("/api/scan_progress", methods=["GET"])
@app.route("/api/scan_progress/<scan_id>")
def api_scan_progress(scan_id: str = ""):
    """Poll fallback for video scan progress when Socket.IO is unavailable."""
    token = (scan_id or request.args.get("scan_id") or "").strip()
    if not token:
        with _device_scan_results_lock:
            active = sorted(_active_device_probe_scans)
        if active:
            token = active[-1]
    if not token:
        return _nocache_jsonify(
            {"phase": "unknown", "is_scanning": False, "progress": 0}, 404
        )
    out = _scan_progress_payload(token)
    if not out:
        return _nocache_jsonify(
            {
                "phase": "unknown",
                "scan_id": token,
                "is_scanning": False,
                "progress": 0,
            },
            404,
        )
    return _nocache_jsonify(out)


@app.route("/api/target_detail", methods=["POST"])
def api_target_detail():
    """
    Enrich a single detection hit with age/gender for the target detail modal.
    Accepts JSON body with hit fields (frame, thumbnail, video_file, etc.).
    """
    payload = request.get_json(silent=True) or {}
    hit = payload.get("hit") if isinstance(payload.get("hit"), dict) else payload
    folder_raw = (payload.get("folder_path") or "").strip().strip('"').strip("'")
    folder_path = Path(folder_raw) if folder_raw else None
    video_raw = (payload.get("video_path") or "").strip()
    video_path = Path(video_raw) if video_raw else None

    enriched = enrich_target_detail_payload(
        hit,
        video_path=video_path if video_path and video_path.is_file() else None,
        folder_path=folder_path if folder_path and folder_path.is_dir() else None,
    )
    return jsonify(enriched)


@app.route("/api/upload_target", methods=["POST"])
def api_upload_target():
    """Persist the dashboard reference target image for presence trace scans."""
    upload_file = request.files.get("target_image") or request.files.get("file")
    if upload_file is None:
        return jsonify({"error": "No target image uploaded", "ok": False}), 400
    if not upload_file.filename:
        return jsonify({"error": "Empty target image filename", "ok": False}), 400

    try:
        save_uploaded_image(upload_file, TARGET_UPLOAD_PATH)
    except ValueError as exc:
        return jsonify({"error": str(exc), "ok": False}), 400

    return jsonify(
        {
            "ok": True,
            "success": True,
            "path": "/static/uploads/target.jpg",
            "filename": TARGET_UPLOAD_PATH.name,
        }
    )


def _watchlist_alert_gate_percent() -> int:
    """Live-camera Telegram / blacklist alert floor (0–100)."""
    raw = (
        os.environ.get("TELEGRAM_ALERT_MIN_PERCENT")
        or os.environ.get("WATCHLIST_ALERT_MIN_PERCENT")
        or ""
    ).strip()
    if raw:
        try:
            value = float(raw)
            if value <= 1.0:
                value *= 100.0
            return int(max(0, min(100, round(value))))
        except ValueError:
            pass
    try:
        from orion_frame_pipeline import BLACKLIST_ALERT_SIMILARITY

        return int(round(float(BLACKLIST_ALERT_SIMILARITY) * 100))
    except Exception:
        return 50


@app.route("/api/watchlist/enroll", methods=["POST"])
@app.route("/api/blacklist/enroll", methods=["POST"])
@app.route("/api/named/enroll", methods=["POST"])
def api_watchlist_enroll():
    """
    Enroll one or more face photos into known_persons.

    ``/api/watchlist/enroll`` → Watchlist alerts.
    ``/api/blacklist/enroll`` → Blacklist alerts (separate list).
    ``/api/named/enroll`` → named gallery (Telegram named alerts, not watchlist).
    """
    as_watchlist = True
    as_blacklist = False
    path = str(request.path or "")
    mode = (request.form.get("enroll_mode") or "").strip().lower()
    flag = (request.form.get("as_watchlist") or "").strip().lower()
    bl_flag = (request.form.get("as_blacklist") or "").strip().lower()
    if "/blacklist/" in path or mode == "blacklist" or bl_flag in ("1", "true", "yes", "on"):
        as_blacklist = True
        as_watchlist = False
    elif "/named/" in path or mode == "named" or flag in ("0", "false", "no", "off"):
        as_watchlist = False
        as_blacklist = False
    upload_files = []
    for key in ("target_images", "images", "target_image", "image"):
        for item in request.files.getlist(key):
            if item is not None and getattr(item, "filename", None):
                upload_files.append(item)
    # Deduplicate identical Werkzeug file wrappers by identity.
    seen_ids = set()
    unique_files = []
    for item in upload_files:
        oid = id(item)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        unique_files.append(item)
    upload_files = unique_files[:12]

    if not upload_files:
        # Also accept raw multipart keys browsers sometimes send.
        for key in list(request.files.keys()):
            for item in request.files.getlist(key):
                if item is not None and getattr(item, "filename", None):
                    upload_files.append(item)
        seen_ids = set()
        unique_files = []
        for item in upload_files:
            oid = id(item)
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
            unique_files.append(item)
        upload_files = unique_files[:12]
    if not upload_files:
        return jsonify(
            {
                "ok": False,
                "error": "No face photo received. Click ADD PHOTOS, then save.",
            }
        ), 400

    default_name = (
        (request.form.get("name") or request.form.get("person_name") or "").strip()
    )[:80]
    per_names = [str(n).strip()[:80] for n in request.form.getlist("names")]
    while len(per_names) < len(upload_files):
        per_names.append("")
    if as_watchlist:
        default_name = default_name or "Watchlist Subject"
    elif as_blacklist:
        default_name = default_name or "Blacklist Subject"
    elif not default_name and not any(per_names):
        return jsonify(
            {
                "ok": False,
                "error": "Type a name for this person, then Add to named people.",
            }
        ), 400
    elif not default_name:
        default_name = next((n for n in per_names if n), "Named Person")
    list_enroll = bool(as_watchlist or as_blacklist)
    labeled_uploads = list(zip(upload_files, per_names[: len(upload_files)]))

    force_person_raw = (request.form.get("person_id") or "").strip()
    force_person_id = int(force_person_raw) if force_person_raw.isdigit() else None

    alert_gate = _watchlist_alert_gate_percent()
    db = get_db_path()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _notify_live_cameras() -> None:
        try:
            from watchlist_vector_db import notify_watchlist_changed, sync_watchlist_from_sqlite

            sync_watchlist_from_sqlite(db_path=db, notify=True)
        except Exception:
            try:
                from watchlist_vector_db import notify_watchlist_changed

                notify_watchlist_changed()
            except Exception:
                pass

    def _force_enroll_portrait(
        conn: sqlite3.Connection,
        person_id: int,
        rows: Sequence[Tuple[np.ndarray, Optional[float], np.ndarray, Path, str]],
    ) -> None:
        """Always save a roster thumbnail after manual enroll (even if quality unchanged)."""
        if not rows:
            return
        from identity import _feature_to_blob, _save_best_face_image

        feature, quality, img, _path, _name = max(
            rows, key=lambda row: float(row[1] or 0.0)
        )
        if img is None or getattr(img, "size", 0) == 0:
            return
        image_path = _save_best_face_image(int(person_id), img)
        if not image_path:
            return
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            """
            INSERT INTO person_best_faces
                (known_person_id, encoding, quality_score, image_path, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(known_person_id) DO UPDATE SET
                image_path = excluded.image_path,
                updated_at = excluded.updated_at
            """,
            (
                int(person_id),
                _feature_to_blob(feature),
                float(quality or 0.0),
                image_path,
                now,
            ),
        )

    def _resolve_person_id(
        conn: sqlite3.Connection,
        *,
        name: str,
        feature: np.ndarray,
        quality: Optional[float],
    ) -> Tuple[Optional[int], bool, int]:
        """Return (person_id or None, updated_existing, similarity_pct)."""
        if force_person_id is not None:
            return force_person_id, True, 100
        if name:
            row = conn.execute(
                """
                SELECT id FROM known_persons
                WHERE lower(trim(name)) = lower(trim(?))
                ORDER BY id DESC LIMIT 1
                """,
                (name,),
            ).fetchone()
            if row is not None:
                return (
                    int(row["id"] if isinstance(row, sqlite3.Row) else row[0]),
                    True,
                    100,
                )
            # Named gallery: a new name always creates a new person — never
            # face-merge into someone else (that swapped roster names/photos).
            if not list_enroll:
                return None, False, 100
        matched_id, match_distance = find_best_face_match(
            feature, conn, query_quality=quality
        )
        if matched_id is not None and float(match_distance) <= float(MATCH_THRESHOLD):
            return (
                int(matched_id),
                True,
                int(round(max(0.0, 1.0 - float(match_distance)) * 100)),
            )
        return None, False, 100

    saved_paths: List[Path] = []
    try:
        from identity import (
            embed_named_enrollment_variants,
            face_sharpness,
            load_bgr_image_oriented,
            refresh_gpu_face_index,
            resolve_known_person_name,
            upsert_best_face_profile,
        )

        # Each row: feature, quality, img, path, display_name
        # ``feature`` is the primary (upright) embed; rotation variants are stored after register.
        embedded_rows: List[
            Tuple[np.ndarray, Optional[float], np.ndarray, Path, str]
        ] = []
        embed_errors: List[str] = []
        # path -> all rotation variants for Named enroll
        named_variant_bank: Dict[str, List[Tuple[np.ndarray, float, str]]] = {}

        for upload_file, raw_name in labeled_uploads:
            label = (raw_name or default_name or Path(upload_file.filename).stem.replace("_", " ").strip() or ("Blacklist Subject" if as_blacklist else ("Watchlist Subject" if as_watchlist else "Named Person")))[:80]
            suffix = _image_suffix_from_filename(upload_file.filename)
            save_path = SCAN_CACHE_DIR / f"watchlist_enroll_{uuid.uuid4().hex[:10]}{suffix}"
            try:
                save_path = save_uploaded_image(upload_file, save_path)
            except ValueError as exc:
                embed_errors.append(str(exc))
                continue
            saved_paths.append(save_path)
            img = load_bgr_image_oriented(save_path)
            if img is None or img.size == 0:
                img = cv2.imread(str(save_path))
            if img is None or img.size == 0:
                embed_errors.append(f"{upload_file.filename}: could not read image")
                continue
            if list_enroll:
                from identity import _embed_scan_reference_face

                embedded = _embed_scan_reference_face(img)
                if embedded is None:
                    embed_errors.append(f"{upload_file.filename}: no face detected")
                    continue
                feature, meta = embedded
                quality = None
                try:
                    crop = None
                    if isinstance(meta, dict):
                        crop = meta.get("aligned_face") or meta.get("face_crop")
                    if crop is None:
                        crop = img
                    quality = float(face_sharpness(np.asarray(crop)))
                except Exception:
                    quality = None
                embedded_rows.append((feature, quality, img, save_path, label))
            else:
                variants = embed_named_enrollment_variants(img)
                if not variants:
                    embed_errors.append(f"{upload_file.filename}: no face detected")
                    continue
                primary = next(
                    (v for v in variants if v[2] in ("original", "rot0")),
                    variants[0],
                )
                feature, quality, _lab = primary
                named_variant_bank[str(save_path)] = variants
                embedded_rows.append((feature, quality, img, save_path, label))

        if not embedded_rows:
            detail = "; ".join(embed_errors[:3]) if embed_errors else "no usable faces"
            return jsonify(
                {
                    "ok": False,
                    "error": f"No face detected in the uploaded photo(s). {detail}",
                }
            ), 400

        # Group photos by person name (case-insensitive).
        groups: Dict[str, List[Tuple[np.ndarray, Optional[float], np.ndarray, Path, str]]] = {}
        for row in embedded_rows:
            key = row[4].strip().lower() or "watchlist subject"
            groups.setdefault(key, []).append(row)

        people_results: List[Dict[str, Any]] = []
        total_encodings = 0
        last_db_error = None

        for attempt in range(5):
            conn = None
            try:
                conn = get_connection(db)
                init_identity_db(conn)
                people_results = []
                total_encodings = 0

                for _key, rows in groups.items():
                    display_name = rows[0][4]
                    first_feature, first_quality, first_img, _p, _n = rows[0]
                    target_id, updated_existing, similarity_pct = _resolve_person_id(
                        conn,
                        name=display_name,
                        feature=first_feature,
                        quality=first_quality,
                    )
                    encodings_added = 0
                    if target_id is None:
                        person_id = int(
                            register_person(
                                conn,
                                first_feature,
                                quality_score=first_quality,
                                name=display_name,
                                watchlist=as_watchlist,
                                named=True,
                            )
                        )
                        encodings_added += 1
                        if as_blacklist:
                            try:
                                from identity import mark_person_blacklist

                                mark_person_blacklist(conn, person_id, blacklist=True)
                            except Exception:
                                pass
                        try:
                            upsert_best_face_profile(
                                conn,
                                person_id,
                                first_feature,
                                first_img,
                                first_quality or 0.0,
                                allow_named=not list_enroll,
                            )
                        except TypeError:
                            try:
                                upsert_best_face_profile(
                                    conn,
                                    person_id,
                                    first_feature,
                                    first_img,
                                    first_quality or 0.0,
                                )
                            except Exception:
                                pass
                        except Exception:
                            pass
                        for feature, quality, img, path, _name in rows[1:]:
                            if list_enroll:
                                add_encoding_for_person(conn, person_id, feature, quality)
                                encodings_added += 1
                            else:
                                variants = named_variant_bank.get(str(path)) or [
                                    (feature, float(quality or 0.0), "original")
                                ]
                                for feat, q, _lab in variants:
                                    add_encoding_for_person(conn, person_id, feat, q)
                                    encodings_added += 1
                            try:
                                upsert_best_face_profile(
                                    conn,
                                    person_id,
                                    feature,
                                    img,
                                    quality or 0.0,
                                    allow_named=not list_enroll,
                                )
                            except TypeError:
                                try:
                                    upsert_best_face_profile(
                                        conn, person_id, feature, img, quality or 0.0
                                    )
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        if not list_enroll:
                            # Rotation variants for the first photo (register_person stored upright only).
                            first_path = str(rows[0][3])
                            for feat, q, lab in named_variant_bank.get(first_path) or []:
                                if lab in ("original", "rot0"):
                                    continue
                                add_encoding_for_person(conn, person_id, feat, q)
                                encodings_added += 1
                    else:
                        person_id = int(target_id)
                        try:
                            conn.execute(
                                "UPDATE known_persons SET name = ?, last_seen = ? WHERE id = ?",
                                (
                                    display_name,
                                    datetime.now().isoformat(timespec="seconds"),
                                    person_id,
                                ),
                            )
                        except Exception:
                            pass
                        try:
                            from identity import (
                                mark_person_blacklist,
                                mark_person_named,
                                mark_person_watchlist,
                            )

                            mark_person_named(conn, person_id, named=True)
                            if as_blacklist:
                                mark_person_blacklist(conn, person_id, blacklist=True)
                            elif as_watchlist:
                                mark_person_watchlist(conn, person_id, watchlist=True)
                            else:
                                mark_person_watchlist(conn, person_id, watchlist=False)
                                mark_person_blacklist(conn, person_id, blacklist=False)
                        except Exception:
                            pass
                        for feature, quality, img, path, _name in rows:
                            if list_enroll:
                                add_encoding_for_person(conn, person_id, feature, quality)
                                encodings_added += 1
                            else:
                                variants = named_variant_bank.get(str(path)) or [
                                    (feature, float(quality or 0.0), "original")
                                ]
                                for feat, q, _lab in variants:
                                    add_encoding_for_person(conn, person_id, feat, q)
                                    encodings_added += 1
                            try:
                                upsert_best_face_profile(
                                    conn,
                                    person_id,
                                    feature,
                                    img,
                                    quality or 0.0,
                                    allow_named=not list_enroll,
                                )
                            except TypeError:
                                try:
                                    upsert_best_face_profile(
                                        conn, person_id, feature, img, quality or 0.0
                                    )
                                except Exception:
                                    pass
                            except Exception:
                                pass

                    person_name = (
                        resolve_known_person_name(conn, person_id) or display_name
                    )
                    try:
                        from identity import (
                            mark_person_blacklist,
                            mark_person_named,
                            mark_person_watchlist,
                        )

                        mark_person_named(conn, person_id, named=True)
                        if as_blacklist:
                            mark_person_blacklist(conn, person_id, blacklist=True)
                        elif as_watchlist:
                            mark_person_watchlist(conn, person_id, watchlist=True)
                        else:
                            mark_person_watchlist(conn, person_id, watchlist=False)
                            mark_person_blacklist(conn, person_id, blacklist=False)
                    except Exception:
                        pass
                    try:
                        _force_enroll_portrait(conn, person_id, rows)
                    except Exception:
                        pass
                    if not list_enroll:
                        try:
                            from named_gallery import archive_enroll_batch

                            archive_enroll_batch(conn, person_id, rows)
                        except Exception:
                            pass
                    total_encodings += encodings_added
                    people_results.append(
                        {
                            "person_id": person_id,
                            "name": person_name,
                            "updated_existing": updated_existing,
                            "photos": len(rows),
                            "encodings_added": encodings_added,
                            "similarity_percent": similarity_pct,
                        }
                    )

                conn.commit()
                try:
                    refresh_gpu_face_index(conn)
                except Exception:
                    pass
                last_db_error = None
                break
            except sqlite3.OperationalError as exc:
                last_db_error = exc
                if "locked" not in str(exc).lower() or attempt >= 4:
                    raise
                time.sleep(0.35 * (attempt + 1))
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        if not people_results:
            raise RuntimeError(
                f"Watchlist enroll failed after retries: {last_db_error}"
            )

        _notify_live_cameras()

        try:
            shutil.copy2(embedded_rows[0][3], TARGET_UPLOAD_PATH)
        except OSError:
            pass

        primary = people_results[0]
        skipped = len(upload_files) - len(embedded_rows)
        return jsonify(
            {
                "ok": True,
                "success": True,
                "person_id": primary["person_id"],
                "name": primary["name"],
                "updated_existing": any(p["updated_existing"] for p in people_results),
                "people": people_results,
                "people_enrolled": len(people_results),
                "photos_submitted": len(upload_files),
                "photos_enrolled": len(embedded_rows),
                "photos_skipped": skipped,
                "encodings_added": total_encodings,
                "similarity_percent": primary["similarity_percent"],
                "alert_gate_percent": alert_gate if list_enroll else None,
                "is_watchlist": bool(as_watchlist),
                "is_blacklist": bool(as_blacklist),
                "is_named": True,
                "message": (
                    f"Enrolled {len(embedded_rows)} photo(s) across "
                    f"{len(people_results)} person(s)."
                ),
                "telegram_hint": (
                    f"Live cameras will Telegram-alert when a face matches ≥{alert_gate}%."
                    if list_enroll
                    else "Named people are identified on capture. They are not added to Watchlist/Blacklist."
                ),
                "warnings": embed_errors[:5],
            }
        )
    except Exception as exc:
        kind = (
            "Blacklist"
            if as_blacklist
            else ("Watchlist" if as_watchlist else "Named people")
        )
        return jsonify({"ok": False, "error": f"{kind} enroll failed: {exc}"}), 500
    finally:
        for path in saved_paths:
            try:
                if path.is_file() and path.resolve() != TARGET_UPLOAD_PATH.resolve():
                    path.unlink()
            except OSError:
                pass


@app.route("/api/named/people", methods=["GET"])
def api_named_people():
    """Roster of people enrolled in the Named gallery (not watchlist alerts)."""
    db = get_db_path()
    if not db.exists():
        return jsonify({"ok": True, "people": []})
    conn = get_connection(db)
    try:
        from identity import ensure_named_flag_column, ensure_watchlist_flag_column

        ensure_watchlist_flag_column(conn)
        ensure_named_flag_column(conn)
        portraits = fetch_profile_images_by_person(conn)
        try:
            from named_gallery import person_gallery_summary
        except Exception:
            person_gallery_summary = None
        rows = conn.execute(
            """
            SELECT id, name, last_seen, is_watchlist, is_named
            FROM known_persons
            WHERE COALESCE(is_named, 0) = 1
              AND COALESCE(is_watchlist, 0) = 0
            ORDER BY last_seen DESC, id DESC
            LIMIT 80
            """
        ).fetchall()
        people = []
        for row in rows:
            pid = int(row["id"])
            gallery = (
                person_gallery_summary(pid) if person_gallery_summary else {}
            )
            # If the gallery has no photos anymore, don't show the person in the roster.
            # This keeps the UX consistent after "Delete all photos".
            if int(gallery.get("photo_count", 0) or 0) <= 0:
                continue
            photo_url = gallery.get("photo_url") or portraits.get(pid)
            people.append(
                {
                    "person_id": pid,
                    "name": str(row["name"] or f"Person_{pid}").strip(),
                    "last_seen": row["last_seen"],
                    "photo_url": photo_url,
                    "photo_count": gallery.get("photo_count", 0),
                    "photos": gallery.get("photos") or [],
                    "is_watchlist": False,
                    "is_named": True,
                }
            )
        return jsonify({"ok": True, "people": people, "count": len(people)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "people": []}), 500
    finally:
        conn.close()


@app.route("/api/named/people", methods=["DELETE"])
def api_named_people_clear():
    """Delete all Named people gallery entries and their portrait images."""
    db = get_db_path()
    if not db.exists():
        return jsonify({"ok": True, "deleted_people": 0, "deleted_images": 0})
    conn = get_connection(db)
    try:
        from identity import delete_named_gallery_people

        stats = delete_named_gallery_people(conn)
        return jsonify({"ok": True, **stats})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        conn.close()


@app.route("/api/named/people/<int:person_id>", methods=["PATCH"])
def api_named_person_update(person_id: int):
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "name required"}), 400
    db = get_db_path()
    conn = get_connection(db)
    try:
        from identity import ensure_named_flag_column

        ensure_named_flag_column(conn)
        row = conn.execute(
            """
            SELECT id FROM known_persons
            WHERE id = ? AND COALESCE(is_named, 0) = 1 AND COALESCE(is_watchlist, 0) = 0
            """,
            (person_id,),
        ).fetchone()
        if row is None:
            return jsonify({"ok": False, "error": "Named person not found"}), 404
        conn.execute(
            "UPDATE known_persons SET name = ? WHERE id = ?",
            (name, person_id),
        )
        conn.commit()
        return jsonify({"ok": True, "person_id": person_id, "name": name})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        conn.close()


@app.route("/api/named/people/<int:person_id>", methods=["GET"])
def api_named_person_detail(person_id: int):
    db = get_db_path()
    if not db.exists():
        return jsonify({"ok": False, "error": "Database not found"}), 404
    conn = get_connection(db)
    try:
        from identity import ensure_named_flag_column, ensure_watchlist_flag_column
        from named_gallery import person_gallery_summary

        ensure_named_flag_column(conn)
        ensure_watchlist_flag_column(conn)
        row = conn.execute(
            """
            SELECT id, name, last_seen, is_named, is_watchlist
            FROM known_persons WHERE id = ?
            """,
            (person_id,),
        ).fetchone()
        if row is None or int(row["is_named"] or 0) != 1:
            return jsonify({"ok": False, "error": "Named person not found"}), 404
        if int(row["is_watchlist"] or 0) == 1:
            return jsonify({"ok": False, "error": "Person is on watchlist"}), 400
        gallery = person_gallery_summary(person_id)
        return jsonify(
            {
                "ok": True,
                "person_id": person_id,
                "name": str(row["name"] or "").strip(),
                "last_seen": row["last_seen"],
                **gallery,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        conn.close()


@app.route("/api/named/people/<int:person_id>", methods=["DELETE"])
def api_named_person_delete(person_id: int):
    db = get_db_path()
    conn = get_connection(db)
    try:
        from identity import ensure_named_flag_column

        ensure_named_flag_column(conn)
        row = conn.execute(
            "SELECT id FROM known_persons WHERE id = ? AND COALESCE(is_named,0)=1 AND COALESCE(is_watchlist,0)=0",
            (person_id,),
        ).fetchone()
        if row is None:
            return jsonify({"ok": False, "error": "Named person not found"}), 404
        # Reuse bulk delete for one id via targeted deletes
        from named_gallery import remove_person_gallery
        from identity import refresh_gpu_face_index

        images = remove_person_gallery(person_id)
        conn.execute("DELETE FROM person_encodings WHERE known_person_id = ?", (person_id,))
        conn.execute("DELETE FROM person_best_faces WHERE known_person_id = ?", (person_id,))
        conn.execute("DELETE FROM known_persons WHERE id = ?", (person_id,))
        conn.commit()
        try:
            refresh_gpu_face_index(conn)
        except Exception:
            pass
        try:
            from watchlist_vector_db import notify_watchlist_changed

            notify_watchlist_changed()
        except Exception:
            pass
        return jsonify({"ok": True, "deleted_people": 1, "deleted_images": images})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        conn.close()


@app.route("/api/named/people/<int:person_id>/primary", methods=["POST"])
def api_named_person_set_primary(person_id: int):
    payload = request.get_json(silent=True) or {}
    filename = str(payload.get("filename") or request.form.get("filename") or "").strip()
    if not filename:
        return jsonify({"ok": False, "error": "filename required"}), 400
    db = get_db_path()
    conn = get_connection(db)
    try:
        from named_gallery import set_primary_photo

        if not set_primary_photo(conn, person_id, filename):
            return jsonify({"ok": False, "error": "Photo not found"}), 404
        conn.commit()
        return jsonify({"ok": True, "filename": Path(filename).name})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        conn.close()


@app.route("/api/named/people/<int:person_id>/photos/<filename>", methods=["DELETE"])
def api_named_person_delete_photo(person_id: int, filename: str):
    db = get_db_path()
    conn = get_connection(db)
    try:
        from named_gallery import delete_gallery_photo, person_gallery_summary

        if not delete_gallery_photo(conn, person_id, filename):
            return jsonify({"ok": False, "error": "Photo not found"}), 404
        conn.commit()
        return jsonify({"ok": True, **person_gallery_summary(person_id)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        conn.close()


@app.route("/api/named/people/<int:person_id>/photos", methods=["DELETE"])
def api_named_person_delete_all_photos(person_id: int):
    db = get_db_path()
    conn = get_connection(db)
    try:
        from named_gallery import clear_gallery_photos, person_gallery_summary

        deleted = clear_gallery_photos(conn, person_id)
        conn.commit()
        return jsonify({"ok": True, "deleted_photos": deleted, **person_gallery_summary(person_id)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        conn.close()


@app.route("/api/named/people/<int:person_id>/photos", methods=["POST"])
def api_named_person_add_photos(person_id: int):
    db = get_db_path()
    conn = get_connection(db)
    try:
        from identity import (
            add_encoding_for_person,
            embed_named_enrollment_variants,
            ensure_named_flag_column,
            load_bgr_image_oriented,
            refresh_gpu_face_index,
        )
        from named_gallery import archive_enroll_batch

        ensure_named_flag_column(conn)
        row = conn.execute(
            "SELECT id, name FROM known_persons WHERE id = ? AND COALESCE(is_named,0)=1",
            (person_id,),
        ).fetchone()
        if row is None:
            return jsonify({"ok": False, "error": "Named person not found"}), 404

        upload_files = []
        for key in ("images", "target_images", "image"):
            upload_files.extend(request.files.getlist(key))
        if not upload_files:
            return jsonify({"ok": False, "error": "No photos received"}), 400

        added = 0
        SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        batch_rows = []
        for upload in upload_files[:12]:
            suffix = _image_suffix_from_filename(upload.filename)
            save_path = SCAN_CACHE_DIR / f"named_add_{uuid.uuid4().hex[:10]}{suffix}"
            try:
                save_path = save_uploaded_image(upload, save_path)
            except ValueError:
                continue
            img = load_bgr_image_oriented(save_path)
            if img is None or img.size == 0:
                img = cv2.imread(str(save_path))
            if img is None or img.size == 0:
                try:
                    save_path.unlink()
                except OSError:
                    pass
                continue
            variants = embed_named_enrollment_variants(img)
            if not variants:
                try:
                    save_path.unlink()
                except OSError:
                    pass
                continue
            primary = next(
                (v for v in variants if v[2] in ("original", "rot0")),
                variants[0],
            )
            feature, quality, _lab = primary
            for feat, q, _l in variants:
                add_encoding_for_person(conn, person_id, feat, q)
            batch_rows.append((feature, quality, img, save_path, row["name"]))
            added += 1

        if batch_rows:
            archive_enroll_batch(conn, person_id, batch_rows)
            conn.commit()
            try:
                refresh_gpu_face_index(conn)
            except Exception:
                pass
            try:
                from watchlist_vector_db import notify_watchlist_changed

                notify_watchlist_changed()
            except Exception:
                pass
        for _f, _q, _i, path, _n in batch_rows:
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass

        from named_gallery import person_gallery_summary

        return jsonify(
            {
                "ok": True,
                "photos_added": added,
                **person_gallery_summary(person_id),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        conn.close()


@app.route("/api/telegram/demo_alert", methods=["POST"])
def api_telegram_demo_alert():
    """
    Send a realistic sample watchlist Telegram alert using a live camera frame
    and the enrolled / uploaded face (not a text placeholder canvas).
    """
    try:
        from telegram_alerts import send_demo_watchlist_alert, telegram_configured
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Telegram module unavailable: {exc}"}), 500

    if not telegram_configured():
        return jsonify(
            {
                "ok": False,
                "error": "Telegram is not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.",
            }
        ), 400

    payload = request.get_json(silent=True) or {}
    name = (
        (request.form.get("name") if request.form else None)
        or payload.get("name")
        or ""
    )
    name = str(name).strip()

    person_id = None
    person_name = name or "Watchlist Subject"
    db = get_db_path()
    if db.is_file():
        try:
            conn = get_connection(db)
            try:
                if name:
                    row = conn.execute(
                        "SELECT id, name FROM known_persons WHERE name = ? ORDER BY id DESC LIMIT 1",
                        (name,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT id, name FROM known_persons ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                if row is not None:
                    person_id = int(row["id"] if isinstance(row, sqlite3.Row) else row[0])
                    resolved = row["name"] if isinstance(row, sqlite3.Row) else row[1]
                    if resolved and str(resolved).strip():
                        person_name = str(resolved).strip()
            finally:
                conn.close()
        except Exception:
            pass

    face_path = TARGET_UPLOAD_PATH if TARGET_UPLOAD_PATH.is_file() else None
    ok = send_demo_watchlist_alert(
        face_path=face_path,
        person_name=person_name,
        person_id=person_id,
        confidence=91,
        camera_name="CAM-01",
        location="Main Gate",
    )
    if not ok:
        return jsonify(
            {
                "ok": False,
                "error": "Demo alert failed — need a live camera frame or enrolled face photo.",
            }
        ), 500
    return jsonify(
        {
            "ok": True,
            "success": True,
            "person_id": person_id,
            "name": person_name,
            "message": f"Demo Telegram alert sent for {person_name}.",
        }
    )


@app.route("/api/trace_target", methods=["POST"])
def api_trace_target():
    """
    Compare the saved target image (static/uploads/target.jpg) against a video file
    or RTSP/local video path selected in the UI.
    """
    if not TARGET_UPLOAD_PATH.is_file():
        return jsonify(
            {
                "error": "No target image on server. Upload a target photo first.",
                "success": False,
                "matches": [],
            }
        ), 400

    video_upload = request.files.get("video")
    video_path_raw = (request.form.get("video_path") or "").strip().strip('"').strip("'")
    stream_key = (request.form.get("stream_key") or "").strip().lower()
    has_upload = bool(video_upload and video_upload.filename)
    has_path = bool(video_path_raw)

    if not has_upload and not has_path and stream_key:
        cameras = _load_cameras_config()
        cam_index = 1 if stream_key in ("cam02", "2", "02") else 0
        if cam_index < len(cameras):
            video_path_raw = str(cameras[cam_index].get("source") or "").strip()
            has_path = bool(video_path_raw)

    source_count = sum((has_upload, has_path))
    if source_count == 0:
        return jsonify(
            {
                "error": (
                    "No video source provided. Upload a video file, select a camera "
                    "stream, or supply video_path."
                ),
                "success": False,
                "matches": [],
            }
        ), 400
    if source_count > 1:
        return jsonify(
            {
                "error": "Provide only one video source: video upload or video_path.",
                "success": False,
                "matches": [],
            }
        ), 400

    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    cleanup_paths: List[Path] = []
    video_source_input = ""
    original_filename: Optional[str] = None

    if has_upload:
        vid_suffix = Path(video_upload.filename).suffix.lower()
        if vid_suffix not in VIDEO_SEARCH_EXTENSIONS:
            return jsonify(
                {
                    "error": (
                        "Unsupported video format. Use "
                        + ", ".join(sorted(VIDEO_SEARCH_EXTENSIONS))
                        + "."
                    ),
                    "success": False,
                    "matches": [],
                }
            ), 400
        vid_path = SCAN_CACHE_DIR / f"{session_id}_video{vid_suffix}"
        try:
            save_uploaded_video(video_upload, vid_path)
        except VideoCodecError as exc:
            body, status = _video_upload_error_response(exc)
            return body, status
        cleanup_paths.append(vid_path)
        video_source_input = str(vid_path)
        original_filename = Path(video_upload.filename).name
    else:
        video_source_input = video_path_raw
        if not video_path_raw.startswith("rtsp://"):
            path_obj = Path(video_path_raw)
            if path_obj.is_file():
                original_filename = path_obj.name

    scan_id = (request.form.get("scan_id") or "").strip() or uuid.uuid4().hex[:12]
    frame_step_raw = (request.form.get("frame_step") or "").strip()
    frame_step = int(frame_step_raw) if frame_step_raw.isdigit() else None

    try:
        result, status = _run_tactical_target_scan_api(
            TARGET_UPLOAD_PATH,
            video_source_input,
            scan_id=scan_id,
            frame_step=frame_step,
            original_filename=original_filename,
        )
        return jsonify(result), status
    except VideoCodecError as exc:
        body, err_status = _video_upload_error_response(exc)
        return body, err_status
    finally:
        _release_scan_job(scan_id)
        for path in cleanup_paths:
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass


@app.route("/api/detect_smoking", methods=["POST"])
def api_detect_smoking():
    """
    Scan uploaded video or selected live stream for smoking/cigarette infractions.
    """
    video_upload = request.files.get("video")
    video_path_raw = (request.form.get("video_path") or "").strip().strip('"').strip("'")
    stream_key = (request.form.get("stream_key") or "").strip().lower()
    has_upload = bool(video_upload and video_upload.filename)
    has_path = bool(video_path_raw)

    if not has_upload and not has_path and stream_key:
        cameras = _load_cameras_config()
        cam_index = 1 if stream_key in ("cam02", "2", "02") else 0
        if cam_index < len(cameras):
            cam = cameras[cam_index]
            video_path_raw = str(cam.get("source") or "").strip()
            has_path = bool(video_path_raw)

    source_count = sum((has_upload, has_path))
    if source_count == 0:
        return jsonify(
            {
                "ok": False,
                "error": (
                    "No video source provided. Upload a video file, select a camera "
                    "stream, or supply video_path."
                ),
                "results": [],
            }
        ), 400
    if source_count > 1:
        return jsonify(
            {
                "ok": False,
                "error": "Provide only one video source: video upload or video_path.",
                "results": [],
            }
        ), 400

    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    cleanup_paths: List[Path] = []
    video_source_input = ""
    source_label = "Main Gate - Cam 01"

    if has_upload:
        vid_suffix = Path(video_upload.filename).suffix.lower()
        if vid_suffix not in VIDEO_SEARCH_EXTENSIONS:
            return jsonify(
                {
                    "ok": False,
                    "error": (
                        "Unsupported video format. Use "
                        + ", ".join(sorted(VIDEO_SEARCH_EXTENSIONS))
                        + "."
                    ),
                    "results": [],
                }
            ), 400
        vid_path = SCAN_CACHE_DIR / f"{session_id}_smoking{vid_suffix}"
        try:
            save_uploaded_video(video_upload, vid_path)
        except VideoCodecError as exc:
            body, status = _video_upload_error_response(exc)
            return body, status
        cleanup_paths.append(vid_path)
        video_source_input = str(vid_path)
        source_label = Path(video_upload.filename).name or source_label
    else:
        video_source_input = video_path_raw
        if stream_key in ("cam02", "2", "02"):
            source_label = "Main Gate - Cam 02"
        else:
            source_label = "Main Gate - Cam 01"

    scan_id = (request.form.get("scan_id") or "").strip() or uuid.uuid4().hex[:12]
    frame_step_raw = (request.form.get("frame_step") or "").strip()
    frame_step = int(frame_step_raw) if frame_step_raw.isdigit() else 12

    try:
        _arm_scan_session(scan_id)
        payload = _scan_video_for_smoking_infraction(
            video_source_input,
            source_label=source_label,
            frame_step=frame_step,
            scan_id=scan_id,
        )
        card_results = payload.get("results") or []
        events = payload.get("events") or []
        count = len(card_results)
        if count > 0:
            increment_detection_counters(count)
        _finalize_scan_session(scan_id)
        warning_text = (
            f"⚠️ Warning: Smoking detected at {source_label}"
            if count > 0
            else f"No smoking infraction detected at {source_label}"
        )
        return jsonify(
            {
                "ok": True,
                "success": True,
                "scan_id": scan_id,
                "results": card_results,
                "events": events,
                "count": count,
                "source_label": source_label,
                "warning_text": warning_text,
                "message": (
                    f"Smoking scan completed — {count} infraction(s) detected."
                    if count > 0
                    else "Smoking scan completed — no infractions found."
                ),
            }
        )
    except VideoCodecError as exc:
        _release_scan_session(scan_id, "error")
        body, status = _video_upload_error_response(exc)
        return body, status
    except Exception as exc:
        _release_scan_session(scan_id, "error")
        return (
            jsonify({"ok": False, "error": f"Smoking scan failed: {exc}", "results": []}),
            500,
        )
    finally:
        _release_scan_job(scan_id)
        for path in cleanup_paths:
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass


@app.route("/api/scan_for_target", methods=["POST"])
def api_scan_for_target():
    """
    Unified target-face scan — auto-detects single clip vs batch folder from inputs.

    Requires ``target_image``. Provide exactly one video source:
      - ``video`` file upload (single clip),
      - ``folder_path`` (batch directory, e.g. E:\\msa),
      - ``video_path`` (local file path or rtsp:// stream).
    """
    if "target_image" not in request.files:
        return jsonify({"error": "No target face image uploaded"}), 400

    image_file = request.files["target_image"]
    if not image_file.filename:
        return jsonify({"error": "Empty target image filename"}), 400

    storage_mode = (request.form.get("storage_mode") or "local").strip().lower()
    if storage_mode not in ("local", "server"):
        storage_mode = "local"

    folder_path = (request.form.get("folder_path") or "").strip().strip('"').strip("'")
    video_path_raw = (request.form.get("video_path") or "").strip().strip('"').strip("'")
    video_upload = request.files.get("video")
    has_upload = bool(video_upload and video_upload.filename)
    has_folder = bool(folder_path)
    has_path = bool(video_path_raw)

    source_count = sum((has_upload, has_folder, has_path))
    if source_count == 0:
        return jsonify(
            {
                "error": (
                    "No video source provided. Upload a video file, or supply "
                    "folder_path (batch) or video_path (single clip / RTSP)."
                )
            }
        ), 400
    if source_count > 1:
        return jsonify(
            {"error": "Provide only one video source: video upload, folder_path, or video_path."}
        ), 400

    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    img_suffix = _image_suffix_from_filename(image_file.filename)
    img_path = SCAN_CACHE_DIR / f"{session_id}_target{img_suffix}"
    cleanup_paths: List[Path] = []

    try:
        img_path = save_uploaded_image(image_file, img_path)
    except ValueError as exc:
        return jsonify({"error": str(exc), "success": False, "matches": []}), 400

    video_source_input = ""
    original_filename: Optional[str] = None

    if has_upload:
        vid_suffix = Path(video_upload.filename).suffix.lower()
        if vid_suffix not in VIDEO_SEARCH_EXTENSIONS:
            return jsonify(
                {
                    "error": (
                        "Unsupported video format. Use "
                        + ", ".join(sorted(VIDEO_SEARCH_EXTENSIONS))
                        + "."
                    )
                }
            ), 400
        vid_path = SCAN_CACHE_DIR / f"{session_id}_video{vid_suffix}"
        try:
            save_uploaded_video(video_upload, vid_path)
        except VideoCodecError as exc:
            body, status = _video_upload_error_response(exc)
            return body, status
        cleanup_paths.append(vid_path)
        video_source_input = str(vid_path)
        original_filename = Path(video_upload.filename).name
    elif has_folder:
        video_source_input = folder_path
    else:
        video_source_input = video_path_raw
        if not video_path_raw.startswith("rtsp://"):
            path_obj = Path(video_path_raw)
            if path_obj.is_file():
                original_filename = path_obj.name

    scan_id = (request.form.get("scan_id") or "").strip() or uuid.uuid4().hex[:12]
    frame_step_raw = (request.form.get("frame_step") or "").strip()
    frame_step = int(frame_step_raw) if frame_step_raw.isdigit() else None

    if has_folder:
        with _target_scan_results_lock:
            if scan_id in _active_target_scans:
                return jsonify({"status": "already_scanning", "scan_id": scan_id}), 409
            _active_target_scans.add(scan_id)
            _target_scan_results.pop(scan_id, None)

        worker = threading.Thread(
            target=_target_folder_scan_worker,
            args=(img_path, folder_path, scan_id, storage_mode, frame_step),
            daemon=True,
            name=f"target-folder-scan-{scan_id[:8]}",
        )
        worker.start()
        return jsonify(
            {
                "success": True,
                "status": "scan_started",
                "scan_id": scan_id,
                "message": "Batch folder scan started in background.",
            }
        ), 202

    try:
        result, status = _run_tactical_target_scan_api(
            img_path,
            video_source_input,
            scan_id=scan_id,
            storage_mode=storage_mode,
            frame_step=frame_step,
            original_filename=original_filename,
        )
        return jsonify(result), status
    except VideoCodecError as exc:
        body, err_status = _video_upload_error_response(exc)
        return body, err_status
    finally:
        _release_scan_job(scan_id)
        cleanup_paths.append(img_path)
        for path in cleanup_paths:
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass


@app.route("/api/target_scan_results/<scan_id>")
def api_target_scan_results(scan_id: str = ""):
    """Poll finalized batch folder target scan results."""
    token = (scan_id or request.args.get("scan_id") or "").strip()
    if not token:
        return jsonify({"error": "scan_id required", "success": False, "matches": []}), 400
    with _target_scan_results_lock:
        payload = _target_scan_results.get(token)
        still_active = token in _active_target_scans
    if payload is None:
        if still_active:
            progress = _scan_progress_payload(token)
            return jsonify(
                {
                    "success": False,
                    "status": "scanning",
                    "scan_id": token,
                    "matches": [],
                    "progress": progress,
                }
            ), 202
        return jsonify(
            {"error": "No results for scan_id", "success": False, "scan_id": token, "matches": []}
        ), 404
    result_payload = dict(payload)
    http_status = int(result_payload.pop("http_status", 200))
    return jsonify(result_payload), http_status


@app.route("/api/scan_video_for_target", methods=["POST"])
def api_scan_video_for_target():
    """Scan an uploaded CCTV clip for appearances of a target face photo."""
    if "target_image" not in request.files:
        return jsonify({"error": "No target face image uploaded"}), 400
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    image_file = request.files["target_image"]
    video_file = request.files["video"]
    if not image_file.filename:
        return jsonify({"error": "Empty target image filename"}), 400
    if not video_file.filename:
        return jsonify({"error": "Empty video filename"}), 400

    img_suffix = _image_suffix_from_filename(image_file.filename)
    vid_suffix = Path(video_file.filename).suffix.lower()
    if vid_suffix not in VIDEO_SEARCH_EXTENSIONS:
        return jsonify(
            {
                "error": (
                    "Unsupported video format. Use "
                    + ", ".join(sorted(VIDEO_SEARCH_EXTENSIONS))
                    + "."
                )
            }
        ), 400

    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    img_path = SCAN_CACHE_DIR / f"{session_id}_target{img_suffix}"
    vid_path = SCAN_CACHE_DIR / f"{session_id}_video{vid_suffix}"

    try:
        img_path = save_uploaded_image(image_file, img_path)
    except ValueError as exc:
        return jsonify({"error": str(exc), "success": False, "matches": []}), 400

    try:
        save_uploaded_video(video_file, vid_path)
    except VideoCodecError as exc:
        body, status = _video_upload_error_response(exc)
        return body, status

    scan_id = (request.form.get("scan_id") or "").strip() or uuid.uuid4().hex[:12]
    frame_step_raw = (request.form.get("frame_step") or "").strip()
    frame_step = int(frame_step_raw) if frame_step_raw.isdigit() else None
    original_filename = Path(video_file.filename).name

    try:
        result, status = _run_tactical_target_scan_api(
            img_path,
            str(vid_path),
            scan_id=scan_id,
            frame_step=frame_step,
            original_filename=original_filename,
        )
        return jsonify(result), status
    except VideoCodecError as exc:
        body, err_status = _video_upload_error_response(exc)
        return body, err_status
    finally:
        _release_scan_job(scan_id)
        for path in (img_path, vid_path):
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass


@app.route("/api/scan_folder_for_target", methods=["POST"])
def api_scan_folder_for_target():
    """Batch-scan all videos in a local folder for a target face."""
    if "target_image" not in request.files:
        return jsonify({"error": "No target face image uploaded"}), 400

    image_file = request.files["target_image"]
    if not image_file.filename:
        return jsonify({"error": "Empty target image filename"}), 400

    folder_path = (request.form.get("folder_path") or "").strip().strip('"').strip("'")
    if not folder_path:
        return jsonify({"error": "No folder path provided."}), 400

    storage_mode = (request.form.get("storage_mode") or "local").strip().lower()
    if storage_mode not in ("local", "server"):
        storage_mode = "local"

    try:
        folder = normalize_batch_folder_path(folder_path, storage_mode)
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    if not folder.is_dir():
        return jsonify(
            {"error": f"Folder not found or not accessible: {folder_path}"}
        ), 400

    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    img_suffix = _image_suffix_from_filename(image_file.filename)
    img_path = SCAN_CACHE_DIR / f"{session_id}_target{img_suffix}"

    try:
        img_path = save_uploaded_image(image_file, img_path)
    except ValueError as exc:
        return jsonify({"error": str(exc), "success": False, "matches": []}), 400

    scan_id = (request.form.get("scan_id") or "").strip() or uuid.uuid4().hex[:12]

    try:
        result, status = _run_tactical_target_scan_api(
            img_path,
            folder_path,
            scan_id=scan_id,
            storage_mode=storage_mode,
        )
        return jsonify(result), status
    except VideoCodecError as exc:
        body, err_status = _video_upload_error_response(exc)
        return body, err_status
    finally:
        _release_scan_job(scan_id)
        if img_path.is_file():
            try:
                img_path.unlink()
            except OSError:
                pass


_UNAUTHORIZED_DEVICE_YOLO: Dict[str, Tuple[str, str]] = {
    "cell phone": ("Smartphone", "HIGH"),
    "laptop": ("Laptop", "HIGH"),
    "mouse": ("USB Storage", "MEDIUM"),
}
_DEVICE_PROBE_TARGET_CLASS_IDS = frozenset({63, 67})  # COCO: 63 laptop, 67 cell phone
_DEVICE_PROBE_CONF_MIN = 0.5
_DEVICE_PROBE_LOG_THROTTLE_SEC = 1.5
_SMOKING_CONF_MIN = 0.35
_SMOKING_LABEL_HINTS = ("smoke", "smoking", "cigarette", "cigar", "vape", "vaping")
_IMAGE_FACE_ANALYZER: Any = None
_IMAGE_FACE_ANALYZER_LOCK = threading.Lock()


def _get_image_face_analyzer() -> Any:
    global _IMAGE_FACE_ANALYZER
    with _IMAGE_FACE_ANALYZER_LOCK:
        if _IMAGE_FACE_ANALYZER is None:
            from face_analyzer import FaceAnalyzer

            _IMAGE_FACE_ANALYZER = FaceAnalyzer()
        return _IMAGE_FACE_ANALYZER


def _encode_bgr_frame_as_data_uri(frame: np.ndarray, quality: int = 88) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _annotate_device_probe_frame(
    frame: np.ndarray,
    detections: List[Tuple[str, float, np.ndarray]],
) -> np.ndarray:
    annotated = frame.copy()
    for label, conf_frac, bbox in detections:
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        h, w = annotated.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"{label} ({int(conf_frac * 100)}%)",
            (x1, max(y1 - 10, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _save_device_probe_snapshot(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    label: str,
    scan_token: str,
) -> str:
    """Draw a red bounding box and persist a probe still under captures/device_probe/."""
    DEVICE_PROBE_DIR.mkdir(parents=True, exist_ok=True)
    annotated = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy[:4]]
    h, w = annotated.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
    tag = label.replace(" ", "_")[:24]
    cv2.putText(
        annotated,
        label,
        (x1, max(y1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    fname = f"{scan_token}_{tag}_{uuid.uuid4().hex[:8]}.jpg"
    rel = f"device_probe/{fname}"
    out_path = CAPTURES_DIR / rel
    cv2.imwrite(str(out_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return rel.replace("\\", "/")


def _save_annotated_probe_jpeg(frame: np.ndarray, scan_token: str, tag: str) -> str:
    DEVICE_PROBE_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{scan_token}_{tag}_{uuid.uuid4().hex[:8]}.jpg"
    rel = f"device_probe/{fname}"
    out_path = CAPTURES_DIR / rel
    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return rel.replace("\\", "/")


def _infraction_public_url(filename: str) -> str:
    name = Path(str(filename or "")).name
    return f"/static/infractions/{name}"


def _local_infraction_path(url_or_name: str) -> Optional[Path]:
    raw = str(url_or_name or "").strip()
    if not raw or raw.startswith("data:") or raw.startswith("http"):
        return None
    for prefix in ("/static/infractions/", "static/infractions/"):
        if raw.startswith(prefix):
            return INFRACTIONS_STATIC_DIR / Path(raw[len(prefix) :]).name
    if raw.startswith("/static/snapshots/") or raw.startswith("static/snapshots/"):
        name = Path(raw).name
        candidate = SNAPSHOTS_STATIC_DIR / name
        return candidate if candidate.is_file() else None
    name = Path(raw).name
    if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        for folder in (INFRACTIONS_STATIC_DIR, SNAPSHOTS_STATIC_DIR):
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


def _register_device_log_snapshot(row_id: int, snapshot_rel: Optional[str]) -> Optional[str]:
    """
    Persist a unique review JPEG at static/infractions/{event_id}_{timestamp}.jpg
    and return its public URL for the Live AI Infraction modal.
    """
    if not row_id:
        return None

    INFRACTIONS_STATIC_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_name = f"{int(row_id)}_{ts}.jpg"
    dest = INFRACTIONS_STATIC_DIR / dest_name
    public_url = _infraction_public_url(dest_name)

    src: Optional[Path] = None
    if snapshot_rel:
        snap = str(snapshot_rel).replace("\\", "/").strip()
        if snap.startswith("/static/infractions/") or snap.startswith("static/infractions/"):
            existing = _local_infraction_path(snap)
            if existing is not None and existing.is_file():
                return _infraction_public_url(existing.name)
        src = CAPTURES_DIR / snap.replace("/", os.sep)
        if not src.is_file():
            src = _local_capture_path(snap)
        if src is None or not src.is_file():
            src = _local_infraction_path(snap)

    if src is not None and src.is_file():
        try:
            shutil.copy2(src, dest)
            # Keep a stable alias for older clients that still request /static/snapshots/<id>.jpg
            SNAPSHOTS_STATIC_DIR.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, SNAPSHOTS_STATIC_DIR / f"{int(row_id)}.jpg")
            except OSError:
                pass
            return public_url
        except OSError:
            if str(snapshot_rel).startswith("/"):
                return str(snapshot_rel)
            return _capture_public_url(str(snapshot_rel)) if snapshot_rel else None

    # Already stored under static/infractions with this event id prefix
    matches = sorted(INFRACTIONS_STATIC_DIR.glob(f"{int(row_id)}_*.jpg"))
    if matches:
        return _infraction_public_url(matches[-1].name)

    legacy = SNAPSHOTS_STATIC_DIR / f"{int(row_id)}.jpg"
    if legacy.is_file():
        try:
            shutil.copy2(legacy, dest)
            return public_url
        except OSError:
            return f"/static/snapshots/{int(row_id)}.jpg"
    return None


def _resolve_device_log_image_url(row: Dict[str, Any]) -> Optional[str]:
    """Resolve a browser-loadable snapshot URL for an infraction log row (no remote placeholder)."""
    row_id = row.get("id")
    if row_id is not None:
        matches = sorted(INFRACTIONS_STATIC_DIR.glob(f"{int(row_id)}_*.jpg"))
        if matches:
            return _infraction_public_url(matches[-1].name)
        alias = SNAPSHOTS_STATIC_DIR / f"{int(row_id)}.jpg"
        if alias.is_file():
            return f"/static/snapshots/{int(row_id)}.jpg"

    snap = row.get("snapshot_path")
    image_hint = row.get("image_url") or row.get("snapshot_url")

    for candidate in (image_hint, snap):
        if not candidate:
            continue
        text = str(candidate).strip()
        if text.startswith("/static/infractions/"):
            local = _local_infraction_path(text)
            if local is not None and local.is_file():
                return _infraction_public_url(local.name)
            return text
        if text.startswith("/static/snapshots/"):
            name = Path(text).name
            if (SNAPSHOTS_STATIC_DIR / name).is_file():
                return f"/static/snapshots/{name}"
        if text.startswith("/captures/") or (
            not text.startswith("http")
            and not text.startswith("data:")
            and not text.startswith("/static/")
        ):
            local = _local_capture_path(text) or (
                CAPTURES_DIR / text.replace("/", os.sep)
            )
            if local is not None and local.is_file():
                # Promote durable unique copy when we know the event id.
                if row_id is not None:
                    promoted = _register_device_log_snapshot(int(row_id), text)
                    if promoted:
                        return promoted
                return _capture_public_url(
                    str(local.relative_to(CAPTURES_DIR)).replace("\\", "/")
                )
    return None


def _camera_key_from_row(item: Dict[str, Any]) -> str:
    for key in ("camera_id", "camera", "camera_location", "source_channel"):
        raw = str(item.get(key) or "")
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            return digits
    return ""


def _face_capture_path_from_row(item: Dict[str, Any]) -> Optional[Path]:
    for key in ("capture_path", "snapshot_path", "image_url", "face_url"):
        raw = str(item.get(key) or "").replace("\\", "/").strip()
        if not raw:
            continue
        name = raw.split("/")[-1]
        if not name:
            continue
        if name.startswith("http"):
            continue
        # Prefer real face captures under captures/
        cand = CAPTURES_DIR / name
        if cand.is_file() and ("_face_" in name or name.endswith(".jpg")):
            if "_scene_" in name:
                continue
            return cand
        local = _local_capture_path(raw)
        if local is not None and local.is_file() and "_scene_" not in local.name:
            return local
    return None


def _public_captures_url(local: Path) -> str:
    try:
        rel = local.resolve().relative_to(CAPTURES_DIR.resolve())
        return _capture_public_url(str(rel).replace("\\", "/"))
    except Exception:
        return f"/captures/{local.name}"


def _hallway_review_url_from_scene_file(local: Path) -> Optional[str]:
    """
    Return a Review-friendly full-corridor image URL.

    Legacy ``*_scene_*`` files are often Telegram side-by-side collages
    (hallway | face). Extract the left hallway panel so Review shows the
    wide CCTV corridor, not the face crop or collage strip.
    """
    if local is None or not local.is_file():
        return None
    try:
        import cv2

        im = cv2.imread(str(local))
        if im is None or im.size == 0:
            return None
        h, w = int(im.shape[0]), int(im.shape[1])
        if min(h, w) < 80:
            return None
        ratio = float(w) / float(max(1, h))
        # Normal CCTV / annotated frames — use as-is.
        if ratio <= 2.15:
            return _public_captures_url(local)

        # Side-by-side collage: scene left + portrait panel right (not always square).
        gap = 3
        panel_w = min(h, max(int(round(h * 0.55)), 120))
        cut = max(1, w - panel_w - gap)
        if cut < int(w * 0.45):
            cut = int(round(w * 0.68))
        hall = im[:, :cut]
        if hall.size == 0:
            return _public_captures_url(local)

        out_name = local.name.replace("_scene_", "_hall_", 1)
        if out_name == local.name:
            out_name = f"{local.stem}_hall.jpg"
        out_path = CAPTURES_DIR / out_name
        need_write = True
        if out_path.is_file():
            try:
                need_write = out_path.stat().st_mtime < local.stat().st_mtime
            except OSError:
                need_write = True
        if need_write:
            CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(out_path),
                hall,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            ):
                return _public_captures_url(local)
        return _public_captures_url(out_path)
    except Exception:
        return _public_captures_url(local)


def _paired_scene_url_from_row(item: Dict[str, Any]) -> Optional[str]:
    """Return a full-corridor scene URL for Review (hallway only, not face crop)."""
    for key in ("scene_url", "full_frame", "scene_image", "scene_path"):
        raw = str(item.get(key) or "").replace("\\", "/").strip()
        if not raw or raw.startswith("/live_preview/"):
            continue
        name = raw.split("/")[-1]
        if not name or ("_face_" in name and "_scene_" not in name and "_hall_" not in name):
            continue
        if "review_scene_" in name:
            continue
        local = _local_capture_path(raw) or (CAPTURES_DIR / name)
        if local is not None and local.is_file():
            if "_hall_" in local.name or "_scene_" in local.name:
                url = _hallway_review_url_from_scene_file(local)
                if url:
                    return url

    face_path = _face_capture_path_from_row(item)
    if face_path is not None and "_face_" in face_path.name:
        for token in ("_scene_", "_hall_"):
            paired = CAPTURES_DIR / face_path.name.replace("_face_", token, 1)
            if paired.is_file():
                url = _hallway_review_url_from_scene_file(paired)
                if url:
                    return url
    return None


def _enrich_device_log_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        row_id = item.get("id")
        snap = item.get("snapshot_path")

        face_path = _face_capture_path_from_row(item)
        face_url = (
            f"/captures/{face_path.name}"
            if face_path is not None
            else _resolve_device_log_image_url(item)
        )
        if face_url and ("/_scene_" in face_url.replace("\\", "/") or "/_hall_" in face_url.replace("\\", "/")):
            face_url = None
        cap_path = item.get("capture_path") or face_path
        if face_url and cap_path and not _face_capture_file_ok(str(cap_path)):
            face_url = None
        if (not face_url) and row_id is not None and snap:
            snap_s = str(snap).replace("\\", "/")
            if "_scene_" not in snap_s and "review_scene_" not in snap_s and "_hall_" not in snap_s:
                face_url = _register_device_log_snapshot(int(row_id), snap)

        # Review primary image = full hallway corridor (wide-angle).
        scene_url = _paired_scene_url_from_row(item)

        item["face_url"] = face_url
        if scene_url:
            item["scene_url"] = scene_url
            item["full_frame"] = scene_url
            item["scene_image"] = scene_url
            item["image_url"] = scene_url
            item["snapshot_url"] = scene_url
        else:
            item["snapshot_url"] = face_url
            item["image_url"] = face_url
        enriched.append(item)
    return enriched


def _insert_device_log_event(conn, **kwargs) -> int:
    """Insert a device log row and register a unique static/infractions snapshot."""
    row_id = insert_device_detection_event(conn, **kwargs)
    public_url = _register_device_log_snapshot(row_id, kwargs.get("snapshot_path"))
    if public_url:
        try:
            conn.execute(
                "UPDATE device_detection_events SET snapshot_path = ? WHERE id = ?",
                (public_url, int(row_id)),
            )
            conn.commit()
        except Exception:
            pass
    return row_id


def _image_raw_features_preview(frame: np.ndarray, sample_count: int = 6) -> str:
    """Compact hex preview of the first RGB pixels (Edge Impulse-style)."""
    flat = frame.reshape(-1, 3)[:sample_count]
    parts = []
    for b, g, r in flat:
        parts.append(f"0x{r:02x}{g:02x}{b:02x}")
    return ", ".join(parts)


def _image_dsp_preview(frame: np.ndarray, size: int = 96) -> Tuple[str, str]:
    """Resize + normalize preview for DSP result panel."""
    thumb = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    norm = thumb.astype(np.float32) / 255.0
    flat = norm.reshape(-1, 3)[:8]
    feature_text = ", ".join(f"{v:.4f}" for v in flat[:, 0])
    uri = _encode_bgr_frame_as_data_uri(thumb, quality=82)
    return uri, feature_text


def _detect_faces_in_image(frame: np.ndarray) -> List[Tuple[str, float, np.ndarray]]:
    try:
        analyzer = _get_image_face_analyzer()
        faces = analyzer.detect(frame)
    except Exception:
        return []
    out: List[Tuple[str, float, np.ndarray]] = []
    for face in faces or []:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        score = float(face.score or 0.0)
        if score < 0.4:
            continue
        out.append(("face", score, np.array([x1, y1, x2, y2], dtype=np.float32)))
    return out


def _detect_devices_in_image(frame: np.ndarray) -> List[Tuple[str, float, np.ndarray, str, str]]:
    model = get_yolo()
    results = model.predict(frame, verbose=False, conf=_DEVICE_PROBE_CONF_MIN)
    detected: List[Tuple[str, float, np.ndarray, str, str]] = []
    if not results:
        return detected
    names = results[0].names or {}
    for box in results[0].boxes or []:
        cls_id = int(box.cls[0])
        if cls_id not in _DEVICE_PROBE_TARGET_CLASS_IDS:
            continue
        conf_frac = float(box.conf[0])
        if conf_frac < _DEVICE_PROBE_CONF_MIN:
            continue
        yolo_label = str(names.get(cls_id, "")).lower().strip()
        if yolo_label not in _UNAUTHORIZED_DEVICE_YOLO:
            continue
        device_type, severity = _UNAUTHORIZED_DEVICE_YOLO[yolo_label]
        bbox = box.xyxy[0].detach().cpu().numpy()
        detected.append((device_type, conf_frac, bbox, severity, yolo_label))
    return detected


def _scan_image_for_detections(
    image_path: Path,
    source_label: str,
    *,
    mode: str = "all",
) -> Dict[str, Any]:
    """Run YOLO device + optional face detection on a single still image."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise VideoCodecError("Could not decode image file.")

    mode_key = (mode or "all").lower().strip()
    if mode_key not in {"all", "devices", "face"}:
        mode_key = "all"

    t0 = time.perf_counter()
    draw_list: List[Tuple[str, float, np.ndarray]] = []
    card_meta: List[Tuple[str, float, np.ndarray, str, str, str]] = []

    if mode_key in ("all", "devices"):
        for device_type, conf_frac, bbox, severity, yolo_label in _detect_devices_in_image(
            frame
        ):
            draw_list.append((device_type, conf_frac, bbox))
            card_meta.append(
                (device_type, conf_frac, bbox, severity, yolo_label, "devices")
            )

    if mode_key in ("all", "face"):
        for label, conf_frac, bbox in _detect_faces_in_image(frame):
            draw_list.append((label, conf_frac, bbox))
            card_meta.append((label, conf_frac, bbox, "HIGH", "face", "face"))

    inference_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    annotated = _annotate_device_probe_frame(frame, draw_list) if draw_list else frame.copy()
    snap_token = uuid.uuid4().hex[:10]
    snapshot_rel = _save_annotated_probe_jpeg(annotated, snap_token, "img")
    snapshot_url = _capture_public_url(snapshot_rel)
    image_base64 = _encode_bgr_frame_as_data_uri(annotated)
    raw_features = _image_raw_features_preview(frame)
    dsp_uri, dsp_features = _image_dsp_preview(frame)
    time_str = datetime.now().strftime("%H:%M:%S")

    card_results: List[Dict[str, Any]] = []
    for idx, (label, conf_frac, bbox, severity, tag, kind) in enumerate(card_meta, start=1):
        conf_pct = round(conf_frac * 100.0, 1)
        display = "Face" if kind == "face" else f"{label} Detected"
        card_results.append(
            {
                "id": f"IMG-{idx:02d}",
                "type": display,
                "time": time_str,
                "match_score": f"{int(conf_frac * 100)}%",
                "image_base64": image_base64,
                "device_type": label if kind == "devices" else "Face",
                "source_channel": source_label,
                "severity": severity,
                "confidence": conf_pct,
                "frame_number": 0,
                "scan_source": "image_probe",
                "yolo_label": tag,
                "snapshot_path": snapshot_rel,
                "snapshot_url": snapshot_url,
            }
        )

    db = get_db_path()
    conn = get_connection(db)
    inserted: List[Dict[str, Any]] = []
    try:
        for card in card_results:
            row_id = _insert_device_log_event(
                conn,
                device_type=str(card.get("device_type") or "Unknown"),
                source_channel=source_label,
                severity=str(card.get("severity") or "HIGH"),
                confidence=float(card.get("confidence") or 0),
                scan_source="image_probe",
                snapshot_path=card.get("snapshot_path"),
                frame_number=0,
            )
            card["db_id"] = row_id
            inserted.append(dict(card))
    finally:
        conn.close()

    h, w = frame.shape[:2]
    return {
        "ok": True,
        "results": card_results,
        "events": inserted,
        "count": len(card_results),
        "annotated_image": image_base64,
        "raw_features": raw_features,
        "dsp_result": dsp_uri,
        "dsp_features": dsp_features,
        "inference_ms": inference_ms,
        "peak_ram_kb": max(4, int((w * h * 3) / 1024)),
        "width": w,
        "height": h,
        "message": (
            f"Image scan — {len(card_results)} detection(s) found."
            if card_results
            else "Image scan — no targets detected."
        ),
    }


def _scan_video_for_unauthorized_devices(
    video_path: Path,
    source_label: str,
    frame_step: int = 12,
    scan_id: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Sample frames with YOLO, emit live progress, and build visual breach cards."""
    token = str(scan_id or "").strip()
    model = get_yolo()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoCodecError("Could not open video file.")

    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

    def _push_progress(phase: str, pct: int, frame_current: int = 0) -> None:
        if not token:
            return
        clamped = max(0, min(100, int(pct)))
        _publish_scan_progress(
            {
                "scan_id": token,
                "phase": phase,
                "percent": clamped,
                "progress": clamped,
                "frame_current": frame_current,
                "frame_total": total_frames,
                "is_scanning": phase not in _DONE_SCAN_PHASES,
            }
        )

    best: Dict[str, Dict[str, Any]] = {}
    card_results: List[Dict[str, Any]] = []
    frame_idx = 0
    step = max(1, int(frame_step))
    last_published = -1
    last_logged_time = 0.0
    snap_token = token or uuid.uuid4().hex[:10]
    _push_progress("scanning", 0, 0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if token:
                pct = min(99, int((frame_idx / total_frames) * 100))
                if pct != last_published:
                    _push_progress("scanning", pct, frame_idx)
                    last_published = pct

            if frame_idx % step != 0:
                time.sleep(0.002)
                continue

            try:
                results = model.predict(frame, verbose=False, conf=_DEVICE_PROBE_CONF_MIN)
            except Exception:
                time.sleep(0.002)
                continue

            detected_objects: List[Tuple[str, float, np.ndarray, str, str]] = []
            if results:
                names = results[0].names or {}
                for box in results[0].boxes or []:
                    cls_id = int(box.cls[0])
                    if cls_id not in _DEVICE_PROBE_TARGET_CLASS_IDS:
                        continue
                    conf_frac = float(box.conf[0])
                    if conf_frac < _DEVICE_PROBE_CONF_MIN:
                        continue
                    yolo_label = str(names.get(cls_id, "")).lower().strip()
                    if yolo_label not in _UNAUTHORIZED_DEVICE_YOLO:
                        continue
                    device_type, severity = _UNAUTHORIZED_DEVICE_YOLO[yolo_label]
                    bbox = box.xyxy[0].detach().cpu().numpy()
                    conf_pct = conf_frac * 100.0
                    detected_objects.append(
                        (device_type, conf_frac, bbox, severity, yolo_label)
                    )
                    prev = best.get(yolo_label)
                    if prev is None or conf_pct > float(prev["conf"]):
                        best[yolo_label] = {
                            "conf": conf_pct,
                            "frame": frame.copy(),
                            "bbox": bbox,
                            "frame_idx": frame_idx,
                            "device_type": device_type,
                            "severity": severity,
                        }

            now = time.time()
            if (
                detected_objects
                and (now - last_logged_time) >= _DEVICE_PROBE_LOG_THROTTLE_SEC
            ):
                draw_list = [(d[0], d[1], d[2]) for d in detected_objects]
                annotated = _annotate_device_probe_frame(frame, draw_list)
                image_base64 = _encode_bgr_frame_as_data_uri(annotated)
                snapshot_rel = _save_annotated_probe_jpeg(
                    annotated, snap_token, f"frm{frame_idx}"
                )
                snapshot_url = _capture_public_url(snapshot_rel)
                time_str = datetime.now().strftime("%H:%M:%S")

                for device_type, conf_frac, bbox, severity, yolo_label in detected_objects:
                    breach_id = f"BRCH-{len(card_results) + 1:02d}"
                    conf_pct = round(conf_frac * 100.0, 1)
                    card_results.append(
                        {
                            "id": breach_id,
                            "type": f"{device_type} Detected",
                            "time": time_str,
                            "match_score": f"{int(conf_frac * 100)}%",
                            "image_base64": image_base64,
                            "device_type": device_type,
                            "source_channel": source_label,
                            "severity": severity,
                            "confidence": conf_pct,
                            "frame_number": frame_idx,
                            "scan_source": "video_probe",
                            "yolo_label": yolo_label,
                            "snapshot_path": snapshot_rel,
                            "snapshot_url": snapshot_url,
                        }
                    )
                last_logged_time = now

            time.sleep(0.002)
    finally:
        cap.release()

    if token:
        _push_progress("scanning", 99, total_frames)

    db = get_db_path()
    conn = get_connection(db)
    inserted: List[Dict[str, Any]] = []
    try:
        if card_results:
            for card in card_results:
                row_id = _insert_device_log_event(
                    conn,
                    device_type=str(card.get("device_type") or "Unknown"),
                    source_channel=source_label,
                    severity=str(card.get("severity") or "HIGH"),
                    confidence=float(card.get("confidence") or 0),
                    scan_source="video_probe",
                    snapshot_path=card.get("snapshot_path"),
                    frame_number=int(card.get("frame_number") or 0),
                )
                card["db_id"] = row_id
                inserted.append(dict(card))
        else:
            for yolo_label, hit in best.items():
                device_type = str(hit.get("device_type") or _UNAUTHORIZED_DEVICE_YOLO[yolo_label][0])
                severity = str(hit.get("severity") or _UNAUTHORIZED_DEVICE_YOLO[yolo_label][1])
                conf = round(float(hit["conf"]), 1)
                snapshot_rel = _save_device_probe_snapshot(
                    hit["frame"],
                    hit["bbox"],
                    device_type,
                    snap_token,
                )
                image_base64 = _encode_bgr_frame_as_data_uri(
                    _annotate_device_probe_frame(
                        hit["frame"],
                        [(device_type, conf / 100.0, hit["bbox"])],
                    )
                )
                breach_id = f"BRCH-{len(card_results) + 1:02d}"
                card = {
                    "id": breach_id,
                    "type": f"{device_type} Detected",
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "match_score": f"{int(conf)}%",
                    "image_base64": image_base64,
                    "device_type": device_type,
                    "source_channel": source_label,
                    "severity": severity,
                    "confidence": conf,
                    "frame_number": int(hit.get("frame_idx") or 0),
                    "scan_source": "video_probe",
                    "snapshot_path": snapshot_rel,
                    "snapshot_url": _capture_public_url(snapshot_rel),
                    "yolo_label": yolo_label,
                }
                row_id = _insert_device_log_event(
                    conn,
                    device_type=device_type,
                    source_channel=source_label,
                    severity=severity,
                    confidence=conf,
                    scan_source="video_probe",
                    snapshot_path=snapshot_rel,
                    frame_number=int(hit.get("frame_idx") or 0),
                )
                card["db_id"] = row_id
                card_results.append(card)
                inserted.append(dict(card))
    finally:
        conn.close()

    return {"events": inserted, "results": card_results}


def _is_smoking_label(label: str) -> bool:
    tag = str(label or "").strip().lower()
    if not tag:
        return False
    return any(hint in tag for hint in _SMOKING_LABEL_HINTS)


def _scan_video_for_smoking_infraction(
    video_source: str,
    source_label: str,
    frame_step: int = 12,
    scan_id: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Sample frames with YOLO and extract smoking/cigarette infractions."""
    token = str(scan_id or "").strip()
    model = get_yolo()
    cap = cv2.VideoCapture(str(video_source))
    if not cap.isOpened():
        raise VideoCodecError("Could not open video source.")

    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    step = max(1, int(frame_step))
    frame_idx = 0
    last_published = -1
    last_logged_time = 0.0
    snap_token = token or uuid.uuid4().hex[:10]
    card_results: List[Dict[str, Any]] = []

    def _push_progress(phase: str, pct: int, frame_current: int = 0) -> None:
        if not token:
            return
        clamped = max(0, min(100, int(pct)))
        _publish_scan_progress(
            {
                "scan_id": token,
                "phase": phase,
                "percent": clamped,
                "progress": clamped,
                "frame_current": frame_current,
                "frame_total": total_frames,
                "is_scanning": phase not in _DONE_SCAN_PHASES,
            }
        )

    _push_progress("scanning", 0, 0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if token and total_frames > 0:
                pct = min(99, int((frame_idx / total_frames) * 100))
                if pct != last_published:
                    _push_progress("scanning", pct, frame_idx)
                    last_published = pct

            if frame_idx % step != 0:
                time.sleep(0.002)
                continue

            try:
                results = model.predict(frame, verbose=False, conf=_SMOKING_CONF_MIN)
            except Exception:
                time.sleep(0.002)
                continue

            smoking_hits: List[Tuple[str, float, np.ndarray]] = []
            if results:
                names = results[0].names or {}
                for box in results[0].boxes or []:
                    conf_frac = float(box.conf[0])
                    if conf_frac < _SMOKING_CONF_MIN:
                        continue
                    cls_id = int(box.cls[0])
                    yolo_label = str(names.get(cls_id, "")).strip().lower()
                    if not _is_smoking_label(yolo_label):
                        continue
                    bbox = box.xyxy[0].detach().cpu().numpy()
                    smoking_hits.append((yolo_label or "smoking", conf_frac, bbox))

            now = time.time()
            if smoking_hits and (now - last_logged_time) >= _DEVICE_PROBE_LOG_THROTTLE_SEC:
                draw_list = [("Smoking", hit[1], hit[2]) for hit in smoking_hits]
                annotated = _annotate_device_probe_frame(frame, draw_list)
                image_base64 = _encode_bgr_frame_as_data_uri(annotated)
                snapshot_rel = _save_annotated_probe_jpeg(
                    annotated, snap_token, f"smoking{frame_idx}"
                )
                snapshot_url = _capture_public_url(snapshot_rel)
                time_str = datetime.now().strftime("%H:%M:%S")
                for yolo_label, conf_frac, _bbox in smoking_hits:
                    conf_pct = round(conf_frac * 100.0, 1)
                    card_results.append(
                        {
                            "id": f"SMK-{len(card_results) + 1:02d}",
                            "type": "Smoking detected",
                            "time": time_str,
                            "match_score": f"{int(conf_frac * 100)}%",
                            "image_base64": image_base64,
                            "device_type": "Smoking",
                            "source_channel": source_label,
                            "severity": "HIGH",
                            "confidence": conf_pct,
                            "frame_number": frame_idx,
                            "scan_source": "smoking_scan",
                            "yolo_label": yolo_label,
                            "snapshot_path": snapshot_rel,
                            "snapshot_url": snapshot_url,
                        }
                    )
                last_logged_time = now
            time.sleep(0.002)
    finally:
        cap.release()

    if token:
        _push_progress("scanning", 99, total_frames)

    db = get_db_path()
    conn = get_connection(db)
    inserted: List[Dict[str, Any]] = []
    try:
        for card in card_results:
            row_id = _insert_device_log_event(
                conn,
                device_type="Smoking",
                source_channel=source_label,
                severity="HIGH",
                confidence=float(card.get("confidence") or 0),
                scan_source="smoking_scan",
                snapshot_path=card.get("snapshot_path"),
                frame_number=int(card.get("frame_number") or 0),
            )
            card["db_id"] = row_id
            inserted.append(dict(card))
    finally:
        conn.close()

    return {"events": inserted, "results": card_results}


def _target_folder_scan_worker(
    target_image_path: Path,
    folder_path: str,
    scan_id: str,
    storage_mode: str,
    frame_step: Optional[int],
) -> None:
    """Background thread — batch folder target trace scan without blocking HTTP."""
    token = str(scan_id or "").strip()
    try:
        result, status = _run_tactical_target_scan_api(
            target_image_path,
            folder_path,
            scan_id=token,
            storage_mode=storage_mode,
            frame_step=frame_step,
        )
        payload = dict(result)
        payload["http_status"] = int(status)
        with _target_scan_results_lock:
            _target_scan_results[token] = payload
    except VideoCodecError as exc:
        _, err_status = _video_upload_error_response(exc)
        with _target_scan_results_lock:
            _target_scan_results[token] = {
                "success": False,
                "error": VIDEO_CODEC_USER_MESSAGE,
                "matches": [],
                "http_status": err_status,
            }
    except Exception as exc:
        with _target_scan_results_lock:
            _target_scan_results[token] = {
                "success": False,
                "error": str(exc),
                "matches": [],
                "http_status": 500,
            }
    finally:
        with _target_scan_results_lock:
            _active_target_scans.discard(token)
        _release_scan_job(token)
        if target_image_path.is_file():
            try:
                target_image_path.unlink()
            except OSError:
                pass


def _device_probe_scan_worker(
    vid_path: Path,
    source_label: str,
    frame_step: int,
    scan_id: str,
) -> None:
    """Background thread — YOLO probe with live progress updates."""
    token = str(scan_id or "").strip()
    try:
        _require_cuda_video_scan()
        _arm_scan_session(token)
        events_payload = _scan_video_for_unauthorized_devices(
            vid_path,
            source_label=source_label,
            frame_step=frame_step,
            scan_id=token,
        )
        events = events_payload.get("events") or []
        card_results = events_payload.get("results") or []
        _finalize_scan_session(token)
        message = (
            f"Scan complete — {len(card_results)} unauthorized device(s) logged."
            if card_results
            else "Scan complete — no unauthorized devices found."
        )
        with _device_scan_results_lock:
            _device_scan_results[token] = {
                "ok": True,
                "scan_id": token,
                "events": events,
                "results": card_results,
                "count": len(card_results),
                "message": message,
            }
    except VideoCodecError as exc:
        _release_scan_session(token, "error")
        with _device_scan_results_lock:
            _device_scan_results[token] = {
                "ok": False,
                "scan_id": token,
                "error": str(exc),
                "events": [],
                "results": [],
                "count": 0,
            }
    except Exception as exc:
        _release_scan_session(token, "error")
        with _device_scan_results_lock:
            _device_scan_results[token] = {
                "ok": False,
                "scan_id": token,
                "error": str(exc),
                "events": [],
                "results": [],
                "count": 0,
            }
    finally:
        with _device_scan_results_lock:
            _active_device_probe_scans.discard(token)
        if vid_path.is_file():
            try:
                vid_path.unlink()
            except OSError:
                pass


@app.route("/api/device_detection_logs", methods=["GET", "POST"])
def api_device_detection_logs():
    """Fetch or append hardware integrity events stored in SQLite."""
    db = get_db_path()
    conn = get_connection(db)
    try:
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            row_id = _insert_device_log_event(
                conn,
                device_type=str(payload.get("device_type") or "Unknown"),
                source_channel=str(payload.get("source_channel") or "Live feed"),
                severity=str(payload.get("severity") or "HIGH").upper(),
                confidence=payload.get("confidence"),
                camera_id=(
                    str(payload.get("camera_id"))
                    if payload.get("camera_id") not in (None, "")
                    else None
                ),
                scan_source=str(payload.get("scan_source") or "live"),
                snapshot_path=(
                    str(payload.get("snapshot_path"))
                    if payload.get("snapshot_path") not in (None, "")
                    else None
                ),
                frame_number=payload.get("frame_number"),
            )
            rows = fetch_device_detection_logs(conn, limit=1)
            entry = _enrich_device_log_rows(rows[:1])[0] if rows else {"id": row_id}
            return jsonify({"ok": True, "id": row_id, "entry": entry})
        limit = min(request.args.get("limit", 50, type=int), 200)
        since_id = request.args.get("since_id", type=int)
        if since_id is not None and since_id > 0:
            # Incremental poll stays on native device_detection_events ids only.
            rows = fetch_device_detection_logs(conn, limit=limit, since_id=since_id)
        else:
            # Live face captures (tracking_logs) + smoking/device events.
            rows = fetch_live_infraction_feed(conn, limit=limit)
        return jsonify({"ok": True, "entries": _enrich_device_log_rows(rows), "count": len(rows)})
    finally:
        conn.close()


@app.route("/api/device_detection_logs/clear", methods=["POST"])
def api_device_detection_logs_clear():
    db = get_db_path()
    conn = get_connection(db)
    try:
        removed = clear_device_detection_events(conn)
        return jsonify({"ok": True, "removed": removed})
    finally:
        conn.close()


@app.route("/api/device_detection_scan_image", methods=["POST"])
def api_device_detection_scan_image():
    """Synchronous YOLO + face probe on a single uploaded image."""
    image_file = request.files.get("image")
    if image_file is None or not image_file.filename:
        return jsonify({"error": "No image uploaded", "ok": False, "status": "error"}), 400

    img_suffix = _image_suffix_from_filename(image_file.filename)
    if img_suffix not in IMAGE_UPLOAD_SUFFIXES:
        return jsonify(
            {
                "error": (
                    "Unsupported image format. Use "
                    + ", ".join(sorted(STANDARD_IMAGE_SUFFIXES))
                    + "."
                ),
                "ok": False,
                "status": "error",
            }
        ), 400

    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    img_path = SCAN_CACHE_DIR / f"{session_id}_device{img_suffix}"
    original_filename = Path(image_file.filename).name
    mode = (request.form.get("mode") or "all").strip().lower()

    try:
        save_uploaded_image(image_file, img_path)
        payload = _scan_image_for_detections(
            img_path,
            f"Image · {original_filename}",
            mode=mode,
        )
        return jsonify(payload)
    except VideoCodecError as exc:
        return jsonify({"error": str(exc), "ok": False, "status": "error"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc), "ok": False, "status": "error"}), 500
    finally:
        if img_path.is_file():
            try:
                img_path.unlink()
            except OSError:
                pass


@app.route("/api/device_detection_scan_video", methods=["POST"])
@app.route("/api/start_scan", methods=["POST"])
def api_device_detection_scan_video():
    """Start a background YOLO hardware probe — returns immediately for progress polling."""
    video_file = request.files.get("video")
    if video_file is None or not video_file.filename:
        return jsonify({"error": "No video uploaded", "ok": False, "status": "error"}), 400

    vid_suffix = Path(video_file.filename).suffix.lower()
    if vid_suffix not in VIDEO_SEARCH_EXTENSIONS:
        return jsonify(
            {
                "error": (
                    "Unsupported video format. Use "
                    + ", ".join(sorted(VIDEO_SEARCH_EXTENSIONS))
                    + "."
                ),
                "ok": False,
                "status": "error",
            }
        ), 400

    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    vid_path = SCAN_CACHE_DIR / f"{session_id}_device{vid_suffix}"
    original_filename = Path(video_file.filename).name
    frame_step_raw = (request.form.get("frame_step") or "").strip()
    frame_step = int(frame_step_raw) if frame_step_raw.isdigit() else 12
    scan_id = (request.form.get("scan_id") or "").strip() or uuid.uuid4().hex[:12]

    with _device_scan_results_lock:
        if scan_id in _active_device_probe_scans:
            return jsonify({"status": "already_scanning", "scan_id": scan_id, "ok": True})

    try:
        save_uploaded_video(video_file, vid_path)
    except VideoCodecError as exc:
        body, status = _video_upload_error_response(exc)
        return body, status

    with _device_scan_results_lock:
        _active_device_probe_scans.add(scan_id)
        _device_scan_results.pop(scan_id, None)

    worker = threading.Thread(
        target=_device_probe_scan_worker,
        args=(
            vid_path,
            f"Video · {original_filename}",
            frame_step,
            scan_id,
        ),
        daemon=True,
    )
    worker.start()
    return jsonify(
        {
            "ok": True,
            "status": "scan_started",
            "scan_id": scan_id,
            "message": "Hardware integrity probe started.",
        }
    )


@app.route("/api/device_detection_scan_results/<scan_id>")
@app.route("/api/scan_results")
def api_device_detection_scan_results(scan_id: str = ""):
    """Fetch finalized probe hits once background inference completes."""
    token = (scan_id or request.args.get("scan_id") or "").strip()
    if not token:
        return jsonify({"ok": False, "error": "scan_id required", "results": []}), 400
    with _device_scan_results_lock:
        payload = _device_scan_results.get(token)
    if not payload:
        progress = _scan_progress_payload(token)
        if progress and progress.get("is_scanning"):
            return jsonify(
                {
                    "ok": False,
                    "status": "scanning",
                    "scan_id": token,
                    "results": [],
                }
            ), 202
        return jsonify(
            {
                "ok": False,
                "error": "No results for scan_id",
                "scan_id": token,
                "results": [],
            }
        ), 404
    out = dict(payload)
    out["results"] = out.get("results") or out.get("events") or []
    return jsonify(out)


def _is_localhost_request() -> bool:
    addr = (request.remote_addr or "").strip()
    return addr in ("127.0.0.1", "::1", "localhost") or addr.startswith("127.")


@app.route("/internal/database-purged", methods=["POST"])
def internal_database_purged():
    """Localhost-only hook for run_system storage cleanup → Socket.IO broadcast."""
    if not _is_localhost_request():
        return jsonify({"error": "forbidden"}), 403
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "clear_all")
    socketio.emit("database_purged", {"message": message})
    return jsonify({"ok": True, "message": message})


def main() -> None:
    parser = argparse.ArgumentParser(description="Visitor tracking web dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    app.config["DB_PATH"] = args.db
    UPLOAD_DIR.mkdir(exist_ok=True)
    CAPTURES_DIR.mkdir(exist_ok=True)
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Database:  {args.db}")
    try:
        from telegram_alerts import telegram_configured

        print(
            "Telegram:    "
            + ("enabled" if telegram_configured() else "not configured (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env)")
        )
    except Exception as exc:
        print(f"Telegram:    unavailable ({exc})")
    print("Photo search: upload an image on the dashboard home page")
    try:
        attr_backend = initialize_age_gender_models()
        if attr_backend != "none":
            print(f"Age/gender models: {attr_backend} (buckets: {', '.join(FACE_ATTR_AGE_BUCKETS)})")
        else:
            print(
                "Age/gender models: unavailable — run download_face_attributes.py "
                "or pip install deepface"
            )
    except Exception as exc:
        print(f"Age/gender model preload skipped ({exc})")
    try:
        _require_cuda_video_scan()
        gc.collect()
    except Exception as exc:
        print(f"Note: CUDA video-scan preload skipped ({exc})")
    try:
        _ensure_person_alert_emitter()
    except Exception as exc:
        print(f"Note: person-alert WebSocket emitter skipped ({exc})")
    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=False,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
