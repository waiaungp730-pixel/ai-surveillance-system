#!/usr/bin/env python3
"""
Run CCTV tracking (main.py) and the web dashboard (dashboard.py) together.

Default with many cameras: --unified-tracker (one GPU process, far less CPU).
Use --per-camera-processes only when you need isolated RTSP processes.

Performance knobs (passed through to main.py):
  --frame-skip 3|5|8   YOLO every N frames; preview stays smooth
  --inference-size 640x360   AI letterbox only; display/crops stay full-res

  python run_system.py --device cuda
  python run_system.py --device cuda --unified-tracker --frame-skip 8
  python run_system.py --device cuda --per-camera-processes   # high CPU

Requires Python 3.12+ with PyTorch installed (use: py -3.12 run_system.py).
"""

from __future__ import annotations

import numpy_compat  # noqa: F401  — NumPy 1.x / 2.x import bridge

import argparse
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from camera_stats import read_faces_detected
from inference_device import (
    DEVICE_CHOICES,
    configure,
    describe_device,
    resolve_torch_device,
)
from orion_frame_pipeline import (
    audit_hardware_dependencies,
    init_face_restoration_stack,
    init_pure_gpu_video_scan,
)
from live_frame_hub import ensure_live_frame_dir
from storage_cleanup import (
    CLEANUP_INTERVAL_SEC,
    DEFAULT_DB_PATH,
    RETENTION_HOURS,
    start_storage_cleanup_daemon,
)
from telegram_alerts import TELEGRAM_ALERT_STYLE, telegram_configured
import tracker_config

CameraConfig = Union[Tuple[str, str], Tuple[str, str, str]]

PROJECT_DIR = Path(__file__).resolve().parent
CAMERAS_CONFIG_PATH = PROJECT_DIR / "cameras.json"
ENV_PATH = PROJECT_DIR / ".env"
from yolo_loader import DEFAULT_MODEL as YOLO_MODEL
SYSTEM_STATUS_PATH = PROJECT_DIR / "system_status.json"

STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")
CAMERA_VALIDATE_TIMEOUT = 5.0
CAMERA_VALIDATE_TCP_PROBE_SEC = float(os.environ.get("CAMERA_VALIDATE_TCP_PROBE_SEC", "1.2"))
CAMERA_VALIDATE_WORKERS = max(1, int(os.environ.get("CAMERA_VALIDATE_WORKERS", "8")))
CAMERA_START_STAGGER_SEC = 2.0
OSNET_WEIGHTS = PROJECT_DIR / "models" / "osnet_x1_0_reid.pth"

_CREDENTIAL_MASK = re.compile(r"//([^:]+):([^@]+)@")

# ── Restart policy ─────────────────────────────────────────────────────────────
MAX_RESTARTS = 10
RESTART_BASE_DELAY = 3.0
RESTART_MAX_DELAY = 60.0
CRASH_STALE_SECONDS = 600.0

_ENV_PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("[SYSTEM]")

MIN_PYTHON = (3, 12)


def _ensure_runtime_python() -> Optional[int]:
    """
    Refuse to start on the wrong interpreter (e.g. bare `python` -> 3.8 with no torch).
    Returns an exit code when startup should abort, else None.
    """
    if sys.version_info < MIN_PYTHON:
        exe = sys.executable
        log.error(
            "Python %s.%s is too old and missing project dependencies.",
            sys.version_info.major,
            sys.version_info.minor,
        )
        log.error("Interpreter: %s", exe)
        log.error(
            "Use Python 3.12 instead:\n"
            "  py -3.12 run_system.py --device cuda"
        )
        return 1
    try:
        import torch  # noqa: F401
    except ImportError:
        log.error("PyTorch is not installed for this Python: %s", sys.executable)
        log.error(
            "Install on Python 3.12:\n"
            "  py -3.12 -m pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu128\n"
            "Then run:\n"
            "  py -3.12 run_system.py --device cuda"
        )
        return 1
    return None

# ── Camera config ──────────────────────────────────────────────────────────────
# Edit cameras.json (or keep the list below) — never put credentials in code.
#
#   cameras.json example:
#   [
#     {"id": "1", "source": "${CAM_1_URL}", "location": "front_door"},
#     {"id": "2", "source": "${CAM_2_URL}", "location": "back_door"}
#   ]
#
# Fallback list used only when cameras.json does NOT exist yet.
# 27 cameras: G1–G3 (7 each) + G4 (6) → 192.168.2.201–227
_RTSP_PATH = "/cam/realmonitor?channel=1&subtype=0"


def _camera_group_location(cam_id: int) -> Tuple[str, str]:
    if cam_id <= 7:
        return "G1", f"G1-{cam_id:02d}"
    if cam_id <= 14:
        return "G2", f"G2-{cam_id - 7:02d}"
    if cam_id <= 21:
        return "G3", f"G3-{cam_id - 14:02d}"
    return "G4", f"G4-{cam_id - 21:02d}"


def _build_fallback_tracker_configs() -> List[CameraConfig]:
    configs: List[CameraConfig] = []
    for cam_id in range(1, 28):
        _group, location = _camera_group_location(cam_id)
        ip = 200 + cam_id
        source = f"rtsp://192.168.2.{ip}:554{_RTSP_PATH}"
        configs.append((str(cam_id), source, location))
    return configs


_FALLBACK_TRACKER_CONFIGS: List[CameraConfig] = _build_fallback_tracker_configs()


def _load_dotenv(path: Path = ENV_PATH) -> None:
    """Load .env into os.environ without overwriting existing variables."""
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        log.info("Loaded environment from %s (python-dotenv)", path.name)
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

    log.info("Loaded environment from %s (built-in parser)", path.name)


def _rtsp_env_credentials() -> Tuple[str, str]:
    user = (
        os.environ.get("RTSP_USER")
        or os.environ.get("RTSP_USERNAME")
        or ""
    ).strip()
    password = (os.environ.get("RTSP_PASSWORD") or "").strip()
    return user, password


def _apply_rtsp_env_credentials(source: str) -> str:
    """Inject RTSP_USER/RTSP_PASSWORD when the URL has no userinfo."""
    if not isinstance(source, str) or not source.lower().startswith("rtsp://"):
        return source
    user, password = _rtsp_env_credentials()
    if not user or not password:
        return source
    rest = source[7:]
    authority = rest.split("/", 1)[0]
    if "@" in authority:
        return source
    from main import sanitize_rtsp_url

    return sanitize_rtsp_url(f"rtsp://{user}:{password}@{rest}")


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} placeholders with environment values."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if var_name in os.environ:
            return os.environ[var_name]
        log.warning(
            "Environment variable %s is not set (placeholder in camera source)",
            var_name,
        )
        return match.group(0)

    return _ENV_PLACEHOLDER.sub(_replace, value)


def _load_tracker_configs() -> List[Tuple[str, str, str]]:
    """Load camera list from cameras.json when present, else fall back."""
    if CAMERAS_CONFIG_PATH.is_file():
        raw = json.loads(CAMERAS_CONFIG_PATH.read_text(encoding="utf-8"))
        configs = []
        for entry in raw:
            name = entry.get("name") or entry.get("id")
            if not name:
                raise ValueError(
                    f"Camera entry in {CAMERAS_CONFIG_PATH.name} needs 'id' or 'name': {entry!r}"
                )
            source = _apply_rtsp_env_credentials(_resolve_env_vars(entry["source"]))
            try:
                from main import sanitize_rtsp_url

                if str(source).lower().startswith("rtsp://"):
                    source = sanitize_rtsp_url(source)
            except Exception:
                pass
            location = entry.get("location", _default_location(name))
            configs.append((name, source, location))
        return configs

    configs: List[Tuple[str, str, str]] = []
    for entry in _FALLBACK_TRACKER_CONFIGS:
        name, source, location = _normalize_camera_config(entry)
        env_key = f"CAM_{name}_URL"
        source = os.environ.get(env_key, _resolve_env_vars(source))
        source = _apply_rtsp_env_credentials(source)
        configs.append((name, source, location))
    return configs


def _default_location(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_camera_config(entry: CameraConfig) -> Tuple[str, str, str]:
    if len(entry) == 3:
        name, source, location = entry
        return name, source, location
    if len(entry) == 2:
        name, source = entry
        return name, source, _default_location(name)
    raise ValueError(
        f"Each camera entry needs 2 or 3 values (name, url[, location]), got {len(entry)}: {entry!r}"
    )


def _is_stream_url(source: str) -> bool:
    return source.lower().startswith(STREAM_PREFIXES)


def _mask_source_for_log(source: str) -> str:
    """Hide credentials in RTSP/HTTP URLs when logging."""
    return _CREDENTIAL_MASK.sub(r"//\1:***@", str(source))


def _resolve_probe_target(source: str) -> Optional[str]:
    """Return a path/URL suitable for ffprobe, or None if not probeable."""
    stripped = str(source).strip()
    if not stripped or "${" in stripped:
        return None
    if stripped.isdigit() or _is_stream_url(stripped):
        return stripped
    path = Path(stripped)
    candidates = [path]
    if not path.is_absolute():
        candidates = [PROJECT_DIR / path, Path.cwd() / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _rtsp_host_port(source: str) -> Optional[Tuple[str, int]]:
    try:
        parsed = urlparse(str(source or "").strip())
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


def _tcp_host_reachable(source: str, timeout: float = CAMERA_VALIDATE_TCP_PROBE_SEC) -> Tuple[bool, str]:
    """Fast LAN probe — offline cams fail here instead of waiting on ffprobe."""
    endpoint = _rtsp_host_port(source)
    if endpoint is None:
        return True, "ok"
    host, port = endpoint
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "ok"
    except OSError as exc:
        return False, f"host {host}:{port} unreachable within {timeout:.1f}s ({exc})"


def _validate_camera_source(
    source: str, timeout: float = CAMERA_VALIDATE_TIMEOUT
) -> Tuple[bool, str]:
    """
    Probe camera reachability with a fast TCP check, then optional ffprobe.
    Returns (ok, reason). Never raises.
    """
    try:
        stripped = str(source).strip()
        if not stripped:
            return False, "empty camera source"
        if "${" in stripped:
            return False, "unresolved ${...} placeholder"

        if stripped.isdigit():
            return True, "ok (local device — ffprobe skipped)"

        probe_target = _resolve_probe_target(stripped)
        if probe_target is None:
            return False, f"video file not found: {stripped}"

        # Network URLs: fail fast when the host is offline.
        if str(probe_target).lower().startswith(("rtsp://", "http://", "https://")):
            tcp_ok, tcp_reason = _tcp_host_reachable(probe_target)
            if not tcp_ok:
                return False, tcp_reason

        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            return True, "ffprobe not installed — skipping"

        cmd = [
            ffprobe,
            "-v",
            "error",
            "-rtsp_transport",
            "tcp",
            "-timeout",
            str(int(timeout * 1_000_000)),
            "-i",
            probe_target,
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, "ok"
        detail = (result.stderr or result.stdout or "ffprobe failed").strip()
        lowered = detail.lower()
        if "401" in detail or "unauthorized" in lowered:
            return False, "RTSP 401 Unauthorized (missing or wrong camera login)"
        return False, detail[:240] if detail else "ffprobe non-zero exit"
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except Exception as exc:
        return False, str(exc)


def _validate_cameras_parallel(
    configs: List[Tuple[str, str, str]],
) -> List[Tuple[str, Tuple[bool, str]]]:
    """Validate camera sources concurrently (offline cams don't serialize startup)."""
    if not configs:
        return []

    def _one(item: Tuple[str, str, str]) -> Tuple[str, Tuple[bool, str]]:
        name, source, _location = item
        return name, _validate_camera_source(source)

    results: List[Tuple[str, Tuple[bool, str]]] = []
    workers = min(CAMERA_VALIDATE_WORKERS, max(1, len(configs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, cfg) for cfg in configs]
        for fut in as_completed(futures):
            results.append(fut.result())
    # Keep original camera order for readable logs.
    order = {name: i for i, (name, _src, _loc) in enumerate(configs)}
    results.sort(key=lambda row: order.get(row[0], 0))
    return results


def _check_face_pipeline() -> Tuple[bool, bool, bool]:
    """Return (face_analyzer importable, retinaface, deepface)."""
    try:
        import face_analyzer

        return True, face_analyzer.retinaface_available(), face_analyzer.deepface_available()
    except ImportError:
        return False, False, False


def _run_dry(configs: List[Tuple[str, str, str]]) -> int:
    """Validate environment without starting subprocesses."""
    ok = True
    log.info("AI Visitor Tracking — dry-run validation")
    log.info("Project: %s\n", PROJECT_DIR)

    main_ok = (PROJECT_DIR / "main.py").is_file()
    dash_ok = (PROJECT_DIR / "dashboard.py").is_file()
    log.info("main.py       %s", "✓" if main_ok else "✗")
    log.info("dashboard.py  %s", "✓" if dash_ok else "✗")
    if not main_ok or not dash_ok:
        ok = False

    if not configs:
        log.error("No cameras configured.")
        ok = False
    else:
        for name, source, location in configs:
            masked = _mask_source_for_log(source)
            reachable, reason = _validate_camera_source(source, timeout=CAMERA_VALIDATE_TIMEOUT)
            log.info(
                "Camera %-8s @ %-16s %s  %s",
                name,
                location,
                masked,
                "✓" if reachable else "✗",
            )
            if reason:
                log.info("  → %s", reason)
            if not reachable:
                ok = False

    log.info("yolov8l.pt (Large)    %s", "✓" if YOLO_MODEL.is_file() else "✗")
    log.info("osnet weights (optional)  %s", "✓" if OSNET_WEIGHTS.is_file() else "✗")

    try:
        from face_restoration import GFPGAN_WEIGHTS, weights_ready

        log.info("face restoration weights %s", "✓" if weights_ready() else "✗")
        if GFPGAN_WEIGHTS.is_file():
            log.info("  GFPGAN: %s", GFPGAN_WEIGHTS.name)
    except ImportError:
        log.info("face restoration module  ✗")

    fa_ok, retina_ok, deepface_ok = _check_face_pipeline()
    log.info("face_analyzer import    %s", "✓" if fa_ok else "✗")
    log.info("RetinaFace              %s", "✓" if retina_ok else "✗")
    log.info("DeepFace                %s", "✓" if deepface_ok else "✗")

    log.info("\nDry-run %s.", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CCTV trackers and the web dashboard together.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dependencies without starting processes",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip startup camera reachability checks",
    )
    parser.add_argument(
        "--device",
        choices=list(DEVICE_CHOICES),
        default="cuda",
        help="Inference device (default: cuda / cuda:0 pure-GPU; auto picks CUDA when available)",
    )
    parser.add_argument(
        "--per-camera-processes",
        action="store_true",
        help="One main.py per camera (uses much more CPU; avoid with 12+ streams)",
    )
    parser.add_argument(
        "--unified-tracker",
        action="store_true",
        help="Force single main.py for all cameras (preferred for GPU efficiency)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        metavar="N",
        help="YOLO every N frames (default: auto from camera count; 5–8 for 12+ cams)",
    )
    parser.add_argument(
        "--inference-size",
        default="640x360",
        metavar="WxH",
        help="AI inference letterbox only (display stays full-res). Default: 640x360",
    )
    parser.add_argument(
        "--plates",
        action="store_true",
        help="Force-enable license plate capture on gate cameras",
    )
    parser.add_argument(
        "--no-plates",
        action="store_true",
        help="Disable license plate capture worker",
    )
    return parser.parse_args()


def _auto_frame_skip(camera_count: int) -> int:
    """More cameras ⇒ skip more YOLO frames so CPU decode + GPU stay balanced."""
    n = max(1, int(camera_count))
    if n >= 24:
        return 10
    if n >= 12:
        return 7
    if n >= 6:
        return 5
    return 3


# Prefer one GPU process once camera count hits this (override with --per-camera-processes).
UNIFIED_TRACKER_CAMERA_THRESHOLD = int(os.environ.get("UNIFIED_TRACKER_CAMERA_THRESHOLD", "4"))


# ── Process management ─────────────────────────────────────────────────────────
def _dashboard_cmd() -> List[str]:
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = os.environ.get("DASHBOARD_PORT", "5000")
    return [
        sys.executable,
        str(PROJECT_DIR / "dashboard.py"),
        "--host",
        host,
        "--port",
        str(port),
    ]

_SHUTTING_DOWN = False
_cameras: List["CameraProcess"] = []
_unified_tracker: Optional["UnifiedTrackerProcess"] = None
_use_unified_tracker = False
_cameras_lock = threading.Lock()
_dashboard: Optional[subprocess.Popen] = None
_go2rtc: Optional[subprocess.Popen] = None
_plate_capture: Optional[subprocess.Popen] = None
_INFERENCE_DEVICE: str = "auto"
_FRAME_SKIP: int = 3
_INFERENCE_SIZE: str = "640x360"
_cleanup_stop: Optional[threading.Event] = None


def _stream_output(pipe, prefix: str) -> None:
    """Read child stdout line-by-line and print with a prefix."""
    child_log = logging.getLogger(prefix)
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            child_log.info(line.rstrip())
    finally:
        try:
            pipe.close()
        except Exception:
            pass


class CameraProcess:
    """One camera tracked in an isolated main.py subprocess."""

    def __init__(
        self, name: str, source: str, location: str, device: str = "auto",
        *,
        frame_skip: int = 3,
        inference_size: str = "640x360",
    ) -> None:
        self.name = name
        self.source = source
        self.location = location
        self.device = device
        self.frame_skip = max(1, int(frame_skip))
        self.inference_size = inference_size
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._restart_count = 0
        self._last_crash_time: Optional[float] = None
        self.permanently_failed = False

    @property
    def restart_count(self) -> int:
        return self._restart_count

    @property
    def faces_detected(self) -> int:
        # Tracker may publish stats under camera id (name) or location label.
        return max(read_faces_detected(self.name), read_faces_detected(self.location))

    @property
    def log_prefix(self) -> str:
        return f"[CAM:{self.name}]"

    def _build_cmd(self) -> List[str]:
        return [
            sys.executable,
            str(PROJECT_DIR / "main.py"),
            "--no-display",
            "--rtsp-tcp",
            "--model",
            str(YOLO_MODEL),
            "--device",
            self.device,
            "--frame-skip",
            str(self.frame_skip),
            "--inference-size",
            self.inference_size,
            # Use stable camera id (not location IP) so live_preview/cameras/{id}.jpg
            # matches Live Stream / go2rtc cam{N} stream names.
            "--camera",
            self.name,
            self.source,
        ]

    def _close_stdout(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            if not self._proc.stdout.closed:
                self._proc.stdout.close()
        except Exception:
            pass

    def start(self) -> subprocess.Popen:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self._proc = subprocess.Popen(
            self._build_cmd(),
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        self._reader_thread = threading.Thread(
            target=_stream_output,
            args=(self._proc.stdout, self.log_prefix),
            name=f"log-{self.name}",
            daemon=True,
        )
        self._reader_thread.start()
        return self._proc

    def poll(self) -> Optional[int]:
        if self._proc is None:
            return None
        return self._proc.poll()

    def terminate(self, timeout: float = 10.0) -> None:
        if self._proc is None or self.poll() is not None:
            return
        try:
            self._proc.terminate()
        except Exception:
            pass

        deadline = time.time() + timeout
        while self.poll() is None and time.time() < deadline:
            time.sleep(0.15)

        if self.poll() is None:
            self.kill()

    def kill(self) -> None:
        if self._proc is None or self.poll() is not None:
            self._close_stdout()
            return
        try:
            self._proc.kill()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=3)
        except Exception:
            pass
        self._close_stdout()

    def maybe_restart(self) -> bool:
        """
        Restart after a crash with exponential back-off.
        Returns True if restarted, False when the restart limit is exceeded.
        """
        if self.permanently_failed:
            return False

        now = time.time()
        if (
            self._last_crash_time is not None
            and (now - self._last_crash_time) > CRASH_STALE_SECONDS
        ):
            self._restart_count = 0

        self._last_crash_time = now
        self._restart_count += 1

        if self._restart_count > MAX_RESTARTS:
            self.permanently_failed = True
            log.error(
                "%s crashed %d times — giving up on this camera.",
                self.log_prefix,
                MAX_RESTARTS,
            )
            self._close_stdout()
            return False

        delay = min(
            RESTART_BASE_DELAY * (2 ** (self._restart_count - 1)),
            RESTART_MAX_DELAY,
        )
        log.warning(
            "%s stopped (restart %d/%d). Waiting %.1fs before restart…",
            self.log_prefix,
            self._restart_count,
            MAX_RESTARTS,
            delay,
        )

        self._close_stdout()
        time.sleep(delay)
        self.start()
        log.info("%s restarted (attempt %d).", self.log_prefix, self._restart_count)
        return True


class UnifiedTrackerProcess:
    """All cameras tracked in one main.py subprocess (single YOLO on GPU)."""

    def __init__(
        self,
        configs: List[Tuple[str, str, str]],
        device: str = "auto",
        *,
        frame_skip: int = 3,
        inference_size: str = "640x360",
    ) -> None:
        self.configs = list(configs)
        self.device = device
        self.frame_skip = max(1, int(frame_skip))
        self.inference_size = inference_size
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._restart_count = 0
        self._last_crash_time: Optional[float] = None
        self.permanently_failed = False

    @property
    def restart_count(self) -> int:
        return self._restart_count

    @property
    def log_prefix(self) -> str:
        return "[TRACKER:unified]"

    def _build_cmd(self) -> List[str]:
        cmd = [
            sys.executable,
            str(PROJECT_DIR / "main.py"),
            "--no-display",
            "--rtsp-tcp",
            "--model",
            str(YOLO_MODEL),
            "--device",
            self.device,
            "--frame-skip",
            str(self.frame_skip),
            "--inference-size",
            self.inference_size,
        ]
        if resolve_torch_device(self.device).startswith("cuda"):
            cmd.append("--gpu-unified")
        for name, source, _location in self.configs:
            # Stable id key — same as go2rtc cam{N} and /live_preview/cameras/{id}.jpg
            cmd.extend(["--camera", str(name), source])
        return cmd

    def _close_stdout(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            if not self._proc.stdout.closed:
                self._proc.stdout.close()
        except Exception:
            pass

    def start(self) -> subprocess.Popen:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self._proc = subprocess.Popen(
            self._build_cmd(),
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        self._reader_thread = threading.Thread(
            target=_stream_output,
            args=(self._proc.stdout, self.log_prefix),
            name="log-unified-tracker",
            daemon=True,
        )
        self._reader_thread.start()
        return self._proc

    def poll(self) -> Optional[int]:
        if self._proc is None:
            return None
        return self._proc.poll()

    def terminate(self, timeout: float = 10.0) -> None:
        if self._proc is None or self.poll() is not None:
            return
        try:
            self._proc.terminate()
        except Exception:
            pass

        deadline = time.time() + timeout
        while self.poll() is None and time.time() < deadline:
            time.sleep(0.15)

        if self.poll() is None:
            self.kill()

    def kill(self) -> None:
        if self._proc is None or self.poll() is not None:
            self._close_stdout()
            return
        try:
            self._proc.kill()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=3)
        except Exception:
            pass
        self._close_stdout()

    def maybe_restart(self) -> bool:
        if self.permanently_failed:
            return False

        now = time.time()
        if (
            self._last_crash_time is not None
            and (now - self._last_crash_time) > CRASH_STALE_SECONDS
        ):
            self._restart_count = 0

        self._last_crash_time = now
        self._restart_count += 1

        if self._restart_count > MAX_RESTARTS:
            self.permanently_failed = True
            log.error(
                "%s crashed %d times — giving up on unified tracker.",
                self.log_prefix,
                MAX_RESTARTS,
            )
            self._close_stdout()
            return False

        delay = min(
            RESTART_BASE_DELAY * (2 ** (self._restart_count - 1)),
            RESTART_MAX_DELAY,
        )
        log.warning(
            "%s stopped (restart %d/%d). Waiting %.1fs before restart…",
            self.log_prefix,
            self._restart_count,
            MAX_RESTARTS,
            delay,
        )

        self._close_stdout()
        time.sleep(delay)
        self.start()
        log.info("%s restarted (attempt %d).", self.log_prefix, self._restart_count)
        return True


def _list_dashboard_pids() -> List[int]:
    """Return PIDs for dashboard.py child processes (Windows-friendly)."""
    pids: List[int] = []
    dashboard_py = str(PROJECT_DIR / "dashboard.py").replace("\\", "/")
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(
                [
                    "wmic",
                    "process",
                    "where",
                    f"CommandLine like '%{dashboard_py}%'",
                    "get",
                    "ProcessId",
                    "/format:csv",
                ],
                text=True,
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            for line in out.splitlines():
                cell = line.strip()
                if cell.isdigit():
                    pids.append(int(cell))
        else:
            out = subprocess.check_output(["pgrep", "-f", "dashboard.py"], text=True)
            for line in out.splitlines():
                if line.strip().isdigit():
                    pids.append(int(line.strip()))
    except Exception as exc:
        log.warning("Could not scan for dashboard processes: %s", exc)
    return sorted(set(pids))


def _port_listener_pids(port: int = 5000) -> List[int]:
    """PIDs currently LISTENING on ``port`` (Windows netstat; empty on failure)."""
    found: List[int] = []
    if sys.platform != "win32":
        return found
    try:
        out = subprocess.check_output(
            ["netstat", "-ano"],
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception:
        return found
    needle = f":{port} "
    for line in out.splitlines():
        if "LISTENING" not in line.upper() or needle not in line:
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            found.append(int(parts[-1]))
        except ValueError:
            continue
    return sorted(set(found))


def _terminate_stale_dashboard_processes() -> None:
    """Kill orphan dashboard.py instances (duplicate port 5000 → random API errors)."""
    pids = sorted(set(_list_dashboard_pids() + _port_listener_pids(5000)))
    # Never kill ourselves if somehow listed.
    pids = [p for p in pids if p != os.getpid()]
    if not pids:
        return
    log.warning(
        "Stopping %d existing dashboard/port-5000 process(es) before fresh start: %s",
        len(pids),
        pids,
    )
    for pid in pids:
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                subprocess.run(["kill", "-TERM", str(pid)], check=False)
        except Exception as exc:
            log.warning("Failed to stop dashboard pid %s: %s", pid, exc)
    # Wait until :5000 is free — overlapping listeners cause Errno 22 / flaky scans.
    deadline = time.time() + 8.0
    while time.time() < deadline:
        left = _port_listener_pids(5000)
        if not left:
            break
        time.sleep(0.4)
    else:
        log.warning(
            "Port 5000 still held by PID(s) %s — Target Trace scans may fail "
            "with [Errno 22] Invalid argument until only one dashboard owns the port.",
            _port_listener_pids(5000),
        )


def _start_dashboard() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DEVICE"] = _INFERENCE_DEVICE
    env["ORION_PURE_GPU"] = "1" if resolve_torch_device(_INFERENCE_DEVICE).startswith("cuda") else "0"
    env.setdefault("NVIDIA_VISIBLE_DEVICES", "all")
    env.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")

    proc = subprocess.Popen(
        _dashboard_cmd(),
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )

    thread = threading.Thread(
        target=_stream_output,
        args=(proc.stdout, "[DASHBOARD]"),
        name="log-dashboard",
        daemon=True,
    )
    thread.start()
    return proc


def _start_go2rtc() -> Optional[subprocess.Popen]:
    """Launch go2rtc WebRTC on :1986 so Fast Live can use H.264 (not JPEG).
    MSA_CCTV_100 binds :1984 and MSA_CCTV binds :1985 — keep this project on GO2RTC_PORT (default 1986).
    """
    nvr_dir = PROJECT_DIR / "nvr"
    exe = nvr_dir / "go2rtc.exe"
    if not exe.is_file():
        found = shutil.which("go2rtc")
        if found:
            exe = Path(found)
        else:
            log.warning("go2rtc not found — WebRTC live view disabled. Put go2rtc.exe in nvr\\")
            return None
    try:
        subprocess.run(
            [sys.executable, str(nvr_dir / "generate_configs.py")],
            cwd=str(PROJECT_DIR),
            check=True,
            timeout=30,
        )
    except Exception as exc:
        log.warning("go2rtc config generate failed: %s", exc)
        if not (nvr_dir / "go2rtc.yaml").is_file():
            return None
    cfg = nvr_dir / "go2rtc.yaml"
    log_path = nvr_dir / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    out = open(log_path / "go2rtc.log", "ab")
    proc = subprocess.Popen(
        [str(exe), "-config", str(cfg)],
        cwd=str(nvr_dir),
        stdout=out,
        stderr=subprocess.STDOUT,
    )
    port = (os.environ.get("GO2RTC_PORT") or "1986").strip() or "1986"
    log.info("go2rtc WebRTC:     http://127.0.0.1:%s  (pid %s)", port, proc.pid)
    return proc


def _plates_enabled(args: argparse.Namespace) -> bool:
    if getattr(args, "no_plates", False):
        return False
    if getattr(args, "plates", False):
        return True
    raw = os.environ.get("ENABLE_LICENSE_PLATE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _start_plate_capture() -> Optional[subprocess.Popen]:
    """Background YOLO+EasyOCR plate reader (separate process from face tracker)."""
    script = PROJECT_DIR / "plate_capture.py"
    if not script.is_file():
        log.warning("plate_capture.py missing — license plate detection disabled")
        return None
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cams = (os.environ.get("LICENSE_PLATE_CAMERAS") or "all").strip()
    every = (os.environ.get("LICENSE_PLATE_OCR_EVERY") or "").strip()
    cmd = [sys.executable, str(script)]
    scope_label = cams
    if cams.lower() in ("all", "*", "every", ""):
        cmd.append("--all")
        scope_label = "ALL cameras"
        if not every:
            every = "15"
    else:
        cmd.extend(["--cam", cams])
        if not every:
            every = "12"
    cmd.extend(["--every", every, "--no-display"])
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    thread = threading.Thread(
        target=_stream_output,
        args=(proc.stdout, "[PLATE]"),
        name="log-plate-capture",
        daemon=True,
    )
    thread.start()
    log.info(
        "License plates:    %s (OCR every %s frames, pid %s)",
        scope_label,
        every,
        proc.pid,
    )
    return proc


def _terminate_all() -> None:
    global _SHUTTING_DOWN, _dashboard, _go2rtc, _plate_capture, _cleanup_stop, _unified_tracker, _cameras
    if _SHUTTING_DOWN:
        return
    _SHUTTING_DOWN = True

    if _cleanup_stop is not None:
        _cleanup_stop.set()

    plate = _plate_capture
    _plate_capture = None
    if plate is not None and plate.poll() is None:
        try:
            plate.terminate()
        except Exception:
            pass
        try:
            plate.wait(timeout=4)
        except Exception:
            try:
                plate.kill()
            except Exception:
                pass

    go2rtc = _go2rtc
    _go2rtc = None
    if go2rtc is not None and go2rtc.poll() is None:
        try:
            go2rtc.terminate()
        except Exception:
            pass
        try:
            go2rtc.wait(timeout=3)
        except Exception:
            try:
                go2rtc.kill()
            except Exception:
                pass

    with _cameras_lock:
        cameras = list(_cameras)
        unified = _unified_tracker
        _unified_tracker = None
        _cameras = []

    if unified is not None:
        unified.terminate()
    for cam in cameras:
        cam.terminate()

    dashboard = _dashboard
    if dashboard is not None and dashboard.poll() is None:
        try:
            dashboard.terminate()
        except Exception:
            pass

        deadline = time.time() + 10.0
        while dashboard.poll() is None and time.time() < deadline:
            time.sleep(0.15)

        if dashboard.poll() is None:
            try:
                dashboard.kill()
            except Exception:
                pass
            try:
                dashboard.wait(timeout=3)
            except Exception:
                pass
            try:
                if dashboard.stdout and not dashboard.stdout.closed:
                    dashboard.stdout.close()
            except Exception:
                pass


def _shutdown(exit_code: int = 0) -> None:
    log.info("Shutting down all services…")
    _terminate_all()
    sys.exit(exit_code)


def _on_signal(signum, frame) -> None:  # noqa: ARG001
    raise KeyboardInterrupt


def _write_cameras_config(configs: List[Tuple[str, str, str]]) -> None:
    """Save configured camera zones for the dashboard (resolved sources)."""
    data = [
        {"id": name, "location": location, "source": source}
        for name, source, location in configs
    ]
    CAMERAS_CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _all_cameras_permanently_failed() -> bool:
    with _cameras_lock:
        if _unified_tracker is not None:
            return _unified_tracker.permanently_failed
        cameras = list(_cameras)
    return bool(cameras) and all(cam.permanently_failed for cam in cameras)


def _any_camera_running() -> bool:
    with _cameras_lock:
        if _unified_tracker is not None:
            return (
                not _unified_tracker.permanently_failed
                and _unified_tracker.poll() is None
            )
        cameras = list(_cameras)
    return any(
        not cam.permanently_failed and cam.poll() is None for cam in cameras
    )


def _write_status() -> None:
    """Atomically publish process health for the dashboard."""
    try:
        with _cameras_lock:
            if _unified_tracker is not None:
                alive = (
                    _unified_tracker.poll() is None
                    and not _unified_tracker.permanently_failed
                )
                camera_rows = [
                    {
                        "name": name,
                        "location": location,
                        "alive": alive,
                        "restart_count": _unified_tracker.restart_count,
                        "faces_detected": max(
                            read_faces_detected(name),
                            read_faces_detected(location),
                        ),
                    }
                    for name, _source, location in _unified_tracker.configs
                ]
            else:
                camera_rows = [
                    {
                        "name": cam.name,
                        "location": cam.location,
                        "alive": cam.poll() is None and not cam.permanently_failed,
                        "restart_count": cam.restart_count,
                        "faces_detected": cam.faces_detected,
                    }
                    for cam in _cameras
                ]

        dashboard = _dashboard
        dash_alive = dashboard is not None and dashboard.poll() is None
        payload = {
            "timestamp": time.time(),
            "dashboard": {
                "pid": dashboard.pid if dashboard else None,
                "alive": dash_alive,
            },
            "cameras": camera_rows,
        }

        tmp_path = SYSTEM_STATUS_PATH.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(SYSTEM_STATUS_PATH)
    except Exception as exc:
        log.debug("Failed to write system_status.json: %s", exc)


def _reload_cameras() -> None:
    """Apply cameras.json changes: add, remove, or replace camera processes."""
    global _cameras, _unified_tracker

    try:
        new_configs = _load_tracker_configs()
    except Exception as exc:
        log.error("[RELOAD] Failed to load cameras.json: %s", exc)
        return

    with _cameras_lock:
        if _unified_tracker is not None:
            old_configs = _unified_tracker.configs
            if old_configs == new_configs:
                log.info("[RELOAD] Unified tracker configs unchanged")
                return
            log.info(
                "[RELOAD] Restarting unified tracker (%d cameras)",
                len(new_configs),
            )
            _unified_tracker.terminate()
            _unified_tracker = UnifiedTrackerProcess(
                new_configs,
                device=_INFERENCE_DEVICE,
                frame_skip=_FRAME_SKIP,
                inference_size=_INFERENCE_SIZE,
            )
            _unified_tracker.start()
            return

        current_by_name: Dict[str, CameraProcess] = {cam.name: cam for cam in _cameras}
        next_cameras: List[CameraProcess] = []

        for name, source, location in new_configs:
            existing = current_by_name.get(name)
            if existing is None:
                log.info("[RELOAD] Adding camera %s (%s)", name, location)
                cam = CameraProcess(
                    name,
                    source,
                    location,
                    device=_INFERENCE_DEVICE,
                    frame_skip=_FRAME_SKIP,
                    inference_size=_INFERENCE_SIZE,
                )
                cam.start()
                next_cameras.append(cam)
                continue

            if existing.source == source and existing.location == location:
                log.info("[RELOAD] Keeping camera %s unchanged", name)
                next_cameras.append(existing)
                continue

            log.info(
                "[RELOAD] Replacing camera %s (source or location changed)",
                name,
            )
            existing.terminate()
            cam = CameraProcess(
                name,
                source,
                location,
                device=_INFERENCE_DEVICE,
                frame_skip=_FRAME_SKIP,
                inference_size=_INFERENCE_SIZE,
            )
            cam.start()
            next_cameras.append(cam)

        new_names = {name for name, _, _ in new_configs}
        for name, cam in current_by_name.items():
            if name not in new_names:
                log.info("[RELOAD] Removing camera %s", name)
                cam.terminate()

        _cameras = next_cameras


def _config_watcher(interval: float = 2.0) -> None:
    """Poll cameras.json mtime and hot-reload camera processes on change."""
    last_mtime: Optional[float] = None
    try:
        last_mtime = CAMERAS_CONFIG_PATH.stat().st_mtime
    except OSError as exc:
        log.warning("Cannot stat cameras.json at watcher start: %s", exc)

    while not _SHUTTING_DOWN:
        time.sleep(interval)
        if _SHUTTING_DOWN:
            break
        try:
            mtime = CAMERAS_CONFIG_PATH.stat().st_mtime
        except OSError as exc:
            log.warning("cameras.json temporarily unreadable: %s", exc)
            continue

        if last_mtime is None or mtime != last_mtime:
            last_mtime = mtime
            log.info("cameras.json changed — reloading camera processes")
            try:
                _reload_cameras()
            except Exception as exc:
                log.error("[RELOAD] Camera reload failed: %s", exc)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    global _cameras, _dashboard, _go2rtc, _plate_capture, _INFERENCE_DEVICE, _FRAME_SKIP, _INFERENCE_SIZE
    global _cleanup_stop, _unified_tracker, _use_unified_tracker

    args = _parse_args()
    _INFERENCE_DEVICE = args.device
    _INFERENCE_SIZE = str(args.inference_size).strip()
    # Placeholder until camera count is known (auto frame-skip below).
    _FRAME_SKIP = 3
    if not args.dry_run:
        runtime_err = _ensure_runtime_python()
        if runtime_err is not None:
            return runtime_err
    configure(_INFERENCE_DEVICE, pure_gpu=resolve_torch_device(_INFERENCE_DEVICE).startswith("cuda"))
    if args.unified_tracker and args.per_camera_processes:
        log.error("Choose either --unified-tracker or --per-camera-processes, not both.")
        return 1
    _load_dotenv()
    try:
        from security_bootstrap import resolve_dashboard_secrets

        resolve_dashboard_secrets()
    except Exception as exc:
        log.warning("Security bootstrap skipped: %s", exc)

    if not (PROJECT_DIR / "main.py").is_file():
        log.error("main.py not found.")
        return 1
    if not (PROJECT_DIR / "dashboard.py").is_file():
        log.error("dashboard.py not found.")
        return 1

    configs = _load_tracker_configs()
    cam_count = len(configs)
    rtsp_missing_login = [
        name
        for name, source, _location in configs
        if str(source).lower().startswith("rtsp://")
        and "@" not in str(source)[7:].split("/", 1)[0]
    ]
    if rtsp_missing_login and not args.skip_validation and not args.dry_run:
        log.error(
            "RTSP cameras have no login (camera 1 returned 401 Unauthorized).\n"
            "  Create .env in this folder:\n"
            "    RTSP_USER=admin\n"
            "    RTSP_PASSWORD=your_camera_password\n"
            "  Or put user:password in each cameras.json source URL."
        )
        return 1
    if args.frame_skip is not None:
        _FRAME_SKIP = max(1, int(args.frame_skip))
    else:
        _FRAME_SKIP = _auto_frame_skip(cam_count)

    # GPU-first default: one tracker process for multi-camera installs.
    # Per-camera processes decode 30× RTSP on CPU and thrash CUDA contexts.
    if args.per_camera_processes:
        _use_unified_tracker = False
    elif args.unified_tracker or cam_count >= UNIFIED_TRACKER_CAMERA_THRESHOLD:
        _use_unified_tracker = True
    else:
        _use_unified_tracker = False

    if args.dry_run:
        return _run_dry(configs)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _on_signal)

    if not OSNET_WEIGHTS.is_file():
        log.info(
            "OSNet body Re-ID weights not in models/ yet.\n"
            "  Run once: python download_body_model.py --no-proxy\n"
            "  (Optional — improves dashboard photo search fallback.)"
        )

    yolo_weights = YOLO_MODEL
    if not yolo_weights.is_file():
        log.info(
            "YOLO weights not found.\n"
            "  Run once: py -3.12 download_model.py --no-proxy --size l\n"
            "  (Downloads yolov8l.pt — Large model for blurry / partial figures.)"
        )

    try:
        from face_restoration import weights_ready

        if not weights_ready():
            log.info(
                "Face restoration weights not in models/ yet.\n"
                "  Run once: python download_face_restoration.py --no-proxy\n"
                "  (GFPGAN/CodeFormer — upscales blurry command-center face crops before ArcFace.)"
            )
    except ImportError:
        pass

    if not configs:
        log.error("No cameras configured.")
        return 1

    if not CAMERAS_CONFIG_PATH.is_file():
        _write_cameras_config(configs)
    ensure_live_frame_dir()

    fa_ok, retina_ok, deepface_ok = _check_face_pipeline()
    resolved_device = audit_hardware_dependencies(device_choice=_INFERENCE_DEVICE)
    init_face_restoration_stack()
    try:
        if resolve_torch_device(_INFERENCE_DEVICE).startswith("cuda"):
            init_pure_gpu_video_scan(device_choice=_INFERENCE_DEVICE, strict=True)
            log.info("Video target scan: CUDA-only pipeline ready (stride-5, 50%% detect downscale)")
    except RuntimeError as exc:
        log.error("Video target scan GPU gate failed: %s", exc)
        return 1
    try:
        from face_analyzer import active_pipeline_name
        from face_restoration import active_backend_name

        log.info("Face GPU backend:  %s", active_pipeline_name())
        log.info("Face restoration:  %s", active_backend_name())
    except Exception as exc:
        log.info("Face GPU backend:  unavailable (%s)", exc)
    log.info("AI Visitor Tracking — unified launcher")
    log.info("Inference device:  %s (%s)", _INFERENCE_DEVICE, describe_device())
    log.info("Resolved torch:    %s", resolved_device)
    log.info(
        "Pipeline:          frame-skip=%d, inference-size=%s (display full-res, RTSP buffer=1)",
        _FRAME_SKIP,
        _INFERENCE_SIZE,
    )
    log.info(
        "Telegram alerts:   %s (style=%s)",
        "enabled" if telegram_configured() else "off (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)",
        TELEGRAM_ALERT_STYLE,
    )
    log.info("Working directory: %s", PROJECT_DIR)
    log.info("Dashboard:         http://127.0.0.1:5000")
    log.info("Live web preview:  go2rtc WebRTC (JPEG fallback if go2rtc is down)")
    log.info(
        "Face pipeline:     analyzer=%s retina=%s deepface=%s",
        "✓" if fa_ok else "✗",
        "✓" if retina_ok else "✗",
        "✓" if deepface_ok else "✗",
    )
    mode_label = (
        "unified tracker (single GPU process — preferred)"
        if _use_unified_tracker
        else "one process per camera (high CPU — decode×N)"
    )
    log.info(
        "Trackers (%s): %d camera(s), frame-skip=%d",
        mode_label,
        cam_count,
        _FRAME_SKIP,
    )
    for name, source, location in configs:
        log.info("  - %s: %s (%s)", name, location, _mask_source_for_log(source))

    if not args.skip_validation:
        validation_results = _validate_cameras_parallel(configs)
        reachable_count = sum(1 for _name, (ok, _reason) in validation_results if ok)
        for name, (ok, reason) in validation_results:
            if ok:
                detail = f" ({reason})" if reason and reason != "ok" else ""
                log.info("  ✓ Camera %s reachable at startup%s", name, detail)
            else:
                log.warning(
                    "  ✗ Camera %s unreachable at startup: %s (will still start and retry)",
                    name,
                    reason,
                )

        if reachable_count == 0:
            reasons = " ".join(reason for _name, (ok, reason) in validation_results if not ok)
            if "401" in reasons or "Unauthorized" in reasons:
                log.error(
                    "All cameras rejected RTSP login (401).\n"
                    "  Put USER:PASSWORD in cameras.json, or set in .env:\n"
                    "    RTSP_USER=admin\n"
                    "    RTSP_PASSWORD=your_camera_password"
                )
                return 1
            # Network may come up later (new VLAN/route). Start and keep retrying.
            log.warning(
                "All cameras unreachable at startup — launching anyway; "
                "trackers will retry until RTSP is reachable."
            )
    else:
        log.warning("Startup camera validation skipped (--skip-validation).")

    log.info(
        "Storage cleanup:    every %.1f h, keep captures %.0f days then CSV+delete",
        CLEANUP_INTERVAL_SEC / 3600.0,
        RETENTION_HOURS / 24.0,
    )
    log.info(
        "Gate cameras %s: frame-skip=%d, RTSP/TCP, relaxed face gates",
        ", ".join(sorted(tracker_config.OUTDOOR_GATE_CAMERAS)),
        tracker_config.OUTDOOR_GATE_FRAME_SKIP,
    )

    try:
        _terminate_stale_dashboard_processes()
        _dashboard = _start_dashboard()
        _go2rtc = _start_go2rtc()
        if _plates_enabled(args):
            _plate_capture = _start_plate_capture()
        else:
            log.info("License plates:    disabled (--no-plates or ENABLE_LICENSE_PLATE=0)")
            _plate_capture = None
        with _cameras_lock:
            if _use_unified_tracker:
                _unified_tracker = UnifiedTrackerProcess(
                    configs,
                    device=_INFERENCE_DEVICE,
                    frame_skip=_FRAME_SKIP,
                    inference_size=_INFERENCE_SIZE,
                )
                _cameras = []
                _unified_tracker.start()
            else:
                _unified_tracker = None
                _cameras = [
                    CameraProcess(
                        name,
                        source,
                        location,
                        device=_INFERENCE_DEVICE,
                        frame_skip=_FRAME_SKIP,
                        inference_size=_INFERENCE_SIZE,
                    )
                    for name, source, location in configs
                ]
        if not _use_unified_tracker:
            for idx, cam in enumerate(_cameras):
                cam.start()
                if idx < len(_cameras) - 1:
                    time.sleep(CAMERA_START_STAGGER_SEC)
    except Exception as exc:
        log.error("Failed to start a process: %s", exc)
        _terminate_all()
        return 1

    watcher = threading.Thread(
        target=_config_watcher,
        kwargs={"interval": 2.0},
        name="config-watcher",
        daemon=True,
    )
    watcher.start()

    _cleanup_thread, _cleanup_stop = start_storage_cleanup_daemon(
        db_path=DEFAULT_DB_PATH,
        interval_sec=CLEANUP_INTERVAL_SEC,
        retention_hours=RETENTION_HOURS,
    )

    try:
        while True:
            dash_code = _dashboard.poll() if _dashboard else None
            if dash_code is not None:
                log.warning(
                    "Dashboard stopped (exit code %d). Stopping all cameras…",
                    dash_code,
                )
                _terminate_all()
                return dash_code if dash_code != 0 else 0

            if _plate_capture is not None and _plate_capture.poll() is not None:
                code = _plate_capture.poll()
                log.warning("License plate worker stopped (exit code %s) — restarting…", code)
                try:
                    _plate_capture = _start_plate_capture()
                except Exception as exc:
                    log.warning("Plate worker restart failed: %s", exc)
                    _plate_capture = None

            with _cameras_lock:
                unified = _unified_tracker
                cameras_snapshot = list(_cameras)

            if unified is not None:
                if unified.permanently_failed:
                    pass
                else:
                    code = unified.poll()
                    if code is not None:
                        log.warning("%s stopped (exit code %d).", unified.log_prefix, code)
                        unified.maybe_restart()
            else:
                for cam in cameras_snapshot:
                    if cam.permanently_failed:
                        continue
                    code = cam.poll()
                    if code is None:
                        continue
                    log.warning("%s stopped (exit code %d).", cam.log_prefix, code)
                    cam.maybe_restart()

            if _all_cameras_permanently_failed():
                log.error("All cameras permanently failed — shutting down.")
                _write_status()
                _terminate_all()
                return 1

            _write_status()

            if not _any_camera_running() and not _all_cameras_permanently_failed():
                # Transient gap while a camera is between crash and restart sleep.
                time.sleep(0.4)
                continue

            time.sleep(0.4)

    except KeyboardInterrupt:
        _shutdown(0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
