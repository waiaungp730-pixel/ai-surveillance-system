"""
Shared live preview frames between the CCTV tracker (main.py) and dashboard.

The tracker encodes each preview frame with cv2.imencode and writes it here;
dashboard.py serves it as MJPEG on /api/live_cctv_stream (no desktop window).
"""

import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
LIVE_FRAME_DIR = PROJECT_DIR / "live_preview"
CAMERA_FRAME_DIR = LIVE_FRAME_DIR / "cameras"
LIVE_FRAME_JPEG = LIVE_FRAME_DIR / "frame.jpg"
# Alternate slots so the dashboard can read one file while the tracker writes the other.
LIVE_FRAME_SLOTS: List[Path] = [
    LIVE_FRAME_DIR / "frame_0.jpg",
    LIVE_FRAME_DIR / "frame_1.jpg",
]

# Frames older than this are treated as stale (tracker stopped or not started).
DEFAULT_MAX_AGE_SEC = 30.0
_PUBLISH_RETRIES = 5
_PUBLISH_RETRY_DELAY_SEC = 0.01
_READ_RETRIES = 3

_write_slot = 0


def ensure_live_frame_dir() -> None:
    LIVE_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    CAMERA_FRAME_DIR.mkdir(parents=True, exist_ok=True)


def _safe_camera_key(camera_key: str) -> str:
    safe = re.sub(r"[^\w\-]+", "_", str(camera_key).strip())
    return (safe[:64] if safe else "cam")


def _write_with_retries(path: Path, jpeg_bytes: bytes) -> bool:
    """Atomically write JPEG bytes (tmp + replace) to avoid half-read flicker."""
    tmp = path.with_name(path.name + ".tmp")
    for attempt in range(_PUBLISH_RETRIES):
        try:
            with open(tmp, "wb") as handle:
                handle.write(jpeg_bytes)
                handle.flush()
                # Skip fsync — live preview prioritizes low latency over crash durability.
            os.replace(tmp, path)
            return True
        except (PermissionError, OSError):
            if attempt < _PUBLISH_RETRIES - 1:
                time.sleep(_PUBLISH_RETRY_DELAY_SEC * (attempt + 1))
    try:
        if tmp.is_file():
            tmp.unlink()
    except OSError:
        pass
    return False


def publish_jpeg(jpeg_bytes: bytes) -> None:
    """Store the latest combined JPEG from a multi-camera tracker process."""
    global _write_slot
    if not jpeg_bytes:
        return
    ensure_live_frame_dir()

    target = LIVE_FRAME_SLOTS[_write_slot]
    if _write_with_retries(target, jpeg_bytes):
        _write_slot = 1 - _write_slot

    # Best-effort mirror for older readers; failure must not stop the tracker.
    _write_with_retries(LIVE_FRAME_JPEG, jpeg_bytes)


def publish_camera_jpeg(camera_key: str, jpeg_bytes: bytes) -> None:
    """Store the latest JPEG for one isolated per-camera tracker process."""
    if not jpeg_bytes:
        return
    ensure_live_frame_dir()
    path = CAMERA_FRAME_DIR / f"{_safe_camera_key(camera_key)}.jpg"
    _write_with_retries(path, jpeg_bytes)


def _read_jpeg_file(path: Path, max_age_sec: float) -> Optional[bytes]:
    """Read one JPEG path with short retries for concurrent dashboard access."""
    now = time.time()
    for attempt in range(_READ_RETRIES):
        try:
            if not path.is_file():
                return None
            age = now - path.stat().st_mtime
            if age > max_age_sec:
                return None
            data = path.read_bytes()
            # Reject truncated files mid-write (EOI marker missing).
            if len(data) < 24 or data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
                if attempt < _READ_RETRIES - 1:
                    time.sleep(_PUBLISH_RETRY_DELAY_SEC)
                    continue
                return None
            return data
        except (OSError, PermissionError):
            if attempt < _READ_RETRIES - 1:
                time.sleep(_PUBLISH_RETRY_DELAY_SEC)
    return None


def list_camera_jpegs(
    max_age_sec: float = DEFAULT_MAX_AGE_SEC,
) -> List[Tuple[str, bytes]]:
    """Return recent per-camera JPEGs sorted by camera key (stable grid order)."""
    ensure_live_frame_dir()
    now = time.time()
    feeds: List[Tuple[str, bytes]] = []
    if not CAMERA_FRAME_DIR.is_dir():
        return feeds
    for path in sorted(CAMERA_FRAME_DIR.glob("*.jpg")):
        try:
            age = now - path.stat().st_mtime
            if age > max_age_sec:
                continue
            jpeg = _read_jpeg_file(path, max_age_sec)
            if jpeg:
                feeds.append((path.stem, jpeg))
        except OSError:
            continue
    return feeds


def get_camera_jpeg(
    camera_key: str, max_age_sec: float = DEFAULT_MAX_AGE_SEC
) -> Optional[bytes]:
    """Return the latest JPEG for one camera if it was written recently."""
    if not camera_key:
        return None
    path = CAMERA_FRAME_DIR / f"{_safe_camera_key(camera_key)}.jpg"
    return _read_jpeg_file(path, max_age_sec)


def get_latest_jpeg(max_age_sec: float = DEFAULT_MAX_AGE_SEC) -> Optional[bytes]:
    """Return the newest tracker frame if it was written recently."""
    best_jpeg: Optional[bytes] = None
    best_mtime = 0.0
    for path in (*LIVE_FRAME_SLOTS, LIVE_FRAME_JPEG):
        try:
            if not path.is_file():
                continue
            mtime = path.stat().st_mtime
            if time.time() - mtime > max_age_sec or mtime <= best_mtime:
                continue
            jpeg = _read_jpeg_file(path, max_age_sec)
            if jpeg is not None:
                best_jpeg = jpeg
                best_mtime = mtime
        except OSError:
            continue
    return best_jpeg


def is_feed_available(max_age_sec: float = DEFAULT_MAX_AGE_SEC) -> bool:
    if get_latest_jpeg(max_age_sec=max_age_sec) is not None:
        return True
    return bool(list_camera_jpegs(max_age_sec=max_age_sec))
