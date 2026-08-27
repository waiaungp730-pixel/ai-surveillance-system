"""Shared tracker/YOLO constants (import-safe for GPU worker)."""

from __future__ import annotations

import os
from typing import Tuple

PERSON_CLASS_ID = 0
# Lower than indoor hallway defaults so distant outdoor gate walkers still track.
TRACK_CONF = 0.35
# Outdoor gate cams (high angle / far subjects) keep weaker YOLO hits.
OUTDOOR_GATE_TRACK_CONF = 0.25
# High-angle outdoor mounts: include Gate-03 (cam 3) + rear gates 28–30.
# Override with OUTDOOR_GATE_CAMERAS=1,2,3,28,29,30
_OUTDOOR_GATE_DEFAULT = "3,28,29,30"
OUTDOOR_GATE_CAMERAS = frozenset(
    c.strip()
    for c in os.environ.get("OUTDOOR_GATE_CAMERAS", _OUTDOOR_GATE_DEFAULT).split(",")
    if c.strip()
)
# Gate-only tuning: more AI passes + relaxed face quality for rain/angle.
OUTDOOR_GATE_FRAME_SKIP = int(os.environ.get("OUTDOOR_GATE_FRAME_SKIP", "3"))
OUTDOOR_GATE_MIN_PERSON_HEIGHT_PX = int(
    os.environ.get("OUTDOOR_GATE_MIN_PERSON_HEIGHT_PX", "20")
)
OUTDOOR_GATE_MIN_FACE_LAPLACIAN = float(
    os.environ.get("OUTDOOR_GATE_MIN_FACE_LAPLACIAN", "1.5")
)
OUTDOOR_GATE_SKIN_MIN_FRACTION = float(
    os.environ.get("OUTDOOR_GATE_SKIN_MIN_FRACTION", "0.05")
)
OUTDOOR_GATE_FACE_MIN_CONF = float(
    os.environ.get("OUTDOOR_GATE_FACE_MIN_CONF", "0.22")
)
OUTDOOR_GATE_MIN_FACE_BOX_PX = int(
    os.environ.get("OUTDOOR_GATE_MIN_FACE_BOX_PX", "28")
)
OUTDOOR_GATE_DISPLAY_MIN_SIDE = int(
    os.environ.get("OUTDOOR_GATE_DISPLAY_MIN_SIDE", "12")
)
OUTDOOR_GATE_LIVE_FACE_FORCE_EVERY = max(
    1,
    int(os.environ.get("OUTDOOR_GATE_LIVE_FACE_FORCE_EVERY", "1")),
)# YOLOv8s-P2 uses stride 4; yolov8l / standard heads use 32.
# Inference canvas should be divisible by this for clean GPU BCHW tensors.
YOLO_STRIDE = 32

# Default AI inference size (low VRAM for multi-camera). Full-res frames are
# kept for dashboard preview / face crops; only YOLO sees this canvas.
INFERENCE_WIDTH = 640
INFERENCE_HEIGHT = 360

TRACK_KWARGS = {
    "persist": True,
    "classes": [PERSON_CLASS_ID],
    "tracker": "bytetrack.yaml",
    # Keep detector floor below TRACK_CONF so outdoor soft-hits reach post-filters.
    "conf": 0.22,
    # Lower IoU keeps overlapping walkers as separate people (groups at gates).
    "iou": 0.45,
    "verbose": False,
    # Match live letterbox tensor — avoid a second Ultralytics downscale.
    "imgsz": [INFERENCE_HEIGHT, INFERENCE_WIDTH],
}


def is_outdoor_gate_camera(location: str) -> bool:
    return str(location).strip() in OUTDOOR_GATE_CAMERAS


def frame_skip_for_camera(location: str, *, default_skip: int) -> int:
    """Outdoor gates analyze more often than the global multi-cam stride."""
    if is_outdoor_gate_camera(location):
        return max(1, int(OUTDOOR_GATE_FRAME_SKIP))
    return max(1, int(default_skip))


def live_face_force_every_for_camera(location: str, *, default_every: int) -> int:
    if is_outdoor_gate_camera(location):
        return max(1, int(OUTDOOR_GATE_LIVE_FACE_FORCE_EVERY))
    return max(1, int(default_every))


def gate_min_person_height_px(location: str) -> float:
    if is_outdoor_gate_camera(location):
        return float(OUTDOOR_GATE_MIN_PERSON_HEIGHT_PX)
    return 28.0


def gate_min_face_laplacian(location: str) -> float:
    if is_outdoor_gate_camera(location):
        return float(OUTDOOR_GATE_MIN_FACE_LAPLACIAN)
    return 3.0


def gate_skin_min_fraction(location: str) -> float:
    if is_outdoor_gate_camera(location):
        return float(OUTDOOR_GATE_SKIN_MIN_FRACTION)
    return 0.04


def gate_face_min_confidence(location: str) -> float:
    if is_outdoor_gate_camera(location):
        return float(OUTDOOR_GATE_FACE_MIN_CONF)
    return 0.18


def gate_min_face_box_px(location: str) -> int:
    if is_outdoor_gate_camera(location):
        return int(OUTDOOR_GATE_MIN_FACE_BOX_PX)
    return int(FACE_MIN_BOX_PX)


def gate_display_min_side(location: str, *, default_side: int) -> int:
    if is_outdoor_gate_camera(location):
        return int(OUTDOOR_GATE_DISPLAY_MIN_SIDE)
    return int(default_side)


def track_conf_for_camera(location: str) -> float:
    if is_outdoor_gate_camera(location):
        return float(OUTDOOR_GATE_TRACK_CONF)
    return float(TRACK_CONF)

# Face detection — strict gating to reject backs of heads, shoulders, hair blobs.
FACE_DETECT_MIN_CONFIDENCE = 0.88
# Per-landmark proxy threshold (face-level score when per-point scores unavailable).
FACE_LANDMARK_MIN_CONFIDENCE = 0.85
FACE_MIN_BOX_PX = 56
FACE_ASPECT_RATIO_MIN = 0.72  # width / height — reject wide shoulder/hair strips
FACE_ASPECT_RATIO_MAX = 1.38  # reject tall slivers
FACE_MIN_EYE_SPAN_RATIO = 0.22
FACE_MAX_EYE_SPAN_RATIO = 0.78


def _align_dim(value: int, stride: int) -> int:
    """Round up to the next multiple of ``stride`` (minimum one stride)."""
    stride = max(1, int(stride))
    value = max(1, int(value))
    return ((value + stride - 1) // stride) * stride


def configure_inference_size(
    width: int,
    height: int,
    *,
    align_stride: bool = True,
) -> Tuple[int, int]:
    """
    Set YOLO letterbox / imgsz at runtime (e.g. 640x360 for multi-stream VRAM).

    When ``align_stride`` is True, width/height are rounded up to ``YOLO_STRIDE``
    so GPU tensor paths stay aligned (640x360 → 640x384 with stride 32).
    Returns the effective (width, height) applied.
    """
    global INFERENCE_WIDTH, INFERENCE_HEIGHT, TRACK_KWARGS

    w = max(32, int(width))
    h = max(32, int(height))
    if align_stride:
        w = _align_dim(w, YOLO_STRIDE)
        h = _align_dim(h, YOLO_STRIDE)

    INFERENCE_WIDTH = w
    INFERENCE_HEIGHT = h
    TRACK_KWARGS = {
        **TRACK_KWARGS,
        "imgsz": [INFERENCE_HEIGHT, INFERENCE_WIDTH],
        "conf": 0.22,
    }
    return INFERENCE_WIDTH, INFERENCE_HEIGHT


def parse_inference_size(spec: str) -> Tuple[int, int]:
    """Parse ``WIDTHxHEIGHT`` / ``WIDTH,HEIGHT`` strings (e.g. ``640x360``)."""
    text = str(spec).strip().lower().replace(" ", "")
    for sep in ("x", ",", "*"):
        if sep in text:
            left, right = text.split(sep, 1)
            return int(left), int(right)
    raise ValueError(
        f"Invalid inference size '{spec}'. Use WIDTHxHEIGHT, e.g. 640x360"
    )
