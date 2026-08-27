"""
Aspect-preserving resize (letterbox) and bbox coordinate mapping.

Used by identity scan, main.py inference, and gpu_pipeline so faces are never
stretched when frames are fitted to fixed WxH tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]
PadColor = Tuple[int, int, int]


@dataclass(frozen=True)
class StretchResizeTransform:
    """Maps bbox coordinates from a stretched WxH resize back to source frame."""

    scale_x: float
    scale_y: float

    def unmap_bbox(self, bbox: Union[BBox, np.ndarray, Tuple[float, ...]]) -> BBox:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        return (
            int(round(x1 * self.scale_x)),
            int(round(y1 * self.scale_y)),
            int(round(x2 * self.scale_x)),
            int(round(y2 * self.scale_y)),
        )


@dataclass(frozen=True)
class LetterboxTransform:
    """Maps coordinates from a letterboxed canvas back to the source frame."""

    scale: float
    pad_x: float
    pad_y: float
    out_w: int
    out_h: int

    @property
    def is_identity(self) -> bool:
        return (
            self.scale == 1.0
            and self.pad_x == 0.0
            and self.pad_y == 0.0
        )

    def unmap_xyxy(self, x1: float, y1: float, x2: float, y2: float) -> BBox:
        if self.is_identity:
            return int(x1), int(y1), int(x2), int(y2)
        inv = 1.0 / self.scale if self.scale > 0 else 1.0
        return (
            int(round((x1 - self.pad_x) * inv)),
            int(round((y1 - self.pad_y) * inv)),
            int(round((x2 - self.pad_x) * inv)),
            int(round((y2 - self.pad_y) * inv)),
        )

    def unmap_bbox(self, bbox: Union[BBox, np.ndarray, Tuple[float, ...]]) -> BBox:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        return self.unmap_xyxy(x1, y1, x2, y2)


@dataclass(frozen=True)
class UniformScaleTransform:
    """Maps coordinates from a uniform downscale (e.g. width→640) back to source pixels."""

    scale: float  # work = source * scale (scale <= 1 when downscaling)

    @property
    def is_identity(self) -> bool:
        return self.scale >= 1.0 - 1e-9

    def unmap_xy(self, x: float, y: float) -> Tuple[float, float]:
        if self.is_identity:
            return x, y
        inv = 1.0 / self.scale
        return x * inv, y * inv

    def unmap_bbox(self, bbox: Union[BBox, np.ndarray, Tuple[float, ...]]) -> BBox:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        ux1, uy1 = self.unmap_xy(x1, y1)
        ux2, uy2 = self.unmap_xy(x2, y2)
        return (
            int(round(ux1)),
            int(round(uy1)),
            int(round(ux2)),
            int(round(uy2)),
        )

    def unmap_landmarks(
        self, landmarks: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
        if self.is_identity:
            return dict(landmarks)
        return {
            name: self.unmap_xy(float(pt[0]), float(pt[1]))
            for name, pt in landmarks.items()
        }


def downscale_frame_to_width(
    frame: np.ndarray,
    max_width: int = 640,
) -> Tuple[np.ndarray, UniformScaleTransform]:
    """Aspect-preserving downscale so output width ≤ max_width (no letterbox pad)."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, UniformScaleTransform(1.0)
    scale = max_width / float(w)
    new_w = max_width
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, UniformScaleTransform(scale=float(scale))


def downscale_frame_by_factor(
    frame: np.ndarray,
    factor: float = 0.5,
) -> Tuple[np.ndarray, UniformScaleTransform]:
    """Uniform downscale (e.g. 50%) — maps detect coords back via UniformScaleTransform."""
    scale = float(factor)
    if scale >= 0.999:
        return frame, UniformScaleTransform(1.0)
    if scale <= 0.0:
        raise ValueError("downscale factor must be positive")
    h, w = frame.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, UniformScaleTransform(scale=scale)


def letterbox_resize(
    frame: np.ndarray,
    width: int,
    height: int,
    pad_color: PadColor = (12, 16, 24),
) -> Tuple[np.ndarray, LetterboxTransform]:
    """Uniform-scale frame into width×height canvas with centered pad borders."""
    h, w = frame.shape[:2]
    if w == width and h == height:
        return frame, LetterboxTransform(1.0, 0.0, 0.0, width, height)

    canvas = np.full((height, width, 3), pad_color, dtype=frame.dtype)
    if w < 1 or h < 1:
        return canvas, LetterboxTransform(1.0, 0.0, 0.0, width, height)

    scale = min(width / float(w), height / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_x = (width - new_w) // 2
    pad_y = (height - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, LetterboxTransform(
        scale=float(scale),
        pad_x=float(pad_x),
        pad_y=float(pad_y),
        out_w=width,
        out_h=height,
    )


def square_padded_crop(
    frame: np.ndarray,
    bbox: Union[BBox, Tuple[float, ...]],
    pad_ratio: float = 0.25,
) -> np.ndarray:
    """Crop a square region around bbox center so thumbnails are not tall strips."""
    if frame is None or frame.size == 0:
        return frame
    x1, y1, x2, y2 = (int(v) for v in bbox[:4])
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)
    half = max(1, int(max(fw, fh) * (0.5 + pad_ratio)))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h, w = frame.shape[:2]
    nx1 = max(0, cx - half)
    ny1 = max(0, cy - half)
    nx2 = min(w, cx + half)
    ny2 = min(h, cy + half)
    crop = frame[ny1:ny2, nx1:nx2]
    return crop if crop.size else frame
