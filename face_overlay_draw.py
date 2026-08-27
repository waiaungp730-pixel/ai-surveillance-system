"""
Corner-bracket face boxes (CCTV facial-recognition UI style).

Used on live JPEG overlays and optional server-side preview burns.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


def _clamp_xyxy(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    w: int,
    h: int,
) -> BBox:
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def draw_corner_bracket_box(
    frame: np.ndarray,
    bbox: Union[BBox, Sequence[float]],
    *,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    arm_ratio: float = 0.22,
    crosshair: bool = True,
) -> np.ndarray:
    """White corner brackets + optional center crosshair (stock FR UI style)."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return frame
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
    except (TypeError, ValueError, IndexError):
        return frame
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_xyxy(x1, y1, x2, y2, w, h)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return frame
    arm = max(3, int(min(bw, bh) * float(arm_ratio)))
    t = max(1, int(thickness))
    c = tuple(int(v) for v in color[:3])
    line = cv2.LINE_AA

    # Top-left
    cv2.line(frame, (x1, y1), (x1 + arm, y1), c, t, line)
    cv2.line(frame, (x1, y1), (x1, y1 + arm), c, t, line)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - arm, y1), c, t, line)
    cv2.line(frame, (x2, y1), (x2, y1 + arm), c, t, line)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + arm, y2), c, t, line)
    cv2.line(frame, (x1, y2), (x1, y2 - arm), c, t, line)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - arm, y2), c, t, line)
    cv2.line(frame, (x2, y2), (x2, y2 - arm), c, t, line)

    if crosshair:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ch = max(2, min(bw, bh) // 12)
        cv2.line(frame, (cx - ch, cy), (cx + ch, cy), c, 1, line)
        cv2.line(frame, (cx, cy - ch), (cx, cy + ch), c, 1, line)

    return frame


def draw_corner_bracket_boxes(
    frame: np.ndarray,
    boxes: Sequence[Union[BBox, Sequence[float]]],
    *,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    out = frame
    for box in boxes or []:
        out = draw_corner_bracket_box(
            out,
            box,
            color=color,
            thickness=thickness,
        )
    return out
