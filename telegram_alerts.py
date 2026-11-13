"""
Telegram Bot API alerts for CCTV Person-of-Interest / watchlist matches.

Uses ``requests`` against the Bot API (no extra telegram SDK required).

Configure via environment:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
  TELEGRAM_ALERT_COOLDOWN_SEC   (optional, default 60)
  TELEGRAM_ALERT_MIN_PERCENT    (optional, default from blacklist tier ≥85)

Async delivery runs on a background thread pool so the tracker never blocks.
"""

from __future__ import annotations

import html
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Optional, Sequence, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
import requests

log = logging.getLogger("telegram_alerts")

try:
    from orion_frame_pipeline import BLACKLIST_ALERT_SIMILARITY

    _DEFAULT_MIN_PERCENT = int(round(BLACKLIST_ALERT_SIMILARITY * 100))
except Exception:
    _DEFAULT_MIN_PERCENT = 50

# Tier-1 watchlist / blacklist gate (percent similarity).
TELEGRAM_ALERT_MIN_PERCENT = int(
    os.environ.get("TELEGRAM_ALERT_MIN_PERCENT", str(_DEFAULT_MIN_PERCENT))
)
# Per (camera, person) cooldown — default 60s to avoid spam.
TELEGRAM_ALERT_COOLDOWN_SEC = float(
    os.environ.get("TELEGRAM_ALERT_COOLDOWN_SEC", "8")
)
# collage = scene + face panel (default); scene = full frame only; both = 2 photos
TELEGRAM_ALERT_STYLE = os.environ.get("TELEGRAM_ALERT_STYLE", "collage").strip().lower()


def _env_bool_alert(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None and name == "TELEGRAM_ALERT_NON_WATCHLIST":
        raw = os.environ.get("TELEGRAM_ALERT_STRANGERS")
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# Strangers-only by default — set TELEGRAM_ALERT_NAMED=1 or WATCHLIST=1 to re-enable.
TELEGRAM_ALERT_NON_WATCHLIST = _env_bool_alert("TELEGRAM_ALERT_NON_WATCHLIST", True)
TELEGRAM_ALERT_WATCHLIST = _env_bool_alert("TELEGRAM_ALERT_WATCHLIST", False)
TELEGRAM_ALERT_NAMED = _env_bool_alert("TELEGRAM_ALERT_NAMED", False)
TELEGRAM_SCENE_MAX_SIDE = int(os.environ.get("TELEGRAM_SCENE_MAX_SIDE", "1920"))
TELEGRAM_SCENE_JPEG_QUALITY = int(os.environ.get("TELEGRAM_SCENE_JPEG_QUALITY", "92"))
# Sharpen Telegram right panel only; saved captures / scene JPEG stay original.
# Default off — keep original CCTV pixels (set 1 for CLAHE/GFPGAN preview).
TELEGRAM_ENHANCE_DISPLAY = os.environ.get("TELEGRAM_ENHANCE_DISPLAY", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
CATCH_EVERY_PERSON = os.environ.get("CATCH_EVERY_PERSON", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
CATCH_BACKS = os.environ.get("CATCH_BACKS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FACE_ONLY_CATCH = os.environ.get("FACE_ONLY_CATCH", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Keep CCTV native pixels on the right panel (no 10–20× face blow-up).
TELEGRAM_NATIVE_PIXELS = os.environ.get("TELEGRAM_NATIVE_PIXELS", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
try:
    TELEGRAM_MAX_FACE_UPSCALE = float(
        os.environ.get(
            "TELEGRAM_MAX_FACE_UPSCALE",
            "3.5" if TELEGRAM_NATIVE_PIXELS else "2.5",
        )
    )
except (TypeError, ValueError):
    TELEGRAM_MAX_FACE_UPSCALE = 3.5 if TELEGRAM_NATIVE_PIXELS else 2.5
TELEGRAM_MAX_FACE_UPSCALE = max(1.0, min(TELEGRAM_MAX_FACE_UPSCALE, 8.0))
try:
    TELEGRAM_COLLAGE_HEIGHT = int(os.environ.get("TELEGRAM_COLLAGE_HEIGHT", "720"))
except (TypeError, ValueError):
    TELEGRAM_COLLAGE_HEIGHT = 720
TELEGRAM_COLLAGE_HEIGHT = max(320, min(TELEGRAM_COLLAGE_HEIGHT, 1200))
# Collage right panel must be a usable person plate (big upper-body), not a
# tiny forehead/face square letterboxed in black.
try:
    TELEGRAM_PANEL_MIN_HEIGHT = int(os.environ.get("TELEGRAM_PANEL_MIN_HEIGHT", "100"))
except (TypeError, ValueError):
    TELEGRAM_PANEL_MIN_HEIGHT = 100
TELEGRAM_PANEL_MIN_HEIGHT = max(72, min(TELEGRAM_PANEL_MIN_HEIGHT, 480))
try:
    TELEGRAM_PANEL_MIN_FILL = float(os.environ.get("TELEGRAM_PANEL_MIN_FILL", "0.28"))
except (TypeError, ValueError):
    TELEGRAM_PANEL_MIN_FILL = 0.28
TELEGRAM_PANEL_MIN_FILL = max(0.20, min(TELEGRAM_PANEL_MIN_FILL, 0.90))

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="telegram-alert")
_cooldown_lock = threading.Lock()
_last_sent: Dict[Tuple[str, int], float] = {}


def _telegram_panel_worth_sending(panel_bgr: Optional[np.ndarray]) -> Tuple[bool, str]:
    """
    True for person panels worth putting on the collage right side.

    Distant hallway faces (~56–90px) are allowed when they already clear the
    human/face gates — previously these were dropped as ``tiny_letterbox`` /
    ``small_square`` and no Telegram was sent at all.
    """
    if panel_bgr is None or getattr(panel_bgr, "size", 0) == 0:
        return False, "empty"
    try:
        arr = np.ascontiguousarray(panel_bgr)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if min(h, w) < 48:
            return False, f"too_tiny({w}x{h})"
        try:
            from face_print_filter import (
                is_likely_looking_down,
                is_likely_rear_view,
                is_likely_unusable_face_plate,
                capture_crop_passes_human_gate,
            )

            human_ok, human_why = capture_crop_passes_human_gate(arr)
            if not human_ok:
                return False, human_why or "non_human"
            rear, rear_why = is_likely_rear_view(arr)
            if rear and not CATCH_BACKS:
                return False, rear_why or "rear_view"
            looking, look_why = is_likely_looking_down(arr)
            if looking and h / float(max(1, w)) < 1.25:
                # Looking-down is OK on a tall bust plate; reject tight crown squares.
                return False, look_why or "looking_down"
            mush, mush_why = is_likely_unusable_face_plate(arr)
            if mush:
                return False, mush_why or "unusable"
        except Exception:
            pass
        portrait = h / float(max(1, w)) >= 1.18
        # Native letterbox: after max upscale, panel must still fill ~half collage.
        up = max(1.0, float(TELEGRAM_MAX_FACE_UPSCALE))
        fill_h = float(h) * up / float(max(1, TELEGRAM_COLLAGE_HEIGHT))
        if portrait and h >= int(TELEGRAM_PANEL_MIN_HEIGHT * 0.85):
            return True, "portrait"
        # Compact but real face plates (common on hallway cams) — still alert.
        if min(h, w) >= 56 and fill_h >= min(0.22, float(TELEGRAM_PANEL_MIN_FILL)):
            return True, f"compact({w}x{h},fill={fill_h:.2f})"
        if fill_h < float(TELEGRAM_PANEL_MIN_FILL) and h < TELEGRAM_PANEL_MIN_HEIGHT:
            return False, f"tiny_letterbox({w}x{h},fill={fill_h:.2f})"
        if (not portrait) and min(h, w) < 56:
            return False, f"small_square({w}x{h})"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def telegram_configured() -> bool:
    token, chat_id = _get_config()
    return bool(token and chat_id)


def _get_config() -> Tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "8505145239:AAH8qsPNsRgcACeebBquBV1i58dASYUXcKk").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "5242360409").strip()
    return token, chat_id


def _telegram_enhance_display() -> bool:
    return TELEGRAM_ENHANCE_DISPLAY


def _cooldown_seconds() -> float:
    try:
        return max(0.0, float(TELEGRAM_ALERT_COOLDOWN_SEC))
    except (TypeError, ValueError):
        return 60.0


def _telegram_alert_style() -> str:
    s = (TELEGRAM_ALERT_STYLE or "collage").strip().lower()
    if s in ("scene", "collage", "both"):
        return s
    return "collage"


def _encode_scene_for_alert(scene_bgr: np.ndarray) -> Optional[bytes]:
    """Full-resolution annotated CCTV frame — no face upscaling or collage."""
    return encode_bgr_jpeg(
        scene_bgr,
        quality=TELEGRAM_SCENE_JPEG_QUALITY,
        max_side=TELEGRAM_SCENE_MAX_SIDE,
    )


def encode_bgr_jpeg(
    image_bgr: Optional[np.ndarray],
    *,
    quality: int = 88,
    max_side: int = 1280,
) -> Optional[bytes]:
    """Encode a BGR ndarray to JPEG bytes (optionally downscale large frames)."""
    if image_bgr is None:
        return None
    try:
        frame = np.asarray(image_bgr)
        if frame.size == 0 or frame.ndim < 2:
            return None
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] > 3:
            frame = frame[:, :, :3]
        h, w = frame.shape[:2]
        side = max(h, w)
        if max_side > 0 and side > max_side:
            scale = max_side / float(side)
            frame = cv2.resize(
                frame,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        return buf.tobytes() if ok else None
    except Exception as exc:
        log.debug("JPEG encode failed: %s", exc)
        return None


# Back-compat alias used by older call sites / tests.
def encode_face_jpeg(face_bgr: np.ndarray) -> Optional[bytes]:
    return encode_bgr_jpeg(face_bgr, quality=88, max_side=640)


def annotate_alert_frame(
    frame_bgr: np.ndarray,
    *,
    person_name: str,
    confidence: int,
    bbox: Optional[Sequence[float]] = None,
    camera_label: str = "",
) -> np.ndarray:
    """
    Draw a red watchlist box + HUD strip on a full CCTV frame.
    Kept for demos / optional overlays; live Telegram style uses side-by-side collage.
    """
    out = np.ascontiguousarray(np.asarray(frame_bgr).copy())
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]

    x1 = y1 = x2 = y2 = None
    if bbox is not None and len(bbox) >= 4:
        try:
            bx = [float(v) for v in bbox[:4]]
            x1, y1, x2, y2 = (int(round(v)) for v in bx)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                x1 = y1 = x2 = y2 = None
        except (TypeError, ValueError):
            x1 = y1 = x2 = y2 = None

    alert_bgr = (40, 40, 220)  # vivid red in BGR
    if x1 is not None:
        cv2.rectangle(out, (x1, y1), (x2, y2), alert_bgr, 3, cv2.LINE_AA)
        tag = f"POI {confidence}%"
        ty = max(22, y1 - 8)
        cv2.putText(
            out,
            tag,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            alert_bgr,
            2,
            cv2.LINE_AA,
        )

    # Top HUD ribbon
    ribbon_h = min(48, max(28, h // 18))
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, ribbon_h), (20, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
    title = f"CCTV ALERT · {person_name} · {confidence}%"
    if camera_label:
        title = f"{title} · {camera_label}"
    cv2.putText(
        out,
        title[:90],
        (10, int(ribbon_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def face_square_xyxy(
    bbox: Sequence[float],
    frame_shape: Tuple[int, ...],
    *,
    pad_ratio: float = 0.12,
    max_frame_frac: float = 0.22,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Build a tight square around a face bbox for drawing on the CCTV scene.

    Rejects boxes that look like full-body / floor regions (too large for a face).
    """
    try:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError, IndexError):
        return None
    if x2 <= x1 or y2 <= y1 or h < 2 or w < 2:
        return None
    bw = x2 - x1
    bh = y2 - y1
    area_frac = (bw * bh) / float(h * w)
    # Body boxes / misplaced floors occupy too much of the frame.
    if area_frac > float(max_frame_frac) or max(bw, bh) > 0.55 * max(h, w):
        return None

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(bw, bh)
    half = 0.5 * side * (1.0 + float(pad_ratio))
    sx1 = int(round(cx - half))
    sy1 = int(round(cy - half))
    sx2 = int(round(cx + half))
    sy2 = int(round(cy + half))
    sx1 = max(0, min(w - 1, sx1))
    sy1 = max(0, min(h - 1, sy1))
    sx2 = max(0, min(w - 1, sx2))
    sy2 = max(0, min(h - 1, sy2))
    if sx2 - sx1 < 8 or sy2 - sy1 < 8:
        return None
    return sx1, sy1, sx2, sy2


def draw_face_square_on_scene(
    frame_bgr: np.ndarray,
    face_bbox: Optional[Sequence[float]],
) -> np.ndarray:
    """Copy scene and draw a red square on the face only (never a body/floor box)."""
    return draw_face_squares_on_scene(frame_bgr, [face_bbox] if face_bbox is not None else None)


def draw_face_squares_on_scene(
    frame_bgr: np.ndarray,
    face_bboxes: Optional[Sequence[Optional[Sequence[float]]]],
) -> np.ndarray:
    """Copy scene and draw red squares on every valid face box."""
    out = np.ascontiguousarray(np.asarray(frame_bgr).copy())
    if not face_bboxes:
        return out
    for face_bbox in face_bboxes:
        if face_bbox is None:
            continue
        sq = face_square_xyxy(face_bbox, out.shape)
        if sq is None:
            continue
        x1, y1, x2, y2 = sq
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 40, 220), 3, cv2.LINE_AA)
    return out


def compose_alert_collage(
    scene_bgr: np.ndarray,
    face_bgr: np.ndarray,
    *,
    target_height: Optional[int] = None,
    gap: int = 4,
    max_face_upscale: Optional[float] = None,
    enhance_face: bool = False,
) -> np.ndarray:
    """
    Side-by-side Telegram style: full CCTV scene (left) + person crop (right).

    Right panel keeps native CCTV pixels by default (letterbox), instead of
    blowing a distant 40px face up to collage height.
    """
    return compose_alert_collage_multi(
        scene_bgr,
        [face_bgr],
        target_height=target_height,
        gap=gap,
        max_face_upscale=max_face_upscale,
        enhance_face=enhance_face,
    )


def _telegram_person_panel(
    face_bgr: np.ndarray,
    *,
    frame_bgr: Optional[np.ndarray] = None,
    bbox: Optional[Sequence[float]] = None,
    person_bbox: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Prefer a native person / upper-body crop from the scene for the right panel.

    Distant tight face squares look like pixel mush when enlarged — a hallway
    person plate keeps real CCTV detail (shirt, gait, context).
    """
    crop = np.ascontiguousarray(face_bgr) if face_bgr is not None else None
    if frame_bgr is not None:
        try:
            from captures import (
                CAPTURE_UPPER_BODY_BELOW_FACE_RATIO,
                CAPTURE_UPPER_BODY_SIDE_PAD_RATIO,
                CAPTURE_UPPER_BODY_TOP_PAD_RATIO,
                crop_upper_body_from_frame,
                display_image_is_portrait_body,
            )

            # Prefer YOLO person box (full body / bust plate) when available.
            if person_bbox is not None and len(person_bbox) >= 4:
                fh, fw = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
                x1, y1, x2, y2 = (float(v) for v in list(person_bbox)[:4])
                bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                pad = 0.12
                xx1 = max(0, int(x1 - pad * bw))
                yy1 = max(0, int(y1 - pad * bh))
                xx2 = min(fw, int(x2 + pad * bw))
                yy2 = min(fh, int(y2 + pad * bh))
                if xx2 - xx1 >= 24 and yy2 - yy1 >= 24:
                    person_plate = np.ascontiguousarray(frame_bgr[yy1:yy2, xx1:xx2])
                    try:
                        from face_print_filter import capture_crop_passes_human_gate

                        plate_ok, _ = capture_crop_passes_human_gate(person_plate)
                        if not plate_ok:
                            person_plate = None
                    except Exception:
                        pass
                    if person_plate is not None:
                        # Always prefer a larger person plate over a tight face square.
                        if crop is None or (
                            int(person_plate.shape[0]) * int(person_plate.shape[1])
                            >= int(crop.shape[0]) * int(crop.shape[1])
                        ):
                            crop = person_plate

            if bbox is not None and len(bbox) >= 4:
                bigger = crop_upper_body_from_frame(
                    np.ascontiguousarray(frame_bgr),
                    bbox,
                    person_bbox,
                    draw_face_box=False,
                    below_face_ratio=max(2.4, CAPTURE_UPPER_BODY_BELOW_FACE_RATIO),
                    side_pad_ratio=max(1.05, CAPTURE_UPPER_BODY_SIDE_PAD_RATIO),
                    top_pad_ratio=max(0.45, CAPTURE_UPPER_BODY_TOP_PAD_RATIO),
                )
                if bigger is not None and getattr(bigger, "size", 0) > 0:
                    # Prefer bust/upper-body whenever it is larger than the face plate.
                    if crop is None or (
                        int(bigger.shape[0]) * int(bigger.shape[1])
                        >= int(crop.shape[0]) * int(crop.shape[1])
                    ):
                        crop = bigger
                    elif display_image_is_portrait_body(bigger) and not display_image_is_portrait_body(
                        crop
                    ):
                        crop = bigger
        except Exception:
            pass
    if crop is not None and getattr(crop, "size", 0) > 0:
        try:
            from face_print_filter import is_likely_streaked_or_corrupt_capture

            streak, _ = is_likely_streaked_or_corrupt_capture(crop)
            if streak and face_bgr is not None:
                crop = np.ascontiguousarray(face_bgr)
        except Exception:
            pass
    return crop if crop is not None else np.ascontiguousarray(face_bgr)


def _letterbox_face_panel(
    face_bgr: np.ndarray,
    *,
    target_height: int,
    max_face_upscale: float,
    enhance: bool = False,
) -> np.ndarray:
    """
    Fit the capture into the Telegram side panel.

    Native mode (default): never blow up a distant face — pad with black so the
    right panel shows real CCTV pixels, same sharpness as the hallway scene.
    """
    face = np.asarray(face_bgr)
    if enhance:
        from captures import enhance_capture_for_display

        face = enhance_capture_for_display(face)
    else:
        face = np.ascontiguousarray(face)
    if face.ndim == 2:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
    elif face.shape[2] > 3:
        face = face[:, :, :3]
    th = max(160, int(target_height))
    fh, fw = int(face.shape[0]), int(face.shape[1])
    # Clip full-body panels — keep head→chest/waist portrait only.
    if fw > 0 and fh / float(fw) > 1.65:
        keep_h = max(1, int(round(fw * 1.48)))
        face = face[:keep_h]
        fh, fw = int(face.shape[0]), int(face.shape[1])
    panel_w = max(int(round(th * 0.78)), 360)
    if fw < 1 or fh < 1:
        return np.zeros((th, panel_w, 3), dtype=np.uint8)

    fit = min(panel_w / float(fw), th / float(fh))
    # Downscale freely; upscale up to TELEGRAM_MAX_FACE_UPSCALE (default 3.5×).
    up_cap = max(1.0, float(max_face_upscale))
    if TELEGRAM_NATIVE_PIXELS:
        up_cap = min(up_cap, TELEGRAM_MAX_FACE_UPSCALE)
    if fit >= 1.0:
        scale = min(fit, up_cap)
    else:
        scale = fit
    nw = max(1, int(round(fw * scale)))
    nh = max(1, int(round(fh * scale)))
    # Prefer filling more of the panel when still tiny after the first scale.
    if nh < int(th * 0.50) and fw > 0 and fh > 0:
        boost = min(up_cap, (th * 0.55) / float(fh), panel_w / float(fw))
        if boost > scale:
            scale = boost
            nw = max(1, int(round(fw * scale)))
            nh = max(1, int(round(fh * scale)))
    interp = cv2.INTER_AREA if scale < 0.95 else (
        cv2.INTER_LANCZOS4 if scale > 1.05 else cv2.INTER_LINEAR
    )
    resized = cv2.resize(face, (nw, nh), interpolation=interp)

    # Letterbox — real pixels centered, no cover-fill mush.
    canvas = np.zeros((th, panel_w, 3), dtype=np.uint8)
    x0 = max(0, (panel_w - nw) // 2)
    y0 = max(0, (th - nh) // 2)
    x1 = min(panel_w, x0 + nw)
    y1 = min(th, y0 + nh)
    canvas[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
    return canvas


def _panels_look_identical(
    panel_a: Optional[np.ndarray],
    panel_b: Optional[np.ndarray],
    *,
    thresh: float = 6.0,
) -> bool:
    """True when two BGR panels are the same crop (duplicate multi-catch slots)."""
    if panel_a is None or panel_b is None:
        return False
    try:
        a = np.asarray(panel_a)
        b = np.asarray(panel_b)
        if a.size == 0 or b.size == 0:
            return False
        sa = cv2.resize(a, (48, 48), interpolation=cv2.INTER_AREA)
        sb = cv2.resize(b, (48, 48), interpolation=cv2.INTER_AREA)
        return float(np.mean(cv2.absdiff(sa, sb))) < float(thresh)
    except Exception:
        return False


def _dedupe_face_panels(panels: Sequence[np.ndarray]) -> List[np.ndarray]:
    unique: List[np.ndarray] = []
    for panel in panels:
        if panel is None or getattr(panel, "size", 0) == 0:
            continue
        arr = np.ascontiguousarray(panel)
        if any(_panels_look_identical(arr, kept) for kept in unique):
            continue
        unique.append(arr)
    return unique


def compose_alert_collage_multi(
    scene_bgr: np.ndarray,
    faces_bgr: Sequence[np.ndarray],
    *,
    target_height: Optional[int] = None,
    gap: int = 4,
    max_face_upscale: Optional[float] = None,
    enhance_face: bool = False,
) -> np.ndarray:
    """Scene left + one or more person panels on the right (multi-person catch)."""
    scene = np.asarray(scene_bgr)
    if scene.ndim == 2:
        scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    elif scene.shape[2] > 3:
        scene = scene[:, :, :3]

    th = max(160, int(target_height if target_height is not None else TELEGRAM_COLLAGE_HEIGHT))
    up_cap = float(
        max_face_upscale if max_face_upscale is not None else TELEGRAM_MAX_FACE_UPSCALE
    )
    sh, sw = scene.shape[:2]
    scale = th / float(max(1, sh))
    scene_w = max(1, int(round(sw * scale)))
    # Keep scene readable but leave room for a large person panel.
    max_scene_w = int(th * 1.85)
    if scene_w > max_scene_w:
        scene_w = max_scene_w
    scene_r = cv2.resize(scene, (scene_w, th), interpolation=cv2.INTER_AREA)

    panels = [
        _letterbox_face_panel(
            face,
            target_height=th,
            max_face_upscale=up_cap,
            enhance=enhance_face,
        )
        for face in faces_bgr
        if face is not None and getattr(face, "size", 0) > 0
    ]
    panels = _dedupe_face_panels(panels)
    if not panels:
        return scene_r

    if len(panels) == 1:
        parts = [scene_r]
        if gap > 0:
            parts.append(np.zeros((th, int(gap), 3), dtype=np.uint8))
        parts.append(panels[0])
        return np.hstack(parts)

    # Two or more people: stack large portrait panels in one right column.
    n = len(panels)
    col_w = max(int(p.shape[1]) for p in panels)
    col_w = max(col_w, int(round(th * 0.78)), 360)
    row_h = max(1, th // n)
    stacked_rows: list[np.ndarray] = []
    vgap = max(0, int(gap))
    for idx, panel in enumerate(panels):
        ph, pw = int(panel.shape[0]), int(panel.shape[1])
        scale = min(col_w / float(max(1, pw)), row_h / float(max(1, ph)))
        nw = max(1, int(round(pw * scale)))
        nh = max(1, int(round(ph * scale)))
        interp = cv2.INTER_AREA if scale < 0.95 else cv2.INTER_LINEAR
        resized = cv2.resize(panel, (nw, nh), interpolation=interp)
        row = np.zeros((row_h, col_w, 3), dtype=np.uint8)
        x0 = max(0, (col_w - nw) // 2)
        y0 = max(0, (row_h - nh) // 2)
        row[y0:y0 + nh, x0:x0 + nw] = resized[:nh, :nw]
        stacked_rows.append(row)
        if vgap > 0 and idx < n - 1:
            stacked_rows.append(np.zeros((vgap, col_w, 3), dtype=np.uint8))
    person_col = np.vstack(stacked_rows)
    if person_col.shape[0] != th:
        person_col = cv2.resize(
            person_col,
            (col_w, th),
            interpolation=cv2.INTER_AREA,
        )
    strip = np.zeros((th, int(max(0, gap)), 3), dtype=np.uint8) if gap > 0 else None
    parts = [scene_r]
    if strip is not None:
        parts.append(strip)
    parts.append(person_col)
    return np.hstack(parts)


def build_alert_caption_html(
    *,
    camera_name: str,
    location: str,
    person_name: str,
    confidence: int,
    timestamp: str,
    person_id: Optional[int] = None,
    alert_kind: str = "watchlist",
    image_quality: Optional[str] = None,
) -> str:
    """
    HTML caption matching the dashboard review modal.

    ``alert_kind`` is ``watchlist`` for enrolled POI matches, ``named`` for
    Named People CCTV matches (not on the watchlist), or ``incident`` for
    stranger catches.
    """
    cam = html.escape(str(camera_name or "CAM"))
    loc = html.escape(str(location or cam))
    name = html.escape(str(person_name or "Unknown"))
    ts = html.escape(str(timestamp or ""))
    kind = str(alert_kind or "watchlist").strip().lower()
    if kind == "incident":
        event = f"Face detected — {name}"
    elif kind == "named":
        event = f"Named match — {name}"
    else:
        event = f"Watchlist match — {name}"
    lines = [
        "<b>LIVE AI INFRACTION</b>",
        event,
        f"<b>Name:</b> {name}",
        f"<b>Camera:</b> {cam}",
        f"<b>Location:</b> {loc}",
        f"<b>Time:</b> {ts}",
    ]
    quality = str(image_quality or "").strip() or "—"
    lines.append(f"<b>Quality:</b> {html.escape(quality)}")
    return "\n".join(lines)


def _allow_send(camera_key: str, person_id: int, cooldown_sec: float) -> bool:
    """Return True once and stamp cooldown; False if still cooling down."""
    key = (str(camera_key), int(person_id))
    now = time.time()
    with _cooldown_lock:
        last = _last_sent.get(key, 0.0)
        if cooldown_sec > 0 and (now - last) < cooldown_sec:
            return False
        _last_sent[key] = now
        return True


def _release_cooldown(camera_key: str, person_id: int) -> None:
    """Undo a cooldown reservation when the send failed before delivery."""
    key = (str(camera_key), int(person_id))
    with _cooldown_lock:
        _last_sent.pop(key, None)


def send_watchlist_alert(
    *,
    face_jpeg: bytes,
    frame_jpeg: Optional[bytes] = None,
    camera_name: str,
    location: str,
    person_name: str,
    confidence: int,
    timestamp: str,
    person_id: Optional[int] = None,
    alert_kind: str = "watchlist",
    image_quality: Optional[str] = None,
) -> bool:
    """
    POST a Telegram photo alert.

    Default (``TELEGRAM_ALERT_STYLE=collage``): side-by-side scene (red face
    box) + natural face crop. Set ``scene`` for full frame only, or ``both``
    for two separate photos.
    """
    token, chat_id = _get_config()
    if not token or not chat_id:
        return False

    caption = build_alert_caption_html(
        camera_name=camera_name,
        location=location,
        person_name=person_name,
        confidence=confidence,
        timestamp=timestamp,
        person_id=person_id,
        alert_kind=alert_kind,
        image_quality=image_quality,
    )

    style = _telegram_alert_style()
    photo_jpeg = face_jpeg
    if frame_jpeg and style == "scene":
        photo_jpeg = frame_jpeg
    elif frame_jpeg and style == "collage":
        try:
            scene = cv2.imdecode(
                np.frombuffer(frame_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            face = cv2.imdecode(
                np.frombuffer(face_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if scene is not None and face is not None:
                collage = compose_alert_collage(
                    scene, face, enhance_face=_telegram_enhance_display()
                )
                encoded = encode_bgr_jpeg(collage, quality=90, max_side=1600)
                if isinstance(encoded, (bytes, bytearray)) and len(encoded) > 0:
                    photo_jpeg = encoded
        except Exception as exc:
            log.debug("Collage compose failed, sending scene: %s", exc)
            photo_jpeg = frame_jpeg
    elif frame_jpeg:
        photo_jpeg = frame_jpeg

    if style == "both" and frame_jpeg and face_jpeg:
        try:
            media = json.dumps(
                [
                    {
                        "type": "photo",
                        "media": "attach://scene",
                        "caption": caption,
                        "parse_mode": "HTML",
                    },
                    {"type": "photo", "media": "attach://face"},
                ]
            )
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMediaGroup",
                data={"chat_id": chat_id, "media": media},
                files={
                    "scene": ("scene.jpg", frame_jpeg, "image/jpeg"),
                    "face": ("face.jpg", face_jpeg, "image/jpeg"),
                },
                timeout=25,
            )
            if response.status_code < 400:
                log.info(
                    "Telegram POI alert sent (scene+face) — %s @ %s (%s%%)",
                    person_name,
                    camera_name,
                    confidence,
                )
                return True
            log.debug(
                "sendMediaGroup failed (%s), falling back to scene photo",
                response.status_code,
            )
        except Exception as exc:
            log.debug("sendMediaGroup failed: %s", exc)

    try:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        response = requests.post(
            url,
            data={
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": "HTML",
            },
            files={"photo": ("alert.jpg", photo_jpeg, "image/jpeg")},
            timeout=20,
        )

        if response.status_code >= 400:
            log.warning(
                "Telegram alert HTTP %s: %s",
                response.status_code,
                response.text[:300],
            )
            return False

        log.info(
            "Telegram POI alert sent — %s @ %s (%s%%)",
            person_name,
            camera_name,
            confidence,
        )
        return True
    except requests.RequestException as exc:
        log.warning("Telegram alert failed: %s", exc)
        return False


def send_target_photo(
    jpeg_bytes: bytes,
    person_id: int,
    cam_zone: str,
    timestamp: str,
) -> bool:
    """Legacy single-photo helper (kept for older callers)."""
    return send_watchlist_alert(
        face_jpeg=jpeg_bytes,
        frame_jpeg=None,
        camera_name=str(cam_zone),
        location=str(cam_zone),
        person_name=f"Person #{int(person_id)}",
        confidence=TELEGRAM_ALERT_MIN_PERCENT,
        timestamp=timestamp,
        person_id=int(person_id),
    )


def maybe_send_target_alert_async(
    face_bgr: np.ndarray,
    person_id: int,
    camera_zone: str,
    match_percent: int,
    *,
    frame_bgr: Optional[np.ndarray] = None,
    person_name: Optional[str] = None,
    camera_name: Optional[str] = None,
    location: Optional[str] = None,
    bbox: Optional[Sequence[float]] = None,
    person_bbox: Optional[Sequence[float]] = None,
    cooldown_sec: Optional[float] = None,
    min_percent: Optional[int] = None,
    alert_kind: str = "watchlist",
    image_quality: Optional[str] = None,
) -> bool:
    """
    Queue an asynchronous Telegram POI / watchlist alert.

    Parameters
    ----------
    face_bgr :
        Cropped face thumbnail (BGR).
    person_id :
        Known / watchlist person id.
    camera_zone :
        Camera id or zone key used for cooldown grouping.
    match_percent :
        Similarity confidence 0–100.
    frame_bgr :
        Optional full CCTV frame for an annotated scene snapshot.
    person_name / camera_name / location :
        Display fields for the HTML caption.
    bbox :
        Optional xyxy box on ``frame_bgr`` for annotation.
    cooldown_sec :
        Override per-(camera, person) silence window (default 60s).
    """
    try:
        return _queue_target_alert(
            face_bgr,
            person_id,
            camera_zone,
            match_percent,
            frame_bgr=frame_bgr,
            person_name=person_name,
            camera_name=camera_name,
            location=location,
            bbox=bbox,
            person_bbox=person_bbox,
            cooldown_sec=cooldown_sec,
            min_percent=min_percent,
            alert_kind=alert_kind,
            image_quality=image_quality,
        )
    except Exception:
        log.exception(
            "Telegram queue failed — person #%s @ %s",
            person_id,
            camera_zone,
        )
        return False


def _jpeg_bytes(value: object) -> Optional[bytes]:
    if isinstance(value, (bytes, bytearray)) and len(value) > 0:
        return bytes(value)
    return None


def _text_or(value: object, fallback: str) -> str:
    if value is None or isinstance(value, np.ndarray):
        return fallback
    text = str(value).strip()
    return text or fallback


def _alert_kind_allowed(alert_kind: str) -> bool:
    """Whether this alert type is enabled via dashboard settings / env flags."""
    kind = (alert_kind or "watchlist").strip().lower()
    try:
        from announcement_settings import (
            telegram_alert_named_enabled,
            telegram_alert_strangers_enabled,
            telegram_alert_watchlist_enabled,
        )

        if kind in ("incident", "stranger"):
            return telegram_alert_strangers_enabled()
        if kind == "named":
            return telegram_alert_named_enabled()
        # Blacklist shares the Watchlist Telegram toggle (same chat / gate).
        if kind in ("watchlist", "blacklist", "poi"):
            return telegram_alert_watchlist_enabled()
        return telegram_alert_watchlist_enabled()
    except Exception:
        if kind in ("incident", "stranger"):
            return TELEGRAM_ALERT_NON_WATCHLIST
        if kind == "named":
            return TELEGRAM_ALERT_NAMED
        return TELEGRAM_ALERT_WATCHLIST


def _queue_target_alert(
    face_bgr: np.ndarray,
    person_id: int,
    camera_zone: str,
    match_percent: int,
    *,
    frame_bgr: Optional[np.ndarray] = None,
    person_name: Optional[str] = None,
    camera_name: Optional[str] = None,
    location: Optional[str] = None,
    bbox: Optional[Sequence[float]] = None,
    person_bbox: Optional[Sequence[float]] = None,
    cooldown_sec: Optional[float] = None,
    min_percent: Optional[int] = None,
    alert_kind: str = "watchlist",
    image_quality: Optional[str] = None,
) -> bool:
    if person_id is None:
        return False
    try:
        from announcement_settings import announcements_allowed_now

        if not announcements_allowed_now():
            log.info(
                "Telegram skip — outside announcement window (person #%s @ %s)",
                person_id,
                camera_zone,
            )
            return False
    except Exception:
        pass
    if not _alert_kind_allowed(alert_kind):
        log.info(
            "Telegram skip — %s alerts disabled (person #%s @ %s)",
            alert_kind,
            person_id,
            camera_zone,
        )
        return False
    try:
        from face_print_filter import (
            is_junk_person_plate_reason,
            is_keepable_person_pose_reason,
            is_likely_non_face_object,
            is_likely_rear_view,
        )

        # Always drop walking-away / back-of-head unless CATCH_BACKS=1.
        rear, reason = is_likely_rear_view(face_bgr)
        if bool(rear) and not CATCH_BACKS:
            print(f"[Telegram] skip back-of-head @ {camera_zone} ({reason})")
            log.info(
                "Telegram skip — back-of-head @ %s (%s)",
                camera_zone,
                reason,
            )
            return False
        try:
            from face_print_filter import is_likely_unusable_face_plate

            mush, mush_why = is_likely_unusable_face_plate(face_bgr)
            if mush:
                print(f"[Telegram] skip unusable @ {camera_zone} ({mush_why})")
                log.info(
                    "Telegram skip — unusable @ %s (%s)",
                    camera_zone,
                    mush_why,
                )
                return False
        except Exception:
            pass
        try:
            from face_print_filter import is_likely_looking_down

            looking, look_why = is_likely_looking_down(face_bgr)
            if looking:
                try:
                    fh, fw = int(face_bgr.shape[0]), int(face_bgr.shape[1])
                    tall = fh / float(max(1, fw)) >= 1.25 and min(fh, fw) >= 120
                except Exception:
                    tall = False
                if not tall:
                    print(f"[Telegram] skip looking-down @ {camera_zone} ({look_why})")
                    log.info(
                        "Telegram skip — looking-down @ %s (%s)",
                        camera_zone,
                        look_why,
                    )
                    return False
        except Exception:
            pass
        try:
            h, w = int(face_bgr.shape[0]), int(face_bgr.shape[1])
            min_side = int(os.environ.get("TELEGRAM_FACE_MIN_SIDE", "40"))
            if min(h, w) < max(40, min_side):
                print(
                    f"[Telegram] skip tiny @ {camera_zone} ({w}x{h} < {min_side}px)"
                )
                return False
        except Exception:
            pass
        bad_obj, obj_reason = is_likely_non_face_object(face_bgr)
        if bool(bad_obj):
            keep_pose = CATCH_EVERY_PERSON and (
                is_keepable_person_pose_reason(obj_reason)
                or not is_junk_person_plate_reason(obj_reason)
            )
            # Keepable "pose" that is actually a rear view must still be blocked.
            if keep_pose and (not CATCH_BACKS or FACE_ONLY_CATCH):
                rear_tokens = (
                    "rear_view",
                    "back_of_head",
                    "back_head",
                    "walking_away",
                    "nape",
                    "crown_neck",
                    "hand_crown",
                )
                if any(t in str(obj_reason).lower() for t in rear_tokens):
                    keep_pose = False
            if FACE_ONLY_CATCH and keep_pose:
                # Faces-only mode: do not Telegram body/pose substitutes.
                keep_pose = False
            if not keep_pose:
                print(f"[Telegram] skip non-face @ {camera_zone} ({obj_reason})")
                log.info(
                    "Telegram skip — non-face @ %s (%s)",
                    camera_zone,
                    obj_reason,
                )
                return False
        from face_print_filter import is_likely_streaked_or_corrupt_capture

        streak, streak_reason = is_likely_streaked_or_corrupt_capture(face_bgr)
        if bool(streak):
            soft_ok = CATCH_EVERY_PERSON and any(
                tok in str(streak_reason).lower()
                for tok in ("narrow_strip", "thin_body", "too_tiny")
            )
            if not soft_ok:
                print(f"[Telegram] skip bad panel @ {camera_zone} ({streak_reason})")
                log.info(
                    "Telegram skip — bad panel @ %s (%s)",
                    camera_zone,
                    streak_reason,
                )
                return False
    except Exception:
        pass
    quality_label = str(image_quality or "").strip()
    if not quality_label:
        try:
            from identity import describe_face_sharpness

            quality_label, _ = describe_face_sharpness(face_bgr)
        except Exception:
            quality_label = ""
    try:
        percent = int(np.asarray(match_percent).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        percent = 0
    gate = int(min_percent if min_percent is not None else TELEGRAM_ALERT_MIN_PERCENT)
    if percent < gate:
        log.info(
            "Telegram skip — person #%s @ %s conf %s%% < gate %s%%",
            person_id,
            camera_zone,
            percent,
            gate,
        )
        return False
    if not telegram_configured():
        log.warning(
            "Telegram skip — TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in this process"
        )
        return False

    cool = float(cooldown_sec) if cooldown_sec is not None else _cooldown_seconds()
    cam_key = _text_or(camera_zone, _text_or(camera_name, "cam"))
    if not _allow_send(cam_key, int(person_id), cool):
        log.info(
            "Telegram cooldown — skip person #%s @ %s",
            person_id,
            cam_key,
        )
        return False

    display_name = _text_or(person_name, f"Person #{int(person_id)}")
    cam_label = _text_or(camera_name, str(camera_zone))
    loc_label = _text_or(location, cam_label)
    conf = percent
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    style = _telegram_alert_style()
    try:
        face_copy = np.ascontiguousarray(face_bgr)
    except Exception:
        _release_cooldown(cam_key, int(person_id))
        return False
    scene_copy = None
    if isinstance(frame_bgr, np.ndarray) and getattr(frame_bgr, "size", 0) > 0:
        scene_copy = np.ascontiguousarray(frame_bgr)
    bbox_copy = list(bbox[:4]) if bbox is not None and len(bbox) >= 4 else None
    person_bbox_copy = (
        list(person_bbox[:4])
        if person_bbox is not None and len(person_bbox) >= 4
        else None
    )

    def _worker() -> None:
        natural = _jpeg_bytes(encode_bgr_jpeg(face_copy, quality=90, max_side=640))
        if natural is None:
            _release_cooldown(cam_key, int(person_id))
            log.warning(
                "Telegram skip — face JPEG encode failed for person #%s", person_id
            )
            return
        frame_jpeg: Optional[bytes] = None
        face_jpeg = natural
        if scene_copy is not None:
            scene_jpeg = _jpeg_bytes(_encode_scene_for_alert(scene_copy))
            if style == "scene":
                face_jpeg = scene_jpeg or natural
            elif style == "collage":
                try:
                    from face_print_filter import (
                        is_junk_person_plate_reason,
                        is_keepable_person_pose_reason,
                        is_likely_non_face_object,
                        is_likely_rear_view,
                    )

                    panel = _telegram_person_panel(
                        face_copy,
                        frame_bgr=scene_copy,
                        bbox=bbox_copy,
                        person_bbox=person_bbox_copy,
                    )
                    # Prefer the big upper-body plate. If it looks like a back /
                    # junk, try the face crop only when that crop is also a large
                    # usable panel — never fall back to a tiny forehead postage stamp.
                    rear_panel, rear_why = is_likely_rear_view(panel)
                    bad_panel, bad_why = is_likely_non_face_object(panel)
                    if rear_panel or bad_panel:
                        why = rear_why or bad_why
                        face_ok, face_ok_why = _telegram_panel_worth_sending(face_copy)
                        if rear_panel and not CATCH_BACKS:
                            if face_ok:
                                log.info(
                                    "Telegram collage back panel @ %s (%s) — using large face plate",
                                    cam_key,
                                    why,
                                )
                                panel = face_copy
                            else:
                                _release_cooldown(cam_key, int(person_id))
                                print(
                                    f"[Telegram] skip small/back panel @ {cam_key} "
                                    f"({why}; face={face_ok_why})"
                                )
                                return
                        else:
                            keep_person = CATCH_EVERY_PERSON and (
                                (rear_panel and CATCH_BACKS)
                                or is_keepable_person_pose_reason(why)
                                or not is_junk_person_plate_reason(why)
                            )
                            if keep_person and rear_panel and not CATCH_BACKS:
                                keep_person = False
                            if keep_person:
                                log.info(
                                    "Telegram collage person-plate @ %s (%s)",
                                    cam_key,
                                    why,
                                )
                            elif face_ok:
                                log.info(
                                    "Telegram collage panel bad @ %s (%s) — using large face plate",
                                    cam_key,
                                    why,
                                )
                                panel = face_copy
                            else:
                                _release_cooldown(cam_key, int(person_id))
                                print(
                                    f"[Telegram] skip bad/small panel @ {cam_key} "
                                    f"({why}; face={face_ok_why})"
                                )
                                return
                    # Never send a banister/wood/rail plate as the person panel.
                    bad_face, face_why = is_likely_non_face_object(panel)
                    if bad_face and is_junk_person_plate_reason(face_why):
                        _release_cooldown(cam_key, int(person_id))
                        print(
                            f"[Telegram] skip non-face panel @ {cam_key} ({face_why})"
                        )
                        return
                    panel_ok, panel_why = _telegram_panel_worth_sending(panel)
                    if not panel_ok:
                        # Don't drop the alert — send the full CCTV scene instead.
                        if scene_jpeg is not None:
                            print(
                                f"[Telegram] weak collage panel @ {cam_key} "
                                f"({panel_why}) — sending scene"
                            )
                            log.info(
                                "Telegram weak panel @ %s (%s) — scene fallback",
                                cam_key,
                                panel_why,
                            )
                            face_jpeg = scene_jpeg
                        else:
                            _release_cooldown(cam_key, int(person_id))
                            print(
                                f"[Telegram] skip tiny collage panel @ {cam_key} ({panel_why})"
                            )
                            return
                    else:
                        collage = compose_alert_collage(
                            scene_copy,
                            panel,
                            enhance_face=_telegram_enhance_display(),
                        )
                        encoded = _jpeg_bytes(
                            encode_bgr_jpeg(collage, quality=90, max_side=1600)
                        )
                        face_jpeg = encoded or scene_jpeg or natural
                except Exception as exc:
                    log.debug("Collage build failed: %s", exc)
                    face_jpeg = scene_jpeg or natural
            elif style == "both" and scene_jpeg is not None:
                frame_jpeg = scene_jpeg
                face_jpeg = natural
        ok = send_watchlist_alert(
            face_jpeg=face_jpeg,
            frame_jpeg=frame_jpeg,
            camera_name=cam_label,
            location=loc_label,
            person_name=display_name,
            confidence=conf,
            timestamp=timestamp,
            person_id=int(person_id),
            alert_kind=alert_kind,
            image_quality=quality_label or None,
        )
        if ok:
            print(
                f"[Telegram] sent {display_name} (#{person_id}) @ {cam_label} "
                f"{conf}% [{alert_kind}]"
            )
            log.info(
                "Telegram sent — %s (#%s) @ %s · %s%% [%s]",
                display_name,
                person_id,
                cam_label,
                conf,
                alert_kind,
            )
        else:
            # Allow a quick retry if the network send failed.
            _release_cooldown(cam_key, int(person_id))
            print(f"[Telegram] send failed — {display_name} (#{person_id}) @ {cam_label}")
            log.warning(
                "Telegram send failed — %s (#%s) @ %s",
                display_name,
                person_id,
                cam_label,
            )

    _executor.submit(_worker)
    return True


def maybe_send_multi_face_alert_async(
    *,
    camera_zone: str,
    frame_bgr: Optional[np.ndarray],
    faces: Sequence[dict],
    camera_name: Optional[str] = None,
    location: Optional[str] = None,
    cooldown_sec: Optional[float] = None,
    min_percent: Optional[int] = None,
) -> bool:
    """
    One Telegram collage when several people are caught in the same frame.

    ``faces`` items: person_id, person_name, match_percent, face_bgr,
    optional bbox (xyxy on ``frame_bgr``).
    """
    if not faces:
        return False
    try:
        from announcement_settings import announcements_allowed_now

        if not announcements_allowed_now():
            log.info(
                "Telegram skip — outside announcement window (multi-catch @ %s)",
                camera_zone,
            )
            return False
    except Exception:
        pass
    if not telegram_configured():
        log.warning(
            "Telegram skip — TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in this process"
        )
        return False

    gate = int(min_percent if min_percent is not None else TELEGRAM_ALERT_MIN_PERCENT)
    cool = float(cooldown_sec) if cooldown_sec is not None else _cooldown_seconds()
    cam_key = str(camera_zone or camera_name or "cam")
    cam_label = (camera_name or "").strip() or str(camera_zone)
    loc_label = (location or "").strip() or cam_label

    eligible: list[dict] = []
    for item in faces:
        try:
            pid = int(item.get("person_id"))
            percent = int(item.get("match_percent", 0) or 0)
        except (TypeError, ValueError):
            continue
        if percent < gate:
            continue
        face = item.get("face_bgr")
        if face is None or getattr(face, "size", 0) == 0:
            continue
        eligible.append(item)

    if not eligible:
        return False

    # Single person → normal alert path (handles its own cooldown).
    if len(eligible) == 1:
        item = eligible[0]
        return maybe_send_target_alert_async(
            item["face_bgr"],
            int(item["person_id"]),
            cam_key,
            int(item.get("match_percent", 0) or 0),
            frame_bgr=frame_bgr,
            person_name=item.get("person_name"),
            camera_name=cam_label,
            location=loc_label,
            bbox=item.get("bbox"),
            cooldown_sec=cooldown_sec,
            min_percent=min_percent,
        )

    reserved: list[Tuple[str, int]] = []
    kept: list[dict] = []
    for item in eligible:
        pid = int(item["person_id"])
        if not _allow_send(cam_key, pid, cool):
            continue
        kept.append(item)
        reserved.append((cam_key, pid))
    if not kept:
        return False
    if len(kept) == 1:
        item = kept[0]
        # Cooldown already reserved — send with cool=0 so we do not double-block.
        return maybe_send_target_alert_async(
            item["face_bgr"],
            int(item["person_id"]),
            cam_key,
            int(item.get("match_percent", 0) or 0),
            frame_bgr=frame_bgr,
            person_name=item.get("person_name"),
            camera_name=cam_label,
            location=loc_label,
            bbox=item.get("bbox"),
            cooldown_sec=0.0,
            min_percent=min_percent,
        )

    eligible = kept
    scene = None if frame_bgr is None else np.ascontiguousarray(frame_bgr)
    face_panels = []
    seen_person_ids: set[int] = set()
    for item in eligible:
        try:
            pid = int(item.get("person_id"))
        except (TypeError, ValueError):
            pid = None
        if pid is not None and pid in seen_person_ids:
            continue
        panel = _telegram_person_panel(
            np.ascontiguousarray(item["face_bgr"]),
            frame_bgr=scene,
            bbox=item.get("bbox"),
            person_bbox=item.get("person_bbox"),
        )
        ok, why = _telegram_panel_worth_sending(panel)
        if not ok:
            # Prefer a large face crop over dropping the whole multi-catch.
            face_ok, _ = _telegram_panel_worth_sending(
                np.ascontiguousarray(item["face_bgr"])
            )
            if face_ok:
                panel = np.ascontiguousarray(item["face_bgr"])
            else:
                log.info(
                    "Telegram multi-catch skip panel person #%s (%s)",
                    item.get("person_id"),
                    why,
                )
                continue
        if any(_panels_look_identical(panel, kept) for kept in face_panels):
            continue
        face_panels.append(panel)
        if pid is not None:
            seen_person_ids.add(pid)
    if not face_panels:
        for key_cam, pid in reserved:
            _release_cooldown(key_cam, pid)
        return False
    caption_bits: list[str] = []
    for item in eligible:
        name = html.escape(str(item.get("person_name") or f"Person #{item.get('person_id')}"))
        percent = int(item.get("match_percent", 0) or 0)
        caption_bits.append(f"#{int(item['person_id'])} {name} {percent}%")

    def _worker() -> None:
        token, chat_id = _get_config()
        if not token or not chat_id:
            for key_cam, pid in reserved:
                _release_cooldown(key_cam, pid)
            return
        try:
            encoded: Optional[bytes] = None
            if scene is not None:
                style = _telegram_alert_style()
                if style == "scene":
                    encoded = _encode_scene_for_alert(scene)
                elif style == "collage":
                    collage = compose_alert_collage_multi(
                        scene, face_panels, enhance_face=_telegram_enhance_display()
                    )
                    encoded = encode_bgr_jpeg(collage, quality=90, max_side=1600)
                else:
                    encoded = _encode_scene_for_alert(scene)
            elif len(face_panels) == 1:
                encoded = encode_bgr_jpeg(face_panels[0], quality=90, max_side=640)
            elif face_panels:
                collage = compose_alert_collage_multi(
                    face_panels[0], face_panels[1:], enhance_face=_telegram_enhance_display()
                )
                encoded = encode_bgr_jpeg(collage, quality=90, max_side=1600)
            if not encoded:
                for key_cam, pid in reserved:
                    _release_cooldown(key_cam, pid)
                return
            when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            n = len(eligible)
            caption = (
                f"<b>LIVE AI INFRACTION</b>\n"
                f"Face detected — {n} people\n"
                f"<b>Camera:</b> {html.escape(cam_label)}\n"
                f"<b>Location:</b> {html.escape(loc_label)}\n"
                f"{' · '.join(caption_bits)}\n"
                f"<b>Time:</b> {html.escape(when)}"
            )
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data={
                    "chat_id": chat_id,
                    "caption": caption,
                    "parse_mode": "HTML",
                },
                files={"photo": ("multi_catch.jpg", encoded, "image/jpeg")},
                timeout=25,
            )
            if response.status_code >= 400:
                for key_cam, pid in reserved:
                    _release_cooldown(key_cam, pid)
                log.warning(
                    "Telegram multi-catch HTTP %s: %s",
                    response.status_code,
                    response.text[:300],
                )
                return
            log.info(
                "Telegram multi-catch sent — ×%s @ %s",
                n,
                cam_label,
            )
        except Exception as exc:
            for key_cam, pid in reserved:
                _release_cooldown(key_cam, pid)
            log.warning("Telegram multi-catch failed: %s", exc)

    _executor.submit(_worker)
    return True


# Friendly alias matching the product wording.
maybe_send_poi_alert_async = maybe_send_target_alert_async


def _load_bgr(path: Path) -> Optional[np.ndarray]:
    try:
        if not path.is_file():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return None
        return img
    except Exception:
        return None


def _pick_demo_scene_frame() -> Optional[np.ndarray]:
    """Prefer a fresh live camera JPEG; fall back to any available preview."""
    cameras_dir = Path(__file__).resolve().parent / "live_preview" / "cameras"
    preferred = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]
    now = time.time()
    for name in preferred:
        path = cameras_dir / name
        try:
            if path.is_file() and (now - path.stat().st_mtime) < 120.0:
                frame = _load_bgr(path)
                if frame is not None:
                    return frame
        except OSError:
            continue
    # Any recent numeric cam file
    try:
        candidates = sorted(
            cameras_dir.glob("[0-9]*.jpg"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in candidates[:8]:
            frame = _load_bgr(path)
            if frame is not None:
                return frame
    except OSError:
        pass
    return None


def _pick_demo_face_bgr(
    *,
    face_path: Optional[Union[str, Path]] = None,
    scene: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[Sequence[float]]]:
    """
    Return (face_crop, bbox_on_scene).

    Prefers an enrolled/upload face photo. When only a scene exists, use a
    centered crop as a visual stand-in (still a real camera image, not text).
    """
    if face_path:
        face = _load_bgr(Path(face_path))
        if face is not None:
            return face, None

    project = Path(__file__).resolve().parent
    for candidate in (
        project / "static" / "uploads" / "target.jpg",
        project / "uploads" / "target.jpg",
    ):
        face = _load_bgr(candidate)
        if face is not None:
            return face, None

    if scene is not None and scene.size:
        h, w = scene.shape[:2]
        side = int(min(h, w) * 0.28)
        side = max(80, min(side, min(h, w) - 4))
        cx, cy = w // 2, int(h * 0.42)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        crop = scene[y1:y2, x1:x2].copy()
        return crop, (float(x1), float(y1), float(x2), float(y2))
    return None, None


def build_demo_watchlist_images(
    *,
    face_path: Optional[Union[str, Path]] = None,
    person_name: str = "Watchlist Subject",
    confidence: int = 91,
    camera_label: str = "CAM-01",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Sequence[float]]]:
    """
    Build a realistic Telegram demo: live CCTV scene + face crop.

    Never draws placeholder 'DEMO CCTV FRAME' text — uses real camera pixels.
    Returns (face_bgr, scene_bgr, bbox).
    """
    scene = _pick_demo_scene_frame()
    face, bbox = _pick_demo_face_bgr(face_path=face_path, scene=scene)

    if scene is None and face is not None:
        # Compose a simple scene canvas from the face so Telegram still gets
        # an annotated "CCTV" photo, not a text stub.
        fh, fw = face.shape[:2]
        canvas_w, canvas_h = 960, 540
        scene = np.full((canvas_h, canvas_w, 3), (28, 24, 22), dtype=np.uint8)
        # Subtle vignette / floor gradient so it does not look like a blank plate.
        for y in range(canvas_h):
            shade = int(18 + (y / max(1, canvas_h - 1)) * 22)
            scene[y, :] = (shade + 6, shade, shade - 2)
        scale = min((canvas_w * 0.34) / max(1, fw), (canvas_h * 0.48) / max(1, fh))
        nw = max(1, int(round(fw * scale)))
        nh = max(1, int(round(fh * scale)))
        resized = cv2.resize(face, (nw, nh), interpolation=cv2.INTER_AREA)
        x1 = (canvas_w - nw) // 2
        y1 = int(canvas_h * 0.28)
        x2, y2 = x1 + nw, y1 + nh
        scene[y1:y2, x1:x2] = resized
        bbox = (float(x1), float(y1), float(x2), float(y2))

    if scene is not None and bbox is None and face is not None:
        # Paste enrolled face onto the live scene so demos still show a subject.
        h, w = scene.shape[:2]
        fh, fw = face.shape[:2]
        box_w = int(min(w, h) * 0.26)
        box_h = int(box_w * (fh / max(1, fw)))
        box_w = max(90, min(box_w, w - 20))
        box_h = max(90, min(box_h, h - 20))
        pasted = cv2.resize(face, (box_w, box_h), interpolation=cv2.INTER_AREA)
        x1 = (w - box_w) // 2
        y1 = max(10, int(h * 0.30))
        x2, y2 = x1 + box_w, y1 + box_h
        if y2 <= h and x2 <= w:
            scene = scene.copy()
            scene[y1:y2, x1:x2] = pasted
            bbox = (float(x1), float(y1), float(x2), float(y2))

    # Raw scene + face; collage is built at send time (sample Telegram style).
    _ = (person_name, confidence, camera_label)
    return face, scene, bbox


def send_demo_watchlist_alert(
    *,
    face_path: Optional[Union[str, Path]] = None,
    person_name: str = "Watchlist Subject",
    person_id: Optional[int] = None,
    confidence: int = 91,
    camera_name: str = "CAM-01",
    location: str = "Main Gate",
) -> bool:
    """
    Send a realistic sample watchlist Telegram alert (real camera/face pixels).
    """
    if not telegram_configured():
        log.warning("Demo Telegram skipped — bot not configured")
        return False

    face_bgr, scene_bgr, _bbox = build_demo_watchlist_images(
        face_path=face_path,
        person_name=person_name,
        confidence=confidence,
        camera_label=camera_name,
    )
    if face_bgr is None and scene_bgr is None:
        log.warning("Demo Telegram skipped — no live frame or face image available")
        return False

    if scene_bgr is not None and face_bgr is not None:
        collage = compose_alert_collage(scene_bgr, face_bgr)
        face_jpeg = encode_bgr_jpeg(collage, quality=86, max_side=1600)
        frame_jpeg = None
    else:
        face_jpeg = encode_bgr_jpeg(
            face_bgr if face_bgr is not None else scene_bgr, quality=90, max_side=640
        )
        frame_jpeg = None
    if not face_jpeg:
        return False

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return send_watchlist_alert(
        face_jpeg=face_jpeg,
        frame_jpeg=frame_jpeg,
        camera_name=camera_name,
        location=location,
        person_name=person_name,
        confidence=int(confidence),
        timestamp=timestamp,
        person_id=person_id,
    )
