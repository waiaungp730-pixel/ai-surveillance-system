"""
Save cropped person images when visits are logged.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]

PROJECT_DIR = Path(__file__).parent
CAPTURES_DIR = PROJECT_DIR / "captures"

# Display / save target — larger than before so Telegram face panel is readable.
CAPTURE_MIN_SIDE = 320
CAPTURE_MAX_UPSCALE = 3.0
CAPTURE_JPEG_QUALITY = 95
CAPTURE_FACE_PAD = float(os.environ.get("CAPTURE_FACE_PAD", "0.0"))
# Display crop: ``person_box`` = live detect rectangle (default); ``upper_body`` = head→waist; ``face`` / ``full_frame``.
CAPTURE_DISPLAY_MODE = os.environ.get("CAPTURE_DISPLAY_MODE", "person_box").strip().lower()
CAPTURE_PERSON_BOX_PAD_RATIO = float(
    os.environ.get("CAPTURE_PERSON_BOX_PAD_RATIO", "0.02")
)
SCENE_MAX_SIDE = int(os.environ.get("CAPTURE_SCENE_MAX_SIDE", "1920"))
CAPTURE_OUTPUT_SIDE = int(os.environ.get("CAPTURE_OUTPUT_SIDE", "256"))
CAPTURE_UPPER_BODY_BELOW_FACE_RATIO = float(os.environ.get("CAPTURE_UPPER_BODY_BELOW_FACE_RATIO", "2.2"))
CAPTURE_UPPER_BODY_SIDE_PAD_RATIO = float(os.environ.get("CAPTURE_UPPER_BODY_SIDE_PAD_RATIO", "0.85"))
CAPTURE_UPPER_BODY_TOP_PAD_RATIO = float(os.environ.get("CAPTURE_UPPER_BODY_TOP_PAD_RATIO", "0.40"))
# Waist line as fraction of YOLO person height (head top → waist).
CAPTURE_UPPER_BODY_WAIST_RATIO = float(os.environ.get("CAPTURE_UPPER_BODY_WAIST_RATIO", "0.52"))
# When a YOLO person box exists, expand the crop toward the waist — never clip the head.
CAPTURE_UPPER_BODY_USE_PERSON_BOX = os.environ.get(
    "CAPTURE_UPPER_BODY_USE_PERSON_BOX", "1"
).strip().lower() in ("1", "true", "yes", "on")
# Never burn live overlays into saved JPEGs unless explicitly enabled.
CAPTURE_DISPLAY_DRAW_FACE_BOX = os.environ.get(
    "CAPTURE_DISPLAY_DRAW_FACE_BOX", "0"
).strip().lower() in ("1", "true", "yes", "on")
# Refuse saves when the native crop is still too small to look photographic.
CAPTURE_NATIVE_MIN_SIDE = int(os.environ.get("CAPTURE_NATIVE_MIN_SIDE", "24"))
# GFPGAN display enhancement when native face is smaller than this (px).
CAPTURE_ENHANCE_BELOW_SIDE = 160
# Telegram / preview panel — upscale soft crops before letterboxing (display-only).
TELEGRAM_DISPLAY_MIN_SIDE = int(os.environ.get("TELEGRAM_DISPLAY_MIN_SIDE", "400"))
PERSON_CROP_PAD_RATIO = 0.08


def _display_mode() -> str:
    return (os.environ.get("CAPTURE_DISPLAY_MODE") or "person_box").strip().lower()


def capture_display_is_full_frame() -> bool:
    """True when galleries save/show the entire CCTV frame instead of a person crop."""
    return _display_mode() in ("full_frame", "fullframe", "scene", "frame", "full")


def capture_display_is_person_box() -> bool:
    """True when saves crop exactly to the live YOLO/detect rectangle."""
    if capture_display_is_full_frame():
        return False
    return _display_mode() in (
        "person_box",
        "personbox",
        "bbox",
        "box",
        "detect_box",
        "detection_box",
        "yolo_box",
    )


def capture_display_is_body_crop() -> bool:
    """True when saves/gallery use head-and-body crops instead of tight face squares."""
    if capture_display_is_full_frame() or capture_display_is_person_box():
        return False
    return _display_mode() in (
        "upper_body",
        "upperbody",
        "head_shoulders",
        "torso",
        "full_body",
        "body",
    )


def display_image_is_portrait_body(img: Optional[np.ndarray]) -> bool:
    """True when an image is a tall person cut, not a tight face square."""
    if img is None or getattr(img, "size", 0) == 0 or img.ndim < 2:
        return False
    h, w = int(img.shape[0]), int(img.shape[1])
    if min(h, w) < 24:
        return False
    return h / float(max(1, w)) >= 1.18


def ensure_capture_column(conn) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    if "capture_path" not in cols:
        conn.execute("ALTER TABLE tracking_logs ADD COLUMN capture_path TEXT")
        conn.commit()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    if "scene_path" not in cols:
        conn.execute("ALTER TABLE tracking_logs ADD COLUMN scene_path TEXT")
        conn.commit()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    if "display_name" not in cols:
        conn.execute("ALTER TABLE tracking_logs ADD COLUMN display_name TEXT")
        conn.commit()


_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_CAPTURE_NAME_RE = re.compile(
    r"^(?P<ts>\d{8}_\d{6})_(?P<kind>face|scene)_(?P<rest>.+)\.(?P<ext>jpe?g|png|webp)$",
    re.IGNORECASE,
)


def capture_filename_slug(person_name: Optional[str], fallback_id: object) -> str:
    """Filesystem-safe token for a capture JPEG. Auto Stranger_/Visitor_ labels stay as the id."""
    fallback = str(fallback_id if fallback_id is not None else "unknown")
    raw = str(person_name or "").strip()
    if not raw:
        return fallback
    try:
        from identity import is_auto_identity_name

        if is_auto_identity_name(raw):
            return fallback
    except Exception:
        pass
    slug = _UNSAFE_FILENAME.sub("", raw)
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "", slug).strip("._-")
    return slug[:40] if slug else fallback


def build_capture_filename(
    kind: str,
    *,
    timestamp: Optional[str] = None,
    person_name: Optional[str] = None,
    person_id: object = "unknown",
) -> str:
    """``20260819_103000_face_Jane_2838.jpg`` when a real name exists."""
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = person_id if person_id is not None else "unknown"
    slug = capture_filename_slug(person_name, pid)
    token = kind if kind in ("face", "scene") else "face"
    if slug and str(slug) != str(pid):
        return f"{ts}_{token}_{slug}_{pid}.jpg"
    return f"{ts}_{token}_{pid}.jpg"


def apply_person_slug_to_capture_filename(
    filename: Optional[str],
    person_name: Optional[str],
) -> Optional[str]:
    """Rewrite a capture JPEG name so it includes the person slug, keeping timestamp + id."""
    if not filename:
        return filename
    name = Path(str(filename).replace("\\", "/")).name
    match = _CAPTURE_NAME_RE.match(name)
    if not match:
        return name
    rest = match.group("rest")
    pid = rest
    if "_" in rest:
        maybe_pid = rest.rsplit("_", 1)[-1]
        if maybe_pid.isdigit() or maybe_pid.lower().startswith("track"):
            pid = maybe_pid
    return build_capture_filename(
        match.group("kind").lower(),
        timestamp=match.group("ts"),
        person_name=person_name,
        person_id=pid,
    )


def rename_capture_file_with_name(
    relative_or_name: Optional[str],
    person_name: Optional[str],
) -> Optional[str]:
    """Rename a JPEG under captures/ to include the person name. Returns the new relative name."""
    if not relative_or_name:
        return relative_or_name
    raw = str(relative_or_name).replace("\\", "/").lstrip("/")
    if raw.startswith("captures/"):
        raw = raw[len("captures/") :]
    src = CAPTURES_DIR / Path(raw).name
    new_name = apply_person_slug_to_capture_filename(src.name, person_name)
    if not new_name or new_name == src.name:
        return Path(raw).name
    dest = CAPTURES_DIR / new_name
    if dest.resolve() == src.resolve():
        return src.name
    if not src.is_file():
        return Path(raw).name
    if dest.is_file():
        return dest.name
    try:
        src.rename(dest)
        return dest.name
    except OSError:
        return src.name


def rename_log_captures_with_name(
    conn,
    person_name: str,
    *,
    log_id: Optional[int] = None,
    known_person_id: Optional[int] = None,
) -> None:
    """Rename saved face/scene JPEGs for a log or person so the files carry the name."""
    ensure_capture_column(conn)
    clauses = []
    params = []
    if log_id is not None:
        clauses.append("id = ?")
        params.append(int(log_id))
    if known_person_id is not None:
        clauses.append("known_person_id = ?")
        params.append(int(known_person_id))
    if not clauses:
        return
    rows = conn.execute(
        f"""
        SELECT id, capture_path, scene_path
        FROM tracking_logs
        WHERE {" OR ".join(clauses)}
        """,
        params,
    ).fetchall()
    for row in rows:
        row_id = int(row["id"] if hasattr(row, "keys") else row[0])
        cap = row["capture_path"] if hasattr(row, "keys") else row[1]
        scene = row["scene_path"] if hasattr(row, "keys") else row[2]
        new_cap = rename_capture_file_with_name(cap, person_name)
        new_scene = rename_capture_file_with_name(scene, person_name)
        if new_cap != cap or new_scene != scene:
            conn.execute(
                "UPDATE tracking_logs SET capture_path = ?, scene_path = ? WHERE id = ?",
                (new_cap, new_scene, row_id),
            )


def upscale_face_minimum(
    face_bgr: np.ndarray,
    min_side: int = CAPTURE_MIN_SIDE,
    *,
    max_scale: float = CAPTURE_MAX_UPSCALE,
) -> np.ndarray:
    """
    Mild upscale for small crops only.

    Caps scale so a 20px CCTV head is never blown to 256px (that creates the
    flat/cartoon look). Prefer waiting for a larger native face instead.
    """
    if face_bgr is None or face_bgr.size == 0:
        return face_bgr
    h, w = face_bgr.shape[:2]
    side = float(max(h, w))
    if side <= 0 or side >= float(min_side):
        return face_bgr
    scale = min(float(min_side) / side, float(max_scale))
    if scale <= 1.01:
        return face_bgr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # LANCZOS4 for upscaling small CCTV crops (clearer than LINEAR/CUBIC).
    interp = cv2.INTER_LANCZOS4 if scale > 1.05 else cv2.INTER_AREA
    return cv2.resize(face_bgr, (new_w, new_h), interpolation=interp)


def _unsharp_mask(
    face_bgr: np.ndarray,
    *,
    amount: float = 0.65,
    sigma: float = 1.0,
) -> np.ndarray:
    """Light sharpen after upscale — helps soft CCTV JPEGs without GFPGAN."""
    if face_bgr is None or face_bgr.size == 0:
        return face_bgr
    blur = cv2.GaussianBlur(face_bgr, (0, 0), sigma)
    return cv2.addWeighted(face_bgr, 1.0 + amount, blur, -amount, 0)


def enhance_capture_for_display(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Clearer Telegram / preview image without changing saved capture JPEGs.

    Upper-body crops get GFPGAN on the head band when small; face squares use
    the full face path. Scene frames should not be passed here.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    out = np.ascontiguousarray(crop_bgr)
    h, w = out.shape[:2]
    aspect = h / float(max(1, w))
    is_body = aspect >= 1.15

    if is_body:
        face_h = max(1, int(round(h * 0.38)))
        face_band = out[0:face_h]
        fh, fw = int(face_band.shape[0]), int(face_band.shape[1])
        if min(fh, fw) < int(CAPTURE_ENHANCE_BELOW_SIDE):
            try:
                from face_restoration import restore_for_scan

                enhanced, restored = restore_for_scan(face_band, force=True)
                if restored and enhanced is not None and enhanced.size > 0:
                    if enhanced.shape[:2] != (fh, fw):
                        enhanced = cv2.resize(
                            enhanced,
                            (fw, fh),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                    out[0:face_h] = enhanced
            except Exception:
                pass
        try:
            from face_analyzer import apply_face_clahe

            out = apply_face_clahe(out)
        except Exception:
            pass
        min_side = max(int(CAPTURE_MIN_SIDE), int(TELEGRAM_DISPLAY_MIN_SIDE))
        out = upscale_face_minimum(
            out,
            min_side=min_side,
            max_scale=CAPTURE_MAX_UPSCALE,
        )
        out = _unsharp_mask(out, amount=0.55, sigma=1.15)
        return out

    return enhance_face_for_display(out)


def enhance_face_for_display(face_bgr: np.ndarray) -> np.ndarray:
    """
    Clearer face for Telegram collages and capture JPEGs.

    Small distant CCTV crops get GFPGAN (when weights are present), then CLAHE
    + mild sharpen + capped upscale. Embedding still uses the natural crop in
    main.py — this is display-only.
    """
    if face_bgr is None or face_bgr.size == 0:
        return face_bgr
    out = np.ascontiguousarray(face_bgr)
    h, w = out.shape[:2]
    native_side = min(h, w)
    restored = False
    if native_side < int(CAPTURE_ENHANCE_BELOW_SIDE):
        try:
            from face_restoration import restore_for_scan

            enhanced, restored = restore_for_scan(out, force=True)
            if enhanced is not None and enhanced.size > 0:
                out = np.ascontiguousarray(enhanced)
        except Exception:
            pass
    if not restored:
        try:
            from face_analyzer import apply_face_clahe

            out = apply_face_clahe(out)
        except Exception:
            pass
    out = upscale_face_minimum(
        out,
        min_side=CAPTURE_MIN_SIDE,
        max_scale=CAPTURE_MAX_UPSCALE,
    )
    out = _unsharp_mask(out)
    return out


def crop_face_from_frame(
    frame: np.ndarray,
    bbox: Union[BBox, Tuple[int, int, int, int], np.ndarray],
    *,
    pad_ratio: float = CAPTURE_FACE_PAD,
) -> Optional[np.ndarray]:
    """
    Crop the detection box itself (hairline → chin), matching the live red square.

    pad_ratio=0 keeps the detector box. Larger values expand into neck/shoulders.
    """
    if frame is None or frame.size == 0:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in np.asarray(bbox).reshape(-1)[:4])
    except (TypeError, ValueError, IndexError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    fw, fh = x2 - x1, y2 - y1
    if fw < 6 or fh < 6:
        return None
    pad = max(0.0, float(pad_ratio))
    # Square covering the face box only — do not grow by (0.5 + pad) around center.
    side = max(fw, fh) * (1.0 + 2.0 * pad)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half = 0.5 * side
    img_h, img_w = frame.shape[:2]
    nx1 = int(max(0, np.floor(cx - half)))
    ny1 = int(max(0, np.floor(cy - half)))
    nx2 = int(min(img_w, np.ceil(cx + half)))
    ny2 = int(min(img_h, np.ceil(cy + half)))
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return None
    crop = frame[ny1:ny2, nx1:nx2]
    if crop is None or crop.size == 0:
        return None
    return np.ascontiguousarray(crop.copy())


def crop_upper_body_from_frame(
    frame: np.ndarray,
    face_bbox: Union[BBox, Tuple[int, int, int, int], np.ndarray],
    person_bbox: Optional[Union[BBox, Tuple[int, int, int, int], np.ndarray]] = None,
    *,
    below_face_ratio: float = CAPTURE_UPPER_BODY_BELOW_FACE_RATIO,
    side_pad_ratio: float = CAPTURE_UPPER_BODY_SIDE_PAD_RATIO,
    top_pad_ratio: float = CAPTURE_UPPER_BODY_TOP_PAD_RATIO,
    draw_face_box: Optional[bool] = None,
) -> Optional[np.ndarray]:
    """
    Head-to-waist crop anchored on the face box (portrait, not a tight face square).

    Recognition still uses the tight face crop; this is the gallery / preview image.
    """
    if frame is None or frame.size == 0:
        return None
    try:
        fx1, fy1, fx2, fy2 = (float(v) for v in np.asarray(face_bbox).reshape(-1)[:4])
    except (TypeError, ValueError, IndexError):
        return None
    if fx2 <= fx1 or fy2 <= fy1:
        return None
    fh, fw = fy2 - fy1, fx2 - fx1
    if fh < 8 or fw < 8:
        return None
    cx = 0.5 * (fx1 + fx2)
    img_h, img_w = frame.shape[:2]
    # Hairline with a little room above; chin → waist (~2 face-heights below).
    top = fy1 - top_pad_ratio * fh
    bottom = fy2 + below_face_ratio * fh
    half_w = (0.5 + side_pad_ratio) * fw
    left = cx - half_w
    right = cx + half_w
    waist_cap: Optional[float] = None
    if person_bbox is not None:
        try:
            px1, py1, px2, py2 = (
                float(v) for v in np.asarray(person_bbox).reshape(-1)[:4]
            )
            if px2 > px1 and py2 > py1:
                ph = py2 - py1
                person_w = px2 - px1
                top = min(top, py1 - 0.02 * ph)
                # Distant subjects: widen crop toward the YOLO person box.
                half_w = max(half_w, 0.42 * person_w)
                if CAPTURE_UPPER_BODY_USE_PERSON_BOX:
                    waist_cap = py1 + CAPTURE_UPPER_BODY_WAIST_RATIO * ph
        except (TypeError, ValueError, IndexError):
            pass
    # Head-to-waist only — never extend to knees/feet.
    min_bottom = fy2 + 1.6 * fh
    max_bottom = fy2 + max(2.0, below_face_ratio + 0.35) * fh
    bottom = float(np.clip(bottom, min_bottom, max_bottom))
    if waist_cap is not None:
        bottom = min(bottom, waist_cap)
    top = min(top, fy1 - 0.15 * fh)
    left = min(left, fx1 - 0.35 * fw)
    right = max(right, fx2 + 0.35 * fw)
    # Avoid needle-thin strips when the face box is tiny (distant subjects).
    min_crop_w = max(72.0, 2.6 * fw)
    crop_w = right - left
    if crop_w < min_crop_w:
        left = cx - 0.5 * min_crop_w
        right = cx + 0.5 * min_crop_w
    # Keep a portrait plate (head→waist), not a wide hallway strip.
    crop_h = max(1.0, bottom - top)
    max_w = crop_h / 1.45
    crop_w = right - left
    if crop_w > max_w:
        left = cx - 0.5 * max_w
        right = cx + 0.5 * max_w
    sx1 = int(max(0, np.floor(left)))
    sy1 = int(max(0, np.floor(top)))
    sx2 = int(min(img_w, np.ceil(right)))
    sy2 = int(min(img_h, np.ceil(bottom)))
    if sx2 - sx1 < 24 or sy2 - sy1 < 40:
        return None
    crop = frame[sy1:sy2, sx1:sx2]
    if crop is None or crop.size == 0:
        return None
    out = np.ascontiguousarray(crop.copy())
    should_draw = CAPTURE_DISPLAY_DRAW_FACE_BOX if draw_face_box is None else bool(draw_face_box)
    if should_draw:
        bx1 = int(round(fx1 - sx1))
        by1 = int(round(fy1 - sy1))
        bx2 = int(round(fx2 - sx1))
        by2 = int(round(fy2 - sy1))
        pad = max(2, int(round(0.08 * max(bx2 - bx1, by2 - by1))))
        ch, cw = out.shape[:2]
        bx1 = max(0, bx1 - pad)
        by1 = max(0, by1 - pad)
        bx2 = min(cw - 1, bx2 + pad)
        by2 = min(ch - 1, by2 + pad)
        if bx2 - bx1 >= 8 and by2 - by1 >= 8:
            cv2.rectangle(out, (bx1, by1), (bx2, by2), (40, 40, 220), 2, cv2.LINE_AA)
    return out


def display_body_crop_is_valid(crop: np.ndarray) -> bool:
    """
    Reject arm/torso-only panels when upper-body display mode is active.

    Real head-and-body crops are portrait with face-like content in the top band.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return False
    h, w = int(crop.shape[0]), int(crop.shape[1])
    if min(h, w) < 24:
        return False
    if w < 72:
        return False
    if h / float(max(1, w)) > 2.0:
        return False
    if not display_image_is_portrait_body(crop):
        return False
    try:
        from face_print_filter import (
            _horizontal_skin_bands,
            capture_crop_passes_human_gate,
            is_likely_invalid_face_crop,
            is_likely_rear_view,
            is_likely_streaked_or_corrupt_capture,
        )

        streak, _ = is_likely_streaked_or_corrupt_capture(crop)
        if streak:
            return False

        human_ok, _ = capture_crop_passes_human_gate(crop)
        if not human_ok:
            return False

        top_third = crop[0 : max(1, h // 3)]
        rear, _ = is_likely_rear_view(crop)
        if rear:
            return False
        rear_top, _ = is_likely_rear_view(top_third)
        if rear_top:
            return False
        bad_top, _ = is_likely_invalid_face_crop(top_third)
        if bad_top:
            return False
        upper, mid, lower = _horizontal_skin_bands(crop)
        if upper < 0.04 and mid < 0.05:
            return False
        if lower > upper + 0.18 and upper < 0.07:
            return False
    except Exception:
        pass
    return True


def _resized_full_frame(frame: np.ndarray, max_side: int = SCENE_MAX_SIDE) -> np.ndarray:
    """Copy the full CCTV frame, downscaling only if wider/taller than max_side."""
    scene = np.ascontiguousarray(frame)
    h, w = int(scene.shape[0]), int(scene.shape[1])
    side = max(h, w)
    limit = int(max_side)
    if side > limit > 0:
        scale = float(limit) / float(side)
        scene = cv2.resize(
            scene,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return scene


def choose_display_capture_crop(
    frame: Optional[np.ndarray],
    face_bbox: Optional[Union[BBox, Tuple[int, int, int, int], np.ndarray]],
    aligned_face: np.ndarray,
    person_bbox: Optional[Union[BBox, Tuple[int, int, int, int], np.ndarray]] = None,
) -> np.ndarray:
    """
    Image saved/shown in galleries and alerts.

    ``person_box`` (default): crop to the live detect rectangle (YOLO person /
    blue frame on CCTV). ``upper_body`` = head-to-waist portrait. ``face`` =
    tight face square.
    """
    if (
        capture_display_is_full_frame()
        and frame is not None
        and getattr(frame, "size", 0) > 0
    ):
        return _resized_full_frame(frame)

    if capture_display_is_person_box() and frame is not None and getattr(frame, "size", 0) > 0:
        pad = float(CAPTURE_PERSON_BOX_PAD_RATIO)
        # Prefer the person detect box drawn on the live stream.
        if person_bbox is not None:
            box_crop = crop_person_from_frame(frame, person_bbox, pad_ratio=pad)
            if box_crop is not None and getattr(box_crop, "size", 0) > 0:
                return np.ascontiguousarray(box_crop)
        if face_bbox is not None:
            # Face-only detect: still crop the rectangle, with a little pad.
            face_crop = crop_face_from_frame(frame, face_bbox, pad_ratio=max(0.08, pad))
            if face_crop is not None and getattr(face_crop, "size", 0) > 0:
                return np.ascontiguousarray(face_crop)
        return np.ascontiguousarray(aligned_face)

    if capture_display_is_body_crop():
        if (
            frame is not None
            and getattr(frame, "size", 0) > 0
            and face_bbox is not None
        ):
            upper = crop_upper_body_from_frame(
                frame, face_bbox, person_bbox, draw_face_box=False
            )
            if upper is None or upper.size == 0 or not display_image_is_portrait_body(upper):
                upper = crop_upper_body_from_frame(
                    frame,
                    face_bbox,
                    None,
                    draw_face_box=False,
                    below_face_ratio=CAPTURE_UPPER_BODY_BELOW_FACE_RATIO,
                )
            if upper is not None and upper.size > 0 and display_body_crop_is_valid(upper):
                return np.ascontiguousarray(upper)
        return np.ascontiguousarray(aligned_face)
    if frame is not None and getattr(frame, "size", 0) > 0 and face_bbox is not None:
        tight = crop_face_from_frame(frame, face_bbox, pad_ratio=0.0)
        if tight is not None and tight.size > 0:
            return tight
    return np.ascontiguousarray(aligned_face)


def crop_person_from_frame(
    frame: np.ndarray,
    person_bbox: Union[BBox, Tuple[int, int, int, int], np.ndarray],
    *,
    pad_ratio: float = 0.04,
) -> Optional[np.ndarray]:
    """YOLO person box crop — fallback when head-to-waist plate cannot be built."""
    if frame is None or frame.size == 0:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in np.asarray(person_bbox).reshape(-1)[:4])
    except (TypeError, ValueError, IndexError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    pw, ph = x2 - x1, y2 - y1
    if pw < 12 or ph < 20:
        return None
    pad = max(0.0, float(pad_ratio))
    img_h, img_w = frame.shape[:2]
    nx1 = int(max(0, np.floor(x1 - pad * pw)))
    ny1 = int(max(0, np.floor(y1 - pad * ph)))
    nx2 = int(min(img_w, np.ceil(x2 + pad * pw)))
    ny2 = int(min(img_h, np.ceil(y2 + pad * ph)))
    if nx2 - nx1 < 16 or ny2 - ny1 < 24:
        return None
    crop = frame[ny1:ny2, nx1:nx2]
    if crop is None or crop.size == 0:
        return None
    return np.ascontiguousarray(crop)


def fallback_display_capture_crop(
    frame: Optional[np.ndarray],
    face_bbox: Optional[Union[BBox, Tuple[int, int, int, int], np.ndarray]],
    aligned_face: np.ndarray,
    person_bbox: Optional[Union[BBox, Tuple[int, int, int, int], np.ndarray]] = None,
) -> np.ndarray:
    """
    When upper-body display mode fails validation, still return a saveable crop.

    Prefer a padded face cut from the scene, then the YOLO person box, then the
    aligned face square used for embedding.
    """
    if frame is not None and getattr(frame, "size", 0) > 0 and face_bbox is not None:
        for pad in (0.28, 0.18, 0.12):
            tight = crop_face_from_frame(frame, face_bbox, pad_ratio=pad)
            if tight is not None and tight.size > 0:
                return np.ascontiguousarray(tight)
    if (
        frame is not None
        and getattr(frame, "size", 0) > 0
        and person_bbox is not None
    ):
        person = crop_person_from_frame(frame, person_bbox)
        if person is not None and person.size > 0:
            return person
    return np.ascontiguousarray(aligned_face)


def map_bbox_from_person_crop_to_source(
    person_bbox: np.ndarray,
    local_bbox: Union[Tuple[int, int, int, int], np.ndarray],
    frame_shape: Tuple[int, ...],
    *,
    pad_ratio: float = PERSON_CROP_PAD_RATIO,
    upscale_min_side: int = 0,
    crop_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int, int]:
    """
    Map a face box from a person sub-crop (optionally upscaled) back to source-frame pixels.
    """
    px1, py1, px2, py2 = (int(v) for v in person_bbox[:4])
    fh, fw = frame_shape[:2]
    pad_px = int(pad_ratio * max(px2 - px1, py2 - py1))
    ox1 = max(0, px1 - pad_px)
    oy1 = max(0, py1 - pad_px)

    lx1, ly1, lx2, ly2 = (int(v) for v in local_bbox[:4])
    scale = 1.0
    if crop_shape is not None and upscale_min_side > 0:
        ch, cw = crop_shape[:2]
        if max(ch, cw) < upscale_min_side:
            scale = upscale_min_side / float(max(ch, cw))

    return (
        max(0, ox1 + int(round(lx1 / scale))),
        max(0, oy1 + int(round(ly1 / scale))),
        min(fw, ox1 + int(round(lx2 / scale))),
        min(fh, oy1 + int(round(ly2 / scale))),
    )


def face_crop_is_photographic(
    face_bgr: Optional[np.ndarray],
    *,
    min_side: int = CAPTURE_NATIVE_MIN_SIDE,
) -> bool:
    """True when a crop is large enough to look like a real photo after mild resize."""
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return False
    h, w = int(face_bgr.shape[0]), int(face_bgr.shape[1])
    return min(h, w) >= int(min_side)


def pad_to_square(img: np.ndarray, fill: int = 0) -> np.ndarray:
    """Make a square by center-cropping the longer side (no black bars)."""
    if img is None or getattr(img, "size", 0) == 0:
        return img
    out = np.ascontiguousarray(img)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.shape[2] > 3:
        out = out[:, :, :3]
    h, w = int(out.shape[0]), int(out.shape[1])
    side = min(h, w)
    if side < 1:
        return out
    if h == w:
        return out
    y = max(0, (h - side) // 2)
    x = max(0, (w - side) // 2)
    return np.ascontiguousarray(out[y : y + side, x : x + side])


def normalize_capture_output(
    img: np.ndarray,
    *,
    side: Optional[int] = None,
) -> np.ndarray:
    """Square face fill at a fixed size — no letterbox, no extra body."""
    if img is None or getattr(img, "size", 0) == 0:
        return img
    out = pad_to_square(img)
    target = int(side if side is not None else CAPTURE_OUTPUT_SIDE)
    if target < 32:
        target = 256
    h = int(out.shape[0])
    if h == target:
        return out
    interp = cv2.INTER_LANCZOS4 if h < target else cv2.INTER_AREA
    return cv2.resize(out, (target, target), interpolation=interp)


def balance_portrait_body_crop(img: np.ndarray, *, target_aspect: float = 1.42) -> np.ndarray:
    """
    Return the native head-to-waist crop unchanged.

    Widening narrow strips with ``BORDER_REPLICATE`` produced ugly colored side
    bars on Telegram / gallery saves — prefer a valid native crop or a face fallback.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return img
    return np.ascontiguousarray(img)


def prepare_face_thumbnail(
    face_bgr: np.ndarray,
    *,
    enhance: bool = False,
) -> np.ndarray:
    """
    Prepare a face crop for disk / dashboard / Telegram display.

    Default: original pixels, padded/resized to a consistent square.
    Pass ``enhance=True`` for GFPGAN + CLAHE (then still normalize).
    """
    if face_bgr is None or face_bgr.size == 0:
        return face_bgr
    if enhance:
        out = enhance_face_for_display(face_bgr)
    else:
        out = np.ascontiguousarray(face_bgr)
    if capture_display_is_full_frame() or capture_display_is_body_crop() or capture_display_is_person_box():
        h, w = out.shape[:2]
        thumb_min = int(CAPTURE_THUMB_MIN_SIDE)
        short = min(h, w)
        if short < thumb_min and short > 0:
            scale = min(float(thumb_min) / float(short), float(CAPTURE_THUMB_MAX_UPSCALE))
            if scale > 1.01:
                out = cv2.resize(
                    out,
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_LANCZOS4 if scale > 1.05 else cv2.INTER_AREA,
                )
        try:
            from face_print_filter import is_likely_streaked_or_corrupt_capture

            streak, _ = is_likely_streaked_or_corrupt_capture(out)
            if streak:
                return face_bgr
        except Exception:
            pass
        return out
    return normalize_capture_output(out)


def save_face_capture(
    aligned_face: np.ndarray,
    known_person_id: Optional[int],
    track_id: int,
    *,
    person_name: Optional[str] = None,
) -> Optional[str]:
    """Save aligned face crop; return filename relative to captures/ folder."""
    if aligned_face is None or aligned_face.size == 0:
        return None
    # Tiny heads become cartoon blobs when forced large — skip those saves.
    if not face_crop_is_photographic(aligned_face):
        return None

    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = known_person_id if known_person_id is not None else f"track{track_id}"
    filename = build_capture_filename(
        "face",
        timestamp=ts,
        person_name=person_name,
        person_id=pid,
    )
    path = CAPTURES_DIR / filename

    if capture_display_is_full_frame():
        thumbnail = np.ascontiguousarray(aligned_face)
    else:
        thumbnail = prepare_face_thumbnail(aligned_face, enhance=CAPTURE_FACE_ENHANCE)
    try:
        from face_print_filter import is_likely_streaked_or_corrupt_capture

        streak, reason = is_likely_streaked_or_corrupt_capture(thumbnail)
        if streak:
            catch_every = os.environ.get("CATCH_EVERY_PERSON", "1").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            soft_ok = catch_every and any(
                tok in str(reason).lower()
                for tok in ("narrow_strip", "thin_body", "too_tiny")
            )
            if not soft_ok:
                print(f"[CAPTURE] skip streaky/distorted save: {reason}")
                return None
    except Exception:
        pass
    if not cv2.imwrite(
        str(path),
        thumbnail,
        [int(cv2.IMWRITE_JPEG_QUALITY), CAPTURE_JPEG_QUALITY],
    ):
        return None
    return filename


# Saved face JPEGs: 0 = mild upscale only; 1 = GFPGAN/CLAHE on disk.
CAPTURE_FACE_ENHANCE = os.environ.get("CAPTURE_FACE_ENHANCE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
CAPTURE_THUMB_MIN_SIDE = int(os.environ.get("CAPTURE_THUMB_MIN_SIDE", "96"))
CAPTURE_THUMB_MAX_UPSCALE = float(os.environ.get("CAPTURE_THUMB_MAX_UPSCALE", "1.5"))
SCENE_JPEG_QUALITY = int(os.environ.get("CAPTURE_SCENE_JPEG_QUALITY", "90"))
# Saved scene JPEGs: 0 = original CCTV frame (no red box); 1 = draw face box on disk.
CAPTURE_SCENE_DRAW_BOX = os.environ.get("CAPTURE_SCENE_DRAW_BOX", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _scene_filename_for_face(
    face_filename: Optional[str],
    pid: object,
    ts: str,
    *,
    person_name: Optional[str] = None,
) -> str:
    if face_filename and "_face_" in str(face_filename):
        return str(face_filename).replace("_face_", "_scene_", 1)
    return build_capture_filename(
        "scene",
        timestamp=ts,
        person_name=person_name,
        person_id=pid,
    )


def save_scene_capture(
    frame_bgr: Optional[np.ndarray],
    known_person_id: Optional[int],
    track_id: int,
    *,
    face_bbox: Optional[Union[Tuple[float, ...], np.ndarray]] = None,
    face_filename: Optional[str] = None,
    person_name: Optional[str] = None,
    max_side: int = SCENE_MAX_SIDE,
) -> Optional[str]:
    """
    Save a full CCTV scene for the Review modal (original frame, no HUD overlays).

    Returns filename relative to captures/ folder. When ``face_filename`` is set,
    uses the matching ``*_scene_*`` name so face + scene stay paired.

    Set CAPTURE_SCENE_DRAW_BOX=1 to burn a red face box into the saved JPEG.
    """
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return None
    scene = np.ascontiguousarray(frame_bgr)
    if CAPTURE_SCENE_DRAW_BOX and face_bbox is not None:
        try:
            from telegram_alerts import draw_face_square_on_scene

            scene = draw_face_square_on_scene(scene, face_bbox)
        except Exception:
            pass

    h, w = int(scene.shape[0]), int(scene.shape[1])
    side = max(h, w)
    if side > int(max_side) > 0:
        scale = float(max_side) / float(side)
        scene = cv2.resize(
            scene,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = known_person_id if known_person_id is not None else f"track{track_id}"
    filename = _scene_filename_for_face(face_filename, pid, ts, person_name=person_name)
    path = CAPTURES_DIR / filename
    if not cv2.imwrite(
        str(path),
        scene,
        [int(cv2.IMWRITE_JPEG_QUALITY), SCENE_JPEG_QUALITY],
    ):
        return None
    return filename
