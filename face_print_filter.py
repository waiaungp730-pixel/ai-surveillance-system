"""
Reject faces that live on wall photos / posters / framed prints.

CCTV RetinaFace happily locks onto portraits hanging on a wall. Those are not
live visitors — block them before capture / Telegram / overlay boxes.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


def _as_xyxy(bbox: Sequence[float]) -> Optional[BBox]:
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
    except (TypeError, ValueError, IndexError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _rect_encloses(
    rx1: int,
    ry1: int,
    rx2: int,
    ry2: int,
    px: float,
    py: float,
    *,
    margin: float = 0.02,
) -> bool:
    w = max(1, rx2 - rx1)
    h = max(1, ry2 - ry1)
    return (
        (rx1 + margin * w) <= px <= (rx2 - margin * w)
        and (ry1 + margin * h) <= py <= (ry2 - margin * h)
    )


def _annulus_mask(
    shape_hw: Tuple[int, int],
    face: BBox,
    inner_scale: float,
    outer_scale: float,
) -> np.ndarray:
    """Binary mask: ring around face between inner_scale and outer_scale of face size."""
    h, w = shape_hw
    fx1, fy1, fx2, fy2 = face
    fh, fw = max(1, fy2 - fy1), max(1, fx2 - fx1)
    side = float(max(fw, fh))
    pad_in = int(side * inner_scale)
    pad_out = int(side * outer_scale)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(
        mask,
        (max(0, fx1 - pad_out), max(0, fy1 - pad_out)),
        (min(w - 1, fx2 + pad_out), min(h - 1, fy2 + pad_out)),
        1,
        -1,
    )
    cv2.rectangle(
        mask,
        (max(0, fx1 - pad_in), max(0, fy1 - pad_in)),
        (min(w - 1, fx2 + pad_in), min(h - 1, fy2 + pad_in)),
        0,
        -1,
    )
    return mask


def _gold_and_wall_fracs(
    frame_bgr: np.ndarray,
    face: BBox,
    *,
    inner_scale: float = 1.3,
    outer_scale: float = 5.0,
) -> Tuple[float, float]:
    """
    Fraction of gold/brass frame color and painted-wall color in an outer ring.

    Framed wall photos sit inside a gold/wood border with wall paint outside;
    live desk workers usually do not.
    """
    img_h, img_w = frame_bgr.shape[:2]
    mask = _annulus_mask((img_h, img_w), face, inner_scale, outer_scale)
    n = int(cv2.countNonZero(mask))
    if n < 80:
        return 0.0, 0.0

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Gold / brass / warm wood frames (require saturation so gray desks don't hit).
    gold = cv2.inRange(hsv, (8, 55, 55), (40, 255, 255))
    # Office wall paint (green / yellow-green / beige).
    wall = cv2.inRange(hsv, (18, 20, 55), (100, 180, 255))
    gold_n = int(cv2.countNonZero(cv2.bitwise_and(gold, mask * 255)))
    wall_n = int(cv2.countNonZero(cv2.bitwise_and(wall, mask * 255)))
    return gold_n / float(n), wall_n / float(n)


def _has_nearby_portrait_pair(
    face: BBox,
    other_faces: Sequence[Sequence[float]],
) -> bool:
    """True when another similar-sized face sits beside this one (group portrait)."""
    fx1, fy1, fx2, fy2 = face
    fh, fw = max(1, fy2 - fy1), max(1, fx2 - fx1)
    fcx = 0.5 * (fx1 + fx2)
    fcy = 0.5 * (fy1 + fy2)
    for other in other_faces:
        ob = _as_xyxy(other)
        if ob is None:
            continue
        ox1, oy1, ox2, oy2 = ob
        if (ox1, oy1, ox2, oy2) == face:
            continue
        oh, ow = max(1, oy2 - oy1), max(1, ox2 - ox1)
        size_ratio = max(fh, fw) / float(max(oh, ow))
        if size_ratio < 0.55 or size_ratio > 1.85:
            continue
        ocx = 0.5 * (ox1 + ox2)
        ocy = 0.5 * (oy1 + oy2)
        if abs(ocy - fcy) > max(fh, oh) * 0.95:
            continue
        if abs(ocx - fcx) > max(fw, ow) * 5.5:
            continue
        if abs(ocx - fcx) < max(fw, ow) * 0.55:
            continue
        return True
    return False


def _rectangular_frame_hit(
    roi_bgr: np.ndarray,
    lx1: int,
    ly1: int,
    lx2: int,
    ly2: int,
) -> bool:
    """Edge contour that looks like a picture frame enclosing the face."""
    fh, fw = max(1, ly2 - ly1), max(1, lx2 - lx1)
    face_area = float(fw * fh)
    fcx = 0.5 * (lx1 + lx2)
    fcy = 0.5 * (ly1 + ly2)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        # Group photos: frame area can be >> face.
        if area < face_area * 2.0 or area > face_area * 120.0:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 40:
            continue
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) < 4 or len(approx) > 8:
            continue
        rx, ry, rw_c, rh_c = cv2.boundingRect(approx)
        if rw_c < fw * 1.25 or rh_c < fh * 1.25:
            continue
        if rw_c > fw * 14 or rh_c > fh * 14:
            continue
        aspect = rw_c / float(max(1, rh_c))
        if aspect < 0.4 or aspect > 2.6:
            continue
        if not _rect_encloses(rx, ry, rx + rw_c, ry + rh_c, fcx, fcy, margin=0.06):
            continue
        fill = face_area / float(max(1, rw_c * rh_c))
        # Single close-up portrait ~0.2–0.5; group photo faces often << 0.1.
        if 0.012 <= fill <= 0.55:
            return True
    return False


def is_likely_wall_print_or_poster(
    frame_bgr: np.ndarray,
    face_bbox: Sequence[float],
    *,
    person_bbox: Optional[Sequence[float]] = None,
    other_face_bboxes: Optional[Sequence[Sequence[float]]] = None,
) -> Tuple[bool, str]:
    """
    Return (True, reason) when the face is probably a photo/poster on a wall.

    Signals (strong combo → reject):
    - Gold/brass frame ring + painted wall around the face
    - Decorative rectangular frame contour
    - Neighbor face of similar size (group portrait in one frame)
    - YOLO person box that is only a bust / hanging mid-wall with frame cues
    """
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return False, ""
    box = _as_xyxy(face_bbox)
    if box is None:
        return False, ""
    fx1, fy1, fx2, fy2 = box
    fh, fw = fy2 - fy1, fx2 - fx1
    if fh < 8 or fw < 8:
        return False, ""

    img_h, img_w = frame_bgr.shape[:2]
    gold_frac, wall_frac = _gold_and_wall_fracs(frame_bgr, box)
    paired = _has_nearby_portrait_pair(box, other_face_bboxes or ())

    # Bust-in-frame: only reject when framed-photo cues exist.
    # Plain hallway YOLO boxes are often head–chest and must NOT be dropped.
    if person_bbox is not None:
        pb = _as_xyxy(person_bbox)
        if pb is not None:
            px1, py1, px2, py2 = pb
            ph = max(1, py2 - py1)
            pw = max(1, px2 - px1)
            framed = gold_frac >= 0.08 or wall_frac >= 0.12 or paired
            if framed and ph < fh * 2.35 and pw < fw * 3.2:
                return True, "bust_person_box"
            if framed and fh / float(ph) >= 0.42 and (fy1 - py1) / float(ph) <= 0.2:
                return True, "face_fills_person"
            # Hanging mid-wall torso (photo people) with frame/wall paint cues.
            bottom_frac = float(py2) / float(max(1, img_h))
            height_frac = float(ph) / float(max(1, img_h))
            if (
                bottom_frac <= 0.52
                and height_frac <= 0.38
                and gold_frac >= 0.08
                and (wall_frac >= 0.10 or paired)
            ):
                return True, "hanging_wall_person"

    # Strong framed-photo cues (no dark-clothing false positives).
    if gold_frac >= 0.12 and wall_frac >= 0.10:
        return True, f"gold_wall_frame(g={gold_frac:.2f},w={wall_frac:.2f})"
    if gold_frac >= 0.18:
        return True, f"gold_frame(g={gold_frac:.2f})"
    if paired and gold_frac >= 0.08:
        return True, f"group_portrait(g={gold_frac:.2f})"

    # Local ROI contour / flat-mat checks (expanded for group photos).
    pad = int(max(fw, fh) * 5.0)
    x1 = max(0, fx1 - pad)
    y1 = max(0, fy1 - pad)
    x2 = min(img_w, fx2 + pad)
    y2 = min(img_h, fy2 + pad)
    if x2 - x1 < 40 or y2 - y1 < 40:
        return False, ""

    roi = frame_bgr[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    lx1, ly1 = fx1 - x1, fy1 - y1
    lx2, ly2 = fx2 - x1, fy2 - y1

    frame_hit = _rectangular_frame_hit(roi, lx1, ly1, lx2, ly2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 160)
    ring = np.ones((rh, rw), dtype=np.uint8)
    cv2.rectangle(
        ring,
        (max(0, lx1 - 2), max(0, ly1 - 2)),
        (min(rw - 1, lx2 + 2), min(rh - 1, ly2 + 2)),
        0,
        -1,
    )
    band = np.zeros((rh, rw), dtype=np.uint8)
    band_pad = int(max(fw, fh) * 1.8)
    cv2.rectangle(
        band,
        (max(0, lx1 - band_pad), max(0, ly1 - band_pad)),
        (min(rw - 1, lx2 + band_pad), min(rh - 1, ly2 + band_pad)),
        1,
        -1,
    )
    ring = cv2.bitwise_and(ring, band)
    ring_n = int(cv2.countNonZero(ring))
    ring_edges = int(cv2.countNonZero(cv2.bitwise_and(edges, ring)))
    ring_edge_frac = ring_edges / float(max(1, ring_n))

    # Gold only in the near ring — never treat dark shirts as a "frame".
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gold = cv2.inRange(hsv, (8, 55, 55), (40, 255, 255))
    gold_near = float(cv2.countNonZero(cv2.bitwise_and(gold, ring * 255))) / float(
        max(1, ring_n)
    )

    below_y1 = min(rh, ly2 + 2)
    below_y2 = min(rh, ly2 + int(fh * 2.2))
    below_x1 = max(0, lx1 - int(fw * 0.35))
    below_x2 = min(rw, lx2 + int(fw * 0.35))
    body_missing = False
    if below_y2 > below_y1 and below_x2 > below_x1:
        below = roi[below_y1:below_y2, below_x1:below_x2]
        if below.size > 0:
            b_std = float(np.std(below.astype(np.float32)))
            b_edges = cv2.Canny(cv2.cvtColor(below, cv2.COLOR_BGR2GRAY), 50, 140)
            b_edge_frac = float(cv2.countNonZero(b_edges)) / float(max(1, b_edges.size))
            if b_std < 28.0 and b_edge_frac < 0.045:
                body_missing = True

    score = 0.0
    if frame_hit:
        score += 0.55
    if ring_edge_frac >= 0.085:
        score += 0.20
    if gold_near >= 0.08:
        score += 0.35
    elif gold_near >= 0.04:
        score += 0.18
    if body_missing:
        score += 0.22
    if paired:
        score += 0.20
    if gold_frac >= 0.08:
        score += 0.15

    if score >= 0.55:
        return True, f"wall_print(score={score:.2f},frame={int(frame_hit)})"
    return False, ""


def _skin_fraction_bgr(bgr: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return float(cv2.countNonZero(mask)) / float(max(1, mask.size))


def _horizontal_skin_bands(arr: np.ndarray) -> Tuple[float, float, float]:
    """Skin fraction in upper / middle / lower thirds of a crop."""
    h = int(arr.shape[0])
    if h < 9:
        return 0.0, 0.0, 0.0
    t = max(1, h // 3)
    upper = _skin_fraction_bgr(arr[0:t])
    mid = _skin_fraction_bgr(arr[t : 2 * t])
    lower = _skin_fraction_bgr(arr[2 * t : h])
    return upper, mid, lower


def is_likely_rear_view(face_bgr: Optional[np.ndarray]) -> Tuple[bool, str]:
    """
    True when the crop is the back of a head (or a head-to-waist shot of someone
    walking away), not a visible face.

    Used before save / Telegram so "Face detected" is not sent for backs.
    """
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return False, ""
    try:
        full = np.asarray(face_bgr)
        if full.ndim < 3 or full.shape[2] < 3:
            return False, ""
        full_h, full_w = int(full.shape[0]), int(full.shape[1])
        if min(full_h, full_w) < 12:
            return False, ""

        # Walking-away body/bust plates: dark shirt mid-band + bright floor below
        # + little usable face in the head strip (hallway CAM backs).
        if full_h > int(full_w * 1.20):
            head_full = full[0 : max(12, int(full_h * 0.34))]
            top_band = head_full[0 : max(1, int(head_full.shape[0] / 3))]
            top_skin = _skin_fraction_bgr(top_band)
            torso = full[int(full_h * 0.28) : max(int(full_h * 0.29), int(full_h * 0.55))]
            feet = full[int(full_h * 0.58) :]
            if torso.size >= 64 and feet.size >= 64:
                torso_gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
                feet_gray = cv2.cvtColor(feet, cv2.COLOR_BGR2GRAY)
                torso_dark = float(np.mean(torso_gray < 105))
                floor_v = float(
                    cv2.cvtColor(feet, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
                )
                floor_edge = float(cv2.Canny(feet_gray, 40, 120).mean()) / 255.0
                head_u, head_m, head_l = _horizontal_skin_bands(head_full)
                # Dark/colored shirt + bright tiled floor + hair/crown on top = walking away.
                # Blue/bright shirts fail a pure "torso_dark" gate — also accept low torso skin
                # with saturated clothing.
                torso_skin = _skin_fraction_bgr(torso)
                torso_hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
                torso_sat = float(torso_hsv[:, :, 1].mean())
                clothing_torso = torso_skin <= 0.16 and (
                    torso_dark >= 0.28 or torso_sat >= 40.0
                )
                # Light shirts + bare arms read as high torso_skin — still a back when
                # the head strip has almost no face and the lower plate is floor/legs.
                head_inner_skin = _skin_fraction_bgr(
                    head_full[
                        int(head_full.shape[0] * 0.12) : int(head_full.shape[0] * 0.78),
                        int(head_full.shape[1] * 0.16) : int(head_full.shape[1] * 0.84),
                    ]
                ) if min(head_full.shape[:2]) >= 16 else _skin_fraction_bgr(head_full)
                full_body_back = (
                    floor_v >= 120.0
                    and floor_edge >= 0.10
                    and top_skin < 0.28
                    and head_inner_skin < 0.12
                    and full_h >= int(full_w * 1.35)
                )
                if (
                    (clothing_torso or full_body_back)
                    and floor_v >= 120.0
                    and top_skin < 0.35
                    and floor_edge >= 0.10
                ):
                    return True, (
                        f"rear_view_walking_away(td={torso_dark:.2f},"
                        f"tSat={torso_sat:.0f},tSkin={torso_skin:.2f},"
                        f"headInner={head_inner_skin:.2f},floorV={floor_v:.0f},"
                        f"topSkin={top_skin:.2f},edge={floor_edge:.2f})"
                    )
                # Ears/neck only in the head strip — no mouth/chin band.
                if (
                    head_l < 0.06
                    and head_m >= 0.10
                    and head_u >= 0.08
                    and _skin_fraction_bgr(head_full) < 0.22
                ):
                    return True, (
                        f"rear_view_ears_neck(u={head_u:.2f},m={head_m:.2f},"
                        f"l={head_l:.2f})"
                    )

        # Head-to-waist plates: only the top (head) decides front vs back.
        arr = full
        h, w = full_h, full_w
        if h > int(w * 1.22):
            arr = full[0 : max(12, int(h * 0.34))]
            h, w = int(arr.shape[0]), int(arr.shape[1])
        iy1, iy2 = int(h * 0.12), int(h * 0.78)
        ix1, ix2 = int(w * 0.16), int(w * 0.84)
        if iy2 - iy1 < 8 or ix2 - ix1 < 8:
            inner = arr
        else:
            inner = arr[iy1:iy2, ix1:ix2]
        inner_skin = _skin_fraction_bgr(inner)
        inner_gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        dark_inner = float(np.mean(inner_gray < 88))
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        mean_s = float(hsv[:, :, 1].mean())
        mean_v = float(hsv[:, :, 2].mean())
        upper, mid, lower = _horizontal_skin_bands(arr)
        face_skin = _skin_fraction_bgr(arr)

        # Hair filling the face plate — no cheeks/forehead.
        if inner_skin < 0.09 and dark_inner >= 0.28 and mean_v < 110:
            return True, (
                f"rear_view_hair(skin={inner_skin:.2f},dark={dark_inner:.2f})"
            )
        if inner_skin < 0.08 and mean_s < 70 and mean_v < 100:
            return True, f"rear_view_dull_hair(s={mean_s:.0f},v={mean_v:.0f})"

        # Hand scratching the crown from behind: skin only in the upper band
        # and strongly one-sided (palm), while mid/lower is shirt — not a face.
        # Must use the FULL plate (before head-strip shrink) so the blue shirt
        # stays in the bottom third.
        fh, fw = full_h, full_w
        top_third = full[0 : max(1, fh // 3)]
        mid_third = full[fh // 3 : max(fh // 3 + 1, 2 * fh // 3)]
        bot_third = full[2 * fh // 3 :]
        top_skin_t = _skin_fraction_bgr(top_third)
        mid_skin_t = _skin_fraction_bgr(mid_third) if mid_third.size >= 16 else 0.0
        bot_skin_t = _skin_fraction_bgr(bot_third) if bot_third.size >= 16 else 0.0
        top_w = int(top_third.shape[1])
        if top_w >= 8:
            top_l = _skin_fraction_bgr(top_third[:, : top_w // 2])
            top_r = _skin_fraction_bgr(top_third[:, top_w // 2 :])
            top_lr = abs(top_l - top_r)
        else:
            top_lr = 0.0
        bot_hsv = (
            cv2.cvtColor(bot_third, cv2.COLOR_BGR2HSV)
            if bot_third.size >= 16
            else None
        )
        bot_sat = float(bot_hsv[:, :, 1].mean()) if bot_hsv is not None else 0.0
        full_skin = _skin_fraction_bgr(full)
        if (
            top_skin_t >= 0.18
            and mid_skin_t <= 0.10
            and bot_skin_t <= 0.08
            and top_lr >= 0.14
            and (bot_sat >= 50.0 or bot_skin_t <= 0.03)
            and full_skin < 0.50
        ):
            return True, (
                f"rear_view_hand_crown(top={top_skin_t:.2f},mid={mid_skin_t:.2f},"
                f"bot={bot_skin_t:.2f},lr={top_lr:.2f},botS={bot_sat:.0f})"
            )

        # Bright hallway / window glare: still a back if there is almost no face skin.
        if inner_skin < 0.10 and upper < 0.08 and mid < 0.10:
            return True, (
                f"rear_view_no_face_bright(u={upper:.2f},skin={inner_skin:.2f})"
            )

        # No face skin in forehead/cheek bands.
        if upper < 0.07 and mid < 0.08 and face_skin < 0.14:
            return True, f"rear_view_no_face(u={upper:.2f},m={mid:.2f})"

        # Hair on top, only neck/ear skin at the bottom of the box.
        if upper < 0.05 and mid < 0.09 and lower >= 0.10 and dark_inner >= 0.22:
            return True, f"rear_view_neck_only(l={lower:.2f})"

        # Crown / rear scalp: dark hair on top, skin only as neck/jaw below.
        top = arr[0 : max(1, h // 3)]
        bot = arr[h // 2 :]
        top_skin = _skin_fraction_bgr(top)
        bot_skin = _skin_fraction_bgr(bot)
        top_dark = float(np.mean(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY) < 90))
        # Nape / back-of-head: dark hair crown + neck skin filling the mid band
        # (RetinaFace often locks onto this as a "face").
        if (
            top_dark >= 0.55
            and top_skin <= 0.16
            and mid >= 0.40
            and upper <= 0.18
            and dark_inner >= 0.35
            and face_skin < 0.50
        ):
            return True, (
                f"rear_view_nape(topDark={top_dark:.2f},mid={mid:.2f},"
                f"topSkin={top_skin:.2f})"
            )
        if (
            top_skin < 0.16
            and top_dark >= 0.35
            and bot_skin >= 0.28
            and bot_skin > top_skin + 0.15
            and face_skin < 0.45
        ):
            return True, (
                f"rear_view_crown_neck(top={top_skin:.2f},bot={bot_skin:.2f})"
            )

        # Ears/neck with no chin/mouth on a tight face plate.
        if lower < 0.06 and mid >= 0.10 and upper >= 0.08 and face_skin < 0.22:
            return True, (
                f"rear_view_ears_neck(u={upper:.2f},m={mid:.2f},l={lower:.2f})"
            )

        return False, ""
    except Exception:
        return False, ""


def is_likely_looking_down(face_bgr: Optional[np.ndarray]) -> Tuple[bool, str]:
    """
    High-angle CCTV looking-at-phone / floor: hair/scalp on top, face in the lower
    half. InsightFace often flips these men to Female — treat as Unknown instead.
    """
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return False, ""
    try:
        arr = np.asarray(face_bgr)
        if arr.ndim < 3 or arr.shape[2] < 3:
            return False, ""
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if min(h, w) < 20:
            return False, ""
        upper, mid, lower = _horizontal_skin_bands(arr)
        top = arr[0 : max(1, h // 3)]
        bot = arr[int(h * 0.45) :]
        top_skin = _skin_fraction_bgr(top)
        bot_skin = _skin_fraction_bgr(bot) if bot.size >= 32 else 0.0
        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        top_dark = float(np.mean(gray < 95))
        # Hair / scalp / cap crown without much face skin on top.
        if (
            top_skin <= 0.20
            and top_dark >= 0.22
            and bot_skin >= 0.12
            and bot_skin >= top_skin + 0.08
            and upper <= 0.22
            and (mid + lower) >= 0.18
        ):
            return True, (
                f"looking_down(topSkin={top_skin:.2f},bot={bot_skin:.2f},"
                f"topDark={top_dark:.2f})"
            )
        # High-angle office cams: dark hair band + chin/face lower in frame.
        # Looser than above so olive-uniform men looking at desk aren't Female.
        top_band = arr[0 : max(1, int(h * 0.28))]
        mid_face = arr[int(h * 0.35) : int(h * 0.78)]
        top_band_dark = float(
            np.mean(cv2.cvtColor(top_band, cv2.COLOR_BGR2GRAY) < 110)
        )
        top_band_skin = _skin_fraction_bgr(top_band)
        mid_skin = _skin_fraction_bgr(mid_face) if mid_face.size >= 32 else 0.0
        if (
            h >= int(w * 0.85)
            and top_band_dark >= 0.28
            and top_band_skin <= 0.18
            and mid_skin >= 0.10
            and mid_skin >= top_band_skin + 0.05
            and upper <= 0.28
        ):
            return True, (
                f"looking_down_hi(topDark={top_band_dark:.2f},"
                f"midSkin={mid_skin:.2f})"
            )
        return False, ""
    except Exception:
        return False, ""


def is_likely_profile_view(face_bgr: Optional[np.ndarray]) -> Tuple[bool, str]:
    """
    True for visible side/profile faces (one cheek toward camera).

    Backs of heads must be filtered with ``is_likely_rear_view`` first.
    """
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return False, ""
    try:
        arr = np.asarray(face_bgr)
        if arr.ndim < 3 or arr.shape[2] < 3:
            return False, ""
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if min(h, w) < 16:
            return False, ""
        rear, rear_reason = is_likely_rear_view(arr)
        if rear:
            return False, rear_reason
        face_skin = _skin_fraction_bgr(arr)
        if face_skin < 0.10:
            return False, "low_face_skin"
        mid = max(1, w // 2)
        left_skin = _skin_fraction_bgr(arr[:, :mid])
        right_skin = _skin_fraction_bgr(arr[:, mid:])
        diff = abs(left_skin - right_skin)
        dominant = max(left_skin, right_skin)
        weaker = min(left_skin, right_skin)
        if dominant >= 0.11 and diff >= 0.035 and weaker <= dominant * 0.72:
            side = "left" if left_skin > right_skin else "right"
            return True, f"profile_asym({side},{diff:.2f})"
        upper, mid_band, lower = _horizontal_skin_bands(arr)
        if face_skin >= 0.12 and diff >= 0.03 and upper >= 0.08 and mid_band >= 0.07:
            return True, f"profile_bands({diff:.2f})"
        return False, ""
    except Exception:
        return False, ""


def is_likely_invalid_face_crop(face_bgr: Optional[np.ndarray]) -> Tuple[bool, str]:
    """
    Reject hands, mouse/desk blobs, backs of heads, and crown-only crops.

    RetinaFace sometimes returns landmarks on non-faces; this uses skin layout
    and scene texture cues that real frontal/profile faces satisfy.
    """
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return False, ""
    try:
        arr = np.asarray(face_bgr)
        if arr.ndim < 3 or arr.shape[2] < 3:
            return False, ""
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if min(h, w) < 12:
            return False, ""
        rear, rear_reason = is_likely_rear_view(arr)
        if rear:
            return True, rear_reason
        wheel, wheel_reason = is_likely_wheel_or_bicycle(arr)
        if wheel:
            return True, wheel_reason

        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
        skin = _skin_fraction_bgr(arr)
        upper, mid, lower = _horizontal_skin_bands(arr)

        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 140)
        edge_frac = float(cv2.countNonZero(edges)) / float(max(1, edges.size))
        white = cv2.inRange(hsv, (0, 0, 175), (180, 55, 255))
        white_frac = float(cv2.countNonZero(white)) / float(max(1, white.size))
        aspect = w / float(max(1, h))
        top = arr[0 : max(1, h // 2)]
        bottom = arr[max(1, h // 2) : h]
        top_v = float(cv2.cvtColor(top, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
        top_s = float(cv2.cvtColor(top, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
        bottom_skin = _skin_fraction_bgr(bottom)
        upper_band = arr[0 : max(1, h // 3)]
        dark_upper = float(np.mean(cv2.cvtColor(upper_band, cv2.COLOR_BGR2GRAY) < 80))
        bands_with_skin = sum(1 for v in (upper, mid, lower) if v >= 0.05)

        # Hand on desk — skin isolated in middle band on bright desk.
        if (
            white_frac >= 0.55
            and mid >= 0.35
            and lower < 0.05
            and upper < 0.12
            and dark_upper < 0.08
        ):
            return True, f"hand_mid_desk(mid={mid:.2f},white={white_frac:.2f})"

        # Back of head — hair top, skin in lower bands, no forehead.
        if (
            upper < 0.05
            and mid >= 0.15
            and lower >= 0.15
            and 0.08 <= skin <= 0.30
        ):
            return True, f"back_head_lower_skin(u={upper:.2f},l={lower:.2f})"

        # Hand on white desk / mouse — bright desk, skin in lower band, no eye region.
        if (
            white_frac >= 0.45
            and lower >= 0.30
            and dark_upper < 0.12
            and edge_frac >= 0.15
            and 0.08 <= skin <= 0.32
        ):
            return True, (
                f"hand_on_desk(white={white_frac:.2f},lower={lower:.2f},dark={dark_upper:.2f})"
            )

        # Skin blob on desk — single band, no dark eye/hair region above.
        if white_frac >= 0.50 and bands_with_skin <= 1 and dark_upper < 0.06:
            return True, f"desk_skin_blob(bands={bands_with_skin})"

        # Back of head — hair/dark upper, heavy mid band, minimal forehead skin.
        if upper < 0.08 and mid >= 0.35 and dark_upper >= 0.45:
            return True, f"back_of_head(mid={mid:.2f},dark={dark_upper:.2f})"

        # Crown / top of head — dark textured top, skin cluster mid, no lower face.
        if (
            upper < 0.04
            and mid >= 0.25
            and lower < 0.10
            and top_v < 120
            and dark_upper >= 0.40
        ):
            return True, f"crown_only(mid={mid:.2f},topV={top_v:.0f})"

        # Top / back of head — heavy mid band, dark hair region, minimal lower face.
        if mid >= 0.55 and upper < 0.25 and lower < 0.22 and dark_upper >= 0.45:
            return True, f"top_of_head(mid={mid:.2f},dark={dark_upper:.2f})"

        # Back of head / crown — skin/hair top-heavy, no mid-face or mouth band.
        if (
            upper >= 0.06
            and mid < 0.045
            and lower < 0.05
            and skin < 0.42
        ):
            return True, (
                f"back_of_head(u={upper:.2f},m={mid:.2f},l={lower:.2f})"
            )

        if top_v < 95.0 and top_s < 55.0 and bottom_skin < 0.06 and upper >= 0.04:
            return True, f"crown_hair(topV={top_v:.0f},bot_skin={bottom_skin:.2f})"

        # Wide shallow blob (arm/desk edge) — not a face aspect.
        if aspect > 1.55 and skin < 0.35 and edge_frac >= 0.11:
            return True, f"wide_non_face(aspect={aspect:.2f})"

        # Tiny desk crops without eye contrast (CAM desk angles).
        if min(h, w) < 56 and white_frac >= 0.40 and dark_upper < 0.15 and lower >= 0.25:
            return True, f"tiny_desk_blob({w}x{h})"

        # Real faces need skin in at least two vertical bands.
        if skin >= 0.10 and bands_with_skin < 2:
            return True, f"flat_skin_layout({upper:.2f},{mid:.2f},{lower:.2f})"

        # Outdoor foliage / tree texture — almost no skin, heavy green vegetation.
        green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_frac = float(cv2.countNonZero(green)) / float(max(1, green.size))
        if skin < 0.06 and green_frac >= 0.22:
            return True, f"foliage_background(skin={skin:.2f},green={green_frac:.2f})"

        # Arm / shoulder skin patch — uniform skin, no dark eye/hair band above.
        if skin >= 0.28 and dark_upper < 0.14:
            spread = max(upper, mid, lower) - min(upper, mid, lower)
            if spread < 0.14 and bands_with_skin >= 2:
                return True, f"uniform_skin_patch(skin={skin:.2f},spread={spread:.2f})"

        return False, ""
    except Exception:
        return False, ""


def is_likely_non_face_object(face_bgr: Optional[np.ndarray]) -> Tuple[bool, str]:
    """
    Reject RetinaFace pareidolia on non-faces (wood finials/knobs, neon blobs).

    Real hallway faces stay accepted; only strong non-face signatures are blocked.
    """
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return False, ""
    bad, reason = is_likely_invalid_face_crop(face_bgr)
    if bad:
        return True, reason
    try:
        arr = np.asarray(face_bgr)
        if arr.ndim < 3 or arr.shape[2] < 3:
            return False, ""
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if min(h, w) < 12:
            return False, ""

        # Ignore letterboxed black margins from dashboard thumbnails.
        gray_full = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray_full > 8)
        if xs.size >= 16:
            arr = arr[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
            h, w = int(arr.shape[0]), int(arr.shape[1])
            if min(h, w) < 12:
                return False, ""

        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        sat_mean = float(hsv[:, :, 1].mean())

        # Neon / hi-sat plastic or orange blob — not a photographic face.
        if sat_mean >= 90.0:
            return True, f"oversaturated_object(S={sat_mean:.0f})"
        orange = cv2.inRange(hsv, (4, 120, 70), (28, 255, 255))
        orange_frac = float(cv2.countNonZero(orange)) / float(max(1, orange.size))
        if sat_mean >= 78.0 and orange_frac >= 0.12:
            return True, f"neon_object(S={sat_mean:.0f},orange={orange_frac:.2f})"

        # Smooth polished sphere / finial / knob (wood or metal).
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_std = float(lap.std())
        hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).ravel()
        hist = hist / float(hist.sum() + 1e-9)
        hue_peak = float(hist.max())
        brown = cv2.inRange(hsv, (4, 28, 45), (30, 210, 220))
        brown_frac = float(cv2.countNonZero(brown)) / float(max(1, brown.size))

        yy, xx = np.mgrid[0:h, 0:w]
        cy = (h - 1) * 0.5
        cx = (w - 1) * 0.5
        rn = np.sqrt(((yy - cy) / max(1.0, h * 0.5)) ** 2 + ((xx - cx) / max(1.0, w * 0.5)) ** 2)
        center_vals = gray[rn < 0.40]
        rim_vals = gray[(rn > 0.55) & (rn < 0.95)]
        if center_vals.size < 8 or rim_vals.size < 8:
            return False, ""
        center_rim = float(center_vals.mean()) / float(float(rim_vals.mean()) + 1e-3)

        # Sphere shading: smooth texture + concentrated warm hue + darker center.
        if (
            lap_std < 16.5
            and hue_peak >= 0.42
            and brown_frac >= 0.18
            and center_rim < 1.02
        ):
            return True, (
                f"spherical_object(lap={lap_std:.1f},hue={hue_peak:.2f},"
                f"brown={brown_frac:.2f},cr={center_rim:.2f})"
            )

        # Extra knob/post cue: very smooth warm blob with plain background fringe.
        plain = cv2.inRange(hsv, (0, 0, 70), (180, 45, 230))
        plain_frac = float(cv2.countNonZero(plain)) / float(max(1, plain.size))
        if (
            lap_std < 15.0
            and brown_frac >= 0.20
            and plain_frac >= 0.30
            and center_rim < 1.05
            and hue_peak >= 0.38
        ):
            return True, f"decor_knob(lap={lap_std:.1f},plain={plain_frac:.2f})"

        # Overexposed moth / lens flare / white door blob near the camera.
        ycrcb = cv2.cvtColor(arr, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        skin_frac = float(cv2.countNonZero(skin)) / float(max(1, skin.size))
        v_mean = float(hsv[:, :, 2].mean())
        white = cv2.inRange(hsv, (0, 0, 170), (180, 55, 255))
        white_frac = float(cv2.countNonZero(white)) / float(max(1, white.size))
        if (
            white_frac >= 0.28
            and skin_frac < 0.14
            and sat_mean <= 55.0
            and v_mean >= 140.0
        ):
            return True, (
                f"overexposed_blob(white={white_frac:.2f},skin={skin_frac:.2f})"
            )

        # Architecture / door / wall texture with almost no skin.
        edges = cv2.Canny(gray, 50, 140)
        edge_frac = float(cv2.countNonZero(edges)) / float(max(1, edges.size))
        if skin_frac < 0.055 and (plain_frac >= 0.35 or white_frac >= 0.22):
            return True, f"non_skin_surface(skin={skin_frac:.2f})"
        if skin_frac < 0.08 and edge_frac >= 0.12 and sat_mean <= 45.0:
            # Strong linear edges + low skin → door frame / wall fixture.
            return True, f"structure_texture(skin={skin_frac:.2f},edge={edge_frac:.2f})"

        # Hallway floor tiles / pavement pareidolia: warm beige reads as YCrCb
        # "skin", but there is almost no dark eye/hair mass in the upper half.
        beige = cv2.inRange(hsv, (8, 15, 90), (40, 110, 240))
        beige_frac = float(cv2.countNonZero(beige)) / float(max(1, beige.size))
        top_gray = gray[0 : max(1, h // 2)]
        dark_top = float(np.mean(top_gray < 70))
        if (
            skin_frac >= 0.55
            and beige_frac >= 0.28
            and dark_top < 0.055
            and v_mean >= 125.0
            and edge_frac >= 0.10
        ):
            return True, (
                f"floor_tile(skin={skin_frac:.2f},beige={beige_frac:.2f},"
                f"darkTop={dark_top:.2f})"
            )

        # Hand / arm on floor or desk — skin blob low in frame, no eye band.
        upper_skin = _skin_fraction_bgr(arr[0 : max(1, h // 3)])
        lower_skin = _skin_fraction_bgr(arr[max(1, (2 * h) // 3) : h])
        if (
            lower_skin >= 0.35
            and upper_skin < 0.12
            and dark_top < 0.08
            and skin_frac >= 0.25
            and beige_frac >= 0.12
        ):
            return True, (
                f"hand_floor(lower={lower_skin:.2f},upper={upper_skin:.2f})"
            )

        # Tiny dark gadget / camera lens on a desk (ring highlight, tiny skin).
        if (
            min(h, w) <= 72
            and skin_frac < 0.10
            and center_rim < 0.92
            and lap_std < 28.0
            and plain_frac >= 0.20
        ):
            return True, f"desk_gadget(skin={skin_frac:.2f},cr={center_rim:.2f})"

        # Bicycle / scooter wheel / hub: circular metal spokes, almost no face skin.
        wheel, wheel_reason = is_likely_wheel_or_bicycle(arr)
        if wheel:
            return True, wheel_reason

        # Wooden railing baluster / post / finial (common hallway false positive).
        wood, wood_reason = is_likely_wood_post_or_rail(arr)
        if wood:
            return True, wood_reason

        # Backpack / harness strap over shoulder or torso — no face.
        tan_strap = cv2.inRange(hsv, (5, 35, 50), (40, 220, 230))
        strap_frac = float(cv2.countNonZero(tan_strap)) / float(max(1, tan_strap.size))
        upper_b, mid_b, lower_b = _horizontal_skin_bands(arr)
        dark_frac = float(np.mean(gray < 70))
        if (
            strap_frac >= 0.06
            and lower_b <= 0.14
            and skin_frac <= 0.32
            and dark_frac >= 0.18
        ):
            return True, (
                f"torso_strap(strap={strap_frac:.2f},skin={skin_frac:.2f},"
                f"lower={lower_b:.2f})"
            )
        # Shoulder / upper-torso only: mid skin, almost no lower face, dark clothing.
        if (
            min(h, w) <= 120
            and upper_b <= 0.28
            and mid_b >= 0.18
            and lower_b <= 0.12
            and skin_frac <= 0.30
            and dark_frac >= 0.22
            and strap_frac >= 0.03
        ):
            return True, (
                f"shoulder_torso(skin={skin_frac:.2f},uml={upper_b:.2f}/"
                f"{mid_b:.2f}/{lower_b:.2f})"
            )

        return False, ""
    except Exception:
        return False, ""


def is_likely_wheel_or_bicycle(face_bgr: Optional[np.ndarray]) -> Tuple[bool, str]:
    """
    True for bicycle / scooter wheels and similar circular metal hubs.

    RetinaFace sometimes locks onto a wheel hub as a 'face' at outdoor gates.
    """
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return False, ""
    try:
        arr = np.asarray(face_bgr)
        if arr.ndim < 3 or arr.shape[2] < 3:
            return False, ""
        h, w = int(arr.shape[0]), int(arr.shape[1])
        side = min(h, w)
        if side < 24:
            return False, ""

        skin = _skin_fraction_bgr(arr)
        if skin >= 0.28:
            return False, ""

        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
        sat_mean = float(hsv[:, :, 1].mean())
        metal = float(
            np.mean((hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 35) & (hsv[:, :, 2] < 210))
        )
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        edge_frac = float(cv2.countNonZero(edges)) / float(max(1, edges.size))

        yy, xx = np.mgrid[0:h, 0:w]
        cy = (h - 1) * 0.5
        cx = (w - 1) * 0.5
        rn = np.sqrt(
            ((yy - cy) / max(1.0, h * 0.5)) ** 2
            + ((xx - cx) / max(1.0, w * 0.5)) ** 2
        )
        ring = edges[(rn > 0.22) & (rn < 0.88)]
        ring_edge = float(np.mean(ring > 0)) if ring.size else 0.0

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(16, side // 3),
            param1=110,
            param2=26,
            minRadius=max(8, side // 10),
            maxRadius=max(12, side // 2),
        )
        n_circ = 0 if circles is None else int(circles.shape[1])
        # Dominant circle covering most of the plate (wheel rim).
        big_rim = False
        if circles is not None:
            for c in circles[0]:
                if float(c[2]) >= side * 0.28:
                    big_rim = True
                    break

        if (
            (n_circ >= 2 or big_rim)
            and metal >= 0.45
            and sat_mean <= 55.0
            and skin < 0.22
            and (ring_edge >= 0.08 or edge_frac >= 0.09)
        ):
            return True, (
                f"bicycle_wheel(circ={n_circ},metal={metal:.2f},skin={skin:.2f},"
                f"ring={ring_edge:.2f},S={sat_mean:.0f})"
            )
        # Spoke-heavy metal plate without enough face skin bands.
        upper, mid, lower = _horizontal_skin_bands(arr)
        if (
            metal >= 0.65
            and sat_mean <= 40.0
            and skin < 0.18
            and edge_frac >= 0.10
            and lower < 0.10
            and max(upper, mid) < 0.22
        ):
            return True, (
                f"metal_spokes(metal={metal:.2f},skin={skin:.2f},edge={edge_frac:.2f})"
            )
        return False, ""
    except Exception:
        return False, ""


def is_likely_unusable_face_plate(face_bgr: Optional[np.ndarray]) -> Tuple[bool, str]:
    """
    Reject tiny mush / pixel-noise plates that Laplacian still scores as 'Clear'.

    These flood Telegram when CATCH_EVERY_PERSON is on.
    """
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return True, "empty"
    try:
        arr = np.asarray(face_bgr)
        if arr.ndim < 3 or arr.shape[2] < 3:
            return True, "bad_shape"
        h, w = int(arr.shape[0]), int(arr.shape[1])
        side = min(h, w)
        if side < 48:
            return True, f"too_tiny({w}x{h})"
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edge = float(cv2.Canny(gray, 40, 120).mean()) / 255.0
        # Sub-72px crops with extreme laplacian = aliasing mush, not a sharp face.
        if side < 72 and lap >= 400.0 and edge >= 0.18:
            return True, f"tiny_mush({w}x{h},lap={lap:.0f},edge={edge:.2f})"
        if side < 72 and lap >= 700.0:
            return True, f"tiny_mush_noise({w}x{h},lap={lap:.0f})"
        # Soft blurry blob with no usable structure.
        if side < 80 and lap < 25.0 and edge < 0.06:
            return True, f"soft_blob({w}x{h},lap={lap:.0f})"
        return False, ""
    except Exception:
        return False, ""


# Real people the camera sees from behind / side / looking down.
_KEEP_PERSON_POSE_MARKERS = (
    "rear_view",
    "looking_down",
    "profile",
    "back_head",
    "back_of_head",
    "walking_away",
    "crown_neck",
    "nape",
    "ears_neck",
    "neck_only",
    "dull_hair",
    "no_face_bright",
    "no_face(",
)
# Banisters, floor tiles, knobs — never treat these as a person catch.
_JUNK_NON_PERSON_MARKERS = (
    "wood",
    "rail",
    "floor_tile",
    "hand_floor",
    "hand_on_desk",
    "hand_mid_desk",
    "desk_skin",
    "desk_gadget",
    "decor_knob",
    "neon",
    "oversaturated",
    "non_skin_surface",
    "structure_texture",
    "spherical",
    "overexposed_blob",
    "streak",
    "empty_panel",
    "wall_print",
    "poster",
    "torso_strap",
    "shoulder_torso",
    "bicycle_wheel",
    "metal_spokes",
    "too_tiny",
    "tiny_mush",
    "soft_blob",
)


def is_keepable_person_pose_reason(reason: str) -> bool:
    """True when a reject reason is a real person pose (back, profile, looking down)."""
    text = str(reason or "").lower()
    return any(marker in text for marker in _KEEP_PERSON_POSE_MARKERS)


def is_junk_non_person_reason(reason: str) -> bool:
    """True for wood / floor / furniture false positives — not a person."""
    if is_keepable_person_pose_reason(reason):
        return False
    text = str(reason or "").lower()
    return any(marker in text for marker in _JUNK_NON_PERSON_MARKERS)


# Full-body YOLO plates: clothing has almost no skin, so face-crop "non_skin"
# / "structure" heuristics must not drop a real person.
_JUNK_PERSON_PLATE_MARKERS = (
    "wood",
    "rail",
    "floor_tile",
    "decor_knob",
    "neon",
    "oversaturated",
    "spherical",
    "desk_gadget",
    "overexposed_blob",
    "wall_print",
    "poster",
    "empty_panel",
    "streak",
    "torso_strap",
    "shoulder_torso",
    "bicycle_wheel",
    "metal_spokes",
    "too_tiny",
    "tiny_mush",
    "soft_blob",
)


def is_junk_person_plate_reason(reason: str) -> bool:
    """True only for definite non-people on a YOLO person-box crop."""
    if is_keepable_person_pose_reason(reason):
        return False
    text = str(reason or "").lower()
    return any(marker in text for marker in _JUNK_PERSON_PLATE_MARKERS)


def is_likely_wood_post_or_rail(face_bgr: Optional[np.ndarray]) -> Tuple[bool, str]:
    """
    True for wooden balusters, posts, and similar furniture false positives.

    Hallway railings are tall, brown, low-skin, and lack face structure.
    """
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return False, ""
    try:
        arr = np.asarray(face_bgr)
        if arr.ndim < 3 or arr.shape[2] < 3:
            return False, ""
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if min(h, w) < 12:
            return False, ""
        aspect = w / float(max(1, h))
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        skin_frac = _skin_fraction_bgr(arr)
        brown = cv2.inRange(hsv, (4, 35, 35), (32, 230, 230))
        brown_frac = float(cv2.countNonZero(brown)) / float(max(1, brown.size))
        # Wood grain often has low-mid saturation; also count warm gray-brown.
        warm = cv2.inRange(hsv, (0, 15, 40), (35, 255, 220))
        warm_frac = float(cv2.countNonZero(warm)) / float(max(1, warm.size))
        wood_frac = max(brown_frac, warm_frac * 0.85)
        # Strong vertical edges (post silhouette).
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
        vert = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        horiz = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        vert_energy = float(np.mean(np.abs(vert)))
        horiz_energy = float(np.mean(np.abs(horiz))) + 1e-3
        vert_ratio = vert_energy / horiz_energy
        edge_frac = float(cv2.countNonZero(edges)) / float(max(1, edges.size))
        # Column center band vs sides (posts are darker/solid in the middle).
        mid_x0, mid_x1 = int(w * 0.28), int(w * 0.72)
        mid_col = gray[:, mid_x0:mid_x1]
        side_col = np.hstack([gray[:, : max(1, mid_x0)], gray[:, mid_x1:]])
        mid_mean = float(mid_col.mean()) if mid_col.size else 0.0
        side_mean = float(side_col.mean()) if side_col.size else 255.0
        center_darker = mid_mean < side_mean - 8.0
        upper, mid_skin, lower = _horizontal_skin_bands(arr)
        # Real faces need forehead/cheek skin — wood mid-bands often fake YCrCb skin.
        face_like_skin = (
            skin_frac >= 0.14
            and upper >= 0.09
            and mid_skin >= 0.08
            and sum(1 for v in (upper, mid_skin, lower) if v >= 0.05) >= 2
        )
        if face_like_skin:
            return False, ""

        tall_thin = aspect <= 0.72 and h >= int(w * 1.15)
        woodish = wood_frac >= 0.12 and upper < 0.09 and skin_frac < 0.28
        vertical_post = vert_ratio >= 1.10 and edge_frac >= 0.03
        if tall_thin and woodish and (vertical_post or center_darker or wood_frac >= 0.18):
            return True, (
                f"wood_post(aspect={aspect:.2f},wood={wood_frac:.2f},"
                f"skin={skin_frac:.2f},vr={vert_ratio:.2f})"
            )
        # Washed-out / gray-brown post with almost zero forehead skin (hallway rail).
        if (
            tall_thin
            and upper < 0.08
            and mid_skin < 0.40
            and wood_frac >= 0.10
            and (vertical_post or wood_frac >= 0.16)
        ):
            return True, (
                f"rail_post(aspect={aspect:.2f},wood={wood_frac:.2f},"
                f"u={upper:.2f})"
            )
        # Any tall crop with weak forehead skin and strong vertical structure.
        if tall_thin and upper < 0.07 and vert_ratio >= 1.20 and edge_frac >= 0.03:
            return True, (
                f"vertical_object(aspect={aspect:.2f},u={upper:.2f},"
                f"vr={vert_ratio:.2f})"
            )
        # Rounded wood finial / knob occupying most of a square crop.
        if (
            aspect >= 0.55
            and aspect <= 1.35
            and wood_frac >= 0.28
            and skin_frac < 0.08
            and edge_frac <= 0.18
        ):
            return True, (
                f"wood_finial(aspect={aspect:.2f},wood={wood_frac:.2f},"
                f"skin={skin_frac:.2f})"
            )
        return False, ""
    except Exception:
        return False, ""


def _edge_replicate_pad_fractions(
    gray: np.ndarray,
    *,
    max_scan: int = 48,
    diff_thresh: float = 0.75,
) -> Tuple[float, float]:
    """
    Fraction of width occupied by identical adjacent columns at left/right edges.

    ``cv2.BORDER_REPLICATE`` widening in body crops produces this artifact.
    """
    if gray is None or getattr(gray, "size", 0) == 0:
        return 0.0, 0.0
    h, w = int(gray.shape[0]), int(gray.shape[1])
    if w < 8 or h < 8:
        return 0.0, 0.0
    g = gray.astype(np.float32)
    left_cols = 0
    for i in range(1, min(max_scan, w // 2)):
        d = float(np.mean(np.abs(g[:, i] - g[:, i - 1])))
        if d <= diff_thresh:
            left_cols = i
        else:
            break
    right_cols = 0
    for i in range(w - 2, max(w // 2 - 1, w - max_scan - 1), -1):
        d = float(np.mean(np.abs(g[:, i + 1] - g[:, i])))
        if d <= diff_thresh:
            right_cols = w - 1 - i
        else:
            break
    return left_cols / float(w), right_cols / float(w)


def is_likely_streaked_or_corrupt_capture(
    img: Optional[np.ndarray],
) -> Tuple[bool, str]:
    """
    True for ugly RTSP streaks, rolling-shutter smears, and replicate-padded edges.

    Rejects captures like narrow vertical strips with colored side bars before
    save / Telegram.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return False, ""
    try:
        arr = np.asarray(img)
        if arr.ndim < 2:
            return False, ""
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if min(h, w) < 12:
            return False, ""
        aspect = h / float(max(1, w))
        if aspect >= 2.05 and w < 100:
            return True, f"narrow_strip(aspect={aspect:.2f},w={w})"
        if aspect >= 1.75 and w < 64:
            return True, f"thin_body(aspect={aspect:.2f},w={w})"

        gray = (
            cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            if arr.ndim >= 3 and arr.shape[2] >= 3
            else arr.astype(np.uint8)
        )
        left_frac, right_frac = _edge_replicate_pad_fractions(gray)
        pad_total = left_frac + right_frac
        if left_frac >= 0.035 or right_frac >= 0.05 or pad_total >= 0.065:
            return True, (
                f"edge_replicate_pad(l={left_frac:.2f},r={right_frac:.2f})"
            )

        cx0, cx1 = int(w * 0.22), int(w * 0.78)
        if cx1 - cx0 >= 8 and h >= 16:
            band = gray[:, cx0:cx1]
            row_diff = np.mean(
                np.abs(band[1:].astype(np.float32) - band[:-1].astype(np.float32)),
                axis=1,
            )
            if row_diff.size:
                spike_frac = float(np.mean(row_diff > 80.0))
                mean_diff = float(np.mean(row_diff))
                if spike_frac >= 0.07 and mean_diff >= 10.0:
                    return True, (
                        f"row_streak(spike={spike_frac:.2f},mean={mean_diff:.1f})"
                    )

        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        short = float(min(h, w))
        # Only reject near-blank mush — distant CCTV busts are soft (lap ~8–30)
        # and must still save / Telegram. Old gate lap<35 dropped those.
        if short >= 64 and lap < 6.0 and aspect >= 1.35:
            return True, f"soft_body(lap={lap:.1f},aspect={aspect:.2f})"

        # Tiny mush / noise patch — only looks like a face after aggressive upscale.
        # Compression noise can inflate Laplacian on 40px blobs, so size alone gates.
        if short < 48:
            return True, f"too_tiny(side={short:.0f})"
        if short < 64 and lap < 22.0:
            return True, f"tiny_mush(side={short:.0f},lap={lap:.1f})"

        # Letterboxed / empty panel: solid black side bars around a thin content strip.
        col_ok = np.mean(gray.astype(np.float32) > 14.0, axis=0) > 0.12
        if col_ok.size >= 16:
            content_frac = float(np.mean(col_ok))
            black_frac = float(np.mean(gray < 10))
            if black_frac >= 0.28 and content_frac <= 0.55:
                return True, (
                    f"letterbox_bars(black={black_frac:.2f},content={content_frac:.2f})"
                )
            # Near-empty canvas (almost all black/near-black).
            if black_frac >= 0.55 and float(gray.mean()) < 40.0:
                return True, f"empty_panel(black={black_frac:.2f},mean={float(gray.mean()):.0f})"

        return False, ""
    except Exception:
        return False, ""


def filter_face_dets_reject_prints(
    frame_bgr: np.ndarray,
    face_dets: Optional[list],
) -> list:
    """Drop FaceDetection objects that look like wall photos/posters or non-faces."""
    dets = list(face_dets or [])
    all_boxes: List[Sequence[float]] = [
        getattr(det, "bbox") for det in dets if getattr(det, "bbox", None) is not None
    ]
    kept = []
    img_h = int(frame_bgr.shape[0]) if frame_bgr is not None else 0
    img_w = int(frame_bgr.shape[1]) if frame_bgr is not None else 0
    for det in dets:
        bbox = getattr(det, "bbox", None)
        if bbox is None:
            continue
        bad, _reason = is_likely_wall_print_or_poster(
            frame_bgr,
            bbox,
            other_face_bboxes=all_boxes,
        )
        if not bad and frame_bgr is not None and img_h > 0 and img_w > 0:
            try:
                x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
                x1 = max(0, min(img_w - 1, x1))
                x2 = max(0, min(img_w, x2))
                y1 = max(0, min(img_h - 1, y1))
                y2 = max(0, min(img_h, y2))
                if x2 > x1 and y2 > y1:
                    crop = frame_bgr[y1:y2, x1:x2]
                    bad, _reason = is_likely_non_face_object(crop)
            except Exception:
                bad = False
        if bad:
            try:
                det.quality_ok = False
                det.reject_reason = _reason or "wall_print"
            except Exception:
                pass
            continue
        kept.append(det)
    return kept
