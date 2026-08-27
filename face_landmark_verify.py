"""
Visible-face landmark verification for capture gating.

Before a face crop is saved to disk or logged in SQLite, verify:
  1. Detection confidence >= FACE_LANDMARK_MIN_CONFIDENCE (default 0.85).
  2. All five core landmarks are present: left/right eye, nose tip, mouth corners.
  3. Each landmark lies inside the face bounding box.
  4. Landmark geometry matches a visible face (frontal or side/profile).
     Backs of heads, shoulders, and hair blobs are rejected.

RetinaFace/YuNet do not expose per-landmark confidence scores; the face-level
detection score is used as the landmark-confidence proxy (standard practice).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np

from tracker_config import (
    FACE_ASPECT_RATIO_MAX,
    FACE_ASPECT_RATIO_MIN,
    FACE_LANDMARK_MIN_CONFIDENCE,
    FACE_MAX_EYE_SPAN_RATIO,
    FACE_MIN_BOX_PX,
    FACE_MIN_EYE_SPAN_RATIO,
)

# RetinaFace + YuNet canonical landmark keys
REQUIRED_LANDMARKS = (
    "right_eye",
    "left_eye",
    "nose",
    "mouth_right",
    "mouth_left",
)

_LANDMARK_ALIASES = {
    "nose_tip": "nose",
    "mouth_left_corner": "mouth_left",
    "mouth_right_corner": "mouth_right",
}

# Profiles compress horizontal eye span; allow lower than frontal-only gate.
FACE_MIN_EYE_SPAN_RATIO_PROFILE = 0.16
# |nose_x - eye_mid_x| / eye_span — ~0 frontal, ~0.5–1.0 strong side view.
FACE_MAX_YAW_FOR_SIDE_VIEW = 1.15


def _get_landmark(
    landmarks: Dict[str, Tuple[float, float]],
    name: str,
) -> Optional[Tuple[float, float]]:
    if not landmarks:
        return None
    pt = landmarks.get(name)
    if pt and len(pt) >= 2:
        return float(pt[0]), float(pt[1])
    alias = _LANDMARK_ALIASES.get(name)
    if alias:
        pt = landmarks.get(alias)
        if pt and len(pt) >= 2:
            return float(pt[0]), float(pt[1])
    return None


def _point_in_bbox(
    pt: Tuple[float, float],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    margin_ratio: float = 0.08,
) -> bool:
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    margin_x = margin_ratio * face_w
    margin_y = margin_ratio * face_h
    return (
        x1 - margin_x <= pt[0] <= x2 + margin_x
        and y1 - margin_y <= pt[1] <= y2 + margin_y
    )


def estimate_face_yaw_ratio(
    landmarks: Dict[str, Tuple[float, float]],
) -> Optional[float]:
    """
    Approximate yaw from 5-point landmarks.

    0 ≈ frontal; larger values ≈ side/profile. Backs of heads usually lack
    usable landmarks and never reach this helper.
    """
    re = _get_landmark(landmarks, "right_eye")
    le = _get_landmark(landmarks, "left_eye")
    nose = _get_landmark(landmarks, "nose")
    if not (re and le and nose):
        return None
    eye_mid_x = (re[0] + le[0]) / 2.0
    eye_span = abs(re[0] - le[0])
    if eye_span < 1e-3:
        return None
    return abs(float(nose[0]) - eye_mid_x) / eye_span


def verify_frontal_landmark_geometry(
    landmarks: Dict[str, Tuple[float, float]],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    allow_side_view: bool = True,
    max_yaw: float = FACE_MAX_YAW_FOR_SIDE_VIEW,
    allow_high_angle: bool = False,
) -> Tuple[bool, str]:
    """
    Eyes + nose + mouth in a plausible face layout.

    Allows straight-on and side/profile views when ``allow_side_view`` is True.
    Rejects backs of heads and non-face blobs (missing/illegal landmark order).
    """
    re = _get_landmark(landmarks, "right_eye")
    le = _get_landmark(landmarks, "left_eye")
    nose = _get_landmark(landmarks, "nose")
    mouth_r = _get_landmark(landmarks, "mouth_right")
    mouth_l = _get_landmark(landmarks, "mouth_left")
    if not (re and le and nose and mouth_r and mouth_l):
        return False, "missing_key_landmarks"

    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    eye_span = abs(re[0] - le[0])
    span_ratio = eye_span / float(face_w)
    min_span = (
        FACE_MIN_EYE_SPAN_RATIO_PROFILE
        if allow_side_view
        else FACE_MIN_EYE_SPAN_RATIO
    )
    if span_ratio < min_span or span_ratio > FACE_MAX_EYE_SPAN_RATIO:
        return False, f"eye_span_ratio({span_ratio:.2f})"
    # Backs of heads: RetinaFace packs fake eyes in the hair blob.
    if span_ratio < 0.16:
        return False, f"rear_eye_span({span_ratio:.2f})"

    eye_y = (re[1] + le[1]) / 2.0
    # Extreme looking-down / crown-of-head crops put eyes near the bottom of the box.
    # High outdoor gate mounts often push eyes mid/lower in the box — allow more.
    rel_eye = (eye_y - float(y1)) / float(face_h)
    eye_limit = 0.88 if allow_high_angle else 0.62
    if rel_eye > eye_limit:
        return False, f"looking_down({rel_eye:.2f})"
    # High CCTV angles can place the nose nearly level with the eyes.
    nose_slack = 0.06 * face_h if allow_high_angle else 0.02 * face_h
    if nose[1] < eye_y - nose_slack:
        return False, "nose_above_eyes"

    mouth_y = (mouth_r[1] + mouth_l[1]) / 2.0
    if mouth_y < nose[1] - 0.02 * face_h:
        return False, "mouth_above_nose"

    mouth_span = abs(mouth_r[0] - mouth_l[0])
    min_mouth = 0.04 * face_w if allow_side_view else 0.12 * face_w
    if mouth_span < min_mouth:
        return False, "mouth_too_narrow"

    yaw = estimate_face_yaw_ratio(landmarks)
    if yaw is None:
        return False, "yaw_unavailable"
    if yaw > float(max_yaw):
        return False, f"yaw_extreme({yaw:.2f})"

    # Both mouth corners nearly on top of each other + tiny eye span ⇒ not a face.
    if span_ratio < 0.08 and mouth_span < 0.05 * face_w:
        return False, "collapsed_landmarks"

    return True, ""


def verify_landmarks_for_capture(
    landmarks: Dict[str, Tuple[float, float]],
    bbox: Tuple[int, int, int, int],
    *,
    detection_confidence: float,
    min_confidence: float = FACE_LANDMARK_MIN_CONFIDENCE,
    allow_side_view: bool = True,
    max_yaw: float = FACE_MAX_YAW_FOR_SIDE_VIEW,
    allow_high_angle: bool = False,
) -> Tuple[bool, str]:
    """
    Full pre-save landmark gate. Returns (ok, reject_reason).
    Classifies sub-threshold or structurally invalid detections as non-face artifacts.
    """
    if detection_confidence < min_confidence:
        return False, f"landmark_confidence({detection_confidence:.2f})"

    for name in REQUIRED_LANDMARKS:
        if _get_landmark(landmarks, name) is None:
            return False, f"missing_{name}"

    x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    for name in REQUIRED_LANDMARKS:
        pt = _get_landmark(landmarks, name)
        assert pt is not None
        if not _point_in_bbox(pt, x1, y1, x2, y2):
            return False, f"landmark_outside_{name}"

    return verify_frontal_landmark_geometry(
        landmarks,
        x1,
        y1,
        x2,
        y2,
        allow_side_view=allow_side_view,
        max_yaw=max_yaw,
        allow_high_angle=allow_high_angle,
    )


def verify_face_detection_for_capture(
    det,
    *,
    min_confidence: float = FACE_LANDMARK_MIN_CONFIDENCE,
    allow_side_view: bool = True,
    max_yaw: float = FACE_MAX_YAW_FOR_SIDE_VIEW,
    allow_distant: bool = False,
    allow_high_angle: bool = False,
    min_box_px: int = FACE_MIN_BOX_PX,
) -> Tuple[bool, str]:
    """Gate a RetinaFace FaceDetection before crop/save (frontal + side OK; back rejected)."""
    try:
        x1, y1, x2, y2 = (int(det.bbox[0]), int(det.bbox[1]), int(det.bbox[2]), int(det.bbox[3]))
        box_side = max(1, max(x2 - x1, y2 - y1))
    except Exception:
        box_side = 0
    ok, reason = verify_landmarks_for_capture(
        det.landmarks,
        det.bbox,
        detection_confidence=float(det.score),
        min_confidence=min_confidence,
        allow_side_view=allow_side_view,
        max_yaw=max_yaw,
        allow_high_angle=allow_high_angle,
    )
    if ok:
        return True, ""
    # Tiny boxes (arm/shoulder pareidolia) must never bypass via soft-accept.
    if box_side < int(min_box_px):
        return False, reason
    # Soft-accept looking_down on high mounts even when the face is large
    # (phone walkers / Gate cams) — do not require allow_distant.
    if allow_high_angle and str(reason).startswith("looking_down"):
        if float(getattr(det, "score", 0.0) or 0.0) >= float(min_confidence) * 0.7:
            return True, ""
    if not allow_distant:
        return False, reason

    # Far hallway / high-angle faces often have noisy landmarks. Soft-accept
    # looking_down and missing points; keep rejecting clear backs / non-faces.
    hard_reject = [
        "yaw_extreme",
        "collapsed_landmarks",
        "nose_above_eyes",
        "mouth_above_nose",
        "landmark_outside_",
        "rear_eye_span",
        "rear_view",
    ]
    # Looking-down (phone / high CCTV mount) and partial landmark miss are common
    # on real walkers — never hard-kill those when allow_distant / catch-everyone.
    if not allow_high_angle and not allow_side_view:
        hard_reject.append("looking_down")
        hard_reject.append("missing_")
    elif not allow_high_angle:
        hard_reject.append("looking_down")
    if any(token in str(reason) for token in hard_reject):
        return False, reason
    if float(getattr(det, "score", 0.0) or 0.0) < float(min_confidence) * 0.85:
        return False, reason
    return True, ""


def face_bbox_in_person_head_band(
    face_bbox: Union[Tuple[float, ...], np.ndarray],
    person_bbox: Union[Tuple[float, ...], np.ndarray],
    *,
    outdoor: bool = False,
) -> Tuple[bool, str]:
    """
    Reject arm/shoulder/torso false positives inside a YOLO person box.

    Real faces sit in the upper portion of the person rectangle; bare-arm
    RetinaFace hits land mid-body.
    """
    try:
        fx1, fy1, fx2, fy2 = (float(v) for v in np.asarray(face_bbox).reshape(-1)[:4])
        px1, py1, px2, py2 = (float(v) for v in np.asarray(person_bbox).reshape(-1)[:4])
    except (TypeError, ValueError, IndexError):
        return True, ""
    if fx2 <= fx1 or fy2 <= fy1 or px2 <= px1 or py2 <= py1:
        return True, ""
    ph = max(1.0, py2 - py1)
    pw = max(1.0, px2 - px1)
    rel_top = (fy1 - py1) / ph
    rel_cy = (0.5 * (fy1 + fy2) - py1) / ph
    face_h = max(1.0, fy2 - fy1)
    # High-angle outdoor gates: allow slightly lower faces, still block arms.
    top_limit = 0.62 if outdoor else 0.48
    cy_limit = 0.68 if outdoor else 0.50
    if rel_top > top_limit:
        return False, f"face_below_head_band(top={rel_top:.2f})"
    if rel_cy > cy_limit:
        return False, f"face_center_too_low(cy={rel_cy:.2f})"
    # Arm patches are tiny vs the full person box.
    person_area = pw * ph
    face_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
    if face_area / person_area < 0.0008:
        return False, f"face_too_small_vs_person({face_area / person_area:.4f})"
    if face_h > 0.42 * ph:
        return False, f"face_too_tall_vs_person({face_h / ph:.2f})"
    return True, ""


def verify_yunet_face_for_capture(
    face: np.ndarray,
    *,
    min_confidence: float = FACE_LANDMARK_MIN_CONFIDENCE,
    allow_side_view: bool = True,
    max_yaw: float = FACE_MAX_YAW_FOR_SIDE_VIEW,
) -> Tuple[bool, str]:
    """Gate a YuNet face row (15 floats) before crop/save."""
    if face is None or len(face) < 15:
        return False, "invalid_yunet_row"

    x, y, w, h = float(face[0]), float(face[1]), float(face[2]), float(face[3])
    score = float(face[14])
    if w < FACE_MIN_BOX_PX or h < FACE_MIN_BOX_PX:
        return False, f"too_small({int(w)}x{int(h)})"

    ratio = w / h if h > 0 else 0.0
    if ratio < FACE_ASPECT_RATIO_MIN or ratio > FACE_ASPECT_RATIO_MAX:
        return False, f"bad_aspect({ratio:.2f})"

    landmarks = {
        "right_eye": (float(face[4]), float(face[5])),
        "left_eye": (float(face[6]), float(face[7])),
        "nose": (float(face[8]), float(face[9])),
        "mouth_right": (float(face[10]), float(face[11])),
        "mouth_left": (float(face[12]), float(face[13])),
    }
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    return verify_landmarks_for_capture(
        landmarks,
        (x1, y1, x2, y2),
        detection_confidence=score,
        min_confidence=min_confidence,
        allow_side_view=allow_side_view,
        max_yaw=max_yaw,
    )


def yunet_landmarks_dict(face: Union[np.ndarray, list]) -> Dict[str, Tuple[float, float]]:
    """Build canonical landmark dict from a YuNet detection row."""
    return {
        "right_eye": (float(face[4]), float(face[5])),
        "left_eye": (float(face[6]), float(face[7])),
        "nose": (float(face[8]), float(face[9])),
        "mouth_right": (float(face[10]), float(face[11])),
        "mouth_left": (float(face[12]), float(face[13])),
    }
