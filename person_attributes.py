"""
Person-level attribute parsing for live CCTV HUD overlays.

Traits: Gender (Male/Female), Age group (Adult/Child), Clothing
(Shorts/Dress/Suit/Casual). Uses face attribute models on the head crop when
available; clothing is inferred from fast lower-body heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_attributes import UNKNOWN_GENDER, attribute_models_ready, predict_face_attributes

CLOTHING_CHOICES = ("Shorts", "Dress", "Suit", "Casual")
AGE_ADULT = "Adult"
AGE_CHILD = "Child"


@dataclass(frozen=True)
class PersonAttributes:
    gender: str
    age_group: str
    clothing: str

    def hud_lines(self) -> List[str]:
        if self.gender == "Male":
            gender_line = "■ Male"
        elif self.gender == "Female":
            gender_line = "■ Female"
        else:
            gender_line = "■ Unknown"
        return [gender_line, f"■ {self.age_group}", f"■ {self.clothing}"]


def _skin_ratio_bgr(region: np.ndarray) -> float:
    if region is None or region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 28, 50), (25, 180, 255))
    mask2 = cv2.inRange(hsv, (160, 28, 50), (180, 180, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    return float(mask.sum()) / float(max(mask.size * 255, 1))


def _classify_clothing(body_bgr: np.ndarray) -> str:
    h, w = body_bgr.shape[:2]
    if h < 36 or w < 18:
        return "Casual"

    upper = body_bgr[: max(1, int(h * 0.42))]
    lower = body_bgr[max(1, int(h * 0.42)) :]
    lower_skin = _skin_ratio_bgr(lower)

    upper_gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
    lower_gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    upper_mean = float(upper_gray.mean())
    lower_mean = float(lower_gray.mean())
    upper_std = float(upper_gray.std())

    if (
        upper_mean < 88.0
        and lower_mean < 98.0
        and abs(upper_mean - lower_mean) < 28.0
        and upper_std < 48.0
    ):
        return "Suit"

    if lower_skin > 0.11 and (h / max(w, 1)) > 1.45:
        return "Shorts"

    upper_hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    lower_hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    hue_diff = abs(float(upper_hsv[:, :, 0].mean()) - float(lower_hsv[:, :, 0].mean()))
    if hue_diff < 14.0 and upper_std < 58.0 and lower_skin < 0.09:
        return "Dress"

    return "Casual"


def _head_from_body(body_bgr: np.ndarray) -> np.ndarray:
    h = body_bgr.shape[0]
    return body_bgr[: max(1, int(h * 0.45))]


def _parse_gender_age(head_bgr: np.ndarray, body_h: int) -> Tuple[str, str]:
    gender = UNKNOWN_GENDER
    age_group = AGE_ADULT

    if head_bgr is not None and head_bgr.size > 0 and max(head_bgr.shape[:2]) >= 32:
        if attribute_models_ready():
            gender, age = predict_face_attributes(head_bgr)
            if age is not None:
                age_group = AGE_CHILD if age < 18 else AGE_ADULT
            elif gender != UNKNOWN_GENDER:
                age_group = AGE_ADULT
        else:
            gender = UNKNOWN_GENDER

    if body_h < 120:
        age_group = AGE_CHILD
    return gender, age_group


def parse_person_attributes(
    body_bgr: np.ndarray,
    head_bgr: Optional[np.ndarray] = None,
) -> PersonAttributes:
    """Parse gender, age group, and clothing from a person crop (BGR numpy)."""
    if body_bgr is None or body_bgr.size == 0:
        return PersonAttributes(UNKNOWN_GENDER, AGE_ADULT, "Casual")

    head = head_bgr if head_bgr is not None and head_bgr.size else _head_from_body(body_bgr)
    gender, age_group = _parse_gender_age(head, int(body_bgr.shape[0]))
    clothing = _classify_clothing(body_bgr)
    return PersonAttributes(gender, age_group, clothing)


class AttributeHUDCache:
    """Per-track attribute cache — avoids re-parsing every preview frame."""

    def __init__(
        self,
        refresh_frames: int = 20,
        max_entries: int = 256,
    ) -> None:
        self.refresh_frames = refresh_frames
        self.max_entries = max_entries
        self._store: Dict[Tuple[str, int], Tuple[int, PersonAttributes]] = {}

    def get(
        self,
        camera_location: str,
        track_id: int,
        frame_index: int,
        body_bgr: np.ndarray,
        head_bgr: Optional[np.ndarray] = None,
    ) -> PersonAttributes:
        key = (camera_location, int(track_id))
        cached = self._store.get(key)
        if cached is not None:
            last_idx, attrs = cached
            if frame_index - last_idx < self.refresh_frames:
                return attrs

        attrs = parse_person_attributes(body_bgr, head_bgr)
        self._store[key] = (frame_index, attrs)
        if len(self._store) > self.max_entries:
            oldest = min(self._store.items(), key=lambda item: item[1][0])[0]
            self._store.pop(oldest, None)
        return attrs

    def drop_track(self, camera_location: str, track_id: int) -> None:
        self._store.pop((camera_location, int(track_id)), None)
