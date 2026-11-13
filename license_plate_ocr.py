#!/usr/bin/env python3
"""
License plate detection (YOLOv8 plate detector) + OCR (PaddleOCR).

ANPR two-step format (detect plate box, then OCR the crop):
    https://github.com/xavierkoo/computer_vision_anpr_alpr

Install dependencies:
    pip install opencv-python ultralytics paddleocr numpy
    pip install paddlepaddle   # or paddlepaddle-gpu for your CUDA build

Optional EasyOCR fallback:
    pip install easyocr
    set LICENSE_PLATE_OCR=easy

Usage:
    python license_plate_ocr.py path/to/image.jpg
    python license_plate_ocr.py path/to/video.mp4 --video
    python license_plate_ocr.py 0 --realtime              # webcam
    python license_plate_ocr.py --cctv 3 --realtime       # CCTV from cameras.json
    python license_plate_ocr.py --all-cctv --realtime     # ALL cameras from cameras.json
    python license_plate_ocr.py --gates --realtime        # Gate cameras 28, 29, 30
    python license_plate_ocr.py --cctv 28,29,30 --realtime
    python license_plate_ocr.py rtsp://... --realtime --rtsp-tcp

Note:
    Place plate weights at ``license_plate_detector.pt``,
    or set LICENSE_PLATE_MODEL=/path/to/your_weights.pt

    Without custom plate weights, realtime mode uses YOLO vehicle detection
    (car / truck / bus / motorcycle) and OCR on the lower bumper region.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PLATE_MODEL = PROJECT_DIR / "license_plate_detector.pt"
FALLBACK_YOLO_MODEL = "yolov8n.pt"
OCR_LANGUAGES = ("en",)
MIN_PLATE_CONFIDENCE = 0.25
MIN_CROP_SIDE_PX = 28
MIN_PLATE_TEXT_LEN = 4
MIN_OCR_CONFIDENCE = float(os.environ.get("LICENSE_PLATE_MIN_OCR", "0.50"))
PLATE_SAVE_COOLDOWN_SEC = float(os.environ.get("LICENSE_PLATE_COOLDOWN_SEC", "60"))
PLATE_CAMERA_COOLDOWN_SEC = float(
    os.environ.get("LICENSE_PLATE_CAMERA_COOLDOWN_SEC", "25")
)
MIN_VEHICLE_AREA_FRAC = float(os.environ.get("LICENSE_PLATE_MIN_VEHICLE_FRAC", "0.012"))
REALTIME_OCR_EVERY_N_FRAMES = 3
ALL_CCTV_OCR_EVERY_N_FRAMES = 15


def _plate_camera_ids_from_env() -> Tuple[str, ...]:
    """Gate / plate cameras — default outdoor gates including CAM-3."""
    raw = os.environ.get("LICENSE_PLATE_CAMERAS", "3,28,29,30").strip()
    ids = tuple(c.strip() for c in raw.split(",") if c.strip())
    return ids or ("3", "28", "29", "30")


GATE_CAMERA_IDS = _plate_camera_ids_from_env()
ALL_CCTV_START_STAGGER_SEC = 0.4
VEHICLE_CLASS_IDS = frozenset({2, 3, 5, 7})  # car, motorcycle, bus, truck (COCO)
CAMERAS_JSON = PROJECT_DIR / "cameras.json"
PLATE_SAVE_DIR = PROJECT_DIR / "plate_captures"
PLATE_EVENTS_PATH = PROJECT_DIR / "live_preview" / "plate_events.jsonl"
_plate_event_lock = threading.Lock()

_yolo_model = None
_yolo_model_path: Optional[str] = None
_vehicle_yolo_model = None
_ocr_reader = None
_ocr_backend: Optional[str] = None  # "paddle" | "easy"
_ocr_fallback_logged = False
_inference_lock = threading.Lock()
_last_crop_phash: Dict[str, Tuple[str, float]] = {}
_last_cam_save_at: Dict[str, float] = {}
_last_plate_text: Dict[str, Tuple[str, float]] = {}
_vote_lock = threading.Lock()
# Per-camera open ballot: accumulate OCR reads before committing a save.
_pending_plate_votes: Dict[str, Dict[str, Any]] = {}
# Sticky memory of accepted plates so the same car keeps the same ID later.
_known_plate_memory: List[Dict[str, Any]] = []
_KNOWN_PLATE_MEMORY_MAX = 64
_KNOWN_PLATE_MEMORY_SEC = float(
    os.environ.get("LICENSE_PLATE_KNOWN_MEMORY_SEC", "14400")
)  # 4 hours

# Multi-frame voting — wait for a stable reading instead of saving the first guess.
PLATE_VOTE_MIN = max(1, int(os.environ.get("LICENSE_PLATE_VOTE_MIN", "2")))
PLATE_VOTE_WINDOW_SEC = float(os.environ.get("LICENSE_PLATE_VOTE_WINDOW_SEC", "1.6"))
PLATE_VOTE_MAX_WAIT_SEC = float(os.environ.get("LICENSE_PLATE_VOTE_MAX_WAIT_SEC", "2.8"))
LICENSE_PLATE_OCR = (
    os.environ.get("LICENSE_PLATE_OCR", "paddle").strip().lower() or "paddle"
)

# Prefer local go2rtc RTSP restream (no extra sessions on the IP cameras).
LICENSE_PLATE_USE_GO2RTC = os.environ.get("LICENSE_PLATE_USE_GO2RTC", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
GO2RTC_RTSP_BASE = (
    os.environ.get("GO2RTC_RTSP_URL")
    or f"rtsp://127.0.0.1:{os.environ.get('GO2RTC_RTSP_PORT', '8558').strip() or '8558'}"
).rstrip("/")
# Plate OCR runs even outside Settings capture schedule (true realtime).
LICENSE_PLATE_IGNORE_SCHEDULE = os.environ.get(
    "LICENSE_PLATE_IGNORE_SCHEDULE", "1"
).strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Model loaders (lazy singletons — expensive to initialize)
# ---------------------------------------------------------------------------


def _fallback_yolo_weights() -> str:
    """Prefer existing project weights (no internet download)."""
    for name in ("yolov8s.pt", "yolov8l.pt", "yolov8n.pt"):
        path = PROJECT_DIR / name
        if path.is_file():
            return str(path)
    return FALLBACK_YOLO_MODEL


def _resolve_model_path() -> str:
    """Pick custom plate weights when present, else fall back to local YOLO."""
    env_path = os.environ.get("LICENSE_PLATE_MODEL", "").strip()
    if env_path and Path(env_path).is_file():
        return env_path
    if DEFAULT_PLATE_MODEL.is_file():
        return str(DEFAULT_PLATE_MODEL)
    return _fallback_yolo_weights()


def _using_custom_plate_model() -> bool:
    path = _resolve_model_path()
    return Path(path).name not in (
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8l.pt",
        "yolov8m.pt",
    )


def _get_yolo_model():
    """Load Ultralytics YOLO once and reuse (plate detector or COCO fallback)."""
    global _yolo_model, _yolo_model_path
    if _yolo_model is None:
        from ultralytics import YOLO

        model_path = _resolve_model_path()
        _yolo_model_path = model_path
        _yolo_model = YOLO(model_path)
        if _using_custom_plate_model():
            print(
                f"[INFO] Plate detector: {Path(model_path).name} (direct plate boxes).",
                flush=True,
            )
        else:
            print(
                "[INFO] Realtime mode: vehicle bumper OCR fallback "
                "(add license_plate_detector.pt for direct plate detection).",
                file=sys.stderr,
            )
    return _yolo_model


def _get_vehicle_yolo_model():
    """COCO vehicle detector for bumper fallback when plate YOLO finds nothing."""
    global _vehicle_yolo_model
    if _vehicle_yolo_model is None:
        from ultralytics import YOLO

        path = _fallback_yolo_weights()
        _vehicle_yolo_model = YOLO(path)
        print(f"[INFO] Vehicle bumper fallback: {Path(path).name}", flush=True)
    return _vehicle_yolo_model


def _wanted_ocr_backend() -> str:
    raw = (os.environ.get("LICENSE_PLATE_OCR") or LICENSE_PLATE_OCR or "paddle").strip().lower()
    if raw in ("easy", "easyocr"):
        return "easy"
    return "paddle"


def _load_paddle_ocr():
    """Create a PaddleOCR reader (prefer v2 API; tolerate v3 kwargs)."""
    from paddleocr import PaddleOCR

    use_gpu = os.environ.get("PADDLE_OCR_GPU", "0").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    # Disable hoster connectivity check (faster cold start).
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    # Prefer classic PP-OCRv3/v4 English pipeline (stable with paddle 3.0).
    try:
        return PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=use_gpu,
            show_log=False,
            enable_mkldnn=True,
        )
    except TypeError:
        try:
            return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except TypeError:
            return PaddleOCR(
                lang="en",
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )


def _load_easy_ocr():
    import easyocr

    use_gpu = os.environ.get("EASYOCR_GPU", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    return easyocr.Reader(list(OCR_LANGUAGES), gpu=use_gpu)


def _get_ocr_reader():
    """Load OCR backend once (PaddleOCR default, EasyOCR fallback)."""
    global _ocr_reader, _ocr_backend, _ocr_fallback_logged
    if _ocr_reader is not None:
        return _ocr_reader

    want = _wanted_ocr_backend()
    if want == "paddle":
        try:
            _ocr_reader = _load_paddle_ocr()
            _ocr_backend = "paddle"
            print("[INFO] Plate OCR backend: PaddleOCR", flush=True)
            return _ocr_reader
        except Exception as exc:
            if not _ocr_fallback_logged:
                print(
                    f"[WARN] PaddleOCR unavailable ({exc}); falling back to EasyOCR.",
                    file=sys.stderr,
                    flush=True,
                )
                _ocr_fallback_logged = True

    _ocr_reader = _load_easy_ocr()
    _ocr_backend = "easy"
    print("[INFO] Plate OCR backend: EasyOCR", flush=True)
    return _ocr_reader


def _ocr_detect_raw(img: np.ndarray) -> List[Tuple[str, float]]:
    """
    Run the active OCR backend on one image.

    Returns a list of ``(text, confidence)`` fragments (unsorted).
    """
    reader = _get_ocr_reader()
    backend = _ocr_backend or "paddle"
    out: List[Tuple[str, float]] = []

    if backend == "paddle":
        try:
            # PaddleOCR 3.x: predict(); 2.x: ocr(..., cls=True)
            if hasattr(reader, "predict"):
                results = reader.predict(img)
            else:
                results = reader.ocr(img, cls=True)
        except TypeError:
            try:
                results = reader.ocr(img)
            except Exception:
                return out
        except Exception:
            return out

        # Normalize PaddleOCR 2.x / 3.x result shapes into (text, conf).
        for item in results or []:
            if item is None:
                continue
            # 3.x OCRResult / dict-like
            if isinstance(item, dict) or hasattr(item, "get"):
                try:
                    texts = item.get("rec_texts") or item.get("texts") or []
                    scores = item.get("rec_scores") or item.get("scores") or []
                    for text, score in zip(list(texts), list(scores)):
                        out.append((str(text), float(score)))
                    continue
                except Exception:
                    pass
            # 2.x: list of [box, (text, conf)]
            if isinstance(item, (list, tuple)):
                # Whole-page list
                for line in item:
                    try:
                        if (
                            isinstance(line, (list, tuple))
                            and len(line) >= 2
                            and isinstance(line[1], (list, tuple))
                        ):
                            text, conf = line[1][0], float(line[1][1])
                            out.append((str(text), conf))
                        elif isinstance(line, (list, tuple)) and len(line) >= 2:
                            # Already (text, conf)
                            out.append((str(line[0]), float(line[1])))
                    except Exception:
                        continue
        return out

    # EasyOCR
    allowlist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
    try:
        detections = reader.readtext(
            img,
            detail=1,
            paragraph=False,
            allowlist=allowlist,
        )
    except TypeError:
        try:
            detections = reader.readtext(img, detail=1, paragraph=False)
        except Exception:
            return out
    except Exception:
        return out
    for _bbox, text, conf in detections:
        out.append((str(text), float(conf)))
    return out


# ---------------------------------------------------------------------------
# Image pre-processing for OCR
# ---------------------------------------------------------------------------


def preprocess_plate_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Enhance a cropped plate region for OCR.

    Returns the primary grayscale view; use ``preprocess_plate_variants`` for multi-pass OCR.
    """
    variants = preprocess_plate_variants(crop_bgr)
    if not variants:
        raise ValueError("Empty plate crop")
    return variants[0]


def preprocess_plate_variants(crop_bgr: np.ndarray) -> List[np.ndarray]:
    """Build several OCR-friendly views of the same plate crop."""
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return []

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # Upscale small / mid crops — local night plates are often tiny.
    # PaddleOCR needs taller glyphs; push harder than EasyOCR did.
    target_h = 200
    if h < target_h:
        scale = target_h / float(max(1, h))
        gray = cv2.resize(
            gray,
            (max(1, int(w * scale)), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=55, sigmaSpace=55)
    inverted = cv2.bitwise_not(denoised)
    # Dark plate / white glyphs: black-hat emphasizes bright chars on dark bg.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    adaptive_inv = cv2.bitwise_not(adaptive)
    _t, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _ti, otsu_inv = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sharp = cv2.filter2D(denoised, -1, np.array([[0, -1, 0], [-1, 5, -1]], np.float32))

    out: List[np.ndarray] = []
    seen: set[int] = set()
    for img in (
        gray,
        denoised,
        inverted,
        blackhat,
        sharp,
        adaptive,
        adaptive_inv,
        tophat,
        otsu,
        otsu_inv,
    ):
        key = hash(img.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(img)
    return out


def refine_plate_roi(bumper_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Tighten a vehicle bumper crop onto plate-like rectangles.

    Prefers dark plates with bright glyphs (common local private plates) and
    returns up to a few refined crops plus the original bumper band.
    """
    if bumper_bgr is None or getattr(bumper_bgr, "size", 0) == 0:
        return []
    base = np.ascontiguousarray(bumper_bgr)
    h, w = base.shape[:2]
    if min(h, w) < 28:
        return [base]

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Dark rectangle candidates (black/blue plates).
    _, dark_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 18), max(3, h // 14)))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored: List[Tuple[float, Tuple[int, int, int, int]]] = []
    frame_area = float(max(1, h * w))
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 28 or bh < 12:
            continue
        aspect = float(bw) / float(max(1, bh))
        if aspect < 1.6 or aspect > 7.5:
            continue
        area = float(bw * bh)
        if area / frame_area < 0.04 or area / frame_area > 0.95:
            continue
        # Prefer horizontally centered plates of moderate height.
        cx = (x + bw * 0.5) / float(max(1, w))
        cy = (y + bh * 0.5) / float(max(1, h))
        score = area * min(aspect, 4.5)
        score *= 1.0 - abs(cx - 0.5) * 0.6
        score *= 1.0 - abs(cy - 0.55) * 0.4
        # Pad slightly so glyphs are not clipped.
        pad_x = max(2, int(bw * 0.06))
        pad_y = max(2, int(bh * 0.12))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        if (x2 - x1) < MIN_CROP_SIDE_PX or (y2 - y1) < 12:
            continue
        scored.append((score, (x1, y1, x2, y2)))

    scored.sort(key=lambda item: item[0], reverse=True)
    # Tight plate rectangles first — OCR is more accurate on small ROIs.
    out: List[np.ndarray] = []
    for _score, (x1, y1, x2, y2) in scored[:3]:
        crop = base[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        out.append(np.ascontiguousarray(crop))

    # Fixed center strip — often cleaner than the full bumper band.
    sy1 = int(h * 0.18)
    sy2 = int(h * 0.92)
    sx1 = int(w * 0.08)
    sx2 = int(w * 0.92)
    if sy2 - sy1 >= 16 and sx2 - sx1 >= 32:
        strip = base[sy1:sy2, sx1:sx2]
        if strip.size:
            out.append(np.ascontiguousarray(strip))

    out.append(base)
    return out


def clean_plate_text(raw: str) -> str:
    """
    Normalize OCR output to typical plate characters.

    Keeps A–Z, 0–9, and common separators; removes noise characters.
    """
    if not raw:
        return ""
    text = raw.upper()
    text = re.sub(r"[^A-Z0-9\-\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Collapse common OCR confusions in digit positions when isolated.
    text = text.replace("O", "0") if re.fullmatch(r"[0O]+", text.replace(" ", "")) else text
    return text


_BRAND_JUNK_RE = re.compile(
    r"(BIGHORN|ISUZU|TOYOTA|HONDA|NISSAN|HYUNDAI|SUZUKI|MITSUBISHI|FORD|CHEVY|TATA)",
    re.I,
)


def format_local_plate(text: str) -> str:
    """Force common local plate shapes to a stable display form."""
    cleaned = clean_plate_text(text)
    compact = re.sub(r"[^A-Z0-9]", "", cleaned)
    if not compact:
        return ""

    m = re.fullmatch(r"(\d)([A-Z])(\d{4})", compact)
    if m:
        return f"{m.group(1)}{m.group(2)}-{m.group(3)}"

    m = re.fullmatch(r"(\d{2})(\d{4})", compact)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.fullmatch(r"([A-Z]{1,3})(\d{3,5})", compact)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    m = re.fullmatch(r"([A-Z]{2,3})(\d{1,2})([A-Z])(\d{4,5})", compact)
    if m:
        return f"{m.group(1)}-{m.group(2)}{m.group(3)}-{m.group(4)}"
    m = re.fullmatch(r"([A-Z]{2,3})(\d{1,2})(\d{4,5})", compact)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    return cleaned.replace(" ", "-")


def normalize_plate_reading(text: str) -> str:
    """Repair frequent EasyOCR mistakes on local plates, then format."""
    cleaned = clean_plate_text(text)
    compact = re.sub(r"[^A-Z0-9]", "", cleaned)
    if not compact:
        return ""

    # Spurious leading 1 before digit+letter plates (17S7676 → 7S7676).
    if re.fullmatch(r"1(\d[A-Z]\d{4})", compact):
        compact = compact[1:]
    # 1S5870 is almost always 5S-5870 on these cameras.
    if re.fullmatch(r"1S\d{4}", compact):
        compact = "5" + compact[1:]
    if re.fullmatch(r"55S\d{4}", compact):
        compact = compact[1:]
    # Trailing extra digit on otherwise good plates (5S58705 → 5S5870).
    if re.fullmatch(r"(\d[A-Z]\d{4})\d", compact):
        compact = compact[:-1]
    # Leading junk letter before digit-letter plate (B5S5870 rare; B558705 below).

    m = re.fullmatch(r"(\d)([A-Z0-9])(\d{4})", compact)
    if m:
        mid = m.group(2)
        # Only map digit→letter when the mid char is already alphabetic after OCR
        # cleanup — do NOT force 1→I / 0→O (that creates 9I/0I flicker from 91/90).
        if mid.isalpha():
            compact = f"{m.group(1)}{mid}{m.group(3)}"

    compact = re.sub(r"^MD[T0]", "MDY", compact)
    compact = re.sub(r"^M0Y", "MDY", compact)
    return format_local_plate(compact)


def _digit_mid_letter_guesses(compact: str) -> List[str]:
    """Turn 6-digit OCR (213447) into digit-letter plates (2J-3447)."""
    if not re.fullmatch(r"\d{6}", compact):
        return []
    digit_to_letters = {
        "0": "O",
        "1": "JIL",  # J first — common on local plates (2J-3447)
        "2": "Z",
        "5": "S",
        "8": "B",
    }
    d0, d1, rest = compact[0], compact[1], compact[2:]
    out: List[str] = []
    for letter in digit_to_letters.get(d1, ""):
        out.append(f"{d0}{letter}{rest}")
    # Second 5 often means S: 555870 → 5S5870
    if d1 == "5":
        out.insert(0, f"{d0}S{rest}")
    return out


def _extract_shaped_plates(*raw_texts: str) -> List[str]:
    """Regex-hunt local plate shapes inside noisy OCR strings."""
    found: List[str] = []
    cleaned_bits = [clean_plate_text(t) for t in raw_texts if t]
    blobs = [re.sub(r"[^A-Z0-9]", "", t) for t in cleaned_bits]
    blobs.append("".join(blobs))
    for compact in blobs:
        if not compact or len(compact) < 5:
            continue
        # Strongest: digit + letter + 4 digits (9I-7088), optional leading junk 1.
        for m in re.finditer(r"(?:1)?(\d[A-Z]\d{4})", compact):
            found.append(normalize_plate_reading(m.group(1)))
        # Digit-digit + 4 digits (91-7088) when mid letter is misread as 1.
        for m in re.finditer(r"(\d{2}\d{4})", compact):
            found.append(normalize_plate_reading(m.group(1)))
        # Region-style long plates (MDY-73W-62370).
        for m in re.finditer(r"([A-Z]{2,3}\d{1,2}[A-Z]?\d{4,5})", compact):
            found.append(normalize_plate_reading(m.group(1)))
        # Do NOT extract short letter-prefix junk (POG127 / PS125 / E3170) —
        # EasyOCR invents these from Myanmar top-line + bumper noise.
        if re.fullmatch(r"\d{6}", compact):
            for guess in _digit_mid_letter_guesses(compact):
                found.append(normalize_plate_reading(guess))
        # [S-5870] missing leading digit — try 5S.
        m = re.fullmatch(r"S(\d{4})", compact)
        if m:
            found.append(normalize_plate_reading(f"5S{m.group(1)}"))
    # Deduplicate preserve order.
    out: List[str] = []
    seen: set[str] = set()
    for item in found:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _ocr_consensus_score(text: str, raw_bits: Sequence[str]) -> int:
    """How well a candidate matches raw EasyOCR fragments."""
    compact = re.sub(r"[^A-Z0-9]", "", clean_plate_text(text))
    if not compact:
        return 0
    score = 0
    suffix = compact[-4:]
    for raw in raw_bits:
        rc = re.sub(r"[^A-Z0-9]", "", clean_plate_text(raw))
        if not rc:
            continue
        nr = re.sub(r"[^A-Z0-9]", "", normalize_plate_reading(raw) or "")
        if compact == nr:
            score += 45
        if compact in rc or rc in compact:
            score += 20
        if len(suffix) == 4 and rc.endswith(suffix):
            score += 12
        # Letter from OCR present in candidate mid slot.
        if re.fullmatch(r"\d[A-Z]\d{4}", compact) and compact[1] in rc:
            score += 8
        # 555870 strongly supports 5S-5870 (S misread as digit 5).
        m = re.fullmatch(r"(\d)\1(\d{4})", rc)
        if m:
            expected = f"{m.group(1)}S{m.group(2)}"
            if compact == expected:
                score += 80
            elif (
                re.fullmatch(r"\d[A-Z]\d{4}", compact)
                and compact[-4:] == m.group(2)
                and compact != expected
            ):
                score -= 50
        # Prefer dropping leading 1: 17S7676 → 7S7676
        m = re.fullmatch(r"1(\d[A-Z]\d{4})", rc)
        if m and compact == m.group(1):
            score += 50
    return score


def _plate_shape_score(text: str) -> int:
    """Higher = more like a real local plate."""
    compact = re.sub(r"[^A-Z0-9]", "", clean_plate_text(text))
    if not compact:
        return -10
    if _BRAND_JUNK_RE.search(compact):
        return -20
    # Primary local private-car form on these cams: 9I-7088 / 5S-5870 / 2J-3447.
    if re.fullmatch(r"\d[A-Z]\d{4}", compact):
        return 100
    # Region-style long plates — only known local prefixes (not ZZC OCR junk).
    if re.fullmatch(r"[A-Z]{2,3}\d{1,2}[A-Z]?\d{4,5}", compact):
        if compact.startswith(("MD", "YG", "SH", "AY", "SG", "NP", "MG", "KN")):
            return 95
        return -10
    # Mid letter often OCR'd as digit (91-7088 ≈ 9I-7088).
    if re.fullmatch(r"\d{2}\d{4}", compact):
        # Leading-zero digit tails (99-0829) are almost always misreads.
        if compact[-4:].startswith("0"):
            return -5
        return 90
    # Letter-prefix short readings (POG-127, E-3170, PS-125) are bumper junk.
    if re.fullmatch(r"[A-Z]{1,3}\d{3,5}", compact):
        return -5
    letters = sum(ch.isalpha() for ch in compact)
    digits = sum(ch.isdigit() for ch in compact)
    if letters and digits and len(compact) >= 6:
        return 40 + min(digits, 5) + min(letters, 3)
    if digits >= 5 and letters == 0:
        return 25
    return 0


def _plate_fingerprint(text: str) -> str:
    """Normalize plate text for fuzzy cooldown (ignore separators / short OCR flicker)."""
    raw = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    if not raw:
        return ""
    digits = re.sub(r"[^0-9]", "", raw)
    letters = re.sub(r"[^A-Z]", "", raw)
    # Digit + letter + 4 digits (9I-7088 / 9T-7088 / 91-7088 flicker) → same key.
    m = re.fullmatch(r"(\d)[A-Z0-9](\d{4})", raw)
    if m:
        return f"X:{m.group(1)}{m.group(2)}"
    m = re.fullmatch(r"(\d{2})(\d{4})", raw)
    if m:
        return f"X:{m.group(1)[0]}{m.group(2)}"
    if letters and digits:
        return f"M:{letters[:3]}{digits[:4]}"
    # Digit-only bottom lines share a cooldown key (62370 / 6237 / 6234).
    if len(digits) >= 4:
        return f"D:{digits[:5]}"
    return f"R:{raw[:8]}"


def _digit_core(text: str) -> str:
    """Last 4–5 digits used to collapse OCR flicker (7088 / 7038 / 7083)."""
    digits = re.sub(r"[^0-9]", "", (text or "").upper())
    if len(digits) >= 4:
        return digits[-4:]
    return digits


def _cores_soft_match(da: str, db: str) -> bool:
    """True when 4-digit tails look like OCR flicker of the same plate."""
    if len(da) != 4 or len(db) != 4:
        return False
    if da == db:
        return True
    diff = sum(x != y for x, y in zip(da, db))
    # One digit typo anywhere (7088 / 7038).
    if diff <= 1:
        return True
    # Same family prefix (67xx) with ≤2 typos (6761 / 6733).
    if da[:2] == db[:2] and diff <= 2:
        return True
    return False


def _plates_near_duplicate(a: str, b: str) -> bool:
    """True when two readings are likely the same physical plate."""
    ca = re.sub(r"[^A-Z0-9]", "", (a or "").upper())
    cb = re.sub(r"[^A-Z0-9]", "", (b or "").upper())
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    fa, fb = _plate_fingerprint(a), _plate_fingerprint(b)
    if fa and fa == fb:
        return True
    da, db = _digit_core(a), _digit_core(b)
    if len(da) == 4 and len(db) == 4:
        diff = sum(x != y for x, y in zip(da, db))
        # Same leading digit family (9xxx) with ≤2 digit typos (7088/7031/7038).
        if ca[0] == cb[0] and diff <= 2:
            return True
        # Same 4-digit core regardless of letter (9I-7088 / 91-7088).
        if da == db:
            return True
        # Soft core match even when prefix OCR flips (06-6761 / 11-6733).
        if _cores_soft_match(da, db):
            return True
    # Equal-length digit plates with a few OCR swaps (06-6761 / 86-6761).
    if len(ca) == len(cb) == 6 and ca.isdigit() and cb.isdigit():
        if sum(x != y for x, y in zip(ca, cb)) <= 2:
            return True
    return False


def is_plausible_plate_text(text: str, ocr_conf: float) -> bool:
    """Reject OCR hallucinations on pavement / partial motorcycle lines."""
    cleaned = normalize_plate_reading(text) or clean_plate_text(text)
    compact = re.sub(r"[^A-Z0-9]", "", cleaned)
    if len(compact) < MIN_PLATE_TEXT_LEN:
        return False
    if _BRAND_JUNK_RE.search(compact):
        return False
    letters = sum(ch.isalpha() for ch in compact)
    digits = sum(ch.isdigit() for ch in compact)
    # Letter-prefix short junk: POG-127 / PS-125 / ER-121 / E-3170 / EED-184.
    # Real plates on these cameras are digit+letter+4digits (or 6 digits).
    if re.fullmatch(r"[A-Z]{1,3}\d{3,5}", compact):
        return False
    # Trailing letter after a digit run (91786B) — OCR noise, not a plate.
    if re.fullmatch(r"\d{5,}[A-Z]+", compact) or re.fullmatch(r"\d{2,}\d{3,}[A-Z]", compact):
        return False
    # Ambiguous mid-letter plates need stronger OCR than the global floor.
    min_conf = MIN_OCR_CONFIDENCE
    if re.fullmatch(r"\d[A-Z]\d{4}", compact):
        min_conf = max(min_conf, 0.58)
    if ocr_conf < min_conf:
        return False
    # Strong digit-only local form: 91-7088 (mid letter misread as digit).
    if re.fullmatch(r"\d{2}\d{4}", compact):
        if compact[-4:].startswith("0"):
            return False
        # Camera OSD clock: ``12 03:08:00`` → ``12-0308``.
        day = int(compact[:2])
        hh = int(compact[2:4])
        mm = int(compact[4:6])
        if 1 <= day <= 31 and 0 <= hh <= 23 and 0 <= mm <= 59:
            return False
        return ocr_conf >= max(min(MIN_OCR_CONFIDENCE, 0.48), 0.45)
    if letters == 0:
        return digits >= 5 and ocr_conf >= max(MIN_OCR_CONFIDENCE, 0.62)
    if digits == 0:
        return False
    if digits < 4 or len(compact) < 5:
        return False
    # Reject leading-zero "plates" like 0I-7033 (OCR flicker).
    if re.fullmatch(r"0[A-Z]\d{4}", compact):
        return False
    # Fake region codes (ZZC-30-8410) — only allow known local prefixes.
    if re.fullmatch(r"[A-Z]{2,3}\d{1,2}[A-Z]?\d{4,5}", compact) and not compact.startswith(
        ("MD", "YG", "SH", "AY", "SG", "NP", "MG", "KN")
    ):
        return False
    # Two-letter + 4-digit junk (IC-1364 / IP-1284) is almost always bumper OCR noise.
    if re.fullmatch(r"[A-Z]{2}\d{4}", compact) and not compact.startswith(
        ("MD", "YG", "SH", "AY", "SG")
    ):
        return False
    if _plate_shape_score(cleaned) >= 75:
        return True
    if letters >= 2 and digits >= 4 and len(compact) >= 6 and ocr_conf >= 0.75:
        return True
    return False


def plate_crop_looks_valid(crop_bgr: np.ndarray) -> bool:
    """Basic geometry / sharpness gate before OCR (bumper bands are soft)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    h, w = crop_bgr.shape[:2]
    # Distant dedicated plate boxes can be short (wide × short).
    if w < 48 or h < 20:
        return False
    if min(h, w) < 20:
        return False
    # Reject full-frame / scene dumps mistaken for plates (CAM-01 plaza junk).
    if w > 520 or h > 220 or (w * h) > 120_000:
        return False
    # Reject postage-stamp night crops — OCR invents 10-1021 / 90-7083 junk.
    if (w * h) < 2200:
        return False
    aspect = float(w) / float(max(1, h))
    # Plates are usually wider than tall; allow tall motorcycle plates too.
    if aspect < 0.55 or aspect > 8.5:
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Night bumper crops are soft; gate lightly so OCR can still try.
    if sharp < 8.0:
        return False
    # Reject nearly-flat texture (pavement blur).
    if float(np.std(gray)) < 10.0:
        return False
    # Reject IR / headlight blowouts — white glare with no real glyphs
    # (e.g. cam30_57-5575 night Gate-03: mean~200, >35% pixels near white).
    mean = float(np.mean(gray))
    hot_frac = float(np.mean(gray >= 220))
    if mean >= 175.0 and hot_frac >= 0.28:
        return False
    if hot_frac >= 0.45:
        return False
    return True

def _crop_phash(crop_bgr: np.ndarray) -> str:
    """Tiny perceptual hash to skip near-duplicate crops on the same camera."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = "".join("1" if float(v) >= mean else "0" for v in small.flatten())
    return bits


def _hamming(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 999
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


# ---------------------------------------------------------------------------
# Detection + OCR
# ---------------------------------------------------------------------------


def _clip_bbox(
    x1: int, y1: int, x2: int, y2: int, width: int, height: int
) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def detect_plate_boxes(
    image_bgr: np.ndarray,
    *,
    conf: float = MIN_PLATE_CONFIDENCE,
) -> List[Dict[str, Any]]:
    """
    Run YOLOv8 on ``image_bgr`` and return plate bounding boxes.

    Each item: ``{x1, y1, x2, y2, confidence, class_id, class_name}``.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    model = _get_yolo_model()
    results = model.predict(
        source=image_bgr,
        conf=conf,
        verbose=False,
    )
    if not results:
        return []

    height, width = image_bgr.shape[:2]
    boxes: List[Dict[str, Any]] = []

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        names = result.names or {}
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            # Distant CCTV plates are tiny — pad before size gate so OCR gets
            # enough pixels (typical raw box ~25x12 on Gate-03 scenes).
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            pad_x = max(4, int(bw * 0.18))
            pad_y = max(4, int(bh * 0.35))
            x1, y1, x2, y2 = _clip_bbox(
                x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, width, height
            )
            if (x2 - x1) < 20 or (y2 - y1) < 12:
                continue
            class_id = int(box.cls[0].item()) if box.cls is not None else -1
            confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
            boxes.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": str(names.get(class_id, "license_plate")),
                }
            )

    # Highest confidence first.
    boxes.sort(key=lambda b: b["confidence"], reverse=True)
    return boxes


def detect_vehicle_plate_regions(
    image_bgr: np.ndarray,
    *,
    conf: float = 0.40,
) -> List[Dict[str, Any]]:
    """
    Fallback when no custom plate model: detect vehicles and crop the bumper area.

    Emits a few overlapping lower-vehicle bands so angled / distant plates are
    less likely to be missed by a single fixed bumper crop.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    model = _get_vehicle_yolo_model() if _using_custom_plate_model() else _get_yolo_model()
    results = model.predict(source=image_bgr, conf=conf, verbose=False)
    if not results:
        return []

    height, width = image_bgr.shape[:2]
    frame_area = float(max(1, height * width))
    names = results[0].names or {}
    boxes: List[Dict[str, Any]] = []

    # (y0_frac, y1_frac, x_inset) — multiple bumper windows per vehicle.
    bands = (
        (0.52, 0.98, 0.08),
        (0.58, 0.97, 0.12),
        (0.45, 0.88, 0.10),
    )

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].item()) if box.cls is not None else -1
            if class_id not in VEHICLE_CLASS_IDS:
                continue
            xyxy = box.xyxy[0].cpu().numpy()
            vx1, vy1, vx2, vy2 = [int(round(v)) for v in xyxy]
            vh = max(1, vy2 - vy1)
            vw = max(1, vx2 - vx1)
            if (vw * vh) / frame_area < MIN_VEHICLE_AREA_FRAC:
                continue
            if vh < 40 or vw < 40:
                continue
            confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
            for y0f, y1f, xin in bands:
                py1 = vy1 + int(vh * y0f)
                py2 = vy1 + int(vh * y1f)
                px1 = vx1 + int(vw * xin)
                px2 = vx2 - int(vw * xin)
                x1, y1, x2, y2 = _clip_bbox(px1, py1, px2, py2, width, height)
                if (x2 - x1) < MIN_CROP_SIDE_PX or (y2 - y1) < MIN_CROP_SIDE_PX:
                    continue
                boxes.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": f"{names.get(class_id, 'vehicle')}_plate_region",
                    }
                )

    boxes.sort(key=lambda b: b["confidence"], reverse=True)
    # Cap bands so OCR stays realtime (top vehicles × bands).
    return boxes[:6]


def detect_all_plate_candidates(
    image_bgr: np.ndarray,
    *,
    conf: float = MIN_PLATE_CONFIDENCE,
) -> List[Dict[str, Any]]:
    """Prefer dedicated plate boxes; fall back to vehicle bumper regions."""
    height, width = (0, 0)
    if image_bgr is not None and getattr(image_bgr, "size", 0):
        height, width = image_bgr.shape[:2]
    frame_area = float(max(1, height * width))

    def _not_huge(box: Dict[str, Any]) -> bool:
        bw = max(1, int(box["x2"]) - int(box["x1"]))
        bh = max(1, int(box["y2"]) - int(box["y1"]))
        # Plate / bumper ROI must stay a small fraction of the frame.
        if bw > 520 or bh > 280:
            return False
        if (bw * bh) / frame_area > 0.18:
            return False
        return True

    if _using_custom_plate_model():
        boxes = [b for b in detect_plate_boxes(image_bgr, conf=conf) if _not_huge(b)]
        if boxes:
            return boxes
        # Plate model found nothing — bumper OCR as secondary path.
        return [
            b
            for b in detect_vehicle_plate_regions(image_bgr, conf=max(conf, 0.35))
            if _not_huge(b)
        ]
    return [
        b
        for b in detect_vehicle_plate_regions(image_bgr, conf=max(conf, 0.35))
        if _not_huge(b)
    ]


def load_cctv_url(camera_id: str) -> str:
    """Resolve an RTSP URL from ``cameras.json`` by camera id or location."""
    for row in load_all_cctv_cameras():
        rid = str(row.get("id", "")).strip().lower()
        rloc = str(row.get("location", "")).strip().lower()
        needle = str(camera_id).strip().lower()
        if needle in (rid, rloc, f"cam-{rid}"):
            url = str(row.get("source", "")).strip()
            if url:
                return url
    raise KeyError(f"Camera {camera_id!r} not found in {CAMERAS_JSON}")


def load_all_cctv_cameras(camera_ids: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Return camera rows from ``cameras.json``, optionally filtered by id."""
    if not CAMERAS_JSON.is_file():
        raise FileNotFoundError(f"Camera list not found: {CAMERAS_JSON}")
    rows = json.loads(CAMERAS_JSON.read_text(encoding="utf-8"))
    want: Optional[set[str]] = None
    if camera_ids:
        want = {str(c).strip().lower().lstrip("cam-") for c in camera_ids if str(c).strip()}

    out: List[Dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", "")).strip()
        if not source:
            continue
        cam_id = str(row.get("id", row.get("location", ""))).strip()
        if want is not None:
            loc = str(row.get("location", cam_id)).strip().lower()
            cid = cam_id.lower()
            if cid not in want and loc not in want:
                continue
        out.append(
            {
                "id": cam_id,
                "location": str(row.get("location", cam_id)).strip(),
                "source": source,
            }
        )
    return out


def parse_cctv_ids(raw: str) -> List[str]:
    """Parse ``28,29,30`` or ``CAM-28 CAM-29`` into id strings."""
    parts = re.split(r"[\s,;]+", str(raw or "").strip())
    return [p.strip().lstrip("cam-").lstrip("CAM-") for p in parts if p.strip()]


# Fail unreachable RTSP hosts in ~1–4s instead of OpenCV's default ~30s stall.
_PLATE_TCP_PROBE_SEC = float(os.environ.get("PLATE_RTSP_TCP_PROBE_SEC", "1.2"))
_PLATE_OPEN_TIMEOUT_MS = int(os.environ.get("PLATE_RTSP_OPEN_TIMEOUT_MS", "3500"))
_PLATE_HOST_COOLDOWN_SEC = float(os.environ.get("PLATE_HOST_COOLDOWN_SEC", "45"))
_plate_host_dead_until: Dict[str, float] = {}
_plate_host_dead_lock = threading.Lock()


def _rtsp_host_port(source: str) -> Optional[Tuple[str, int]]:
    try:
        parsed = urlparse(source)
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


def _host_cooldown_key(source: str) -> str:
    endpoint = _rtsp_host_port(source)
    if endpoint is None:
        return str(source or "").strip()
    return f"{endpoint[0]}:{endpoint[1]}"


def _mark_host_unreachable(source: str, *, seconds: Optional[float] = None) -> None:
    key = _host_cooldown_key(source)
    if not key:
        return
    cool = float(seconds if seconds is not None else _PLATE_HOST_COOLDOWN_SEC)
    with _plate_host_dead_lock:
        _plate_host_dead_until[key] = time.time() + max(5.0, cool)


def _host_in_cooldown(source: str) -> bool:
    key = _host_cooldown_key(source)
    if not key:
        return False
    with _plate_host_dead_lock:
        until = float(_plate_host_dead_until.get(key) or 0.0)
    return until > time.time()


def _ensure_rtsp_host_reachable(source: str, timeout: float = _PLATE_TCP_PROBE_SEC) -> None:
    """TCP probe before OpenCV open — dead cams fail in ~1s, not ~30s."""
    if _host_in_cooldown(source):
        endpoint = _rtsp_host_port(source)
        where = f"{endpoint[0]}:{endpoint[1]}" if endpoint else source
        raise RuntimeError(
            f"Could not open video source: {source} "
            f"(host {where} recently unreachable — cooling down)"
        )
    endpoint = _rtsp_host_port(source)
    if endpoint is None:
        return
    host, port = endpoint
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as exc:
        _mark_host_unreachable(source)
        raise RuntimeError(
            f"Could not open video source: {source} "
            f"(host {host}:{port} unreachable within {timeout:.1f}s: {exc})"
        ) from exc


def open_video_capture(
    source: Union[int, str],
    *,
    rtsp_tcp: bool = True,
) -> cv2.VideoCapture:
    """Open webcam, file, or RTSP with low-latency settings.

    Network streams get a fast TCP probe + short FFmpeg timeouts so offline
    cameras fail in ~1–4s instead of OpenCV's default ~30s stall.
    """
    src: Any = int(source) if isinstance(source, str) and source.isdigit() else source
    url = str(src)

    if isinstance(src, str) and url.lower().startswith(("rtsp://", "http://", "https://")):
        _ensure_rtsp_host_reachable(url)
        sock_us = max(1_000_000, _PLATE_OPEN_TIMEOUT_MS * 1000)
        opts = [
            "fflags;nobuffer",
            "flags;low_delay",
            "max_delay;0",
            f"stimeout;{sock_us}",
            f"rw_timeout;{sock_us}",
        ]
        if rtsp_tcp and url.lower().startswith("rtsp://"):
            opts.insert(0, "rtsp_transport;tcp")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)

        cap = cv2.VideoCapture()
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(_PLATE_OPEN_TIMEOUT_MS))
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(_PLATE_OPEN_TIMEOUT_MS))
        except Exception:
            pass
        if not cap.open(url, cv2.CAP_FFMPEG) or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            raise RuntimeError(f"Could not open video source: {source}")
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    if url.lower().startswith("rtsp://") and rtsp_tcp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def read_plate_text(crop_bgr: np.ndarray) -> Tuple[str, float]:
    """
    OCR a cropped plate image.

    Returns ``(cleaned_text, best_confidence)``. Empty text when unreadable.
    Runs several preprocess variants and prefers local plate shapes
    (``7S-7676``, ``MDY-73W-62370``, …) so the first save is the real number.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0
    if not plate_crop_looks_valid(crop_bgr):
        return "", 0.0

    _get_ocr_reader()
    # Dedicated plate boxes are already tight — light ROI refine only.
    # Bumper fallback still benefits from dark-rectangle tightening first.
    if _using_custom_plate_model():
        roi_list = [np.ascontiguousarray(crop_bgr)]
        # Small center pad strip as a second view when glare clips edges.
        h, w = crop_bgr.shape[:2]
        if h >= 20 and w >= 40:
            sy1, sy2 = int(h * 0.05), int(h * 0.95)
            sx1, sx2 = int(w * 0.04), int(w * 0.96)
            strip = crop_bgr[sy1:sy2, sx1:sx2]
            if strip.size:
                roi_list.append(np.ascontiguousarray(strip))
    else:
        roi_list = refine_plate_roi(crop_bgr)
    # Deduplicate near-identical sizes while preserving order.
    crops: List[np.ndarray] = []
    seen_shapes: set[Tuple[int, int]] = set()
    for roi in roi_list:
        key = (int(roi.shape[0]), int(roi.shape[1]))
        if key in seen_shapes and len(crops) >= 2:
            continue
        seen_shapes.add(key)
        crops.append(roi)
    if not crops:
        crops = [crop_bgr]

    raw_bits: List[str] = []
    candidates: List[Tuple[str, float]] = []

    for crop in crops[:4]:
        variants = preprocess_plate_variants(crop)
        if not variants:
            continue
        for img in variants[:6]:
            detections = _ocr_detect_raw(img)
            if not detections:
                continue

            line_parts: List[Tuple[str, float]] = []
            for text, conf in detections:
                cleaned = clean_plate_text(text)
                if not cleaned or float(conf) < 0.20:
                    continue
                raw_bits.append(cleaned)
                normalized = normalize_plate_reading(cleaned)
                if normalized:
                    candidates.append((normalized, float(conf)))
                else:
                    candidates.append((cleaned, float(conf)))
                line_parts.append((cleaned, float(conf)))

            if len(line_parts) >= 2:
                merged_raw = "".join(part[0] for part in line_parts[:3])
                merged_conf = float(sum(p[1] for p in line_parts[:2]) / 2.0)
                for shaped in _extract_shaped_plates(merged_raw, *[p[0] for p in line_parts]):
                    candidates.append((shaped, merged_conf))

    # Hunt strong plate-shaped substrings across noisy OCR fragments.
    for shaped in _extract_shaped_plates(*raw_bits):
        # Only promote strong shapes; never invent letter-prefix junk at 0.55.
        if _plate_shape_score(shaped) < 75:
            continue
        candidates.append((shaped, 0.55))

    if not candidates:
        return "", 0.0

    scored: List[Tuple[int, int, float, int, str]] = []
    best_conf: Dict[str, float] = {}
    for text, conf in candidates:
        norm = normalize_plate_reading(text) or clean_plate_text(text)
        if not norm:
            continue
        if _plate_shape_score(norm) < 40:
            continue
        prev = best_conf.get(norm)
        if prev is None or float(conf) > prev:
            best_conf[norm] = float(conf)
    for norm, conf in best_conf.items():
        scored.append(
            (
                _plate_shape_score(norm),
                _ocr_consensus_score(norm, raw_bits),
                float(conf),
                len(re.sub(r"[^A-Z0-9]", "", norm)),
                norm,
            )
        )
    if not scored:
        return "", 0.0
    # Consensus beats inflated letter-guess shape (91-7088 from 917088 wins
    # over speculative 9J/9I/9L guesses).
    scored.sort(
        key=lambda item: (item[0] + item[1], item[1], item[2], item[3]),
        reverse=True,
    )

    # Local OCR often reads 5S as 55 (555870). Same digit twice → digit+S+####.
    scored_texts = {row[-1] for row in scored}
    for raw in raw_bits:
        rc = re.sub(r"[^A-Z0-9]", "", clean_plate_text(raw))
        m = re.fullmatch(r"(\d)\1(\d{4})", rc)
        if not m:
            continue
        forced = normalize_plate_reading(f"{m.group(1)}S{m.group(2)}")
        if forced and forced in scored_texts and is_plausible_plate_text(forced, 0.58):
            return forced, max(0.58, MIN_OCR_CONFIDENCE)

    for shape, consensus, conf, _n, text in scored:
        if shape < 75:
            break
        compact = re.sub(r"[^A-Z0-9]", "", text)
        suffix = compact[-4:] if len(compact) >= 4 else ""
        # Require the digit tail to actually appear in raw OCR — blocks
        # invented plates like 7C-3127 when the bumper really shows 7088.
        suffix_hit = False
        soft_hit = False
        if suffix:
            for r in raw_bits:
                rc = re.sub(r"[^A-Z0-9]", "", clean_plate_text(r))
                if not rc:
                    continue
                if suffix in rc or rc.endswith(suffix):
                    suffix_hit = True
                    break
                if len(rc) >= 4:
                    overlap = sum(a == b for a, b in zip(suffix, rc[-4:]))
                    if overlap >= 3:
                        soft_hit = True
        if consensus <= 0 and conf <= 0.60 and not suffix_hit and not soft_hit:
            continue
        if not suffix_hit and not soft_hit and consensus < 12:
            continue
        gate_conf = conf
        if is_plausible_plate_text(text, gate_conf):
            return text, conf
        # High raw consensus may lift confidence to the configured floor once.
        lift = max(conf, MIN_OCR_CONFIDENCE)
        if consensus >= 40:
            lift = max(lift, MIN_OCR_CONFIDENCE, 0.55)
        if (suffix_hit or consensus >= 12) and is_plausible_plate_text(text, lift):
            return text, lift
    return "", 0.0



def draw_plate_results(
    image_bgr: np.ndarray,
    plates: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw bounding boxes and recognized text on a copy of the image."""
    output = image_bgr.copy()
    for plate in plates:
        x1, y1, x2, y2 = plate["x1"], plate["y1"], plate["x2"], plate["y2"]
        text = plate.get("text") or "?"
        conf = plate.get("ocr_confidence", 0.0)
        det_conf = plate.get("confidence", 0.0)

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{text} ({conf:.0%})" if text and text != "?" else f"No text ({det_conf:.0%})"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1 - 8, th + 4)
        cv2.rectangle(output, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), (0, 255, 0), -1)
        cv2.putText(
            output,
            label,
            (x1 + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def detect_and_read_plate(image_path: str) -> Dict[str, Any]:
    """
    Detect license plates in an image and read their text.

    Parameters
    ----------
    image_path :
        Path to an image file readable by OpenCV.

    Returns
    -------
    dict
        {
            "success": bool,
            "image_path": str,
            "plates": [
                {
                    "x1", "y1", "x2", "y2",
                    "confidence",        # YOLO detection score
                    "text",              # cleaned OCR string (may be empty)
                    "ocr_confidence",    # EasyOCR score
                    "error",             # per-plate OCR note, if any
                },
                ...
            ],
            "annotated_image": np.ndarray | None,
            "error": str | None,         # top-level error message
        }
    """
    path = Path(image_path)
    result: Dict[str, Any] = {
        "success": False,
        "image_path": str(path),
        "plates": [],
        "annotated_image": None,
        "error": None,
    }

    if not path.is_file():
        result["error"] = f"File not found: {path}"
        return result

    image = cv2.imread(str(path))
    if image is None:
        result["error"] = f"Could not decode image: {path}"
        return result

    try:
        boxes = detect_all_plate_candidates(image)
    except Exception as exc:
        result["error"] = f"Detection failed: {exc}"
        return result

    if not boxes:
        result["error"] = "No license plate detected."
        result["annotated_image"] = image.copy()
        return result

    plates_out: List[Dict[str, Any]] = []
    for box in boxes:
        crop = image[box["y1"] : box["y2"], box["x1"] : box["x2"]]
        plate_entry = dict(box)
        plate_entry["text"] = ""
        plate_entry["ocr_confidence"] = 0.0
        plate_entry["error"] = None

        try:
            text, ocr_conf = read_plate_text(crop)
            plate_entry["text"] = text
            plate_entry["ocr_confidence"] = ocr_conf
            if not text:
                plate_entry["error"] = "Plate detected but text could not be read."
        except Exception as exc:
            plate_entry["error"] = f"OCR failed: {exc}"

        plates_out.append(plate_entry)

    result["plates"] = plates_out
    result["annotated_image"] = draw_plate_results(image, plates_out)
    result["success"] = any(p.get("text") for p in plates_out)
    if not result["success"]:
        result["error"] = "Plate(s) detected but no readable text found."
    return result


def _plate_captures_allowed() -> bool:
    """Honor Settings → Capture schedule (unless LICENSE_PLATE_IGNORE_SCHEDULE=1)."""
    if LICENSE_PLATE_IGNORE_SCHEDULE:
        return True
    try:
        from announcement_settings import captures_allowed_now

        return bool(captures_allowed_now())
    except Exception:
        return True


def resolve_plate_stream_source(camera: Dict[str, str]) -> str:
    """
    Prefer go2rtc local RTSP so plate OCR reuses the already-open camera restream
    instead of opening a second session on the Dahua/Hikvision camera.
    """
    cam_id = str(camera.get("id", camera.get("location", ""))).strip()
    direct = str(camera.get("source", "")).strip()
    if not LICENSE_PLATE_USE_GO2RTC or not cam_id:
        return direct
    # High-res plate restream from nvr/go2rtc.yaml (plate_cam*).
    return f"{GO2RTC_RTSP_BASE}/plate_cam{cam_id}"


def _append_plate_event(event: Dict[str, Any]) -> None:
    """Append one plate hit for the dashboard recent-plates feed."""
    try:
        PLATE_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False)
        with _plate_event_lock:
            with PLATE_EVENTS_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except OSError:
        pass


def _crop_sharpness(crop_bgr: np.ndarray) -> float:
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _vote_identity_key(text: str, phash: str) -> str:
    """Group OCR flicker of the same car (digit core + coarse visual hash)."""
    core = _digit_core(text)
    visual = (phash or "")[:32]
    if core and len(core) >= 4:
        return f"C:{core[0]}:{core}"
    return f"V:{visual}"


def _remember_known_plate(camera_id: str, text: str, phash: str, conf: float) -> None:
    """Remember an accepted plate so later OCR flicker maps back to it."""
    now = time.time()
    entry = {
        "camera": str(camera_id or ""),
        "text": text,
        "phash": phash,
        "conf": float(conf),
        "ts": now,
        "shape": _plate_shape_score(text),
    }
    with _vote_lock:
        _known_plate_memory[:] = [
            row
            for row in _known_plate_memory
            if now - float(row.get("ts") or 0) < _KNOWN_PLATE_MEMORY_SEC
        ]
        _known_plate_memory.insert(0, entry)
        del _known_plate_memory[_KNOWN_PLATE_MEMORY_MAX:]


def _seed_known_plates_from_disk(save_root: Optional[Path] = None) -> int:
    """Load recent strong plate saves into sticky memory (survives worker restart)."""
    root = save_root or PLATE_SAVE_DIR
    if not root.is_dir():
        return 0
    seeded = 0
    files = sorted(root.glob("cam*_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    # Prefer majority text per soft-core family so flicker IDs don't seed as equals.
    family_counts: Dict[str, Dict[str, int]] = {}
    parsed: List[Tuple[str, str, Path]] = []
    for path in files:
        if path.name.endswith("_scene.jpg"):
            continue
        m = re.match(r"cam([^_]+)_(.+?)_\d{8}_\d{6}\.jpg$", path.name, re.I)
        if not m:
            continue
        cam_id, text = m.group(1), m.group(2)
        text = normalize_plate_reading(text) or clean_plate_text(text)
        if not text or _plate_shape_score(text) < 90:
            continue
        age = time.time() - path.stat().st_mtime
        if age > _KNOWN_PLATE_MEMORY_SEC:
            continue
        core = _digit_core(text) or text
        fam_key = f"{cam_id}:{core[:2] if len(core) >= 2 else core}"
        bucket = family_counts.setdefault(fam_key, {})
        bucket[text] = bucket.get(text, 0) + 1
        parsed.append((cam_id, text, path))

    majority_for: Dict[str, str] = {}
    for fam_key, bucket in family_counts.items():
        majority_for[fam_key] = max(
            bucket.items(), key=lambda kv: (kv[1], _plate_shape_score(kv[0]))
        )[0]

    for cam_id, text, path in parsed:
        core = _digit_core(text) or text
        fam_key = f"{cam_id}:{core[:2] if len(core) >= 2 else core}"
        # Collapse soft-family flicker to the majority label while seeding.
        canon = majority_for.get(fam_key, text)
        if _cores_soft_match(_digit_core(canon), _digit_core(text)):
            text = canon
        try:
            img = cv2.imread(str(path))
            if img is None:
                continue
            phash = _crop_phash(img)
        except Exception:
            continue
        _remember_known_plate(cam_id, text, phash, 0.7)
        seeded += 1
        if seeded >= 24:
            break
    if seeded:
        print(f"[PLATE] Seeded {seeded} known plate ID(s) from disk", flush=True)
    return seeded


def _canonical_known_plate(
    camera_id: str, text: str, phash: str
) -> Optional[str]:
    """
    If this looks like a car we already ID'd correctly, reuse that plate text.

    Stops 9I-7088 from coming back as 91-7028 / 9C-3282 hours later.
    Also collapses soft OCR flicker (06-6761 ↔ 11-6733) to the majority ID.
    """
    now = time.time()
    cam = str(camera_id or "")
    with _vote_lock:
        memory = list(_known_plate_memory)

    # Frequency of each known text on this camera (majority wins flicker wars).
    freq: Dict[str, int] = {}
    for row in memory:
        if now - float(row.get("ts") or 0) >= _KNOWN_PLATE_MEMORY_SEC:
            continue
        row_cam = str(row.get("camera") or "")
        if cam and row_cam not in ("", cam):
            continue
        row_text = str(row.get("text") or "")
        if row_text:
            freq[row_text] = freq.get(row_text, 0) + 1

    # Soft-core family on this camera → always prefer the majority plate text.
    in_core = _digit_core(text)
    family: Dict[str, int] = {}
    for row_text, count in freq.items():
        if _cores_soft_match(in_core, _digit_core(row_text)):
            family[row_text] = family.get(row_text, 0) + count
    if family:
        majority = max(family.items(), key=lambda kv: (kv[1], _plate_shape_score(kv[0])))[0]
        if majority and majority != text:
            return majority
        # Even if OCR already matches majority, keep it (no downgrade).
        if majority == text and family[majority] >= 2:
            return majority

    best_text = None
    best_score = -1.0
    for row in memory:
        if now - float(row.get("ts") or 0) >= _KNOWN_PLATE_MEMORY_SEC:
            continue
        if cam and str(row.get("camera") or "") not in ("", cam):
            same_cam = False
        else:
            same_cam = True
        row_text = str(row.get("text") or "")
        if not row_text:
            continue
        near = _plates_near_duplicate(row_text, text)
        ham = _hamming(str(row.get("phash") or ""), phash)
        soft_core = _cores_soft_match(_digit_core(row_text), _digit_core(text))
        # Bumper-band crops of a parked car can hash very differently (~70).
        # Same-camera soft core still maps flicker back (06-6761 ↔ 11-6733).
        if not near and ham > 34 and not (same_cam and soft_core):
            continue
        shape = float(row.get("shape") or 0)
        score = shape + (20.0 if same_cam else 0.0) + (15.0 if near else 0.0)
        score += max(0.0, 34.0 - float(ham))
        if same_cam and soft_core:
            score += 40.0
        # Majority known ID beats exact self-match of a rare flicker.
        score += 14.0 * float(freq.get(row_text, 1))
        if row_text == text:
            score -= 10.0
        # Strong known IDs win even when OCR text is unrelated junk (10-9431).
        if shape >= 90 and (ham <= 34 or (same_cam and soft_core)):
            score += 25.0
        if score > best_score:
            best_score = score
            best_text = row_text
    return best_text


def _pick_voted_plate(
    votes: Sequence[Dict[str, Any]],
) -> Optional[Tuple[str, float, np.ndarray, Optional[np.ndarray], float]]:
    """
    Choose the most stable plate text from multi-frame OCR votes.

    Prefers a shared 4-digit core, then digit+letter form (9I-7088) when the
    mid letter appeared in votes; otherwise digit-digit (91-7088).
    """
    if not votes:
        return None

    # Weight votes by confidence + shape.
    core_weights: Dict[str, float] = {}
    text_weights: Dict[str, float] = {}
    letter_mid_weights: Dict[str, Dict[str, float]] = {}
    best_by_text: Dict[str, Dict[str, Any]] = {}

    for vote in votes:
        text = str(vote.get("text") or "")
        conf = float(vote.get("conf") or 0.0)
        if not text:
            continue
        shape = float(_plate_shape_score(text))
        weight = max(0.05, conf) * (1.0 + shape / 100.0)
        core = _digit_core(text)
        if core:
            core_weights[core] = core_weights.get(core, 0.0) + weight
        text_weights[text] = text_weights.get(text, 0.0) + weight
        compact = re.sub(r"[^A-Z0-9]", "", text)
        m = re.fullmatch(r"(\d)([A-Z])(\d{4})", compact)
        if m:
            key = f"{m.group(1)}|{m.group(3)}"
            bucket = letter_mid_weights.setdefault(key, {})
            bucket[m.group(2)] = bucket.get(m.group(2), 0.0) + weight
        prev = best_by_text.get(text)
        if prev is None or conf >= float(prev.get("conf") or 0.0):
            best_by_text[text] = vote

    if not core_weights and not text_weights:
        return None

    winning_core = ""
    if core_weights:
        winning_core = max(core_weights.items(), key=lambda kv: kv[1])[0]

    # Build preferred display text for the winning core.
    chosen = ""
    if winning_core:
        # Prefer digit+letter if we saw a letter mid for this core.
        for key, letters in letter_mid_weights.items():
            lead, core = key.split("|", 1)
            if core != winning_core:
                continue
            mid = max(letters.items(), key=lambda kv: kv[1])[0]
            chosen = normalize_plate_reading(f"{lead}{mid}{core}") or f"{lead}{mid}-{core}"
            break
        if not chosen:
            # Prefer exact texts whose digit core matches.
            matching = [
                (text_weights[t], t)
                for t in text_weights
                if _digit_core(t) == winning_core
            ]
            if matching:
                matching.sort(reverse=True)
                chosen = matching[0][1]
            else:
                lead = "9"
                for vote in votes:
                    compact = re.sub(r"[^A-Z0-9]", "", str(vote.get("text") or ""))
                    if compact and compact[0].isdigit():
                        lead = compact[0]
                        break
                chosen = normalize_plate_reading(f"{lead}1{winning_core}") or (
                    f"{lead}1-{winning_core}"
                )

    if not chosen:
        chosen = max(text_weights.items(), key=lambda kv: kv[1])[0]

    # Best crop: sharpest among votes for this core / text family.
    pool = [
        v
        for v in votes
        if _plates_near_duplicate(str(v.get("text") or ""), chosen)
        or (_digit_core(str(v.get("text") or "")) == winning_core and winning_core)
    ] or list(votes)
    best_vote = max(
        pool,
        key=lambda v: (
            float(v.get("sharp") or 0.0),
            float(v.get("conf") or 0.0),
        ),
    )
    conf = float(best_vote.get("conf") or 0.0)
    agree = sum(
        1
        for v in votes
        if _plates_near_duplicate(str(v.get("text") or ""), chosen)
        or _digit_core(str(v.get("text") or "")) == winning_core
    )
    conf = min(0.99, conf + 0.05 * max(0, agree - 1))
    crop = best_vote.get("crop")
    frame = best_vote.get("frame")
    det = float(best_vote.get("det") or 0.0)
    if crop is None:
        return None
    return chosen, conf, crop, frame, det


def _vote_and_maybe_save(
    text: str,
    ocr_conf: float,
    crop: np.ndarray,
    *,
    camera_id: str,
    save_root: Path,
    seen_plates: Dict[str, float],
    seen_lock: Optional[threading.Lock] = None,
    cooldown_sec: float = PLATE_SAVE_COOLDOWN_SEC,
    det_confidence: float = 0.0,
    frame_bgr: Optional[np.ndarray] = None,
    log_hits: bool = True,
) -> Optional[Path]:
    """
    Buffer OCR reads per camera and save only when the reading stabilizes.

    Fast cars still save after ``PLATE_VOTE_MAX_WAIT_SEC`` using the best
    consensus so far (better than committing the first noisy frame).
    """
    text = normalize_plate_reading(text) or clean_plate_text(text)
    if not text or not is_plausible_plate_text(text, ocr_conf):
        return None
    if not plate_crop_looks_valid(crop):
        return None

    cam_key = str(camera_id or "cam")
    now = time.time()
    phash = _crop_phash(crop)
    ident = _vote_identity_key(text, phash)
    # Keep a compact scene snapshot only (full-res copies would thrash RAM).
    scene_snap = None
    if frame_bgr is not None and getattr(frame_bgr, "size", 0):
        try:
            h, w = frame_bgr.shape[:2]
            max_w = 960
            if w > max_w:
                scale = max_w / float(w)
                scene_snap = cv2.resize(
                    frame_bgr,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                scene_snap = frame_bgr.copy()
        except Exception:
            scene_snap = None

    vote = {
        "text": text,
        "conf": float(ocr_conf),
        "det": float(det_confidence),
        "crop": crop.copy(),
        "frame": scene_snap,
        "phash": phash,
        "sharp": _crop_sharpness(crop),
        "ts": now,
        "ident": ident,
    }

    ready: Optional[Dict[str, Any]] = None
    with _vote_lock:
        ballot = _pending_plate_votes.get(cam_key)
        if ballot is not None:
            age = now - float(ballot.get("started") or now)
            same_car = False
            for prev in ballot.get("votes") or []:
                prev_text = str(prev.get("text") or "")
                if _plates_near_duplicate(prev_text, text):
                    same_car = True
                    break
                if _hamming(str(prev.get("phash") or ""), phash) <= 30:
                    same_car = True
                    break
            if not same_car:
                # Different vehicle arrived — flush prior ballot first.
                if ballot.get("votes"):
                    ready = dict(ballot)
                ballot = None
            elif age > PLATE_VOTE_MAX_WAIT_SEC * 2:
                # Stale ballot — start clean.
                ballot = None

        if ballot is None:
            ballot = {"started": now, "votes": [], "ident": ident}
            _pending_plate_votes[cam_key] = ballot

        ballot["votes"].append(vote)
        votes = list(ballot["votes"])
        age = now - float(ballot.get("started") or now)
        cores = [_digit_core(str(v.get("text") or "")) for v in votes]
        cores = [c for c in cores if len(c) == 4]
        majority = 0
        if cores:
            top_core = max(set(cores), key=cores.count)
            majority = cores.count(top_core)

        should_commit = False
        if len(votes) >= PLATE_VOTE_MIN and majority >= max(2, (PLATE_VOTE_MIN + 1) // 2):
            should_commit = True
        elif age >= PLATE_VOTE_WINDOW_SEC and len(votes) >= 2 and majority >= 2:
            should_commit = True
        elif age >= PLATE_VOTE_MAX_WAIT_SEC and len(votes) >= 1:
            should_commit = True

        commit_ballot = ready
        if should_commit:
            commit_ballot = dict(ballot)
            _pending_plate_votes.pop(cam_key, None)
        elif ready is not None:
            # Keep only the new car's current vote after flushing the old ballot.
            _pending_plate_votes[cam_key] = {
                "started": now,
                "votes": [vote],
                "ident": ident,
            }

    if commit_ballot is None:
        return None

    picked = _pick_voted_plate(commit_ballot.get("votes") or [])
    if picked is None:
        return None
    win_text, win_conf, win_crop, win_frame, win_det = picked
    win_phash = _crop_phash(win_crop)
    canonical = _canonical_known_plate(camera_id, win_text, win_phash)
    if canonical and canonical != win_text:
        print(
            f"[PLATE] CAM-{camera_id} corrected {win_text} -> {canonical} (known plate)",
            flush=True,
        )
        win_text = canonical
    saved = _save_plate_crop(
        win_text,
        win_conf,
        win_crop,
        camera_id=camera_id,
        save_root=save_root,
        seen_plates=seen_plates,
        seen_lock=seen_lock,
        cooldown_sec=cooldown_sec,
        det_confidence=win_det,
        frame_bgr=win_frame,
    )
    if saved:
        _remember_known_plate(camera_id, win_text, win_phash, win_conf)
    if saved and log_hits:
        n = len(commit_ballot.get("votes") or [])
        cam_label = f"CAM-{camera_id}" if camera_id else "CAM"
        print(
            f"[PLATE] {cam_label} {win_text}  ocr={win_conf:.0%}  "
            f"votes={n}  saved={saved.name}",
            flush=True,
        )
    return saved


def _save_plate_crop(
    text: str,
    ocr_conf: float,
    crop: np.ndarray,
    *,
    camera_id: str,
    save_root: Path,
    seen_plates: Dict[str, float],
    seen_lock: Optional[threading.Lock] = None,
    cooldown_sec: float = PLATE_SAVE_COOLDOWN_SEC,
    det_confidence: float = 0.0,
    frame_bgr: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """Save a plate crop (+ optional scene) once per camera+text within cooldown."""
    if not _plate_captures_allowed():
        return None
    text = normalize_plate_reading(text) or clean_plate_text(text)
    if not text or not is_plausible_plate_text(text, ocr_conf):
        return None
    if not plate_crop_looks_valid(crop):
        return None
    cam_key = str(camera_id or "cam")
    fp = _plate_fingerprint(text)
    key = f"{cam_key}:{fp}" if fp else f"{cam_key}:{text}"
    now = time.time()
    phash = _crop_phash(crop)

    def _claim() -> bool:
        # Per-camera rate limit (stops OCR flicker floods).
        last_cam = _last_cam_save_at.get(cam_key, 0.0)
        if now - last_cam < PLATE_CAMERA_COOLDOWN_SEC:
            return False
        last = seen_plates.get(key, 0.0)
        if now - last < cooldown_sec:
            return False
        # Near-identical crop already saved recently on this camera.
        prev = _last_crop_phash.get(cam_key)
        if prev is not None:
            prev_hash, prev_ts = prev
            # Bumper crops of the same car flicker a lot — be aggressive.
            if now - prev_ts < cooldown_sec and _hamming(prev_hash, phash) <= 36:
                return False
        # Same physical plate, OCR letter/digit flicker (9I-7088 / 9T-7088 / 0I-7033).
        for existing, ts in list(seen_plates.items()):
            if not existing.startswith(f"{cam_key}:"):
                continue
            if now - float(ts) >= cooldown_sec:
                continue
            other_fp = existing.split(":", 1)[-1]
            # Fingerprint match (X:97088) or digit-core near-duplicate.
            if other_fp == fp:
                return False
            other_text = existing[len(cam_key) + 1 :]
            # seen key stores fingerprint; also check recent saved texts via phash window.
            if fp.startswith("X:") and other_fp.startswith("X:"):
                if fp[2:3] == other_fp[2:3] and _hamming(
                    fp[3:].ljust(4, "0")[:4], other_fp[3:].ljust(4, "0")[:4]
                ) <= 1:
                    return False
            # Soft core on digit fingerprints (D:6761 vs D:6733).
            if fp.startswith("D:") and other_fp.startswith("D:"):
                if _cores_soft_match(fp[2:6], other_fp[2:6]):
                    return False
        # Block near-neighbor OCR flicker keys (6237 / 62370 / 6234).
        if fp.startswith("D:") and len(fp) >= 4:
            stem = fp[:6]
            for existing, ts in list(seen_plates.items()):
                if not existing.startswith(f"{cam_key}:D:"):
                    continue
                if existing == key:
                    continue
                other = existing[len(cam_key) + 1 :]
                if other.startswith(stem) or stem.startswith(other[: min(6, len(other))]):
                    if now - float(ts) < cooldown_sec:
                        return False
        # Track last accepted plate text per camera for near-duplicate checks.
        prev_text = _last_plate_text.get(cam_key)
        if prev_text is not None:
            prev_val, prev_ts = prev_text
            if now - prev_ts < cooldown_sec:
                if _plates_near_duplicate(prev_val, text):
                    return False
                # Never replace a strong real reading (9I-7088) with weaker OCR
                # junk (POG-127) for the same camera dwell window.
                prev_shape = _plate_shape_score(prev_val)
                new_shape = _plate_shape_score(text)
                if prev_shape >= 90 and new_shape < 90:
                    return False
                if (
                    prev_shape >= 90
                    and new_shape >= 90
                    and _cores_soft_match(_digit_core(prev_val), _digit_core(text))
                ):
                    return False
        seen_plates[key] = now
        _last_cam_save_at[cam_key] = now
        _last_crop_phash[cam_key] = (phash, now)
        _last_plate_text[cam_key] = (text, now)
        return True

    if seen_lock is not None:
        with seen_lock:
            if not _claim():
                return None
    elif not _claim():
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_text = re.sub(r"[^A-Z0-9\-]+", "", text.upper())[:24] or "PLATE"
    prefix = f"cam{camera_id}_" if camera_id else ""
    stem = f"{prefix}{safe_text}_{stamp}"
    out_path = save_root / f"{stem}.jpg"
    save_root.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)

    scene_url = ""
    if frame_bgr is not None and getattr(frame_bgr, "size", 0):
        try:
            scene_path = save_root / f"{stem}_scene.jpg"
            # Downscale huge scenes for dashboard review.
            scene = frame_bgr
            h, w = scene.shape[:2]
            max_w = 1280
            if w > max_w:
                scale = max_w / float(w)
                scene = cv2.resize(
                    scene,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(str(scene_path), scene, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            scene_url = f"/plate_captures/{scene_path.name}"
        except Exception:
            scene_url = ""

    event_id = f"plt_{stamp}_{safe_text[:8].lower()}"
    _append_plate_event(
        {
            "id": event_id,
            "event": "plate_capture",
            "camera": str(camera_id or ""),
            "text": text,
            "ocr_confidence": round(float(ocr_conf) * 100.0, 1),
            "det_confidence": round(float(det_confidence) * 100.0, 1),
            "image_url": f"/plate_captures/{out_path.name}",
            "face_url": f"/plate_captures/{out_path.name}",
            "scene_url": scene_url,
            "filename": out_path.name,
            "iso_timestamp": datetime.now().isoformat(timespec="seconds"),
            "review_status": "OPEN",
            "severity": "INFO",
        }
    )
    return out_path


def _analyze_frame_plates(
    frame: np.ndarray,
    *,
    camera_id: str = "",
    save_root: Optional[Path] = None,
    seen_plates: Optional[Dict[str, float]] = None,
    seen_lock: Optional[threading.Lock] = None,
    cooldown_sec: float = PLATE_SAVE_COOLDOWN_SEC,
    log_hits: bool = True,
) -> List[Dict[str, Any]]:
    """Detect + OCR one frame (thread-safe via ``_inference_lock``)."""
    plates: List[Dict[str, Any]] = []
    fh = int(frame.shape[0]) if frame is not None and getattr(frame, "size", 0) else 0
    with _inference_lock:
        boxes = detect_all_plate_candidates(frame)
        for box in boxes:
            # Skip camera OSD bands (burned-in clock / IPC labels look like plates).
            if fh > 0:
                cy = 0.5 * (float(box["y1"]) + float(box["y2"]))
                if cy <= fh * 0.14 or cy >= fh * 0.92:
                    continue
            crop = frame[box["y1"] : box["y2"], box["x1"] : box["x2"]]
            if not plate_crop_looks_valid(crop):
                continue
            entry = dict(box)
            text, ocr_conf = read_plate_text(crop)
            entry["text"] = text
            entry["ocr_confidence"] = ocr_conf
            if not text:
                continue
            plates.append(entry)

            if save_root is not None and seen_plates is not None:
                saved = _vote_and_maybe_save(
                    text,
                    ocr_conf,
                    crop,
                    camera_id=camera_id,
                    save_root=save_root,
                    seen_plates=seen_plates,
                    seen_lock=seen_lock,
                    cooldown_sec=cooldown_sec,
                    det_confidence=float(entry.get("confidence") or 0.0),
                    frame_bgr=frame,
                    log_hits=log_hits,
                )
                if saved:
                    entry["saved"] = str(saved.name)
    return plates


def _camera_realtime_worker(
    camera: Dict[str, str],
    *,
    stop_event: threading.Event,
    save_root: Path,
    seen_plates: Dict[str, float],
    seen_lock: threading.Lock,
    rtsp_tcp: bool = True,
    ocr_every: int = ALL_CCTV_OCR_EVERY_N_FRAMES,
    max_frames: int = 0,
) -> None:
    """Background worker: one stream per camera (go2rtc restream preferred)."""
    cam_id = str(camera.get("id", camera.get("location", "?")))
    direct = str(camera.get("source", "")).strip()
    preferred = resolve_plate_stream_source(camera)
    sources = []
    for src in (preferred, f"{GO2RTC_RTSP_BASE}/cam{cam_id}", direct):
        if src and src not in sources:
            sources.append(src)
    label = f"CAM-{cam_id}"
    frame_idx = 0
    last_plates: List[Dict[str, Any]] = []
    reconnect_delay = 1.0
    source_idx = 0
    open_failures = 0
    last_warn_at = 0.0

    while not stop_event.is_set():
        cap = None
        # Prefer a source whose host is not in cooldown.
        picked = None
        for offset in range(len(sources)):
            candidate = sources[(source_idx + offset) % len(sources)]
            if not _host_in_cooldown(candidate):
                picked = candidate
                source_idx = (source_idx + offset) % len(sources)
                break
        if picked is None:
            # All upstream hosts cooling down — wait briefly, don't hammer FFmpeg.
            open_failures += 1
            delay = min(12.0, reconnect_delay * (2 ** min(open_failures, 3)))
            now = time.time()
            if now - last_warn_at >= 60.0:
                print(
                    f"[{label}] all sources unreachable — retry in {delay:.0f}s",
                    flush=True,
                )
                last_warn_at = now
            if stop_event.wait(delay):
                break
            source_idx = (source_idx + 1) % max(1, len(sources))
            continue

        source = picked
        try:
            cap = open_video_capture(source, rtsp_tcp=rtsp_tcp)
            print(f"[{label}] stream connected ({source})", flush=True)
            open_failures = 0
            reconnect_delay = 1.0
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    print(f"[{label}] stream dropped — reconnecting", flush=True)
                    break

                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    break

                if frame_idx % max(1, ocr_every) != 0:
                    continue

                # Skip OCR outside Settings → Capture schedule (unless ignored).
                if not _plate_captures_allowed():
                    continue

                last_plates = _analyze_frame_plates(
                    frame,
                    camera_id=cam_id,
                    save_root=save_root,
                    seen_plates=seen_plates,
                    seen_lock=seen_lock,
                )
        except Exception as exc:
            open_failures += 1
            _mark_host_unreachable(source)
            now = time.time()
            if now - last_warn_at >= 45.0:
                print(f"[{label}] error ({source}): {exc}", flush=True)
                last_warn_at = now
            source_idx = (source_idx + 1) % max(1, len(sources))
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

        if stop_event.is_set():
            break
        if max_frames and frame_idx >= max_frames:
            break
        delay = min(12.0, reconnect_delay * (2 ** min(max(0, open_failures - 1), 3)))
        time.sleep(delay)

    print(f"[{label}] stopped (frames={frame_idx}, last_plates={len(last_plates)})", flush=True)


def run_multi_cctv_realtime(
    cameras: List[Dict[str, str]],
    *,
    label: str = "CCTV",
    rtsp_tcp: bool = True,
    ocr_every: int = ALL_CCTV_OCR_EVERY_N_FRAMES,
    save_dir: Optional[Path] = None,
    stagger_sec: float = ALL_CCTV_START_STAGGER_SEC,
    max_frames: int = 0,
) -> None:
    """
    Run license plate OCR on a list of cameras at once.

    One thread per camera; YOLO + OCR share a lock so GPU inference stays safe.
    Press Ctrl+C to stop all cameras.
    """
    if not cameras:
        raise RuntimeError("No cameras selected.")

    save_root = save_dir or PLATE_SAVE_DIR
    save_root.mkdir(parents=True, exist_ok=True)
    seen_plates: Dict[str, float] = {}
    seen_lock = threading.Lock()
    stop_event = threading.Event()

    print(f"[{label}] Loading YOLO plate detector + PaddleOCR...", flush=True)
    with _inference_lock:
        # Load YOLO before Paddle — avoids Windows torch DLL clashes.
        _get_yolo_model()
        if _using_custom_plate_model():
            _get_vehicle_yolo_model()
        _get_ocr_reader()
    _seed_known_plates_from_disk(save_root)

    cam_names = ", ".join(f"CAM-{c.get('id', '?')}" for c in cameras)
    print(
        f"[{label}] Starting {len(cameras)} camera(s): {cam_names} "
        f"(OCR every {max(1, ocr_every)} frames, saves -> {save_root})",
        flush=True,
    )
    print(f"[{label}] Press Ctrl+C to stop.", flush=True)

    threads: List[threading.Thread] = []
    for camera in cameras:
        thread = threading.Thread(
            target=_camera_realtime_worker,
            kwargs={
                "camera": camera,
                "stop_event": stop_event,
                "save_root": save_root,
                "seen_plates": seen_plates,
                "seen_lock": seen_lock,
                "rtsp_tcp": rtsp_tcp,
                "ocr_every": max(1, ocr_every),
                "max_frames": max_frames,
            },
            name=f"plate-cam-{camera.get('id', '?')}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)
        if stagger_sec > 0:
            time.sleep(stagger_sec)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"[{label}] Stopping cameras...", flush=True)
        stop_event.set()

    for thread in threads:
        thread.join(timeout=8.0)
    print(f"[{label}] Done.", flush=True)


def run_all_cctv_realtime(
    *,
    rtsp_tcp: bool = True,
    ocr_every: int = ALL_CCTV_OCR_EVERY_N_FRAMES,
    save_dir: Optional[Path] = None,
    stagger_sec: float = ALL_CCTV_START_STAGGER_SEC,
    max_frames: int = 0,
) -> None:
    """Run license plate OCR on every camera in ``cameras.json``."""
    cameras = load_all_cctv_cameras()
    run_multi_cctv_realtime(
        cameras,
        label="ALL-CCTV",
        rtsp_tcp=rtsp_tcp,
        ocr_every=ocr_every,
        save_dir=save_dir,
        stagger_sec=stagger_sec,
        max_frames=max_frames,
    )


def run_gate_cctv_realtime(
    *,
    rtsp_tcp: bool = True,
    ocr_every: int = REALTIME_OCR_EVERY_N_FRAMES,
    save_dir: Optional[Path] = None,
    max_frames: int = 0,
) -> None:
    """Run license plate OCR on configured gate cameras (default 3,28,29,30)."""
    cam_ids = _plate_camera_ids_from_env()
    cameras = load_all_cctv_cameras(list(cam_ids))
    if len(cameras) != len(cam_ids):
        found = {c["id"] for c in cameras}
        missing = [c for c in cam_ids if c not in found]
        print(f"[GATES] Warning: missing cameras: {', '.join(missing)}", flush=True)
    run_multi_cctv_realtime(
        cameras,
        label="GATES",
        rtsp_tcp=rtsp_tcp,
        ocr_every=ocr_every,
        save_dir=save_dir,
        stagger_sec=1.0,
        max_frames=max_frames,
    )


def _plate_range_cutoff(range_key: str) -> datetime:
    """Start of window for today / past 7 days / past 30 days."""
    key = str(range_key or "today").strip().lower()
    start_today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if key in ("7d", "week", "past_7_days"):
        return start_today - timedelta(days=6)
    if key in ("30d", "month", "past_30_days"):
        return start_today - timedelta(days=29)
    return start_today


def _plate_row_datetime(row: Dict[str, Any]) -> Optional[datetime]:
    """Best-effort timestamp from event row or filename."""
    iso = str(row.get("iso_timestamp") or row.get("timestamp") or "").strip()
    if iso:
        try:
            return datetime.fromisoformat(iso.replace("Z", ""))
        except ValueError:
            pass
    filename = str(row.get("filename") or row.get("image_url") or "").split("/")[-1]
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return None


def list_recent_plates(*, limit: int = 48, range_key: str = "today") -> List[Dict[str, Any]]:
    """
    Recent plate hits for the dashboard.

    Merges ``plate_events.jsonl`` with ``plate_captures/``, filters by
    ``range_key`` (today / 7d / 30d), newest first.
    """
    limit = max(1, min(200, int(limit)))
    cutoff = _plate_range_cutoff(range_key)
    by_name: Dict[str, Dict[str, Any]] = {}

    def _enrich(row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        filename = str(out.get("filename") or "").strip()
        image_url = str(out.get("image_url") or "").strip()
        if not filename and image_url:
            filename = image_url.rsplit("/", 1)[-1]
            out["filename"] = filename
        if not image_url and filename:
            out["image_url"] = f"/plate_captures/{filename}"
            image_url = out["image_url"]
        if not out.get("face_url"):
            out["face_url"] = out.get("image_url") or ""
        scene_url = str(out.get("scene_url") or "").strip()
        # Drop stale companion URLs that point at missing files.
        if scene_url.startswith("/plate_captures/"):
            scene_name = scene_url.rsplit("/", 1)[-1].split("?", 1)[0]
            if scene_name and not (PLATE_SAVE_DIR / scene_name).is_file():
                scene_url = ""
                out["scene_url"] = ""
        if not scene_url and filename:
            stem = Path(filename).stem
            scene_name = f"{stem}_scene.jpg"
            if (PLATE_SAVE_DIR / scene_name).is_file():
                out["scene_url"] = f"/plate_captures/{scene_name}"
                scene_url = out["scene_url"]
        if not scene_url:
            cam = str(out.get("camera") or "").strip()
            if cam.lower().startswith("cam"):
                cam = cam[3:]
            cam = cam.strip()
            if cam:
                still = PROJECT_DIR / "live_preview" / "cameras" / f"{cam}.jpg"
                if still.is_file():
                    out["scene_url"] = f"/live_preview/cameras/{cam}.jpg"
        out.setdefault("review_status", "OPEN")
        out.setdefault("severity", "INFO")
        return out

    def _accept(row: Dict[str, Any]) -> None:
        enriched = _enrich(row)
        filename = str(enriched.get("filename") or "").strip()
        if not filename:
            return
        # Drop event-log rows whose crop was deleted (OCR junk cleanup).
        if not (PLATE_SAVE_DIR / Path(filename).name).is_file():
            return
        ts = _plate_row_datetime(enriched)
        if ts is not None and ts < cutoff:
            return
        # Prefer event-log rows (may have corrected text / review status).
        prev = by_name.get(filename)
        if prev is None:
            by_name[filename] = enriched
            return
        # Keep the row that has review/text corrections when both exist.
        if enriched.get("id") and not prev.get("id"):
            by_name[filename] = enriched

    if PLATE_EVENTS_PATH.is_file():
        try:
            lines = PLATE_EVENTS_PATH.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    _accept(row)
        except OSError:
            pass

    if PLATE_SAVE_DIR.is_dir():
        files = sorted(
            (
                p
                for p in PLATE_SAVE_DIR.glob("*.jpg")
                if not p.name.lower().endswith("_scene.jpg")
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in files:
            name = path.name
            if name in by_name:
                continue
            camera = ""
            text = ""
            m = re.match(r"^cam(\d+)_(.+?)_(\d{8}_\d{6})\.jpg$", name, re.I)
            if m:
                camera = m.group(1)
                text = m.group(2)
            else:
                text = path.stem
            scene_name = f"{path.stem}_scene.jpg"
            scene_url = (
                f"/plate_captures/{scene_name}"
                if (PLATE_SAVE_DIR / scene_name).is_file()
                else ""
            )
            _accept(
                {
                    "event": "plate_capture",
                    "camera": camera,
                    "text": text,
                    "ocr_confidence": None,
                    "det_confidence": None,
                    "image_url": f"/plate_captures/{name}",
                    "face_url": f"/plate_captures/{name}",
                    "scene_url": scene_url,
                    "filename": name,
                    "iso_timestamp": datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                        timespec="seconds"
                    ),
                }
            )

    items = list(by_name.values())

    def _sort_key(row: Dict[str, Any]) -> float:
        ts = _plate_row_datetime(row)
        return ts.timestamp() if ts is not None else 0.0

    items.sort(key=_sort_key, reverse=True)
    return items[:limit]


def _rename_plate_capture_file(old_name: str, new_text: str) -> Tuple[str, Optional[str]]:
    """
    Rename ``camX_OLD_stamp.jpg`` → ``camX_NEW_stamp.jpg`` (+ companion scene).

    Returns ``(new_filename, error_or_none)``.
    """
    old_name = str(old_name or "").strip()
    new_text = clean_plate_text(new_text)
    if not old_name or not new_text:
        return old_name, "filename and text required"
    src = PLATE_SAVE_DIR / old_name
    if not src.is_file():
        return old_name, f"File not found: {old_name}"
    safe_text = re.sub(r"[^A-Z0-9\-]+", "", new_text.upper())[:24] or "PLATE"
    m = re.match(r"^(cam\d+_)(.+?)_(\d{8}_\d{6})(\.jpe?g)$", old_name, re.I)
    if m:
        new_name = f"{m.group(1)}{safe_text}_{m.group(3)}{m.group(4).lower()}"
    else:
        stem = Path(old_name).stem
        new_name = f"{stem}_{safe_text}.jpg"
    if new_name == old_name:
        return old_name, None
    dest = PLATE_SAVE_DIR / new_name
    if dest.exists() and dest.resolve() != src.resolve():
        return old_name, f"Target already exists: {new_name}"
    try:
        src.rename(dest)
    except OSError as exc:
        return old_name, str(exc)
    # Companion scene uses ``{stem}_scene.jpg``.
    old_scene = PLATE_SAVE_DIR / f"{Path(old_name).stem}_scene.jpg"
    if old_scene.is_file():
        new_scene = PLATE_SAVE_DIR / f"{Path(new_name).stem}_scene.jpg"
        try:
            if not new_scene.exists():
                old_scene.rename(new_scene)
        except OSError:
            pass
    return new_name, None


def update_plate_label(
    *,
    filename: str = "",
    text: str = "",
    review_status: Optional[str] = None,
) -> Dict[str, Any]:
    """Correct plate OCR text and/or review status (renames capture file when needed)."""
    filename = str(filename or "").strip()
    text = normalize_plate_reading(text) or clean_plate_text(text)
    if not filename:
        return {"ok": False, "error": "filename required"}

    new_filename = filename
    if text:
        renamed, rename_err = _rename_plate_capture_file(filename, text)
        if rename_err and "not found" in rename_err.lower():
            return {"ok": False, "error": rename_err}
        if rename_err and "already exists" in rename_err.lower():
            return {"ok": False, "error": rename_err}
        if not rename_err:
            new_filename = renamed

    updated: Optional[Dict[str, Any]] = None
    lines_out: List[str] = []
    with _plate_event_lock:
        raw_lines: List[str] = []
        if PLATE_EVENTS_PATH.is_file():
            try:
                raw_lines = PLATE_EVENTS_PATH.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                return {"ok": False, "error": str(exc)}
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                lines_out.append(stripped)
                continue
            if not isinstance(row, dict):
                lines_out.append(stripped)
                continue
            row_name = str(row.get("filename") or "").strip()
            row_url = str(row.get("image_url") or "").strip()
            match = row_name == filename or row_url.endswith(f"/{filename}")
            if match:
                if text:
                    row["text"] = text
                if new_filename != filename:
                    row["filename"] = new_filename
                    row["image_url"] = f"/plate_captures/{new_filename}"
                    row["face_url"] = f"/plate_captures/{new_filename}"
                    scene_name = f"{Path(new_filename).stem}_scene.jpg"
                    if (PLATE_SAVE_DIR / scene_name).is_file():
                        row["scene_url"] = f"/plate_captures/{scene_name}"
                if review_status:
                    row["review_status"] = str(review_status).upper()
                updated = row
                lines_out.append(json.dumps(row, ensure_ascii=False))
            else:
                lines_out.append(json.dumps(row, ensure_ascii=False))

        # No event log row yet (disk-scan only) — create one from the file.
        if updated is None:
            cam = ""
            stamp = ""
            m = re.match(
                r"^cam(\d+)_(.+?)_(\d{8}_\d{6})\.jpe?g$",
                new_filename,
                re.I,
            )
            if m:
                cam = m.group(1)
                stamp = m.group(3)
            plate_text = text or (
                m.group(2) if m else Path(new_filename).stem
            )
            scene_name = f"{Path(new_filename).stem}_scene.jpg"
            scene_url = (
                f"/plate_captures/{scene_name}"
                if (PLATE_SAVE_DIR / scene_name).is_file()
                else ""
            )
            if not scene_url and cam:
                still = PROJECT_DIR / "live_preview" / "cameras" / f"{cam}.jpg"
                if still.is_file():
                    scene_url = f"/live_preview/cameras/{cam}.jpg"
            updated = {
                "id": f"plt_{stamp or Path(new_filename).stem}_{re.sub(r'[^a-z0-9]+', '', plate_text.lower())[:8]}",
                "event": "plate_capture",
                "camera": cam,
                "text": plate_text,
                "image_url": f"/plate_captures/{new_filename}",
                "face_url": f"/plate_captures/{new_filename}",
                "scene_url": scene_url,
                "filename": new_filename,
                "iso_timestamp": (
                    datetime.strptime(stamp, "%Y%m%d_%H%M%S").isoformat(timespec="seconds")
                    if stamp
                    else datetime.now().isoformat(timespec="seconds")
                ),
                "review_status": str(review_status or "OPEN").upper(),
                "severity": "INFO",
            }
            lines_out.append(json.dumps(updated, ensure_ascii=False))

        try:
            PLATE_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            PLATE_EVENTS_PATH.write_text(
                "\n".join(lines_out) + ("\n" if lines_out else ""),
                encoding="utf-8",
            )
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
    return {"ok": True, "plate": updated, "filename": new_filename}


def run_realtime(
    source: Union[int, str],
    *,
    output_path: Optional[str] = None,
    display: bool = True,
    rtsp_tcp: bool = True,
    ocr_every: int = REALTIME_OCR_EVERY_N_FRAMES,
    save_dir: Optional[Path] = None,
    max_frames: int = 0,
    camera_id: str = "",
) -> None:
    """
    Real-time license plate capture from webcam, RTSP, or video file.

    Press ``q`` in the preview window to stop. Recognized plates are printed
    to the console and optionally saved under ``plate_captures/``.
    """
    cap = open_video_capture(source, rtsp_tcp=rtsp_tcp)
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    save_root = save_dir or PLATE_SAVE_DIR
    save_root.mkdir(parents=True, exist_ok=True)
    seen_plates: Dict[str, float] = {}
    plate_cooldown_sec = PLATE_SAVE_COOLDOWN_SEC

    window = "License Plate OCR - Realtime (q=quit)"
    frame_idx = 0
    last_plates: List[Dict[str, Any]] = []

    cam_label = f"CAM-{camera_id}" if camera_id else "REALTIME"
    print(f"[{cam_label}] Source: {source}", flush=True)
    print(f"[REALTIME] OCR every {max(1, ocr_every)} frame(s). Saves -> {save_root}", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[REALTIME] Stream ended or frame read failed.", flush=True)
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            run_ocr = (frame_idx % max(1, ocr_every)) == 0
            plates = last_plates if not run_ocr else []

            if run_ocr:
                last_plates = _analyze_frame_plates(
                    frame,
                    camera_id=camera_id,
                    save_root=save_root,
                    seen_plates=seen_plates,
                    cooldown_sec=plate_cooldown_sec,
                )
                plates = last_plates

            annotated = draw_plate_results(frame, plates)
            cv2.putText(
                annotated,
                f"Frame {frame_idx} | plates {len(plates)} | q=quit",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(annotated)
            if display:
                cv2.imshow(window, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[REALTIME] Stopped by user.", flush=True)
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()


def process_video_stream(
    source: str,
    *,
    output_path: Optional[str] = None,
    display: bool = True,
    rtsp_tcp: bool = True,
) -> None:
    """Backward-compatible wrapper — use ``run_realtime``."""
    run_realtime(
        source,
        output_path=output_path,
        display=display,
        rtsp_tcp=rtsp_tcp,
        ocr_every=1,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect license plates with YOLOv8 and read text with PaddleOCR.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="",
        help="Image path, video path, RTSP URL, or camera index (0=webcam).",
    )
    parser.add_argument(
        "--cctv",
        metavar="ID",
        default="",
        help="Camera from cameras.json (e.g. 3). Comma-list for several: 28,29,30",
    )
    parser.add_argument(
        "--all-cctv",
        action="store_true",
        help="Run realtime OCR on ALL cameras in cameras.json (29 streams).",
    )
    parser.add_argument(
        "--gates",
        action="store_true",
        help="Run realtime OCR on gate cameras 28, 29, 30 only.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Live capture from webcam, RTSP, or video (recommended).",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Alias for --realtime.",
    )
    parser.add_argument(
        "--rtsp-tcp",
        action="store_true",
        default=True,
        help="Use RTSP over TCP (default, lower latency drops).",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=REALTIME_OCR_EVERY_N_FRAMES,
        help=f"Run OCR every N frames in realtime mode (default {REALTIME_OCR_EVERY_N_FRAMES}).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run until quit). For testing.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="",
        help="Save annotated image (.jpg/.png) or video (.mp4) to this path.",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help=f"Folder for plate crop JPEGs (default: {PLATE_SAVE_DIR.name}/).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a preview window (realtime mode only).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.gates:
        save_dir = Path(args.save_dir) if args.save_dir else PLATE_SAVE_DIR
        try:
            run_gate_cctv_realtime(
                rtsp_tcp=bool(args.rtsp_tcp),
                ocr_every=max(1, int(args.every)),
                save_dir=save_dir,
                max_frames=max(0, int(args.max_frames)),
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Gate CCTV failed: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.all_cctv:
        save_dir = Path(args.save_dir) if args.save_dir else PLATE_SAVE_DIR
        ocr_every = max(1, int(args.every))
        if ocr_every == REALTIME_OCR_EVERY_N_FRAMES:
            ocr_every = ALL_CCTV_OCR_EVERY_N_FRAMES
        try:
            run_all_cctv_realtime(
                rtsp_tcp=bool(args.rtsp_tcp),
                ocr_every=ocr_every,
                save_dir=save_dir,
                max_frames=max(0, int(args.max_frames)),
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"All-CCTV failed: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.cctv and ("," in args.cctv or ";" in args.cctv):
        save_dir = Path(args.save_dir) if args.save_dir else PLATE_SAVE_DIR
        ids = parse_cctv_ids(args.cctv)
        try:
            cameras = load_all_cctv_cameras(ids)
            run_multi_cctv_realtime(
                cameras,
                label="CCTV",
                rtsp_tcp=bool(args.rtsp_tcp),
                ocr_every=max(1, int(args.every)),
                save_dir=save_dir,
                max_frames=max(0, int(args.max_frames)),
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Multi-CCTV failed: {exc}", file=sys.stderr)
            return 2
        return 0

    source: Union[int, str] = args.input
    if args.cctv:
        try:
            source = load_cctv_url(args.cctv)
            print(f"[CCTV] Camera {args.cctv} -> {source.split('@')[-1]}", flush=True)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            print(f"Camera lookup failed: {exc}", file=sys.stderr)
            return 2

    realtime = bool(args.realtime or args.video or args.cctv or str(source).isdigit())
    if args.cctv and not args.input:
        realtime = True

    if realtime:
        if not source and not args.cctv:
            source = 0
        if not source:
            print("Provide input URL/path, --cctv ID, or camera index 0.", file=sys.stderr)
            return 2
        save_dir = Path(args.save_dir) if args.save_dir else PLATE_SAVE_DIR
        run_realtime(
            source,
            output_path=args.output or None,
            display=not args.no_display,
            rtsp_tcp=bool(args.rtsp_tcp),
            ocr_every=max(1, int(args.every)),
            save_dir=save_dir,
            max_frames=max(0, int(args.max_frames)),
            camera_id=str(args.cctv or ""),
        )
        return 0

    if not source:
        _build_arg_parser().print_help()
        return 2

    result = detect_and_read_plate(str(source))
    if result["error"]:
        print(result["error"], file=sys.stderr)

    for i, plate in enumerate(result["plates"], start=1):
        print(
            f"Plate {i}: text={plate.get('text')!r} "
            f"det={plate.get('confidence', 0):.2f} "
            f"ocr={plate.get('ocr_confidence', 0):.2f} "
            f"bbox=({plate['x1']},{plate['y1']},{plate['x2']},{plate['y2']})"
        )
        if plate.get("error"):
            print(f"  note: {plate['error']}")

    if args.output and result["annotated_image"] is not None:
        cv2.imwrite(args.output, result["annotated_image"])
        print(f"Saved annotated image -> {args.output}")
    elif result["annotated_image"] is not None:
        cv2.imshow("License Plate OCR", result["annotated_image"])
        print("Press any key to close the preview window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
