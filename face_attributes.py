"""
Age and gender prediction from aligned YuNet/SFace face crops.

Backend priority (auto):
  1. InsightFace genderage.onnx (buffalo_l, already used for recognition)
  2. DeepFace analyze (if installed)
  3. OpenCV DNN Levi-Hassner Caffe models (OpenCV < 5 only)

Predictions below CONFIDENCE_THRESHOLD are stored as gender="Unknown", age=None.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"

AGE_PROTO = MODELS_DIR / "deploy_age.prototxt"
AGE_MODEL = MODELS_DIR / "age_net.caffemodel"
GENDER_PROTO = MODELS_DIR / "deploy_gender.prototxt"
GENDER_MODEL = MODELS_DIR / "gender_net.caffemodel"

MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LABELS_OPENCV = ("Male", "Female")
GENDER_LABELS_INSIGHTFACE = ("Female", "Male")
AGE_MIDPOINTS = (1, 5, 10, 17, 28, 40, 50, 65)

CONFIDENCE_THRESHOLD = float(os.environ.get("FACE_ATTR_MIN_CONFIDENCE", "0.70"))
# Female false-positives are common on CCTV profiles / looking-down men.
FEMALE_CONFIDENCE_THRESHOLD = float(
    os.environ.get("FACE_ATTR_FEMALE_MIN_CONFIDENCE", "0.88")
)
FACE_PADDING = 0.20
UNKNOWN_GENDER = "Unknown"
# Soft / rear CCTV crops flip gender often — require a clearer plate.
GENDER_MIN_SIDE_PX = int(float(os.environ.get("FACE_ATTR_MIN_SIDE", "48")))
FEMALE_MIN_SIDE_PX = int(float(os.environ.get("FACE_ATTR_FEMALE_MIN_SIDE", "72")))
# InsightFace over-ages soft CCTV / body plates into 60+ — gate hard.
AGE_SENIOR_MIN_SIDE_PX = int(float(os.environ.get("FACE_ATTR_AGE_SENIOR_MIN_SIDE", "96")))
AGE_SENIOR_MIN_CONF = float(os.environ.get("FACE_ATTR_AGE_SENIOR_MIN_CONF", "0.85"))
AGE_SOFT_MAX = int(float(os.environ.get("FACE_ATTR_AGE_SOFT_MAX", "54")))

_age_net = None
_gender_net = None
_opencv_load_failed = False
_deepface_checked = False
_deepface_ok = False
_insightface_attr_model = None


class AttributePrediction(NamedTuple):
    gender: str
    age: Optional[int]
    gender_conf: float
    age_conf: float
    backend: str


def _insightface_genderage_path() -> Path:
    root = Path(os.path.expanduser(os.environ.get("INSIGHTFACE_HOME", "~/.insightface")))
    return root / "models" / "buffalo_l" / "genderage.onnx"


def _insightface_available() -> bool:
    return _insightface_genderage_path().is_file()


def _opencv_models_ready() -> bool:
    global _opencv_load_failed
    if _opencv_load_failed:
        return False
    if not all(
        p.is_file() and p.stat().st_size > 1000
        for p in (AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL)
    ):
        return False
    try:
        _load_nets()
        return True
    except Exception:
        _opencv_load_failed = True
        return False


def _deepface_available() -> bool:
    global _deepface_checked, _deepface_ok
    if _deepface_checked:
        return _deepface_ok
    _deepface_checked = True
    try:
        from deepface import DeepFace  # noqa: F401

        _deepface_ok = True
    except Exception:
        _deepface_ok = False
    return _deepface_ok


def attribute_models_ready() -> bool:
    return (
        _insightface_available()
        or _deepface_available()
        or _opencv_models_ready()
    )


def active_backend_name() -> str:
    pref = os.environ.get("FACE_ATTR_BACKEND", "auto").lower()
    if pref == "insightface":
        return "insightface" if _insightface_available() else "none"
    if pref == "opencv":
        return "opencv" if _opencv_models_ready() else "none"
    if pref == "deepface":
        return "deepface" if _deepface_available() else "none"
    if _insightface_available():
        return "insightface"
    if _deepface_available():
        return "deepface"
    if _opencv_models_ready():
        return "opencv"
    return "none"


def ensure_attribute_columns(conn) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
    if "gender" not in cols:
        conn.execute("ALTER TABLE tracking_logs ADD COLUMN gender TEXT")
    if "age" not in cols:
        conn.execute("ALTER TABLE tracking_logs ADD COLUMN age INTEGER")
    conn.commit()


def _load_nets() -> Tuple[cv2.dnn_Net, cv2.dnn_Net]:
    global _age_net, _gender_net
    if _age_net is not None and _gender_net is not None:
        return _age_net, _gender_net
    if not all(
        p.is_file() and p.stat().st_size > 1000
        for p in (AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL)
    ):
        raise FileNotFoundError(
            f"Age/gender models missing in {MODELS_DIR}\n"
            "Run: python download_face_attributes.py --no-proxy\n"
            "Or install InsightFace buffalo_l / DeepFace"
        )
    _age_net = cv2.dnn.readNet(str(AGE_MODEL), str(AGE_PROTO))
    _gender_net = cv2.dnn.readNet(str(GENDER_MODEL), str(GENDER_PROTO))
    return _age_net, _gender_net


def _load_insightface_genderage():
    global _insightface_attr_model
    if _insightface_attr_model is not None:
        return _insightface_attr_model
    from insightface.model_zoo.attribute import Attribute

    path = _insightface_genderage_path()
    _insightface_attr_model = Attribute(str(path))
    return _insightface_attr_model


def preload_attribute_models() -> None:
    """Load attribute models once at tracker startup."""
    backend = active_backend_name()
    if backend == "insightface":
        _load_insightface_genderage()
    elif backend == "opencv":
        _load_nets()


def _pad_face_crop(face_bgr: np.ndarray, margin: float = FACE_PADDING) -> np.ndarray:
    """Add border padding — attribute models need context around eyes/nose/chin."""
    h, w = face_bgr.shape[:2]
    pad_y = max(1, int(h * margin))
    pad_x = max(1, int(w * margin))
    return cv2.copyMakeBorder(
        face_bgr, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64).flatten()
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / max(exp_x.sum(), 1e-9)


def _apply_confidence_gate(
    gender: Optional[str],
    age: Optional[int],
    gender_conf: float,
    age_conf: float,
) -> Tuple[str, Optional[int]]:
    out_gender = gender if gender and gender_conf >= CONFIDENCE_THRESHOLD else UNKNOWN_GENDER
    out_age = age if age is not None and age_conf >= CONFIDENCE_THRESHOLD else None
    return out_gender, out_age


def _predict_insightface(face_bgr: np.ndarray) -> Optional[AttributePrediction]:
    if not _insightface_available():
        return None
    try:
        model = _load_insightface_genderage()
    except Exception:
        return None

    face = _pad_face_crop(face_bgr)
    input_size = model.input_size
    resized = cv2.resize(face, input_size, interpolation=cv2.INTER_LINEAR)
    blob = cv2.dnn.blobFromImage(
        resized,
        1.0 / model.input_std,
        input_size,
        (model.input_mean, model.input_mean, model.input_mean),
        swapRB=True,
    )
    pred = model.session.run(model.output_names, {model.input_name: blob})[0][0]
    gender_probs = _softmax(pred[:2])
    gender_idx = int(gender_probs.argmax())
    gender_conf = float(gender_probs[gender_idx])
    gender = GENDER_LABELS_INSIGHTFACE[gender_idx] if 0 <= gender_idx < len(GENDER_LABELS_INSIGHTFACE) else None
    age = int(np.round(float(pred[2]) * 100))
    age_conf = gender_conf
    gender, age = _apply_confidence_gate(gender, age, gender_conf, age_conf)
    return AttributePrediction(gender, age, gender_conf, age_conf, "insightface")


def _predict_opencv(face_bgr: np.ndarray) -> Optional[AttributePrediction]:
    if not _opencv_models_ready():
        return None
    try:
        age_net, gender_net = _load_nets()
    except Exception:
        return None

    face = _pad_face_crop(face_bgr)
    if max(face.shape[:2]) < 64:
        face = cv2.resize(face, (227, 227), interpolation=cv2.INTER_LINEAR)

    blob = cv2.dnn.blobFromImage(
        face, scalefactor=1.0, size=(227, 227), mean=MODEL_MEAN, swapRB=False
    )

    gender_net.setInput(blob)
    gender_probs = _softmax(gender_net.forward())
    gender_idx = int(gender_probs.argmax())
    gender_conf = float(gender_probs[gender_idx])
    gender = GENDER_LABELS_OPENCV[gender_idx] if 0 <= gender_idx < len(GENDER_LABELS_OPENCV) else None

    age_net.setInput(blob)
    age_probs = _softmax(age_net.forward())
    age_idx = int(age_probs.argmax())
    age_conf = float(age_probs[age_idx])
    age = int(AGE_MIDPOINTS[age_idx]) if 0 <= age_idx < len(AGE_MIDPOINTS) else None

    gender, age = _apply_confidence_gate(gender, age, gender_conf, age_conf)
    return AttributePrediction(gender, age, gender_conf, age_conf, "opencv")


def _map_deepface_gender(label: str) -> Optional[str]:
    text = (label or "").strip().lower()
    if text in ("man", "male"):
        return "Male"
    if text in ("woman", "female"):
        return "Female"
    return None


def _predict_deepface(face_bgr: np.ndarray) -> Optional[AttributePrediction]:
    if not _deepface_available():
        return None
    try:
        from deepface import DeepFace
    except Exception:
        return None

    face = _pad_face_crop(face_bgr)
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    try:
        results = DeepFace.analyze(
            img_path=rgb,
            actions=["age", "gender"],
            enforce_detection=False,
            detector_backend="skip",
            silent=True,
        )
    except Exception:
        return None

    row = results[0] if isinstance(results, list) else results
    if not isinstance(row, dict):
        return None

    gender_raw = row.get("dominant_gender") or row.get("gender")
    gender_conf = 0.0
    gender = None

    if isinstance(gender_raw, dict):
        if gender_raw:
            label = max(gender_raw, key=gender_raw.get)
            gender_conf = float(gender_raw[label]) / 100.0
            gender = _map_deepface_gender(label)
    elif isinstance(gender_raw, str):
        gender = _map_deepface_gender(gender_raw)
        gender_conf = 0.85 if gender else 0.0

    age_val = row.get("age")
    try:
        age = int(round(float(age_val))) if age_val is not None else None
    except (TypeError, ValueError):
        age = None

    age_conf = gender_conf if age is not None else 0.0
    gender, age = _apply_confidence_gate(gender, age, gender_conf, age_conf)
    return AttributePrediction(gender, age, gender_conf, age_conf, "deepface")


def _predict_with_backend(
    face_bgr: np.ndarray,
    backend: str,
) -> Optional[AttributePrediction]:
    if backend == "insightface":
        return _predict_insightface(face_bgr)
    if backend == "deepface":
        return _predict_deepface(face_bgr)
    if backend == "opencv":
        return _predict_opencv(face_bgr)
    return None


def _predict_best(face_bgr: np.ndarray) -> Optional[AttributePrediction]:
    pref = os.environ.get("FACE_ATTR_BACKEND", "auto").lower()
    if pref != "auto":
        return _predict_with_backend(face_bgr, active_backend_name())

    for backend in ("insightface", "deepface", "opencv"):
        pred = _predict_with_backend(face_bgr, backend)
        if pred is not None and pred.gender != UNKNOWN_GENDER:
            return pred
    for backend in ("insightface", "deepface", "opencv"):
        pred = _predict_with_backend(face_bgr, backend)
        if pred is not None:
            return pred
    return None


def predict_face_attributes(
    face_bgr: np.ndarray,
) -> Tuple[str, Optional[int]]:
    """
    Predict gender and age from an aligned face crop (BGR).
    Returns (gender, age) with Unknown/None when confidence is below threshold.
    """
    if face_bgr is None or face_bgr.size == 0:
        return UNKNOWN_GENDER, None

    try:
        h, w = int(face_bgr.shape[0]), int(face_bgr.shape[1])
        if min(h, w) < int(GENDER_MIN_SIDE_PX):
            return UNKNOWN_GENDER, None
    except Exception:
        return UNKNOWN_GENDER, None

    # Back-of-head / walking-away / non-face plates must not vote Male/Female.
    looking_down = False
    profile = False
    try:
        from face_print_filter import (
            is_likely_rear_view,
            is_likely_invalid_face_crop,
            is_likely_non_face_object,
            is_likely_looking_down,
            is_likely_profile_view,
            is_likely_animal_or_dog,
            is_likely_unusable_face_plate,
            capture_crop_passes_human_gate,
        )

        rear, _ = is_likely_rear_view(face_bgr)
        if rear:
            return UNKNOWN_GENDER, None
        bad, _ = is_likely_invalid_face_crop(face_bgr)
        if bad:
            return UNKNOWN_GENDER, None
        non_face, _ = is_likely_non_face_object(face_bgr)
        if non_face:
            return UNKNOWN_GENDER, None
        animal, _ = is_likely_animal_or_dog(face_bgr)
        if animal:
            return UNKNOWN_GENDER, None
        mush, _ = is_likely_unusable_face_plate(face_bgr)
        if mush:
            return UNKNOWN_GENDER, None
        human_ok, _ = capture_crop_passes_human_gate(face_bgr)
        if not human_ok:
            return UNKNOWN_GENDER, None
        looking_down, _ = is_likely_looking_down(face_bgr)
        profile, _ = is_likely_profile_view(face_bgr)
    except Exception:
        pass

    prediction = _predict_best(face_bgr)
    if prediction is None:
        return UNKNOWN_GENDER, None

    gender = prediction.gender
    age = prediction.age
    conf = float(prediction.gender_conf or 0.0)
    side = min(h, w)
    tall_body = h >= int(w * 1.35)

    # Looking-down hallway crops flip Male→Female often.
    if looking_down and gender == "Female":
        return UNKNOWN_GENDER, None
    # Soft / profile / tiny plates: Female needs a much stronger score.
    if gender == "Female":
        if conf < float(FEMALE_CONFIDENCE_THRESHOLD):
            return UNKNOWN_GENDER, None
        if side < int(FEMALE_MIN_SIDE_PX) and conf < 0.93:
            return UNKNOWN_GENDER, None
        if profile and conf < 0.93:
            return UNKNOWN_GENDER, None
        if profile and side < 110:
            return UNKNOWN_GENDER, None
        # Tall body / hat / blurry plates must not count as Female.
        if tall_body and conf < 0.95:
            return UNKNOWN_GENDER, None
        if side < 96:
            return UNKNOWN_GENDER, None
    elif gender == "Male":
        if conf < float(CONFIDENCE_THRESHOLD):
            return UNKNOWN_GENDER, None
        if tall_body and conf < 0.85:
            return UNKNOWN_GENDER, None
    else:
        return UNKNOWN_GENDER, None

    # Age on soft CCTV is noisy — InsightFace often pushes mid-age Asians to 60+.
    if age is not None:
        soft_pose = bool(looking_down or profile or tall_body or side < int(AGE_SENIOR_MIN_SIDE_PX))
        if soft_pose and int(age) > int(AGE_SOFT_MAX):
            age = None
        elif int(age) >= 60:
            if (
                soft_pose
                or conf < float(AGE_SENIOR_MIN_CONF)
                or side < int(AGE_SENIOR_MIN_SIDE_PX)
            ):
                age = None

    return gender, age


def predict_face_attributes_verbose(
    face_bgr: np.ndarray,
) -> AttributePrediction:
    """Like predict_face_attributes but includes confidence and backend name."""
    if face_bgr is None or face_bgr.size == 0:
        return AttributePrediction(UNKNOWN_GENDER, None, 0.0, 0.0, "none")

    pred = _predict_best(face_bgr)
    if pred is None:
        return AttributePrediction(UNKNOWN_GENDER, None, 0.0, 0.0, active_backend_name())
    return pred
