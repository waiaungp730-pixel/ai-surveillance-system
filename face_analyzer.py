"""
RetinaFace detection → ArcFace embedding pipeline via DeepFace.

Tiers:
  1 — quality gating + weighted FaceGallery averages
  2 — SharedFaceGallery (SQLite WAL) + AdaptiveFrameSkip
  3 — optional ONNX RetinaFace + AntiSpoof liveness
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from gpu_pipeline import (
    crop_xyxy,
    laplacian_variance,
    preprocess_retinaface_gpu,
    tensor_to_bgr_numpy,
)
from inference_device import tensor_to_numpy
from face_landmark_verify import verify_landmarks_for_capture
from tracker_config import (
    FACE_ASPECT_RATIO_MAX,
    FACE_ASPECT_RATIO_MIN,
    FACE_DETECT_MIN_CONFIDENCE,
    FACE_LANDMARK_MIN_CONFIDENCE,
    FACE_MAX_EYE_SPAN_RATIO,
    FACE_MIN_BOX_PX,
    FACE_MIN_EYE_SPAN_RATIO,
)

log = logging.getLogger("face_analyzer")

PROJECT_DIR = Path(__file__).parent
DEFAULT_GALLERY_DB = PROJECT_DIR / "face_gallery.db"
DEFAULT_ONNX_MODEL = PROJECT_DIR / "retinaface.onnx"
DEFAULT_ANTI_SPOOF_MODEL = PROJECT_DIR / "models" / "anti_spoof.onnx"

_checked = False
_retina_ok = False
_deepface_ok = False
_insightface_ok = False
_insightface_app: Optional[Any] = None
_insightface_lock = threading.Lock()
_insightface_thread_local = threading.local()

ARCFACE_MODEL = "ArcFace"
DETECTOR_BACKEND = "retinaface"
MIN_FACE_PX = FACE_MIN_BOX_PX

_CLAHE: Optional[cv2.CLAHE] = None


def _get_clahe() -> cv2.CLAHE:
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _CLAHE


def apply_face_clahe(face_bgr: np.ndarray) -> np.ndarray:
    """Normalize local contrast on a BGR face crop before ArcFace embedding."""
    if face_bgr is None or face_bgr.size == 0:
        return face_bgr
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    eq = _get_clahe().apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Dependency probes (preserved API)
# ---------------------------------------------------------------------------


def _check_deps() -> None:
    global _checked, _retina_ok, _deepface_ok
    if _checked:
        return
    _checked = True
    try:
        from retinaface import RetinaFace  # noqa: F401

        _retina_ok = True
    except ImportError:
        try:
            from retina_face import RetinaFace  # noqa: F401

            _retina_ok = True
        except ImportError:
            _retina_ok = False
    try:
        from deepface import DeepFace  # noqa: F401

        _deepface_ok = True
    except ImportError:
        _deepface_ok = False


def retinaface_available() -> bool:
    _check_deps()
    return _retina_ok


def deepface_available() -> bool:
    _check_deps()
    return _deepface_ok


def insightface_available() -> bool:
    _check_deps()
    return _insightface_ok


def _configure_tf_gpu() -> None:
    """မြန်မာ: TensorFlow/Keras RetinaFace + DeepFace ArcFace ကို GPU ပေါ်တင်ရန် ကြိုးစားသည်။"""
    try:
        from inference_device import torch_cuda_active

        if not torch_cuda_active():
            return
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print(
                "WARNING: TensorFlow has no GPU — RetinaFace/DeepFace fallback will use CPU (oneDNN).\n"
                "  Prefer InsightFace GPU: pip install insightface --no-deps onnx>=1.16.0 scikit-image"
            )
            return
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(f"TensorFlow GPU enabled ({len(gpus)} device(s)) for RetinaFace/DeepFace fallback")
    except Exception as exc:
        log.debug("TensorFlow GPU config skipped: %s", exc)


def _insightface_providers() -> List[str]:
    """CUDA primary; CPUExecutionProvider listed as ONNX terminal fallback only."""
    try:
        from inference_device import torch_cuda_active

        if torch_cuda_active():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def _verify_insightface_cuda(app: Any) -> None:
    """Refuse silent CPU fallback when CUDA was requested."""
    for task_name, model in app.models.items():
        session = getattr(model, "session", None)
        if session is None:
            continue
        active = session.get_providers()
        if not active:
            continue
        if active[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"InsightFace {task_name} fell back to {active[0]} — "
                "install onnxruntime-gpu matching your CUDA Toolkit"
            )


_pt_retinaface_model: Optional[Any] = None
_pt_retinaface_lock = threading.Lock()


def _load_pytorch_retinaface_cuda() -> Optional[Any]:
    """Load PyTorch RetinaFaceModel on CUDA when the native Keras path is unavailable."""
    global _pt_retinaface_model
    with _pt_retinaface_lock:
        if _pt_retinaface_model is not None:
            return _pt_retinaface_model
        try:
            from inference_device import get_device, move_model_to_device, torch_cuda_active

            if not torch_cuda_active():
                return None
            from retinaface.model.retinaface_model import RetinaFaceModel  # type: ignore

            model = RetinaFaceModel()
            model = move_model_to_device(model)
            model.eval()
            _pt_retinaface_model = model
            print(f"RetinaFace PyTorch: .to('{get_device()}')")
            return model
        except ImportError:
            return None
        except Exception as exc:
            log.debug("PyTorch RetinaFace CUDA load skipped: %s", exc)
            return None


def _warm_deepface_arcface_gpu() -> None:
    """Preload DeepFace ArcFace weights on GPU via TensorFlow when InsightFace is unavailable."""
    if not deepface_available():
        return
    try:
        from inference_device import torch_cuda_active

        if not torch_cuda_active():
            return
        _configure_tf_gpu()
        from deepface import DeepFace

        DeepFace.build_model(ARCFACE_MODEL)
        print(f"ArcFace (DeepFace/{ARCFACE_MODEL}): TensorFlow GPU warmup complete")
    except Exception as exc:
        log.debug("DeepFace ArcFace GPU warmup skipped: %s", exc)


def _create_insightface_cuda_app(
    *, quiet: bool = False, det_thresh: float = 0.5, det_size: Tuple[int, int] = (640, 640)
) -> Optional[Any]:
    """Build a new InsightFace FaceAnalysis on CUDA (caller owns the instance)."""
    try:
        from inference_device import torch_cuda_active

        if not quiet:
            print("CUDA Available:", torch.cuda.is_available())
        if not torch_cuda_active():
            return None
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" not in available:
            if not quiet:
                print(
                    "WARNING: InsightFace GPU skipped — CUDAExecutionProvider unavailable.\n"
                    "  pip install onnxruntime-gpu  (must match PyTorch CUDA / CUDA Toolkit)"
                )
            return None
        from insightface.app import FaceAnalysis

        providers = _insightface_providers()
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        side = max(160, int(det_size[0])), max(160, int(det_size[1]))
        app.prepare(ctx_id=0, det_size=(side, side), det_thresh=float(det_thresh))
        _verify_insightface_cuda(app)
        if not quiet:
            print("InsightFace loaded: providers=['CUDAExecutionProvider'], ctx_id=0")
        return app
    except ImportError:
        if not quiet:
            print(
                "WARNING: insightface not installed — face pipeline will use CPU TensorFlow.\n"
                "  pip install insightface --no-deps onnx>=1.16.0 scikit-image imageio\n"
                "  (keep onnxruntime-gpu; do not install CPU onnxruntime)"
            )
        return None
    except Exception as exc:
        log.warning("InsightFace CUDA init failed (%s) — using DeepFace/ONNX fallback", exc)
        if not quiet:
            print(f"WARNING: InsightFace CUDA init failed ({exc})")
        return None


def _try_load_insightface_cuda_for_thread() -> Optional[Any]:
    """Per-thread InsightFace app for parallel folder video scans."""
    app = getattr(_insightface_thread_local, "app", None)
    if app is not None:
        return app
    from identity import VIDEO_SCAN_RETINAFACE_DET_THRESH

    det_side = max(160, int(os.environ.get("VIDEO_SCAN_DET_SIZE", "320")))
    app = _create_insightface_cuda_app(
        quiet=True,
        det_thresh=VIDEO_SCAN_RETINAFACE_DET_THRESH,
        det_size=(det_side, det_side),
    )
    if app is not None:
        _insightface_thread_local.app = app
    return app


def _try_load_insightface_cuda() -> Optional[Any]:
    """
    မြန်မာ: InsightFace ကို CUDAExecutionProvider + ctx_id=0 ဖြင့် GPU ပေါ်တင်သည်။
    RetinaFace detection + ArcFace embedding တို့ကို တစ်ခါတည်း GPU ပေါ်တွင် လုပ်ဆောင်သည်။
    """
    global _insightface_app, _insightface_ok
    with _insightface_lock:
        if _insightface_app is not None:
            return _insightface_app
        # Live CCTV needs a low RetinaFace floor (matches LIVE_FACE_DETECT_CONF ~0.28).
        # Default InsightFace 0.5 silently drops hallway / side-view faces.
        live_thresh = 0.18
        try:
            raw = os.environ.get("LIVE_FACE_DETECT_CONF", "").strip()
            if raw:
                live_thresh = float(raw)
        except ValueError:
            pass
        try:
            from identity import VIDEO_SCAN_RETINAFACE_DET_THRESH

            live_thresh = min(float(live_thresh), float(VIDEO_SCAN_RETINAFACE_DET_THRESH))
        except Exception:
            pass
        live_thresh = max(0.10, min(0.40, float(live_thresh)))
        app = _create_insightface_cuda_app(
            quiet=False,
            det_thresh=live_thresh,
            det_size=(640, 640),
        )
        if app is None:
            return None
        _insightface_app = app
        _insightface_ok = True
        return app


def analyzer_ready() -> bool:
    _check_deps()
    _try_load_insightface_cuda()
    return _deepface_ok or _insightface_ok


def active_pipeline_name() -> str:
    _check_deps()
    if _insightface_ok or _insightface_app is not None:
        return "InsightFace GPU (RetinaFace + ArcFace, CUDAExecutionProvider)"
    try:
        from inference_device import torch_cuda_active

        if torch_cuda_active():
            if _try_load_insightface_cuda() is not None:
                return "InsightFace GPU (RetinaFace + ArcFace, CUDAExecutionProvider)"
    except Exception:
        pass
    if not _deepface_ok:
        return "unavailable"
    if _retina_ok:
        return "RetinaFace + ArcFace (TensorFlow/DeepFace)"
    return "ArcFace (opencv detector)"


def detector_backend() -> str:
    return DETECTOR_BACKEND if retinaface_available() else "opencv"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FaceDetection:
    bbox: Tuple[int, int, int, int]
    score: float
    landmarks: Dict[str, Tuple[float, float]]
    embedding: Optional[np.ndarray] = None
    quality_ok: bool = True
    reject_reason: Optional[str] = None
    anti_spoof_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Tier 1 — FaceGallery (weighted embedding average)
# ---------------------------------------------------------------------------


class FaceGallery:
    """Per-track in-memory embedding history with weighted mean matching."""

    def __init__(self, threshold: float = 0.45, max_history: int = 5) -> None:
        self._threshold = threshold
        self._max_history = max_history
        self._store: Dict[int, List[np.ndarray]] = {}

    def add(self, track_id: int, embedding: np.ndarray) -> None:
        history = self._store.setdefault(track_id, [])
        history.append(np.asarray(embedding, dtype=np.float32).reshape(-1))
        if len(history) > self._max_history:
            history.pop(0)

    def _weighted_mean(self, embeddings: List[np.ndarray]) -> np.ndarray:
        n = len(embeddings)
        weights = np.exp(np.linspace(-1.0, 0.0, n))
        weights /= weights.sum()
        avg = sum(w * e for w, e in zip(weights, embeddings))
        norm = np.linalg.norm(avg)
        return avg / (norm + 1e-6)

    def match(
        self, embedding: np.ndarray, exclude: Optional[int] = None
    ) -> Tuple[Optional[int], float]:
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        qn = np.linalg.norm(query)
        if qn < 1e-6:
            return None, 0.0
        query = query / qn

        best_id: Optional[int] = None
        best_score = 0.0
        for track_id, history in self._store.items():
            if exclude is not None and track_id == exclude:
                continue
            if not history:
                continue
            gallery_vec = self._weighted_mean(history)
            score = float(np.dot(query, gallery_vec))
            if score > best_score:
                best_score = score
                best_id = track_id

        if best_id is not None and best_score >= self._threshold:
            return best_id, best_score
        return None, best_score

    def remove(self, track_id: int) -> None:
        self._store.pop(track_id, None)


# ---------------------------------------------------------------------------
# Tier 2 — SharedFaceGallery (SQLite WAL)
# ---------------------------------------------------------------------------


def _pack_embedding_history(embeddings: List[np.ndarray]) -> bytes:
    if not embeddings:
        return np.array([0, 0], dtype=np.int32).tobytes()
    n = len(embeddings)
    dim = int(embeddings[0].size)
    header = np.array([n, dim], dtype=np.int32).tobytes()
    body = b"".join(e.astype(np.float32).reshape(-1).tobytes() for e in embeddings)
    return header + body


def _unpack_embedding_history(blob: bytes) -> List[np.ndarray]:
    if not blob or len(blob) < 8:
        return []
    header = np.frombuffer(blob[:8], dtype=np.int32)
    n, dim = int(header[0]), int(header[1])
    if n <= 0 or dim <= 0:
        return []
    offset = 8
    out: List[np.ndarray] = []
    for _ in range(n):
        end = offset + dim * 4
        if end > len(blob):
            break
        out.append(np.frombuffer(blob[offset:end], dtype=np.float32).copy())
        offset = end
    return out


class SharedFaceGallery:
    """SQLite-backed gallery safe for multi-process camera workers (WAL)."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS gallery (
        track_id   TEXT    NOT NULL,
        camera_id  TEXT    NOT NULL,
        embedding  BLOB    NOT NULL,
        updated_at REAL    NOT NULL,
        PRIMARY KEY (track_id, camera_id)
    );
    """

    def __init__(self, db_path: Path, threshold: float = 0.45) -> None:
        self._db_path = Path(db_path)
        self._threshold = threshold
        self._lock = threading.Lock()
        self._max_history = 5
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(self._SCHEMA)
            self._conn.commit()

    def _weighted_mean(self, embeddings: List[np.ndarray]) -> np.ndarray:
        n = len(embeddings)
        weights = np.exp(np.linspace(-1.0, 0.0, n))
        weights /= weights.sum()
        avg = sum(w * e for w, e in zip(weights, embeddings))
        norm = np.linalg.norm(avg)
        return avg / (norm + 1e-6)

    def add(self, track_id: str, camera_id: str, embedding: np.ndarray) -> None:
        now = time.time()
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        with self._lock:
            self._conn.execute(
                "DELETE FROM gallery WHERE updated_at < ?",
                (now - 3600.0,),
            )
            row = self._conn.execute(
                "SELECT embedding FROM gallery WHERE track_id = ? AND camera_id = ?",
                (track_id, camera_id),
            ).fetchone()
            history: List[np.ndarray] = []
            if row and row[0]:
                history = _unpack_embedding_history(row[0])
            history.append(vec)
            if len(history) > self._max_history:
                history = history[-self._max_history :]
            blob = _pack_embedding_history(history)
            self._conn.execute(
                """
                INSERT INTO gallery (track_id, camera_id, embedding, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(track_id, camera_id) DO UPDATE SET
                    embedding = excluded.embedding,
                    updated_at = excluded.updated_at
                """,
                (track_id, camera_id, blob, now),
            )
            self._conn.commit()

    def match(
        self,
        embedding: np.ndarray,
        exclude_camera: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        qn = np.linalg.norm(query)
        if qn < 1e-6:
            return None, 0.0
        query = query / qn

        with self._lock:
            rows = self._conn.execute(
                "SELECT track_id, camera_id, embedding FROM gallery"
            ).fetchall()

        best_id: Optional[str] = None
        best_score = 0.0
        for tid, cam, blob in rows:
            if exclude_camera is not None and cam == exclude_camera:
                continue
            history = _unpack_embedding_history(blob)
            if not history:
                single = np.frombuffer(blob, dtype=np.float32)
                if single.size:
                    history = [single]
            if not history:
                continue
            gallery_vec = self._weighted_mean(history)
            score = float(np.dot(query, gallery_vec))
            if score > best_score:
                best_score = score
                best_id = str(tid)

        if best_id is not None and best_score >= self._threshold:
            return best_id, best_score
        return None, best_score

    def remove(self, track_id: str, camera_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM gallery WHERE track_id = ? AND camera_id = ?",
                (track_id, camera_id),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Tier 2 — AdaptiveFrameSkip
# ---------------------------------------------------------------------------


class AdaptiveFrameSkip:
    """EMA-smoothed inference timing → dynamic frame skip interval."""

    def __init__(
        self,
        target_ms: float = 100.0,
        min_skip: int = 1,
        max_skip: int = 10,
        smoothing: float = 0.2,
    ) -> None:
        self.target_ms = target_ms
        self.min_skip = min_skip
        self.max_skip = max_skip
        self._smoothing = smoothing
        self._ema = target_ms
        self._skip = min_skip

    @property
    def skip(self) -> int:
        return self._skip

    def update(self, inference_ms: float) -> None:
        self._ema = self._smoothing * inference_ms + (1.0 - self._smoothing) * self._ema
        raw = int(self._ema / self.target_ms) + 1
        new_skip = max(self.min_skip, min(self.max_skip, raw))
        if new_skip != self._skip:
            log.debug("AdaptiveFrameSkip: %d → %d (ema=%.1f ms)", self._skip, new_skip, self._ema)
            self._skip = new_skip


SCAN_DETECT_MIN_CONFIDENCE = 0.28
SCAN_MIN_FACE_PX = 14
SCAN_MAX_BLUR = 2.0
SCAN_MAX_YAW = 0.98
SCAN_LANDMARK_MIN_CONFIDENCE = 0.25


# ---------------------------------------------------------------------------
# Quality gating
# ---------------------------------------------------------------------------


def _bbox_aspect_ok(width: int, height: int) -> bool:
    if width < 1 or height < 1:
        return False
    ratio = width / float(height)
    return FACE_ASPECT_RATIO_MIN <= ratio <= FACE_ASPECT_RATIO_MAX


def _is_usable_face_scan(
    det: FaceDetection,
    frame: np.ndarray,
    *,
    min_size: int = SCAN_MIN_FACE_PX,
    max_blur: float = SCAN_MAX_BLUR,
    min_confidence: float = SCAN_DETECT_MIN_CONFIDENCE,
) -> Tuple[bool, str]:
    """Relaxed quality gate for dashboard video scan (live CCTV gates stay strict)."""
    if det.score < min_confidence:
        return False, f"low_confidence({det.score:.2f})"

    x1, y1, x2, y2 = det.bbox
    w = x2 - x1
    h = y2 - y1
    if max(w, h) < min_size:
        return False, f"too_small({w}x{h})"

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "empty_crop"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < max_blur:
        return False, f"blurry({blur_score:.1f})"

    return True, ""


def _is_usable_face(
    det: FaceDetection,
    frame: np.ndarray,
    min_size: int = FACE_MIN_BOX_PX,
    max_blur: float = 80.0,
    max_yaw: float = 0.35,
    min_confidence: float = FACE_DETECT_MIN_CONFIDENCE,
) -> Tuple[bool, str]:
    if det.score < min_confidence:
        return False, f"low_confidence({det.score:.2f})"

    x1, y1, x2, y2 = det.bbox
    w = x2 - x1
    h = y2 - y1
    if w < min_size or h < min_size:
        return False, f"too_small({w}x{h})"
    if not _bbox_aspect_ok(w, h):
        return False, f"bad_aspect({w}/{h})"

    ok_lm, lm_reason = verify_landmarks_for_capture(
        det.landmarks,
        det.bbox,
        detection_confidence=det.score,
        # Live CCTV uses a lower detect_threshold — keep landmark gate in the same band.
        min_confidence=(
            FACE_LANDMARK_MIN_CONFIDENCE
            if min_confidence >= 0.8
            else SCAN_LANDMARK_MIN_CONFIDENCE
        ),
        allow_side_view=True,
        max_yaw=max_yaw,
    )
    if not ok_lm:
        return False, lm_reason

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "empty_crop"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < max_blur:
        return False, f"blurry({blur_score:.1f})"

    return True, ""


def _is_usable_face_gpu(
    det: FaceDetection,
    frame: torch.Tensor,
    min_size: int = FACE_MIN_BOX_PX,
    max_blur: float = 80.0,
    max_yaw: float = 0.35,
    min_confidence: float = FACE_DETECT_MIN_CONFIDENCE,
) -> Tuple[bool, str]:
    if det.score < min_confidence:
        return False, f"low_confidence({det.score:.2f})"

    x1, y1, x2, y2 = det.bbox
    w = x2 - x1
    h = y2 - y1
    if w < min_size or h < min_size:
        return False, f"too_small({w}x{h})"
    if not _bbox_aspect_ok(w, h):
        return False, f"bad_aspect({w}/{h})"

    ok_lm, lm_reason = verify_landmarks_for_capture(
        det.landmarks,
        det.bbox,
        detection_confidence=det.score,
        min_confidence=(
            FACE_LANDMARK_MIN_CONFIDENCE
            if min_confidence >= 0.8
            else SCAN_LANDMARK_MIN_CONFIDENCE
        ),
        allow_side_view=True,
        max_yaw=max_yaw,
    )
    if not ok_lm:
        return False, lm_reason

    crop = crop_xyxy(frame, det.bbox)
    if crop is None or crop.numel() == 0:
        return False, "empty_crop"
    blur_score = laplacian_variance(crop)
    if blur_score < max_blur:
        return False, f"blurry({blur_score:.1f})"

    return True, ""


# ---------------------------------------------------------------------------
# FaceAnalyzer
# ---------------------------------------------------------------------------


class FaceAnalyzer:
    """RetinaFace (+ optional ONNX) → ArcFace with quality gating and anti-spoof."""

    def __init__(
        self,
        min_size: int = FACE_MIN_BOX_PX,
        max_blur: float = 80.0,
        max_yaw: float = 0.35,
        onnx_model_path: Optional[Path] = None,
        anti_spoof_model: Optional[Path] = None,
        detect_threshold: float = FACE_DETECT_MIN_CONFIDENCE,
        isolated_insightface: bool = False,
    ) -> None:
        _check_deps()
        self.min_size = min_size
        self.max_blur = max_blur
        self.max_yaw = max_yaw
        self.detect_threshold = detect_threshold
        self._ort_session = None
        self._ort_input_name: Optional[str] = None
        self._anti_spoof = None
        self._insightface_app: Optional[Any] = None

        # မြန်မာ: CUDA ရှိပါက InsightFace GPU backend ကို ဦးစားပေးတင်သည်
        from inference_device import torch_cuda_active

        if torch_cuda_active():
            if isolated_insightface:
                self._insightface_app = _try_load_insightface_cuda_for_thread()
            else:
                self._insightface_app = _try_load_insightface_cuda()
            if self._insightface_app is None:
                _configure_tf_gpu()
                _load_pytorch_retinaface_cuda()
                _warm_deepface_arcface_gpu()

        onnx_path = Path(onnx_model_path) if onnx_model_path else DEFAULT_ONNX_MODEL
        if onnx_path.is_file():
            try:
                from inference_device import create_onnx_session

                self._ort_session = create_onnx_session(
                    str(onnx_path), label="RetinaFace ONNX"
                )
                self._ort_input_name = self._ort_session.get_inputs()[0].name
                active = self._ort_session.get_providers()
                print(f"RetinaFace ONNX: {active[0] if active else 'unknown'}")
                log.debug("RetinaFace ONNX session loaded: %s", onnx_path)
            except Exception as exc:
                log.warning("ONNX RetinaFace failed to load (%s) — using native", exc)
                self._ort_session = None
        elif torch_cuda_active() and self._insightface_app is None:
            print(
                f"WARNING: {onnx_path.name} not found — RetinaFace ONNX GPU disabled.\n"
                "  Export: python export_retinaface_onnx.py"
            )

        self._anti_spoof = None
        if anti_spoof_model is not None:
            from anti_spoof import AntiSpoof

            self._anti_spoof = AntiSpoof(Path(anti_spoof_model))

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        if self._insightface_app is not None:
            return self._detect_insightface(frame)
        if self._ort_session is not None:
            return self._detect_onnx(frame)
        return self._detect_native(frame)

    def detect_gpu(self, frame: torch.Tensor) -> List[FaceDetection]:
        if self._insightface_app is not None:
            return self._detect_insightface(tensor_to_bgr_numpy(frame))
        if self._ort_session is not None:
            return self._detect_onnx_gpu(frame)
        return self._detect_native(tensor_to_bgr_numpy(frame))

    def _detect_insightface(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        """မြန်မာ: InsightFace GPU — RetinaFace detect + ArcFace embed တစ်ခါတည်း။"""
        if self._insightface_app is None or frame_bgr is None or frame_bgr.size == 0:
            return []
        t0 = time.perf_counter()
        try:
            faces = self._insightface_app.get(frame_bgr)
        except Exception as exc:
            log.debug("InsightFace detect failed: %s", exc)
            return []

        dets: List[FaceDetection] = []
        lm_names = ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")
        for face in faces:
            bbox = face.bbox.astype(int)
            if bbox.shape[0] < 4:
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            score = float(getattr(face, "det_score", 1.0))
            if score < self.detect_threshold:
                continue
            landmarks: Dict[str, Tuple[float, float]] = {}
            kps = getattr(face, "kps", None)
            if kps is not None:
                for idx, name in enumerate(lm_names):
                    if idx < len(kps):
                        landmarks[name] = (float(kps[idx][0]), float(kps[idx][1]))
            det = FaceDetection(
                bbox=(x1, y1, x2, y2),
                score=score,
                landmarks=landmarks,
            )
            emb = getattr(face, "normed_embedding", None)
            if emb is not None:
                det.embedding = np.asarray(emb, dtype=np.float32).reshape(-1)
            dets.append(det)

        ms = (time.perf_counter() - t0) * 1000.0
        log.debug("InsightFace GPU detect+embed %.1f ms (%d faces)", ms, len(dets))
        return dets

    def _ort_forward(self, im_tensor) -> list:
        """Run RetinaFace ONNX — GPU IO binding in pure-GPU mode when possible."""
        if self._ort_session is None or self._ort_input_name is None:
            raise RuntimeError("ONNX session not loaded")

        from inference_device import pure_gpu_active

        if (
            pure_gpu_active()
            and isinstance(im_tensor, torch.Tensor)
            and im_tensor.is_cuda
        ):
            im_tensor = im_tensor.contiguous().float()
            io_binding = self._ort_session.io_binding()
            device_id = im_tensor.device.index or 0
            io_binding.bind_input(
                name=self._ort_input_name,
                device_type="cuda",
                device_id=device_id,
                element_type=np.float32,
                shape=tuple(im_tensor.shape),
                buffer_ptr=im_tensor.data_ptr(),
            )
            for out_meta in self._ort_session.get_outputs():
                io_binding.bind_output(out_meta.name, "cuda", device_id)
            self._ort_session.run_with_iobinding(io_binding)
            return io_binding.copy_outputs_to_cpu()

        if isinstance(im_tensor, torch.Tensor):
            im_tensor_np = tensor_to_numpy(im_tensor)
        else:
            im_tensor_np = np.asarray(im_tensor, dtype=np.float32)
        return self._ort_session.run(None, {self._ort_input_name: im_tensor_np})

    def _detect_native(self, frame: np.ndarray) -> List[FaceDetection]:
        if not retinaface_available():
            return []
        t0 = time.perf_counter()
        try:
            from retinaface import RetinaFace
        except ImportError:
            from retina_face import RetinaFace  # type: ignore

        try:
            raw = RetinaFace.detect_faces(
                frame,
                threshold=self.detect_threshold,
            )
        except Exception as exc:
            log.debug("RetinaFace native detect failed: %s", exc)
            return []

        ms = (time.perf_counter() - t0) * 1000.0
        log.debug("RetinaFace native detect %.1f ms (%d faces)", ms, len(raw))

        dets: List[FaceDetection] = []
        for _key, face in raw.items():
            area = face.get("facial_area") or face.get("bbox")
            if not area or len(area) < 4:
                continue
            x1, y1, x2, y2 = (int(area[0]), int(area[1]), int(area[2]), int(area[3]))
            lm_raw = face.get("landmarks") or {}
            landmarks: Dict[str, Tuple[float, float]] = {}
            for name, pt in lm_raw.items():
                if pt and len(pt) >= 2:
                    landmarks[name] = (float(pt[0]), float(pt[1]))
            score = float(face.get("score", 0.0))
            if score < self.detect_threshold:
                continue
            dets.append(
                FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    landmarks=landmarks,
                )
            )
        return dets

    def _detect_onnx(self, frame: np.ndarray) -> List[FaceDetection]:
        """ONNX forward + RetinaFace postprocess (same decode as native package)."""
        if self._ort_session is None or self._ort_input_name is None:
            return self._detect_native(frame)

        t0 = time.perf_counter()
        try:
            from retinaface.commons import postprocess, preprocess

            im_tensor, im_info, im_scale = preprocess.preprocess_image(
                frame, allow_upscaling=True
            )
            outputs = self._ort_session.run(None, {self._ort_input_name: im_tensor})
            net_out = list(outputs)

            threshold = self.detect_threshold
            nms_threshold = 0.4
            decay4 = 0.5
            _feat_stride_fpn = [32, 16, 8]
            _anchors_fpn = {
                "stride32": np.array(
                    [[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]],
                    dtype=np.float32,
                ),
                "stride16": np.array(
                    [[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]],
                    dtype=np.float32,
                ),
                "stride8": np.array(
                    [[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]],
                    dtype=np.float32,
                ),
            }
            _num_anchors = {"stride32": 2, "stride16": 2, "stride8": 2}

            proposals_list: List[np.ndarray] = []
            scores_list: List[np.ndarray] = []
            landmarks_list: List[np.ndarray] = []
            sym_idx = 0

            for s in _feat_stride_fpn:
                scores = net_out[sym_idx]
                scores = scores[:, :, :, _num_anchors[f"stride{s}"] :]
                bbox_deltas = net_out[sym_idx + 1]
                height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]
                a = _num_anchors[f"stride{s}"]
                k = height * width
                anchors_fpn = _anchors_fpn[f"stride{s}"]
                anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
                anchors = anchors.reshape((k * a, 4))
                scores = scores.reshape((-1, 1))
                bbox_stds = [1.0, 1.0, 1.0, 1.0]
                bbox_pred_len = bbox_deltas.shape[3] // a
                bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
                bbox_deltas[:, 0::4] *= bbox_stds[0]
                bbox_deltas[:, 1::4] *= bbox_stds[1]
                bbox_deltas[:, 2::4] *= bbox_stds[2]
                bbox_deltas[:, 3::4] *= bbox_stds[3]
                proposals = postprocess.bbox_pred(anchors, bbox_deltas)
                proposals = postprocess.clip_boxes(proposals, im_info[:2])
                if s == 4 and decay4 < 1.0:
                    scores *= decay4
                order = np.where(scores.ravel() >= threshold)[0]
                proposals = proposals[order, :]
                scores = scores[order]
                proposals[:, 0:4] /= im_scale
                proposals_list.append(proposals)
                scores_list.append(scores)
                landmark_deltas = net_out[sym_idx + 2]
                landmark_pred_len = landmark_deltas.shape[3] // a
                landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len // 5))
                landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]
                landmarks[:, :, 0:2] /= im_scale
                landmarks_list.append(landmarks)
                sym_idx += 3

            if not proposals_list:
                return self._detect_native(frame)

            proposals = np.vstack(proposals_list)
            if proposals.shape[0] == 0:
                return []

            scores = np.vstack(scores_list)
            order = scores.ravel().argsort()[::-1]
            proposals = proposals[order, :]
            scores = scores[order]
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)

            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
            keep = postprocess.cpu_nms(pre_det, nms_threshold)
            det = pre_det[keep, :]
            landmarks = landmarks[keep]

            dets: List[FaceDetection] = []
            for idx, face in enumerate(det):
                score = float(face[4])
                if score < self.detect_threshold:
                    continue
                x1, y1, x2, y2 = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
                lm: Dict[str, Tuple[float, float]] = {}
                names = ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]
                for j, name in enumerate(names):
                    lm[name] = (float(landmarks[idx][j][0]), float(landmarks[idx][j][1]))
                dets.append(
                    FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        score=score,
                        landmarks=lm,
                    )
                )

            ms = (time.perf_counter() - t0) * 1000.0
            log.debug("RetinaFace ONNX detect %.1f ms (%d faces)", ms, len(dets))
            return dets
        except Exception as exc:
            log.warning("ONNX detect failed (%s) — falling back to native", exc)
            return self._detect_native(frame)

    def _detect_onnx_gpu(self, frame: torch.Tensor) -> List[FaceDetection]:
        """ONNX RetinaFace with GPU letterbox preprocess (no full-frame host copy)."""
        if self._ort_session is None or self._ort_input_name is None:
            return self._detect_native(tensor_to_bgr_numpy(frame))

        t0 = time.perf_counter()
        try:
            from retinaface.commons import postprocess

            im_tensor, im_info, im_scale = preprocess_retinaface_gpu(
                frame, allow_upscaling=True
            )
            outputs = self._ort_forward(im_tensor)
            net_out = list(outputs)

            threshold = self.detect_threshold
            nms_threshold = 0.4
            decay4 = 0.5
            _feat_stride_fpn = [32, 16, 8]
            _anchors_fpn = {
                "stride32": np.array(
                    [[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]],
                    dtype=np.float32,
                ),
                "stride16": np.array(
                    [[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]],
                    dtype=np.float32,
                ),
                "stride8": np.array(
                    [[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]],
                    dtype=np.float32,
                ),
            }
            _num_anchors = {"stride32": 2, "stride16": 2, "stride8": 2}

            proposals_list: List[np.ndarray] = []
            scores_list: List[np.ndarray] = []
            landmarks_list: List[np.ndarray] = []
            sym_idx = 0

            for s in _feat_stride_fpn:
                scores = net_out[sym_idx]
                scores = scores[:, :, :, _num_anchors[f"stride{s}"] :]
                bbox_deltas = net_out[sym_idx + 1]
                height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]
                a = _num_anchors[f"stride{s}"]
                k = height * width
                anchors_fpn = _anchors_fpn[f"stride{s}"]
                anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
                anchors = anchors.reshape((k * a, 4))
                scores = scores.reshape((-1, 1))
                bbox_stds = [1.0, 1.0, 1.0, 1.0]
                bbox_pred_len = bbox_deltas.shape[3] // a
                bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
                bbox_deltas[:, 0::4] *= bbox_stds[0]
                bbox_deltas[:, 1::4] *= bbox_stds[1]
                bbox_deltas[:, 2::4] *= bbox_stds[2]
                bbox_deltas[:, 3::4] *= bbox_stds[3]
                proposals = postprocess.bbox_pred(anchors, bbox_deltas)
                proposals = postprocess.clip_boxes(proposals, im_info)
                if s == 4 and decay4 < 1.0:
                    scores *= decay4
                order = np.where(scores.ravel() >= threshold)[0]
                proposals = proposals[order, :]
                scores = scores[order]
                proposals[:, 0:4] /= im_scale
                proposals_list.append(proposals)
                scores_list.append(scores)
                landmark_deltas = net_out[sym_idx + 2]
                landmark_pred_len = landmark_deltas.shape[3] // a
                landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len // 5))
                landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]
                landmarks[:, :, 0:2] /= im_scale
                landmarks_list.append(landmarks)
                sym_idx += 3

            if not proposals_list:
                return self._detect_native(tensor_to_bgr_numpy(frame))

            proposals = np.vstack(proposals_list)
            if proposals.shape[0] == 0:
                return []

            scores = np.vstack(scores_list)
            order = scores.ravel().argsort()[::-1]
            proposals = proposals[order, :]
            scores = scores[order]
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)

            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
            keep = postprocess.cpu_nms(pre_det, nms_threshold)
            det = pre_det[keep, :]
            landmarks = landmarks[keep]

            dets: List[FaceDetection] = []
            for idx, face in enumerate(det):
                score = float(face[4])
                if score < self.detect_threshold:
                    continue
                x1, y1, x2, y2 = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
                lm: Dict[str, Tuple[float, float]] = {}
                names = ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]
                for j, name in enumerate(names):
                    lm[name] = (float(landmarks[idx][j][0]), float(landmarks[idx][j][1]))
                dets.append(
                    FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        score=score,
                        landmarks=lm,
                    )
                )

            ms = (time.perf_counter() - t0) * 1000.0
            log.debug("RetinaFace ONNX GPU detect %.1f ms (%d faces)", ms, len(dets))
            return dets
        except Exception as exc:
            log.warning("ONNX GPU detect failed (%s) — falling back to native", exc)
            return self._detect_native(tensor_to_bgr_numpy(frame))

    @staticmethod
    def _canonical_face_landmarks(
        landmarks: Optional[Dict[str, Tuple[float, float]]],
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """Normalize RetinaFace / InsightFace landmark keys for 5-point align."""
        if not landmarks:
            return None
        src = dict(landmarks)
        if "mouth_right" in src and "right_mouth" not in src:
            src["right_mouth"] = src["mouth_right"]
        if "mouth_left" in src and "left_mouth" not in src:
            src["left_mouth"] = src["mouth_left"]
        need = ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")
        if not all(name in src for name in need):
            return None
        return {name: (float(src[name][0]), float(src[name][1])) for name in need}

    @staticmethod
    def _landmarks_in_crop(
        landmarks: Dict[str, Tuple[float, float]],
        x1: int,
        y1: int,
    ) -> Dict[str, Tuple[float, float]]:
        return {
            name: (float(pt[0]) - x1, float(pt[1]) - y1)
            for name, pt in landmarks.items()
        }

    def embed(self, frame: np.ndarray, det: FaceDetection) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = det.bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
        # Match enroll / reference path: pad crop, then 5-point ArcFace align.
        pad = int(0.15 * max(x2 - x1, y2 - y1))
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
        crop = frame[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            return None
        canon = self._canonical_face_landmarks(det.landmarks)
        if canon is not None:
            local_lm = self._landmarks_in_crop(canon, x1p, y1p)
            emb = self.embed_crop_recognition_only(
                crop,
                landmarks=local_lm,
                align_size=112,
            )
            if emb is not None:
                return emb
        return self.embed_crop(crop)

    def embed_crop(
        self, face_bgr: np.ndarray, *, apply_clahe: bool = True
    ) -> Optional[np.ndarray]:
        """ArcFace embedding on a pre-cropped BGR face (CLAHE applied first)."""
        if face_bgr is None or face_bgr.size == 0:
            return None

        # မြန်မာ: InsightFace GPU recognition path (CUDAExecutionProvider)
        if self._insightface_app is not None:
            enhanced = apply_face_clahe(face_bgr) if apply_clahe else face_bgr
            t0 = time.perf_counter()
            try:
                faces = self._insightface_app.get(enhanced)
                if not faces:
                    return None
                emb = np.asarray(faces[0].normed_embedding, dtype=np.float32).reshape(-1)
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    emb = emb / norm
                ms = (time.perf_counter() - t0) * 1000.0
                log.debug("InsightFace ArcFace embed %.1f ms", ms)
                return emb
            except Exception as exc:
                log.debug("InsightFace embed failed: %s", exc)
                return None

        if not deepface_available():
            return None

        enhanced = apply_face_clahe(face_bgr) if apply_clahe else face_bgr
        t0 = time.perf_counter()
        try:
            from deepface import DeepFace

            reps = DeepFace.represent(
                img_path=enhanced,
                model_name=ARCFACE_MODEL,
                detector_backend="skip",
                enforce_detection=False,
                align=True,
            )
            if not reps:
                return None
            emb = np.asarray(reps[0]["embedding"], dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm
            ms = (time.perf_counter() - t0) * 1000.0
            log.debug("ArcFace embed %.1f ms", ms)
            return emb
        except Exception as exc:
            log.debug("ArcFace embed failed: %s", exc)
            return None

    def prepare_arcface_input(
        self,
        face_bgr: np.ndarray,
        *,
        landmarks: Optional[Dict[str, Tuple[float, float]]] = None,
        apply_clahe: bool = True,
        align_size: int = 112,
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        Build the BGR tensor fed to InsightFace recognition (112×112 aligned).

        Returns (aligned_bgr_hwc, onnx_nchw_shape).
        """
        if face_bgr is None or face_bgr.size == 0:
            return None, (0, 0, 0, 0)

        if self._insightface_app is not None:
            rec = self._insightface_app.models.get("recognition")
            if rec is not None:
                from insightface.utils import face_align

                enhanced = apply_face_clahe(face_bgr) if apply_clahe else face_bgr
                size = int(align_size or rec.input_size[0])
                kps = self._landmarks_to_kps(landmarks, enhanced.shape)
                if kps is not None:
                    aimg = face_align.norm_crop(
                        enhanced,
                        landmark=kps,
                        image_size=size,
                    )
                else:
                    aimg = cv2.resize(
                        enhanced,
                        (size, size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                if aimg.shape[0] != size or aimg.shape[1] != size:
                    aimg = cv2.resize(
                        aimg,
                        (size, size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                onnx_shape = (1, 3, size, size)
                return aimg, onnx_shape

        size = int(align_size)
        aimg = cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
        return aimg, (1, 3, size, size)

    @staticmethod
    def describe_arcface_recognition_model(analyzer: Optional["FaceAnalyzer"]) -> str:
        """Diagnostic: which recognition ONNX session backs ArcFace embeds."""
        if analyzer is None:
            return "unavailable (no FaceAnalyzer)"
        if analyzer._insightface_app is not None:
            rec = analyzer._insightface_app.models.get("recognition")
            if rec is None:
                return "insightface/buffalo_l (recognition model missing)"
            model_file = getattr(rec, "model_file", None) or getattr(
                rec, "onnx_file", None
            )
            input_size = getattr(rec, "input_size", None)
            providers = []
            session = getattr(rec, "session", None)
            if session is not None:
                try:
                    providers = list(session.get_providers())
                except Exception:
                    pass
            return (
                f"insightface/buffalo_l recognition "
                f"weights={model_file or 'w600k_r50.onnx'} "
                f"input_size={input_size} providers={providers}"
            )
        if analyzer._recognizer is not None:
            return "opencv_sface onnx=face_recognition_sface_2021dec.onnx"
        return "deepface_arcface_fallback"

    def embed_crop_recognition_only(
        self,
        face_bgr: np.ndarray,
        *,
        landmarks: Optional[Dict[str, Tuple[float, float]]] = None,
        apply_clahe: bool = True,
        align_size: int = 112,
        debug_out: Optional[Dict[str, object]] = None,
    ) -> Optional[np.ndarray]:
        """
        ArcFace on a full-resolution crop without re-running RetinaFace detect.

        Used by video scan: detection at 640w, embed at native source resolution.
        """
        if face_bgr is None or face_bgr.size == 0:
            return None

        if self._insightface_app is not None:
            rec = self._insightface_app.models.get("recognition")
            if rec is not None:
                t0 = time.perf_counter()
                try:
                    aimg, onnx_shape = self.prepare_arcface_input(
                        face_bgr,
                        landmarks=landmarks,
                        apply_clahe=apply_clahe,
                        align_size=align_size,
                    )
                    if aimg is None:
                        return None
                    if debug_out is not None:
                        debug_out["aligned_bgr"] = aimg.copy()
                        debug_out["arcface_input_shape"] = onnx_shape
                        debug_out["arcface_model"] = (
                            FaceAnalyzer.describe_arcface_recognition_model(self)
                        )
                    emb = rec.get_feat(aimg).reshape(-1).astype(np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        emb = emb / norm
                    ms = (time.perf_counter() - t0) * 1000.0
                    log.debug("InsightFace ArcFace recognition-only %.1f ms", ms)
                    return emb
                except Exception as exc:
                    log.debug("InsightFace recognition-only embed failed: %s", exc)

        return self.embed_crop(face_bgr, apply_clahe=apply_clahe)

    @staticmethod
    def _landmarks_to_kps(
        landmarks: Optional[Dict[str, Tuple[float, float]]],
        shape: Tuple[int, ...],
    ) -> Optional[np.ndarray]:
        if not landmarks:
            return None
        canon = FaceAnalyzer._canonical_face_landmarks(landmarks)
        if canon is None:
            return None
        order = (
            "left_eye",
            "right_eye",
            "nose",
            "left_mouth",
            "right_mouth",
        )
        pts = [[float(canon[name][0]), float(canon[name][1])] for name in order]
        return np.asarray(pts, dtype=np.float32)

    def embed_gpu(self, frame: torch.Tensor, det: FaceDetection) -> Optional[np.ndarray]:
        if det.embedding is not None:
            return det.embedding
        crop = crop_xyxy(frame, det.bbox)
        if crop is None or crop.numel() == 0:
            return None
        crop_bgr = tensor_to_bgr_numpy(crop)
        x1, y1, x2, y2 = det.bbox
        canon = self._canonical_face_landmarks(det.landmarks)
        if canon is not None:
            local_lm = self._landmarks_in_crop(canon, int(x1), int(y1))
            emb = self.embed_crop_recognition_only(
                crop_bgr,
                landmarks=local_lm,
                align_size=112,
            )
            if emb is not None:
                return emb
        return self.embed_crop(crop_bgr)

    def detect_and_embed(self, frame: np.ndarray) -> List[FaceDetection]:
        results: List[FaceDetection] = []
        for det in self.detect(frame):
            ok, reason = _is_usable_face(
                det,
                frame,
                min_size=self.min_size,
                max_blur=self.max_blur,
                max_yaw=self.max_yaw,
                min_confidence=self.detect_threshold,
            )
            if not ok:
                det.quality_ok = False
                det.reject_reason = reason
                log.debug("Face rejected: %s", reason)
                results.append(det)
                continue

            if self._anti_spoof is not None and self._anti_spoof.enabled:
                is_real, conf = self._anti_spoof.is_real(frame, det.bbox)
                det.anti_spoof_score = conf
                if not is_real:
                    det.quality_ok = False
                    det.reject_reason = f"spoof({conf:.2f})"
                    log.debug("Face rejected: %s", det.reject_reason)
                    results.append(det)
                    continue

            emb = self.embed(frame, det)
            if emb is None and det.embedding is not None:
                emb = det.embedding
            if emb is not None:
                det.embedding = emb
            else:
                det.quality_ok = False
                det.reject_reason = "embed_failed"
            results.append(det)

        try:
            from face_print_filter import filter_face_dets_reject_prints

            results = filter_face_dets_reject_prints(frame, results)
        except Exception:
            pass
        try:
            from face_stats import record_face_detections_batch

            record_face_detections_batch(results)
        except Exception:
            pass
        return results

    def detect_and_embed_scan(self, frame: np.ndarray) -> List[FaceDetection]:
        """Video scan / live CCTV path — relaxed detect/embed for hallway faces."""
        from face_landmark_verify import verify_landmarks_for_capture

        results: List[FaceDetection] = []
        for det in self.detect(frame):
            ok, reason = _is_usable_face_scan(
                det,
                frame,
                min_size=max(16, int(self.min_size)),
                max_blur=min(float(self.max_blur), SCAN_MAX_BLUR),
                min_confidence=min(float(self.detect_threshold), SCAN_DETECT_MIN_CONFIDENCE),
            )
            if not ok:
                log.debug("Scan face rejected: %s", reason)
                continue

            # Soft pose gate: keep frontal + side + looking-down; drop clear backs.
            if det.landmarks:
                ok_lm, lm_reason = verify_landmarks_for_capture(
                    det.landmarks,
                    det.bbox,
                    detection_confidence=float(det.score),
                    min_confidence=0.18,
                    allow_side_view=True,
                    max_yaw=max(float(self.max_yaw), SCAN_MAX_YAW),
                    allow_high_angle=True,
                )
                # Soft-accept noisy hallway landmarks, but never looking_down —
                # that path was letting walking-away / crown crops through.
                soft_ok = (
                    str(lm_reason).startswith("missing_")
                    or str(lm_reason).startswith("eye_span_ratio")
                    or str(lm_reason).startswith("mouth_too_narrow")
                )
                if not ok_lm and not soft_ok:
                    log.debug("Scan face pose rejected: %s", lm_reason)
                    continue

            emb = det.embedding
            if emb is None:
                emb = self.embed(frame, det)
            if emb is None:
                log.debug("Scan embed failed for bbox %s", det.bbox)
                continue

            # Optional anti-spoof when weights are present (scan path used to skip it).
            if self._anti_spoof is not None and self._anti_spoof.enabled:
                is_real, conf = self._anti_spoof.is_real(frame, det.bbox)
                det.anti_spoof_score = conf
                if not is_real:
                    log.debug("Scan face rejected: spoof(%.2f)", conf)
                    continue

            try:
                from face_print_filter import is_likely_invalid_face_crop

                x1, y1, x2, y2 = det.bbox
                crop = frame[y1:y2, x1:x2]
                bad_crop, crop_reason = is_likely_invalid_face_crop(crop)
                if bad_crop:
                    log.debug("Scan face rejected: %s", crop_reason)
                    continue
            except Exception:
                pass

            det.embedding = emb
            det.quality_ok = True
            results.append(det)

        # Wall-print filter after all faces are known (group portraits need neighbors).
        try:
            from face_print_filter import filter_face_dets_reject_prints

            results = filter_face_dets_reject_prints(frame, results)
        except Exception:
            pass
        return results

    def detect_and_embed_gpu(self, frame: torch.Tensor) -> List[FaceDetection]:
        results: List[FaceDetection] = []
        for det in self.detect_gpu(frame):
            ok, reason = _is_usable_face_gpu(
                det,
                frame,
                min_size=self.min_size,
                max_blur=self.max_blur,
                max_yaw=self.max_yaw,
                min_confidence=self.detect_threshold,
            )
            if not ok:
                det.quality_ok = False
                det.reject_reason = reason
                log.debug("Face rejected: %s", reason)
                results.append(det)
                continue

            if self._anti_spoof is not None and self._anti_spoof.enabled:
                is_real, conf = self._anti_spoof.is_real_gpu(frame, det.bbox)
                det.anti_spoof_score = conf
                if not is_real:
                    det.quality_ok = False
                    det.reject_reason = f"spoof({conf:.2f})"
                    log.debug("Face rejected: %s", det.reject_reason)
                    results.append(det)
                    continue

            emb = self.embed_gpu(frame, det)
            if emb is not None:
                det.embedding = emb
            else:
                det.quality_ok = False
                det.reject_reason = "embed_failed"
            results.append(det)

        try:
            from face_print_filter import filter_face_dets_reject_prints

            results = filter_face_dets_reject_prints(tensor_to_bgr_numpy(frame), results)
        except Exception:
            pass
        try:
            from face_stats import record_face_detections_batch

            record_face_detections_batch(results)
        except Exception:
            pass
        return results


# ---------------------------------------------------------------------------
# Legacy helper (preserved API)
# ---------------------------------------------------------------------------


def _crop_person(frame: np.ndarray, bbox: np.ndarray, pad: float = 0.08) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = frame.shape[:2]
    pad_px = int(pad * max(x2 - x1, y2 - y1))
    x1, y1 = max(0, x1 - pad_px), max(0, y1 - pad_px)
    x2, y2 = min(w, x2 + pad_px), min(h, y2 + pad_px)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def _face_in_person_bbox(
    face_bbox: Tuple[int, int, int, int],
    person_bbox: np.ndarray,
    min_overlap: float = 0.3,
) -> bool:
    fx1, fy1, fx2, fy2 = face_bbox
    px1, py1, px2, py2 = person_bbox.astype(float)
    ix1 = max(fx1, px1)
    iy1 = max(fy1, py1)
    ix2 = min(fx2, px2)
    iy2 = min(fy2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    face_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
    return (inter / face_area) >= min_overlap


def _face_person_match_score(
    face_bbox: Tuple[int, int, int, int],
    person_bbox: np.ndarray,
    *,
    outdoor: bool = False,
) -> float:
    """Higher = better unique match. Prefers faces in the head/upper body band."""
    if not _face_in_person_bbox(face_bbox, person_bbox):
        return -1.0
    try:
        from face_landmark_verify import face_bbox_in_person_head_band

        ok, _reason = face_bbox_in_person_head_band(
            face_bbox, person_bbox, outdoor=outdoor
        )
        if not ok:
            return -1.0
    except Exception:
        pass
    fx1, fy1, fx2, fy2 = (float(v) for v in face_bbox[:4])
    px1, py1, px2, py2 = (float(v) for v in np.asarray(person_bbox).reshape(-1)[:4])
    face_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
    fcy = (fy1 + fy2) * 0.5
    band = max(1.0, py2 - py1)
    # Head is typically in the top ~55% of a standing person box.
    rel_y = (fcy - py1) / band
    head_bonus = 1.35 if rel_y <= 0.55 else (0.85 if rel_y <= 0.75 else 0.35)
    # Prefer face center near horizontal middle of the person box.
    fcx = (fx1 + fx2) * 0.5
    pcx = (px1 + px2) * 0.5
    width = max(1.0, px2 - px1)
    center_bonus = 1.0 - min(1.0, abs(fcx - pcx) / width)
    return face_area * head_bonus * (0.65 + 0.35 * center_bonus)


def assign_unique_faces_to_persons(
    person_bboxes: List[np.ndarray],
    face_dets: List["FaceDetection"],
) -> Dict[int, "FaceDetection"]:
    """
    1:1 face→person assignment so two people side-by-side each keep their own face.

    Without this, the largest overlapping face is greedily given to every person
    box that overlaps it — the second person never gets a crop.
    """
    usable = [
        d
        for d in (face_dets or [])
        if getattr(d, "quality_ok", False)
        and getattr(d, "embedding", None) is not None
        and getattr(d, "bbox", None) is not None
    ]
    if not person_bboxes or not usable:
        return {}

    pairs: List[Tuple[float, int, int]] = []
    for pi, pb in enumerate(person_bboxes):
        for fi, det in enumerate(usable):
            score = _face_person_match_score(det.bbox, pb)
            if score > 0:
                pairs.append((score, pi, fi))
    pairs.sort(key=lambda item: item[0], reverse=True)

    assigned: Dict[int, FaceDetection] = {}
    used_faces: set[int] = set()
    used_persons: set[int] = set()
    for score, pi, fi in pairs:
        if pi in used_persons or fi in used_faces:
            continue
        assigned[pi] = usable[fi]
        used_persons.add(pi)
        used_faces.add(fi)
    return assigned


def describe_arcface_recognition_model(analyzer: Optional[FaceAnalyzer]) -> str:
    return FaceAnalyzer.describe_arcface_recognition_model(analyzer)


def extract_face_from_person_crop(
    frame: np.ndarray,
    bbox: np.ndarray,
    min_face_px: int = MIN_FACE_PX,
    analyzer: Optional[FaceAnalyzer] = None,
    source_frame: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Detect a face inside a YOLO person box and return an ArcFace embedding.
    Returns (embedding, hi-res face_bgr, face_area).
    """
    if not analyzer_ready():
        return None

    fa = analyzer or FaceAnalyzer()
    host = source_frame if source_frame is not None else frame
    crop = _crop_person(frame, bbox)
    if crop is None:
        return None

    t0 = time.perf_counter()
    # Live CCTV: use relaxed scan gate inside the person crop (strict studio
    # detect_and_embed rejects most hallway faces).
    dets = fa.detect_and_embed_scan(crop)
    ms = (time.perf_counter() - t0) * 1000.0
    log.debug("extract_face_from_person_crop %.1f ms", ms)

    usable = [d for d in dets if d.quality_ok and d.embedding is not None]
    if not usable:
        return None

    best = max(
        usable,
        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
    )
    from captures import crop_face_from_frame, map_bbox_from_person_crop_to_source

    abs_bbox = map_bbox_from_person_crop_to_source(
        bbox,
        best.bbox,
        host.shape,
        crop_shape=crop.shape[:2],
        upscale_min_side=0,
    )
    aligned = crop_face_from_frame(host, abs_bbox)
    if aligned is None or aligned.size == 0:
        x1, y1, x2, y2 = best.bbox
        aligned = crop[y1:y2, x1:x2]
        if aligned.size == 0:
            aligned = crop
    fw, fh = abs_bbox[2] - abs_bbox[0], abs_bbox[3] - abs_bbox[1]
    if max(fw, fh) < min_face_px:
        return None
    return best.embedding, aligned, float(fw * fh)


def extract_face_from_person_crop_gpu(
    frame: torch.Tensor,
    bbox: torch.Tensor,
    min_face_px: int = MIN_FACE_PX,
    analyzer: Optional[FaceAnalyzer] = None,
    source_frame: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    VRAM path: crop person on GPU, detect/embed on GPU tensors.
    Returns (embedding, hi-res face_bgr_numpy, face_area).
    """
    if not analyzer_ready():
        return None

    fa = analyzer or FaceAnalyzer()
    host = source_frame if source_frame is not None else tensor_to_bgr_numpy(frame)
    host_bbox = tensor_to_numpy(bbox)
    crop = crop_xyxy(frame, bbox, pad_ratio=0.08)
    if crop is None or crop.numel() == 0:
        return None

    t0 = time.perf_counter()
    dets = fa.detect_and_embed_gpu(crop)
    ms = (time.perf_counter() - t0) * 1000.0
    log.debug("extract_face_from_person_crop_gpu %.1f ms", ms)

    usable = [d for d in dets if d.quality_ok and d.embedding is not None]
    if not usable:
        return None

    best = max(
        usable,
        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
    )
    from captures import crop_face_from_frame, map_bbox_from_person_crop_to_source

    crop_np = tensor_to_bgr_numpy(crop)
    abs_bbox = map_bbox_from_person_crop_to_source(
        host_bbox,
        best.bbox,
        host.shape,
        crop_shape=crop_np.shape[:2],
        upscale_min_side=0,
    )
    aligned = crop_face_from_frame(host, abs_bbox)
    if aligned is None or aligned.size == 0:
        x1, y1, x2, y2 = best.bbox
        aligned_t = crop_xyxy(crop, (x1, y1, x2, y2))
        if aligned_t is None or aligned_t.numel() == 0:
            aligned_t = crop
        aligned = tensor_to_bgr_numpy(aligned_t)
    fw, fh = abs_bbox[2] - abs_bbox[0], abs_bbox[3] - abs_bbox[1]
    if max(fw, fh) < min_face_px:
        return None
    return best.embedding, aligned, float(fw * fh)


def match_face_to_person_detections(
    frame: Optional[np.ndarray],
    person_bbox: np.ndarray,
    face_dets: List[FaceDetection],
) -> Optional[FaceDetection]:
    """Pick the best quality-gated face overlapping a YOLO person box."""
    del frame  # bbox overlap only; frame kept for API compatibility
    matches = [
        d
        for d in face_dets
        if d.quality_ok
        and d.embedding is not None
        and _face_in_person_bbox(d.bbox, person_bbox)
    ]
    if not matches:
        return None
    return max(
        matches,
        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
    )
