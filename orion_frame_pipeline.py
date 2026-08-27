"""
Orion Tactical — Production-grade RTSP frame processing pipeline.

မြန်မာ: CCTV RTSP စတရီးမများမှ frame များကို လုံခြုံစွာ ဖတ်ယူပြီး
RetinaFace + ArcFace inference လုပ်ငန်းစဉ်ကို စနစ်တကျ စီမံခန့်ခွဲပေးသော မော်ဂျူး။
"""

from __future__ import annotations

import gc
import logging
import os
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import cv2
import numpy as np

# ── Tier thresholds (similarity 0.0–1.0, higher = stronger match) ──────────
# Override with TELEGRAM_ALERT_MIN_PERCENT or WATCHLIST_ALERT_MIN_PERCENT (0–100).
def _env_alert_similarity_floor(default: float = 0.50) -> float:
    raw = (
        os.environ.get("TELEGRAM_ALERT_MIN_PERCENT")
        or os.environ.get("WATCHLIST_ALERT_MIN_PERCENT")
        or ""
    ).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value > 1.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


BLACKLIST_ALERT_SIMILARITY = _env_alert_similarity_floor(0.50)
# Gray-zone band sits just below the live alert floor.
GRAY_ZONE_SIMILARITY_MAX = max(0.0, BLACKLIST_ALERT_SIMILARITY - 0.01)
GRAY_ZONE_SIMILARITY_MIN = max(0.0, GRAY_ZONE_SIMILARITY_MAX - 0.09)

# မြန်မာ: RTSP ပြတ်တောက်ပါက ပြန်ချိတ်မီ စောင့်ချိန် (စက္ကန့်)
RTSP_RECONNECT_WAIT_SEC = 2.0

# မြန်မာ: infinite loop တွင် gc.collect() ခေါ်မည့် frame ကြားအကွာအဝေး
MEMORY_GC_INTERVAL_FRAMES = 30

T = TypeVar("T")

_logger = logging.getLogger("orion.pipeline")


# ── Hardware dependency audit (entry-point diagnostics) ─────────────────────


def audit_hardware_dependencies(
    *,
    device_choice: str = "auto",
    hybrid_face_on_cpu: bool = False,
    pure_gpu: bool = False,
) -> str:
    """
    Entry-point diagnostic: CUDA visibility, cuDNN, ONNX providers, face GPU stack.

    Call once before the infinite processing loop. Returns resolved torch device string.
    """
    from inference_device import configure, describe_device, get_torch_device, normalize_device_choice

    want_cuda = normalize_device_choice(device_choice).startswith("cuda")
    resolved = configure(
        device_choice,
        hybrid_face_on_cpu=hybrid_face_on_cpu,
        pure_gpu=pure_gpu or want_cuda,
    )
    print("=" * 60)
    print("ORION TACTICAL — Hardware Dependency Audit")
    print("=" * 60)
    print(f"Resolved inference: {describe_device()}")
    print(f"Torch device string: {get_torch_device()}")

    try:
        import onnxruntime as ort

        print(f"ONNX Runtime providers: {ort.get_available_providers()}")
    except ImportError:
        print("ONNX Runtime providers: unavailable (install onnxruntime-gpu)")

    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception:
        pass

    try:
        from face_analyzer import _try_load_insightface_cuda, active_pipeline_name

        if resolved.startswith("cuda"):
            _try_load_insightface_cuda()
        print(f"Face pipeline backend: {active_pipeline_name()}")
    except Exception as exc:
        print(f"Face pipeline backend: unavailable ({exc})")

    try:
        from identity import gpu_scan_ready

        if resolved.startswith("cuda"):
            ready = gpu_scan_ready()
            print(f"GPU scan ready: {ready}")
        else:
            print("GPU scan ready: False (CUDA not active)")
    except Exception as exc:
        print(f"GPU scan ready: False ({exc})")

    try:
        from face_restoration import active_backend_name, weights_ready

        if resolved.startswith("cuda"):
            if weights_ready():
                print(f"Face restoration: weights present ({active_backend_name()})")
            else:
                print(
                    "Face restoration: weights missing — "
                    "run: python download_face_restoration.py --no-proxy"
                )
    except Exception as exc:
        print(f"Face restoration: unavailable ({exc})")

    print("=" * 60)
    return resolved


# ── Frame memory management (RAM leak prevention) ───────────────────────────


def release_frame_buffers(*refs: Any) -> None:
    """
    Force-release ndarray/tensor references after a frame is processed.

    Clears face crops, bbox tuples, intermediate detection lists, and host frames
    so Python can reclaim buffers before the next RTSP read.
    """
    for ref in refs:
        if ref is None:
            continue
        if isinstance(ref, (list, dict, set)):
            ref.clear()
        elif isinstance(ref, np.ndarray):
            del ref
        else:
            try:
                del ref
            except Exception:
                pass


class FrameMemoryManager:
    """
    Periodic garbage collection inside the infinite surveillance loop.

    Keeps RSS stable by explicitly collecting unreachable frame buffers every
    MEMORY_GC_INTERVAL_FRAMES iterations (target: < 8 GB under normal load).
    """

    def __init__(self, gc_interval: int = MEMORY_GC_INTERVAL_FRAMES) -> None:
        self.gc_interval = max(1, int(gc_interval))
        self._last_gc_frame = 0

    def tick(self, frame_index: int) -> None:
        if frame_index <= 0:
            return
        if frame_index - self._last_gc_frame < self.gc_interval:
            return
        self._last_gc_frame = frame_index
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def init_gpu_face_stack() -> Optional[Any]:
    """
    Initialize InsightFace on NVIDIA CUDA (RetinaFace detect + ArcFace embed).

    providers=['CUDAExecutionProvider'] with CPUExecutionProvider as ONNX fallback
    is configured in face_analyzer / inference_device — similarity thresholds are
    unchanged after this context switch (still 85% alert / 75–84% gray-zone).
    """
    try:
        from inference_device import torch_cuda_active
        from face_analyzer import _try_load_insightface_cuda

        if not torch_cuda_active():
            print(
                "WARNING: CUDA inactive — face inference stays on CPU until "
                "--device cuda is used with onnxruntime-gpu installed."
            )
            return None
        return _try_load_insightface_cuda()
    except Exception as exc:
        _logger.warning("GPU face stack init failed: %s", exc)
        print(f"WARNING: GPU face stack init failed: {exc}")
        return None


def init_face_restoration_stack() -> Optional[Any]:
    """
    Initialize GFPGAN / CodeFormer blind face restoration on CUDA.

    Stage 2 of the super-resolution pipeline — runs before ArcFace embedding.
    """
    try:
        from inference_device import torch_cuda_active
        from face_restoration import get_face_restoration_engine

        if not torch_cuda_active():
            print(
                "WARNING: Face restoration disabled — CUDA inactive "
                "(GFPGAN/CodeFormer require device='cuda')."
            )
            return None
        return get_face_restoration_engine(initialize=True)
    except Exception as exc:
        _logger.warning("Face restoration init failed: %s", exc)
        print(f"WARNING: Face restoration init failed: {exc}")
        return None


class MatchTier(str, Enum):
    """Identity match decision tier."""

    BLACKLIST_ALERT = "blacklist_alert"
    GRAY_ZONE = "gray_zone"
    IGNORE = "ignore"


@dataclass(frozen=True)
class TierDecision:
    """Result of two-tier threshold evaluation."""

    tier: MatchTier
    similarity: float
    match_percent: int
    person_id: Optional[int]
    distance: float


class AsyncWarningLogger:
    """
    မြန်မာ: warning များကို မူလ loop ကို မနှောင့်နှေးစေဘဲ background thread မှတဆင့် မှတ်တမ်းတင်သည်။
    """

    def __init__(self, max_queue: int = 512) -> None:
        self._queue: queue.Queue[Tuple[str, str]] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="orion-async-logger",
        )
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                level, message = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if level == "warning":
                _logger.warning(message)
                print(f"[WARN] {message}")
            elif level == "info":
                _logger.info(message)
                print(f"[INFO] {message}")
            self._queue.task_done()

    def warn(self, message: str) -> None:
        try:
            self._queue.put_nowait(("warning", message))
        except queue.Full:
            print(f"[WARN] {message}")

    def info(self, message: str) -> None:
        try:
            self._queue.put_nowait(("info", message))
        except queue.Full:
            print(f"[INFO] {message}")

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


# မြန်မာ: လုပ်ငန်းစဉ်တစ်ခုလုံးအတွက် shared async logger
_async_logger = AsyncWarningLogger()


def get_async_logger() -> AsyncWarningLogger:
    return _async_logger


def distance_to_similarity(distance: float) -> float:
    """Convert cosine distance to similarity score in [0, 1]."""
    if not np.isfinite(distance):
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(distance)))


def similarity_to_percent(similarity: float) -> int:
    return int(round(max(0.0, min(1.0, similarity)) * 100))


class FrameValidator:
    """မြန်မာ: OpenCV cap.read() မှ ပြန်လာသော frame ကို inference မတိုင်မီ စစ်ဆေးသည်။"""

    @staticmethod
    def is_valid(ret: bool, frame: Optional[np.ndarray]) -> bool:
        if not ret:
            return False
        if frame is None:
            return False
        try:
            return int(frame.size) > 0
        except (AttributeError, TypeError, ValueError):
            return False


class RtspAutoReconnect:
    """
    မြန်မာ: RTSP ပြတ်တောက်မှု / ဗလာ frame ကို တွေ့ရင် ၂ စက္ကန့် စောင့်ပြီး
    cv2.VideoCapture ကို ပြန်ဖွင့်သည် — infinite loop ကို မရပ်ပါ။
    """

    def __init__(
        self,
        source: Union[int, str],
        *,
        rtsp_tcp: bool = False,
        open_capture_fn: Optional[Callable[..., cv2.VideoCapture]] = None,
        location: str = "camera",
        wait_sec: float = RTSP_RECONNECT_WAIT_SEC,
        ffmpeg_lock: Optional[threading.RLock] = None,
    ) -> None:
        self.source = source
        self.rtsp_tcp = rtsp_tcp
        self.location = location
        self.wait_sec = wait_sec
        self._open_capture_fn = open_capture_fn
        self._ffmpeg_lock = ffmpeg_lock
        self.cap: Optional[cv2.VideoCapture] = None

    def _open(self) -> cv2.VideoCapture:
        if self._open_capture_fn is None:
            raise RuntimeError("open_capture_fn is required for RtspAutoReconnect")
        if self._ffmpeg_lock is not None:
            with self._ffmpeg_lock:
                return self._open_capture_fn(self.source, rtsp_tcp=self.rtsp_tcp)
        return self._open_capture_fn(self.source, rtsp_tcp=self.rtsp_tcp)

    def ensure_connected(self) -> bool:
        """မြန်မာ: capture ဖွင့်ထားပြီး အဆင်သင့် ဖြစ်ကြောင်း သေချာစေသည်။"""
        if self.cap is not None and self.cap.isOpened():
            return True
        try:
            self.cap = self._open()
            print(f"[CAM] {self.location}: stream reconnected")
            return True
        except Exception as exc:
            _async_logger.warn(
                f"{self.location}: RTSP reconnect failed ({exc}) — retrying in {self.wait_sec:.0f}s"
            )
            time.sleep(self.wait_sec)
            return False

    def read_validated(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        မြန်မာ: frame တစ်ခုဖတ်ပြီး မမှန်ပါက စောင့်ပြီး ပြန်ချိတ်သည်။
        Returns (ok, frame) — ok=False ဆိုရင် caller က continue လုပ်ရမည်။
        """
        if not self.ensure_connected():
            return False, None

        assert self.cap is not None
        try:
            if self._ffmpeg_lock is not None:
                with self._ffmpeg_lock:
                    ret, frame = self.cap.read()
            else:
                ret, frame = self.cap.read()
        except cv2.error as exc:
            _async_logger.warn(f"{self.location}: OpenCV read error ({exc})")
            ret, frame = False, None

        if FrameValidator.is_valid(ret, frame):
            return True, frame

        _async_logger.warn(
            f"{self.location}: invalid frame (ret={ret}) — "
            f"waiting {self.wait_sec:.0f}s then reconnecting"
        )
        self.release()
        time.sleep(self.wait_sec)
        return False, None

    def release(self) -> None:
        if self.cap is None:
            return
        try:
            if self._ffmpeg_lock is not None:
                with self._ffmpeg_lock:
                    self.cap.release()
            else:
                self.cap.release()
        except Exception:
            pass
        self.cap = None


class SafeInferenceRunner:
    """
    မြန်မာ: detection / recognition inference ကို IndexError, TypeError မှ ကာကွယ်ပြီး
    crash မဖြစ်စေဘဲ နောက် frame သို့ ဆက်သွားစေသည်။
    """

    @staticmethod
    def run(
        fn: Callable[[], T],
        *,
        context: str = "frame inference",
        on_skip: Optional[Callable[[], None]] = None,
    ) -> Optional[T]:
        try:
            return fn()
        except (IndexError, TypeError) as exc:
            _async_logger.warn(
                f"{context}: skipped empty/invalid detection result ({type(exc).__name__}: {exc})"
            )
            if on_skip is not None:
                on_skip()
            return None
        except Exception as exc:
            _async_logger.warn(
                f"{context}: unexpected error ({type(exc).__name__}: {exc})"
            )
            if on_skip is not None:
                on_skip()
            return None


def safe_match_score(score: Any) -> float:
    """
    မြန်မာ: OpenCV recognizer.match() ရလဒ်ကို ဘေးကင်းစွာ float အဖြစ် ပြောင်းသည်။
    tuple index out of range ကို ကာကွယ်သည်။
    """
    if score is None:
        return float("nan")
    if isinstance(score, (float, int)):
        return float(score)
    try:
        if hasattr(score, "__len__") and len(score) == 0:
            return float("nan")
        return float(score[0])
    except (IndexError, TypeError, ValueError):
        return float("nan")


class IdentityTierRouter:
    """
    မြန်မာ: Euclidean/cosine distance အပေါ် အခြေခံ၍ နှစ်ဆင့် ဆုံးဖြတ်ချက် —
    Tier 1 [ALERT] ≥ 85%, Tier 2 [GRAY-ZONE] 75–84%, အခြား ignore။
    """

    @staticmethod
    def evaluate(
        person_id: Optional[int],
        distance: float,
    ) -> TierDecision:
        similarity = distance_to_similarity(distance)
        pct = similarity_to_percent(similarity)

        if person_id is not None and similarity >= BLACKLIST_ALERT_SIMILARITY:
            tier = MatchTier.BLACKLIST_ALERT
        elif (
            person_id is not None
            and GRAY_ZONE_SIMILARITY_MIN <= similarity <= GRAY_ZONE_SIMILARITY_MAX
        ):
            tier = MatchTier.GRAY_ZONE
        else:
            tier = MatchTier.IGNORE

        return TierDecision(
            tier=tier,
            similarity=similarity,
            match_percent=pct,
            person_id=person_id,
            distance=distance,
        )

    @staticmethod
    def log_decision(
        decision: TierDecision,
        camera_location: str,
        *,
        async_logger: Optional[AsyncWarningLogger] = None,
    ) -> None:
        log = async_logger or _async_logger
        pid = decision.person_id
        pct = decision.match_percent

        if decision.tier == MatchTier.BLACKLIST_ALERT:
            log.info(
                f"[ALERT] Blacklist match Person #{pid} @ {camera_location} "
                f"— similarity {decision.similarity:.2f} ({pct}%)"
            )
        elif decision.tier == MatchTier.GRAY_ZONE:
            log.warn(
                f"[GRAY-ZONE] High-resemblance Person #{pid} @ {camera_location} "
                f"— similarity {decision.similarity:.2f} ({pct}%) — audit only, no alarm"
            )


def init_gray_zone_audit_db(conn: sqlite3.Connection) -> None:
    """မြန်မာ: gray-zone ဖြစ်ရပ်များကို နောက်မှ စစ်ဆေးနိုင်ရန် SQLite table ဖန်တီးသည်။"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gray_zone_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            known_person_id INTEGER,
            camera_location TEXT NOT NULL,
            similarity REAL NOT NULL,
            match_percent INTEGER NOT NULL,
            distance REAL NOT NULL,
            capture_path TEXT,
            timestamp TEXT NOT NULL,
            original_filename TEXT,
            storage_path TEXT,
            FOREIGN KEY (known_person_id) REFERENCES known_persons(id)
        )
        """
    )
    cols = {
        row[1] for row in conn.execute("PRAGMA table_info(gray_zone_audit)").fetchall()
    }
    if "original_filename" not in cols:
        conn.execute(
            "ALTER TABLE gray_zone_audit ADD COLUMN original_filename TEXT"
        )
    if "storage_path" not in cols:
        conn.execute("ALTER TABLE gray_zone_audit ADD COLUMN storage_path TEXT")
    conn.commit()


def log_gray_zone_event(
    conn: sqlite3.Connection,
    known_person_id: int,
    camera_location: str,
    decision: TierDecision,
    capture_path: Optional[str] = None,
    *,
    original_filename: Optional[str] = None,
    storage_path: Optional[str] = None,
) -> None:
    """မြန်မာ: Tier 2 gray-zone ကို database တွင် မှတ်တမ်းတင်သည် (အချက်ပေးမလုပ်ပါ)။"""
    init_gray_zone_audit_db(conn)
    storage = storage_path or capture_path
    display_name = (original_filename or camera_location or "Live Feed").strip()
    conn.execute(
        """
        INSERT INTO gray_zone_audit
            (known_person_id, camera_location, similarity, match_percent,
             distance, capture_path, timestamp, original_filename, storage_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            known_person_id,
            camera_location,
            round(decision.similarity, 4),
            decision.match_percent,
            round(decision.distance, 4),
            capture_path,
            datetime.now().isoformat(timespec="seconds"),
            display_name,
            storage,
        ),
    )
    conn.commit()


def apply_identity_tiers(
    conn: sqlite3.Connection,
    matched_id: Optional[int],
    match_distance: float,
    camera_location: str,
    aligned_face: Optional[np.ndarray] = None,
    *,
    send_alert_fn: Optional[Callable[..., None]] = None,
    frame_bgr: Optional[np.ndarray] = None,
    bbox: Optional[Sequence[float]] = None,
    person_name: Optional[str] = None,
    camera_name: Optional[str] = None,
) -> TierDecision:
    """
    မြန်မာ: identity matching ရလဒ်ကို tier logic ဖြင့် ဆုံးဖြတ်ပြီး alert / audit လုပ်သည်။
    main.py နှင့် identity.py နှစ်ခုလုံးမှ ခေါ်သုံးနိုင်သည်။
    """
    decision = IdentityTierRouter.evaluate(matched_id, match_distance)

    display_face = aligned_face
    if aligned_face is not None and frame_bgr is not None and bbox is not None:
        try:
            from captures import choose_display_capture_crop

            display_face = choose_display_capture_crop(
                frame_bgr, bbox, aligned_face, None
            )
        except Exception:
            display_face = aligned_face

    is_enrolled_watchlist = False
    if matched_id is not None:
        try:
            from identity import is_enrolled_watchlist_person

            is_enrolled_watchlist = is_enrolled_watchlist_person(conn, int(matched_id))
        except Exception:
            is_enrolled_watchlist = False

    # Do not treat auto Stranger_* gallery hits as blacklist alarms in the log.
    if decision.tier != MatchTier.BLACKLIST_ALERT or is_enrolled_watchlist:
        IdentityTierRouter.log_decision(decision, camera_location)

    if decision.tier == MatchTier.BLACKLIST_ALERT and matched_id is not None:
        display_name = person_name
        if not display_name:
            try:
                from identity import resolve_known_person_name

                display_name = resolve_known_person_name(conn, matched_id)
            except Exception:
                display_name = None
        cam_label = camera_name or f"CAM-{camera_location}"
        if not is_enrolled_watchlist:
            try:
                name_l = str(display_name or "").strip().lower()
                is_enrolled_watchlist = bool(name_l) and not (
                    name_l.startswith("stranger_")
                    or name_l.startswith("passerby_")
                    or name_l.startswith("unknown")
                )
            except Exception:
                pass
        if send_alert_fn is not None and display_face is not None and is_enrolled_watchlist:
            try:
                send_alert_fn(
                    display_face,
                    matched_id,
                    camera_location,
                    decision.match_percent,
                    frame_bgr=frame_bgr,
                    person_name=display_name,
                    camera_name=cam_label,
                    location=str(camera_location),
                    bbox=bbox,
                )
            except TypeError:
                # Legacy alert callbacks that only accept positional args.
                send_alert_fn(
                    display_face,
                    matched_id,
                    camera_location,
                    decision.match_percent,
                )
        # Instant dashboard WebSocket alert (Person of Interest / watchlist).
        if is_enrolled_watchlist:
            try:
                from live_detection_hub import publish_person_alert

                publish_person_alert(
                    camera=str(camera_location),
                    name=str(display_name or f"Person #{matched_id}"),
                    confidence=float(decision.match_percent),
                    face_bgr=display_face,
                    person_id=int(matched_id),
                    known_person_id=int(matched_id),
                    bbox=bbox,
                    alert_type="blacklist",
                    force=True,
                )
            except Exception:
                pass

    if decision.tier == MatchTier.GRAY_ZONE and matched_id is not None:
        capture_path: Optional[str] = None
        if display_face is not None:
            try:
                from captures import save_face_capture

                capture_path = save_face_capture(
                    display_face,
                    matched_id,
                    abs(hash((camera_location, time.time()))) % 900_000 + 100_000,
                    person_name=person_name,
                )
            except Exception:
                capture_path = None
        log_gray_zone_event(
            conn,
            matched_id,
            camera_location,
            decision,
            capture_path=capture_path,
            original_filename=camera_location,
            storage_path=capture_path,
        )

    return decision


class OrionTacticalPipeline:
    """
    မြန်မာ: RTSP စတရီးမတစ်ခုအတွက် စုံလင်သော frame processing loop —
    validation, auto-reconnect, safe inference, tier logic, GPU face stack,
    and bounded RAM via periodic gc.collect() တို့ ပါဝင်သည်။
    """

    def __init__(
        self,
        rtsp_url: Union[int, str],
        camera_location: str,
        *,
        rtsp_tcp: bool = False,
        open_capture_fn: Optional[Callable[..., cv2.VideoCapture]] = None,
        ffmpeg_lock: Optional[threading.RLock] = None,
        frame_processor: Optional[Callable[[np.ndarray, int], None]] = None,
        device_choice: str = "auto",
        gc_interval: int = MEMORY_GC_INTERVAL_FRAMES,
    ) -> None:
        self.camera_location = camera_location
        self._device_choice = device_choice
        self._memory = FrameMemoryManager(gc_interval=gc_interval)
        self._reconnect = RtspAutoReconnect(
            rtsp_url,
            rtsp_tcp=rtsp_tcp,
            open_capture_fn=open_capture_fn,
            location=camera_location,
            ffmpeg_lock=ffmpeg_lock,
        )
        self._frame_processor = frame_processor
        self._running = False

    def process_frame_safe(
        self,
        frame: np.ndarray,
        frame_index: int,
        processor: Callable[[np.ndarray, int], T],
    ) -> Optional[T]:
        """မြန်မာ: frame တစ်ခုအတွက် inference ကို try-except ဖြင့် လုံခြုံစွာ လုပ်ဆောင်သည်။"""
        return SafeInferenceRunner.run(
            lambda: processor(frame, frame_index),
            context=f"{self.camera_location} frame {frame_index}",
        )

    def run_forever(self) -> None:
        """
        Infinite RTSP loop with NVIDIA GPU face stack and bounded host RAM.

        Identity matching still uses IdentityTierRouter (85% alert threshold) —
        only the inference execution provider changes (CPU OpenVINO/TF → CUDA).
        """
        audit_hardware_dependencies(device_choice=self._device_choice)
        init_gpu_face_stack()
        init_face_restoration_stack()
        self._running = True
        frame_index = 0
        print(f"[ORION] Pipeline started — {self.camera_location}")

        while self._running:
            ok, frame = self._reconnect.read_validated()
            if not ok or frame is None:
                continue

            frame_index += 1
            result: Any = None
            if self._frame_processor is not None:
                result = self.process_frame_safe(
                    frame, frame_index, self._frame_processor
                )

            # မြန်မာ: frame / crop / bbox references ကို လက်ရှိ iteration ပြီးချိန်တွင် လွှတ်ပေးသည်
            del result
            del frame
            self._memory.tick(frame_index)

        self._reconnect.release()

    def stop(self) -> None:
        self._running = False
        self._reconnect.release()


def wrap_yolo_results(results: Any) -> Optional[Any]:
    """
    မြန်မာ: YOLO ရလဒ် list ဗလာဖြစ်ပါက None ပြန်ပြီး results[0] crash ကို ကာကွယ်သည်။
    """
    if results is None:
        return None
    if isinstance(results, list):
        if len(results) == 0:
            return None
        return results[0]
    return results


# ── Super-resolution face pipeline (YOLO P2 → GFPGAN/CodeFormer → ArcFace) ──

PERSON_CLASS_ID = 0
PERSON_CROP_PAD = 0.08
MIN_PERSON_HEIGHT_PX = 60
MIN_RESTORED_FACE_PX = 28


@dataclass(frozen=True)
class PersonCrop:
    """Stage 1 output: YOLOv8s-P2 person bounding box + padded BGR crop."""

    bbox_xyxy: np.ndarray
    crop_bgr: np.ndarray


@dataclass(frozen=True)
class RestoredFaceMatch:
    """Stage 3 output: ArcFace embedding from a restored face patch."""

    embedding: np.ndarray
    restored_face_bgr: np.ndarray
    source_face_bbox: Tuple[int, int, int, int]
    face_area: float
    was_restored: bool


class YoloPersonCropStage:
    """
    Stage 1 — YOLOv8s-P2 person detection → accurate human bounding-box crops.

    P2 stride-4 heads preserve distant/small persons in ultra-wide command-center CCTV.
    """

    @staticmethod
    def crop_person_from_frame(
        frame_bgr: np.ndarray,
        bbox_xyxy: np.ndarray,
        *,
        pad: float = PERSON_CROP_PAD,
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float32).astype(int)
        h, w = frame_bgr.shape[:2]
        pad_px = int(pad * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad_px), max(0, y1 - pad_px)
        x2, y2 = min(w, x2 + pad_px), min(h, y2 + pad_px)
        crop = frame_bgr[y1:y2, x1:x2]
        return crop if crop.size else None

    @classmethod
    def crops_from_yolo(
        cls,
        frame_bgr: np.ndarray,
        yolo_result: Any,
        *,
        person_class_id: int = PERSON_CLASS_ID,
        min_conf: float = 0.25,
        min_height_px: float = MIN_PERSON_HEIGHT_PX,
        letterbox: Optional[Any] = None,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> List[PersonCrop]:
        """Extract Person-class crops from a single YOLO P2 result tensor."""
        wrapped = wrap_yolo_results(yolo_result)
        if wrapped is None or wrapped.boxes is None or len(wrapped.boxes) == 0:
            return []

        boxes = wrapped.boxes.xyxy
        confs = wrapped.boxes.conf
        classes = wrapped.boxes.cls

        if letterbox is not None and not getattr(letterbox, "is_identity", True):
            try:
                from gpu_pipeline import unmap_letterbox_xyxy_boxes

                boxes = unmap_letterbox_xyxy_boxes(boxes, letterbox)
            except Exception:
                pass
        elif scale_x != 1.0 or scale_y != 1.0:
            try:
                from gpu_pipeline import scale_xyxy_boxes

                boxes = scale_xyxy_boxes(boxes, scale_x, scale_y)
            except Exception:
                pass

        crops: List[PersonCrop] = []
        for i in range(len(boxes)):
            if classes is not None and int(classes[i].item()) != person_class_id:
                continue
            conf = float(confs[i].item()) if confs is not None else 1.0
            if conf < min_conf:
                continue
            bbox = boxes[i]
            if hasattr(bbox, "detach"):
                host_bbox = bbox.detach().cpu().numpy().astype(np.float32)
            else:
                host_bbox = np.asarray(bbox, dtype=np.float32)
            height = host_bbox[3] - host_bbox[1]
            if height < min_height_px:
                continue
            patch = cls.crop_person_from_frame(frame_bgr, host_bbox)
            if patch is None:
                continue
            crops.append(PersonCrop(bbox_xyxy=host_bbox, crop_bgr=patch))
        return crops


class FaceRestorationStage:
    """Stage 2 — GFPGAN / CodeFormer blind restoration on CUDA."""

    def __init__(self, engine: Optional[Any] = None) -> None:
        self._engine = engine

    @property
    def ready(self) -> bool:
        eng = self._ensure_engine()
        return eng is not None and bool(getattr(eng, "ready", False))

    def _ensure_engine(self) -> Optional[Any]:
        if self._engine is not None:
            return self._engine
        try:
            from face_restoration import get_face_restoration_engine

            self._engine = get_face_restoration_engine(initialize=True)
        except Exception as exc:
            _logger.debug("Face restoration engine unavailable: %s", exc)
            self._engine = None
        return self._engine

    def restore(self, face_bgr: np.ndarray, *, force: bool = False) -> Tuple[np.ndarray, bool]:
        eng = self._ensure_engine()
        if eng is None:
            return face_bgr, False
        return eng.restore_face(face_bgr, force=force)


class RestoredArcFaceStage:
    """Stage 3 — ArcFace embedding on restored high-definition face matrices."""

    @staticmethod
    def embed_restored(
        face_analyzer: Any,
        restored_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:
        if face_analyzer is None or restored_bgr is None or restored_bgr.size == 0:
            return None
        return face_analyzer.embed_crop(restored_bgr, apply_clahe=False)


_sr_pipeline: Optional["SuperResolutionFacePipeline"] = None
_sr_pipeline_lock = threading.Lock()


class SuperResolutionFacePipeline:
    """
    End-to-end command-center face pipeline:

      1. YOLOv8s-P2 person crop
      2. RetinaFace detect inside crop → GFPGAN/CodeFormer restore (CUDA)
      3. ArcFace embed (BLACKLIST_ALERT_SIMILARITY = 0.85 downstream unchanged)
    """

    def __init__(
        self,
        *,
        restoration: Optional[FaceRestorationStage] = None,
        person_crop_stage: Optional[YoloPersonCropStage] = None,
        arcface_stage: Optional[RestoredArcFaceStage] = None,
    ) -> None:
        self.person_crop = person_crop_stage or YoloPersonCropStage()
        self.restoration = restoration or FaceRestorationStage()
        self.arcface = arcface_stage or RestoredArcFaceStage()

    @property
    def ready(self) -> bool:
        return self.restoration.ready

    def process_person_bbox(
        self,
        frame_bgr: np.ndarray,
        person_bbox: np.ndarray,
        *,
        face_analyzer: Optional[Any] = None,
    ) -> Optional[RestoredFaceMatch]:
        """
        Intercept a YOLO person box: crop → detect face → restore → ArcFace embed.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        if face_analyzer is None:
            try:
                from face_analyzer import FaceAnalyzer, analyzer_ready

                if not analyzer_ready():
                    return None
                face_analyzer = FaceAnalyzer()
            except Exception:
                return None

        host_bbox = np.asarray(person_bbox, dtype=np.float32)
        person_crop = self.person_crop.crop_person_from_frame(frame_bgr, host_bbox)
        if person_crop is None:
            return None

        dets = face_analyzer.detect_and_embed_scan(person_crop)
        if not dets:
            return None

        best = max(
            dets,
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
        )
        x1, y1, x2, y2 = best.bbox
        face_patch = person_crop[y1:y2, x1:x2]
        if face_patch is None or face_patch.size == 0:
            return None

        restored_patch, was_restored = self.restoration.restore(face_patch, force=True)
        embedding = self.arcface.embed_restored(face_analyzer, restored_patch)
        if embedding is None:
            return None

        try:
            from captures import crop_face_from_frame, map_bbox_from_person_crop_to_source

            abs_bbox = map_bbox_from_person_crop_to_source(
                host_bbox,
                best.bbox,
                frame_bgr.shape,
                crop_shape=person_crop.shape[:2],
                upscale_min_side=0,
            )
            hi_res = crop_face_from_frame(frame_bgr, abs_bbox)
            if hi_res is None or hi_res.size == 0:
                hi_res = restored_patch
        except Exception:
            abs_bbox = (int(x1), int(y1), int(x2), int(y2))
            hi_res = restored_patch

        fw, fh = abs_bbox[2] - abs_bbox[0], abs_bbox[3] - abs_bbox[1]
        if max(fw, fh) < MIN_RESTORED_FACE_PX and not was_restored:
            return None

        return RestoredFaceMatch(
            embedding=embedding,
            restored_face_bgr=hi_res if hi_res is not None else restored_patch,
            source_face_bbox=(
                int(abs_bbox[0]),
                int(abs_bbox[1]),
                int(abs_bbox[2]),
                int(abs_bbox[3]),
            ),
            face_area=float(max(fw * fh, restored_patch.shape[0] * restored_patch.shape[1])),
            was_restored=was_restored,
        )

    def restore_and_embed_face_patch(
        self,
        face_bgr: np.ndarray,
        *,
        face_analyzer: Optional[Any] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, bool]]:
        """Restore an existing face crop and return (embedding, restored_bgr, was_restored)."""
        if face_analyzer is None:
            return None
        restored, was_restored = self.restoration.restore(face_bgr, force=True)
        emb = self.arcface.embed_restored(face_analyzer, restored)
        if emb is None:
            return None
        return emb, restored, was_restored


def get_super_resolution_pipeline(
    *,
    initialize: bool = True,
) -> Optional[SuperResolutionFacePipeline]:
    """Shared singleton for live CCTV + unified tracker subprocesses."""
    global _sr_pipeline
    with _sr_pipeline_lock:
        if _sr_pipeline is not None:
            return _sr_pipeline
        if not initialize:
            return None
        _sr_pipeline = SuperResolutionFacePipeline()
        return _sr_pipeline


# ── Pure-GPU dashboard video target scan ─────────────────────────────────────

VIDEO_SCAN_FRAME_WIDTH = max(
    320,
    int(os.environ.get("VIDEO_SCAN_MAX_WIDTH", "640")),
)  # optional max width after detect downscale
VIDEO_SCAN_YOLO_IMGSZ = 640
VIDEO_SCAN_FRAME_STEP = max(
    1,
    int(os.environ.get("VIDEO_SCAN_FRAME_INTERVAL", "12")),
)
VIDEO_SCAN_DETECT_SCALE = float(
    os.environ.get("VIDEO_SCAN_DETECT_SCALE", "0.50")
)
VIDEO_SCAN_ADAPTIVE_ENABLED = os.environ.get(
    "VIDEO_SCAN_ADAPTIVE", "1"
).strip().lower() in ("1", "true", "yes")
# When faces appear → denser sampling; empty scenes → sparser (faster).
VIDEO_SCAN_ADAPTIVE_DENSE_STEP = max(
    1, int(os.environ.get("VIDEO_SCAN_ADAPTIVE_DENSE_STEP", "3"))
)
VIDEO_SCAN_ADAPTIVE_SPARSE_MULT = max(
    2, int(os.environ.get("VIDEO_SCAN_ADAPTIVE_SPARSE_MULT", "2"))
)
VIDEO_SCAN_ADAPTIVE_NO_FACE_STREAK = max(
    2, int(os.environ.get("VIDEO_SCAN_ADAPTIVE_NO_FACE_STREAK", "6"))
)
VIDEO_SCAN_ADAPTIVE_DENSE_WINDOW = max(
    4, int(os.environ.get("VIDEO_SCAN_ADAPTIVE_DENSE_WINDOW", "20"))
)
VIDEO_SCAN_YOLO_PERSON_GATE = os.environ.get(
    "VIDEO_SCAN_YOLO_GATE", "1"
).strip().lower() in ("1", "true", "yes")
VIDEO_SCAN_GPU_BATCH_SIZE = max(
    1,
    int(os.environ.get("VIDEO_SCAN_BATCH_SIZE", "20")),
)
VIDEO_SCAN_GC_INTERVAL_FRAMES = 96
VIDEO_SCAN_EARLY_EXIT_MATCH_PERCENT = 55  # stop early on strong target hit
# TEMP — set True to export raw probe JPEGs and skip detection
VIDEO_SCAN_DEBUG_SAVE_RAW_ONLY = False
VIDEO_SCAN_DEBUG_RAW_FRAME_INDICES = frozenset({100, 500, 1000, 1500, 2000})
VIDEO_SCAN_DEBUG_DIR = Path(__file__).resolve().parent / "debug_frames"
# SceneChangeGate is retained for future experiments but disabled for video scan —
# pixel-diff skips most CCTV stride samples incorrectly (overhead / compression).
VIDEO_SCAN_SCENE_THUMB_WIDTH = 160
VIDEO_SCAN_SCENE_THUMB_HEIGHT = 120
VIDEO_SCAN_SCENE_DIFF_THRESHOLD = 8.0
VIDEO_SCAN_SCENE_MAX_STATIC_SKIPS = 12


class SceneChangeGate:
    """
    DISABLED for dashboard video scan — kept for future tuning only.

    Pixel-diff on a grayscale thumbnail falsely treats most CCTV stride samples as
    static (compression, overhead angle, slow movement). Video scan uses stride-only
    sampling instead.
    """

    def __init__(
        self,
        *,
        thumb_width: int = VIDEO_SCAN_SCENE_THUMB_WIDTH,
        thumb_height: int = VIDEO_SCAN_SCENE_THUMB_HEIGHT,
        diff_threshold: float = VIDEO_SCAN_SCENE_DIFF_THRESHOLD,
        max_static_skips: int = VIDEO_SCAN_SCENE_MAX_STATIC_SKIPS,
    ) -> None:
        self.thumb_width = max(32, int(thumb_width))
        self.thumb_height = max(24, int(thumb_height))
        self.diff_threshold = float(diff_threshold)
        self.max_static_skips = max(1, int(max_static_skips))
        self._ref_gray: Optional[np.ndarray] = None
        self._static_skips = 0
        self.skipped = 0

    def _gray_thumb(self, frame_bgr: np.ndarray) -> np.ndarray:
        small = cv2.resize(
            frame_bgr,
            (self.thumb_width, self.thumb_height),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def scene_changed(self, frame_bgr: np.ndarray) -> bool:
        """True when this frame should be face-scanned (scene moved or forced)."""
        gray = self._gray_thumb(frame_bgr)
        if self._ref_gray is None:
            self._ref_gray = gray
            self._static_skips = 0
            return True

        diff = float(
            np.mean(np.abs(gray.astype(np.int16) - self._ref_gray.astype(np.int16)))
        )
        if diff >= self.diff_threshold:
            self._ref_gray = gray
            self._static_skips = 0
            return True

        self._static_skips += 1
        if self._static_skips >= self.max_static_skips:
            self._ref_gray = gray
            self._static_skips = 0
            return True

        self.skipped += 1
        return False


@dataclass
class VideoScanProgress:
    scan_id: str
    video_file: str
    frame_current: int
    frame_total: int
    frames_inspected: int
    faces_compared: int
    best_match_percent: int
    eta_seconds: float
    phase: str = "scanning"
    video_index: int = 0
    videos_total: int = 1
    early_exit: bool = False
    frames_skipped_static: int = 0
    original_filename: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "video_file": self.video_file,
            "original_filename": self.original_filename or self.video_file,
            "frame_current": self.frame_current,
            "frame_total": self.frame_total,
            "frames_inspected": self.frames_inspected,
            "faces_compared": self.faces_compared,
            "best_match_percent": self.best_match_percent,
            "eta_seconds": round(self.eta_seconds, 1),
            "phase": self.phase,
            "video_index": self.video_index,
            "videos_total": self.videos_total,
            "early_exit": self.early_exit,
            "frames_skipped_static": self.frames_skipped_static,
            "progress_percent": (
                round(100.0 * self.frame_current / self.frame_total, 1)
                if self.frame_total > 0
                else 0.0
            ),
        }


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def require_cuda_video_scan(*, device_choice: str = "cuda") -> None:
    """
    Hard gate for dashboard video target scan — raises RuntimeError on CPU fallback.

    YOLOv8 Large + InsightFace RetinaFace/ArcFace must run on CUDA; no OpenVINO/CPU path.
    """
    from inference_device import configure, describe_device, resolve_torch_device, torch_cuda_active

    configure(device_choice, pure_gpu=True)
    print(f"Resolved inference: {describe_device()}")
    if not resolve_torch_device(device_choice).startswith("cuda"):
        raise RuntimeError(
            "CRITICAL: Video target scan requires NVIDIA CUDA.\n"
            "  Run: py -3.12 run_system.py --device cuda"
        )
    if not torch_cuda_active():
        raise RuntimeError(
            "CRITICAL: torch.cuda.is_available() is False — "
            "video scan refuses CPU execution (18k-frame RAM collapse)."
        )
    init_gpu_face_stack()
    from identity import _ensure_video_scan_gpu, gpu_scan_ready

    if not gpu_scan_ready() or not _ensure_video_scan_gpu():
        raise RuntimeError(
            "CRITICAL: InsightFace RetinaFace+ArcFace not on CUDAExecutionProvider.\n"
            "  pip install onnxruntime-gpu && pip install insightface --no-deps\n"
            "  Then: py -3.12 run_system.py --device cuda"
        )
    print("GPU scan ready: True (CUDAExecutionProvider, no CPU/OpenVINO fallback)")


def init_pure_gpu_video_scan(
    *, device_choice: str = "cuda", strict: bool = False
) -> bool:
    """
    Force CUDA-only inference for dashboard video scanning.

    Bypasses CPU OpenVINO/TensorFlow paths — RetinaFace + ArcFace via InsightFace
    on CUDAExecutionProvider. IdentityTierRouter thresholds (85% alert) unchanged.
    """
    try:
        require_cuda_video_scan(device_choice=device_choice)
        return True
    except RuntimeError:
        if strict:
            raise
        print("GPU scan ready: False (CUDA / InsightFace stack unavailable)")
        return False


def resize_frame_for_video_scan(
    frame: np.ndarray,
    *,
    width: int = VIDEO_SCAN_FRAME_WIDTH,
    scale_factor: Optional[float] = None,
) -> Tuple[np.ndarray, "UniformScaleTransform"]:
    """
    Downscale for RetinaFace detection (default 50% of source resolution).

    ArcFace runs later on full-resolution crops mapped via UniformScaleTransform.
    """
    from image_geometry import (
        UniformScaleTransform,
        downscale_frame_by_factor,
        downscale_frame_to_width,
    )
    from identity import VIDEO_SCAN_DETECT_SCALE

    detect_scale = float(
        scale_factor if scale_factor is not None else VIDEO_SCAN_DETECT_SCALE
    )
    if detect_scale < 0.999:
        work, t_primary = downscale_frame_by_factor(frame, detect_scale)
    else:
        work, t_primary = frame, UniformScaleTransform(1.0)

    cap_width = max(0, int(width or 0))
    if cap_width and work.shape[1] > cap_width:
        work, t_cap = downscale_frame_to_width(work, max_width=cap_width)
        combined = UniformScaleTransform(scale=t_primary.scale * t_cap.scale)
        return work, combined
    return work, t_primary


class VideoScanExecutor:
    """
    Production GPU video target scanner — stride sampling, RetinaFace + ArcFace on CUDA.

    Designed for 10-minute wide-angle CCTV clips from the dashboard upload API.
    """

    @staticmethod
    def _debug_save_raw_frames(
        cap: cv2.VideoCapture,
        *,
        video_name: str,
        fps: float,
        total_frames: int,
    ) -> dict:
        """TEMP: export raw BGR frames at fixed indices; no detection."""
        from identity import _format_video_timestamp

        VIDEO_SCAN_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        wanted = VIDEO_SCAN_DEBUG_RAW_FRAME_INDICES
        max_wanted = max(wanted) if wanted else 0
        saved: List[Path] = []
        frame_idx = 0

        while frame_idx <= max_wanted:
            ok, frame = cap.read()
            if not FrameValidator.is_valid(ok, frame):
                break
            if frame_idx in wanted:
                out = VIDEO_SCAN_DEBUG_DIR / f"frame_{frame_idx}_raw.jpg"
                cv2.imwrite(str(out), frame)
                print(f"Saved debug frame: {out}")
                saved.append(out)
            frame_idx += 1

        duration_ts = _format_video_timestamp(max(total_frames - 1, 0), fps)
        names = ", ".join(str(p.name) for p in saved) or "none"
        return {
            "success": True,
            "matches": [],
            "borderline_suspects": [],
            "near_miss_suspects": [],
            "frames_scanned": 0,
            "video_frames_total": total_frames,
            "video_duration": duration_ts,
            "fps": round(fps, 2),
            "video_file": video_name,
            "faces_compared": 0,
            "faces_detected_count": 0,
            "best_match_percent": 0,
            "gpu_scan": False,
            "early_exit": True,
            "frames_skipped_static": 0,
            "message": (
                f"Debug export complete ({len(saved)} frame(s): {names}). "
                "Detection skipped."
            ),
        }

    @classmethod
    def scan_for_target(
        cls,
        target_image_path: Path,
        video_path: Path,
        yolo_model=None,
        frame_step: int = VIDEO_SCAN_FRAME_STEP,
        match_threshold: float = 0.50,
        target_feat: Optional[np.ndarray] = None,
        progress_callback: ProgressCallback = None,
        match_score_callback: Optional[Callable[[float], None]] = None,
        scan_id: Optional[str] = None,
        video_index: int = 0,
        videos_total: int = 1,
        require_cuda: bool = True,
        original_filename: Optional[str] = None,
        face_demographics_callback=None,
        target_hit_callback=None,
    ) -> dict:
        import os
        import time as _time

        from identity import (
            VIDEO_TARGET_MATCH_THRESHOLD,
            VIDEO_TARGET_NEAR_MISS_MIN_PERCENT,
            _build_scan_hit,
            _face_boxes_for_scan_frame,
            _scan_arcface_embed,
            _format_video_timestamp,
            _load_target_feature,
            _scan_embedding_distance,
            build_all_scan_hits_sorted,
            tier_scan_hits_by_percent,
            VIDEO_SCAN_ALERT_MATCH_PERCENT,
            VIDEO_TARGET_CONFIRM_MATCH_PERCENT,
            _dedupe_borderline_hits,
            finalize_scan_embed_diagnostics,
            _record_embed_diag_candidate,
            VIDEO_SCAN_RETINAFACE_DET_THRESH,
            VIDEO_SCAN_DETECT_UPSCALE,
            confidence_score_from_distance,
            scan_min_match_percent,
            resolve_original_video_filename,
        )

        storage_name = Path(video_path).name
        display_name = resolve_original_video_filename(storage_name, original_filename)
        empty = cls._empty_result(storage_name, "", original_filename=display_name)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            empty["message"] = "Could not open video file."
            return empty

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if VIDEO_SCAN_DEBUG_SAVE_RAW_ONLY:
            try:
                return cls._debug_save_raw_frames(
                    cap,
                    video_name=storage_name,
                    fps=fps,
                    total_frames=total_frames,
                )
            finally:
                cap.release()

        device = os.environ.get("DEVICE", "cuda")
        if require_cuda:
            require_cuda_video_scan(device_choice=device)

        if yolo_model is None:
            cap.release()
            empty["message"] = "Search model unavailable."
            return empty

        use_gpu_scan = True

        if target_feat is None:
            target_feat, err = _load_target_feature(
                target_image_path, yolo_model=yolo_model
            )
        if target_feat is None:
            cap.release()
            empty["message"] = err
            return empty

        frame_step = max(1, int(frame_step or VIDEO_SCAN_FRAME_STEP))
        base_frame_step = frame_step
        current_frame_step = base_frame_step
        consecutive_no_face = 0
        dense_inspects_remaining = 0
        sparse_mode = False
        sparse_frame_step = max(
            base_frame_step + 1,
            base_frame_step * VIDEO_SCAN_ADAPTIVE_SPARSE_MULT,
        )
        batch_size = max(1, int(VIDEO_SCAN_GPU_BATCH_SIZE))
        gate_pct = scan_min_match_percent(
            match_threshold or VIDEO_TARGET_MATCH_THRESHOLD
        )
        scan_token = scan_id or uuid.uuid4().hex[:12]
        scan_started = _time.monotonic()
        early_exit = False
        expected_inspected = (
            (total_frames + base_frame_step - 1) // base_frame_step
            if total_frames > 0
            else 0
        )
        print(
            f"Scan config: total_frames={total_frames}, stride={base_frame_step}, "
            f"expected_inspected={expected_inspected}, detection=RetinaFace-only "
            f"(det_thresh={VIDEO_SCAN_RETINAFACE_DET_THRESH}, "
            f"detect_scale={VIDEO_SCAN_DETECT_SCALE}, "
            f"detect_upscale={VIDEO_SCAN_DETECT_UPSCALE}x, metric=cosine)"
        )

        def _advance_to_next_inspect_frame() -> Tuple[bool, Optional[np.ndarray]]:
            """Skip decode-only frames in bulk, then read the next inspect frame."""
            nonlocal frame_count, eof, current_frame_step
            if dense_inspects_remaining > 0 and VIDEO_SCAN_ADAPTIVE_ENABLED:
                current_frame_step = VIDEO_SCAN_ADAPTIVE_DENSE_STEP
            elif sparse_mode and VIDEO_SCAN_ADAPTIVE_ENABLED:
                current_frame_step = sparse_frame_step
            else:
                current_frame_step = base_frame_step

            remainder = frame_count % current_frame_step
            if remainder != 0:
                skip_grabs = current_frame_step - remainder
                for _ in range(skip_grabs):
                    if not cap.grab():
                        eof = True
                        return False, None
                    frame_count += 1

            ok, frame = cap.read()
            if not FrameValidator.is_valid(ok, frame):
                eof = True
                return False, None
            frame_count += 1
            return True, frame

        def _report_progress() -> None:
            if progress_callback is None:
                return
            elapsed = max(_time.monotonic() - scan_started, 0.001)
            eta = 0.0
            if frame_count > 0 and total_frames > frame_count:
                rate = frame_count / elapsed
                eta = (total_frames - frame_count) / max(rate, 1e-6)
            progress_callback(
                VideoScanProgress(
                    scan_id=scan_token,
                    video_file=storage_name,
                    original_filename=display_name,
                    frame_current=frame_count,
                    frame_total=total_frames,
                    frames_inspected=scanned,
                    faces_compared=faces_compared,
                    best_match_percent=best_match_pct,
                    eta_seconds=eta,
                    phase="early_exit" if early_exit else "scanning",
                    video_index=video_index,
                    videos_total=videos_total,
                    early_exit=early_exit,
                    frames_skipped_static=0,
                ).to_dict()
            )

        raw_hits: List[Dict] = []
        near_miss_hits: List[Dict] = []
        frame_count = 0
        scanned = 0
        faces_compared = 0
        faces_detected_count = 0
        best_match_pct = 0
        top_face_matches: List[Dict[str, object]] = []
        memory_mgr = FrameMemoryManager(gc_interval=VIDEO_SCAN_GC_INTERVAL_FRAMES)

        def _infer_faces_on_frame(
            source_frame: np.ndarray,
            work: np.ndarray,
            transform: Any,
            *,
            scan_frame_idx: int,
        ) -> Tuple[bool, int]:
            """Returns (early_exit_hit, face_boxes_found_on_frame)."""
            nonlocal faces_compared, faces_detected_count, best_match_pct, early_exit
            nonlocal top_face_matches

            # Cheap YOLO person gate — skip RetinaFace on empty frames (big speed win).
            if (
                VIDEO_SCAN_YOLO_PERSON_GATE
                and yolo_model is not None
                and work is not None
                and getattr(work, "size", 0)
            ):
                try:
                    from identity import _yolo_predict_persons

                    yolo_out = _yolo_predict_persons(yolo_model, work, for_scan=True)
                    first = wrap_yolo_results(yolo_out)
                    n_persons = (
                        len(first.boxes)
                        if first is not None
                        and getattr(first, "boxes", None) is not None
                        else 0
                    )
                    if n_persons <= 0:
                        return False, 0
                except Exception:
                    pass

            face_hits = _face_boxes_for_scan_frame(
                work,
                scan_frame_idx=scan_frame_idx,
                source_frame=source_frame,
            )
            n_found = len(face_hits) if face_hits else 0
            if n_found:
                faces_detected_count += n_found
            if not face_hits:
                return False, n_found

            unmap_lm = getattr(transform, "unmap_landmarks", None)

            for face_box, landmarks in face_hits:
                full_box = transform.unmap_bbox(face_box)
                full_lm = unmap_lm(landmarks) if landmarks and unmap_lm else landmarks
                packed = _scan_arcface_embed(
                    source_frame,
                    full_box,
                    landmarks=full_lm,
                )
                if packed is None:
                    continue
                compare_feat, embed_meta = packed
                dist = _scan_embedding_distance(
                    target_feat, compare_feat, gpu=use_gpu_scan
                )
                if not np.isfinite(dist):
                    del compare_feat
                    continue
                match_pct = confidence_score_from_distance(dist)
                match_percentage = int(match_pct)
                faces_compared += 1
                if match_score_callback is not None:
                    try:
                        match_score_callback(float(match_pct))
                    except Exception:
                        pass
                if match_pct > best_match_pct:
                    best_match_pct = int(match_pct)

                # Drop weak faces; keep borderline+ for review.
                if match_percentage < VIDEO_TARGET_CONFIRM_MATCH_PERCENT:
                    del compare_feat
                    continue

                crop_w = int(embed_meta.get("crop_w", 0))
                crop_h = int(embed_meta.get("crop_h", 0))
                enhanced = bool(embed_meta.get("enhanced", False))
                top_face_matches.append(
                    {
                        "frame": scan_frame_idx,
                        "similarity": match_pct,
                        "crop_w": crop_w,
                        "crop_h": crop_h,
                        "enhanced": enhanced,
                    }
                )
                _record_embed_diag_candidate(
                    {
                        "frame": scan_frame_idx,
                        "similarity": match_pct,
                        "embedding": compare_feat.copy(),
                        "debug": embed_meta.get("debug"),
                    }
                )
                hit = _build_scan_hit(
                    source_frame,
                    scan_frame_idx,
                    fps,
                    storage_name,
                    full_box,
                    dist,
                    match_pct,
                    original_filename=display_name,
                )
                if face_demographics_callback is not None:
                    try:
                        x1, y1, x2, y2 = (int(v) for v in full_box)
                        h, w = source_frame.shape[:2]
                        pad_x = max(1, int((x2 - x1) * 0.12))
                        pad_y = max(1, int((y2 - y1) * 0.12))
                        cx1 = max(0, x1 - pad_x)
                        cy1 = max(0, y1 - pad_y)
                        cx2 = min(w, x2 + pad_x)
                        cy2 = min(h, y2 + pad_y)
                        face_crop = source_frame[cy1:cy2, cx1:cx2]
                        if face_crop is not None and face_crop.size > 0:
                            demo = face_demographics_callback(face_crop)
                            if isinstance(demo, dict):
                                hit.update(demo)
                            hit["face_box"] = full_box
                    except Exception as exc:
                        get_async_logger().warn(
                            f"Demographics callback skipped frame {scan_frame_idx}: {exc}"
                        )
                raw_hits.append(hit)
                if target_hit_callback is not None:
                    try:
                        target_hit_callback(dict(hit))
                    except Exception:
                        pass
                del compare_feat, hit
                if match_pct >= VIDEO_SCAN_EARLY_EXIT_MATCH_PERCENT:
                    early_exit = True
                    return True, n_found
            del face_hits
            return early_exit, n_found

        def _note_frame_face_stats(n_faces: int) -> None:
            """Dense when people appear; sparse when the scene stays empty."""
            nonlocal consecutive_no_face, dense_inspects_remaining, current_frame_step
            nonlocal sparse_mode
            if not VIDEO_SCAN_ADAPTIVE_ENABLED:
                return
            if n_faces > 0:
                consecutive_no_face = 0
                sparse_mode = False
                dense_inspects_remaining = VIDEO_SCAN_ADAPTIVE_DENSE_WINDOW
                current_frame_step = VIDEO_SCAN_ADAPTIVE_DENSE_STEP
                return
            if dense_inspects_remaining > 0:
                dense_inspects_remaining -= 1
                if dense_inspects_remaining <= 0:
                    current_frame_step = base_frame_step
                return
            consecutive_no_face += 1
            if consecutive_no_face >= VIDEO_SCAN_ADAPTIVE_NO_FACE_STREAK:
                sparse_mode = True
                current_frame_step = sparse_frame_step

        def _process_scan_batch(batch_items: List[Tuple]) -> bool:
            nonlocal scanned, early_exit, current_frame_step
            if not batch_items:
                return early_exit

            processed = 0
            for idx, source_frame, work, transform in batch_items:
                if early_exit:
                    break
                processed += 1
                frame = source_frame
                work_frame = work
                try:

                    def _scan_one(
                        _source=frame,
                        _work=work_frame,
                        _transform=transform,
                        _idx=idx,
                    ) -> Tuple[bool, int]:
                        return _infer_faces_on_frame(
                            _source,
                            _work,
                            _transform,
                            scan_frame_idx=_idx,
                        )

                    scan_result = SafeInferenceRunner.run(
                        _scan_one,
                        context=f"video scan {display_name} frame {idx}",
                    )
                    if scan_result is None:
                        hit_early, n_faces = False, 0
                    else:
                        hit_early, n_faces = scan_result
                    _note_frame_face_stats(n_faces)
                    if hit_early:
                        early_exit = True
                        break
                finally:
                    release_frame_buffers(work_frame, frame)
                    del frame, work_frame, transform

            scanned += processed
            if frame_count % VIDEO_SCAN_GC_INTERVAL_FRAMES == 0:
                gc.collect()
            return early_exit

        _report_progress()
        try:
            eof = False
            while not eof and not early_exit:
                batch_items: List[Tuple[int, np.ndarray, np.ndarray, Any]] = []
                while len(batch_items) < batch_size and not eof:
                    got_frame, frame = _advance_to_next_inspect_frame()
                    if not got_frame or frame is None:
                        break

                    work, transform = resize_frame_for_video_scan(frame)
                    batch_items.append((frame_count - 1, frame, work, transform))

                if not batch_items:
                    break

                if _process_scan_batch(batch_items):
                    break
                del batch_items
                memory_mgr.tick(frame_count)
                _report_progress()
        except (IndexError, TypeError) as exc:
            get_async_logger().warn(
                f"Video scan inference skipped ({type(exc).__name__}: {exc})"
            )
        finally:
            cap.release()
            if progress_callback is not None:
                progress_callback(
                    VideoScanProgress(
                        scan_id=scan_token,
                        video_file=storage_name,
                        original_filename=display_name,
                        frame_current=frame_count,
                        frame_total=total_frames,
                        frames_inspected=scanned,
                        faces_compared=faces_compared,
                        best_match_percent=best_match_pct,
                        eta_seconds=0.0,
                        phase="done",
                        video_index=video_index,
                        videos_total=videos_total,
                        early_exit=early_exit,
                        frames_skipped_static=0,
                    ).to_dict()
                )
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        finalize_scan_embed_diagnostics(target_feat, top_face_matches)

        for entry in sorted(
            top_face_matches, key=lambda e: int(e["similarity"]), reverse=True
        )[:10]:
            print(
                f"Frame {entry['frame']}: similarity={entry['similarity']}%, "
                f"crop_size={entry['crop_w']}x{entry['crop_h']}, "
                f"enhanced={entry['enhanced']}"
            )

        print(
            f"[SCAN] {display_name}: frames_scanned={scanned} "
            f"(expected~{expected_inspected}) "
            f"faces_detected_count={faces_detected_count} "
            f"faces_compared={faces_compared} "
            f"best_similarity_score={best_match_pct}"
            f"{' early_exit=50%' if early_exit else ''}"
        )

        final_frames = total_frames if total_frames > 0 else frame_count
        duration_ts = _format_video_timestamp(max(final_frames - 1, 0), fps)

        all_hits = build_all_scan_hits_sorted(raw_hits)
        merged, merged_borderline = tier_scan_hits_by_percent(all_hits)
        merged_borderline = _dedupe_borderline_hits(merged_borderline)
        near_miss: List[Dict] = []

        if not all_hits:
            if faces_compared == 0:
                no_hit_msg = (
                    "No faces were detected in the scanned frames. "
                    "Try a closer camera angle or a higher-resolution clip."
                )
            elif near_miss:
                no_hit_msg = (
                    f"No confirmed matches (≥{gate_pct}%). "
                    f"Best similarity was {best_match_pct}% — "
                    f"{len(near_miss)} low-confidence candidate(s) listed for review."
                )
            elif best_match_pct > 0:
                no_hit_msg = (
                    f"No matching appearances found. Best similarity was {best_match_pct}% "
                    f"(need ≥{VIDEO_TARGET_CONFIRM_MATCH_PERCENT}% to appear in results)."
                )
            else:
                no_hit_msg = "No matching appearances found in this video."
            empty.update(
                {
                    "success": True,
                    "near_miss_suspects": near_miss,
                    "frames_scanned": scanned,
                    "video_frames_total": final_frames,
                    "video_duration": duration_ts,
                    "fps": round(fps, 2),
                    "faces_compared": faces_compared,
                    "faces_detected_count": faces_detected_count,
                    "best_match_percent": best_match_pct,
                    "message": no_hit_msg,
                    "gpu_scan": use_gpu_scan,
                    "early_exit": early_exit,
                    "frames_skipped_static": 0,
                }
            )
            return empty

        msg_parts = []
        if merged:
            msg_parts.append(f"{len(merged)} confirmed")
        if merged_borderline:
            msg_parts.append(f"{len(merged_borderline)} borderline")
        summary = " and ".join(msg_parts) if msg_parts else "0"
        done_msg = f"Found {summary} appearance(s) in {display_name}."
        if early_exit:
            done_msg += (
                f" Stopped early at {best_match_pct}% match "
                f"(≥{VIDEO_SCAN_ALERT_MATCH_PERCENT}%)."
            )

        return {
            "success": True,
            "matches": all_hits,
            "all_matches": all_hits,
            "confirmed_matches": merged,
            "borderline_suspects": merged_borderline,
            "near_miss_suspects": near_miss,
            "frames_scanned": scanned,
            "video_frames_total": final_frames,
            "video_duration": duration_ts,
            "fps": round(fps, 2),
            "video_file": storage_name,
            "original_filename": display_name,
            "faces_compared": faces_compared,
            "faces_detected_count": faces_detected_count,
            "best_match_percent": best_match_pct,
            "gpu_scan": use_gpu_scan,
            "early_exit": early_exit,
            "frames_skipped_static": 0,
            "message": done_msg,
        }

    @staticmethod
    def _empty_result(
        video_name: str,
        message: str,
        *,
        original_filename: Optional[str] = None,
    ) -> dict:
        from identity import resolve_original_video_filename

        storage_name = str(video_name or "").strip()
        display_name = resolve_original_video_filename(storage_name, original_filename)
        return {
            "success": False,
            "matches": [],
            "borderline_suspects": [],
            "frames_scanned": 0,
            "video_frames_total": 0,
            "video_duration": "",
            "fps": 0.0,
            "message": message,
            "video_file": storage_name,
            "original_filename": display_name,
        }
