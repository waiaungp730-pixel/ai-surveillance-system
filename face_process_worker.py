"""
Background face / detection worker — keeps the live preview loop responsive.

YOLO batch runs on the main tracker thread; per-camera RetinaFace + capture
commit runs here so dashboard MJPEG is not blocked by heavy GPU face work.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from tracker_config import is_outdoor_gate_camera


@dataclass
class CameraAIJob:
    frame_index: int
    location: str
    frame_bgr: np.ndarray
    now: float
    tracked: bool = False
    result: Any = None
    src_frame: Any = None
    letterbox: Any = None


class MultiCamFaceWorker:
    """Serial GPU face pipeline off the hot preview path."""

    def __init__(
        self,
        process_fn: Callable[[CameraAIJob], int],
        *,
        max_queue: int = 24,
    ) -> None:
        self._process_fn = process_fn
        self._queue: queue.Queue[Optional[CameraAIJob]] = queue.Queue(
            maxsize=max(4, int(max_queue))
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="face-process-worker",
            daemon=True,
        )
        self.log_count = 0
        self._log_lock = threading.Lock()

    def start(self) -> None:
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    def pending(self) -> int:
        return self._queue.qsize()

    def submit(self, job: CameraAIJob) -> bool:
        """Enqueue one camera tick; drop stale non-gate work when the queue is full."""
        if self._stop.is_set():
            return False
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            pending: List[Optional[CameraAIJob]] = []
            while True:
                try:
                    pending.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            gate_jobs = [j for j in pending if j is not None and is_outdoor_gate_camera(j.location)]
            other_jobs = [j for j in pending if j is not None and not is_outdoor_gate_camera(j.location)]
            drop_n = min(6, len(other_jobs))
            for stale in other_jobs[:drop_n]:
                if stale.frame_bgr is not None:
                    del stale.frame_bgr
            kept = gate_jobs + other_jobs[drop_n:] + [j for j in pending if j is None]
            for item in kept:
                try:
                    self._queue.put_nowait(item)
                except queue.Full:
                    if item is not None and item.frame_bgr is not None:
                        del item.frame_bgr
            try:
                self._queue.put_nowait(job)
                return True
            except queue.Full:
                if job.frame_bgr is not None:
                    del job.frame_bgr
                return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                n = int(self._process_fn(job))
                if n:
                    with self._log_lock:
                        self.log_count += n
            except Exception as exc:
                print(f"[WARN] face worker @ {job.location}: {exc}")
            finally:
                if job.frame_bgr is not None:
                    del job.frame_bgr
