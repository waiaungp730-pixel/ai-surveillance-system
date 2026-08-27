"""
Centralized GPU inference worker — single YOLO instance on cuda:0.

Camera ingest pushes FrameJob objects into a queue; the worker batch-processes
frames inside VRAM (one YOLO batch call per tick) and emits InferenceResult packets.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from image_geometry import LetterboxTransform
import tracker_config
from tracker_config import TRACK_CONF

log = logging.getLogger("gpu_inference_worker")

BATCH_COLLECT_TIMEOUT_SEC = 0.008
MAX_BATCH_SIZE = 32


def _inference_wh() -> Tuple[int, int]:
    return tracker_config.INFERENCE_WIDTH, tracker_config.INFERENCE_HEIGHT


def _track_kwargs() -> dict:
    return dict(tracker_config.TRACK_KWARGS)


@dataclass
class FrameJob:
    camera_location: str
    slot: int
    frame_index: int
    frame: np.ndarray
    run_inference: bool = True


@dataclass
class TrackPacket:
    bbox: Tuple[float, float, float, float]
    track_id: Optional[int]
    conf: float


@dataclass
class InferenceResult:
    camera_location: str
    slot: int
    frame_index: int
    tracks: List[TrackPacket] = field(default_factory=list)
    letterbox: LetterboxTransform = field(
        default_factory=lambda: LetterboxTransform(1.0, 0.0, 0.0, 1, 1)
    )
    scale_x: float = 1.0
    scale_y: float = 1.0
    raw_result: Any = None


def _tracks_from_result(
    result,
    letterbox: LetterboxTransform,
    *,
    camera_location: str = "",
) -> List[TrackPacket]:
    from gpu_pipeline import unmap_letterbox_xyxy_boxes

    tracks: List[TrackPacket] = []
    if result.boxes is None or len(result.boxes) == 0:
        return tracks

    boxes = result.boxes.xyxy
    if not letterbox.is_identity:
        boxes = unmap_letterbox_xyxy_boxes(boxes, letterbox)
    confs = result.boxes.conf
    ids = result.boxes.id
    min_conf = tracker_config.track_conf_for_camera(camera_location)
    for i in range(len(boxes)):
        conf = float(confs[i].item()) if confs is not None else 1.0
        if conf < min_conf:
            continue
        bbox = boxes[i]
        if hasattr(bbox, "tolist"):
            bx = bbox.tolist()
        else:
            bx = [float(v) for v in bbox[:4]]
        track_id = int(ids[i].item()) if ids is not None else None
        tracks.append(
            TrackPacket(
                bbox=(float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])),
                track_id=track_id,
                conf=conf,
            )
        )
    return tracks


def _process_frame_job(model, job: FrameJob) -> InferenceResult:
    from gpu_pipeline import bgr_numpy_to_gpu, resize_hwc, yolo_track_batch, yolo_track_source
    from inference_device import yolo_device_arg

    track_kw = {**_track_kwargs(), "device": yolo_device_arg(), "persist": True}
    frame_gpu = bgr_numpy_to_gpu(job.frame)
    inf_w, inf_h = _inference_wh()
    scaled, letterbox = resize_hwc(frame_gpu, inf_w, inf_h)
    results = model.track(yolo_track_source(scaled), **track_kw)
    result = results[0] if not isinstance(results, list) else results[0]

    return InferenceResult(
        camera_location=job.camera_location,
        slot=job.slot,
        frame_index=job.frame_index,
        tracks=_tracks_from_result(
            result, letterbox, camera_location=job.camera_location
        ),
        letterbox=letterbox,
        raw_result=result,
    )


def _process_batch(model, jobs: List[FrameJob]) -> List[InferenceResult]:
    """Batch YOLO track — keeps separate ByteTrack state per camera stream."""
    if not jobs:
        return []
    if len(jobs) == 1:
        return [_process_frame_job(model, jobs[0])]

    from gpu_pipeline import bgr_numpy_to_gpu, resize_hwc, yolo_track_batch, yolo_track_source
    from inference_device import yolo_device_arg

    track_kw = {**_track_kwargs(), "device": yolo_device_arg(), "persist": True}
    scaled_frames = []
    meta: List[Tuple[FrameJob, LetterboxTransform]] = []
    inf_w, inf_h = _inference_wh()

    for job in jobs:
        frame_gpu = bgr_numpy_to_gpu(job.frame)
        scaled, letterbox = resize_hwc(frame_gpu, inf_w, inf_h)
        scaled_frames.append(scaled)
        meta.append((job, letterbox))

    results = model.track(yolo_track_batch(scaled_frames), **track_kw)
    if not isinstance(results, list):
        results = [results]
    if len(results) != len(jobs):
        log.warning(
            "GPU batch track returned %d/%d results — falling back to sequential",
            len(results),
            len(jobs),
        )
        return [_process_frame_job(model, job) for job in jobs]

    out: List[InferenceResult] = []
    for (job, letterbox), result in zip(meta, results):
        out.append(
            InferenceResult(
                camera_location=job.camera_location,
                slot=job.slot,
                frame_index=job.frame_index,
                tracks=_tracks_from_result(
                    result, letterbox, camera_location=job.camera_location
                ),
                letterbox=letterbox,
                raw_result=result,
            )
        )
    return out


class GPUInferenceWorker:
    """Background thread owning ONE shared YOLO model on GPU."""

    def __init__(self, model, *, max_queue: int = 64) -> None:
        self.model = model
        self._in_q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._out_q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="gpu-inference-worker",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=5.0)

    def submit(self, job: FrameJob, *, block: bool = False, timeout: float = 0.05) -> bool:
        try:
            if block:
                self._in_q.put(job, timeout=timeout)
            else:
                self._in_q.put_nowait(job)
            return True
        except queue.Full:
            return False
        except Exception:
            return False

    def poll_results(self, max_items: int = 64) -> List[InferenceResult]:
        out: List[InferenceResult] = []
        for _ in range(max_items):
            try:
                item = self._out_q.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                out.append(item)
        return out

    def _collect_batch(self, first: FrameJob) -> List[FrameJob]:
        jobs = [first]
        deadline = time.perf_counter() + BATCH_COLLECT_TIMEOUT_SEC
        while len(jobs) < MAX_BATCH_SIZE:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                job = self._in_q.get(timeout=remaining)
            except queue.Empty:
                break
            if job is None:
                self._stop.set()
                break
            if job.run_inference:
                jobs.append(job)
        return jobs

    def _run_loop(self) -> None:
        from inference_device import yolo_device_arg

        log.info("GPU inference worker online (device=%s)", yolo_device_arg())
        while not self._stop.is_set():
            try:
                job = self._in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                break
            if not job.run_inference:
                continue

            batch = self._collect_batch(job)
            t0 = time.perf_counter()
            try:
                packets = _process_batch(self.model, batch)
                for packet in packets:
                    self._out_q.put(packet)
                ms = (time.perf_counter() - t0) * 1000.0
                log.debug(
                    "GPU worker batch x%d — %.1f ms (%.1f ms/frame)",
                    len(batch),
                    ms,
                    ms / max(len(batch), 1),
                )
            except Exception as exc:
                log.warning("GPU worker batch failed (%d jobs): %s", len(batch), exc)


def start_multiprocessing_worker(
    model_path: str,
    in_q,
    out_q,
) -> None:
    """Process entrypoint: load one YOLO stack on cuda:0."""
    from inference_device import configure
    from ultralytics import YOLO

    configure("cuda:0", hybrid_face_on_cpu=False, pure_gpu=True)
    model = YOLO(model_path)

    while True:
        try:
            job = in_q.get(timeout=0.2)
        except Exception:
            continue
        if job is None:
            break
        if not isinstance(job, FrameJob) or not job.run_inference:
            continue
        try:
            for packet in _process_batch(model, [job]):
                out_q.put(packet)
        except Exception as exc:
            log.warning("MP GPU worker error: %s", exc)
