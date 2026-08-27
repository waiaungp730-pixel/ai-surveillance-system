"""
Silent-Face-Anti-Spoofing ONNX inference (MiniFASNet 80×80).

Model weights: models/anti_spoof.onnx (see create_download_anti_spoof.py).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from gpu_pipeline import bgr_hwc_to_chw_normalized, crop_xyxy, resize_bilinear_hwc
from inference_device import create_onnx_session, tensor_to_numpy

log = logging.getLogger("anti_spoof")


class AntiSpoof:
    """ONNX liveness check; no-op stub when weights are missing."""

    def __init__(self, model_path: Path, threshold: float = 0.85) -> None:
        self._threshold = threshold
        self._session = None
        self._input_name: Optional[str] = None
        path = Path(model_path)
        if not path.is_file():
            log.warning("Anti-spoof model not found at %s — checks disabled", path)
            return
        try:
            self._session = create_onnx_session(str(path), label="Anti-spoof")
            self._input_name = self._session.get_inputs()[0].name
            log.debug("Anti-spoof ONNX loaded: %s", path)
        except Exception as exc:
            log.warning("Anti-spoof session failed to load (%s) — checks disabled", exc)
            self._session = None

    @property
    def enabled(self) -> bool:
        return self._session is not None

    def is_real(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[bool, float]:
        """
        Crop + resize to 80×80, run inference.
        Returns (is_real, confidence). Never raises.
        """
        if self._session is None or self._input_name is None:
            return True, 1.0
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return False, 0.0

            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (80, 80))
            blob = crop.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis]

            t0 = __import__("time").perf_counter()
            outputs = self._session.run(None, {self._input_name: blob})
            ms = (__import__("time").perf_counter() - t0) * 1000.0
            log.debug("Anti-spoof inference %.1f ms", ms)

            logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            if logits.size >= 2:
                exp = np.exp(logits - logits.max())
                probs = exp / (exp.sum() + 1e-6)
                # MiniFASNet: index 1 is typically "real" for 2-class; for 3-class use last as real.
                real_conf = float(probs[-1]) if probs.size >= 3 else float(probs[1])
            else:
                real_conf = float(logits.reshape(-1)[0])

            return real_conf >= self._threshold, real_conf
        except Exception as exc:
            log.debug("Anti-spoof inference failed: %s", exc)
            return True, 1.0

    def is_real_gpu(
        self,
        frame: torch.Tensor,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[bool, float]:
        """VRAM path: crop/resize/normalize on GPU; sync only the 80×80 ONNX blob."""
        if self._session is None or self._input_name is None:
            return True, 1.0
        try:
            crop = crop_xyxy(frame, bbox)
            if crop is None:
                return False, 0.0
            crop = resize_bilinear_hwc(crop, 80, 80)
            blob = bgr_hwc_to_chw_normalized(crop, rgb_order=False)
            blob_np = tensor_to_numpy(blob)

            t0 = __import__("time").perf_counter()
            outputs = self._session.run(None, {self._input_name: blob_np})
            ms = (__import__("time").perf_counter() - t0) * 1000.0
            log.debug("Anti-spoof GPU inference %.1f ms", ms)

            logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            if logits.size >= 2:
                exp = np.exp(logits - logits.max())
                probs = exp / (exp.sum() + 1e-6)
                real_conf = float(probs[-1]) if probs.size >= 3 else float(probs[1])
            else:
                real_conf = float(logits.reshape(-1)[0])

            return real_conf >= self._threshold, real_conf
        except Exception as exc:
            log.debug("Anti-spoof GPU inference failed: %s", exc)
            return True, 1.0
