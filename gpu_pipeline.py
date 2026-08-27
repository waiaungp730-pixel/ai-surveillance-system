"""
GPU-resident frame, bbox, and crop utilities for VRAM-only tracking.

Frames are uploaded once per grab; YOLO boxes stay on-device; person/face
crops are sliced on GPU. CPU sync is limited to small tensors at I/O boundaries
(embeddings, capture JPEG, dashboard preview).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from inference_device import (
    get_device,
    get_torch_device,
    tensor_from_numpy,
    tensor_to_numpy,
    to_device,
    torch_cuda_active,
)
from image_geometry import LetterboxTransform

FrameSource = Union[np.ndarray, torch.Tensor]
BBoxSource = Union[np.ndarray, torch.Tensor, Tuple[float, ...]]
_LETTERBOX_PAD_BGR = (24, 16, 12)  # BGR for cv2-style backgrounds on GPU canvas


def vram_pipeline_active() -> bool:
    return torch_cuda_active()


def frame_device(frame: FrameSource) -> torch.device:
    if isinstance(frame, torch.Tensor):
        return frame.device
    return get_device()


def bgr_numpy_to_gpu(
    frame: np.ndarray,
    *,
    device: Optional[str] = None,
    non_blocking: bool = True,
) -> torch.Tensor:
    """Upload a BGR uint8 HWC numpy frame to a GPU tensor (single host→device copy)."""
    if device is not None and device == "cpu":
        return torch.from_numpy(np.ascontiguousarray(frame))
    return tensor_from_numpy(frame, non_blocking=non_blocking)


def resize_hwc(
    frame: FrameSource,
    width: int,
    height: int,
) -> Tuple[torch.Tensor, LetterboxTransform]:
    """Letterbox HWC BGR frame on GPU/CPU — uniform scale, no stretch."""
    if isinstance(frame, np.ndarray):
        h, w = frame.shape[:2]
        if w == width and h == height:
            return bgr_numpy_to_gpu(frame), LetterboxTransform(
                1.0, 0.0, 0.0, width, height
            )
        frame_t = bgr_numpy_to_gpu(frame)
    else:
        h, w = frame.shape[:2]
        if w == width and h == height:
            return frame, LetterboxTransform(1.0, 0.0, 0.0, width, height)
        frame_t = frame

    if w < 1 or h < 1:
        dev = frame_t.device
        canvas = torch.full(
            (height, width, 3),
            12,
            dtype=torch.uint8,
            device=dev,
        )
        return canvas, LetterboxTransform(1.0, 0.0, 0.0, width, height)

    scale = min(width / float(w), height / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    chw = frame_t.permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(chw, size=(new_h, new_w), mode="area")
    content = resized.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8)
    pad_x = (width - new_w) // 2
    pad_y = (height - new_h) // 2
    canvas = torch.full(
        (height, width, 3),
        12,
        dtype=torch.uint8,
        device=content.device,
    )
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = content
    return canvas, LetterboxTransform(
        scale=float(scale),
        pad_x=float(pad_x),
        pad_y=float(pad_y),
        out_w=width,
        out_h=height,
    )


def unmap_letterbox_xyxy_boxes(
    boxes: torch.Tensor,
    letterbox: LetterboxTransform,
) -> torch.Tensor:
    """Map xyxy boxes from letterboxed inference coords back to source frame."""
    if letterbox.is_identity:
        return boxes
    inv = 1.0 / letterbox.scale if letterbox.scale > 0 else 1.0
    scaled = boxes.clone()
    scaled[:, 0] = (scaled[:, 0] - letterbox.pad_x) * inv
    scaled[:, 2] = (scaled[:, 2] - letterbox.pad_x) * inv
    scaled[:, 1] = (scaled[:, 1] - letterbox.pad_y) * inv
    scaled[:, 3] = (scaled[:, 3] - letterbox.pad_y) * inv
    return scaled


def scale_xyxy_boxes(
    boxes: torch.Tensor,
    scale_x: float,
    scale_y: float,
) -> torch.Tensor:
    """Scale xyxy boxes in-place on the current device."""
    if scale_x == 1.0 and scale_y == 1.0:
        return boxes
    scaled = boxes.clone()
    scaled[:, 0] *= scale_x
    scaled[:, 2] *= scale_x
    scaled[:, 1] *= scale_y
    scaled[:, 3] *= scale_y
    return scaled


def _bbox_ints(
    bbox: BBoxSource,
    frame_h: int,
    frame_w: int,
    *,
    pad_ratio: float = 0.0,
) -> Tuple[int, int, int, int]:
    if isinstance(bbox, torch.Tensor):
        x1, y1, x2, y2 = (int(v) for v in bbox[:4].tolist())
    else:
        arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
        x1, y1, x2, y2 = (int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]))

    if pad_ratio > 0.0:
        pad = int(pad_ratio * max(x2 - x1, y2 - y1))
        x1 -= pad
        y1 -= pad
        x2 += pad
        y2 += pad

    x1 = max(0, min(frame_w, x1))
    y1 = max(0, min(frame_h, y1))
    x2 = max(0, min(frame_w, x2))
    y2 = max(0, min(frame_h, y2))
    return x1, y1, x2, y2


def crop_xyxy(
    frame: torch.Tensor,
    bbox: BBoxSource,
    *,
    pad_ratio: float = 0.0,
) -> Optional[torch.Tensor]:
    """Slice a BGR HWC tensor on-device using integer xyxy coordinates."""
    h, w = int(frame.shape[0]), int(frame.shape[1])
    x1, y1, x2, y2 = _bbox_ints(bbox, h, w, pad_ratio=pad_ratio)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def bgr_hwc_to_chw_normalized(
    crop: torch.Tensor,
    *,
    rgb_order: bool = True,
) -> torch.Tensor:
    """HWC BGR uint8/float → 1×3×H×W float32 in [0, 1] on the same device."""
    x = crop.float()
    if x.max() > 1.5:
        x = x / 255.0
    chw = x.permute(2, 0, 1).unsqueeze(0)
    if rgb_order:
        chw = chw[:, [2, 1, 0], :, :]
    return chw.contiguous()


def laplacian_variance(crop: torch.Tensor) -> float:
    """GPU Laplacian variance sharpness for a BGR HWC crop."""
    if crop.numel() == 0:
        return 0.0
    gray = (
        0.114 * crop[..., 0].float()
        + 0.587 * crop[..., 1].float()
        + 0.299 * crop[..., 2].float()
    )
    g = gray.unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(g, kernel, padding=1)
    return float(lap.var().item())


def tensor_to_bgr_numpy(crop: torch.Tensor) -> np.ndarray:
    """Sync a small GPU crop to host BGR uint8 (I/O boundary only)."""
    arr = tensor_to_numpy(crop)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def yolo_track_source(frame: FrameSource) -> Union[torch.Tensor, np.ndarray]:
    """
    Format a frame for ultralytics track().

    GPU pipeline: 1×3×H×W float RGB on CUDA (H/W divisible by stride 32).
    CPU pipeline: numpy HWC BGR (letterbox handled inside ultralytics).
    """
    if vram_pipeline_active():
        if isinstance(frame, np.ndarray):
            frame = bgr_numpy_to_gpu(frame)
        if isinstance(frame, torch.Tensor) and frame.ndim == 3 and int(frame.shape[-1]) == 3:
            return bgr_hwc_to_chw_normalized(frame, rgb_order=True)
        return frame
    if isinstance(frame, torch.Tensor):
        return tensor_to_bgr_numpy(frame)
    return np.ascontiguousarray(frame)


def yolo_track_batch(frames: list) -> Union[torch.Tensor, list, np.ndarray]:
    """Build a single ultralytics source from one or more scaled frames."""
    sources = [yolo_track_source(frame) for frame in frames]
    if len(sources) == 1:
        return sources[0]
    if all(isinstance(source, torch.Tensor) for source in sources):
        return torch.cat(sources, dim=0)
    return sources


def preprocess_retinaface_gpu(
    frame_hwc: torch.Tensor,
    *,
    allow_upscaling: bool = True,
    target_size: int = 1024,
    max_size: int = 1980,
) -> Tuple[torch.Tensor, Tuple[int, int], float]:
    """
    RetinaFace letterbox resize + BGR→RGB CHW normalize entirely on GPU.
    Returns (1×3×H×W float tensor, (h, w), im_scale).
    """
    h, w = int(frame_hwc.shape[0]), int(frame_hwc.shape[1])
    if w > h:
        im_size_min, im_size_max = h, w
    else:
        im_size_min, im_size_max = w, h

    im_scale = target_size / float(im_size_min)
    if not allow_upscaling:
        im_scale = min(1.0, im_scale)
    if round(im_scale * im_size_max) > max_size:
        im_scale = max_size / float(im_size_max)

    if im_scale != 1.0:
        new_h = max(1, int(round(h * im_scale)))
        new_w = max(1, int(round(w * im_scale)))
        chw = frame_hwc.permute(2, 0, 1).unsqueeze(0).float()
        chw = F.interpolate(chw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        frame_hwc = chw.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8)
        h, w = new_h, new_w

    im_tensor = bgr_hwc_to_chw_normalized(frame_hwc, rgb_order=True)
    return im_tensor, (h, w), im_scale


def resize_bilinear_hwc(crop: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a HWC tensor with bilinear sampling on-device."""
    chw = crop.permute(2, 0, 1).unsqueeze(0).float()
    out = F.interpolate(chw, size=(height, width), mode="bilinear", align_corners=False)
    return out.squeeze(0).permute(1, 2, 0)
