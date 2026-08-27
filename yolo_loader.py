"""
YOLO model loading for Orion Tactical CCTV.

Default: yolov8l.pt (YOLOv8 Large) on NVIDIA CUDA for command-center recall.
Legacy yolov8s-p2.pt (stride-4 small-object head) remains available via --model.

Bounding-box contract (unchanged across all YOLOv8 sizes):
  ultralytics Results → result.boxes.xyxy  (N×4 float tensor/array, xyxy pixels)
  Consumed by orion_frame_pipeline.YoloPersonCropStage, identity scan, face crops.
Similarity thresholds (85% alert tier) are unchanged downstream.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from ultralytics import YOLO

log = logging.getLogger("yolo_loader")

PROJECT_DIR = Path(__file__).resolve().parent

# Default production detector — YOLOv8 Large (CUDA-only at load time)
YOLO_LARGE_WEIGHTS = "yolov8l.pt"
DEFAULT_MODEL = PROJECT_DIR / YOLO_LARGE_WEIGHTS

# Legacy P2 small-object head (optional --model yolov8s-p2.pt)
YOLO_ARCH_YAML = "yolov8s-p2.yaml"
YOLO_BASE_WEIGHTS = "yolov8s.pt"
YOLO_P2_WEIGHTS = "yolov8s-p2.pt"

MIN_LARGE_BYTES = 70_000_000
MIN_BASE_BYTES = 18_000_000
MIN_P2_BYTES = 15_000_000

# Exact terminal status line after a successful Large-model CUDA load
YOLO_LARGE_LOAD_STATUS = (
    "YOLO Large loaded: yolov8l.pt → cuda (GPU-only, no CPU fallback)"
)


def _p2_path(project_dir: Path) -> Path:
    return project_dir / YOLO_P2_WEIGHTS


def _base_path(project_dir: Path) -> Path:
    return project_dir / YOLO_BASE_WEIGHTS


def _large_path(project_dir: Path) -> Path:
    return project_dir / YOLO_LARGE_WEIGHTS


def _requires_cuda(weights_path: Path) -> bool:
    """Large YOLO weights must not run on CPU (10-min scans would stall)."""
    name = weights_path.name.lower()
    return name == YOLO_LARGE_WEIGHTS.lower() or name.startswith("yolov8l")


def _cuda_hardware_ready() -> bool:
    """True when PyTorch can see an NVIDIA device (strict gate for yolov8l)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def p2_weights_ready(project_dir: Optional[Path] = None) -> bool:
    path = _p2_path(project_dir or PROJECT_DIR)
    return path.is_file() and path.stat().st_size >= MIN_P2_BYTES


def build_p2_weights(
    project_dir: Optional[Path] = None,
    *,
    force: bool = False,
) -> Path:
    """
    Fuse yolov8s.pt backbone weights into the P2 architecture and save yolov8s-p2.pt.
    Run once after download_model.py --size s (or automatically on first load).
    """
    root = project_dir or PROJECT_DIR
    out = _p2_path(root)
    base = _base_path(root)

    if not force and out.is_file() and out.stat().st_size >= MIN_P2_BYTES:
        return out

    if not base.is_file() or base.stat().st_size < MIN_BASE_BYTES:
        raise FileNotFoundError(
            f"Base weights missing: {base}\n"
            "  Run: py -3.12 download_model.py --no-proxy --size s"
        )

    print(f"Building {YOLO_P2_WEIGHTS} ({YOLO_ARCH_YAML} + {YOLO_BASE_WEIGHTS})...")
    model = YOLO(YOLO_ARCH_YAML)
    model.load(str(base))
    model.save(str(out))
    size_mb = out.stat().st_size / 1_000_000
    print(f"Saved P2 weights: {out} ({size_mb:.1f} MB)")
    return out


def resolve_model_path(
    model_arg: Union[str, Path],
    project_dir: Optional[Path] = None,
    *,
    auto_build_p2: bool = True,
) -> Path:
    """
    Resolve CLI / env model path. Defaults to yolov8l.pt (CUDA required at load).
    Legacy yolov8s-p2.pt paths still build from yolov8s.pt when requested.
    """
    root = project_dir or PROJECT_DIR
    raw = Path(model_arg)
    path = raw if raw.is_absolute() else root / raw

    name = path.name.lower()
    if name in (YOLO_P2_WEIGHTS.lower(), YOLO_ARCH_YAML.lower()):
        if auto_build_p2 and not p2_weights_ready(root):
            build_p2_weights(root)
        p2 = _p2_path(root)
        if p2.is_file() and p2.stat().st_size >= MIN_P2_BYTES:
            return p2
        if auto_build_p2:
            return build_p2_weights(root)
        raise FileNotFoundError(f"P2 weights not found: {p2}")

    if name == YOLO_LARGE_WEIGHTS.lower() or name.startswith("yolov8l"):
        large = path if path.is_file() else _large_path(root)
        if large.is_file() and large.stat().st_size >= MIN_LARGE_BYTES:
            return large
        raise FileNotFoundError(
            f"YOLOv8 Large weights missing: {large}\n"
            "  Run: py -3.12 download_model.py --no-proxy --size l"
        )

    if name == YOLO_BASE_WEIGHTS.lower():
        if auto_build_p2:
            return build_p2_weights(root)
        if path.exists() and path.stat().st_size >= MIN_BASE_BYTES:
            return path

    if path.exists() and path.stat().st_size >= MIN_BASE_BYTES:
        return path

    if auto_build_p2 and _base_path(root).is_file():
        return build_p2_weights(root)

    raise FileNotFoundError(
        f"YOLO weights not found or too small: {path}\n"
        "Download YOLOv8 Large (recommended):\n"
        "  py -3.12 download_model.py --no-proxy --size l\n"
        "Or legacy P2: py -3.12 download_model.py --no-proxy --size p2"
    )


def _enforce_large_model_cuda(resolved: Path) -> None:
    """Raise immediately if yolov8l cannot run on CUDA — no silent CPU fallback."""
    if not _requires_cuda(resolved):
        return
    if not _cuda_hardware_ready():
        raise RuntimeError(
            "CRITICAL: yolov8l.pt requires NVIDIA CUDA "
            "(torch.cuda.is_available() == False).\n"
            "  CPU fallback is disabled to prevent 10-minute scan latency collapse.\n"
            "  Run: py -3.12 run_system.py --device cuda\n"
            "  Install: py -3.12 -m pip install torch --index-url "
            "https://download.pytorch.org/whl/cu128"
        )
    from inference_device import torch_cuda_active

    if not torch_cuda_active():
        raise RuntimeError(
            "CRITICAL: CUDA hardware present but runtime resolved to CPU for yolov8l.pt.\n"
            "  Run: py -3.12 run_system.py --device cuda"
        )


def load_yolo_model(
    model_path: Optional[Union[str, Path]] = None,
    project_dir: Optional[Path] = None,
) -> YOLO:
    """
    Load YOLO on NVIDIA CUDA. yolov8l.pt is GPU-only — raises on CPU or CUDA failure.

    Downstream bbox contract: result.boxes.xyxy (N×4 float) — unchanged vs yolov8s-p2.
    """
    from inference_device import log_device_info, yolo_device_arg

    root = project_dir or PROJECT_DIR
    if model_path is None:
        resolved = resolve_model_path(DEFAULT_MODEL, root)
    else:
        resolved = resolve_model_path(model_path, root)

    cuda_required = _requires_cuda(resolved)
    if cuda_required:
        _enforce_large_model_cuda(resolved)

    log_device_info()
    model = YOLO(str(resolved))

    if cuda_required or _cuda_hardware_ready():
        try:
            model.to("cuda")
        except Exception as exc:
            if cuda_required:
                raise RuntimeError(
                    f"CRITICAL: model.to('cuda') failed for {resolved.name}: {exc}\n"
                    "  Refusing CPU fallback for YOLOv8 Large."
                ) from exc
            print(f"[WARN] Could not move YOLO to GPU: {exc}")
    elif cuda_required:
        raise RuntimeError(
            f"CRITICAL: {resolved.name} loaded but CUDA is inactive — refusing CPU execution."
        )

    device = yolo_device_arg()
    if resolved.name.lower() == YOLO_P2_WEIGHTS.lower():
        arch = getattr(getattr(model, "model", None), "yaml", {}) or {}
        arch_name = (
            Path(str(arch.get("yaml_file", YOLO_ARCH_YAML))).name
            if isinstance(arch, dict)
            else resolved.name
        )
        print(
            f"YOLO P2 loaded: {resolved.name} → {device} "
            f"(stride-4 small-object head, arch={arch_name})"
        )
    elif cuda_required:
        print(YOLO_LARGE_LOAD_STATUS)
    log.info("YOLO model %s on %s", resolved.name, device)
    return model
