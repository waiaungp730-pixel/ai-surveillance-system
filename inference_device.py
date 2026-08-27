"""
Shared inference device selection for YOLO (PyTorch), ONNX Runtime, and torchreid.

Configured once per process via configure() from main.py / run_system.py.

Usage in inference loops:
    configure("auto")
    device = get_device()                    # torch.device('cuda:0' or 'cpu')
    model = move_model_to_device(my_model)   # .to(device)
    batch = to_device(batch)                 # tensors before forward()
    result = tensor_to_numpy(output)         # .detach().cpu().numpy() at I/O boundary
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, List, Optional, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort
    import torch
    import torch.nn as nn

log = logging.getLogger("inference_device")

DEVICE_CHOICES = ("auto", "cpu", "cuda", "cuda:0")
_RESOLVED_DEVICE_CHOICES = ("auto", "cpu", "cuda:0")


def normalize_device_choice(choice: str) -> str:
    """Map CLI aliases (e.g. cuda) to resolved device strings."""
    normalized = (choice or "auto").strip().lower()
    if normalized == "cuda":
        return "cuda:0"
    return normalized

_runtime_choice: str = "auto"
_resolved_torch_device: str = "cpu"
_cuda_active: bool = False
_hybrid_face_on_cpu: bool = False
_pure_gpu: bool = False
_torch_device_obj: Optional["torch.device"] = None


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _gpu_sm_tag(major: int, minor: int) -> str:
    return f"sm_{major}{minor}"


def _torch_cuda_compatible() -> bool:
    """True when CUDA is available and PyTorch ships kernels for this GPU."""
    if not _torch_cuda_available():
        return False
    try:
        import torch

        arch_list = torch.cuda.get_arch_list()
        if not arch_list:
            return True
        major, minor = torch.cuda.get_device_capability(0)
        sm = _gpu_sm_tag(major, minor)
        if sm in arch_list:
            return True
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "GPU"
        supported = ", ".join(arch_list)
        log.warning(
            "%s (%s) is not supported by this PyTorch build (supports: %s). "
            "Install PyTorch with CUDA 12.8+ for RTX 50-series, or run with --device cpu.",
            name,
            sm,
            supported,
        )
        print(
            f"WARNING: {name} ({sm}) needs PyTorch cu128 - falling back to CPU. "
            "See https://pytorch.org/get-started/locally/ or use --device cpu."
        )
        return False
    except Exception as exc:
        log.warning("CUDA compatibility check failed (%s) — using CPU", exc)
        return False


def resolve_torch_device(choice: str) -> str:
    """Map CLI choice to an ultralytics/torch device string."""
    normalized = normalize_device_choice(choice)
    if normalized not in _RESOLVED_DEVICE_CHOICES:
        normalized = "auto"

    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda:0":
        if _torch_cuda_compatible():
            return "cuda:0"
        log.warning("cuda:0 requested but CUDA is unavailable or incompatible — using CPU")
        return "cpu"

    # auto
    if _torch_cuda_compatible():
        return "cuda:0"
    return "cpu"


def get_device() -> "torch.device":
    """Return the active torch.device (cuda:0 when available, else cpu)."""
    import torch

    global _torch_device_obj
    if _torch_device_obj is None:
        _torch_device_obj = torch.device(_resolved_torch_device)
    return _torch_device_obj


def print_cuda_diagnostics() -> None:
    """
    မြန်မာ: CUDA / cuDNN / ONNX Runtime provider များကို စစ်ဆေးပြီး
    CPU သို့ silent-fallback ဖြစ်နေပါက သတိပေးချက် ပြသသည်။
    """
    import torch

    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")

    if cuda_avail:
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(f"CUDA Device: {name} (sm_{cap[0]}{cap[1]})")
        except Exception as exc:
            print(f"CUDA Device: unavailable ({exc})")
        pt_cuda = getattr(torch.version, "cuda", None)
        if pt_cuda:
            print(f"PyTorch CUDA build: {pt_cuda}")
        try:
            if torch.backends.cudnn.is_available():
                print(f"cuDNN: {torch.backends.cudnn.version()}")
            else:
                print("cuDNN: not available")
        except Exception:
            pass

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        if cuda_avail and "CUDAExecutionProvider" not in providers:
            print(
                "WARNING: CUDAExecutionProvider missing — face ONNX will silent-fallback to CPU.\n"
                "  Fix: pip install onnxruntime-gpu  (must match your CUDA Toolkit version)"
            )
        elif cuda_avail and "CUDAExecutionProvider" in providers:
            print("ONNX CUDAExecutionProvider: ready")
            try:
                import onnxruntime as ort

                ort_pkg = getattr(ort, "__version__", "unknown")
                pt_cuda = getattr(torch.version, "cuda", None)
                print(f"ONNX Runtime: {ort_pkg} | PyTorch CUDA: {pt_cuda or 'n/a'}")
                if pt_cuda and not pt_cuda.startswith("12"):
                    print(
                        "WARNING: PyTorch CUDA major version is not 12.x — "
                        "onnxruntime-gpu wheels expect CUDA 12.x"
                    )
            except Exception:
                pass
    except ImportError:
        print(
            "WARNING: onnxruntime not installed — RetinaFace ONNX GPU disabled.\n"
            "  Fix: pip install onnxruntime-gpu"
        )

    if cuda_avail and not _cuda_active:
        print(
            f"WARNING: CUDA hardware present but runtime resolved to CPU "
            f"(choice={_runtime_choice!r}, resolved={_resolved_torch_device!r})"
        )
    if _pure_gpu and _cuda_active:
        print("GPU mode: pure-gpu (CUDA-only ONNX, no CPU fallback)")


def log_device_info() -> None:
    """Log CUDA availability and the resolved inference device."""
    import torch

    dev = get_device()
    if dev.type == "cuda" and torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(dev.index or 0)
        except Exception:
            name = "CUDA"
        log.info("Inference device: %s (%s)", dev, name)
        print(f"Inference device: {dev} ({name})")
    else:
        log.info("Inference device: cpu (CUDA unavailable)")
        print("Inference device: cpu (CUDA unavailable — install CUDA-enabled PyTorch for GPU)")


def to_device(
    value: Union["torch.Tensor", "nn.Module"],
    *,
    non_blocking: bool = True,
) -> Union["torch.Tensor", "nn.Module"]:
    """Move a tensor or nn.Module to the active inference device."""
    import torch

    dev = get_device()
    if isinstance(value, torch.nn.Module):
        return value.to(dev)
    if isinstance(value, torch.Tensor):
        if value.device == dev:
            return value
        use_non_blocking = non_blocking and dev.type == "cuda"
        return value.to(dev, non_blocking=use_non_blocking)
    raise TypeError(f"to_device() expects Tensor or Module, got {type(value)!r}")


def move_model_to_device(model: "nn.Module") -> "nn.Module":
    """Place a PyTorch model on the active device and set eval mode."""
    model = to_device(model)  # type: ignore[assignment]
    model.eval()
    return model


def tensor_from_numpy(
    array: np.ndarray,
    *,
    dtype: Optional[Any] = None,
    non_blocking: bool = True,
) -> "torch.Tensor":
    """Create a tensor from NumPy on the active device (host→device copy once)."""
    import torch

    arr = np.ascontiguousarray(array)
    tensor = torch.from_numpy(arr)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return to_device(tensor, non_blocking=non_blocking)  # type: ignore[return-value]


def tensor_to_numpy(
    value: Union["torch.Tensor", np.ndarray],
    *,
    copy: bool = False,
) -> np.ndarray:
    """Sync a tensor to host NumPy via .detach().cpu().numpy() (I/O boundary only)."""
    import torch

    if isinstance(value, np.ndarray):
        return value.copy() if copy else value
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        return arr.copy() if copy else arr
    return np.asarray(value)


def configure(
    choice: str = "auto",
    hybrid_face_on_cpu: bool = False,
    pure_gpu: bool = False,
) -> str:
    """Apply runtime device settings for this process. Returns resolved torch device."""
    global _runtime_choice, _resolved_torch_device, _cuda_active, _hybrid_face_on_cpu, _pure_gpu
    global _torch_device_obj
    normalized = normalize_device_choice(choice)
    _runtime_choice = (
        normalized if normalized in _RESOLVED_DEVICE_CHOICES else "auto"
    )
    _pure_gpu = bool(pure_gpu)
    _hybrid_face_on_cpu = bool(hybrid_face_on_cpu) and not _pure_gpu
    _resolved_torch_device = resolve_torch_device(_runtime_choice)
    _cuda_active = _resolved_torch_device.startswith("cuda")
    _torch_device_obj = None
    if not _hybrid_face_on_cpu and _cuda_active:
        _pure_gpu = True
    if _pure_gpu and not _cuda_active:
        log.warning("pure_gpu requested but CUDA unavailable — continuing on CPU")
    if _runtime_choice in ("cuda", "cuda:0") or _runtime_choice == "auto":
        print_cuda_diagnostics()
    return _resolved_torch_device


def get_torch_device() -> str:
    return _resolved_torch_device


def torch_cuda_active() -> bool:
    return _cuda_active


def pure_gpu_active() -> bool:
    return _pure_gpu and _cuda_active


def face_on_cpu() -> bool:
    """True when face ONNX / anti-spoof should stay on CPU (hybrid or no CUDA)."""
    return _hybrid_face_on_cpu or not _cuda_active


def use_gpu_for_reid() -> bool:
    """torchreid OSNet — GPU only when CUDA is active and hybrid mode is off."""
    return _cuda_active and not _hybrid_face_on_cpu


def yolo_device_arg() -> str:
    """Device argument for ultralytics YOLO track/predict calls."""
    return _resolved_torch_device


def describe_device() -> str:
    parts = [f"torch={_resolved_torch_device}"]
    if _pure_gpu and _cuda_active:
        parts.append("mode=pure-gpu")
    if _hybrid_face_on_cpu and _cuda_active:
        parts.append("face=cpu (hybrid)")
    elif face_on_cpu():
        parts.append("face=cpu")
    else:
        parts.append(f"face={onnx_providers()[0]}")
    if use_gpu_for_reid():
        parts.append("reid=gpu")
    else:
        parts.append("reid=cpu")
    return ", ".join(parts)


def onnx_providers() -> List[str]:
    """Ordered ONNX Runtime providers with CPU fallback terminal entry."""
    if _pure_gpu and _cuda_active:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if face_on_cpu():
        return ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except ImportError:
        return ["CPUExecutionProvider"]

    preferred: List[str] = []
    if _cuda_active and "CUDAExecutionProvider" in available:
        preferred.append("CUDAExecutionProvider")
    elif _runtime_choice == "auto" and "DmlExecutionProvider" in available:
        # Windows DirectML when CUDA torch is not installed but GPU EP exists.
        preferred.append("DmlExecutionProvider")

    if "CPUExecutionProvider" not in preferred:
        preferred.append("CPUExecutionProvider")
    return preferred


def create_onnx_session(model_path: str, *, label: str = "ONNX") -> "ort.InferenceSession":
    """Create an ONNX Runtime session with provider fallback on failure."""
    import onnxruntime as ort

    if _pure_gpu and _cuda_active:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_cpu_mem_arena = True
        try:
            cuda_options = {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "DEFAULT",
            }
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=[
                    ("CUDAExecutionProvider", cuda_options),
                    "CPUExecutionProvider",
                ],
            )
        except TypeError:
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        log.info("%s loaded with CUDAExecutionProvider (pure GPU)", label)
        return session

    providers = onnx_providers()
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        active = session.get_providers()
        active_ep = active[0] if active else "unknown"
        log.info("%s loaded with %s", label, active_ep)
        if _cuda_active and active_ep == "CPUExecutionProvider":
            print(
                f"WARNING: {label} silent-fallback to CPUExecutionProvider — "
                "install onnxruntime-gpu matching your CUDA Toolkit"
            )
        return session
    except Exception as exc:
        if _cuda_active and not face_on_cpu():
            log.error(
                "%s CUDA providers %s failed (%s) — refusing CPU fallback in GPU mode",
                label,
                providers,
                exc,
            )
            raise
        log.warning(
            "%s providers %s failed (%s) — retrying CPUExecutionProvider",
            label,
            providers,
            exc,
        )
        session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        log.info("%s loaded with CPUExecutionProvider (fallback)", label)
        return session


# ── Per-thread CUDA streams (parallel folder video scan) ─────────────────────

_thread_local = threading.local()


def thread_cuda_stream() -> Optional["torch.cuda.Stream"]:
    """Dedicated CUDA stream for the current thread (lazy-created)."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    stream = getattr(_thread_local, "cuda_stream", None)
    if stream is None:
        stream = torch.cuda.Stream()
        _thread_local.cuda_stream = stream
    return stream


@contextmanager
def cuda_stream_context(*, sync: bool = True):
    """
    Run PyTorch CUDA kernels on this thread's stream.

    Used by parallel folder scans so two videos can overlap YOLO inference.
    """
    stream = thread_cuda_stream()
    if stream is None:
        yield None
        return
    import torch

    with torch.cuda.stream(stream):
        try:
            yield stream
        finally:
            if sync:
                stream.synchronize()
