"""
Blind face super-resolution for command-center CCTV (GFPGAN / CodeFormer on CUDA).

Lightweight GPU restoration before ArcFace embedding — reconstructs landmarks and
upscales blurry low-res face patches from YOLO person crops.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("face_restoration")

PROJECT_DIR = Path(__file__).resolve().parent
GFPGAN_WEIGHTS = PROJECT_DIR / "models" / "GFPGANv1.4.pth"
CODEFORMER_WEIGHTS = PROJECT_DIR / "models" / "codeformer.pth"

MIN_WEIGHT_BYTES = 300_000_000
DEFAULT_UPSCALE = 2
CODEFORMER_FIDELITY = 0.5


def _patch_basicsr_torchvision() -> None:
    """Newer torchvision dropped functional_tensor; basicsr still imports it."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ModuleNotFoundError:
        import sys
        import types

        import torchvision.transforms.functional as F

        mod = types.ModuleType("torchvision.transforms.functional_tensor")
        mod.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = mod


_patch_basicsr_torchvision()


class RestorationBackend(str, Enum):
    GFPGAN = "gfpgan"
    CODEFORMER = "codeformer"
    NONE = "none"


_engine: Optional["FaceRestorationEngine"] = None
_engine_lock = threading.Lock()


def weights_ready() -> bool:
    gfp = GFPGAN_WEIGHTS.is_file() and GFPGAN_WEIGHTS.stat().st_size >= MIN_WEIGHT_BYTES
    cf = CODEFORMER_WEIGHTS.is_file() and CODEFORMER_WEIGHTS.stat().st_size >= 8_000_000
    return gfp or cf


def active_backend_name() -> str:
    eng = get_face_restoration_engine(initialize=False)
    if eng is None or not eng.ready:
        return "disabled"
    return eng.backend.value


def get_face_restoration_engine(
    *,
    initialize: bool = True,
    device: Optional[str] = None,
    backend: str = "auto",
) -> Optional["FaceRestorationEngine"]:
    global _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        if not initialize:
            return None
        try:
            from inference_device import get_device, torch_cuda_active

            dev = device or (str(get_device()) if torch_cuda_active() else "cpu")
            _engine = FaceRestorationEngine(device=dev, backend=backend)
            _engine.ensure_loaded()
            return _engine
        except Exception as exc:
            log.warning("Face restoration engine unavailable: %s", exc)
            return None


class FaceRestorationEngine:
    """GFPGAN (preferred) or CodeFormer blind face restoration on CUDA."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        upscale: int = DEFAULT_UPSCALE,
        backend: str = "auto",
    ) -> None:
        self.device = device
        self.upscale = max(1, int(upscale))
        self._backend_pref = backend.lower().strip()
        self._gfpgan = None
        self._codeformer = None
        self._codeformer_device = None
        self.backend = RestorationBackend.NONE
        self.ready = False

    def ensure_loaded(self) -> bool:
        if self.ready:
            return True

        if self._backend_pref in ("gfpgan", "auto") and self._try_load_gfpgan():
            self.backend = RestorationBackend.GFPGAN
            self.ready = True
            print(
                f"Face restoration: GFPGAN on {self.device} "
                f"(upscale={self.upscale}x, weights={GFPGAN_WEIGHTS.name})"
            )
            return True

        if self._backend_pref in ("codeformer", "auto") and self._try_load_codeformer():
            self.backend = RestorationBackend.CODEFORMER
            self.ready = True
            print(
                f"Face restoration: CodeFormer on {self.device} "
                f"(weights={CODEFORMER_WEIGHTS.name})"
            )
            return True

        log.warning(
            "Face restoration weights missing — run: python download_face_restoration.py --no-proxy"
        )
        return False

    def _try_load_gfpgan(self) -> bool:
        if not GFPGAN_WEIGHTS.is_file() or GFPGAN_WEIGHTS.stat().st_size < MIN_WEIGHT_BYTES:
            return False
        try:
            from gfpgan import GFPGANer

            self._gfpgan = GFPGANer(
                model_path=str(GFPGAN_WEIGHTS),
                upscale=self.upscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device,
            )
            return True
        except ImportError:
            log.warning("gfpgan not installed — pip install gfpgan basicsr facexlib")
            return False
        except Exception as exc:
            log.warning("GFPGAN load failed: %s", exc)
            return False

    def _try_load_codeformer(self) -> bool:
        if not CODEFORMER_WEIGHTS.is_file():
            return False
        try:
            import torch
            from basicsr.utils.registry import ARCH_REGISTRY

            net = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            )
            ckpt = torch.load(str(CODEFORMER_WEIGHTS), map_location="cpu")
            state = ckpt.get("params_ema", ckpt)
            net.load_state_dict(state, strict=True)
            self._codeformer_device = torch.device(self.device)
            net.eval().to(self._codeformer_device)
            self._codeformer = net
            return True
        except ImportError:
            log.warning("CodeFormer requires basicsr — pip install basicsr")
            return False
        except Exception as exc:
            log.warning("CodeFormer load failed: %s", exc)
            return False

    def restore_face(
        self,
        face_bgr: np.ndarray,
        *,
        force: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Blind face restoration on a BGR patch.

        Returns (restored_bgr, was_restored). When restoration is unavailable or
        the patch is already sharp/large enough, returns the input unchanged.
        """
        if face_bgr is None or face_bgr.size == 0:
            return face_bgr, False
        if not self.ready and not self.ensure_loaded():
            return face_bgr, False

        h, w = face_bgr.shape[:2]
        if not force and min(h, w) >= 160 and _laplacian_variance(face_bgr) >= 120.0:
            return face_bgr, False

        try:
            if self.backend == RestorationBackend.GFPGAN and self._gfpgan is not None:
                restored = self._restore_gfpgan(face_bgr)
                if restored is not None:
                    return restored, True
            if self.backend == RestorationBackend.CODEFORMER and self._codeformer is not None:
                restored = self._restore_codeformer(face_bgr)
                if restored is not None:
                    return restored, True
        except Exception as exc:
            log.debug("Face restoration failed: %s", exc)

        return face_bgr, False

    def _restore_gfpgan(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        assert self._gfpgan is not None
        _, restored_faces, _ = self._gfpgan.enhance(
            face_bgr,
            has_aligned=False,
            only_center_face=True,
            paste_back=False,
        )
        if not restored_faces:
            return None
        out = restored_faces[0]
        return out if out is not None and out.size > 0 else None

    def _restore_codeformer(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        import torch
        from torchvision.transforms.functional import normalize

        assert self._codeformer is not None
        device = self._codeformer_device or torch.device(self.device)

        img = cv2.resize(face_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        tensor = normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=False)

        with torch.no_grad():
            output = self._codeformer(tensor, w=CODEFORMER_FIDELITY, adain=True)[0]

        restored = output.squeeze(0).permute(1, 2, 0).clamp(-1, 1)
        restored = ((restored + 1.0) / 2.0 * 255.0).byte().cpu().numpy()
        restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
        return restored


def _laplacian_variance(face_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def should_enhance_for_scan(
    face_bgr: np.ndarray,
    *,
    max_side: int = 80,
    sharp_min: float = 120.0,
) -> bool:
    """True when max(w,h) < max_side — large sharp crops are left untouched."""
    if face_bgr is None or face_bgr.size == 0:
        return False
    h, w = face_bgr.shape[:2]
    return max(w, h) < max_side


def restore_for_scan(
    face_bgr: np.ndarray,
    *,
    force: bool = False,
) -> Tuple[np.ndarray, bool]:
    """GFPGAN/CodeFormer on small CCTV crops before ArcFace alignment."""
    eng = get_face_restoration_engine(initialize=True)
    if eng is None:
        return face_bgr, False
    return eng.restore_face(face_bgr, force=force)
