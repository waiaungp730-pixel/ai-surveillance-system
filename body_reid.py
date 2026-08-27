"""
Person body Re-ID for CCTV (when faces are too small).
Uses OSNet x1.0 trained on Market-1501 — a proper person re-identification model.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

_body_model = None
_body_transform = None

# Cosine distance: lower = same person (OSNet is stricter than generic ResNet18)
BODY_MATCH_THRESHOLD = 0.45

WEIGHTS_NAME = "osnet_x1_0_reid.pth"
PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_WEIGHTS = PROJECT_DIR / "models" / WEIGHTS_NAME
MIN_WEIGHTS_BYTES = 16_000_000


def _resolve_weights_path() -> Optional[Path]:
    if LOCAL_WEIGHTS.is_file() and LOCAL_WEIGHTS.stat().st_size >= MIN_WEIGHTS_BYTES:
        return LOCAL_WEIGHTS
    return None


def _infer_num_classes(state: dict) -> int:
    for key in ("classifier.weight", "module.classifier.weight"):
        if key in state:
            return int(state[key].shape[0])
    return 751


def _load_osnet(weights_path: Path):
    import torch
    from torchvision import transforms

    from inference_device import get_device, move_model_to_device, use_gpu_for_reid

    try:
        from torchreid.reid.models import build_model
    except ImportError as exc:
        raise ImportError(
            "torchreid is required for body Re-ID.\n"
            "  pip install torchreid gdown\n"
        ) from exc

    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    num_classes = _infer_num_classes(state)
    use_gpu = use_gpu_for_reid()

    model = build_model(
        name="osnet_x1_0",
        num_classes=num_classes,
        loss="softmax",
        pretrained=False,
        use_gpu=use_gpu,
    )
    model.load_state_dict(state, strict=False)
    model.classifier = torch.nn.Identity()
    model = move_model_to_device(model)
    if use_gpu:
        print(f"OSNet body Re-ID: .to('{get_device()}')")

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, transform


def _get_body_model():
    global _body_model, _body_transform
    if _body_model is None:
        weights_path = _resolve_weights_path()
        if weights_path is None:
            raise FileNotFoundError(
                "OSNet body Re-ID weights not found.\n"
                "  Run: python download_body_model.py --no-proxy\n"
                f"  Expected file: {LOCAL_WEIGHTS}"
            )
        print(f"Loading OSNet body Re-ID weights from: {weights_path}")
        _body_model, _body_transform = _load_osnet(weights_path)
    return _body_model, _body_transform


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(1.0 - np.dot(a, b) / denom)


def preload_body_model() -> None:
    """Load OSNet once at startup (avoids long pause on first person detected)."""
    print("Loading body Re-ID model (OSNet)...")
    _get_body_model()
    print("Body Re-ID ready.")


def extract_body_embedding(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 10:
        return None

    import torch

    from inference_device import to_device

    model, transform = _get_body_model()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0)
    tensor = to_device(tensor)
    with torch.no_grad():
        emb = model(tensor).detach().cpu().numpy().flatten()
    norm = np.linalg.norm(emb) + 1e-8
    return emb / norm
