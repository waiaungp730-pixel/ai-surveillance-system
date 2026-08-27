"""
Download OpenCV YuNet + SFace models for face recognition.

This project uses OpenCV's YuNet (face detection) and SFace (face recognition).
These are the models identity.py loads — not DeepFace, InsightFace, or dlib.

  python download_face_models.py --no-proxy
  python download_face_models.py --no-proxy --verify
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

MODELS_DIR = Path(__file__).parent / "models"

# YuNet 2023 + SFace 2021 — OpenCV Zoo production models used by identity.py
FILES: Dict[str, Tuple[str, int]] = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        200_000,
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        30_000_000,
    ),
}


def clear_proxy_env() -> None:
    for key in list(os.environ):
        if "proxy" in key.lower():
            os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"


def download(url: str, dest: Path, min_bytes: int) -> None:
    print(f"  Downloading {dest.name} …")
    last_err = None
    for attempt in range(1, 4):
        try:
            if dest.exists():
                dest.unlink(missing_ok=True)
            urllib.request.urlretrieve(url, dest)
            size = dest.stat().st_size
            if size < min_bytes:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"File too small ({size:,} bytes; need ≥ {min_bytes:,})"
                )
            print(f"    OK ({size:,} bytes)")
            return
        except Exception as exc:
            last_err = exc
            print(f"    Attempt {attempt}/3 failed: {exc}")
    raise RuntimeError(f"Could not download {dest.name}: {last_err}")


def verify_models() -> bool:
    """Load models the same way identity.py does at runtime."""
    yunet = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    sface = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
    if not yunet.is_file() or not sface.is_file():
        print("  Verify failed: model files missing.")
        return False

    try:
        import cv2

        detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320))
        recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
        if detector is None or recognizer is None:
            print("  Verify failed: OpenCV returned null handles.")
            return False
        print("  OpenCV loaded YuNet + SFace successfully.")
        return True
    except Exception as exc:
        print(f"  Verify failed: {exc}")
        return False


def check_identity_ready() -> bool:
    try:
        from identity import models_ready

        return models_ready()
    except Exception as exc:
        print(f"  identity.models_ready() error: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YuNet + SFace face models for AI Visitor Tracker"
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Bypass HTTP proxy (use when proxy is off or broken)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing models (no download)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()

    if args.no_proxy:
        clear_proxy_env()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        urllib.request.install_opener(opener)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        ok = verify_models() and check_identity_ready()
        print("\nFace models ready." if ok else "\nFace models NOT ready.")
        sys.exit(0 if ok else 1)

    print("AI Visitor Tracker — face model installer")
    print("Models: OpenCV YuNet (detect) + SFace (recognize)\n")

    for name, (url, min_bytes) in FILES.items():
        dest = MODELS_DIR / name
        if dest.is_file() and dest.stat().st_size >= min_bytes and not args.force:
            print(f"  {name} already present ({dest.stat().st_size:,} bytes)")
            continue
        try:
            download(url, dest, min_bytes)
        except Exception as exc:
            print(f"\nFailed: {exc}")
            sys.exit(1)

    print("\nVerifying installation …")
    if not verify_models():
        sys.exit(1)
    if not check_identity_ready():
        print("Warning: files exist but identity.models_ready() is False.")
        sys.exit(1)

    print(f"\nFace models ready in {MODELS_DIR}")
    print("Tracking will use face matching first, body matching only as fallback.")
    print("Run: python run_system.py")


if __name__ == "__main__":
    main()
