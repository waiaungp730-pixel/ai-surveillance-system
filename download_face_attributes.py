"""
Download OpenCV DNN age/gender models for face attribute recognition.

  python download_face_attributes.py --no-proxy
  python download_face_attributes.py --no-proxy --verify
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

MODELS_DIR = Path(__file__).parent / "models"

FILES: Dict[str, Tuple[str, int]] = {
    "deploy_age.prototxt": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/"
        "age_deploy.prototxt",
        500,
    ),
    "age_net.caffemodel": (
        "https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1",
        1_000_000,
    ),
    "deploy_gender.prototxt": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/"
        "gender_deploy.prototxt",
        500,
    ),
    "gender_net.caffemodel": (
        "https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=1",
        1_000_000,
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
                    f"File too small ({size:,} bytes; need >= {min_bytes:,})"
                )
            print(f"    OK ({size:,} bytes)")
            return
        except Exception as exc:
            last_err = exc
            print(f"    Attempt {attempt}/3 failed: {exc}")
    raise RuntimeError(f"Could not download {dest.name}: {last_err}")


def verify_models() -> bool:
    try:
        from face_attributes import (
            active_backend_name,
            attribute_models_ready,
            predict_face_attributes_verbose,
        )

        import numpy as np

        if not attribute_models_ready():
            print("  Verify failed: no attribute backend available.")
            print("  Install OpenCV models or: pip install deepface")
            return False
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        pred = predict_face_attributes_verbose(dummy)
        print(
            f"  Backend: {active_backend_name()} — "
            f"sample: {pred.gender}, age={pred.age}, "
            f"conf={pred.gender_conf:.2f}/{pred.age_conf:.2f}"
        )
        return True
    except Exception as exc:
        print(f"  Verify failed: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download OpenCV age/gender DNN models for face attributes"
    )
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.no_proxy:
        clear_proxy_env()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        urllib.request.install_opener(opener)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("AI Visitor Tracker — age/gender model installer")
    print(f"Target folder: {MODELS_DIR}\n")

    for name, (url, min_bytes) in FILES.items():
        dest = MODELS_DIR / name
        if dest.is_file() and dest.stat().st_size >= min_bytes:
            print(f"  {name} already present ({dest.stat().st_size:,} bytes)")
            continue
        download(url, dest, min_bytes)

    print("\nDone. Face attributes will run after YuNet captures a face.")
    if args.verify:
        print("\nVerifying …")
        return 0 if verify_models() else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
