"""
Download Silent-Face-Anti-Spoofing ONNX weights to models/anti_spoof.onnx.

Converted MiniFASNetV2 weights (80×80) from the yakhyo/face-anti-spoofing release,
derived from minivision-ai/Silent-Face-Anti-Spoofing.

  python create_download_anti_spoof.py [--no-proxy]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_PATH = MODELS_DIR / "anti_spoof.onnx"

# MiniFASNetV2 80×80 ONNX (Silent-Face-Anti-Spoofing derivative)
DOWNLOAD_URL = (
    "https://github.com/yakhyo/face-anti-spoofing/releases/download/"
    "weights/MiniFASNetV2.onnx"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(no_proxy: bool = False) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.is_file() and OUTPUT_PATH.stat().st_size > 100_000:
        print(f"Already present: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size:,} bytes)")
        print(f"SHA-256: {_sha256(OUTPUT_PATH)}")
        return OUTPUT_PATH

    if no_proxy:
        urllib.request.install_opener(
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
        )
    else:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            val = os.environ.get(key)
            if val:
                print(f"Using proxy from {key}={val}")

    print(f"Downloading {DOWNLOAD_URL}")
    print(f"  → {OUTPUT_PATH}")
    tmp = OUTPUT_PATH.with_suffix(".onnx.part")
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, tmp)
        tmp.replace(OUTPUT_PATH)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {exc}") from exc

    print(f"Saved {OUTPUT_PATH.stat().st_size:,} bytes")
    print(f"SHA-256: {_sha256(OUTPUT_PATH)}")
    return OUTPUT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Download anti-spoof ONNX weights")
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Ignore HTTP(S)_PROXY environment variables",
    )
    args = parser.parse_args()
    try:
        path = download(no_proxy=args.no_proxy)
        print(f"Anti-spoof model ready at {path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
