"""
Download GFPGAN / CodeFormer weights into models/.

Usage:
  python download_face_restoration.py
  python download_face_restoration.py --no-proxy
  python download_face_restoration.py --backend gfpgan
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
)
CODEFORMER_URL = (
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
)

GFPGAN_NAME = "GFPGANv1.4.pth"
CODEFORMER_NAME = "codeformer.pth"
MIN_GFPGAN_BYTES = 300_000_000
MIN_CODEFORMER_BYTES = 8_000_000


def models_dir() -> Path:
    return Path(__file__).parent / "models"


def proxy_port_open(host: str = "127.0.0.1", port: int = 60705) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def clear_proxy_env() -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"


def download_urllib(url: str, dest: Path, min_bytes: int) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    urllib.request.install_opener(opener)
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, dest)
    return dest.is_file() and dest.stat().st_size >= min_bytes


def download_curl(url: str, dest: Path, min_bytes: int) -> bool:
    cmd = ["curl", "--ssl-no-revoke", "-L", "--fail", "-o", str(dest), url]
    print(f"  curl: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr or result.stdout)
        return False
    return dest.is_file() and dest.stat().st_size >= min_bytes


def fetch(url: str, dest: Path, min_bytes: int, use_proxy: bool) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        print(f"  Already present: {dest.name} ({dest.stat().st_size / 1_000_000:.1f} MB)")
        return True

    if not use_proxy:
        clear_proxy_env()

    if download_curl(url, dest, min_bytes):
        return True
    return download_urllib(url, dest, min_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download face restoration model weights.")
    parser.add_argument(
        "--backend",
        choices=("gfpgan", "codeformer", "both"),
        default="both",
        help="Which weights to download (default: both)",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Bypass local HTTP proxy (recommended on Windows)",
    )
    args = parser.parse_args()

    use_proxy = not args.no_proxy and proxy_port_open()
    if args.no_proxy:
        clear_proxy_env()
        use_proxy = False

    models = models_dir()
    ok = True

    if args.backend in ("gfpgan", "both"):
        dest = models / GFPGAN_NAME
        print(f"GFPGAN -> {dest}")
        if not fetch(GFPGAN_URL, dest, MIN_GFPGAN_BYTES, use_proxy):
            print("  FAILED: GFPGAN download")
            ok = False

    if args.backend in ("codeformer", "both"):
        dest = models / CODEFORMER_NAME
        print(f"CodeFormer -> {dest}")
        if not fetch(CODEFORMER_URL, dest, MIN_CODEFORMER_BYTES, use_proxy):
            print("  FAILED: CodeFormer download")
            ok = False

    if ok:
        print("\nFace restoration weights ready.")
        print("  Install deps: pip install gfpgan basicsr facexlib")
        return 0

    print("\nDownload incomplete — retry with --no-proxy", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
