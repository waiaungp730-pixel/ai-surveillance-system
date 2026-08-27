"""
Download YOLO weights into this project folder.

Default: yolov8l.pt (YOLOv8 Large — command-center recall on CUDA).

Usage:
  py -3.12 download_model.py --no-proxy
  py -3.12 download_model.py --no-proxy --size l    # yolov8l.pt (default)
  py -3.12 download_model.py --no-proxy --size p2   # legacy yolov8s-p2.pt
  py -3.12 download_model.py --no-proxy --size s    # base yolov8s.pt only
"""

import argparse
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

MODEL_CATALOG: Dict[str, Dict[str, object]] = {
    "n": {
        "name": "yolov8n.pt",
        "urls": [
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt",
        ],
        "min_bytes": 5_000_000,
    },
    "s": {
        "name": "yolov8s.pt",
        "urls": [
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8s.pt",
        ],
        "min_bytes": 18_000_000,
    },
    "l": {
        "name": "yolov8l.pt",
        "urls": [
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8l.pt",
        ],
        "min_bytes": 70_000_000,
    },
    "p2": {
        "name": "yolov8s-p2.pt",
        "base": "yolov8s.pt",
        "min_bytes": 15_000_000,
        "urls": [
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8s.pt",
        ],
        "base_min_bytes": 18_000_000,
    },
}
DEFAULT_SIZE = "l"


def target_path(size: str) -> Path:
    info = MODEL_CATALOG[size]
    return Path(__file__).parent / str(info["name"])


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


def get_proxies_from_env() -> Optional[dict]:
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(key)
        if value:
            return {"http": value, "https": value}
    return None


def download_with_urllib(url: str, dest: Path, use_proxy: bool) -> bool:
    if not use_proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        urllib.request.install_opener(opener)
    print(f"  urllib: {url}")
    urllib.request.urlretrieve(url, dest)
    return dest.exists()


def download_with_requests(url: str, dest: Path, use_proxy: bool) -> bool:
    try:
        import requests
    except ImportError:
        return False

    session = requests.Session()
    session.trust_env = use_proxy
    proxies = get_proxies_from_env() if use_proxy else None

    print(f"  requests: {url}")
    response = session.get(url, stream=True, proxies=proxies, timeout=180)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)
    return dest.exists()


def download_with_curl(url: str, dest: Path) -> bool:
    cmd = ["curl", "--ssl-no-revoke", "-L", "--fail", "-o", str(dest), url]
    print(f"  curl: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr or result.stdout)
        return False
    return dest.exists()


def try_download(dest: Path, urls: List[str], min_bytes: int, use_proxy: bool) -> bool:
    methods = (download_with_requests, download_with_urllib, download_with_curl)
    for url in urls:
        if dest.exists():
            dest.unlink()
        for method in methods:
            try:
                if method(url, dest, use_proxy) and dest.stat().st_size >= min_bytes:
                    return True
            except Exception as exc:
                print(f"    failed: {exc}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YOLO detection weights")
    parser.add_argument(
        "--size",
        choices=sorted(MODEL_CATALOG),
        default=DEFAULT_SIZE,
        help="Model: l=YOLOv8 Large (default), p2=legacy P2 head, s=small, n=nano",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Ignore Windows/env proxy (use if proxy app is off)",
    )
    args = parser.parse_args()

    info = MODEL_CATALOG[args.size]
    dest = target_path(args.size)
    min_bytes = int(info["min_bytes"])

    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"Already present: {dest} ({dest.stat().st_size:,} bytes)")
        return

    if dest.exists():
        dest.unlink()

    use_proxy = not args.no_proxy
    if use_proxy and proxy_port_open():
        print("Local proxy detected on 127.0.0.1:60705 — using it.")
        os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:60705")
        os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:60705")
    elif use_proxy:
        print(
            "WARNING: Windows proxy is set to 127.0.0.1:60705 but nothing is listening.\n"
            "  Try: python download_model.py --no-proxy\n"
        )

    if args.no_proxy:
        clear_proxy_env()
        print("Downloading without proxy...")

    if args.size == "p2":
        base_dest = Path(__file__).parent / str(info["base"])
        base_min = int(info["base_min_bytes"])
        urls = list(info["urls"])
        if not base_dest.is_file() or base_dest.stat().st_size < base_min:
            print(f"Downloading base weights: {info['base']}")
            if not try_download(
                base_dest, urls, base_min, use_proxy=use_proxy and not args.no_proxy
            ):
                print("\nBase weight download failed — cannot build P2 model.")
                sys.exit(1)
        from yolo_loader import build_p2_weights

        build_p2_weights(Path(__file__).parent)
        if dest.is_file() and dest.stat().st_size >= min_bytes:
            print(f"\nSuccess: {dest} ({dest.stat().st_size:,} bytes)")
            print("Now run: py -3.12 run_system.py --device cuda")
            return
        sys.exit(1)

    urls = list(info["urls"])
    print(f"Model: {info['name']}")
    print(f"Saving to: {dest.resolve()}")
    if try_download(dest, urls, min_bytes, use_proxy=use_proxy and not args.no_proxy):
        print(f"\nSuccess: {dest} ({dest.stat().st_size:,} bytes)")
        print("Now run: py -3.12 run_system.py --device cuda")
        return

    print(
        "\nAutomatic download failed.\n"
        "Manual download:\n"
        f"  1. Open: {urls[0]}\n"
        f"  2. Save as: {dest.resolve()}\n"
        f"  3. File must be at least {min_bytes // 1_000_000} MB.\n"
        "  4. Run: py -3.12 run_system.py --device cuda\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
