"""
Download OSNet x1.0 body Re-ID weights into models/.

Usage:
  python download_body_model.py
  python download_body_model.py --no-proxy
"""

import argparse
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional

WEIGHTS_NAME = "osnet_x1_0_reid.pth"
WEIGHTS_URLS = [
    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
]
MIN_BYTES = 16_000_000


def target_path() -> Path:
    return Path(__file__).parent / "models" / WEIGHTS_NAME


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


def download_urllib(url: str, dest: Path, use_proxy: bool) -> bool:
    if not use_proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        urllib.request.install_opener(opener)
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, dest)
    return dest.exists() and dest.stat().st_size >= MIN_BYTES


def download_curl(url: str, dest: Path) -> bool:
    cmd = ["curl", "--ssl-no-revoke", "-L", "--fail", "-o", str(dest), url]
    print(f"  curl: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr or result.stdout)
        return False
    return dest.exists() and dest.stat().st_size >= MIN_BYTES


def verify_osnet(path: Path) -> bool:
    try:
        import torch
        from torchreid.reid.models import build_model

        checkpoint = torch.load(str(path), map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        num_classes = 751
        for key in ("classifier.weight", "module.classifier.weight"):
            if key in state:
                num_classes = int(state[key].shape[0])
                break

        model = build_model(
            name="osnet_x1_0",
            num_classes=num_classes,
            loss="softmax",
            pretrained=False,
            use_gpu=False,
        )
        model.load_state_dict(state, strict=False)
        print("  OSNet weights verified.")
        return True
    except Exception as exc:
        print(f"  Verify failed: {exc}")
        return False


def _replace_file(src: Path, dest: Path) -> None:
    """Move completed download into place, replacing any partial file."""
    try:
        if dest.exists():
            dest.unlink()
    except PermissionError:
        backup = dest.with_suffix(dest.suffix + ".old")
        try:
            if backup.exists():
                backup.unlink()
            dest.replace(backup)
        except OSError as exc:
            raise PermissionError(
                f"Cannot replace {dest.name} — stop run_system.py / tracker first, "
                f"then delete models/{dest.name} manually."
            ) from exc
    src.replace(dest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OSNet x1.0 body Re-ID weights (Market-1501)"
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Direct download (use if proxy app is off)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing weights",
    )
    args = parser.parse_args()

    dest = target_path()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if args.verify:
        if not dest.is_file():
            print(f"Missing: {dest}")
            sys.exit(1)
        sys.exit(0 if verify_osnet(dest) else 1)

    if dest.exists() and dest.stat().st_size >= MIN_BYTES:
        print(f"Already present: {dest} ({dest.stat().st_size:,} bytes)")
        verify_osnet(dest)
        return

    partial = dest.with_suffix(dest.suffix + ".part")
    if partial.exists():
        try:
            partial.unlink()
        except PermissionError:
            print(
                f"Another download may still be running ({partial.name} is locked).\n"
                "  Wait for it to finish, or stop run_system.py and try again."
            )
            sys.exit(1)

    if dest.exists() and dest.stat().st_size < MIN_BYTES:
        print(f"Removing incomplete file ({dest.stat().st_size:,} bytes)…")
        try:
            dest.unlink()
        except PermissionError:
            print(
                f"Cannot delete partial {dest.name} — it is in use.\n"
                "  Stop run_system.py (Ctrl+C), then run this script again."
            )
            sys.exit(1)

    use_proxy = not args.no_proxy
    if use_proxy and proxy_port_open():
        print("Local proxy on 127.0.0.1:60705 — using it.")
        os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:60705")
        os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:60705")
    elif use_proxy:
        print(
            "WARNING: Proxy may be set but nothing is listening on 127.0.0.1:60705.\n"
            "  Try: python download_body_model.py --no-proxy\n"
        )

    if args.no_proxy:
        clear_proxy_env()
        print("Downloading without proxy...")

    print(f"Saving to: {dest.resolve()}")
    ok = False
    for url in WEIGHTS_URLS:
        for method in (download_urllib, download_curl):
            try:
                if method is download_curl:
                    ok = method(url, partial)
                else:
                    ok = method(url, partial, use_proxy=use_proxy and not args.no_proxy)
                if ok:
                    break
            except Exception as exc:
                print(f"  failed: {exc}")
        if ok:
            break

    if ok:
        try:
            _replace_file(partial, dest)
        except PermissionError as exc:
            print(f"\nDownload complete but could not install: {exc}")
            print(f"  Completed file kept at: {partial.resolve()}")
            sys.exit(1)

    if ok and verify_osnet(dest):
        print(f"\nSuccess: {dest} ({dest.stat().st_size:,} bytes)")
        print("Install deps if needed: pip install torchreid gdown")
        print("Now run: python run_system.py")
        return

    print(
        "\nDownload failed.\n"
        f"  1. Open in a browser:\n"
        f"     {WEIGHTS_URLS[0]}\n"
        f"  2. Save as: {dest.resolve()}\n"
        "  3. File should be about 17 MB.\n"
        "  4. pip install torchreid gdown\n"
        "  5. Run: python run_system.py\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
