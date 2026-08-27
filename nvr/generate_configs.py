#!/usr/bin/env python3
"""Generate go2rtc.yaml + nginx snapshot ports from cameras.json (real-time WebRTC)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

NVR_DIR = Path(__file__).resolve().parent
PROJECT_DIR = NVR_DIR.parent
CAMERAS_PATH = PROJECT_DIR / "cameras.json"
GO2RTC_OUT = NVR_DIR / "go2rtc.yaml"
SNAPSHOTS_OUT = NVR_DIR / "nginx-snapshots.conf"
SNAPSHOT_PORT_BASE = 18089
# MSA CCTV already owns :1984 — this project's go2rtc listens here.
GO2RTC_PORT = int(os.environ.get("GO2RTC_PORT", "1985"))
# Outdoor gate cams get ffmpeg deshake (wind / mount shake).
_STABILIZE_DEFAULT = "3,28,29,30"
STABILIZE_CAMERAS = frozenset(
    c.strip()
    for c in os.environ.get("GO2RTC_STABILIZE_CAMERAS", _STABILIZE_DEFAULT).split(",")
    if c.strip()
)

# Prefer a full build with libx264 / h264_nvenc when ffmpeg is not on PATH.
_FFMPEG_CANDIDATES = [
    Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
    if "LOCALAPPDATA" in os.environ
    else None,
    Path(r"C:\pinokio\bin\miniforge\Library\bin\ffmpeg.exe"),
    Path(r"C:\pinokio\api\wan.git\app\ffmpeg_bins\ffmpeg.exe"),
    Path(r"C:\Program Files\TRCCCAP\ffmpeg.exe"),
    Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "ffmpeg" / "bin" / "ffmpeg.exe",
    NVR_DIR / "ffmpeg.exe",
]


def resolve_ffmpeg() -> str:
    env = (os.environ.get("FFMPEG_BIN") or os.environ.get("FFMPEG_PATH") or "").strip()
    if env and Path(env).is_file():
        return str(Path(env).resolve())
    which = shutil.which("ffmpeg")
    if which:
        return str(Path(which).resolve())
    for cand in _FFMPEG_CANDIDATES:
        if cand is None:
            continue
        if cand.is_file():
            return str(cand.resolve())
    raise SystemExit(
        "ffmpeg not found. Install ffmpeg or set FFMPEG_BIN to ffmpeg.exe, then re-run."
    )


def main() -> None:
    cams = json.loads(CAMERAS_PATH.read_text(encoding="utf-8"))
    if not isinstance(cams, list) or not cams:
        raise SystemExit(f"No cameras in {CAMERAS_PATH}")

    ffmpeg_bin = resolve_ffmpeg()
    # go2rtc YAML: quote path if it has spaces
    ffmpeg_yaml = ffmpeg_bin.replace("\\", "/")

    lines = [
        "# Auto-generated from cameras.json — real-time WebRTC (not JPEG)",
        "# Re-run: py -3.12 nvr/generate_configs.py",
        f"# go2rtc :{GO2RTC_PORT} — RTSP → WebRTC/MSE via ffmpeg H.264",
        f"# ffmpeg: {ffmpeg_bin}",
        f"# stabilize cams: {', '.join(sorted(STABILIZE_CAMERAS)) or '(none)'}",
        "",
        "api:",
        # Localhost-only bind (security). origin * is OK — Flask proxies /go2rtc.
        f'  listen: "127.0.0.1:{GO2RTC_PORT}"',
        '  origin: "*"',
        "",
        "webrtc:",
        "  candidates:",
        "    - stun:stun.l.google.com:19302",
        "",
        "ffmpeg:",
        f'  bin: "{ffmpeg_yaml}"',
        "",
        "streams:",
    ]
    for cam in cams:
        cid = str(cam["id"]).strip()
        src = str(cam["source"]).strip()
        # Prefer NVR substream (subtype=1). Fixed even height for libx264.
        sub = src.replace("subtype=0", "subtype=1") if "subtype=0" in src else src
        stabilize = cid in STABILIZE_CAMERAS
        # Outdoor gates: deshake reduces wind/mount shake in the 2×2 live view.
        if stabilize:
            stream_url = (
                f"ffmpeg:{sub}#video=h264#width=960#height=540"
                f"#raw=-vf deshake=rx=64:ry=64:edge=mirror"
            )
        else:
            stream_url = f"ffmpeg:{sub}#video=h264#width=960#height=540"
        lines.append(f"  cam{cid}:")
        lines.append(f"    - {stream_url}")
        if sub != src and not stabilize:
            lines.append(f"    - ffmpeg:{src}#video=h264#width=960#height=540")

    GO2RTC_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {GO2RTC_OUT} ({len(cams)} realtime streams @960x540 H.264)")
    print(f"ffmpeg: {ffmpeg_bin}")
    print(f"go2rtc listen: 127.0.0.1:{GO2RTC_PORT}")

    snap: list[str] = [
        "# JPEG snapshot proxies only (fallback stills — not used for live video)",
        f"# Ports {SNAPSHOT_PORT_BASE}–{SNAPSHOT_PORT_BASE + len(cams) - 1}",
        "",
    ]
    for i, cam in enumerate(cams):
        port = SNAPSHOT_PORT_BASE + i
        cid = str(cam["id"]).strip()
        snap.extend(
            [
                "server {",
                f"    listen 127.0.0.1:{port};",
                "    location / {",
                f"        proxy_pass http://127.0.0.1:{GO2RTC_PORT}/api/frame.jpeg?src=cam{cid};",
                "        proxy_http_version 1.1;",
                '        add_header Cache-Control "no-store";',
                "    }",
                "}",
                "",
            ]
        )
    SNAPSHOTS_OUT.write_text("\n".join(snap), encoding="utf-8")
    print(f"Wrote {SNAPSHOTS_OUT}")


if __name__ == "__main__":
    main()
