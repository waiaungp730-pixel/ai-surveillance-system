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
# MSA_CCTV_100 owns :1984, MSA_CCTV owns :1985 — this project uses :1986.
GO2RTC_PORT = int(os.environ.get("GO2RTC_PORT", "1986"))
GO2RTC_RTSP_PORT = int(os.environ.get("GO2RTC_RTSP_PORT", "8558"))
GO2RTC_WEBRTC_PORT = int(os.environ.get("GO2RTC_WEBRTC_PORT", "8559"))
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
    """Return an ffmpeg path safe for go2rtc exec (no unquoted spaces)."""
    env = (os.environ.get("FFMPEG_BIN") or os.environ.get("FFMPEG_PATH") or "").strip()
    if env and Path(env).is_file():
        resolved = str(Path(env).resolve())
        # go2rtc exec does not quote bin paths — spaces break "C:/Program Files/...".
        if " " not in resolved:
            return resolved
    which = shutil.which("ffmpeg")
    if which:
        # Prefer bare name on PATH so go2rtc never sees "Program Files".
        return "ffmpeg"
    for cand in _FFMPEG_CANDIDATES:
        if cand is None:
            continue
        if cand.is_file() and " " not in str(cand):
            return str(cand.resolve())
    # Last resort: copy/link into nvr/ (no spaces).
    nvr_ffmpeg = NVR_DIR / "ffmpeg.exe"
    if nvr_ffmpeg.is_file():
        return str(nvr_ffmpeg.resolve())
    for cand in _FFMPEG_CANDIDATES:
        if cand is None or not cand.is_file():
            continue
        try:
            shutil.copy2(cand, nvr_ffmpeg)
            return str(nvr_ffmpeg.resolve())
        except OSError:
            continue
    raise SystemExit(
        "ffmpeg not found. Install ffmpeg on PATH (no spaces in path), "
        "or set FFMPEG_BIN, then re-run."
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
        f"# rtsp :{GO2RTC_RTSP_PORT}  webrtc :{GO2RTC_WEBRTC_PORT}",
        f"# ffmpeg: {ffmpeg_bin}",
        f"# stabilize cams: {', '.join(sorted(STABILIZE_CAMERAS)) or '(none)'}",
        "",
        "api:",
        # Prefer all-interfaces so Flask proxy + LAN browsers can reach it.
        # Port must not collide with MSA (:1984 / :1985).
        f'  listen: ":{GO2RTC_PORT}"',
        '  origin: "*"',
        "",
        # Required: ffmpeg transcode uses {output} → local RTSP (else "rtsp module disabled").
        # Bind all interfaces like MSA configs — host-only listen can leave rtsp.Port empty in some builds.
        "rtsp:",
        f'  listen: ":{GO2RTC_RTSP_PORT}"',
        "",
        "webrtc:",
        f'  listen: ":{GO2RTC_WEBRTC_PORT}"',
        "  candidates:",
        f"    - 127.0.0.1:{GO2RTC_WEBRTC_PORT}",
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
        # Prefer main stream (subtype=0) first — some NVR channels have broken subtype=1.
        sub = src.replace("subtype=0", "subtype=1") if "subtype=0" in src else src
        stabilize = cid in STABILIZE_CAMERAS
        # Outdoor gates: deshake reduces wind/mount shake in the 2×2 live view.
        # H.265/any → H.264 @ 640px (NVR 2×2 WebRTC budget).
        # #rtsp_transport=tcp matches working MSA go2rtc configs.
        if stabilize:
            main_url = (
                f"ffmpeg:{src}#rtsp_transport=tcp#video=h264#width=640#height=360"
                f"#raw=-vf deshake=rx=64:ry=64:edge=mirror"
            )
            sub_url = (
                f"ffmpeg:{sub}#rtsp_transport=tcp#video=h264#width=640#height=360"
                f"#raw=-vf deshake=rx=64:ry=64:edge=mirror"
            )
        else:
            main_url = f"ffmpeg:{src}#rtsp_transport=tcp#video=h264#width=640#height=360"
            sub_url = f"ffmpeg:{sub}#rtsp_transport=tcp#video=h264#width=640#height=360"
        lines.append(f"  cam{cid}:")
        lines.append(f"    - {main_url}")
        lines.append(f"    - {src}#rtsp_transport=tcp")
        if sub != src:
            lines.append(f"    - {sub_url}")
            lines.append(f"    - {sub}#rtsp_transport=tcp")

    # Higher-res restreams for ANPR plate OCR (gate / configured plate cameras).
    plate_ids = {
        c.strip()
        for c in os.environ.get("LICENSE_PLATE_CAMERAS", "1,3,28,29,30").split(",")
        if c.strip()
    }
    plate_count = 0
    for cam in cams:
        cid = str(cam["id"]).strip()
        if cid not in plate_ids:
            continue
        src = str(cam["source"]).strip()
        sub = src.replace("subtype=0", "subtype=1") if "subtype=0" in src else src
        lines.append(f"  plate_cam{cid}:")
        lines.append(
            f"    - ffmpeg:{src}#rtsp_transport=tcp#video=h264#width=1280#height=720"
        )
        lines.append(f"    - {src}#rtsp_transport=tcp")
        if sub != src:
            lines.append(
                f"    - ffmpeg:{sub}#rtsp_transport=tcp#video=h264#width=1280#height=720"
            )
            lines.append(f"    - {sub}#rtsp_transport=tcp")
        plate_count += 1

    GO2RTC_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {GO2RTC_OUT} ({len(cams)} live @640x360 + {plate_count} plate @1280x720)")
    print(f"ffmpeg: {ffmpeg_bin}")
    print(f"go2rtc listen: :{GO2RTC_PORT}  rtsp:{GO2RTC_RTSP_PORT}  webrtc:{GO2RTC_WEBRTC_PORT}")

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
