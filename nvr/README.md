# Real-time Live Cameras (go2rtc WebRTC)

Live Cameras **2×2 does not use 480×270 JPEG**. It uses **real WebRTC video** from go2rtc.

| Port | Service |
|------|---------|
| 5000 | AI Tracker dashboard |
| **1986** | **go2rtc** real-time WebRTC/MSE (this project) |
| 1985 | MSA_CCTV go2rtc (do not steal) |
| 1984 | MSA_CCTV_100 go2rtc (do not steal) |
| **8088** | Nginx reverse proxy |
| 18089+ | JPEG snapshots only (not live video) |

## Requirements

- **ffmpeg** on PATH  
- **go2rtc** (`nvr/go2rtc.exe` or PATH)  
- nginx optional  

## Start

```bat
py -3.12 run_system.py --device cuda
nvr\start_nvr.bat
```

Open **http://127.0.0.1:8088** (or `:5000` if no nginx). Live Cameras status should read **GO2RTC · WEBRTC**.

If go2rtc is down, tiles fall back to snapshot JPEG (`/go2rtc/api/frame.jpeg` or ports 18089+) — not the tracker disk JPG poll.
