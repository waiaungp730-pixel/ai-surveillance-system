# AI Surveillance System

Dockerized CCTV person tracking with face recognition, visitor logging, and a live web dashboard.

YOLOv8 + ByteTrack detect and track people across RTSP streams. RetinaFace / YuNet gate frontal faces before crops are saved. Matches are logged to SQLite and surfaced on the dashboard at **http://localhost:5000**.

---

## Prerequisites

| Tool | Notes |
|------|--------|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Required for containerized deployment |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | Optional — enables GPU inference (`--gpus all`) |
| Git | To clone this repository |

---

## Quick Start (Docker)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Configure cameras

Copy the example config and edit stream URLs (do **not** commit credentials):

```bash
cp cameras.json data/cameras.json
```

Or mount your own `cameras.json` into the persistent data volume:

```json
[
  {
    "id": "cam-01",
    "location": "Entrance",
    "source": "rtsp://user:pass@192.168.1.100:554/stream1"
  }
]
```

Optional: create a `.env` file for Telegram alerts and other secrets (see `.env.example` if provided).

### 3. Build the image

First build downloads ML weights (YOLO, YuNet, SFace, etc.) and may take **15–30 minutes**:

```bash
docker build -t ai-surveillance-system .
```

### 4. Run on port 5000

```bash
mkdir -p data/captures data/uploads

docker run -d \
  --name guard_up_app \
  -p 5000:5000 \
  -v "$(pwd)/data:/app/data" \
  ai-surveillance-system
```

**With NVIDIA GPU:**

```bash
docker run -d \
  --name guard_up_app \
  --gpus all \
  -p 5000:5000 \
  -v "$(pwd)/data:/app/data" \
  -e DEVICE=cuda \
  ai-surveillance-system
```

### 5. Open the dashboard

| URL | Description |
|-----|-------------|
| http://localhost:5000 | Web dashboard (analytics, search, live stats) |

### 6. Useful commands

```bash
# Follow container logs
docker logs -f guard_up_app

# Stop / remove
docker stop guard_up_app
docker rm guard_up_app

# Rebuild after code changes
docker build -t ai-surveillance-system .
docker rm -f guard_up_app
docker run -d --name guard_up_app -p 5000:5000 -v "$(pwd)/data:/app/data" ai-surveillance-system
```

---

## Persistent data

The `./data` volume stores everything that must survive container restarts:

| Path | Contents |
|------|----------|
| `data/visitor_tracking.db` | Visit / match log (SQLite) |
| `data/face_gallery.db` | Cross-camera face gallery |
| `data/captures/` | Face crop thumbnails |
| `data/uploads/` | User-uploaded search images |
| `data/cameras.json` | Optional camera override |

These paths are excluded from Git via `.gitignore`.

---

## Local development (without Docker)

Requires **Python 3.12+**, CUDA-capable PyTorch for GPU, and FFmpeg.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install onnxruntime-gpu      # or onnxruntime for CPU-only

python download_model.py --no-proxy
python download_face_models.py --no-proxy

python run_system.py --device cuda
```

Dashboard: http://127.0.0.1:5000

---

## Architecture overview

```
RTSP cameras ──► main.py (YOLO + ByteTrack + face pipeline)
                      │
                      ├── SQLite (visitor_tracking.db)
                      ├── captures/ (face thumbnails)
                      └── dashboard.py :5000 (Flask UI)
```

`run_system.py` orchestrates one tracker process per camera (default) plus the dashboard.

---

## Security notes

- Never commit `.env`, `cameras.json` with live credentials, or `data/`.
- Rotate RTSP passwords if they were ever committed to history.
- Bind the dashboard to localhost in production or place it behind a reverse proxy with authentication.

---

## License

Add your license here.
