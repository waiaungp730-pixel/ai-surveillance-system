# Orion Tactical — Pure GPU (NVIDIA CUDA 12.8 + cuDNN)
# Build:  docker build -t ai-tracker .
# Run:    docker run --gpus all -p 5000:5000 -v ./data:/app/data ai-tracker
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    DASHBOARD_HOST=127.0.0.1 \
    DASHBOARD_PORT=5000 \
    DEVICE=cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NO_PROXY="*"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

COPY requirements.txt .

# PyTorch cu128 first (RTX 50-series / sm_120), then GPU ONNX Runtime, then app deps
RUN python3.12 -m pip install --no-cache-dir --upgrade pip \
    && python3.12 -m pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python3.12 -m pip install --no-cache-dir onnxruntime-gpu>=1.18.0 onnx>=1.16.0 \
    && python3.12 -m pip install --no-cache-dir -r requirements.txt \
    && python3.12 -m pip install --no-cache-dir insightface --no-deps

COPY . .

RUN chmod +x docker-entrypoint.sh \
    && python3.12 download_model.py --no-proxy --size p2 \
    && python3.12 download_face_models.py --no-proxy \
    && python3.12 download_face_attributes.py --no-proxy \
    && python3.12 download_body_model.py --no-proxy

EXPOSE 5000

VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3.12 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/api/summary', timeout=3)" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
