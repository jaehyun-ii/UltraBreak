# UltraBreak — GPU execution image (target: H100 / Hopper, CUDA 12.x)
#
# Base already ships torch 2.5.1 + torchvision 0.20.1 built for CUDA 12.1,
# which match requirements.txt exactly and run on H100 (sm_90).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Persist HF model weights (Qwen3-VL-8B ~16GB) on a mounted volume
    HF_HOME=/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Minimal runtime libs: git (some HF repos), libGL/glib for Pillow image ops
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
# torch/torchvision/torchaudio are already satisfied by the base image, so pip
# skips them and only adds transformers (>=4.57 for Qwen3-VL), qwen-vl-utils, etc.
COPY requirements.txt .
RUN pip install --no-cache-dir hf_transfer && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (for standalone use; docker-compose bind-mounts
# the repo over this for live edits).
COPY . .

# No fixed ENTRYPOINT — pick a script at run time, e.g.:
#   docker compose run --rm ultrabreak python optimisation/optimise.py
#   docker compose run --rm ultrabreak python evaluation/attack.py
CMD ["python", "optimisation/optimise.py"]
