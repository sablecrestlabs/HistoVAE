ARG PYTORCH_VERSION=2.10.0
ARG CUDA_VERSION=13.0

FROM ubuntu:22.04

# Re-declare build args after FROM so they can be used in later instructions.
ARG PYTORCH_VERSION
ARG CUDA_VERSION

# Expose CUDA version inside the container as an env var.
ENV CUDA_VERSION=${CUDA_VERSION}
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# System dependencies
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		ca-certificates \
		python3 \
		python3-pip \
		python3-venv \
		libopenslide0 \
		openslide-tools \
	&& rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN set -eux; \
	python3 -m pip install --upgrade pip; \
	cu_tag="$(echo "${CUDA_VERSION}" | tr -d '.')"; \
	python3 -m pip install --no-cache-dir \
		--index-url "https://download.pytorch.org/whl/cu${cu_tag}" \
		"torch==${PYTORCH_VERSION}" torchvision torchaudio \
	|| python3 -m pip install --no-cache-dir \
		--index-url "https://download.pytorch.org/whl/cu${cu_tag}" \
		torch torchvision torchaudio \
	|| python3 -m pip install --no-cache-dir "torch==${PYTORCH_VERSION}" torchvision torchaudio \
	|| python3 -m pip install --no-cache-dir torch torchvision torchaudio; \
	# Install the remaining dependencies (skip torch/torchvision if present).
	grep -v -E '^(torch|torchvision|torchaudio)\b' /workspace/requirements.txt > /tmp/requirements.no_torch.txt; \
	python3 -m pip install --no-cache-dir -r /tmp/requirements.no_torch.txt

# Project files
COPY vae.py /workspace/vae.py
COPY train_vae.sh /workspace/train_vae.sh
COPY tensorboard.sh /workspace/tensorboard.sh
COPY pretrained/ /workspace/pretrained/

# Default to running training; pass args directly to the container.
ENTRYPOINT ["python3", "vae.py"]

