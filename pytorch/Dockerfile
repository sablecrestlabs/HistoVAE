ARG PYTORCH_VERSION=2.10.0
ARG CUDA_VERSION=13.0

FROM ubuntu:22.04

ARG PYTORCH_VERSION
ARG CUDA_VERSION

ENV CUDA_VERSION=${CUDA_VERSION}
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace/pytorch

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		ca-certificates \
		python3 \
		python3-pip \
		python3-venv \
		libopenslide0 \
		openslide-tools \
	&& rm -rf /var/lib/apt/lists/*

COPY pytorch/requirements.txt /tmp/requirements.txt
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
	grep -v -E '^(torch|torchvision|torchaudio)\b' /tmp/requirements.txt > /tmp/requirements.no_torch.txt; \
	python3 -m pip install --no-cache-dir -r /tmp/requirements.no_torch.txt

COPY pytorch/src /workspace/pytorch/src
COPY pretrained /workspace/pretrained

ENTRYPOINT ["python3", "-m", "src.cli"]