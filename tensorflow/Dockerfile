ARG NVIDIA_TENSORFLOW_TAG=25.02-tf2-py3

FROM nvcr.io/nvidia/tensorflow:${NVIDIA_TENSORFLOW_TAG}

ARG NVIDIA_TENSORFLOW_TAG

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false
ENV XLA_FLAGS=--xla_gpu_enable_triton_gemm=false

WORKDIR /workspace/tensorflow

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		libopenslide0 \
		openslide-tools \
	&& rm -rf /var/lib/apt/lists/*

COPY tensorflow/requirements.txt /tmp/requirements.txt
RUN set -eux; \
	python3 -m pip install --upgrade pip; \
	grep -v -E '^tensorflow\b' /tmp/requirements.txt > /tmp/requirements.no_tensorflow.txt; \
	python3 -m pip install --no-cache-dir -r /tmp/requirements.no_tensorflow.txt

COPY tensorflow/src /workspace/tensorflow/src

ENTRYPOINT ["python3", "-m", "src.cli"]