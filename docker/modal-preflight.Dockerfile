FROM python:3.12-slim-bookworm AS python312

FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

# Modal's add_python layer supplies Python 3.12. Copy the official CPython
# runtime into the CUDA image so the local test exercises the same minor
# version rather than Ubuntu 22.04's Python 3.10.
COPY --from=python312 /usr/local /usr/local

ENV DEBIAN_FRONTEND=noninteractive \
    CC=gcc \
    CXX=g++ \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && nvcc --version \
    && python --version \
    && gcc --version \
    && g++ --version

RUN python -m pip install --no-cache-dir \
    torch==2.12.0 \
    ninja packaging setuptools wheel \
    && python -m pip install --no-cache-dir --no-build-isolation \
    causal-conv1d==1.6.2.post1 \
    flash-linear-attention==0.5.1 \
    && python -c "import torch; print('torch_cuda=' + str(torch.version.cuda)); import causal_conv1d; print('causal_conv1d=ok'); import fla; print('fla=ok')"
