# Maintained NVIDIA PyTorch runtime; no bespoke CUDA/compiler stack.
FROM nvcr.io/nvidia/pytorch:25.11-py3

RUN pip install --no-cache-dir \
    transformers \
    tuned-lens==0.2.0 \
    datasets \
    accelerate \
    huggingface-hub \
    bitsandbytes==0.49.2 \
    flash-linear-attention==0.5.1 \
    packaging \
    ninja \
    causal-conv1d==1.6.2.post1 --no-build-isolation

WORKDIR /workspace
