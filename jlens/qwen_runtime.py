"""Runtime controls for optional Qwen3.5 fused kernels."""
from __future__ import annotations

import importlib


def configure_qwen_kernels(mode: str = "auto") -> str:
    """Select Qwen3.5 kernel mode before constructing a model.

    ``auto`` leaves Transformers' default selection unchanged, ``on`` requires
    both optional packages, and ``off`` forces the PyTorch fallback. This is a
    version-specific compatibility shim for the installed Transformers Qwen3.5
    implementation; it is recorded by experiment runners for reproducibility.
    """
    if mode not in {"auto", "on", "off"}:
        raise ValueError(f"unknown Qwen kernel mode: {mode!r}")
    module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    if mode == "auto":
        return "auto"
    if mode == "on":
        missing = []
        try:
            import causal_conv1d  # noqa: F401
        except ImportError:
            missing.append("causal-conv1d")
        try:
            import fla  # noqa: F401
        except ImportError:
            missing.append("flash-linear-attention")
        if missing:
            raise RuntimeError("Qwen fast kernels unavailable: " + ", ".join(missing))
        return "on"

    # Transformers binds these optional functions at module import time.
    # Replacing them before AutoModel construction makes each layer select its
    # ordinary torch implementation without uninstalling anything.
    module.causal_conv1d_fn = None
    module.causal_conv1d_update = None
    module.chunk_gated_delta_rule = None
    module.fused_recurrent_gated_delta_rule = None
    module.FusedRMSNormGated = None
    return "off"
