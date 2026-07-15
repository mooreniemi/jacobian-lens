from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "modal_fit_tuned_lens.py"


def load_guard_module():
    spec = importlib.util.spec_from_file_location("modal_fit_guard", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_modal_script_is_parseable():
    ast.parse(SCRIPT.read_text())


def test_worst_case_cost_has_margin_under_cap():
    module = load_guard_module()
    estimate = module.worst_case_cost()
    assert estimate < module.MAX_BUDGET_USD - module.SAFETY_MARGIN_USD


def test_budget_guard_requires_exact_explicit_confirmation():
    module = load_guard_module()
    with pytest.raises(ValueError, match="confirm-budget"):
        module.validate_budget(24)
    module.validate_budget(5)


def test_modal_function_has_single_container_and_no_retries():
    source = SCRIPT.read_text()
    assert "retries=0" in source
    assert "max_containers=1" in source
    assert 'gpu="A100-80GB"' in source or "gpu=GPU" in source
    assert 'HF_SECRET_NAME = "huggingface-secret"' in source
    assert "secrets=[hf_secret]" in source
    assert 'os.environ.get("HF_TOKEN")' in source
    assert 'os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")' in source
    assert '"huggingface-hub"' in source
    assert '"flash-linear-attention==0.5.1"' in source
    assert 'causal-conv1d==1.6.2.post1' in source
    assert 'nvcr.io/nvidia/pytorch:25.11-py3' in source
    assert 'pip install --no-build-isolation causal-conv1d==1.6.2.post1' in source
    assert "HfApi(token=token)" in source
    assert "token=token" in source
    assert "max_workers=16" in source


def test_model_cache_is_persistent_and_download_is_skipped_when_present():
    source = SCRIPT.read_text()
    assert 'MODEL_VOLUME_NAME = "jlens-qwen36-model-cache"' in source
    assert 'modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)' in source
    assert 'model_id.replace("/", "--")' in source
    assert "if not model_cache_complete(model_path):" in source
    assert "model_cache_complete(path, revision=revision)" in source
    assert 'cache_root / "qwen3.6-27b-hf"' in source
    assert "find_model_cache(Path(\"/models\"), model_id, revision)" in source
    assert 'partial_path = Path("/outputs") / f".{output_name}.partial"' in source
    assert "refusing to overwrite existing output artifact" in source
    assert "partial_path.rename(output_path)" in source
    assert "precision: str" in source
    assert '"--load-in-8bit"' in source
    assert "snapshot_download(" in source
    assert "local_dir=str(model_path)" in source
    assert 'model_volume.commit()' in source
    assert 'print(f"[download] using cached model at {model_path}"' in source


def test_model_download_is_cpu_only_and_precedes_gpu_fit():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    cache_decorators = "\n".join(ast.get_source_segment(source, d) or "" for d in functions["cache_model_remote"].decorator_list)
    fit_decorators = "\n".join(ast.get_source_segment(source, d) or "" for d in functions["fit_remote"].decorator_list)
    assert "cpu=2" in cache_decorators
    assert "gpu=" not in cache_decorators
    assert "gpu=GPU" in fit_decorators
    assert source.index("cache_model_remote.remote") < source.index("fit_remote.remote")
    assert "cache_only: bool = False" in source
    assert 'no GPU fit submitted' in source
    assert "cache-only resources: CPU=2" in source
    assert "image_only: bool = False" in source
    assert "image-only mode" in source
    assert source.index("if image_only:") < source.index("cache_model_remote.remote")


def test_incomplete_model_cache_is_not_accepted(tmp_path):
    module = load_guard_module()
    (tmp_path / "config.json").write_text("{}")
    assert module.model_cache_complete(tmp_path) is False
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    assert module.model_cache_complete(tmp_path) is True


def test_model_cache_requires_matching_revision(tmp_path):
    module = load_guard_module()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    assert module.model_cache_complete(tmp_path, revision="abc") is False
    (tmp_path / ".jlens_revision").write_text("abc\n")
    assert module.model_cache_complete(tmp_path, revision="abc") is True
    assert module.model_cache_complete(tmp_path, revision="different") is False


def test_sharded_model_cache_requires_every_weight_shard(tmp_path):
    module = load_guard_module()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}
    }))
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    assert module.model_cache_complete(tmp_path) is False
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
    assert module.model_cache_complete(tmp_path) is True


def test_artifact_validation_rejects_missing_or_mismatched_outputs(tmp_path):
    module = load_guard_module()
    with pytest.raises(RuntimeError, match="missing"):
        module.validate_artifact(tmp_path, model_id="Qwen/Qwen3.6-27B", revision="abc")


def test_artifact_validation_accepts_complete_finite_artifact(tmp_path):
    import torch

    module = load_guard_module()
    (tmp_path / "config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3.6-27B",
        "base_model_revision": "abc",
    }))
    (tmp_path / "fit_manifest.json").write_text(json.dumps({
        "model": "Qwen/Qwen3.6-27B",
        "model_revision": "abc",
        "load_in_4bit": True,
    }))
    torch.save({"weight": torch.ones(2)}, tmp_path / "params.pt")
    module.validate_artifact(tmp_path, model_id="Qwen/Qwen3.6-27B", revision="abc")
