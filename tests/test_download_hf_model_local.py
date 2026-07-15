from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_hf_model_local.py"


def load_module():
    spec = importlib.util.spec_from_file_location("download_hf_model_local", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_first_download_validates_weights_before_writing_revision_marker(tmp_path):
    module = load_module()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"x": "model-00001-of-00001.safetensors"}
    }))
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    assert module.weights_complete(tmp_path) is True
    assert module.complete_model(tmp_path, "abc") is False
    (tmp_path / ".jlens_revision").write_text("abc\n")
    assert module.complete_model(tmp_path, "abc") is True
