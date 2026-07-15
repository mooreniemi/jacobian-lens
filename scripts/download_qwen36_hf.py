"""Safely download the full Transformers-format Qwen3.6-27B checkpoint."""
from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

repo = "Qwen/Qwen3.6-27B"
out = Path("/home/alex/models/qwen3.6-27b-hf")
free = shutil.disk_usage("/home/alex").free
print(f"download preflight: {free / 2**30:.1f} GiB free; expected checkpoint size is about 55.6 GB", flush=True)
if free < 120 * 2**30:
    raise SystemExit("refusing download: less than 120 GiB free")
out.mkdir(parents=True, exist_ok=True)
path = snapshot_download(repo_id=repo, local_dir=str(out), local_dir_use_symlinks=False)
print(f"download complete: {path}", flush=True)
