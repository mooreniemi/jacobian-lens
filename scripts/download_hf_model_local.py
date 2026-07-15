"""Download and validate a Hugging Face Transformers model locally.

The token is read from a permission-restricted file outside the repository.
The destination is resumable through huggingface_hub's local_dir cache.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def weights_complete(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    try:
        index = json.loads((path / "model.safetensors.index.json").read_text())
        shards = set(index["weight_map"].values())
    except (KeyError, json.JSONDecodeError, OSError):
        return False
    return bool(shards) and all(
        (path / shard).is_file() and (path / shard).stat().st_size > 0 for shard in shards
    )


def complete_model(path: Path, revision: str) -> bool:
    try:
        marker = (path / ".jlens_revision").read_text().strip()
    except OSError:
        return False
    return marker == revision and weights_complete(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--out", type=Path, default=Path("/home/alex/models/qwen3.6-27b-hf"))
    parser.add_argument("--token-file", type=Path, default=Path("/home/alex/.config/huggingface/modal-token"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    token = args.token_file.read_text().strip()
    if not token.startswith("hf_"):
        raise SystemExit("token file does not contain an hf_ token")
    if args.token_file.stat().st_mode & 0o077:
        raise SystemExit("token file permissions are too open; expected mode 600")

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi(token=token).model_info(args.model).sha
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[preflight] model={args.model} revision={revision}", flush=True)
    if complete_model(args.out, revision):
        print(f"[cache] complete local model already exists at {args.out}", flush=True)
        return
    print(f"[download] destination={args.out} workers={args.workers} Xet=high-performance", flush=True)
    snapshot_download(
        args.model,
        revision=revision,
        local_dir=str(args.out),
        token=token,
        max_workers=args.workers,
    )
    if not weights_complete(args.out):
        raise SystemExit("download finished without all indexed Safetensors shards")
    (args.out / ".jlens_revision").write_text(revision + "\n")
    print(f"[complete] validated local model at {args.out}", flush=True)


if __name__ == "__main__":
    main()
