# Artifact storage and retrieval

Git contains the recipe and provenance for experiments. Large model and lens
binary artifacts live in an artifact store and are referenced by manifests and
hashes.

## What is recorded locally

For each reusable artifact, keep or publish:

- canonical base-model ID and immutable revision;
- lens variant and training corpus label;
- precision and quantization;
- fit configuration and software versions;
- SHA256 of the binary artifact;
- artifact-store repository/path;
- the exact retrieval command;
- a small JSON manifest that does not require the binary to parse.

## Hugging Face lens artifacts

Hugging Face is the preferred public home for reusable lens artifacts. We have
not created the upload repository yet; after choosing its name, the intended
layout is:

```text
<hf-user>/jacobian-lens-artifacts/
  qwen3-0.6b/tuned-pile-repro-v1/
  smollm2-135m/tuned-pile-repro-v1/
  qwen3-1.7b/tuned-pile-repro-v1/
```

Upload a validated artifact with the authenticated CLI:

```bash
hf upload <hf-user>/jacobian-lens-artifacts \
  data/lenses/qwen3-0.6b-tuned-pile-repro-v1 \
  qwen3-0.6b/tuned-pile-repro-v1 \
  --repo-type model
```

Retrieve it into the local artifact layout with:

```bash
hf download <hf-user>/jacobian-lens-artifacts \
  --repo-type model \
  --include 'qwen3-0.6b/tuned-pile-repro-v1/*' \
  --local-dir data/lenses/qwen3-0.6b-tuned-pile-repro-v1
```

After download, verify `config.json`, `fit_manifest.json`, `params.pt`, the
recorded SHA256, and model compatibility before causal use. The binary is
ignored by Git; the manifest and retrieval instructions are the durable record.

## Modal artifacts

The 27B Modal artifacts remain in the named Modal volume and are not copied
into Git. Retrieval should use the volume name and artifact subdirectory
recorded in the run log, for example:

```bash
modal volume get jlens-qwen36-tuned-artifacts \
  qwen3.6-27b-tuned-wikitext-nf4 ./data/lenses/qwen3.6-27b-tuned-wikitext-nf4
```

The exact volume/app IDs and billing provenance are recorded in the research
log and `data/provenance/`. Before publishing, verify the command against the
current Modal CLI and add the final artifact hash.

## Git LFS

Git LFS is installed locally and initialized for this checkout, but it is not
currently used. It is reserved for a deliberately published binary smaller
than the applicable GitHub per-file limit. It is not a substitute for HF or
object storage for model checkpoints, Pile shards, Docker images, or the 27B
multi-gigabyte lens artifacts.
