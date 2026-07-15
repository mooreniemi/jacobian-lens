"""Create the read-only training-data exploration notebook."""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "training_data_explorer.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md(
        """
        # Training-data explorer for J-lens causal experiments

        This notebook is a read-only map of what data was used to fit each
        intervention and what data is only used for causal evaluation. It does
        not load a model or allocate GPU memory.

        The key distinction is:

        - **J-lens:** a learned Jacobian/readout map. The released 27B lens
          metadata is available, but its original prompt text is not in this
          checkout. Our smaller local J-lens prompt files are inspectable.
        - **Logit lens:** no fitted data; it uses the model's unembedding rows.
        - **Tuned lens:** a learned affine translator. Our artifacts record a
          Wikitext-2 training source and fit settings.
        - **Random matched:** no fitted data; it matches perturbation norm.

        The causal fixtures are shown separately because they are evaluation
        prompts, not lens-training data.
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        from collections import Counter
        from IPython.display import display, Markdown

        ROOT = Path.cwd()
        if not (ROOT / "data").exists():
            ROOT = Path("/home/alex/Code/jlens/jacobian-lens")
        print("Repository:", ROOT)
        """
    ),
    md("## 1. Intervention data inventory"),
    code(
        """
        inventory = [
            {
                "intervention": "J-lens",
                "fit_data": "Prompt corpus; released 27B prompt text unavailable locally",
                "local_artifact": "data/lenses/*jacobian* plus data/lens-prompts/ for smaller local fits",
                "trained": True,
            },
            {
                "intervention": "Logit lens",
                "fit_data": "None",
                "local_artifact": "Model unembedding matrix",
                "trained": False,
            },
            {
                "intervention": "Tuned lens",
                "fit_data": "Salesforce/wikitext, wikitext-2-raw-v1, train",
                "local_artifact": "data/lenses/*tuned-wikitext*/fit_manifest.json",
                "trained": True,
            },
            {
                "intervention": "Random matched",
                "fit_data": "None",
                "local_artifact": "Runtime-generated norm-matched random vector",
                "trained": False,
            },
        ]
        display(inventory)
        """
    ),
    md("## 2. Local J-lens prompt corpora"),
    code(
        """
        prompt_dir = ROOT / "data" / "lens-prompts"
        for path in sorted(prompt_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            items = payload.get("items", [])
            counts = Counter(item.get("category", "unknown") for item in items)
            print(f"{path.relative_to(ROOT)}: {len(items)} prompts; categories={dict(counts)}")
            for item in items[:3]:
                print(" -", item.get("name"), ":", item.get("prompt"))
            print()
        """
    ),
    md(
        """
        These are our locally assembled prompt mixes for smaller model-matched
        fits. They include typo correction, associations, transformations,
        factual/reasoning-style prompts, and other categories. They are not
        evidence that the released Anthropic 27B J-lens used the same corpus.
        """
    ),
    md("## 3. Tuned-lens fit manifests"),
    code(
        """
        manifests = []
        for path in sorted((ROOT / "data" / "lenses").glob("*/fit_manifest.json")):
            manifest = json.loads(path.read_text())
            manifests.append({
                "artifact": str(path.parent.relative_to(ROOT)),
                "model": manifest.get("model"),
                "dataset": manifest.get("dataset"),
                "split": manifest.get("split"),
                "chunks": manifest.get("max_chunks"),
                "length": manifest.get("max_length"),
                "steps": manifest.get("steps"),
                "objective": manifest.get("objective"),
                "precision": manifest.get("quantization", manifest.get("dtype")),
            })
        display(manifests)
        """
    ),
    md(
        """
        For the 27B tuned fit specifically, the manifest says: 512 Wikitext
        chunks, maximum length 128, 100 steps, BF16 compute over NF4 weights,
        and per-layer KL divergence to the frozen final logits. Wikitext is
        raw text; it is not the same thing as our causal reasoning fixtures.
        """
    ),
    code(
        """
        # Optional: fetch/display raw Wikitext examples if `datasets` is installed.
        # This cell may access the Hugging Face cache or network; it is not needed
        # to inspect the local metadata above.
        try:
            from datasets import load_dataset
            wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            rows = [row["text"] for row in wiki if row["text"].strip()][:5]
            for i, row in enumerate(rows, 1):
                print(f"Wikitext example {i}: {row[:500]!r}")
        except Exception as exc:
            print("Wikitext loading skipped:", type(exc).__name__, exc)
        """
    ),
    md("## 4. Causal evaluation fixtures (not training data)"),
    code(
        """
        fixtures = {
            "verbal report": ROOT / "data/experiments/verbal-report.json",
            "two-hop reasoning": ROOT / "data/experiments/probe-swap.json",
            "flexible generalization": ROOT / "data/experiments/flexible-generalization.json",
            "conjunction candidates": ROOT / "data/benchmarks/proofwriter-strong-conjunctions.jsonl",
        }
        for name, path in fixtures.items():
            if path.suffix == ".jsonl":
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                print(f"{name}: {len(rows)} rows from {path.relative_to(ROOT)}")
                for row in rows[:2]: print(" -", row.get("prompt", row))
            else:
                payload = json.loads(path.read_text())
                rows = payload.get("items", payload.get("categories", payload))
                print(f"{name}: {len(rows) if hasattr(rows, '__len__') else 'structured'} from {path.relative_to(ROOT)}")
                if isinstance(rows, list):
                    for row in rows[:2]: print(" -", row.get("prompt", row))
            print()
        """
    ),
    md(
        """
        These fixtures define what the model is asked and how success is
        scored. They should not be described as the training corpus for any
        lens unless a fit command explicitly points to them.
        """
    ),
    md(
        """
        ## Questions to investigate next

        1. Do the smaller J-lens prompt mixes contain enough category and
           linguistic diversity for the intended claim?
        2. Should the tuned lens use raw Wikitext, task-like prompts, or both?
        3. Should we reserve a held-out corpus for fit-quality checks before
           comparing causal effectiveness?
        4. Can we obtain the original released J-lens fitting prompts or only
           reproduce their reported count and broad corpus description?
        """
    ),
]

notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT)
print(f"wrote {OUT}")
