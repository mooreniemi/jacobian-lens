#!/usr/bin/env python3
"""Build the standalone loop-transformer research report."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "experiments"
REPORTS = ROOT / "reports" / "loop-transformers"
DOCS = ROOT / "docs" / "loop-transformers"


def read(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def best(result: dict | None, method: str) -> tuple[int | None, float | None]:
    if result is None:
        return None, None
    values = {int(k): v for k, v in result["accuracy"][method].items()}
    layer = max(values, key=values.get)
    return layer, values[layer]


def make_plots() -> dict[str, str]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    plot_dir = REPORTS / "plots"
    plot_dir.mkdir(exist_ok=True)
    selection_files = [
        "ouro-proofwriter-virtual-loop1-selection-300.json",
        "ouro-proofwriter-virtual-loop2-selection-300.json",
        "ouro-proofwriter-virtual-loop3-selection-300.json",
        "ouro-proofwriter-virtual-loop4-selection-300.json",
    ]
    selections = [read(name) for name in selection_files]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True, constrained_layout=True)
    for index, (axis, result) in enumerate(zip(axes.flat, selections, strict=True), start=1):
        if result is None:
            axis.set_title(f"Loop {index} (not available)")
            axis.axis("off")
            continue
        layers = [int(k) for k in result["layers"]]
        jlens = [result["accuracy"]["jlens"][str(k)] for k in layers]
        logits = [result["accuracy"]["logit"][str(k)] for k in layers]
        axis.plot(layers, jlens, color="#0072B2", label="J-lens")
        axis.plot(layers, logits, color="#E69F00", label="logit lens")
        axis.axhline(result["accuracy"]["final"], color="#222", linestyle="--", label="final")
        axis.axhline(1 / 3, color="#888", linestyle=":", label="chance")
        axis.set_title(f"Loop {index}: target v{index * 48 - 1}")
        axis.set_xlabel("Virtual source layer")
        axis.set_ylabel("ProofWriter accuracy")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize="small")
    fig.suptitle("Ouro virtual-depth selection readouts")
    fig.savefig(plot_dir / "ouro-virtual-selection-readouts.png", dpi=160)
    plt.close(fig)

    confirmation_files = [
        "ouro-proofwriter-virtual-loop1-confirmation-1500.json",
        "ouro-proofwriter-virtual-loop2-confirmation-1500.json",
        "ouro-proofwriter-virtual-loop3-confirmation-1500.json",
        "ouro-proofwriter-virtual-loop4-confirmation-1500.json",
    ]
    rows = []
    for index, name in enumerate(confirmation_files, start=1):
        result = read(name)
        if result is None:
            continue
        layer, jacc = best(result, "jlens")
        _, lacc = best(result, "logit")
        rows.append((index, layer, jacc, lacc, result["accuracy"]["final"]))
    fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    if rows:
        x = list(range(len(rows)))
        axis.bar([i - 0.2 for i in x], [row[2] for row in rows], width=0.2, label="locked J-lens", color="#0072B2")
        axis.bar(x, [row[3] for row in rows], width=0.2, label="matching logit", color="#E69F00")
        axis.bar([i + 0.2 for i in x], [row[4] for row in rows], width=0.2, label="final Ouro", color="#222")
        axis.set_xticks(x, [f"Loop {row[0]}\nv{row[1]}" for row in rows])
    axis.axhline(1 / 3, color="#888", linestyle=":", label="chance")
    axis.set_ylim(0, 0.7)
    axis.set_ylabel("Confirmation accuracy")
    axis.set_title("Locked virtual readouts on source-disjoint ProofWriter")
    axis.legend(frameon=False, ncols=2)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(plot_dir / "ouro-virtual-confirmation.png", dpi=160)
    plt.close(fig)
    return {
        "selection": "plots/ouro-virtual-selection-readouts.png",
        "confirmation": "plots/ouro-virtual-confirmation.png",
    }


def report_markdown(plot_paths: dict[str, str]) -> str:
    selections = [read(f"ouro-proofwriter-virtual-loop{i}-selection-300.json") for i in range(1, 5)]
    confirmations = [read(f"ouro-proofwriter-virtual-loop{i}-confirmation-1500.json") for i in range(1, 5)]
    selection_rows = []
    for i, result in enumerate(selections, start=1):
        if result is None:
            continue
        jl, ja = best(result, "jlens")
        ll, la = best(result, "logit")
        selection_rows.append(f"| {i} | v{jl} | {ja:.1%} | v{ll} | {la:.1%} | {result['accuracy']['final']:.1%} |")
    confirmation_rows = []
    for i, result in enumerate(confirmations, start=1):
        if result is None:
            continue
        jl, ja = best(result, "jlens")
        ll, la = best(result, "logit")
        confirmation_rows.append(f"| {i} | v{jl} | {ja:.1%} | v{ll} | {la:.1%} | {result['accuracy']['final']:.1%} |")
    return f"""# Reading Looped Transformers with Virtual-Depth Jacobian Lenses

**Standalone research report · {date.today().isoformat()}**

## Abstract

Ordinary Transformer interpretability indexes computation by untied layer. Ouro-2.6B-Thinking instead applies the same 48-block decoder repeatedly for four recurrent passes. We therefore expose a virtual depth of 192 computation stages and fit target-specific Jacobian lenses at each loop boundary, using only the preceding 47 virtual stages for transport. This report compares those readouts with matched logit lenses on balanced ProofWriter.

The experiment follows the loop-specific methodology of Wang and Reid's virtual-unrolling study: recurrent blocks are addressable by `(pass, shared block)`, transport is measured within local recurrence windows, and selection layers are locked before source-disjoint confirmation. The present work is a narrower replication/extension: it evaluates a logical-label readout and does not yet include the full causal workspace suite.

## Contents

1. [Why loops differ from ordinary Transformers](#why-loops-differ-from-ordinary-transformers)
2. [Protocol and best-practice controls](#protocol-and-best-practice-controls)
3. [Selection readouts](#selection-readouts)
   1. [Figure: selection readouts](#figure-selection-readouts)
4. [Locked confirmation](#locked-confirmation)
   1. [Figure: confirmation results](#figure-confirmation-results)
5. [Interpretation and limitations](#interpretation-and-limitations)
6. [Relation to established work](#relation-to-established-work)

## Why loops differ from ordinary Transformers

In a conventional Transformer, layer 20 and layer 40 have different parameters and form a single feed-forward depth axis. In Ouro, block 20 in pass 1 and block 20 in pass 3 share parameters but receive different hidden states and recurrent context. A useful coordinate is therefore `v = 48 × pass + block`.

This changes interpretability in four ways:

- A single final-target lens can lose information crossing loop boundaries.
- The same shared block can have different functional roles at different recurrent passes.
- Readable content may be reconstructed independently on each loop rather than carried forward.
- An intervention may need to persist through later loops to remain behaviorally effective.

The model's four loop-end checkpoints are `v47`, `v95`, `v143`, and `v191`; these are not equivalent to four ordinary untied layers.

## Protocol and best-practice controls

- Model: local BF16 `ByteDance/Ouro-2.6B-Thinking`, 48 shared blocks, four recurrent passes.
- Lens fit: eight generic prompts, sequence length 64, eight leading positions skipped, dimension batch 16.
- Lens family: one target at each loop end; each target uses the immediately preceding 47 virtual layers as sources.
- Baseline: ordinary logit lens evaluated at exactly the same source virtual layers.
- Task: balanced three-way ProofWriter (`True`, `False`, `Unknown`).
- Selection: 300 items, used only to choose the source layer within each loop.
- Confirmation: 1,500 source-disjoint items, with the selected layer locked in advance.

This prevents the main loop-specific failure mode: selecting a layer after looking at the confirmation set, or treating a cross-loop final-target map as evidence that information persists across recurrence.

## Selection readouts

| Loop | Best J-lens source | J-lens | Best logit source | Logit lens | Final Ouro |
|---:|---:|---:|---:|---:|---:|\n""" + "\n".join(selection_rows) + f"""

### Figure: selection readouts

![Virtual-depth selection readouts]({plot_paths['selection']})

## Locked confirmation

| Loop | Locked J-lens source | J-lens | Best logit source | Logit lens | Final Ouro |
|---:|---:|---:|---:|---:|---:|\n""" + ("\n".join(confirmation_rows) or "| — | Pending | — | — | — | — |") + f"""

### Figure: confirmation results

![Virtual-depth confirmation results]({plot_paths['confirmation']})

## Interpretation and limitations

The first confirmed results should be read as a recurrent-depth profile, not as a universal J-lens advantage. Loop 1's selected layer did not beat the final model on confirmation, while the selected layers for later loops currently show stronger readouts. This makes recurrence position part of the hypothesis, rather than a nuisance dimension.

The evaluation remains a three-way observational label readout. It does not establish that the model internally uses the decoded representation, nor that a causal intervention would improve reasoning. The next causal tests should patch or ablate multi-direction subspaces across all remaining loops and compare maintained versus single-step interventions.

The fit is also intentionally modest: eight generic prompts rather than the 1,000-prompt reference-scale lens protocol. Results are therefore a local research pilot until replicated with a larger fit corpus and additional task families.

## Relation to established work

The closest precedent is [Looped Transformers under the Jacobian Lens](https://arxiv.org/abs/2609.01924), which introduced virtual unrolling for Ouro and Huginn and combined lens readouts with transport, workspace, and causal intervention analyses. [Operational Proto-Introspection in Looped Language Models](https://arxiv.org/abs/2607.18553) probes Ouro trajectories for process quality and branch viability, while [Step-resolved data attribution for looped transformers](https://arxiv.org/abs/2602.10097) decomposes influence by recurrent step. [Loop, Think, & Generalize](https://arxiv.org/abs/2604.07822) provides mechanistic analysis on smaller recurrent-depth models trained on synthetic compositional tasks.

Our contribution here is narrower: an independently implemented virtual-depth adapter, loop-end J-lens family, and balanced ProofWriter readout/confirmation track. It should be presented as complementary replication and task-focused evaluation, not as the first interpretability method for loop transformers.
"""


def main() -> None:
    paths = make_plots()
    markdown = report_markdown(paths)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "LOOP_TRANSFORMER_REPORT.md").write_text(markdown)
    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Loop-transformer research report</title><style>body{{margin:0;background:#f3f1ec;color:#222;font:16px/1.58 Georgia,serif}}main{{max-width:1100px;margin:2rem auto;padding:3rem 4rem;background:#fff;border:1px solid #d7d3ca}}h1,h2,h3,table{{font-family:Arial,sans-serif}}h1{{font:700 2.2rem Georgia,serif}}h2{{border-bottom:1px solid #b8b5ae;padding-bottom:.25rem;margin-top:2.5rem}}a{{color:#174a73;text-decoration:none}}a:hover{{text-decoration:underline}}table{{border-collapse:collapse;width:100%;margin:1rem 0 1.5rem;font-size:.88rem}}th,td{{border-bottom:1px solid #d6d3cc;padding:.4rem .5rem;text-align:left}}th{{border-top:1px solid #222;border-bottom:1px solid #222}}img{{max-width:100%;border:1px solid #d6d3cc}}code{{font-family:monospace}}blockquote{{border-left:3px solid #174a73;padding-left:1rem;color:#666}}</style></head><body><main>{markdown_to_html(markdown)}</main></body></html>"""
    (REPORTS / "index.html").write_text(page)
    print(f"wrote {REPORTS / 'index.html'} and {DOCS / 'LOOP_TRANSFORMER_REPORT.md'}")


def markdown_to_html(text: str) -> str:
    import mistune
    return mistune.create_markdown(escape=False, plugins=["table", "strikethrough", "footnotes"])(text)


if __name__ == "__main__":
    main()
