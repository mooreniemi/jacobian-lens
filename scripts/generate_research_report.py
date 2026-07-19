#!/usr/bin/env python3
"""Generate a human-readable HTML report from recorded causal results.

The report deliberately reads the JSON artifacts in data/experiments rather
than duplicating result values in a template.  It is intended to be served
locally and viewed through SSH port forwarding.
"""
from __future__ import annotations

import argparse
import html
import json
import posixpath
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mistune
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "experiments"
DOCS_ROOT = ROOT / "docs"


def documentation_href(source: Path) -> str:
    """Return the URL for a Markdown document in the served report."""
    relative = source.relative_to(DOCS_ROOT).with_suffix(".html")
    return f"docs/{relative.as_posix()}"


def rewrite_document_links(markdown_text: str, source: Path) -> str:
    """Point relative Markdown-document links at the rendered HTML copies."""
    pattern = re.compile(r"(?<!\!)\[([^\]]+)\]\(([^)]+)\)")

    def replace(match: re.Match[str]) -> str:
        label, target = match.groups()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)
        target_path, fragment = (target.split("#", 1) + [""])[:2]
        candidate = (source.parent / target_path).resolve()
        if candidate.is_file() and candidate.suffix.lower() == ".md" and candidate.is_relative_to(DOCS_ROOT):
            candidate_relative = candidate.relative_to(DOCS_ROOT).with_suffix(".html").as_posix()
            source_parent = source.relative_to(DOCS_ROOT).parent.as_posix() or "."
            rendered = posixpath.relpath(candidate_relative, source_parent)
            if fragment:
                rendered += f"#{fragment}"
            return f"[{label}]({rendered})"
        return match.group(0)

    return pattern.sub(replace, markdown_text)


def render_documentation(out_dir: Path) -> list[tuple[str, str, str]]:
    """Render research Markdown into browser-readable pages alongside report."""
    renderer = mistune.create_markdown(escape=False, plugins=["table", "strikethrough", "footnotes"])
    rows = []
    for source in sorted(DOCS_ROOT.rglob("*.md")):
        relative = source.relative_to(DOCS_ROOT)
        output = out_dir / "docs" / relative.with_suffix(".html")
        output.parent.mkdir(parents=True, exist_ok=True)
        markdown_text = rewrite_document_links(source.read_text(), source)
        rendered = renderer(markdown_text)
        heading = re.search(r"^#\s+(.+)$", markdown_text, flags=re.MULTILINE)
        title = heading.group(1).strip() if heading else source.stem.replace("_", " ")
        page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · J-Lens research report</title>
<style>
:root {{ --paper:#fff; --wash:#f3f1ec; --ink:#222; --muted:#666; --rule:#b8b5ae; --accent:#174a73; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--wash); color:var(--ink); font:16px/1.58 Georgia,'Times New Roman',serif; }}
main {{ max-width:1120px; margin:2rem auto; padding:3.5rem 4.5rem; background:var(--paper); border:1px solid #d7d3ca; box-shadow:0 1px 8px #00000012; }}
h1,h2,h3,table,nav,footer {{ font-family:Arial,Helvetica,sans-serif; }} h1 {{ font:700 2.2rem/1.15 Georgia,serif; }} h2 {{ margin-top:2.2rem; border-bottom:1px solid var(--rule); padding-bottom:.25rem; }}
a {{ color:var(--accent); text-decoration:none; }} a:hover {{ text-decoration:underline; }} nav {{ margin-bottom:2rem; }}
table {{ border-collapse:collapse; width:100%; margin:1rem 0 1.5rem; font-size:.86rem; }} th,td {{ border-bottom:1px solid #d6d3cc; padding:.4rem .5rem; text-align:left; vertical-align:top; }} th {{ border-top:1px solid var(--ink); border-bottom:1px solid var(--ink); }}
pre {{ overflow:auto; padding:1rem; background:#f5f4f0; border:1px solid #dedbd3; }} code {{ font:.88em/1.35 'SFMono-Regular',Consolas,monospace; }} blockquote {{ border-left:3px solid var(--accent); margin-left:0; padding-left:1rem; color:var(--muted); }}
img {{ max-width:100%; }} footer {{ margin-top:3rem; padding-top:1rem; border-top:1px solid var(--rule); color:var(--muted); font-size:.8rem; }}
@media (max-width:760px) {{ main {{ margin:0; padding:2rem 1.2rem; }} table {{ display:block; overflow-x:auto; }} }}
</style></head><body><main>
<nav><a href="../index.html">← Research report</a> · <a href="index.html">Documentation index</a></nav>
<article>{rendered}</article>
<footer>Rendered from <code>{html.escape(str(relative))}</code> by <code>scripts/generate_research_report.py</code>.</footer>
</main></body></html>"""
        output.write_text(page)
        rows.append((title, documentation_href(source), str(relative)))
    index_items = "".join(
        f"<li><a href='{html.escape(Path(relative).with_suffix('.html').as_posix())}'>{html.escape(title)}</a>"
        f" <span class='muted'>({html.escape(str(relative))})</span></li>"
        for title, _href, relative in rows
    )
    index_page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Documentation index · J-Lens report</title>
<style>body {{ margin:2rem auto; max-width:900px; padding:0 1.2rem; font:16px/1.58 Georgia,serif; background:#f3f1ec; color:#222; }} main {{ background:#fff; padding:2rem 3rem; border:1px solid #d7d3ca; }} a {{ color:#174a73; text-decoration:none; }} a:hover {{ text-decoration:underline; }} .muted {{ color:#666; }}</style>
</head><body><main><p><a href="../index.html">← Research report</a></p><h1>Research documentation</h1><p>Browser-rendered copies of the Markdown logs, protocols, summaries, and provenance notes.</p><ul>{index_items}</ul></main></body></html>"""
    (out_dir / "docs" / "index.html").write_text(index_page)
    return rows


CAUSAL_SPECS = {
    "verbal": {
        "label": "Verbal report",
        "pattern": "verbal-report-causal-*.json",
        "aggregate_headers": [
            "method",
            "n",
            "improved",
            "median_delta_rank",
            "mean_delta_rank",
            "top1",
            "top10",
        ],
    },
    "multihop": {
        "label": "Two-hop reasoning",
        "pattern": "multihop-causal-*.json",
        "aggregate_headers": ["method", "n", "improved", "median_delta_rank", "top1", "top5"],
    },
    "flexible": {
        "label": "Flexible generalization",
        "pattern": "flexible-causal-*.json",
        "aggregate_headers": ["method", "n", "improved", "median_delta_rank", "top1", "top5"],
    },
}

# The JSON result schema keeps the intervention key ``tuned`` for compatibility,
# but rendered research outputs must expose the actual artifact provenance.
PLOT_METHOD_LABELS = {
    "jlens": "J-lens",
    "logit": "logit lens",
    "random": "random matched",
    "tuned": "tuned-wiki-small-v0",
    "tuned_pile": "tuned-pile-repro-v1",
}

# Semantic colors are fixed across every plot. In particular, random matched
# is always red, regardless of which methods are present in a given panel.
PLOT_METHOD_COLORS = {
    "jlens": "#0072B2",       # blue
    "logit": "#E69F00",       # orange
    "tuned": "#009E73",       # green
    "tuned_short": "#CC79A7", # mauve
    "tuned_pile": "#56B4E9",   # light blue
    "random": "#D62728",      # red
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def model_label(path: Path, data: dict) -> str:
    stem = path.stem
    for suffix in ("-tuned", "-27b", "-4b", "-0.8b", "-0.6b", "-smollm2-135m"):
        stem = stem.removesuffix(suffix)
    model = data.get("model", "")
    if "Qwen3.6-27B" in model or "27b" in path.name:
        return "Qwen3.6-27B"
    if "Qwen3.5-4B" in model or "-4b" in path.name:
        return "Qwen3.5-4B"
    if "Qwen3.5-0.8B" in model or "-0.8b" in path.name:
        return "Qwen3.5-0.8B"
    if "Qwen3-0.6B" in model or "-0.6b" in path.name:
        return "Qwen3-0.6B"
    if "Qwen3-1.7B" in model or "-1.7b" in path.name:
        return "Qwen3-1.7B"
    if "SmolLM2" in model or "smollm2" in path.name:
        return "SmolLM2-135M"
    return stem


def causal_runs(spec: dict) -> list[tuple[str, Path, dict]]:
    runs = []
    for path in sorted(RESULTS.glob(spec["pattern"])):
        # Keep the stronger Pile reproduction track separate from the older
        # Wikitext-pilot plots; both schemas use ``tuned`` internally.
        if "tuned-pile-repro-v1" in path.name:
            continue
        if "smoke" in path.name:
            continue
        data = load_json(path)
        # Avoid treating the generic old fixtures as comparable causal runs.
        if not data.get("aggregate") or not any(
            row[0] in {"jlens", "logit", "random", "tuned"}
            for row in data["aggregate"]
        ):
            continue
        runs.append((model_label(path, data), path, data))
    return runs


def pile_causal_runs(spec: dict) -> list[tuple[str, Path, dict]]:
    """Load Pile causal runs separately from the Wikitext pilot runs."""
    prefix = spec["pattern"].removesuffix("*.json")
    runs = []
    for path in sorted(RESULTS.glob(f"{prefix}*-tuned-pile-repro-v1.json")):
        data = load_json(path)
        if model_label(path, data) == "Qwen3.5-0.8B" and data.get("kernel_mode") != "on":
            continue
        if data.get("aggregate"):
            runs.append((model_label(path, data), path, data))
    return runs


def pile_causal_table() -> str:
    """Render validated Pile causal outputs without pooling them with Wikitext."""
    rows = []
    for key, spec in CAUSAL_SPECS.items():
        prefix = spec["pattern"].removesuffix("*.json")
        pattern = f"{prefix}*-tuned-pile-repro-v1.json"
        for path in sorted(RESULTS.glob(pattern)):
            data = load_json(path)
            if model_label(path, data) == "Qwen3.5-0.8B" and data.get("kernel_mode") != "on":
                continue
            for aggregate in data.get("aggregate", []):
                method = "tuned-pile-repro-v1" if aggregate[0] == "tuned" else PLOT_METHOD_LABELS.get(aggregate[0], aggregate[0])
                rows.append([
                    model_label(path, data), spec["label"], method,
                    aggregate[1], aggregate[2], aggregate[3],
                    aggregate[4], aggregate[5] if len(aggregate) > 5 else "—",
                    path.name,
                ])
    if not rows:
        return "<p class='muted'>No validated Pile causal outputs yet; they are queued behind held-out predictive validation.</p>"
    return table(
        ["Model", "Task", "Intervention", "n", "Improved", "Median Δrank", "Top-1", "Top-5/10", "Artifact"],
        rows,
    )


def provisional_short_pile_causal_table() -> str:
    """Render explicitly provisional short-2M causal outputs."""
    rows = []
    for key, spec in CAUSAL_SPECS.items():
        prefix = spec["pattern"].removesuffix("*.json")
        pattern = f"{prefix}*-tuned-pile-repro-v1-short-2m.json"
        for path in sorted(RESULTS.glob(pattern)):
            data = load_json(path)
            for aggregate in data.get("aggregate", []):
                method = "tuned-pile-repro-v1-short-2m" if aggregate[0] == "tuned" else PLOT_METHOD_LABELS.get(aggregate[0], aggregate[0])
                rows.append([
                    model_label(path, data), spec["label"], method,
                    aggregate[1], aggregate[2], aggregate[3],
                    aggregate[4], aggregate[5] if len(aggregate) > 5 else "—",
                    path.name,
                ])
    if not rows:
        return "<p class='muted'>The provisional short-2M causal queue has not produced a full-suite output yet.</p>"
    return table(
        ["Model", "Task", "Intervention", "n", "Improved", "Median Δrank", "Top-1", "Top-5/10", "Artifact"],
        rows,
    )


def pile_validation_table() -> str:
    """Render held-out Pile predictive checks when their JSON artifacts exist."""
    paths = sorted(RESULTS.glob("*-tuned-pile-repro-v1-eval.json"))
    rows = []
    for path in paths:
        data = load_json(path)
        layers = data.get("layers", [])
        if not layers:
            continue
        checkpoints = [layers[0], layers[len(layers) // 2], layers[-1]]
        rows.append([
            model_label(path, data),
            f"{data.get('tokens', 0):,}",
            len(layers),
            " / ".join(f"{item['tuned_kl']:.3f}" for item in checkpoints),
            " / ".join(f"{item['logit_kl']:.3f}" for item in checkpoints),
            path.name,
        ])
    if not rows:
        return "<p class='muted'>No full held-out Pile artifacts yet.</p>"
    return table(
        ["Model", "Scored tokens", "Layers", "Tuned KL (first / middle / final)", "Logit KL (first / middle / final)", "Artifact"],
        rows,
    )


def causal_task_examples() -> dict[str, str]:
    """Render one concrete fixture example for each causal task."""
    verbal = load_json(RESULTS / "verbal-report.json")["candidates"]
    verbal_words = verbal["country"][:3]
    verbal_rows = [[
        "Think of a country. Answer in one word.",
        verbal_words[0],
        verbal_words[1],
        "Swap the source representation toward the target candidate; score target rank at the answer position.",
    ]]

    multihop_item = load_json(RESULTS / "probe-swap.json")["items"][0]
    multihop_rows = [[
        multihop_item["prompt"],
        f"{multihop_item['intermediate']} → {multihop_item['swap_to']}",
        f"{multihop_item['answer']} → {multihop_item['swap_answer']}",
        "Swap the intermediate entity and test whether the downstream answer follows it.",
    ]]

    flexible = load_json(RESULTS / "flexible-generalization.json")["categories"][0]
    function = flexible["funcs"][0]
    source, target = flexible["args"][:2]
    flexible_rows = [[
        function["template"].format(arg=source),
        f"{source} → {target}",
        f"{function['answers'][source]} → {function['answers'][target]}",
        "Swap the argument representation and test whether the function is recomputed for the new argument.",
    ]]

    return {
        "verbal": fold(
            "Show one verbal-report example",
            table(["Prompt", "Source candidate", "Target candidate", "Test"], verbal_rows),
        ),
        "multihop": fold(
            "Show one two-hop example",
            table(["Prompt", "Intermediate swap", "Expected answer shift", "Test"], multihop_rows),
        ),
        "flexible": fold(
            "Show one flexible-generalization example",
            table(["Prompt", "Argument swap", "Expected answer shift", "Test"], flexible_rows),
        ),
    }


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:+.1f}" if value else "0"
    return str(value)


def table(headers: list[str], rows: list[list[object]], classes: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(fmt(v))}</td>" for v in row) + "</tr>")
    return f'<table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def fold(title: str, content: str) -> str:
    """Render a closed-by-default disclosure section."""
    return f"<details><summary>{html.escape(title)}</summary>{content}</details>"


def paired_causal_table() -> str:
    """Render item-paired contrasts from the durable paired analysis artifact."""
    path = ROOT / "data/analysis/paired_causal_effects.json"
    if not path.exists():
        return "<p class='muted'>Paired analysis has not been generated yet.</p>"

    task_labels = {
        "verbal": "Verbal report",
        "multihop": "Two-hop reasoning",
        "flexible": "Flexible generalization",
    }
    model_labels = {
        "qwen3-0.6b": "Qwen3-0.6B",
        "qwen3-1.7b": "Qwen3-1.7B",
        "qwen3.5-0.8b": "Qwen3.5-0.8B",
        "qwen3.5-4b": "Qwen3.5-4B",
        "qwen3.6-27b": "Qwen3.6-27B",
        "smollm2-135m": "SmolLM2-135M",
    }

    def model_label(raw: str) -> str:
        lowered = raw.lower()
        for key, label in model_labels.items():
            if key in lowered:
                return label
        return Path(raw).name or raw

    def signed_pp(value: float) -> str:
        return f"{value * 100:+.1f} pp"

    def p_value(value: float) -> str:
        return "<0.001" if value < 0.001 else f"{value:.3f}"

    rows = []
    for row in json.loads(path.read_text()):
        # Random-v-logit belongs in the control table above; this section is
        # intended to explain method-v-logit contrasts at the same item.
        if row.get("left") not in {"jlens", "tuned"} or row.get("right") != "logit":
            continue
        rows.append([
            task_labels.get(row["task"], row["task"]),
            model_label(row["model"]),
            "J-lens" if row["left"] == "jlens" else "tuned lens",
            row["n_items"],
            signed_pp(row["paired_success_diff"]),
            f"[{signed_pp(row['paired_success_ci_low'])}, {signed_pp(row['paired_success_ci_high'])}]",
            f"{row['paired_delta_mean']:+.1f}",
            p_value(row["paired_success_p"]),
        ])
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return table(
        ["Task", "Model", "Contrast", "Items", "Success-rate difference", "95% bootstrap CI", "Mean Δrank difference", "Paired sign-flip p"],
        rows,
    )


def task_discriminability_table(all_runs: dict[str, list[tuple[str, Path, dict]]]) -> str:
    """Summarize whether the current two-hop and flexible tasks separate."""
    model_set = {"SmolLM2-135M", "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3.5-0.8B", "Qwen3.5-4B", "Qwen3.6-27B"}
    rates: dict[str, dict[str, dict[str, float]]] = {"multihop": {}, "flexible": {}}
    for task in rates:
        for model, _path, data in all_runs[task]:
            if model not in model_set:
                continue
            for aggregate in data.get("aggregate", []):
                method = aggregate[0]
                if method in {"jlens", "logit", "random", "tuned"}:
                    rates[task].setdefault(method, {})[model] = 100 * aggregate[2] / aggregate[1]

    rows = []
    for method in ("jlens", "logit", "random", "tuned"):
        shared = sorted(set(rates["multihop"].get(method, {})) & set(rates["flexible"].get(method, {})))
        if len(shared) < 3:
            continue
        multihop = np.asarray([rates["multihop"][method][model] for model in shared])
        flexible = np.asarray([rates["flexible"][method][model] for model in shared])
        correlation = float(np.corrcoef(multihop, flexible)[0, 1])
        rows.append([
            PLOT_METHOD_LABELS[method],
            len(shared),
            f"{multihop.mean():.1f}%",
            f"{flexible.mean():.1f}%",
            f"{correlation:+.2f}",
        ])
    return table(
        ["Intervention", "Shared models", "Mean two-hop", "Mean flexible", "Across-model Pearson r"],
        rows,
    )


def coverage_table() -> str:
    """Build a model/intervention-by-measurement coverage matrix."""
    tasks = {
        "verbal": ("Verbal report", "VERBAL_REPORT_CAUSAL_LOG.html"),
        "multihop": ("Two-hop reasoning", "MULTIHOP_CAUSAL_LOG.html"),
        "flexible": ("Flexible generalization", "FLEXIBLE_GENERALIZATION_LOG.html"),
    }
    models = [
        ("SmolLM2-135M", "135M"),
        ("Qwen3-0.6B", "0.6B"),
        ("Qwen3-1.7B", "1.7B"),
        ("Qwen3.5-0.8B", "0.8B"),
        ("Qwen3.5-4B", "4B"),
        ("Qwen3.6-27B", "27B"),
    ]
    interventions = [
        ("J-lens", "jlens"),
        ("logit lens", "logit"),
        ("random matched", "random"),
        ("tuned-wiki-small-v0", "tuned"),
        ("tuned-pile-repro-v1", "pile"),
    ]
    pile_states = {
        "Qwen3-0.6B": "VALIDATED PREDICTIVE + CAUSAL COMPLETE",
        "SmolLM2-135M": "VALIDATED PREDICTIVE + CAUSAL COMPLETE",
        "Qwen3-1.7B": "FULL HELD-OUT GATE IN PROGRESS",
        "Qwen3.5-0.8B": "PREDICTIVE COMPLETE / CAUSAL KERNEL-PARITY RERUN QUEUED",
        "Qwen3.5-4B": "VALIDATED PREDICTIVE + CAUSAL COMPLETE",
        "Qwen3.6-27B": "FIT + SHORT-2M CAUSAL TODO",
    }

    run_index: dict[tuple[str, str, str], Path] = {}
    for task_key, (_task_label, _log_name) in tasks.items():
        for model, path, data in causal_runs(CAUSAL_SPECS[task_key]):
            for aggregate in data.get("aggregate", []):
                method = aggregate[0]
                if method in {"jlens", "logit", "random", "tuned"}:
                    run_index[(model, method, task_key)] = path

    short_run_index: dict[tuple[str, str, str], Path] = {}
    for task_key, spec in CAUSAL_SPECS.items():
        prefix = spec["pattern"].removesuffix("*.json")
        for path in sorted(RESULTS.glob(f"{prefix}*tuned-pile-repro-v1-short-2m.json")):
            data = load_json(path)
            for aggregate in data.get("aggregate", []):
                if aggregate[0] in {"logit", "random"}:
                    short_run_index[(model_label(path, data), aggregate[0], task_key)] = path

    def cell(model: str, method: str, task_key: str) -> str:
        if method == "pile":
            status = pile_states[model]
            return f"<a href='docs/RESEARCH_LOG.html' title='Pile track status'>{html.escape(status)}</a>"
        path = run_index.get((model, method, task_key))
        if path is None and model == "Qwen3-1.7B":
            path = short_run_index.get((model, method, task_key))
            if path is not None:
                return f"<a href='docs/RESEARCH_LOG.html' title='{html.escape(path.name)}'>complete (short-2M)</a>"
        if path is None:
            return "<span class='muted'>not run</span>"
        log_name = tasks[task_key][1]
        return f"<a href='docs/experiments/{log_name}' title='{html.escape(path.name)}'>complete</a>"

    headers = ["Model", "Parameters", "Intervention", *[label for label, _ in tasks.values()]]
    rows = []
    for model, params in models:
        for intervention, method in interventions:
            rows.append([model, params, intervention, *(cell(model, method, task) for task in tasks)])
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell_html}</td>" for cell_html in row) + "</tr>" for row in rows)
    return f'<table class="coverage"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table><p class="muted sans">Each row is one model–intervention pair; columns are the measurement protocols. “complete” links to the experiment log. Pile status: IN PROGRESS = held-out evaluation running; QUEUED = local queue waiting; NOT STARTED = no fit begun.</p>'


def save_plot(path: Path, title: str, series: dict[str, list[float]], labels: list[str], ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    x = list(range(len(labels)))
    width = 0.8 / max(1, len(series))
    for index, (name, values) in enumerate(series.items()):
        offsets = [value + (index - (len(series) - 1) / 2) * width for value in x]
        ax.bar(offsets, values, width=width, label=name)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=min(3, len(series)))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def item_change_clusters(data: dict, method: str) -> list[list[float]]:
    """Return per-item layer changes for cluster bootstrap resampling."""
    if "verbal" in data.get("methods", []) or "target_rank_before" in data.get("items", [{}])[0]:
        before_key, after_key = "target_rank_before", "target_rank_after"
    elif "swap_answer_rank_before" in data.get("items", [{}])[0]:
        before_key, after_key = "swap_answer_rank_before", "swap_answer_rank_after"
    else:
        before_key, after_key = "target_answer_rank_before", "target_answer_rank_after"
    clusters = []
    for item in data.get("items", []):
        changes = []
        before = item.get(before_key)
        for layer in item.get("layers", {}).values():
            if method in layer and before is not None and after_key in layer[method]:
                changes.append(before - layer[method][after_key])
        if changes:
            clusters.append(changes)
    return clusters


def bootstrap_rate_ci(data: dict, method: str, seed: int = 0, n_boot: int = 2000) -> tuple[float, float, float] | None:
    """Estimate success-rate CI, resampling prompts/items as clusters."""
    clusters = item_change_clusters(data, method)
    if not clusters:
        return None
    flat = np.asarray([change for cluster in clusters for change in cluster], dtype=float)
    point = float(np.mean(flat > 0) * 100)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for index in range(n_boot):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        sampled = np.concatenate([np.asarray(clusters[i], dtype=float) for i in selected])
        estimates[index] = np.mean(sampled > 0) * 100
    low, high = np.percentile(estimates, [2.5, 97.5])
    return point, float(low), float(high)


def bootstrap_rate_difference(data: dict, method_a: str, method_b: str, seed: int = 0, n_boot: int = 2000) -> tuple[float, float, float] | None:
    """Bootstrap paired rate difference method_a minus method_b by item."""
    paired = []
    for item in data.get("items", []):
        before_key = "target_rank_before" if "target_rank_before" in item else (
            "swap_answer_rank_before" if "swap_answer_rank_before" in item else "target_answer_rank_before"
        )
        after_key = "target_rank_after" if before_key == "target_rank_before" else (
            "swap_answer_rank_after" if before_key == "swap_answer_rank_before" else "target_answer_rank_after"
        )
        a, b = [], []
        for layer in item.get("layers", {}).values():
            if method_a in layer and after_key in layer[method_a]:
                a.append(item[before_key] - layer[method_a][after_key])
            if method_b in layer and after_key in layer[method_b]:
                b.append(item[before_key] - layer[method_b][after_key])
        if a and b:
            paired.append((a, b))
    if not paired:
        return None

    def rate(clusters: list[list[float]]) -> float:
        values = np.asarray([value for cluster in clusters for value in cluster], dtype=float)
        return float(np.mean(values > 0) * 100)

    point = rate([a for a, _ in paired]) - rate([b for _, b in paired])
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for index in range(n_boot):
        selected = rng.integers(0, len(paired), size=len(paired))
        estimates[index] = rate([paired[i][0] for i in selected]) - rate([paired[i][1] for i in selected])
    low, high = np.percentile(estimates, [2.5, 97.5])
    return point, float(low), float(high)


def bootstrap_median_ci(data: dict, method: str, seed: int = 0, n_boot: int = 2000) -> tuple[float, float, float] | None:
    """Estimate the median rank-change CI with item-cluster resampling."""
    clusters = item_change_clusters(data, method)
    if not clusters:
        return None
    rng = np.random.default_rng(seed)
    point = float(np.median(np.asarray([value for cluster in clusters for value in cluster], dtype=float)))
    estimates = np.empty(n_boot)
    for index in range(n_boot):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        sampled = np.concatenate([np.asarray(clusters[i], dtype=float) for i in selected])
        estimates[index] = np.median(sampled)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return point, float(low), float(high)


def save_ci_plot(path: Path, title: str, stats: dict[str, list[tuple[float, float, float] | None]], labels: list[str], ylabel: str, method_labels: dict[str, str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fig.subplots_adjust(bottom=0.28)
    x = np.arange(len(labels), dtype=float)
    methods = list(stats)
    width = 0.8 / max(1, len(methods))
    for index, method in enumerate(methods):
        values = [entry[0] if entry else np.nan for entry in stats[method]]
        lows = [entry[0] - entry[1] if entry else 0 for entry in stats[method]]
        highs = [entry[2] - entry[0] if entry else 0 for entry in stats[method]]
        positions = x + (index - (len(methods) - 1) / 2) * width
        label = (method_labels or {}).get(method, PLOT_METHOD_LABELS.get(method, method))
        ax.bar(positions, values, width=width, label=label, color=PLOT_METHOD_COLORS.get(method), yerr=[lows, highs], capsize=3)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        frameon=False,
        ncols=min(4, len(methods)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_paired_contrast_plot(path: Path) -> None:
    """Plot same-item method-vs-logit success-rate contrasts from the analysis artifact."""
    analysis_path = ROOT / "data/analysis/paired_causal_effects.json"
    rows = [
        row for row in json.loads(analysis_path.read_text())
        if row.get("right") == "logit" and row.get("left") in {"jlens", "tuned"}
    ]
    task_order = ["verbal", "multihop", "flexible"]
    task_labels = {"verbal": "Verbal report", "multihop": "Two-hop reasoning", "flexible": "Flexible generalization"}
    model_order = ["SmolLM2-135M", "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3.5-0.8B", "Qwen3.5-4B", "Qwen3.6-27B"]
    model_keys = {
        "SmolLM2-135M": "smollm2-135m",
        "Qwen3-0.6B": "qwen3-0.6b",
        "Qwen3-1.7B": "qwen3-1.7b",
        "Qwen3.5-0.8B": "qwen3.5-0.8b",
        "Qwen3.5-4B": "qwen3.5-4b",
        "Qwen3.6-27B": "qwen3.6-27b",
    }
    methods = ["jlens", "tuned"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.4), sharex=True, sharey=True, constrained_layout=True)
    offsets = {"jlens": -0.12, "tuned": 0.12}
    plotted = set()
    all_values = []
    for axis, task in zip(axes, task_order, strict=True):
        for model_index, model in enumerate(model_order):
            key = model_keys[model]
            for method in methods:
                row = next(
                    (item for item in rows if item["task"] == task and item["left"] == method and key in item["model"].lower()),
                    None,
                )
                if row is None:
                    continue
                value = row["paired_success_diff"] * 100
                low = row["paired_success_ci_low"] * 100
                high = row["paired_success_ci_high"] * 100
                all_values.extend([low, high])
                axis.errorbar(
                    value,
                    model_index + offsets[method],
                    xerr=[[value - low], [high - value]],
                    fmt="o",
                    capsize=3,
                    color=PLOT_METHOD_COLORS[method],
                    label=PLOT_METHOD_LABELS[method] if method not in plotted else "_nolegend_",
                )
                plotted.add(method)
        axis.axvline(0, color="black", linewidth=0.9, alpha=0.65)
        axis.set_title(task_labels[task])
        axis.set_yticks(range(len(model_order)), model_order)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    if all_values:
        bound = max(5, float(np.ceil(max(abs(value) for value in all_values) / 5) * 5))
        axes[0].set_xlim(-bound, bound)
    axes[0].set_ylabel("Model")
    axes[1].set_xlabel("Success-rate difference versus logit (percentage points)")
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=PLOT_METHOD_COLORS[method], label=("tuned-wiki-small-v0 (27B only)" if method == "tuned" else PLOT_METHOD_LABELS[method]))
        for method in methods
        if method in plotted
    ]
    legend_handles.insert(0, Line2D([0], [0], color="black", linewidth=0.9, label="logit lens (reference = 0)"))
    axes[-1].legend(handles=legend_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2)
    fig.suptitle("Paired per-item contrasts: positive values favor the named method")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_parameter_plot(
    path: Path,
    all_runs: dict[str, list[tuple[str, Path, dict]]],
    pile_runs: dict[str, list[tuple[str, Path, dict]]],
) -> None:
    """Plot effectiveness against model parameter count for each protocol."""
    parameter_count = {
        "SmolLM2-135M": 0.135,
        "Qwen3-0.6B": 0.6,
        "Qwen3.5-0.8B": 0.8,
        "Qwen3.5-4B": 4.0,
        "Qwen3.6-27B": 27.0,
    }
    model_order = list(parameter_count)
    methods = ["jlens", "logit", "tuned", "tuned_pile", "random"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    for axis, (key, spec) in zip(axes, CAUSAL_SPECS.items(), strict=True):
        for method in methods:
            points = []
            for index, model in enumerate(model_order):
                source_runs = pile_runs[key] if method == "tuned_pile" else all_runs[key]
                aggregate_method = "tuned" if method == "tuned_pile" else method
                data = next(
                    (
                        d
                        for m, p, d in source_runs
                        if m == model and any(row[0] == aggregate_method for row in d.get("aggregate", []))
                    ),
                    None,
                )
                interval = bootstrap_rate_ci(data, aggregate_method, seed=800 + index) if data else None
                if interval:
                    points.append((parameter_count[model], interval))
            if not points:
                continue
            x = np.asarray([point[0] for point in points])
            y = np.asarray([point[1][0] for point in points])
            low = np.asarray([point[1][0] - point[1][1] for point in points])
            high = np.asarray([point[1][2] - point[1][0] for point in points])
            axis.errorbar(
                x, y, yerr=[low, high], marker="o", linestyle="none", capsize=3, color=PLOT_METHOD_COLORS.get(method), label=PLOT_METHOD_LABELS.get(method, method)
            )
        axis.set_xscale("log")
        axis.set_xticks([0.135, 0.6, 0.8, 4, 27], ["135M", "0.6B", "0.8B", "4B", "27B"])
        axis.set_title(spec["label"])
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Improved conditions (%)")
    axes[-1].legend(frameon=False, fontsize="small")
    fig.suptitle("Improvement rate versus model size (points; 95% bootstrap CI)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_parameter_stat_plot(
    path: Path,
    all_runs: dict[str, list[tuple[str, Path, dict]]],
    pile_runs: dict[str, list[tuple[str, Path, dict]]],
    *,
    title: str,
    ylabel: str,
    statistic: str,
) -> None:
    """Plot a paired or absolute statistic against model size."""
    parameter_count = {
        "SmolLM2-135M": 0.135,
        "Qwen3-0.6B": 0.6,
        "Qwen3.5-0.8B": 0.8,
        "Qwen3.5-4B": 4.0,
        "Qwen3.6-27B": 27.0,
    }
    model_order = list(parameter_count)
    methods = ["jlens", "logit", "tuned", "tuned_pile"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False, constrained_layout=True)
    for axis, (key, spec) in zip(axes, CAUSAL_SPECS.items(), strict=True):
        for method in methods:
            points = []
            for index, model in enumerate(model_order):
                source_runs = pile_runs[key] if method == "tuned_pile" else all_runs[key]
                aggregate_method = "tuned" if method == "tuned_pile" else method
                data = next(
                    (
                        d
                        for m, p, d in source_runs
                        if m == model and any(row[0] == aggregate_method for row in d.get("aggregate", []))
                    ),
                    None,
                )
                if not data:
                    continue
                if statistic == "excess_random":
                    interval = bootstrap_rate_difference(data, aggregate_method, "random", seed=1200 + index)
                else:
                    interval = bootstrap_median_ci(data, aggregate_method, seed=1400 + index)
                if interval:
                    points.append((parameter_count[model], interval))
            if not points:
                continue
            x = np.asarray([point[0] for point in points])
            y = np.asarray([point[1][0] for point in points])
            low = np.asarray([point[1][0] - point[1][1] for point in points])
            high = np.asarray([point[1][2] - point[1][0] for point in points])
            axis.errorbar(
                    x, y, yerr=[low, high], marker="o", linestyle="none", capsize=3, color=PLOT_METHOD_COLORS.get(method), label=PLOT_METHOD_LABELS.get(method, method)
            )
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xscale("log")
        axis.set_xticks([0.135, 0.6, 0.8, 4, 27], ["135M", "0.6B", "0.8B", "4B", "27B"])
        axis.set_title(spec["label"])
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False, fontsize="small")
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def completed_fit_record(
    path: Path,
    *,
    model: str,
    method: str,
    hardware: str,
    device_vram_gib: float,
    billed_usd: float | None = None,
    source_suffix: str = "",
) -> dict | None:
    """Read a completed fit's measured timing/memory values from its artifact."""
    manifest_path = path / "fit_manifest.json"
    events_path = path / "run-events.jsonl"
    if not manifest_path.exists() or not events_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    complete = None
    for line in events_path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "complete":
            complete = event
    if complete is None:
        return None
    return {
        "model": model,
        "method": method,
        "fit_minutes": complete["elapsed_s"] / 60,
        "steps": manifest.get("steps"),
        "tokens_per_step": manifest.get("max_length"),
        "dtype": manifest.get("dtype", "not recorded"),
        "quantization": manifest.get("quantization", "not recorded"),
        "peak_vram_gib": complete.get("peak_allocated_gib", manifest.get("cuda_peak_allocated_gib")),
        "device_vram_gib": device_vram_gib,
        "billed_usd": billed_usd,
        "hardware": hardware,
        "source": f"{manifest_path.name} + {events_path.name}{source_suffix}",
    }


def precision_label(record: dict) -> str:
    """Make storage/compute precision explicit without collapsing them."""
    dtype = str(record.get("dtype", "not recorded")).lower()
    quant = str(record.get("quantization", "not recorded")).lower()
    dtype_text = {"bf16": "BF16 compute", "fp16": "FP16 compute", "fp32": "FP32 compute"}.get(dtype, dtype)
    quant_text = {"nf4": "NF4 4-bit base", "int8": "INT8 base", "none": "unquantized base"}.get(quant, quant)
    return f"{dtype_text}; {quant_text}"


def efficiency_records() -> list[dict]:
    """Return the resource measurements currently available in the checkout.

    Artifact sizes are measured from the files used by the report.  Fit-time
    records are deliberately limited to runs with structured timing/resource
    provenance from our actual RTX 3090 or Modal A100 runs.
    """
    artifact_paths = {
        ("SmolLM2-135M", "J-lens"): ROOT / "data/lenses/smollm2-135m-fit-204-lens.pt",
        ("SmolLM2-135M", "tuned-wiki-small-v0"): ROOT / "data/lenses/smollm2-135m-tuned-wikitext/params.pt",
        ("Qwen3-0.6B", "J-lens"): ROOT / "data/lenses/qwen3-0.6b-fit-204-lens.pt",
        ("Qwen3-0.6B", "tuned-wiki-small-v0"): ROOT / "data/lenses/qwen3-0.6b-tuned-wikitext/params.pt",
        ("Qwen3-0.6B", "tuned-pile-repro-v1"): ROOT / "data/lenses/qwen3-0.6b-tuned-pile-repro-v1/params.pt",
        ("Qwen3-1.7B", "tuned-pile-repro-v1"): ROOT / "data/lenses/qwen3-1.7b-tuned-pile-repro-v1/params.pt",
        ("Qwen3.5-0.8B", "J-lens"): ROOT / "data/lenses/qwen3.5-0.8b-fit-204-lens.pt",
        ("Qwen3.5-0.8B", "tuned-wiki-small-v0"): ROOT / "data/lenses/qwen3.5-0.8b-tuned-wikitext/params.pt",
        ("Qwen3.5-4B", "tuned-wiki-small-v0"): ROOT / "data/lenses/qwen3.5-4b-tuned-wikitext/params.pt",
        ("Qwen3.5-4B", "tuned-pile-repro-v1"): ROOT / "data/lenses/qwen3.5-4b-tuned-pile-repro-v1/params.pt",
        ("Qwen3.6-27B", "tuned-wiki-small-v0"): ROOT / "data/lenses/qwen3.6-27b-tuned-wikitext-nf4/params.pt",
    }
    records = []
    for (model, method), path in artifact_paths.items():
        if path.exists():
            records.append({"model": model, "method": method, "artifact_mb": path.stat().st_size / 1e6})
    # The larger J-lens artifacts were used by the causal runs from the
    # published Neuronpedia/Hugging Face repository, not copied into this
    # checkout.  Include their verified repository metadata so the efficiency
    # plots do not silently omit the larger model rows.
    records.extend([
        {
            "model": "Qwen3.5-4B", "method": "J-lens", "artifact_mb": 406332644 / 1e6,
            "source": "neuronpedia/jacobian-lens@qwen-n1000",
        },
        {
            "model": "Qwen3.6-27B", "method": "J-lens", "artifact_mb": 3303032772 / 1e6,
            "source": "neuronpedia/jacobian-lens@qwen-n1000",
        },
    ])
    # Structured run-events provenance for completed native tuned fits. Read
    # timing, step count, and peak allocation from the artifacts themselves so
    # future Pile cells appear automatically after they finish.
    for record in (
        completed_fit_record(
            ROOT / "data/lenses/qwen3-0.6b-tuned-pile-repro-v1",
            model="Qwen3-0.6B", method="tuned-pile-repro-v1", hardware="RTX 3090",
            device_vram_gib=24.0,
        ),
        completed_fit_record(
            ROOT / "data/lenses/smollm2-135m-tuned-pile-repro-v1",
            model="SmolLM2-135M", method="tuned-pile-repro-v1", hardware="RTX 3090",
            device_vram_gib=24.0,
        ),
        completed_fit_record(
            ROOT / "data/lenses/qwen3-1.7b-tuned-pile-repro-v1",
            model="Qwen3-1.7B", method="tuned-pile-repro-v1", hardware="RTX 3090",
            device_vram_gib=24.0,
        ),
        completed_fit_record(
            ROOT / "data/lenses/qwen3.5-4b-tuned-pile-repro-v1",
            model="Qwen3.5-4B", method="tuned-pile-repro-v1", hardware="RTX 3090",
            device_vram_gib=24.0,
        ),
        completed_fit_record(
            ROOT / "data/lenses/qwen3.6-27b-tuned-wikitext-nf4",
            model="Qwen3.6-27B", method="tuned-wiki-small-v0", hardware="A100-SXM4-80GB",
            device_vram_gib=80.0, billed_usd=0.24579145, source_suffix=" + Modal billing",
        ),
    ):
        if record is not None:
            records.append(record)
    return records


def save_efficiency_plots(plot_dir: Path, all_runs: dict[str, list[tuple[str, Path, dict]]]) -> list[dict]:
    """Return measured creation-cost records for the primary efficiency plot."""
    del plot_dir, all_runs
    return efficiency_records()


def save_creation_scaling_plot(plot_dir: Path, records: list[dict]) -> None:
    """Plot creation cost against model scale, with sparse data explicit."""
    parameter_count = {
        "SmolLM2-135M": 0.135,
        "Qwen3-0.6B": 0.6,
        "Qwen3-1.7B": 1.7,
        "Qwen3.5-0.8B": 0.8,
        "Qwen3.5-4B": 4.0,
        "Qwen3.6-27B": 27.0,
    }
    timed = [row for row in records if "fit_minutes" in row]
    colors = {"J-lens": "#0173b2", "tuned-wiki-small-v0": "#de8f05", "tuned-pile-repro-v1": "#029e73"}
    markers = {"J-lens": "o", "tuned-wiki-small-v0": "s", "tuned-pile-repro-v1": "D"}
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)
    for axis, metric, ylabel in (
        (axes[0], "fit_minutes", "Total fit time (minutes, log scale)"),
        (axes[1], "native_rate", "Minutes per 1,000 native workload units (log scale)"),
        (axes[2], "memory_fraction", "Peak VRAM / device VRAM (fraction, log scale)"),
    ):
        for method in ["J-lens", "tuned-wiki-small-v0", "tuned-pile-repro-v1"]:
            rows = []
            for row in timed:
                if row["method"] != method:
                    continue
                if metric == "native_rate":
                    if row["method"] == "J-lens" and row.get("prompts") and row.get("source_layers"):
                        units = row["prompts"] * row["source_layers"]
                    elif row.get("steps"):
                        units = row["steps"]
                    else:
                        continue
                    row = {**row, "native_rate": row["fit_minutes"] * 1000 / units}
                elif metric == "memory_fraction":
                    if "peak_vram_gib" not in row or "device_vram_gib" not in row:
                        continue
                    row = {**row, "memory_fraction": row["peak_vram_gib"] / row["device_vram_gib"]}
                if metric in row:
                    rows.append(row)
            if not rows:
                continue
            axis.scatter(
                [parameter_count[row["model"]] for row in rows],
                [row[metric] for row in rows],
                s=85,
                color=colors[method],
                marker=markers[method],
                label=method,
            )
            for row in rows:
                annotation = [row["model"], row.get("hardware", "hardware n/a")]
                if row.get("steps") is not None:
                    annotation.append(f"{row['steps']:,} steps")
                elif row.get("prompts") is not None:
                    annotation.append(f"{row['prompts']:,} prompts")
                else:
                    annotation.append("fit units n/a")
                axis.annotate(
                    "\n".join(annotation),
                    (parameter_count[row["model"]], row[metric]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks([0.135, 0.6, 0.8, 1.7, 4, 27], ["135M", "0.6B", "0.8B", "1.7B", "4B", "27B"])
        axis.set_xlabel("Base-model parameters (log scale)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
    axes[0].set_title("Raw elapsed time")
    axes[1].set_title("Method-native normalized time")
    axes[2].set_title("Normalized memory pressure")
    axes[0].legend(frameon=False, fontsize="small")
    fig.suptitle("Creation resource scaling versus model size (measured runs; not dollar cost)")
    fig.savefig(plot_dir / "efficiency_creation_scaling.png", dpi=150)
    plt.close(fig)


def creation_provenance_rows(records: list[dict]) -> list[list[str]]:
    """Render completed creation measurements without artifact-only rows."""
    rows = []
    for record in records:
        if record.get("fit_minutes") is None:
            continue
        minutes = record.get("fit_minutes")
        peak = record.get("peak_vram_gib")
        device = record.get("device_vram_gib")
        rows.append([
            record["model"],
            record["method"],
            f"{minutes:.1f}" if minutes is not None else "—",
            f"{record['steps']:,}" if record.get("steps") is not None else "—",
            f"{record['prompts']:,}" if record.get("prompts") is not None else "—",
            f"{peak:.2f}" if peak is not None else "—",
            f"{device:.0f}" if device is not None else "—",
            f"{minutes / 60:.3f}" if minutes is not None else "—",
            f"{peak * minutes / 60:.2f}" if peak is not None and minutes is not None else "—",
            f"{100 * peak / device:.1f}%" if peak is not None and device is not None else "—",
            f"${record['billed_usd']:.4f}" if record.get("billed_usd") is not None else "not priced",
            precision_label(record),
            record.get("hardware", record.get("source", "local artifact file")),
        ])
    return rows


def native_efficiency_rows(records: list[dict]) -> list[list[str]]:
    """Report measured rates and explicitly show unmeasured J-lens telemetry."""
    rows = []
    for record in records:
        if "fit_minutes" not in record:
            continue
        if record["method"] == "J-lens" and record.get("prompts") and record.get("source_layers"):
            units = record["prompts"] * record["source_layers"]
            workload = f"{record['prompts']:,} prompts × {record['source_layers']} layers"
            rate = record["fit_minutes"] * 1000 / units
            unit = "minutes / 1,000 prompt-layers"
        elif record.get("steps"):
            units = record["steps"]
            tokens = record.get("tokens_per_step")
            workload = f"{record['steps']:,} optimizer steps"
            if tokens:
                workload += f" × {tokens} tokens"
            rate = record["fit_minutes"] * 1000 / units
            unit = "minutes / 1,000 optimizer steps"
        else:
            workload, rate, unit = "fit workload not recorded", None, "—"
        rows.append([
            record["model"], record["method"], workload,
            f"{record['fit_minutes']:.1f}",
            f"{rate:.2f}" if rate is not None else "—", unit,
            precision_label(record),
            record.get("hardware", record.get("source", "—")),
        ])
    rows.append([
        "Qwen3.6-27B", "J-lens", "fit workload not recorded", "—", "—", "—",
        "not measured", "RTX 3090 (not run); remote artifact used for causal plots",
    ])
    rows.append(["All models", "logit lens", "no learned artifact (by design)", "0", "0", "minutes", "not applicable", "analytical control"])
    return rows


def headline_27b_efficiency_rows(records: list[dict]) -> list[list[str]]:
    """Render the 27B creation evidence without implying matched costs."""
    rows = []
    for record in records:
        if record.get("model") != "Qwen3.6-27B" or "fit_minutes" not in record:
            continue
        if record["method"] == "J-lens":
            workload = f"{record['prompts']} prompts × {record['source_layers']} layers"
            native = f"{record['fit_minutes'] * 1000 / (record['prompts'] * record['source_layers']):.1f} min / 1k prompt-layers"
            comparison = "External M4 benchmark; no completed local 3090 J-lens fit telemetry"
        else:
            workload = f"{record.get('steps', '—'):,} optimizer steps × {record.get('tokens_per_step', '—')} tokens"
            native = f"{record['fit_minutes'] * 1000 / record['steps']:.1f} min / 1k steps"
            comparison = "Not matched: 100-step Wikitext pilot; no 27B Pile fit"
        rows.append([
            record["method"], workload, record.get("hardware", "—"),
            f"{record['fit_minutes']:.1f}", native,
            f"{record['peak_vram_gib']:.2f}" if record.get("peak_vram_gib") is not None else "—",
            f"${record['billed_usd']:.4f}" if record.get("billed_usd") is not None else "not priced",
            precision_label(record),
            comparison,
        ])
    rows.append([
        "J-lens", "27B local fit not measured", "RTX 3090 (not run)", "—", "—",
        "—", "not priced", "not measured", "Prefit remote artifact; creation telemetry unavailable",
    ])
    rows.append([
        "logit lens", "no learned artifact (by design)", "—", "0", "0 min",
        "0", "not priced", "not applicable", "Analytical baseline; no fitting step exists",
    ])
    rows.append([
        "tuned-pile-repro-v1", "27B Pile fit not yet available", "—", "—", "—",
        "—", "—", "not yet run", "Pending local/remote capacity run",
    ])
    return rows


def runtime_environment_rows() -> list[list[str]]:
    """Keep hardware capabilities separate from our measured run records."""
    return [
        [
            "RTX 3090",
            "local measured environment",
            "24 GiB GDDR6X",
            "10,496 CUDA cores; ~35.6 FP32 TFLOPS",
            "~936 GB/s",
            "BF16 compute; unquantized small-model fits",
            "Our local machine; nvidia-smi confirms 24 GiB",
        ],
        [
            "A100 SXM4 80GB",
            "Modal measured environment",
            "80 GiB HBM2e",
            "6,912 CUDA cores; 19.5 FP32 / 312 BF16 tensor TFLOPS",
            "2,039 GB/s",
            "BF16 compute; NF4 base for 27B tuned pilot",
            "NVIDIA specifications; Modal billing/run telemetry",
        ],
        [
            "Apple M4 Pro",
            "external upstream reference only",
            "up to 64 GiB unified memory (not dedicated VRAM)",
            "up to 20-core GPU; ML TFLOPS not published",
            "273 GB/s",
            "not reported",
            "Apple specifications; 164.7-min J-lens reference is not our run and is excluded from measured plots",
        ],
    ]


def memory_evidence_rows() -> list[list[str]]:
    """Summarize memory evidence without treating missing data as zero."""
    return [
        [
            "J-lens",
            "Qwen3.6-27B",
            "not measured locally",
            "not recorded",
            "not measured",
            "A model-matched remote J-lens artifact was used for causal evaluation; its creation memory was not recorded in our run",
        ],
        [
            "tuned lens",
            "Qwen3.6-27B Wikitext pilot",
            "34.85 GiB",
            "BF16 compute; NF4 4-bit base",
            "80 GiB A100",
            "Successful Modal fit; local 3090 attempt OOMed during backward",
        ],
        [
            "tuned lens",
            "Qwen3-0.6B Pile fit",
            "2.04 GiB",
            "BF16 compute; unquantized base",
            "24 GiB RTX 3090",
            "Completed local fit telemetry",
        ],
        [
            "logit lens",
            "any model",
            "not a fitted artifact",
            "not applicable",
            "base-model inference memory",
            "Zero creation/training cost; it still uses runtime memory for the model, hidden states, and unembedding",
        ],
    ]


def build_report(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    report_version = int(datetime.now().timestamp())
    sns.set_theme(style="whitegrid", context="notebook")
    sns.set_palette("colorblind")

    all_runs: dict[str, list[tuple[str, Path, dict]]] = {
        key: causal_runs(spec) for key, spec in CAUSAL_SPECS.items()
    }
    pile_runs: dict[str, list[tuple[str, Path, dict]]] = {
        key: pile_causal_runs(spec) for key, spec in CAUSAL_SPECS.items()
    }

    # 27B headline plot with prompt/item-cluster bootstrap intervals.
    experiment_labels = [spec["label"] for spec in CAUSAL_SPECS.values()]
    methods = ["jlens", "logit", "tuned", "random"]
    headline_stats = {method: [] for method in methods}
    for method in methods:
        for key in CAUSAL_SPECS:
            data = next((d for m, p, d in all_runs[key] if m == "Qwen3.6-27B"), None)
            headline_stats[method].append(bootstrap_rate_ci(data, method, seed=100 + len(headline_stats[method])) if data else None)
    save_ci_plot(
        plot_dir / "27b_improvement_rate.png",
        "Qwen3.6-27B: fraction of successful causal swaps (95% bootstrap CI)",
        headline_stats,
        experiment_labels,
        "Improved conditions (%)",
    )

    # Cross-model comparisons, including tuned wherever a tuned artifact exists.
    for key in ("verbal", "multihop", "flexible"):
        model_order = ["SmolLM2-135M", "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3.5-0.8B", "Qwen3.5-4B", "Qwen3.6-27B"]
        plot_methods = ["jlens", "logit", "tuned", "tuned_pile", "random"]
        stats = {method: [] for method in plot_methods}
        for method in plot_methods:
            for model in model_order:
                if method == "tuned_pile":
                    source_runs = pile_runs[key]
                    aggregate_method = "tuned"
                else:
                    source_runs = all_runs[key]
                    aggregate_method = method
                data = next(
                    (
                        d
                        for m, p, d in source_runs
                        if m == model
                        and any(row[0] == aggregate_method for row in d.get("aggregate", []))
                    ),
                    None,
                )
                stats[method].append(bootstrap_rate_ci(data, aggregate_method, seed=200 + len(stats[method])) if data else None)
        save_ci_plot(
            plot_dir / f"{key}_cross_model.png",
            f"{CAUSAL_SPECS[key]['label']}: causal improvement rate (95% bootstrap CI)",
            stats,
            model_order,
            "Improved conditions (%)",
            method_labels={"tuned_pile": "tuned-pile-repro-v1"},
        )
    save_parameter_plot(plot_dir / "parameter_effectiveness.png", all_runs, pile_runs)
    save_parameter_stat_plot(
        plot_dir / "parameter_excess_random.png",
        all_runs,
        pile_runs,
        title="Improvement above random versus model size (95% bootstrap CI)",
        ylabel="Improvement-rate difference (percentage points)",
        statistic="excess_random",
    )
    save_parameter_stat_plot(
        plot_dir / "parameter_median_delta.png",
        all_runs,
        pile_runs,
        title="Median rank change versus model size (95% bootstrap CI)",
        ylabel="Median rank improvement",
        statistic="median",
    )
    save_paired_contrast_plot(plot_dir / "paired_causal_contrasts.png")
    efficiency_records_current = save_efficiency_plots(plot_dir, all_runs)
    save_creation_scaling_plot(plot_dir, efficiency_records_current)

    method_rows = []
    for key, spec in CAUSAL_SPECS.items():
        for model, _path, data in all_runs[key]:
            if model not in {"SmolLM2-135M", "Qwen3-0.6B", "Qwen3.5-0.8B", "Qwen3.5-4B", "Qwen3.6-27B"}:
                continue
            for row in data["aggregate"]:
                method_rows.append(
                    [
                        spec["label"],
                        model,
                        PLOT_METHOD_LABELS.get(row[0], row[0]),
                        row[1],
                        row[2],
                        row[3],
                        " / ".join(fmt(value) for value in row[4:]),
                    ]
                )

    headline_rows = []
    for key, spec in CAUSAL_SPECS.items():
        data = next((d for m, p, d in all_runs[key] if m == "Qwen3.6-27B"), None)
        for row in (data or {}).get("aggregate", []):
            headline_rows.append([spec["label"], PLOT_METHOD_LABELS.get(row[0], row[0]), *row[1:], f"{100 * row[2] / row[1]:.1f}%"])

    bootstrap_rows = []
    for experiment_index, spec in enumerate(CAUSAL_SPECS.values()):
        for method in methods:
            interval = headline_stats[method][experiment_index]
            if interval:
                bootstrap_rows.append([spec["label"], PLOT_METHOD_LABELS.get(method, method), f"{interval[0]:.1f}%", f"{interval[1]:.1f}%", f"{interval[2]:.1f}%"])

    contrast_rows = []
    for key, spec in CAUSAL_SPECS.items():
        for model, _path, data in all_runs[key]:
            for method in ("jlens", "tuned"):
                if not any(row[0] == method for row in data.get("aggregate", [])):
                    continue
                contrast = bootstrap_rate_difference(data, method, "random", seed=500 + len(contrast_rows))
                if contrast:
                    contrast_rows.append([spec["label"], model, PLOT_METHOD_LABELS.get(method, method), f"{contrast[0]:+.1f} pp", f"{contrast[1]:+.1f} pp", f"{contrast[2]:+.1f} pp"])

    documentation_rows = render_documentation(out_dir)
    task_examples = causal_task_examples()
    documentation_links = "<ul>" + "".join(
        f"<li><a href='{html.escape(href)}'>{html.escape(title)}</a> <span class='muted'>({html.escape(relative)})</span></li>"
        for title, href, relative in documentation_rows
    ) + "</ul>"

    def plot_figure(filename: str, anchor: str, alt: str, title: str) -> str:
        """Render a plot with a stable, shareable URL fragment."""
        return (
            f"<figure id='{anchor}' class='report-figure'>"
            f"<figcaption><strong>{html.escape(title)}</strong> "
            f"<a class='figure-link' href='#{anchor}' aria-label='Link to {html.escape(title)}'>"
            f"[#{anchor}]</a></figcaption>"
            f"<img class='plot' src='plots/{filename}?v={report_version}' alt='{html.escape(alt)}'>"
            "</figure>"
        )

    html_parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<meta name=viewport content='width=device-width, initial-scale=1'>",
        "<title>J-Lens research report</title>",
        """<style>
        :root { --paper:#fff; --wash:#f3f1ec; --ink:#222; --muted:#666; --rule:#b8b5ae; --accent:#174a73; }
        * { box-sizing:border-box; } body { margin:0; background:var(--wash); color:var(--ink); font:16px/1.58 Georgia, 'Times New Roman', serif; }
        main { max-width:1120px; margin:2rem auto; padding:3.5rem 4.5rem; background:var(--paper); border:1px solid #d7d3ca; box-shadow:0 1px 8px #00000012; }
        h1,h2,h3,summary,.sans,table,footer { font-family:Arial, Helvetica, sans-serif; } h1 { font:700 2.35rem/1.12 Georgia,serif; letter-spacing:-.02em; margin:0 0 .35rem; } h2 { margin:2.8rem 0 .7rem; padding-bottom:.25rem; border-bottom:1px solid var(--rule); font-size:1.28rem; letter-spacing:.01em; } h3 { margin:1.6rem 0 .55rem; font-size:1.02rem; }
        .kicker { font:700 .76rem/1 Arial,sans-serif; letter-spacing:.14em; text-transform:uppercase; color:var(--accent); } .lede { font-size:1.08rem; } .abstract { border-top:2px solid var(--ink); border-bottom:1px solid var(--rule); padding:1rem 0; } .callout { border-left:3px solid var(--accent); padding:.1rem 1rem; margin:1rem 0; } .muted { color:var(--muted); }
        .toc { border:1px solid var(--rule); padding:.8rem 1.1rem; margin:1.6rem 0; } .toc ol { margin:.4rem 0 0; columns:2; } a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
        details { margin:1rem 0 1.4rem; } summary { cursor:pointer; color:var(--accent); font-weight:700; } summary::marker { color:var(--ink); } table { border-collapse:collapse; width:100%; margin:.8rem 0 1.4rem; font-size:.82rem; } th,td { border-bottom:1px solid #d6d3cc; padding:.4rem .5rem; text-align:left; vertical-align:top; } th { border-top:1px solid var(--ink); border-bottom:1px solid var(--ink); font-weight:700; } tr:nth-child(even) { background:#faf9f6; }
        .report-figure { margin:1.5rem 0 2.2rem; scroll-margin-top:1rem; } .report-figure figcaption { font:700 .92rem/1.3 Arial,Helvetica,sans-serif; margin-bottom:.45rem; } .figure-link { font-weight:400; color:var(--muted); } .plot { display:block; background:#fff; width:100%; max-width:100%; margin:0; border:1px solid #d6d3cc; } code { font: .88em/1.3 'SFMono-Regular',Consolas,monospace; } ul { padding-left:1.4rem; } footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--rule); color:var(--muted); font-size:.8rem; }
        @media (max-width:760px) { main { margin:0; padding:2rem 1.2rem; } .toc ol { columns:1; } table { display:block; overflow-x:auto; } }
        @media print { body { background:#fff; } main { max-width:none; margin:0; padding:0; border:0; box-shadow:none; } details { display:block; break-inside:avoid; } details > :not(summary) { display:block !important; } details > summary { display:none; } a { color:inherit; text-decoration:underline; } }
        </style></head><body><main>""",
        "<div class='kicker'>Research report · Jacobian lens</div>",
        f"<h1>Representational readout and causal transport in language models</h1><p class='muted sans'>Generated {datetime.now(ZoneInfo('America/New_York')).date().isoformat()} from recorded JSON artifacts.</p>",
        "<section class='abstract' id='summary'><p><strong>Abstract.</strong> We reproduced the local Jacobian-lens readout path and extended it into controlled causal swap experiments across decoder models from 135M to 27B parameters. The model-matched 27B JLens produced stronger target-rank movement than logit-lens and random matched controls in the current verbal-report, two-hop, and flexible-generalization runs. Existing tuned results are explicitly labelled <code>tuned-wiki-small-v0</code>: model-matched Wikitext pilots, not a strong reproduction.</p><p><strong>Current status:</strong> model-matched <code>tuned-pile-repro-v1</code> predictive and causal results are complete for SmolLM2-135M, Qwen3-0.6B, and Qwen3.5-4B. Qwen3-1.7B fitting is complete and its canonical 16.4M-token held-out gate is still running; its short-2M causal suite remains explicitly provisional. No provisional result is pooled into the validated Pile causal table.</p></section>",
        "<nav class='toc' aria-label='Table of contents'><strong class='sans'>Contents</strong><ol><li><a href='#summary'>Summary and current claim</a></li><li><a href='#coverage'>Model and intervention coverage</a></li><li><a href='#interventions'>Intervention definitions</a></li><li><a href='#measurements'>Measurements and estimands</a></li><li><a href='#results-27b'>27B results and uncertainty</a></li><li><a href='#cross-model'>Cross-model comparisons</a></li><li><a href='#paired-contrasts'>Paired per-item contrasts</a></li><li><a href='#scaling'>Effectiveness versus model size</a></li><li><a href='#efficiency'>Creation cost and scaling</a></li><li><a href='#pile-validation'>Held-out Pile validation</a></li><li><a href='#interpretation'>Interpretation and open questions</a></li><li><a href='#provenance'>Sources and provenance</a></li><li><a href='#documentation'>Documentation and logs</a></li></ol></nav>",
        "<h2 id='coverage'>Model and intervention coverage</h2>",
        fold("Show coverage matrix", coverage_table()),
        """<h3>Model catalog</h3>
        <p>This catalog describes the subjects, not the interventions. “Base/pretrained” means the checkpoint name does not identify an Instruct variant; it is not a claim that the model lacks reasoning behavior. Precision describes the recorded run, and can differ between fit, evaluation, and causal inference.</p>
        <table><thead><tr><th>Subject</th><th>Exact checkpoint ID</th><th>Parameters</th><th>Family</th><th>Checkpoint flavor</th><th>Architecture</th><th>Recorded run precision</th></tr></thead><tbody>
        <tr><td>SmolLM2-135M</td><td><code>HuggingFaceTB/SmolLM2-135M-Instruct</code></td><td>135M</td><td>SmolLM2</td><td>Instruction-tuned</td><td>Decoder-only Transformer</td><td>BF16 compute; unquantized base on RTX 3090</td></tr>
        <tr><td>Qwen3-0.6B</td><td><code>Qwen/Qwen3-0.6B</code></td><td>0.6B</td><td>Qwen3</td><td>Base/pretrained checkpoint</td><td>Decoder-only Transformer</td><td>BF16 compute; unquantized base on RTX 3090</td></tr>
        <tr><td>Qwen3-1.7B</td><td><code>Qwen/Qwen3-1.7B</code></td><td>1.7B</td><td>Qwen3</td><td>Base/pretrained checkpoint</td><td>Decoder-only Transformer</td><td>BF16 compute; unquantized base on RTX 3090</td></tr>
        <tr><td>Qwen3.5-0.8B</td><td><code>Qwen/Qwen3.5-0.8B</code></td><td>0.8B</td><td>Qwen3.5</td><td>Base/pretrained checkpoint</td><td>Decoder-only Transformer</td><td>BF16 compute; unquantized local fit</td></tr>
        <tr><td>Qwen3.5-4B</td><td><code>Qwen/Qwen3.5-4B</code></td><td>4B</td><td>Qwen3.5</td><td>Base/pretrained checkpoint</td><td>Decoder-only Transformer</td><td>BF16 compute; unquantized local fit</td></tr>
        <tr><td>Qwen3.6-27B</td><td><code>Qwen/Qwen3.6-27B</code></td><td>27B</td><td>Qwen3.6 / Qwen3.5-family</td><td>Base/pretrained checkpoint</td><td>Hybrid GDN/attention decoder</td><td>BF16 compute; NF4 4-bit frozen base for the Modal tuned fit and 27B causal runs</td></tr>
        </tbody></table>""",
        """<h2 id='interventions'>Intervention definitions</h2>
        <p>All four methods use the same causal protocol: choose a source and
        target concept, construct their residual-stream directions at the
        selected layer, swap the two coordinates with a two-vector
        pseudoinverse frame, and measure the target answer's rank after the
        model continues forward. For the 27B runs, the intervention layers are
        16, 32, and 60, and the patch is applied at every prompt position.</p>
        <table><thead><tr><th>Method</th><th>Direction used</th><th>What is learned?</th><th>Control/intervention meaning</th></tr></thead>
        <tbody>
        <tr><td>J-lens</td><td>J<sub>layer</sub><sup>T</sup> times the model's unembedding row for the token</td><td>A model- and layer-specific Jacobian lens fitted from prompts</td><td>Swap coordinates in the learned global-workspace/readout basis</td></tr>
        <tr><td>Logit lens</td><td>The raw unembedding row for the token; equivalent to J = I</td><td>Nothing</td><td>Same coordinate-swap machinery, using ordinary logit directions</td></tr>
        <tr><td><code>tuned-wiki-small-v0</code></td><td>(I + W<sup>T</sup>) times the unembedding row, where W is the learned affine translator</td><td>Model- and layer-specific affine translators trained by KL-to-final-logits on short Wikitext-2 fits</td><td>Current tuned causal rows; exported translator-basis swap, with nonlinear final RMSNorm not folded into this patch</td></tr>
        <tr><td><code>tuned-pile-repro-v1</code></td><td>(I + W<sup>T</sup>) times the unembedding row, where W is the learned affine translator</td><td>Same translator objective, fit on Pile validation and accepted only after held-out Pile-test validation</td><td>Validated causal rows pending; Qwen3-1.7B has a separately labelled short-2M provisional track</td></tr>
        <tr><td>Random matched</td><td>A random residual direction scaled to the norm of the corresponding coordinate-swap delta</td><td>Nothing</td><td>Controls for generic perturbation magnitude rather than semantic direction</td></tr>
        </tbody></table>
        <p class='muted'><strong>How to read the random control:</strong> it is not expected to have zero successes. A norm-matched random perturbation can move a target token up or down by chance; its success rate is the measured noise floor for this rank-based metric. A method should therefore be judged by its excess over random, not by raw improvement alone. If random approaches the semantic method, the intervention may be exploiting generic residual sensitivity rather than the intended concept direction.</p>
        <h3>Tuned-lens training-data scale</h3>
        <table><thead><tr><th>Variant</th><th>Corpus</th><th>Fit material</th><th>Optimization</th><th>Held-out evaluation</th><th>Status</th></tr></thead>
        <tbody>
        <tr><td><code>tuned-wiki-small-v0</code></td><td>Wikitext-2 raw train</td><td>32,768 sampled tokens for the smaller models; 65,536 for the 27B pilot</td><td>100 steps, 128-token chunks, learning rate 1e-3</td><td>Not part of the original pilot fit</td><td>Used in current causal plots; sanity-check only</td></tr>
        <tr><td><code>tuned-pile-repro-v1</code></td><td>The Pile validation split</td><td>16,384 chunks per model; 2,097,152 sampled training tokens</td><td>4,096 steps, 128-token chunks, learning rate 1e-3</td><td>Pile test, target 16.4M tokens</td><td>Full predictive/causal track complete for SmolLM2, Qwen3-0.6B, Qwen3.5-0.8B, and Qwen3.5-4B; Qwen3-1.7B full gate in progress</td></tr>
        </tbody></table>
        <p class='muted'>These are materially different training regimes. The Pile first pass exposes the translators to roughly 32–64 times more token positions than the Wikitext pilots, and it has a genuinely held-out test evaluation. Therefore a future Pile causal result must be compared against the corresponding model's Pile-trained lens, not silently pooled with the Wikitext rows.</p>
        <p class='muted'>The 27B released J-lens is a separate model-matched
        1,000-prompt artifact. The new 27B tuned lens was fit separately using
        512 Wikitext chunks of length 128 for 100 optimization steps and is
        labelled <code>tuned-wiki-small-v0</code>. The stronger
        <code>tuned-pile-repro-v1</code> track is now being staged, starting with
        Qwen3-0.6B. Logit and random are not trained lenses. Thus the causal comparison holds the
        model, prompt fixture, patch positions, layers, and swap rule fixed
        while changing the direction basis.</p>""",
        "<h2 id='measurements'>Measurements and estimands</h2>",
        "<p>The plots repeat the key labels, but the estimands are defined here once so the reader can interpret every task consistently. Unless noted otherwise, each scored observation is one valid prompt/item at one intervention layer; multiple layers from the same prompt are retained together for uncertainty estimation.</p>",
        "<table><thead><tr><th>Measurement</th><th>Definition</th><th>Interpretation</th></tr></thead><tbody>",
        "<tr><td>n</td><td>Number of scored item–layer observations after tokenization and validity filtering.</td><td>Denominator for the task's aggregate rate; it is not always the number of unique prompts.</td></tr>",
        "<tr><td>Improved conditions (%)</td><td>100 × fraction where target rank before − target rank after is positive.</td><td>How often the intervention moves the intended target upward; positive does not require top-1.</td></tr>",
        "<tr><td>Δrank</td><td>Target rank before − target rank after.</td><td>Positive is improvement; the median is the primary robust effect-size summary, while the mean is more sensitive to outliers.</td></tr>",
        "<tr><td>Target top-1 / top-5 / top-10</td><td>Fraction of post-intervention observations where the intended target has the indicated rank threshold.</td><td>Absolute endpoint performance after intervention, distinct from rank movement.</td></tr>",
        "<tr><td>Excess over random</td><td>Method improvement rate minus the matched-random improvement rate on the same task/model.</td><td>Estimated direction-specific effect above generic norm-matched perturbation sensitivity.</td></tr>",
        "<tr><td>95% bootstrap interval</td><td>Cluster bootstrap over prompts/items, keeping their scored layers together.</td><td>Uncertainty over examples, not an assumption that layers are independent samples.</td></tr>",
        "</tbody></table>",
        "<p class='muted'>The verbal-report task uses target top-10; the two-hop and flexible tasks use target top-5. Random matched controls are not expected to have zero improved conditions: they establish the noise floor for this rank-based metric.</p>",
        "<h3>Are two-hop reasoning and flexible generalization actually distinct?</h3>",
        "<p>At the level of the current fixtures, the distinction is weaker than the task names suggest. Both intervene on an argument-like representation and ask whether a downstream token changes appropriately. The two-hop prompt adds a nested relation; the flexible prompt makes the relation look like a named function. That may be a meaningful compositional difference, but it is not yet a strong measurement separation.</p>",
        fold("Show across-model task correlations", task_discriminability_table(all_runs)),
        "<p class='muted'>The correlation table uses the six shared models and aggregate improved-condition rates, so it is descriptive and underpowered rather than a formal validation of task independence. A high positive correlation means the tasks may be tracking a shared argument-substitution sensitivity; it does not prove that they are identical.</p>",
        "<table><thead><tr><th>Current fixture</th><th>Swap</th><th>Expected shift</th><th>Operational interpretation</th></tr></thead><tbody>",
        "<tr><td>The capital of France is the city of</td><td>France → Canada</td><td>Paris → Ottawa</td><td>Recompute a one-argument function after an entity substitution; current flexible-generalization fixture.</td></tr>",
        "<tr><td>Fact: The language spoken in the country where the Amazon River ends is</td><td>Brazil → Mexico</td><td>Portuguese → Spanish</td><td>Carry an intermediate entity through a nested relation; current two-hop fixture.</td></tr>",
        "</tbody></table>",
        "<p class='callout'><strong>Interpretive consequence:</strong> we should not present the current two-hop and flexible suites as independent evidence without qualification. The next stronger version should hold the surface operation constant while varying whether the answer requires one substituted argument or a genuinely necessary intermediate chain, and should include matched controls that distinguish direct lookup from composition.</p>",
        "<h2 id='results-27b'>Qwen3.6-27B headline results</h2>",
        "<p class='muted'><strong>Verbal report task:</strong> replace one entity in a factual prompt and test whether the model's next answer or continuation shifts toward the substituted entity. The headline plot summarizes the fraction of causal swap conditions where the target answer's rank improved.</p>",
        fold("Show headline aggregate table", table(["Experiment", "Method", "n", "Improved", "Median Δrank", "Mean Δrank", "Top-1", "Top-k", "Improvement rate"], headline_rows)),
        plot_figure("27b_improvement_rate.png", "fig-27b-improvement-rate", "27B improvement rates", "Figure: 27B headline improvement rates"),
        "<h3>27B bootstrap uncertainty</h3>",
        fold("Show 27B bootstrap intervals", table(["Experiment", "Method", "Observed rate", "95% CI low", "95% CI high"], bootstrap_rows)),
        "<p class='muted'>Intervals use a deterministic cluster bootstrap over prompts/items, resampling each prompt as a unit while retaining its multiple scored layers together. This is more conservative than treating every layer as independent.</p>",
        "<h3>Method versus random control</h3>",
        fold("Show paired method-versus-random contrasts", table(["Experiment", "Model", "Method", "Observed difference", "95% CI low", "95% CI high"], contrast_rows)),
        "<p class='muted'>Differences are paired success-rate contrasts in percentage points: JLens or tuned lens minus the random matched control. A positive interval entirely above zero is the clearest evidence in these pilot metrics that the method beats random for that model and experiment.</p>",
        "<h2 id='cross-model'>Cross-model causal comparisons</h2>",
        "<p>These plots show the fraction of scored conditions in which the target improved after the intervention. Wikitext and full Pile tuned lenses are displayed as separate series; the earlier Qwen3-1.7B short-2M diagnostic is excluded until its full held-out Pile protocol is complete. Prompt counts and tokenization skips should still be inspected in the per-experiment logs.</p>",
        f"<h3>Verbal report</h3><p class='muted'>Task: ask the model to produce a category member, then intervene on the representation of the selected source concept and test whether the target candidate rises at the answer position.</p>{task_examples['verbal']}" + plot_figure("verbal_cross_model.png", "fig-verbal-cross-model", "Verbal-report cross-model improvement rates", "Figure: Verbal-report cross-model comparison"),
        f"<h3>Two-hop reasoning</h3><p class='muted'>Task: the prompt states two linked facts; the intervention swaps the representation of the intermediate entity, and success means the model's downstream answer moves toward the answer implied by that substituted intermediate.</p>{task_examples['multihop']}" + plot_figure("multihop_cross_model.png", "fig-multihop-cross-model", "Two-hop cross-model improvement rates", "Figure: Two-hop reasoning cross-model comparison"),
        f"<h3>Flexible generalization</h3><p class='muted'>Task: substitute one argument for another, then test whether the model correctly carries that replacement through several downstream functions such as ordering, comparison, arithmetic, or relational use.</p>{task_examples['flexible']}" + plot_figure("flexible_cross_model.png", "fig-flexible-cross-model", "Flexible generalization cross-model improvement rates", "Figure: Flexible-generalization cross-model comparison"),
        "<h2 id='paired-contrasts'>Paired per-item method contrasts</h2>",
        "<p>Aggregate bars answer how often each method succeeds overall; they do not show whether the same items improve under one method and fail under another. This analysis pairs methods on the same item and averages over that item's scored layers, then reports the difference in success rate, a prompt/item bootstrap interval, and a paired sign-flip p-value. Positive values favor the named method over logit lens. These are paired descriptive/inferential summaries, not independent-sample tests, and each saved artifact remains a separate row rather than being pooled across models or tasks.</p>",
        plot_figure("paired_causal_contrasts.png", "fig-paired-causal-contrasts", "Paired per-item success-rate contrasts versus logit lens", "Figure: Paired per-item contrasts versus logit lens"),
        "<p class='callout'><strong>Quick read:</strong> logit lens is the reference method and therefore appears as the vertical zero line, not as a separate point series. Points to the right mean the named method wins more same-item comparisons than logit lens; intervals crossing zero are inconclusive at this uncertainty level. The green tuned-lens point is 27B-only because no other model currently has a paired tuned-versus-logit artifact. The plot is most useful for seeing whether an apparent aggregate advantage is consistent across models and tasks, rather than driven by one large bar.</p>",
        fold("Show all J-lens/tuned-versus-logit paired contrasts", paired_causal_table()),
        "<p class='muted'>The corresponding machine-readable artifact is <code>data/analysis/paired_causal_effects.json</code>. Random-versus-logit controls remain in the 27B contrast table and the underlying artifact because their role is to establish the perturbation noise floor, not to replace the method-versus-method comparison.</p>",
        "<h2 id='scaling'>Effectiveness versus model size</h2>",
        "<p>The first view uses the intuitive absolute improvement rate. The second subtracts the matched random-control improvement rate, which is a better summary of technique-specific effect. The third shows median rank movement, capturing effect magnitude rather than only whether movement was positive. All use parameter count on a logarithmic x-axis and item-cluster bootstrap intervals.</p>",
        plot_figure("parameter_effectiveness.png", "fig-parameter-effectiveness", "Technique effectiveness versus model parameter count", "Figure: Absolute effectiveness versus model size"),
        plot_figure("parameter_excess_random.png", "fig-parameter-excess-random", "Improvement above random versus model parameter count", "Figure: Excess over random versus model size"),
        plot_figure("parameter_median_delta.png", "fig-parameter-median-delta", "Median rank change versus model parameter count", "Figure: Median rank movement versus model size"),
        "<h2 id='efficiency'>27B creation-cost evidence</h2>",
        "<p>The primary efficiency question is now stated at the target model scale: what did it take to create each 27B artifact? This is separate from causal coverage. The causal plots include the released model-matched J-lens artifact, logit lens, random matched controls, and the fitted Wikitext tuned lens; only the tuned Wikitext row currently has our 27B fit telemetry. Logit and random have no learned artifact by design, and the J-lens creation run was not recorded in our checkout. The raw times are therefore evidence, not a causal ranking. The method-native rates are also not interchangeable: J-lens uses prompt-layer evaluations, tuned lens uses optimizer steps, and logit lens has no fit. The broad model-scale plot is retained below as exploratory context only.</p>",
        fold("Show runtime environments", table(["Environment", "Role in this project", "Memory", "Compute reference", "Bandwidth", "Recorded precision", "Provenance"], runtime_environment_rows())),
        fold("Show 27B creation evidence", table(["Method", "Workload", "Hardware", "Fit min", "Native normalized rate", "Peak VRAM GiB", "Billed USD", "Precision", "Comparability"], headline_27b_efficiency_rows(efficiency_records_current))),
        "<h3>Memory evidence by intervention</h3>",
        "<p>Peak VRAM in GiB is a directly comparable resource measurement only when model, precision, batch size, and workload are specified. A missing value is not zero. Logit lens has zero learned-artifact creation cost, but it still consumes the base model's runtime memory.</p>",
        fold("Show memory evidence", table(["Intervention", "Model/run", "Peak memory", "Precision", "Device", "Interpretation"], memory_evidence_rows())),
        fold("Show exploratory model-scale resource plot", plot_figure("efficiency_creation_scaling.png", "fig-efficiency-creation-scaling", "Exploratory creation wall time and normalized memory pressure versus model scale", "Figure: Exploratory creation efficiency scaling")),
        fold("Show creation-cost provenance", table(["Model", "Intervention", "Fit min", "Steps", "Prompts", "Peak VRAM GiB", "Device VRAM GiB", "GPU-hours", "VRAM-hours", "VRAM fraction", "Billed USD", "Precision", "Hardware/source"], creation_provenance_rows(efficiency_records_current))),
        fold("Show method-native normalized rates", table(["Model", "Intervention", "Workload", "Total fit min", "Normalized rate", "Unit", "Precision", "Hardware/source"], native_efficiency_rows(efficiency_records_current))),
        "<h2 id='pile-validation'>Held-out Pile predictive validation</h2>",
        "<p>These are predictive checks for the tuned-lens training track, not causal swap results. Each row is evaluated on the held-out Pile test stream; lower KL to the frozen final model distribution is better. The three KL columns are first, middle, and final transformer layers.</p>",
        fold("Show held-out Pile validation table", pile_validation_table()),
        "<h3>Validated Pile causal suites</h3>",
        "<p>These rows are kept separate from the Wikitext-pilot causal plots. They appear only after the corresponding held-out predictive artifact passes validation, and report the Pile-trained tuned basis alongside logit and random controls.</p>",
        fold("Show validated Pile causal table", pile_causal_table()),
        "<p class='muted'>The earlier Qwen3-1.7B short-2M diagnostic artifacts are retained in the data directory for provenance, but are intentionally excluded from comparative plots and result tables. The full model-matched Pile protocol is being completed instead.</p>",
        "<h2 id='interpretation'>Interpretation and open questions</h2>",
        """<ul>
        <li>The original readout path runs locally, and the 27B PyTorch path works in guarded 4-bit NF4 mode.</li>
        <li>The released 27B JLens is model-matched, has a 1,000-prompt fit, and supports the current causal runs.</li>
        <li>JLens, logit, and random controls have been run on the three main causal protocols.</li>
        <li><code>tuned-wiki-small-v0</code> comparisons exist for the smaller models and 27B; their results are pilot evidence only because the fits used short Wikitext runs.</li>
        <li>The proper Pile validation/test reproduction is being staged as <code>tuned-pile-repro-v1</code>, starting with Qwen3-0.6B and proceeding upward only after held-out predictive checks pass.</li>
        <li>A model-matched Qwen3.6-27B Wikitext pilot is available. Its causal comparison uses its exported affine translator basis; the nonlinear final RMSNorm is not folded into that patch.</li>
        <li>The conjunction experiment is still a separate representation/composition study; its pilot results should not be conflated with the headline reproduction.</li>
        </ul>""",
        fold("Show full aggregate table", table(["Experiment", "Model", "Method", "n", "Improved", "Median Δrank", "Additional aggregate metrics"], method_rows)),
        "<h3>Next validation step</h3>",
        "<p>The 27B Wikitext pilot fit and causal evaluation are complete under a capped Modal run. The next validation track is <code>tuned-pile-repro-v1</code>: fit on Pile validation, score held-out Pile test, then rerun causal fixtures only after the predictive checks pass. Publish only with the exact base-model revision, fit corpus, objective, dtype/quantization, hashes, and known limitations.</p>",
        "<h2 id='provenance'>Sources and provenance</h2>",
        "<h3>Research-assistance provenance</h3>",
        "<p>This report and its supporting scripts were developed collaboratively with <strong>GPT-5.6 Luna Medium</strong>, operating through OpenAI Codex in the local workspace under the researcher's direction. The model assisted with implementation, debugging, analysis, and documentation; it did not supply model outputs or replace the recorded experimental runs. Human review and reruns remain responsible for scientific claims.</p>",
        """<h3>Modal execution provenance</h3>
        <p>The 27B tuned lens was fit and evaluated on the persistent Modal
        volumes using the validated NVIDIA PyTorch 25.11 image. These are the
        successful app records; the links open Modal's logs and resource
        telemetry.</p>
        <table><thead><tr><th>Run</th><th>Purpose</th><th>App</th><th>Observed runtime</th><th>Actual billed cost</th><th>Configured worst-case ceiling</th></tr></thead>
        <tbody>
        <tr><td>Fit</td><td>100-step native tuned-lens fit, A100-80GB</td><td><a href='https://modal.com/apps/mooreniemi/main/ap-snTfK7M83nxfrRktoZE10K'>ap-snTfK7M83nxfrRktoZE10K</a></td><td>~193 s fit phase</td><td>$0.2458</td><td>$3.02</td></tr>
        <tr><td>Causal evaluation</td><td>27B tuned rows for verbal, two-hop, flexible</td><td><a href='https://modal.com/apps/mooreniemi/main/ap-rA96vrmDrqJZUPQVUUqQG7'>ap-rA96vrmDrqJZUPQVUUqQG7</a></td><td>~9 min app lifetime</td><td>$0.5071</td><td>~$1.79</td></tr>
        </tbody></table>
        <p class='muted'>The ceilings are conservative resource-request estimates
        from the runner configuration, including the 20-minute startup allowance;
        they are not a Modal invoice. The actual billed subtotal for these two
        successful app IDs was <strong>$0.7528</strong>. The full same-day
        workspace report was <strong>$1.9379</strong> across 23 billing entries,
        including failed/aborted attempts and shell usage. The archived billing
        summary is <code>data/provenance/modal-billing-2026-07-14.json</code>.
        The fit gate was hard-capped at $5, and the fit plus evaluation ceilings
        remain below that cap when considered together.</p>""",
        "<h2 id='documentation'>Documentation and logs</h2>",
        "<p>All Markdown research notes are rendered into browser-readable HTML pages alongside this report. Open the <a href='docs/index.html'>documentation index</a> to browse the full log and experiment archive.</p>",
        documentation_links,
        "<footer>Generated by <code>scripts/generate_research_report.py</code>. The report is reproducible from the JSON files under <code>data/experiments/</code>.</footer>",
        "</main></body></html>",
    ]
    output = out_dir / "index.html"
    output.write_text("\n".join(html_parts))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "research")
    args = parser.parse_args()
    output = build_report(args.out_dir)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
