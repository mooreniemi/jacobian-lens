# Two-hop causal J-lens experiment

This is the stronger internal-reasoning test. Each prompt requires an implicit
intermediate entity or property before producing its answer. We exchange the
J-lens coordinates of the original intermediate and a replacement, then test
whether the downstream answer moves toward the replacement-specific answer.

The primary control is a random perturbation with the same per-position norm as
the coordinate-swap displacement. Results are analyzed per item, not by
treating the three tested layers as independent observations.

The current corpus is `data/experiments/probe-swap.json` (90 prompts). The
current implementation only runs single-token concepts; multi-token cases are
reported as skipped rather than silently truncated.
