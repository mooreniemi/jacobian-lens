.PHONY: report check

# Rebuild the browser-facing report from docs/ and data/ artifacts.
report:
	uv run python scripts/generate_research_report.py

# Fast local checks before committing research or infrastructure changes.
check:
	uv run ruff check .
	uv run pytest -q
