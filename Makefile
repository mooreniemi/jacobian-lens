.PHONY: report report-pdf loop-report check

# Rebuild the browser-facing report from docs/ and data/ artifacts.
report:
	uv run python scripts/generate_research_report.py
	rm -rf "$(CURDIR)/docs/research"
	cp -a "$(CURDIR)/reports/research" "$(CURDIR)/docs/research"

# Print the rendered report with the locally installed Chrome renderer.
report-pdf: report
	google-chrome --headless --disable-gpu --no-sandbox \
		--no-pdf-header-footer \
		--print-to-pdf="$(CURDIR)/reports/research/research-report.pdf" \
		"file://$(CURDIR)/reports/research/index.html"
	cp "$(CURDIR)/reports/research/research-report.pdf" \
		"$(CURDIR)/docs/research/research-report.pdf"

# Build the standalone virtual-depth loop-transformer report.
loop-report:
	uv run python scripts/generate_loop_report.py
	mkdir -p "$(CURDIR)/docs/loop-transformers"
	cp -a "$(CURDIR)/reports/loop-transformers/." "$(CURDIR)/docs/loop-transformers/"

# Fast local checks before committing research or infrastructure changes.
check:
	uv run ruff check .
	uv run pytest -q
