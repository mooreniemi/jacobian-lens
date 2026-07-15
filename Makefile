.PHONY: report report-pdf check

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

# Fast local checks before committing research or infrastructure changes.
check:
	uv run ruff check .
	uv run pytest -q
