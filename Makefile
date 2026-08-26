# Companion repository for the discretized CBC profile space paper.
#
#   make verify    recompute the manuscript numbers from the deposit (no data needed)
#   make check     confirm numbers_en.tex matches results/ (no data needed)
#   make all       re-run the full pipeline (needs the record-level parquet)
#
.PHONY: help verify check numbers all deposit sums clean

PYTHON ?= python

help:
	@echo "verify   - recompute the manuscript's numbers from data/ (no data access needed)"
	@echo "check    - regenerate paper/numbers_en.tex from results/ and diff (no data needed)"
	@echo "numbers  - regenerate paper/numbers_en.tex in place"
	@echo "all      - run experiments W1-W9 (requires CBC_PARQUET or data-restricted/)"
	@echo "deposit  - rebuild data/ from the record-level input"
	@echo "sums     - refresh SHA256SUMS over data/ and results/"

verify:
	$(PYTHON) src/verify_deposit.py

check:
	$(PYTHON) paper/fill_numbers.py --check

numbers:
	$(PYTHON) paper/fill_numbers.py

all:
	$(PYTHON) src/run_all.py

deposit:
	$(PYTHON) src/make_release.py

sums:
	find data results -type f ! -name SHA256SUMS -print0 \
		| sort -z | xargs -0 shasum -a 256 > SHA256SUMS

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
