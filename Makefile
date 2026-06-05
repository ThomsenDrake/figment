PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install test run build-fts

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) app.py

build-fts:
	@if [ -f scripts/build_fts.py ]; then \
		$(PYTHON) scripts/build_fts.py; \
	else \
		echo "scripts/build_fts.py not available yet; skipping FTS build."; \
	fi
