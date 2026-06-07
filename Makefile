PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
HOST ?= 127.0.0.1
PORT ?= 7860

.PHONY: install test run run-hosted-demo build-fts

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) app.py

run-hosted-demo:
	FIGMENT_MODE=hosted MODEL_STACK=omni_native MODEL_BACKEND=hosted_omni AUDIO_BACKEND=omni_native ENABLE_AUDIO_INTAKE=true \
	$(PYTHON) -c 'from app import build_app; build_app().queue().launch(server_name="$(HOST)", server_port=$(PORT))'

build-fts:
	@if [ -f scripts/build_fts.py ]; then \
		$(PYTHON) scripts/build_fts.py; \
	else \
		echo "scripts/build_fts.py not available yet; skipping FTS build."; \
	fi
