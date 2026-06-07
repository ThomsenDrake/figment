PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
HOST ?= 127.0.0.1
PORT ?= 7860
FIGMENT_SMOKE_TIMEOUT_SECONDS ?= 8

.PHONY: install test run run-hosted-demo build-fts audit-claims evidence-gates smoke-model-route smoke-local-model-route local-4b-evidence local-asr-evidence

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

audit-claims:
	$(PYTHON) scripts/audit_submission_claims.py

evidence-gates:
	$(PYTHON) scripts/evidence_gate_status.py --markdown; status=$$?; test $$status -eq 0 -o $$status -eq 2

smoke-model-route:
	PYTHON_DOTENV_DISABLED=true FIGMENT_MODE=canned MODEL_STACK=omni_native MODEL_BACKEND=canned AUDIO_BACKEND=none \
	FIGMENT_SMOKE_ALLOW_NETWORK=false FIGMENT_SMOKE_TIMEOUT_SECONDS=$(FIGMENT_SMOKE_TIMEOUT_SECONDS) \
	$(PYTHON) scripts/smoke_model_route.py

smoke-local-model-route:
	FIGMENT_MODE=local MODEL_BACKEND=llama_cpp AUDIO_BACKEND=none FIGMENT_SMOKE_ALLOW_NETWORK=true FIGMENT_SMOKE_TIMEOUT_SECONDS=$(FIGMENT_SMOKE_TIMEOUT_SECONDS) \
	$(PYTHON) scripts/smoke_model_route.py

local-4b-evidence:
	PYTHON_DOTENV_DISABLED=true $(PYTHON) scripts/run_local_4b_evidence.py --base-url "$${LLAMA_BASE_URL:-http://127.0.0.1:8001/v1}"

local-asr-evidence:
	PYTHON_DOTENV_DISABLED=true $(PYTHON) scripts/run_local_asr_evidence.py
