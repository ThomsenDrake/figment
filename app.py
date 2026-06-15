"""Figment Gradio app scaffold."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from fastapi.responses import HTMLResponse

from figment.audio_intake import confirm_audio_draft as _confirm_audio_draft
from figment.audio_intake import draft_audio_intake as _draft_audio_intake
from figment.config import FigmentConfig, load_config
from figment.model_client import ModelClient, ModelClientError, hosted_audio_limits_text, validate_hosted_audio_file
from figment.navigator import run_navigation
from figment.retrieval import load_protocol_cards, query_from_intake, retrieval_source_summary, search_protocol_cards
from figment.rules import evaluate_rules, run_red_flag_checks
from figment.sbar import render_sbar
from figment.trace import normalize_trace_payload, runtime_route_label, stable_hash, write_trace
from figment.ui_theme import FIGMENT_CSS
from figment.validators import urgency_floor_from_rules, validate_audio_ready

try:
    import gradio as gr
    from gradio.data_classes import FileData
except (ImportError, OSError):  # pragma: no cover - lets unit tests import without gradio installed
    gr = None
    FileData = Any  # type: ignore[misc, assignment]


TAB_TITLES = [
    "Intake",
    "Risk Check",
    "Protocol Guidance",
    "Navigator Output + Handoff",
    "Trace",
]

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_AUDIO_FILENAMES = (
    "case_1_dictated_intake.wav",
    "case_2_dictated_intake.wav",
    "case_3_dictated_intake.wav",
)

INTAKE_FIELD_KEYS = (
    "setting",
    "patient_age",
    "pregnancy_status",
    "chief_concern",
    "symptoms",
    "vitals",
    "allergies",
    "medications",
    "available_supplies",
    "responder_note",
)


DEMO_CASES: dict[str, dict[str, str]] = {
    "Disaster clinic: pediatric dehydration": {
        "setting": "shelter clinic",
        "patient_age": "7",
        "pregnancy_status": "not_applicable",
        "chief_concern": "vomiting and dehydration concern",
        "symptoms": "lethargic, very dry mouth, no urine since morning",
        "vitals": "temperature and blood pressure missing",
        "allergies": "unknown",
        "medications": "none reported",
        "available_supplies": "oral rehydration solution, radio, transport team",
        "responder_note": "Child after flood cleanup cannot keep fluids down.",
    },
    "Disaster injury: wound infection": {
        "setting": "mobile clinic",
        "patient_age": "43",
        "pregnancy_status": "not_applicable",
        "chief_concern": "wound getting worse",
        "symptoms": "spreading redness, swelling, foul drainage",
        "vitals": "temperature unknown",
        "allergies": "unknown",
        "medications": "unknown",
        "available_supplies": "clean dressings, radio",
        "responder_note": "Cut from debris three days ago.",
    },
    "Rural clinic: pregnancy danger sign": {
        "setting": "rural clinic",
        "patient_age": "29",
        "pregnancy_status": "pregnant",
        "chief_concern": "bleeding and severe headache",
        "symptoms": "vaginal bleeding, severe headache, dizziness",
        "vitals": "blood pressure not available",
        "allergies": "unknown",
        "medications": "prenatal vitamin reported",
        "available_supplies": "phone, transport contact",
        "responder_note": "Patient is pregnant and reports bleeding.",
    },
}


def collect_intake(
    setting: str,
    patient_age: str,
    pregnancy_status: str,
    chief_concern: str,
    symptoms: str,
    vitals: str,
    allergies: str,
    medications: str,
    available_supplies: str,
    responder_note: str,
) -> dict[str, Any]:
    return {
        "setting": setting,
        "patient_age": patient_age,
        "pregnancy_status": pregnancy_status,
        "chief_concern": chief_concern,
        "symptoms": symptoms,
        "vitals": vitals,
        "allergies": allergies,
        "medications": medications,
        "available_supplies": available_supplies,
        "responder_note": responder_note,
        "confirmed": False,
    }


def confirm_intake(intake: dict[str, Any], audio_draft: dict[str, Any] | None = None) -> dict[str, Any]:
    audio_validation = validate_audio_ready(audio_draft)
    if not audio_validation.passed:
        raise ValueError("; ".join(audio_validation.failures))
    confirmed = dict(intake)
    confirmed["confirmed"] = True
    return confirmed


def evaluate_red_flags(intake: dict[str, Any]) -> list[dict[str, Any]]:
    if not intake.get("confirmed"):
        return []
    return [rule.to_dict() for rule in run_red_flag_checks(intake)]


def draft_audio_intake(
    transcript: str = "",
    config: FigmentConfig | None = None,
    audio_file: str | None = None,
    provider_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = (config or load_config()).validated()
    provider_error = None
    if audio_file and not transcript.strip() and provider_payload is None and _should_use_hosted_omni_audio(config):
        try:
            validate_hosted_audio_file(audio_file)
        except ModelClientError as exc:
            provider_error = f"Hosted Omni audio draft skipped; typed transcript or canned fallback required. {exc}"
        else:
            try:
                provider_payload = ModelClient(config).generate_audio_draft(audio_file)
            except ModelClientError as exc:
                provider_error = f"Hosted Omni audio draft failed; typed transcript or canned fallback required. {exc}"
    draft = _draft_audio_intake(
        transcript=transcript,
        config=config,
        provider_payload=provider_payload,
        audio_file_received=bool(audio_file),
    )
    if audio_file:
        draft["audio_file_received"] = True
        draft["audio_filename"] = Path(audio_file).name
        draft["raw_audio_stored"] = False
        retention_note = (
            "Original clip bytes are not written to Figment traces; Gradio may keep upload/session files "
            "while the app is running, and committed demo clips stay on disk."
        )
        if _should_use_hosted_omni_audio(config):
            hosted_disclosure = _hosted_audio_disclosure_text()
            draft["hosted_audio_disclosure"] = hosted_disclosure
            retention_note = f"{retention_note} {hosted_disclosure}"
        draft["audio_retention_note"] = retention_note
    if provider_error and draft.get("audio_intake_path") == "audio_received_needs_transcript_or_model":
        draft["processing_status"] = provider_error
    return draft


def confirm_audio_draft(
    intake: dict[str, Any],
    audio_draft: dict[str, Any],
    *,
    accept: bool = True,
    edits: dict[str, str] | None = None,
    reject_fields: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _confirm_audio_draft(intake, audio_draft, accept=accept, edits=edits, reject_fields=reject_fields)


def run_case(intake: dict[str, Any], config: FigmentConfig | None = None, audio_draft: dict[str, Any] | None = None) -> dict[str, Any]:
    confirmed = confirm_intake(intake, audio_draft=audio_draft)
    rules = evaluate_red_flags(confirmed)
    runtime_config = (config or load_config()).validated()
    retrieved_cards = search_protocol_cards(query_from_intake(confirmed))
    output, trace = run_navigation(
        confirmed,
        rules,
        audio_draft=audio_draft,
        config=runtime_config,
        retrieved_cards=retrieved_cards,
    )
    evaluation = evaluate_rules(confirmed)
    trace_payload = normalize_trace_payload(trace.to_dict())
    trace_payload["retrieval"] = retrieval_source_summary(retrieved_cards)
    return {
        "intake": confirmed,
        "risk": evaluation,
        "retrieved_cards": retrieved_cards,
        "navigator_output": output,
        "sbar": render_sbar(output, trace.validator_result),
        "trace": trace_payload,
    }


def trace_download_path(trace: dict[str, Any], config: FigmentConfig | None = None) -> str:
    config = (config or load_config()).validated()
    trace_id = stable_hash(trace or {})
    path = config.trace_dir / f"figment-trace-{trace_id}.json"
    return str(write_trace(trace or {}, path))


class _FallbackDemo:
    def queue(self) -> "_FallbackDemo":
        return self

    def launch(self, *args: Any, **kwargs: Any) -> "_FallbackDemo":
        return self


def build_app(config: FigmentConfig | None = None):
    config = (config or load_config()).validated()
    if gr is None:
        return _FallbackDemo()

    if not hasattr(gr, "Server"):
        raise RuntimeError("Figment Server mode requires gradio>=6.0 so gradio.Server is available.")

    server = gr.Server(
        title="Figment",
        summary="Protocol navigator for field clinics and disaster response.",
        version="1.0.0",
    )

    @server.api(name="runtime", concurrency_limit=None)
    def runtime_api() -> dict[str, Any]:
        return _runtime_payload(config)

    @server.api(name="load_demo_case", concurrency_limit=None)
    def load_demo_case_api(name: str) -> dict[str, Any]:
        fields = _fields_dict_from_values(_load_demo_case(name))
        return {
            "fields": fields,
            "intake": collect_intake(*_field_values(fields)),
            "risk": _empty_risk_result(),
            "risk_html": _risk_summary_html(_empty_risk_result()),
            "guidance_html": _protocol_results_html([]),
            "navigator_html": _navigator_summary_html({}),
            "trace_audit_html": _trace_audit_html({}),
        }

    @server.api(name="draft_audio", concurrency_limit=1)
    def draft_audio_api(audio_file: FileData | None = None, transcript: str = "") -> dict[str, Any]:
        path = _file_data_path(audio_file)
        return draft_audio_intake(transcript=transcript or "", config=config, audio_file=path)

    @server.api(name="apply_audio_draft", concurrency_limit=None)
    def apply_audio_draft_api(fields: dict[str, Any], audio_draft: dict[str, Any] | None = None) -> dict[str, Any]:
        values = _field_values(fields)
        updated = _apply_audio_draft_ui(*values, audio_draft)
        updated_fields = _fields_dict_from_values(updated[: len(INTAKE_FIELD_KEYS)])
        return {
            "fields": updated_fields,
            "audio_draft": updated[-1],
            "intake": collect_intake(*_field_values(updated_fields)),
            "risk": _empty_risk_result(),
            "risk_html": _risk_summary_html(_empty_risk_result()),
            "guidance_html": _protocol_results_html([]),
            "navigator_html": _navigator_summary_html({}),
            "trace_audit_html": _trace_audit_html({}),
        }

    @server.api(name="confirm_intake", concurrency_limit=None)
    def confirm_intake_api(fields: dict[str, Any], audio_draft: dict[str, Any] | None = None) -> dict[str, Any]:
        confirmed, intake_state, updated_audio = _confirm_ui_intake(*_field_values(fields), audio_draft)
        return {"intake": confirmed, "intake_state": intake_state, "audio_draft": updated_audio}

    @server.api(name="risk_check", concurrency_limit=None)
    def risk_check_api(intake: dict[str, Any]) -> dict[str, Any]:
        risk, summary = _risk_ui_with_summary(intake)
        return {"risk": risk, "risk_html": summary}

    @server.api(name="retrieve_protocol_cards", concurrency_limit=None)
    def retrieve_protocol_cards_api(intake: dict[str, Any]) -> dict[str, Any]:
        cards, evidence, summary = _retrieve_with_evidence_and_summary_ui(intake)
        return {"cards": cards, "evidence": evidence, "guidance_html": summary}

    @server.api(name="run_navigator", concurrency_limit=1)
    def run_navigator_api(intake: dict[str, Any], audio_draft: dict[str, Any] | None = None) -> dict[str, Any]:
        output, sbar, trace, trace_state, summary, audit = _navigate_ui_with_summary(intake, audio_draft, config=config)
        return {
            "navigator_output": output,
            "sbar": sbar,
            "trace": trace,
            "trace_state": trace_state,
            "navigator_html": summary,
            "trace_audit_html": audit,
        }

    @server.get("/", response_class=HTMLResponse)
    async def homepage() -> str:
        return _server_homepage_html(config)

    @server.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "mode": "gradio.Server"}

    return server


def _h(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _app_header_html() -> str:
    return """
    <div class="figment-topbar">
      <div class="figment-brand">
        <div class="figment-logo">Figment</div>
        <div class="figment-positioning">Offline protocol support for field clinics and disaster response</div>
      </div>
      <div class="figment-safety">
        <span class="figment-safety-mark">!</span>
        <span>For trained responders only. Not a substitute for clinical judgment.</span>
      </div>
    </div>
    """


def _statusline_html(config: FigmentConfig) -> str:
    audio_chip = "green" if config.enable_audio_intake else "amber"
    backend_chip = "blue" if config.model_backend in {"hosted_omni", "hf_zerogpu"} else "amber"
    return f"""
    <div class="figment-statusline">
      <strong>Runtime</strong>
      <span class="figment-chip {backend_chip}">{_h(_model_mode_label(config))}</span>
      <span class="figment-chip">MODEL_STACK={_h(config.model_stack)}</span>
      <span class="figment-chip">MODEL_BACKEND={_h(config.model_backend)}</span>
      <span class="figment-chip {audio_chip}">ENABLE_AUDIO_INTAKE={_h('ON' if config.enable_audio_intake else 'OFF')}</span>
      <span class="figment-chip green">Privacy: no raw audio retained in traces</span>
    </div>
    """


def _footer_rail_html(config: FigmentConfig) -> str:
    return f"""
    <div class="figment-footer-rail">
      <div class="figment-footer-cluster">
        <strong>Model mode</strong>
        <span class="figment-chip blue">{_h(_model_mode_label(config))}</span>
        <span class="figment-chip">Local 4B + Parakeet stretch</span>
        <span class="figment-chip">Canned Trace fallback</span>
      </div>
      <div class="figment-footer-cluster">
        <strong>Schema</strong>
        <span class="figment-chip green">v1.0.0</span>
        <span class="figment-chip green">Deterministic red-flag floor enabled</span>
        <span class="figment-chip green">Privacy: no raw audio retained</span>
      </div>
    </div>
    """


def _runtime_payload(config: FigmentConfig) -> dict[str, Any]:
    return {
        "model_mode_label": _model_mode_label(config),
        "model_stack": config.model_stack,
        "model_backend": config.model_backend,
        "audio_backend": config.audio_backend,
        "enable_audio_intake": config.enable_audio_intake,
        "audio_section_title": _audio_section_title(config),
        "audio_section_subtitle": _audio_section_subtitle(config),
        "audio_clip_label": _audio_clip_label(config),
        "transcript_label": _transcript_label(config),
        "audio_chips_html": _audio_runtime_chips_html(config),
        "demo_audio_examples": _demo_audio_examples(),
        "status_text": _status_text(config),
    }


def _fields_dict_from_values(values: list[Any] | tuple[Any, ...]) -> dict[str, str]:
    return {key: str(value or "") for key, value in zip(INTAKE_FIELD_KEYS, values, strict=True)}


def _field_values(fields: dict[str, Any] | None) -> list[str]:
    fields = fields or {}
    return [str(fields.get(key, "") or "") for key in INTAKE_FIELD_KEYS]


def _file_data_path(file_data: Any) -> str | None:
    if not file_data:
        return None
    if isinstance(file_data, dict):
        path = file_data.get("path")
        return str(path) if path else None
    path = getattr(file_data, "path", None)
    return str(path) if path else None


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True).replace("</", "<\\/")


def _server_homepage_html(config: FigmentConfig) -> str:
    initial_data = {
        "tabTitles": TAB_TITLES,
        "fieldKeys": INTAKE_FIELD_KEYS,
        "runtime": _runtime_payload(config),
        "emptyRisk": _empty_risk_result(),
        "riskHtml": _risk_summary_html(_empty_risk_result()),
        "protocolLibraryHtml": _protocol_library_html(),
        "guidanceHtml": _protocol_results_html([]),
        "navigatorHtml": _navigator_summary_html({}),
        "traceAuditHtml": _trace_audit_html({}),
    }
    html_doc = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Figment</title>
    <style>
__FIGMENT_CSS__
    </style>
  </head>
  <body>
    <main class="figment-app-shell">
      <aside class="figment-mission-rail" aria-label="Mission context">
      <header class="figment-topbar">
        <div class="figment-brand">
          <a class="figment-logo" href="/" aria-label="Homepage">Figment</a>
          <div class="figment-positioning">Offline protocol support for field clinics and disaster response.</div>
        </div>
        <div class="figment-safety">
          <span class="figment-safety-mark">!</span>
          <span>For trained responders only. Not a substitute for clinical judgment.</span>
        </div>
      </header>

      <section class="figment-statusline" aria-label="Runtime status">
        <strong>Runtime</strong>
        <span class="figment-chip blue" id="runtime-mode"></span>
        <span class="figment-chip" id="runtime-stack"></span>
        <span class="figment-chip" id="runtime-backend"></span>
        <span class="figment-chip" id="runtime-audio"></span>
        <span class="figment-chip green">Privacy: no raw audio retained in traces</span>
      </section>

      <nav class="figment-tabs" aria-label="Workflow">
        <button class="figment-tab-button" type="button" data-view="intake">Intake</button>
        <button class="figment-tab-button" type="button" data-view="risk">Risk Check</button>
        <button class="figment-tab-button" type="button" data-view="protocol">Protocol Guidance</button>
        <button class="figment-tab-button" type="button" data-view="navigator">Navigator Output + Handoff</button>
        <button class="figment-tab-button" type="button" data-view="trace">Trace</button>
      </nav>

      <p class="figment-live-status" id="live-status" aria-live="polite">Server mode ready. Gradio queue endpoints are connected.</p>

      <footer class="figment-footer-rail">
        <div class="figment-footer-cluster">
          <strong>Server mode</strong>
          <span class="figment-chip blue">gradio.Server</span>
          <span class="figment-chip">Custom HTML/CSS/JS frontend</span>
          <span class="figment-chip">Gradio API queue retained</span>
        </div>
        <div class="figment-footer-cluster">
          <strong>Harness</strong>
          <span class="figment-chip green">v1.0.0 schema</span>
          <span class="figment-chip green">Deterministic red-flag floor enabled</span>
        </div>
      </footer>
      </aside>

      <section class="figment-operation-board" aria-label="Case workspace">

      <section class="figment-view" id="view-intake" data-view-panel="intake">
        <div class="figment-workspace figment-workspace-intake">
          <section class="figment-panel figment-intake-panel">
            <div class="figment-panel-heading">
              <div>
                <span class="figment-kicker">Intake</span>
                <h2>Case intake</h2>
                <p>Responder-entered facts for the protocol run.</p>
              </div>
            </div>
            <form id="intake-form" class="figment-field-grid">
              <label class="figment-control" for="field-setting">
                <span>Setting</span>
                <input id="field-setting" name="setting" type="text">
              </label>
              <label class="figment-control" for="field-patient_age">
                <span>Patient age</span>
                <input id="field-patient_age" name="patient_age" type="text">
              </label>
              <label class="figment-control" for="field-pregnancy_status">
                <span>Pregnancy status</span>
                <input id="field-pregnancy_status" name="pregnancy_status" type="text">
              </label>
              <label class="figment-control figment-control-wide" for="field-chief_concern">
                <span>Chief concern</span>
                <input id="field-chief_concern" name="chief_concern" type="text">
              </label>
              <label class="figment-control figment-control-wide" for="field-symptoms">
                <span>Symptoms</span>
                <input id="field-symptoms" name="symptoms" type="text">
              </label>
              <label class="figment-control figment-control-wide" for="field-vitals">
                <span>Vitals</span>
                <input id="field-vitals" name="vitals" type="text">
              </label>
              <label class="figment-control" for="field-allergies">
                <span>Allergies</span>
                <input id="field-allergies" name="allergies" type="text">
              </label>
              <label class="figment-control" for="field-medications">
                <span>Medications</span>
                <input id="field-medications" name="medications" type="text">
              </label>
              <label class="figment-control figment-control-wide" for="field-available_supplies">
                <span>Available supplies</span>
                <input id="field-available_supplies" name="available_supplies" type="text">
              </label>
              <label class="figment-control figment-control-wide" for="field-responder_note">
                <span>Responder note</span>
                <textarea id="field-responder_note" name="responder_note" rows="4"></textarea>
              </label>
            </form>

            <div class="figment-section-divider"></div>

            <section class="figment-audio-draft" aria-labelledby="audio-title">
              <div class="figment-panel-heading figment-panel-heading-action">
                <div>
                  <span class="figment-kicker">Audio draft</span>
                  <h2 id="audio-title"></h2>
                  <p id="audio-subtitle"></p>
                </div>
                <div id="audio-runtime-chips" class="figment-chip-row"></div>
              </div>
              <div class="figment-audio-grid">
                <label class="figment-control" for="audio-file">
                  <span id="audio-file-label"></span>
                  <input id="audio-file" name="audio_file" type="file" accept="audio/*">
                </label>
                <div class="figment-audio-actions">
                  <button class="figment-button figment-button-secondary" id="draft-audio" type="button">Draft audio fields</button>
                  <button class="figment-button figment-button-secondary" id="apply-audio" type="button">Apply audio draft</button>
                </div>
              </div>
              <label class="figment-control" for="transcript">
                <span id="transcript-label"></span>
                <textarea id="transcript" name="transcript" rows="3"></textarea>
              </label>
            </section>
          </section>

          <aside class="figment-panel figment-sticky-panel figment-review-panel">
            <div class="figment-panel-heading figment-panel-heading-action figment-review-heading">
              <div>
                <span class="figment-kicker">Review</span>
                <h2>Confirmed intake</h2>
                <p>Protocol rules and navigation use this payload.</p>
              </div>
              <button class="figment-button figment-button-primary" id="confirm-intake" type="button">Confirm intake</button>
            </div>
            <pre class="figment-json" id="intake-json">{}</pre>
            <div class="figment-section-divider"></div>
            <div class="figment-panel-heading">
              <div>
                <span class="figment-kicker">Draft</span>
                <h2>Audio suggestions</h2>
                <p>Timecoded field suggestions before apply.</p>
              </div>
            </div>
            <pre class="figment-json" id="audio-json">{}</pre>
          </aside>
        </div>
      </section>

      <section class="figment-view" id="view-risk" data-view-panel="risk" hidden>
        <div class="figment-workspace">
          <section class="figment-panel">
            <div class="figment-panel-heading">
              <div>
                <h2>Deterministic red-flag checklist</h2>
                <p>Reference checklist for the frozen safety floor. These rules are deterministic.</p>
              </div>
            </div>
            __RED_FLAG_CHECKLIST__
          </section>
          <section class="figment-panel">
            <div class="figment-panel-heading figment-panel-heading-action">
              <div>
                <h2>Rule output</h2>
                <p>The model cannot lower the deterministic protocol_urgency floor.</p>
              </div>
              <button class="figment-button figment-button-primary" id="run-risk" type="button">Run risk check</button>
            </div>
            <div id="risk-html"></div>
            <details class="figment-disclosure">
              <summary>Raw deterministic red flags JSON</summary>
              <pre class="figment-json" id="risk-json">{}</pre>
            </details>
          </section>
        </div>
      </section>

      <section class="figment-view" id="view-protocol" data-view-panel="protocol" hidden>
        <div class="figment-workspace">
          <section class="figment-panel">
            <div class="figment-panel-heading figment-panel-heading-action">
              <div>
                <h2>Protocol card browser</h2>
                <p>Local protocol cards retrieved from the confirmed intake.</p>
              </div>
              <button class="figment-button figment-button-primary" id="retrieve-cards" type="button">Retrieve protocol cards</button>
            </div>
            <div id="protocol-library"></div>
          </section>
          <section class="figment-panel">
            <div id="guidance-html"></div>
            <div class="figment-section-divider"></div>
            <label class="figment-control" for="guidance-evidence">
              <span>Protocol evidence panel</span>
              <textarea id="guidance-evidence" name="guidance_evidence" rows="8" readonly></textarea>
            </label>
            <details class="figment-disclosure">
              <summary>Retrieved protocol cards JSON</summary>
              <pre class="figment-json" id="guidance-json">[]</pre>
            </details>
          </section>
        </div>
      </section>

      <section class="figment-view" id="view-navigator" data-view-panel="navigator" hidden>
        <div class="figment-workspace">
          <section class="figment-panel">
            <div class="figment-panel-heading figment-panel-heading-action">
              <div>
                <h2>Navigator output JSON</h2>
                <p>Machine-readable protocol navigation output.</p>
              </div>
              <button class="figment-button figment-button-primary" id="run-navigator" type="button">Run navigator</button>
            </div>
            <pre class="figment-json figment-json-tall" id="output-json">{}</pre>
          </section>
          <section class="figment-panel">
            <div id="navigator-html"></div>
            <label class="figment-control" for="sbar-text">
              <span>SBAR handoff</span>
              <textarea id="sbar-text" name="sbar_text" rows="8" readonly></textarea>
            </label>
          </section>
        </div>
      </section>

      <section class="figment-view" id="view-trace" data-view-panel="trace" hidden>
        <div class="figment-workspace">
          <section class="figment-panel">
            <div class="figment-panel-heading figment-panel-heading-action">
              <div>
                <h2>Run steps timeline</h2>
                <p>Audit trail from intake through validation.</p>
              </div>
              <button class="figment-button figment-button-secondary" id="export-trace" type="button">Export trace</button>
            </div>
            <div id="trace-audit-html"></div>
          </section>
          <section class="figment-panel">
            <div class="figment-panel-heading">
              <div>
                <h2>Trace JSON</h2>
                <p>Raw audit object for review and export.</p>
              </div>
            </div>
            <pre class="figment-json figment-json-tall" id="trace-json">{}</pre>
          </section>
        </div>
      </section>

      </section>
    </main>

    <script id="figment-data" type="application/json">__FIGMENT_DATA__</script>
    <script type="module">
      import { Client, handle_file } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

      const initial = JSON.parse(document.getElementById("figment-data").textContent);
      const fieldKeys = initial.fieldKeys;
      const clientPromise = Client.connect(window.location.origin);
      const state = {
        fields: emptyFields(),
        intake: {},
        audioDraft: null,
        risk: initial.emptyRisk,
        cards: [],
        evidence: "",
        navigatorOutput: {},
        sbar: "",
        trace: {},
        riskHtml: initial.riskHtml,
        guidanceHtml: initial.guidanceHtml,
        navigatorHtml: initial.navigatorHtml,
        traceAuditHtml: initial.traceAuditHtml,
      };

      const $ = (selector) => document.querySelector(selector);
      const $$ = (selector) => Array.from(document.querySelectorAll(selector));

      function emptyFields() {
        return Object.fromEntries(initial.fieldKeys.map((key) => [key, ""]));
      }

      async function predict(name, args = []) {
        const client = await clientPromise;
        const result = await client.predict(`/${name}`, args);
        return result.data[0];
      }

      function setLiveStatus(message) {
        $("#live-status").textContent = message;
      }

      function setBusy(button, busy) {
        if (!button) return;
        button.disabled = busy;
        button.classList.toggle("figment-button-loading", busy);
      }

      async function runAction(button, pendingMessage, doneMessage, action) {
        try {
          setBusy(button, true);
          setLiveStatus(pendingMessage);
          await action();
          setLiveStatus(doneMessage);
        } catch (error) {
          console.error(error);
          setLiveStatus(error?.message || "Action failed. Check the browser console.");
        } finally {
          setBusy(button, false);
        }
      }

      function readFields() {
        const fields = {};
        for (const key of fieldKeys) {
          fields[key] = $(`#field-${key}`).value;
        }
        return fields;
      }

      function writeFields(fields) {
        for (const key of fieldKeys) {
          const input = $(`#field-${key}`);
          if (input) input.value = fields?.[key] || "";
        }
        state.fields = { ...emptyFields(), ...(fields || {}) };
      }

      function resetDownstream() {
        state.intake = {};
        state.risk = initial.emptyRisk;
        state.cards = [];
        state.evidence = "";
        state.navigatorOutput = {};
        state.sbar = "";
        state.trace = {};
        state.riskHtml = initial.riskHtml;
        state.guidanceHtml = initial.guidanceHtml;
        state.navigatorHtml = initial.navigatorHtml;
        state.traceAuditHtml = initial.traceAuditHtml;
      }

      function renderJson(selector, value) {
        $(selector).textContent = JSON.stringify(value ?? {}, null, 2);
      }

      function render() {
        renderJson("#audio-json", state.audioDraft || {});
        renderJson("#intake-json", state.intake || {});
        renderJson("#risk-json", state.risk || {});
        renderJson("#guidance-json", state.cards || []);
        renderJson("#output-json", state.navigatorOutput || {});
        renderJson("#trace-json", state.trace || {});
        $("#risk-html").innerHTML = state.riskHtml || initial.riskHtml;
        $("#guidance-html").innerHTML = state.guidanceHtml || initial.guidanceHtml;
        $("#guidance-evidence").value = state.evidence || "";
        $("#navigator-html").innerHTML = state.navigatorHtml || initial.navigatorHtml;
        $("#sbar-text").value = state.sbar || "";
        $("#trace-audit-html").innerHTML = state.traceAuditHtml || initial.traceAuditHtml;
      }

      function setView(name) {
        for (const panel of $$("[data-view-panel]")) {
          panel.hidden = panel.dataset.viewPanel !== name;
        }
        for (const button of $$(".figment-tab-button")) {
          const active = button.dataset.view === name;
          button.classList.toggle("is-active", active);
          button.setAttribute("aria-current", active ? "page" : "false");
        }
      }

      async function ensureConfirmed() {
        if (state.intake?.confirmed) return;
        const payload = await predict("confirm_intake", [readFields(), state.audioDraft]);
        state.intake = payload.intake || {};
        state.audioDraft = payload.audio_draft || state.audioDraft;
        render();
      }

      async function draftAudio() {
        const file = $("#audio-file").files[0];
        const transcript = $("#transcript").value;
        const args = file ? [handle_file(file), transcript] : [null, transcript];
        state.audioDraft = await predict("draft_audio", args);
        resetDownstream();
        render();
      }

      async function applyAudioDraft() {
        const payload = await predict("apply_audio_draft", [readFields(), state.audioDraft]);
        writeFields(payload.fields || {});
        state.audioDraft = payload.audio_draft || null;
        resetDownstream();
        state.risk = payload.risk || initial.emptyRisk;
        state.riskHtml = payload.risk_html || initial.riskHtml;
        state.guidanceHtml = payload.guidance_html || initial.guidanceHtml;
        state.navigatorHtml = payload.navigator_html || initial.navigatorHtml;
        state.traceAuditHtml = payload.trace_audit_html || initial.traceAuditHtml;
        render();
      }

      async function confirmIntake() {
        const payload = await predict("confirm_intake", [readFields(), state.audioDraft]);
        state.intake = payload.intake || {};
        state.audioDraft = payload.audio_draft || state.audioDraft;
        render();
      }

      async function runRisk() {
        await ensureConfirmed();
        const payload = await predict("risk_check", [state.intake]);
        state.risk = payload.risk || initial.emptyRisk;
        state.riskHtml = payload.risk_html || initial.riskHtml;
        render();
      }

      async function retrieveCards() {
        await ensureConfirmed();
        const payload = await predict("retrieve_protocol_cards", [state.intake]);
        state.cards = payload.cards || [];
        state.evidence = payload.evidence || "";
        state.guidanceHtml = payload.guidance_html || initial.guidanceHtml;
        render();
      }

      async function runNavigator() {
        await ensureConfirmed();
        const payload = await predict("run_navigator", [state.intake, state.audioDraft]);
        state.navigatorOutput = payload.navigator_output || {};
        state.sbar = payload.sbar || "";
        state.trace = payload.trace || {};
        state.navigatorHtml = payload.navigator_html || initial.navigatorHtml;
        state.traceAuditHtml = payload.trace_audit_html || initial.traceAuditHtml;
        render();
      }

      function exportTrace() {
        if (!state.trace || Object.keys(state.trace).length === 0) {
          setLiveStatus("Run the navigator before exporting a trace.");
          return;
        }
        const blob = new Blob([JSON.stringify(state.trace, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "figment-trace.json";
        link.click();
        URL.revokeObjectURL(url);
        setLiveStatus("Trace export prepared in the browser.");
      }

      function initialize() {
        const runtime = initial.runtime;
        $("#runtime-mode").textContent = runtime.model_mode_label;
        $("#runtime-stack").textContent = `MODEL_STACK=${runtime.model_stack}`;
        $("#runtime-backend").textContent = `MODEL_BACKEND=${runtime.model_backend}`;
        $("#runtime-audio").textContent = `ENABLE_AUDIO_INTAKE=${runtime.enable_audio_intake ? "ON" : "OFF"}`;
        $("#runtime-audio").classList.add(runtime.enable_audio_intake ? "green" : "amber");
        $("#audio-title").textContent = runtime.audio_section_title;
        $("#audio-subtitle").textContent = runtime.audio_section_subtitle;
        $("#audio-file-label").textContent = runtime.audio_clip_label;
        $("#transcript-label").textContent = runtime.transcript_label;
        $("#audio-runtime-chips").innerHTML = runtime.audio_chips_html;
        $("#protocol-library").innerHTML = initial.protocolLibraryHtml;

        for (const button of $$(".figment-tab-button")) {
          button.addEventListener("click", () => setView(button.dataset.view));
        }
        for (const input of $$("#intake-form input, #intake-form textarea")) {
          input.addEventListener("input", () => {
            state.fields = readFields();
            resetDownstream();
            render();
          });
        }

        $("#draft-audio").addEventListener("click", () => runAction($("#draft-audio"), "Drafting audio fields through Gradio Server.", "Audio draft updated.", draftAudio));
        $("#apply-audio").addEventListener("click", () => runAction($("#apply-audio"), "Applying audio draft.", "Audio draft applied to editable intake.", applyAudioDraft));
        $("#confirm-intake").addEventListener("click", () => runAction($("#confirm-intake"), "Confirming intake.", "Confirmed intake is ready for rules and navigation.", confirmIntake));
        $("#run-risk").addEventListener("click", () => runAction($("#run-risk"), "Running deterministic red-flag checks.", "Risk check complete.", runRisk));
        $("#retrieve-cards").addEventListener("click", () => runAction($("#retrieve-cards"), "Retrieving local protocol cards.", "Protocol cards retrieved.", retrieveCards));
        $("#run-navigator").addEventListener("click", () => runAction($("#run-navigator"), "Running navigator through Gradio Server.", "Navigator output and trace complete.", runNavigator));
        $("#export-trace").addEventListener("click", exportTrace);

        const audioEnabled = Boolean(runtime.enable_audio_intake);
        $("#audio-file").disabled = !audioEnabled;
        $("#transcript").disabled = !audioEnabled;
        $("#draft-audio").disabled = !audioEnabled;
        $("#apply-audio").disabled = !audioEnabled;

        setView("intake");
        render();
      }

      initialize();
    </script>
  </body>
</html>
"""
    return (
        html_doc.replace("__FIGMENT_CSS__", FIGMENT_CSS)
        .replace("__FIGMENT_DATA__", _json_for_script(initial_data))
        .replace("__RED_FLAG_CHECKLIST__", _red_flag_checklist_html())
    )


def _model_mode_label(config: FigmentConfig) -> str:
    if config.model_backend == "hosted_omni":
        return "Configured backend: hosted_omni"
    if config.model_backend == "hf_zerogpu":
        return "Configured backend: hf_zerogpu"
    if config.model_backend == "llama_cpp":
        return "Configured backend: llama_cpp"
    return "Configured backend: canned"


def _audio_section_title(config: FigmentConfig) -> str:
    if not config.enable_audio_intake or config.audio_backend == "none":
        return "Audio draft intake disabled"
    if config.audio_backend == "omni_native" and config.model_backend == "hosted_omni":
        return "Hosted Omni audio draft"
    if config.audio_backend == "parakeet_nemo":
        return "Local Parakeet ASR draft"
    if config.audio_backend == "canned":
        return "Canned audio demo draft"
    return "Audio draft intake"


def _audio_section_subtitle(config: FigmentConfig) -> str:
    if not config.enable_audio_intake or config.audio_backend == "none":
        return "Typed confirmed intake remains the only active source for rules and navigation."
    if config.audio_backend == "omni_native" and config.model_backend == "hosted_omni":
        return (
            "Record or upload responder dictation for a provisional Omni draft. Audio is sent to the configured "
            f"hosted endpoint; use only synthetic or de-identified clips. Limit: {hosted_audio_limits_text()}."
        )
    if config.audio_backend == "parakeet_nemo":
        return "Use gated local ASR for provisional field suggestions, then confirm fields before rules run."
    if config.audio_backend == "canned":
        return "Use canned clips only as repeatable demo input, then confirm fields before rules run."
    return "Draft suggestions are provisional until the confirmed intake form is reviewed."


def _audio_clip_label(config: FigmentConfig) -> str:
    if not config.enable_audio_intake or config.audio_backend == "none":
        return "Audio intake disabled"
    if config.audio_backend == "parakeet_nemo":
        return "Parakeet audio intake"
    if config.audio_backend == "canned":
        return "Demo audio intake"
    return "Hosted Omni audio intake"


def _transcript_label(config: FigmentConfig) -> str:
    if not config.enable_audio_intake or config.audio_backend == "none":
        return "Typed transcript heuristic disabled"
    return "Typed transcript heuristic"


def _audio_runtime_chips_html(config: FigmentConfig) -> str:
    if not config.enable_audio_intake or config.audio_backend == "none":
        return '<span class="figment-chip amber">Audio intake disabled</span>'
    chips = ['<span class="figment-chip amber">Confirm before rules run</span>']
    if config.audio_backend == "omni_native" and config.model_backend == "hosted_omni":
        chips.insert(0, '<span class="figment-chip green">Hosted Omni audio</span>')
        chips.append('<span class="figment-chip amber">Hosted endpoint: synthetic/de-identified only</span>')
    elif config.audio_backend == "parakeet_nemo":
        chips.insert(0, '<span class="figment-chip blue">Parakeet ASR</span>')
    elif config.audio_backend == "canned":
        chips.insert(0, '<span class="figment-chip amber">Canned demo audio</span>')
    else:
        chips.insert(0, '<span class="figment-chip amber">Typed transcript heuristic</span>')
    return " ".join(chips)


def _section_header_html(title: str, subtitle: str = "") -> str:
    subtitle_html = f'<div class="figment-section-subtitle">{_h(subtitle)}</div>' if subtitle else ""
    return f'<div class="figment-section-title">{_h(title)}</div>{subtitle_html}'


def _red_flag_checklist_html() -> str:
    categories = {
        "Airway / Breathing": [
            "Unable to speak full sentences",
            "O2 sat below local threshold",
            "Stridor or severe wheeze",
            "RR very high or very low",
        ],
        "Circulation": [
            "SBP below local threshold",
            "Cap refill prolonged",
            "Active bleeding not controlled",
            "Pulse thready or collapsing",
        ],
        "Neurologic": [
            "Unresponsive or difficult to arouse",
            "New confusion or disorientation",
            "Seizure activity",
            "Severe headache with danger signs",
        ],
        "Pregnancy": [
            "Vaginal bleeding",
            "Severe headache or visual changes",
            "Convulsions",
            "Severe abdominal pain",
        ],
        "Pediatric": [
            "Lethargic or not waking",
            "Poor feeding or refuses fluids",
            "Cap refill prolonged",
            "No urine reported",
        ],
        "Infection / Wound": [
            "Spreading redness",
            "Foul drainage",
            "Suspected sepsis cues",
            "Rapidly worsening pain",
        ],
        "Chest Pain / Stroke": [
            "Crushing or pressure pain",
            "Radiates to arm, jaw, or back",
            "Face droop or arm weakness",
            "Speech difficulty",
        ],
    }
    panels = []
    for title, items in categories.items():
        panels.append(
            '<div class="figment-panel-soft">'
            f'<div class="figment-section-title">{_h(title)}</div>'
            f'<ul class="figment-checklist">{"".join(f"<li>{_h(item)}</li>" for item in items)}</ul>'
            "</div>"
        )
    return f'<div style="display:grid; gap:10px;">{"".join(panels)}</div>'


def _risk_ui_with_summary(intake: dict[str, Any]) -> tuple[dict[str, Any], str]:
    result = _risk_ui(intake)
    return result, _risk_summary_html(result)


def _risk_summary_html(result: dict[str, Any]) -> str:
    urgency = str(result.get("protocol_urgency") or "routine").lower()
    if urgency not in {"routine", "monitor", "urgent", "emergency"}:
        urgency = "routine"
    rules = result.get("red_flags") if isinstance(result.get("red_flags"), list) else []
    rows = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{_h(rule.get('rule_id'))}</td>"
            f"<td>{_h(rule.get('evidence'))}</td>"
            f"<td>{_h(rule.get('card_id'))}</td>"
            f"<td>{_urgency_chip_html(str(rule.get('urgency') or urgency))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="4" class="figment-muted">No confirmed intake red flags have fired yet.</td></tr>')

    source_cards = sorted({str(rule.get("card_id")) for rule in rules if isinstance(rule, dict) and rule.get("card_id")})
    if not source_cards:
        source_cards = ["Run rules after confirming intake"]

    return f"""
    <div class="figment-urgency-banner">
      <div>
        <div class="figment-muted" style="font-weight:760;">PROTOCOL_URGENCY</div>
        <div class="figment-urgency-word {urgency}">{_h(urgency.upper())}</div>
      </div>
      <div class="figment-lockout">
        <strong>Deterministic safety floor locked</strong><br>
        Rules enforce this minimum. AI cannot lower this floor.
      </div>
    </div>
    <div style="height:12px"></div>
    <div class="figment-section-title">Fired Rules <span class="figment-muted">(deterministic)</span></div>
    <table class="figment-table">
      <thead><tr><th>Rule ID</th><th>Evidence</th><th>Protocol Card</th><th>Urgency Floor</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <div style="height:12px"></div>
    <div class="figment-card-grid">
      <div class="figment-mini-card">
        <h4>Validation Messages</h4>
        <ul>
          <li>Confirmed intake required before rules.</li>
          <li>Deterministic rules triggered: {_h(len(rules))}</li>
          <li>Schema validation: ready for navigator.</li>
        </ul>
      </div>
      <div class="figment-mini-card">
        <h4>Source Protocol Cards</h4>
        <div>{''.join(f'<span class="figment-chip">{_h(card)}</span> ' for card in source_cards)}</div>
      </div>
    </div>
    """


def _retrieve_with_evidence_and_summary_ui(intake: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    cards, evidence = _retrieve_with_evidence_ui(intake)
    return cards, evidence, _protocol_results_html(cards)


def _protocol_library_html() -> str:
    rows = []
    for card in load_protocol_cards()[:10]:
        card_id = str(card.get("card_id", ""))
        rows.append(
            "<tr>"
            f"<td>{_h(card_id)}</td>"
            f"<td>{_h(_protocol_condition(card))}</td>"
            f"<td>{_protocol_card_badge_html(card)}</td>"
            "<td>v1</td>"
            "</tr>"
        )
    return f"""
    <div class="figment-panel-soft">
      <div class="figment-section-subtitle">Search and filters are represented by the confirmed intake query in this prototype.</div>
      <table class="figment-table">
        <thead><tr><th>Card ID</th><th>Condition</th><th>Urgency</th><th>Version</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def _protocol_results_html(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return """
        <div class="figment-section-title">Selected Protocol Card</div>
        <div class="figment-panel-soft">
          Confirm intake, then retrieve protocol cards to populate this browser.
        </div>
        """
    first = cards[0]
    card = first.get("card") if isinstance(first.get("card"), dict) else first
    title = str(card.get("title") or first.get("title") or "Selected protocol card")
    card_id = str(card.get("card_id") or first.get("card_id") or "")
    rationale_rows = []
    for item in cards:
        item_card = item.get("card") if isinstance(item.get("card"), dict) else item
        rationale_rows.append(
            "<tr>"
            f"<td>{_h(item.get('card_id') or item_card.get('card_id'))}</td>"
            f"<td>{_h(_protocol_condition(item_card))}</td>"
            f"<td>{_h(item.get('source') or 'unknown')}</td>"
            f"<td>{_h(_relevance_text(item))}</td>"
            "</tr>"
        )
    return f"""
    <div class="figment-section-title">Selected Protocol Card <span class="figment-chip blue">Version: v1</span></div>
    <h3 style="margin:0 0 10px;">{_h(card_id)}</h3>
    <div class="figment-section-subtitle">{_h(title)}</div>
    <div class="figment-card-grid">
      {_protocol_detail_card_html("When Relevant", card.get("applies_to"))}
      {_protocol_detail_card_html("Red Flags (Escalate)", card.get("red_flags"))}
      {_protocol_detail_card_html("Collect Next", card.get("required_observations"))}
      {_protocol_detail_card_html("Responder Checklist", card.get("local_actions"))}
      {_protocol_detail_card_html("Do Not Do", card.get("forbidden_actions"))}
      {_protocol_detail_card_html("Source Note", [card.get("source_note"), card.get("safety_boundary")])}
    </div>
    <div style="height:12px"></div>
    <div class="figment-section-title">Why these cards were retrieved</div>
    <table class="figment-table">
      <thead><tr><th>Card ID</th><th>Matched Context</th><th>Source</th><th>Relevance Reason</th></tr></thead>
      <tbody>{''.join(rationale_rows)}</tbody>
    </table>
    """


def _protocol_detail_card_html(title: str, values: Any) -> str:
    items = _as_list(values)
    if not items:
        items = ["No value available yet."]
    return (
        '<div class="figment-mini-card">'
        f'<h4>{_h(title)}</h4>'
        f'<ul>{"".join(f"<li>{_h(item)}</li>" for item in items)}</ul>'
        "</div>"
    )


def _protocol_condition(card: dict[str, Any]) -> str:
    applies_to = _as_list(card.get("applies_to"))
    if applies_to:
        return str(applies_to[0]).replace("_", " ").title()
    card_id = str(card.get("card_id", ""))
    return card_id.split("-")[0].title() if card_id else "General"


def _protocol_card_badge_html(card: dict[str, Any]) -> str:
    card_text = " ".join(_as_list(card.get("red_flags")) + _as_list(card.get("escalation_criteria"))).lower()
    if "emergency" in card_text:
        return '<span class="figment-chip red">Emergency</span>'
    if "urgent" in card_text or card.get("red_flags"):
        return '<span class="figment-chip amber">Urgent</span>'
    return '<span class="figment-chip blue">All</span>'


def _navigate_ui_with_summary(
    intake: dict[str, Any],
    audio_draft: dict[str, Any] | None,
    config: FigmentConfig | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any], str, str]:
    output, sbar, trace, trace_state = _navigate_ui(intake, audio_draft, config=config)
    return output, sbar, trace, trace_state, _navigator_summary_html(output, trace), _trace_audit_html(trace)


def _navigator_summary_html(output: dict[str, Any], trace: dict[str, Any] | None = None) -> str:
    if not output:
        return """
        <div class="figment-section-title">Protocol Urgency</div>
        <div class="figment-panel-soft">Run the navigator after confirming intake and red-flag checks.</div>
        """
    urgency = str(output.get("protocol_urgency") or "routine").lower()
    if urgency not in {"routine", "monitor", "urgent", "emergency"}:
        urgency = "routine"
    handoff = output.get("handoff_note_sbar") if isinstance(output.get("handoff_note_sbar"), dict) else {}
    runtime_card = _runtime_contribution_card_html(trace)
    evidence_card = _harness_evidence_card_html(output, trace)
    return f"""
    <div class="figment-urgency-banner">
      <div>
        <div class="figment-section-title">Protocol Urgency</div>
        <div class="figment-urgency-word {urgency}">{_h(urgency.upper())}</div>
      </div>
      <div class="figment-lockout">
        <strong>Deterministic safety floor locked</strong><br>
        Minimum rules enforced. AI cannot lower this floor.
      </div>
    </div>
    <div style="height:12px"></div>
    {runtime_card}
    <div style="height:12px"></div>
    {evidence_card}
    <div style="height:12px"></div>
    <div class="figment-card-grid">
      {_navigator_list_card_html("Missing Observations", output.get("missing_info_to_collect"))}
      {_navigator_list_card_html("Responder Checklist", output.get("responder_checklist"), checked=True)}
      {_navigator_list_card_html("Do-Not-Do", output.get("do_not_do"))}
      {_navigator_list_card_html("Source Cards", output.get("source_cards"))}
    </div>
    <div style="height:12px"></div>
    <div class="figment-mini-card">
      <h4>Responder Script <span class="figment-muted">(plain language)</span></h4>
      <p>{_h(output.get("responder_plain_language_script") or "No script generated yet.")}</p>
    </div>
    <div style="height:12px"></div>
    <div class="figment-section-title">SBAR Handoff</div>
    <table class="figment-table">
      <thead><tr><th>Situation</th><th>Background</th><th>Assessment Observations Only</th><th>Handoff Request</th></tr></thead>
      <tbody><tr>
        <td>{_h(handoff.get("situation"))}</td>
        <td>{_h(handoff.get("background"))}</td>
        <td>{_h(handoff.get("assessment_observations_only"))}</td>
        <td>{_h(handoff.get("handoff_request"))}</td>
      </tr></tbody>
    </table>
    """


def _runtime_contribution_card_html(trace: dict[str, Any] | None) -> str:
    if not trace:
        return ""
    payload = normalize_trace_payload(trace)
    route = payload.get("model_route") if isinstance(payload.get("model_route"), dict) else {}
    retrieval = payload.get("retrieval") if isinstance(payload.get("retrieval"), dict) else {}
    provenance_summary = payload.get("field_provenance_summary") if isinstance(payload.get("field_provenance_summary"), dict) else {}
    final_route = str(route.get("final_route") or "unknown")
    fallback_reason = route.get("fallback_reason") or "none"
    retrieval_source = retrieval.get("primary_source") or "not traced"
    return f"""
    <div class="figment-mini-card">
      <h4>Runtime contribution</h4>
      <span class="figment-chip {_route_chip_class(final_route)}">{_h(runtime_route_label(route))}</span>
      <span class="figment-chip">Configured backend: {_h(route.get('raw_route'))}</span>
      <span class="figment-chip">validation={_h(route.get('validation_status'))}</span>
      <span class="figment-chip">retrieval={_h(retrieval_source)}</span>
      {_field_provenance_counts_html(provenance_summary)}
      {_repair_metrics_inline_html(route)}
      <div class="figment-section-subtitle">fallback_reason={_h(fallback_reason)}</div>
    </div>
    """


def _harness_evidence_card_html(output: dict[str, Any] | None, trace: dict[str, Any] | None = None) -> str:
    evidence = _harness_evidence_from(output, trace)
    if not evidence:
        return ""
    retrieved_count = len(_as_list(evidence.get("retrieved_card_ids")))
    rule_count = len(_as_list(evidence.get("deterministic_rule_ids")))
    source_count = len(_as_list(evidence.get("source_card_ids")))
    final_route = str(evidence.get("final_route") or "unknown")
    return f"""
    <div class="figment-mini-card">
      <h4>Harness Evidence</h4>
      <span class="figment-chip green">Intake confirmed: {_h(evidence.get('confirmed_intake'))}</span>
      <span class="figment-chip green">Validation: {_h(evidence.get('validator_status'))}</span>
      <span class="figment-chip">Retrieved cards: {_h(retrieved_count)}</span>
      <span class="figment-chip">Rule results: {_h(rule_count)}</span>
      <span class="figment-chip">Source cards: {_h(source_count)}</span>
      <span class="figment-chip">Urgency floor: {_h(evidence.get('urgency_floor'))}</span>
      <span class="figment-chip {_route_chip_class(final_route)}">Route: {_h(runtime_route_label(final_route))}</span>
      <span class="figment-chip">Audio correction: {_h(evidence.get('audio_correction_status'))}</span>
    </div>
    """


def _harness_evidence_from(output: dict[str, Any] | None, trace: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(output, dict) and isinstance(output.get("harness_evidence"), dict):
        return output["harness_evidence"]
    if isinstance(trace, dict):
        normalized = normalize_trace_payload(trace)
        if isinstance(normalized.get("harness_evidence"), dict):
            return normalized["harness_evidence"]
        navigator_output = normalized.get("navigator_output")
        if isinstance(navigator_output, dict) and isinstance(navigator_output.get("harness_evidence"), dict):
            return navigator_output["harness_evidence"]
    return {}


def _navigator_list_card_html(title: str, values: Any, *, checked: bool = False) -> str:
    items = _as_list(values) or ["No items generated yet."]
    cls = "figment-checklist checked" if checked else "figment-checklist"
    return (
        '<div class="figment-mini-card">'
        f'<h4>{_h(title)}</h4>'
        f'<ul class="{cls}">{"".join(f"<li>{_h(item)}</li>" for item in items)}</ul>'
        "</div>"
    )


def _route_chip_class(final_route: str) -> str:
    return {
        "live_model_generated": "green",
        "model_repaired": "blue",
        "model_with_deterministic_patches": "blue",
        "validation_fallback": "amber",
        "canned_backend": "amber",
    }.get(final_route, "amber")


def _field_provenance_counts_html(summary: dict[str, Any]) -> str:
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    if not counts:
        return '<span class="figment-chip">Field provenance: not traced</span>'
    chips = [
        f'<span class="figment-chip">Field provenance: {_h(name)}={_h(count)}</span>'
        for name, count in sorted(counts.items())
    ]
    return "".join(chips)


def _repair_metrics_inline_html(route: dict[str, Any]) -> str:
    attempts = route.get("repair_attempt_count", 0)
    if not attempts:
        return '<span class="figment-chip">Repair calls: 0</span>'
    cap = route.get("repair_attempt_cap", 0)
    latency = route.get("repair_latency_ms", 0.0)
    capped = " capped" if route.get("repair_capped") else ""
    return (
        f'<span class="figment-chip">Repair calls: {_h(attempts)} / {_h(cap)}{capped}</span>'
        f'<span class="figment-chip">Repair latency: {_h(latency)} ms</span>'
    )


def _trace_audit_html(trace: dict[str, Any]) -> str:
    if not trace:
        return """
        <div class="figment-panel-soft">Run the navigator to populate timeline, validation, model route, and trace metadata.</div>
        """
    trace = normalize_trace_payload(trace)
    events = _as_list(trace.get("events"))
    if not events:
        events = ["input captured", "rules evaluated", "cards retrieved", "navigator output generated", "validation complete"]
    rows = []
    for index, event in enumerate(events, start=1):
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{_h(event)}</td>"
            '<td><span class="figment-chip green">OK</span></td>'
            f"<td>{_h(_trace_event_detail(event, trace))}</td>"
            "</tr>"
        )
    validator = trace.get("validator_result") if isinstance(trace.get("validator_result"), dict) else {}
    route = trace.get("model_route") if isinstance(trace.get("model_route"), dict) else {}
    retrieval = trace.get("retrieval") if isinstance(trace.get("retrieval"), dict) else {}
    provenance_summary = trace.get("field_provenance_summary") if isinstance(trace.get("field_provenance_summary"), dict) else {}
    evidence = _harness_evidence_from(None, trace)
    final_route = str(route.get("final_route") or "unknown")
    return f"""
    <table class="figment-table">
      <thead><tr><th>Step</th><th>Component</th><th>Status</th><th>Details</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <div style="height:12px"></div>
    <div class="figment-card-grid">
      <div class="figment-mini-card">
        <h4>Audit Summary</h4>
        <span class="figment-chip green">Raw audio retained: false</span>
        <span class="figment-chip green">Schema valid: {_h(validator.get('passed'))}</span>
        <span class="figment-chip {_route_chip_class(final_route)}">{_h(runtime_route_label(route))}</span>
        <span class="figment-chip">Retrieval: {_h(retrieval.get('primary_source') or 'not traced')}</span>
        <span class="figment-chip">Harness evidence: {_h('visible' if evidence else 'not traced')}</span>
      </div>
      {_harness_evidence_card_html(None, trace)}
      <div class="figment-mini-card">
        <h4>Model &amp; Performance</h4>
        <table class="figment-table">
          <tbody>
            <tr><td>Raw route</td><td>{_h(route.get('raw_route'))}</td></tr>
            <tr><td>Final route</td><td>{_h(route.get('final_route'))}</td></tr>
            <tr><td>Model ID</td><td>{_h(route.get('model_id'))}</td></tr>
            <tr><td>Fallback tier</td><td>{_h(route.get('fallback_tier'))}</td></tr>
            <tr><td>Fallback reason</td><td>{_h(route.get('fallback_reason') or 'none')}</td></tr>
            <tr><td>Validation status</td><td>{_h(route.get('validation_status'))}</td></tr>
            <tr><td>Repair calls</td><td>{_h(route.get('repair_attempt_count', 0))} / {_h(route.get('repair_attempt_cap', 0))}</td></tr>
            <tr><td>Repair latency ms</td><td>{_h(route.get('repair_latency_ms', 0.0))}</td></tr>
          </tbody>
        </table>
      </div>
      <div class="figment-mini-card">
        <h4>Field provenance</h4>
        {_field_provenance_counts_html(provenance_summary)}
        <table class="figment-table">
          <tbody>
            <tr><td>Total fields</td><td>{_h(provenance_summary.get('total_fields', 0))}</td></tr>
            <tr><td>Deterministic patches</td><td>{_h(provenance_summary.get('deterministic_patch_count', 0))}</td></tr>
            <tr><td>Model retained</td><td>{_h(provenance_summary.get('model_retained_count', 0))}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
    """


def _trace_event_detail(event: str, trace: dict[str, Any]) -> str:
    if "rules" in event:
        return f"{len(trace.get('red_flags') or [])} deterministic red-flag result(s)."
    if "cards" in event:
        retrieval = trace.get("retrieval") if isinstance(trace.get("retrieval"), dict) else {}
        source = retrieval.get("primary_source")
        suffix = f" via {source}" if source else ""
        return f"{len(trace.get('retrieved_card_ids') or [])} protocol card(s) retrieved{suffix}."
    if "validation" in event:
        validator = trace.get("validator_result") if isinstance(trace.get("validator_result"), dict) else {}
        return "Output conforms to schema." if validator.get("passed") else "Validation failures present."
    if "model" in event or "navigator" in event:
        route = trace.get("model_route") if isinstance(trace.get("model_route"), dict) else {}
        return runtime_route_label(route)
    return "Trace step recorded."


def _urgency_chip_html(urgency: str) -> str:
    value = urgency.lower()
    cls = {
        "routine": "blue",
        "monitor": "green",
        "urgent": "amber",
        "emergency": "red",
    }.get(value, "blue")
    return f'<span class="figment-chip {cls}">{_h(value.upper())}</span>'


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, tuple | set):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def _status_text(config: FigmentConfig) -> str:
    audio = "enabled" if config.enable_audio_intake else "disabled"
    return f"`MODEL_STACK={config.model_stack}` | `MODEL_BACKEND={config.model_backend}` | audio {audio}"


def _demo_audio_examples() -> list[list[str]]:
    examples = []
    for filename in DEMO_AUDIO_FILENAMES:
        path = PROJECT_ROOT / "data" / "demo_audio" / filename
        if path.exists():
            examples.append([str(path), ""])
    return examples


def _load_demo_case(name: str) -> list[str]:
    case = DEMO_CASES.get(name or "", next(iter(DEMO_CASES.values())))
    return [
        case["setting"],
        case["patient_age"],
        case["pregnancy_status"],
        case["chief_concern"],
        case["symptoms"],
        case["vitals"],
        case["allergies"],
        case["medications"],
        case["available_supplies"],
        case["responder_note"],
    ]


def _load_demo_case_and_reset(name: str) -> list[Any]:
    return [
        *_load_demo_case(name),
        None,
        "",
        None,
        None,
        _empty_risk_result(),
        _risk_summary_html(_empty_risk_result()),
        [],
        "",
        _protocol_results_html([]),
        {},
        "",
        _navigator_summary_html({}),
        {},
        None,
        _trace_audit_html({}),
        {},
        None,
        {},
    ]


def _empty_risk_result() -> dict[str, Any]:
    return {"red_flags": [], "protocol_urgency": "routine"}


def _clear_source_outputs() -> list[Any]:
    return [
        None,
        _empty_risk_result(),
        _risk_summary_html(_empty_risk_result()),
        [],
        "",
        _protocol_results_html([]),
        {},
        "",
        _navigator_summary_html({}),
        {},
        None,
        _trace_audit_html({}),
        {},
        {},
    ]


def _clear_audio_outputs() -> list[Any]:
    return [
        None,
        None,
        _empty_risk_result(),
        _risk_summary_html(_empty_risk_result()),
        [],
        "",
        _protocol_results_html([]),
        {},
        "",
        _navigator_summary_html({}),
        {},
        None,
        _trace_audit_html({}),
        {},
        None,
        {},
    ]


def _draft_audio_ui(audio_file: str | None, transcript: str, config: FigmentConfig | None = None) -> dict[str, Any]:
    return draft_audio_intake(transcript=transcript, config=config, audio_file=audio_file)


def _should_use_hosted_omni_audio(config: FigmentConfig) -> bool:
    return (
        config.enable_audio_intake
        and config.audio_backend == "omni_native"
        and config.model_backend == "hosted_omni"
    )


def _hosted_audio_disclosure_text() -> str:
    return (
        "Hosted audio is sent to the configured hosted endpoint for drafting; use only synthetic or "
        f"de-identified audio. Hosted upload cap: {hosted_audio_limits_text()}."
    )


def _apply_audio_draft_ui(*values: Any) -> list[Any]:
    *field_values, audio_draft = values
    if not audio_draft:
        return [*field_values, None, None]
    intake = collect_intake(*field_values)
    updated_audio_draft = dict(audio_draft)
    suggestions = []
    for suggestion in audio_draft.get("suggested_fields", []):
        item = dict(suggestion)
        field = str(suggestion.get("field", ""))
        value = str(suggestion.get("draft_value", "")).strip()
        if field in intake and value and not intake.get(field):
            intake[field] = value
            item["status"] = "applied_unreviewed"
            item["needs_confirmation"] = True
        suggestions.append(item)
    updated_audio_draft["suggested_fields"] = suggestions
    if suggestions:
        updated_audio_draft["confirmed_intake_required"] = True
        updated_audio_draft["confirmation_status"] = "unconfirmed"
    fields = [
        intake["setting"],
        intake["patient_age"],
        intake["pregnancy_status"],
        intake["chief_concern"],
        intake["symptoms"],
        intake["vitals"],
        intake["allergies"],
        intake["medications"],
        intake["available_supplies"],
        intake["responder_note"],
    ]
    return [*fields, updated_audio_draft, updated_audio_draft]


def _confirm_ui_intake(*values: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    *field_values, audio_draft = values
    intake = collect_intake(*field_values)
    if audio_draft and audio_draft.get("confirmation_status") != "confirmed":
        edits = {
            str(suggestion.get("field")): str(intake.get(str(suggestion.get("field")), ""))
            for suggestion in audio_draft.get("suggested_fields", [])
            if suggestion.get("field") and intake.get(str(suggestion.get("field")))
        }
        intake, audio_draft = confirm_audio_draft(intake, audio_draft, accept=False, edits=edits)
    confirmed = confirm_intake(intake, audio_draft=audio_draft)
    return confirmed, confirmed, audio_draft


def _risk_ui(intake: dict[str, Any]) -> dict[str, Any]:
    if not intake:
        return _empty_risk_result()
    rules = evaluate_red_flags(intake)
    return {"red_flags": rules, "protocol_urgency": urgency_floor_from_rules(rules)}


def _retrieve_ui(intake: dict[str, Any]) -> list[dict[str, Any]]:
    if not intake:
        return []
    return search_protocol_cards(query_from_intake(intake))


def _retrieve_with_evidence_ui(intake: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    cards = _retrieve_ui(intake)
    return cards, protocol_evidence_panel(cards)


def protocol_evidence_panel(retrieved_cards: list[dict[str, Any]]) -> str:
    if not retrieved_cards:
        return (
            "Prototype evidence/source material for trained-responder review only; "
            "no protocol cards retrieved. Use local protocol, supervisor, clinician, "
            "or emergency pathway rather than improvising."
        )

    lines = [
        "Prototype evidence/source material for trained-responder review only; not medical advice.",
        "",
        "| Card ID | Title | Source | Cue / boundary | Relevance |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in retrieved_cards:
        card = item.get("card") if isinstance(item.get("card"), dict) else item
        card_id = _compact_cell(str(item.get("card_id") or card.get("card_id") or "unknown"))
        title = _compact_cell(str(item.get("title") or card.get("title") or "Untitled card"))
        source = _compact_cell(str(item.get("source") or "unknown"))
        cue = _compact_cell(_evidence_cue(card))
        relevance = _compact_cell(_relevance_text(item))
        lines.append(f"| {card_id} | {title} | {source} | {cue} | {relevance} |")
    return "\n".join(lines)


def _evidence_cue(card: dict[str, Any]) -> str:
    for field in ("escalation_criteria", "red_flags", "safety_boundary"):
        value = card.get(field)
        if isinstance(value, list):
            for item in value:
                text = str(item).strip()
                if text:
                    return text
        elif value:
            return str(value).strip()
    return "No escalation cue or safety boundary summary available."


def _relevance_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    score = result.get("score")
    if isinstance(score, int | float):
        parts.append(f"score={float(score):.2f}")
    elif score not in (None, ""):
        parts.append(f"score={score}")

    snippet = result.get("snippet") or result.get("matched_text") or result.get("summary")
    if snippet:
        parts.append(str(snippet).strip())

    return "; ".join(parts) if parts else "No relevance score or snippet available."


def _compact_cell(value: str, max_chars: int = 180) -> str:
    text = " ".join(value.split())
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text.replace("|", "\\|")


def _navigate_ui(
    intake: dict[str, Any],
    audio_draft: dict[str, Any] | None,
    config: FigmentConfig | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    if not intake:
        return {}, "", {}, {}
    result = run_case(intake, (config or load_config()).validated(), audio_draft=audio_draft)
    return result["navigator_output"], result["sbar"], result["trace"], result["trace"]


if __name__ == "__main__":
    build_app().launch()
