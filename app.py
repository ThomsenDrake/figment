"""Figment Gradio app scaffold."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from figment.audio_intake import confirm_audio_draft as _confirm_audio_draft
from figment.audio_intake import draft_audio_intake as _draft_audio_intake
from figment.config import FigmentConfig, load_config
from figment.model_client import ModelClient, ModelClientError
from figment.navigator import run_navigation
from figment.retrieval import load_protocol_cards, query_from_intake, search_protocol_cards
from figment.rules import evaluate_rules, run_red_flag_checks
from figment.sbar import render_sbar
from figment.trace import stable_hash, write_trace
from figment.ui_theme import FIGMENT_CSS
from figment.validators import urgency_floor_from_rules, validate_audio_ready

try:
    import gradio as gr
except (ImportError, OSError):  # pragma: no cover - lets unit tests import without gradio installed
    gr = None


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
        draft["audio_retention_note"] = (
            "Original clip bytes are not written to Figment traces; Gradio may keep upload/session files "
            "while the app is running, and committed demo clips stay on disk."
        )
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
    output, trace = run_navigation(confirmed, rules, audio_draft=audio_draft, config=runtime_config)
    evaluation = evaluate_rules(confirmed)
    return {
        "intake": confirmed,
        "risk": evaluation,
        "retrieved_cards": search_protocol_cards(query_from_intake(confirmed)),
        "navigator_output": output,
        "sbar": render_sbar(output, trace.validator_result),
        "trace": trace.to_dict(),
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

    with gr.Blocks(title="Figment", css=FIGMENT_CSS, theme=gr.themes.Base(), fill_width=True) as demo:
        gr.HTML(_app_header_html())
        gr.HTML(_statusline_html(config))
        intake_state = gr.State({})
        audio_state = gr.State(None)
        trace_state = gr.State({})

        with gr.Tabs(elem_classes=["figment-tabs"]):
            with gr.Tab(TAB_TITLES[0]):
                with gr.Column(elem_classes=["figment-tab-body"]):
                    with gr.Row():
                        with gr.Column(scale=11, elem_classes=["figment-panel"]):
                            gr.HTML(_section_header_html("1. Quick start", "Load a frozen synthetic demo case, or type directly into the confirmed intake form."))
                            with gr.Row():
                                demo_case = gr.Dropdown(list(DEMO_CASES), label="Demo case", scale=4)
                                load_demo = gr.Button("Load", scale=1)
                            gr.HTML(_demo_case_pills_html())

                            gr.HTML(_section_header_html("2. Omni audio intake", "Audio only drafts editable fields. Manual entries and edits always win."))
                            with gr.Row():
                                with gr.Column(scale=3):
                                    audio_clip = gr.Audio(label="Audio intake clip", sources=["microphone", "upload"], type="filepath")
                                with gr.Column(scale=2):
                                    draft_btn = gr.Button("Draft Audio Fields", elem_classes=["primary"])
                                    apply_audio = gr.Button("Apply Audio Draft")
                                    gr.HTML('<span class="figment-chip amber">Confirm before rules run</span>')
                            transcript = gr.Textbox(label="Dictated intake transcript", lines=3)
                            if examples := _demo_audio_examples():
                                gr.Examples(examples=examples, inputs=[audio_clip, transcript], label="Demo audio clips")

                            gr.HTML(_section_header_html("3. Confirmed intake", "Protocol rules and navigation run only after this intake is confirmed."))
                            with gr.Row():
                                setting = gr.Textbox(label="Setting")
                                patient_age = gr.Textbox(label="Patient age")
                                pregnancy_status = gr.Textbox(label="Pregnancy status")
                            chief_concern = gr.Textbox(label="Chief concern")
                            symptoms = gr.Textbox(label="Symptoms")
                            vitals = gr.Textbox(label="Vitals")
                            with gr.Row():
                                allergies = gr.Textbox(label="Allergies")
                                medications = gr.Textbox(label="Medications")
                            supplies = gr.Textbox(label="Available supplies")
                            note = gr.Textbox(label="Responder note", lines=4)

                        with gr.Column(scale=9, elem_classes=["figment-panel"]):
                            gr.HTML(_section_header_html("Audio draft field suggestions", "Review timecoded suggestions before applying them to the editable intake."))
                            audio_json = gr.JSON(label="Audio draft", elem_classes=["figment-json-compact"])
                            gr.HTML(_section_header_html("Live confirmed intake preview", "This is the only source allowed to feed deterministic rules and navigation."))
                            intake_json = gr.JSON(label="Confirmed intake", elem_classes=["figment-json-compact"])
                            confirm_btn = gr.Button("Confirm Intake", elem_classes=["primary"])

            with gr.Tab(TAB_TITLES[1]):
                with gr.Column(elem_classes=["figment-tab-body"]):
                    with gr.Row():
                        with gr.Column(scale=8, elem_classes=["figment-panel"]):
                            gr.HTML(_section_header_html("Deterministic Red-Flag Checklist", "Reference checklist for the frozen safety floor. These rules are deterministic."))
                            gr.HTML(_red_flag_checklist_html())
                        with gr.Column(scale=10, elem_classes=["figment-panel"]):
                            gr.HTML(_section_header_html("Rule Output", "The model cannot lower the deterministic protocol_urgency floor."))
                            risk_btn = gr.Button("Run Risk Check", elem_classes=["primary"])
                            risk_html = gr.HTML(_risk_summary_html(_empty_risk_result()))
                            with gr.Accordion("Raw deterministic red flags JSON", open=False):
                                risk_json = gr.JSON(label="Deterministic red flags", elem_classes=["figment-json-compact"])

            with gr.Tab(TAB_TITLES[2]):
                with gr.Column(elem_classes=["figment-tab-body"]):
                    with gr.Row():
                        with gr.Column(scale=8, elem_classes=["figment-panel"]):
                            gr.HTML(_section_header_html("Protocol Card Browser", "Local protocol cards retrieved from the confirmed intake."))
                            gr.HTML(_protocol_library_html())
                            retrieve_btn = gr.Button("Retrieve Protocol Cards", elem_classes=["primary"])
                        with gr.Column(scale=10, elem_classes=["figment-panel"]):
                            guidance_html = gr.HTML(_protocol_results_html([]))
                            guidance_evidence = gr.Textbox(label="Protocol evidence panel", lines=8, interactive=False)
                            with gr.Accordion("Retrieved protocol cards JSON", open=False):
                                guidance_json = gr.JSON(label="Retrieved protocol cards", elem_classes=["figment-json-compact"])

            with gr.Tab(TAB_TITLES[3]):
                with gr.Column(elem_classes=["figment-tab-body"]):
                    with gr.Row():
                        with gr.Column(scale=8, elem_classes=["figment-panel"]):
                            gr.HTML(_section_header_html("Navigator Output JSON", "Machine-readable protocol navigation output."))
                            nav_btn = gr.Button("Run Navigator", elem_classes=["primary"])
                            output_json = gr.JSON(label="Navigator output", elem_classes=["figment-json-tall"])
                        with gr.Column(scale=10, elem_classes=["figment-panel"]):
                            navigator_html = gr.HTML(_navigator_summary_html({}))
                            sbar_text = gr.Textbox(label="SBAR handoff", lines=8)

            with gr.Tab(TAB_TITLES[4]):
                with gr.Column(elem_classes=["figment-tab-body"]):
                    with gr.Row():
                        with gr.Column(scale=8, elem_classes=["figment-panel"]):
                            gr.HTML(_section_header_html("Run Steps (Timeline)", "Audit trail from intake through validation."))
                            trace_audit_html = gr.HTML(_trace_audit_html({}))
                            export_trace = gr.Button("Export Trace")
                            trace_file = gr.File(label="Trace download", interactive=False)
                        with gr.Column(scale=10, elem_classes=["figment-panel"]):
                            gr.HTML(_section_header_html("Trace JSON", "Raw audit object for review and export."))
                            trace_json = gr.JSON(label="Trace", elem_classes=["figment-json-tall"])

        gr.HTML(_footer_rail_html(config))

        fields = [setting, patient_age, pregnancy_status, chief_concern, symptoms, vitals, allergies, medications, supplies, note]
        source_outputs = [
            intake_json,
            risk_json,
            risk_html,
            guidance_json,
            guidance_evidence,
            guidance_html,
            output_json,
            sbar_text,
            navigator_html,
            trace_json,
            trace_file,
            trace_audit_html,
            intake_state,
            trace_state,
        ]
        audio_source_outputs = [
            audio_json,
            intake_json,
            risk_json,
            risk_html,
            guidance_json,
            guidance_evidence,
            guidance_html,
            output_json,
            sbar_text,
            navigator_html,
            trace_json,
            trace_file,
            trace_audit_html,
            intake_state,
            audio_state,
            trace_state,
        ]
        load_demo.click(
            _load_demo_case_and_reset,
            inputs=[demo_case],
            outputs=[
                *fields,
                audio_clip,
                transcript,
                audio_json,
                intake_json,
                risk_json,
                risk_html,
                guidance_json,
                guidance_evidence,
                guidance_html,
                output_json,
                sbar_text,
                navigator_html,
                trace_json,
                trace_file,
                trace_audit_html,
                intake_state,
                audio_state,
                trace_state,
            ],
        )
        draft_btn.click(
            lambda audio_file, transcript_text: _draft_audio_ui(audio_file, transcript_text, config=config),
            inputs=[audio_clip, transcript],
            outputs=[audio_json],
        ).then(lambda x: x, inputs=[audio_json], outputs=[audio_state]).then(_clear_source_outputs, outputs=source_outputs)
        apply_audio.click(_apply_audio_draft_ui, inputs=[*fields, audio_state], outputs=[*fields, audio_json, audio_state]).then(
            _clear_source_outputs,
            outputs=source_outputs,
        )
        for source in fields:
            source.change(_clear_source_outputs, outputs=source_outputs)
        audio_clip.change(_clear_audio_outputs, outputs=audio_source_outputs)
        transcript.change(_clear_audio_outputs, outputs=audio_source_outputs)
        confirm_btn.click(_confirm_ui_intake, inputs=[*fields, audio_state], outputs=[intake_json, intake_state, audio_state])
        risk_btn.click(_risk_ui_with_summary, inputs=[intake_state], outputs=[risk_json, risk_html])
        retrieve_btn.click(_retrieve_with_evidence_and_summary_ui, inputs=[intake_state], outputs=[guidance_json, guidance_evidence, guidance_html])
        nav_btn.click(
            lambda intake, audio_draft: _navigate_ui_with_summary(intake, audio_draft, config=config),
            inputs=[intake_state, audio_state],
            outputs=[output_json, sbar_text, trace_json, trace_state, navigator_html, trace_audit_html],
        )
        export_trace.click(lambda trace: trace_download_path(trace, config=config) if trace else None, inputs=[trace_state], outputs=[trace_file])
    return demo


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
    backend_chip = "blue" if config.model_backend == "hosted_omni" else "amber"
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


def _model_mode_label(config: FigmentConfig) -> str:
    if config.model_backend == "hosted_omni":
        return "Hosted Omni (live)"
    if config.model_backend == "llama_cpp":
        return "Local 4B text navigator"
    return "Canned Trace (offline)"


def _section_header_html(title: str, subtitle: str = "") -> str:
    subtitle_html = f'<div class="figment-section-subtitle">{_h(subtitle)}</div>' if subtitle else ""
    return f'<div class="figment-section-title">{_h(title)}</div>{subtitle_html}'


def _demo_case_pills_html() -> str:
    pills = "".join(f'<div class="figment-demo-pill">{_h(name)}</div>' for name in DEMO_CASES)
    return f'<div class="figment-quick-grid">{pills}</div>'


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
      <thead><tr><th>Card ID</th><th>Matched Context</th><th>Relevance Reason</th></tr></thead>
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
    return output, sbar, trace, trace_state, _navigator_summary_html(output), _trace_audit_html(trace)


def _navigator_summary_html(output: dict[str, Any]) -> str:
    if not output:
        return """
        <div class="figment-section-title">Protocol Urgency</div>
        <div class="figment-panel-soft">Run the navigator after confirming intake and red-flag checks.</div>
        """
    urgency = str(output.get("protocol_urgency") or "routine").lower()
    if urgency not in {"routine", "monitor", "urgent", "emergency"}:
        urgency = "routine"
    handoff = output.get("handoff_note_sbar") if isinstance(output.get("handoff_note_sbar"), dict) else {}
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


def _navigator_list_card_html(title: str, values: Any, *, checked: bool = False) -> str:
    items = _as_list(values) or ["No items generated yet."]
    cls = "figment-checklist checked" if checked else "figment-checklist"
    return (
        '<div class="figment-mini-card">'
        f'<h4>{_h(title)}</h4>'
        f'<ul class="{cls}">{"".join(f"<li>{_h(item)}</li>" for item in items)}</ul>'
        "</div>"
    )


def _trace_audit_html(trace: dict[str, Any]) -> str:
    if not trace:
        return """
        <div class="figment-panel-soft">Run the navigator to populate timeline, validation, model route, and trace metadata.</div>
        """
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
        <span class="figment-chip green">Source cards present</span>
      </div>
      <div class="figment-mini-card">
        <h4>Model &amp; Performance</h4>
        <table class="figment-table">
          <tbody>
            <tr><td>Model mode</td><td>{_h(route.get('model_backend'))}</td></tr>
            <tr><td>Model ID</td><td>{_h(route.get('model_id'))}</td></tr>
            <tr><td>Fallback tier</td><td>{_h(route.get('fallback_tier'))}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
    """


def _trace_event_detail(event: str, trace: dict[str, Any]) -> str:
    if "rules" in event:
        return f"{len(trace.get('red_flags') or [])} deterministic red-flag result(s)."
    if "cards" in event:
        return f"{len(trace.get('retrieved_card_ids') or [])} protocol card(s) retrieved."
    if "validation" in event:
        validator = trace.get("validator_result") if isinstance(trace.get("validator_result"), dict) else {}
        return "Output conforms to schema." if validator.get("passed") else "Validation failures present."
    if "model" in event or "navigator" in event:
        route = trace.get("model_route") if isinstance(trace.get("model_route"), dict) else {}
        return str(route.get("model_backend") or "navigator output generated")
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


def _apply_audio_draft_ui(*values: Any) -> list[Any]:
    *field_values, audio_draft = values
    if not audio_draft:
        return [*field_values, None, None]
    intake = collect_intake(*field_values)
    for suggestion in audio_draft.get("suggested_fields", []):
        field = str(suggestion.get("field", ""))
        value = str(suggestion.get("draft_value", "")).strip()
        if field in intake and value and not intake.get(field):
            intake[field] = value
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
    return [*fields, audio_draft, audio_draft]


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
        "| Card ID | Title | Cue / boundary | Relevance |",
        "| --- | --- | --- | --- |",
    ]
    for item in retrieved_cards:
        card = item.get("card") if isinstance(item.get("card"), dict) else item
        card_id = _compact_cell(str(item.get("card_id") or card.get("card_id") or "unknown"))
        title = _compact_cell(str(item.get("title") or card.get("title") or "Untitled card"))
        cue = _compact_cell(_evidence_cue(card))
        relevance = _compact_cell(_relevance_text(item))
        lines.append(f"| {card_id} | {title} | {cue} | {relevance} |")
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
    build_app().queue().launch()
