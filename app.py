"""Figment Gradio app scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from figment.audio_intake import confirm_audio_draft as _confirm_audio_draft
from figment.audio_intake import draft_audio_intake as _draft_audio_intake
from figment.config import FigmentConfig, load_config
from figment.model_client import ModelClient, ModelClientError
from figment.navigator import run_navigation
from figment.retrieval import query_from_intake, search_protocol_cards
from figment.rules import evaluate_rules, run_red_flag_checks
from figment.sbar import render_sbar
from figment.trace import stable_hash, write_trace
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
    output, trace = run_navigation(confirmed, rules, audio_draft=audio_draft, config=config or FigmentConfig().validated())
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

    with gr.Blocks(title="Figment") as demo:
        gr.Markdown("## Figment")
        status = gr.Markdown(_status_text(config))
        intake_state = gr.State({})
        audio_state = gr.State(None)
        trace_state = gr.State({})

        with gr.Tabs():
            with gr.Tab(TAB_TITLES[0]):
                demo_case = gr.Dropdown(list(DEMO_CASES), label="Demo case")
                load_demo = gr.Button("Load")
                setting = gr.Textbox(label="Setting")
                patient_age = gr.Textbox(label="Patient age")
                pregnancy_status = gr.Textbox(label="Pregnancy status")
                chief_concern = gr.Textbox(label="Chief concern")
                symptoms = gr.Textbox(label="Symptoms")
                vitals = gr.Textbox(label="Vitals")
                allergies = gr.Textbox(label="Allergies")
                medications = gr.Textbox(label="Medications")
                supplies = gr.Textbox(label="Available supplies")
                note = gr.Textbox(label="Responder note", lines=4)
                audio_clip = gr.Audio(label="Audio intake clip", sources=["microphone", "upload"], type="filepath")
                transcript = gr.Textbox(label="Dictated intake transcript", lines=3)
                draft_btn = gr.Button("Draft Audio Fields")
                audio_json = gr.JSON(label="Audio draft")
                apply_audio = gr.Button("Apply Audio Draft")
                confirm_btn = gr.Button("Confirm Intake")
                intake_json = gr.JSON(label="Confirmed intake")

            with gr.Tab(TAB_TITLES[1]):
                risk_btn = gr.Button("Run Risk Check")
                risk_json = gr.JSON(label="Deterministic red flags")

            with gr.Tab(TAB_TITLES[2]):
                retrieve_btn = gr.Button("Retrieve Protocol Cards")
                guidance_json = gr.JSON(label="Retrieved protocol cards")
                guidance_evidence = gr.Textbox(label="Protocol evidence panel", lines=8, interactive=False)

            with gr.Tab(TAB_TITLES[3]):
                nav_btn = gr.Button("Run Navigator")
                output_json = gr.JSON(label="Navigator output")
                sbar_text = gr.Textbox(label="SBAR handoff", lines=8)

            with gr.Tab(TAB_TITLES[4]):
                trace_json = gr.JSON(label="Trace")
                export_trace = gr.Button("Export Trace")
                trace_file = gr.File(label="Trace download", interactive=False)

        fields = [setting, patient_age, pregnancy_status, chief_concern, symptoms, vitals, allergies, medications, supplies, note]
        load_demo.click(_load_demo_case, inputs=[demo_case], outputs=fields)
        draft_btn.click(
            lambda audio_file, transcript_text: _draft_audio_ui(audio_file, transcript_text, config=config),
            inputs=[audio_clip, transcript],
            outputs=[audio_json],
        ).then(lambda x: x, inputs=[audio_json], outputs=[audio_state])
        apply_audio.click(_apply_audio_draft_ui, inputs=[*fields, audio_state], outputs=[*fields, audio_json, audio_state])
        confirm_btn.click(_confirm_ui_intake, inputs=[*fields, audio_state], outputs=[intake_json, intake_state, audio_state])
        risk_btn.click(_risk_ui, inputs=[intake_state], outputs=[risk_json])
        retrieve_btn.click(_retrieve_with_evidence_ui, inputs=[intake_state], outputs=[guidance_json, guidance_evidence])
        nav_btn.click(
            lambda intake, audio_draft: _navigate_ui(intake, audio_draft, config=config),
            inputs=[intake_state, audio_state],
            outputs=[output_json, sbar_text, trace_json, trace_state],
        )
        export_trace.click(lambda trace: trace_download_path(trace, config=config) if trace else None, inputs=[trace_state], outputs=[trace_file])
        status.value = _status_text(config)
    return demo


def _status_text(config: FigmentConfig) -> str:
    audio = "enabled" if config.enable_audio_intake else "disabled"
    return f"`MODEL_STACK={config.model_stack}` | `MODEL_BACKEND={config.model_backend}` | audio {audio}"


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
        return {"red_flags": [], "protocol_urgency": "routine"}
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
