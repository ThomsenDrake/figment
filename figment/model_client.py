"""Model client adapters for hosted, local, and canned Figment modes."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
import urllib.request
from typing import Any

from .config import FigmentConfig, load_config


class ModelClientError(RuntimeError):
    """Raised when a configured model backend cannot return a usable response."""


NVIDIA_HOSTED_MAX_TOKENS = 8192
DEFAULT_MAX_TOKENS = 4096

AUDIO_DRAFT_PROMPT = """Transcribe this responder audio and return ONLY JSON with:
{
  "transcript": "verbatim or lightly normalized transcript",
  "suggested_fields": [
    {"field": "one allowed field name", "draft_value": "", "source_snippet": "", "source_timecode": ""}
  ],
  "missing_or_unclear_fields": [],
  "provisional_red_flag_mentions": []
}
Allowed field names are: setting, patient_age, pregnancy_status, chief_concern, symptoms, vitals, allergies, medications, available_supplies, responder_note.
Each suggested_fields item must use exactly one allowed field name, not a pipe-delimited list.
The output is a provisional intake draft for a trained responder. Do not diagnose, prescribe, or decide urgency. Do not include raw audio data."""


def canned_navigator_output(
    intake: dict[str, Any],
    rule_results: list[dict[str, Any]],
    retrieved_cards: list[dict[str, Any]],
    urgency_floor: str = "routine",
) -> dict[str, Any]:
    """Return a deterministic demo-safe navigator output."""
    cards = [item.get("card", item) for item in retrieved_cards]
    source_cards = [str(card.get("card_id")) for card in cards if card.get("card_id")]
    if not source_cards:
        source_cards = ["SAFETY-BOUNDARIES-v1"]

    concern = intake.get("chief_concern") or intake.get("responder_note") or "Reported field concern"
    red_labels = [rule.get("label", rule.get("rule_id", "")) for rule in rule_results]
    escalation = "; ".join(red_labels) if red_labels else "No deterministic red flag fired"
    return {
        "protocol_urgency": urgency_floor,
        "red_flags": rule_results,
        "intake_facts": [
            {"fact": f"Concern: {concern}", "status": "reported", "source": "structured_field"},
            {"fact": f"Vitals: {intake.get('vitals') or 'not recorded'}", "status": "reported" if intake.get("vitals") else "missing", "source": "structured_field"},
        ],
        "candidate_protocol_pathways": [
            {
                "card_id": card_id,
                "reason_relevant": "Retrieved from the confirmed intake and deterministic rule context.",
            }
            for card_id in source_cards[:3]
        ],
        "missing_info_to_collect": ["repeat vitals", "time course", "available referral route"],
        "next_observations_to_collect": ["level of alertness", "work of breathing", "hydration/perfusion signs"],
        "conflicts_or_uncertainties": ["Prototype output; responder must verify all observations."],
        "responder_checklist": [
            "Keep deterministic red-flag result visible.",
            "Collect the missing observations before relying on protocol guidance.",
            "Use local escalation pathway if the case worsens or protocol cards do not fit.",
        ],
        "do_not_do": [
            "Do not assign a condition label.",
            "Do not add medication instructions beyond cited local protocol text.",
            "Do not downgrade red flags.",
        ],
        "source_cards": source_cards[:6],
        "handoff_note_sbar": {
            "situation": str(concern),
            "background": f"Setting: {intake.get('setting', 'field setting')}. {escalation}.",
            "assessment_observations_only": f"Observed/reported: {intake.get('symptoms') or intake.get('responder_note') or 'details pending'}. Vitals: {intake.get('vitals') or 'missing'}.",
            "handoff_request": "Request review/escalation per cited local protocol cards.",
        },
        "responder_plain_language_script": "I am going to check the next protocol observations and escalate if the danger signs remain present.",
        "safety_boundary": "Prototype protocol navigation only; no condition label, medication order, or autonomous routing.",
    }


class ModelClient:
    def __init__(self, config: FigmentConfig | None = None, timeout_seconds: float = 45.0):
        self.config = (config or load_config()).validated()
        self.timeout_seconds = timeout_seconds

    def generate_json(self, prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        if self.config.model_backend == "canned":
            return canned_navigator_output(
                context.get("intake", {}),
                context.get("rule_results", []),
                context.get("retrieved_cards", []),
                context.get("urgency_floor", "routine"),
            )
        if self.config.model_backend == "llama_cpp":
            return self._call_openai_compatible(
                self.config.llama_base_url,
                prompt,
                model_id=self.config.local_model_id,
                auth_headers={},
            )
        if self.config.model_backend == "hosted_omni":
            endpoint = self.config.omni_endpoint_url or self.config.hf_endpoint_url or self.config.nvidia_base_url
            if not endpoint:
                raise ModelClientError("hosted model backend requires NVIDIA_BASE_URL, OMNI_ENDPOINT_URL, or HF_ENDPOINT_URL")
            is_nvidia_endpoint = "integrate.api.nvidia.com" in endpoint
            return self._call_openai_compatible(
                endpoint,
                prompt,
                model_id=self.config.active_model_id,
                auth_headers=self._hosted_auth_headers(endpoint),
                include_nvidia_options=is_nvidia_endpoint,
            )
        raise ModelClientError(f"unsupported model backend: {self.config.model_backend}")

    def generate_audio_draft(self, audio_path: str | Path) -> dict[str, Any]:
        if self.config.model_backend != "hosted_omni":
            raise ModelClientError("hosted Omni audio drafting requires MODEL_BACKEND=hosted_omni")
        endpoint = self.config.omni_endpoint_url or self.config.hf_endpoint_url or self.config.nvidia_base_url
        if not endpoint:
            raise ModelClientError("hosted Omni audio drafting requires NVIDIA_BASE_URL, OMNI_ENDPOINT_URL, or HF_ENDPOINT_URL")
        is_nvidia_endpoint = "integrate.api.nvidia.com" in endpoint
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": AUDIO_DRAFT_PROMPT},
                    {"type": "audio_url", "audio_url": {"url": _audio_file_data_url(audio_path)}},
                ],
            }
        ]
        return self._call_openai_compatible_messages(
            endpoint,
            messages,
            model_id=self.config.active_model_id,
            auth_headers=self._hosted_auth_headers(endpoint),
            include_nvidia_options=is_nvidia_endpoint,
        )

    def _call_openai_compatible(
        self,
        base_url: str,
        prompt: str,
        *,
        model_id: str,
        auth_headers: dict[str, str],
        include_nvidia_options: bool = False,
    ) -> dict[str, Any]:
        return self._call_openai_compatible_messages(
            base_url,
            [{"role": "user", "content": prompt}],
            model_id=model_id,
            auth_headers=auth_headers,
            include_nvidia_options=include_nvidia_options,
        )

    def _call_openai_compatible_messages(
        self,
        base_url: str,
        messages: list[dict[str, Any]],
        *,
        model_id: str,
        auth_headers: dict[str, str],
        include_nvidia_options: bool = False,
    ) -> dict[str, Any]:
        url = base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        body = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        if include_nvidia_options:
            body["max_tokens"] = NVIDIA_HOSTED_MAX_TOKENS
            body["stream"] = False
            body["chat_template_kwargs"] = {"enable_thinking": False}
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", **auth_headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelClientError(f"model backend failed: {exc}") from exc
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelClientError("model response did not include choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ModelClientError("model response choice was not an object")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ModelClientError("model response did not include a message")
        content = _message_content_text(message)
        try:
            return _parse_json_object(content)
        except json.JSONDecodeError as exc:
            raise ModelClientError("model response was not valid JSON") from exc

    def _hosted_auth_headers(self, endpoint: str) -> dict[str, str]:
        if "integrate.api.nvidia.com" in endpoint:
            if not self.config.nvidia_api_key:
                raise ModelClientError("hosted NVIDIA backend requires NVIDIA_API_KEY")
            return {"Authorization": f"Bearer {self.config.nvidia_api_key}"}
        if self.config.hf_token:
            return {"Authorization": f"Bearer {self.config.hf_token}"}
        return {}


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise
    if isinstance(parsed, dict):
        return parsed
    raise json.JSONDecodeError("model response JSON was not an object", text, 0)


def _message_content_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _message_part_text(content)
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                part_text = _message_part_text(part)
                if part_text:
                    text_parts.append(part_text)
        if text_parts:
            return "\n".join(text_parts)
        raise ModelClientError("model response content did not include text parts")
    raise ModelClientError("model response content was not text")


def _message_part_text(part: dict[str, Any]) -> str:
    for key in ("text", "content", "value"):
        value = part.get(key)
        if isinstance(value, str):
            return value
    return ""


def _audio_file_data_url(audio_path: str | Path) -> str:
    path = Path(audio_path)
    try:
        audio_bytes = path.read_bytes()
    except OSError as exc:
        raise ModelClientError(f"could not read audio file for hosted Omni draft: {exc}") from exc
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    if not mime_type.startswith("audio/"):
        mime_type = "audio/wav"
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
