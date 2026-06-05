"""Model client adapters for hosted, local, and canned Figment modes."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import FigmentConfig, load_config


class ModelClientError(RuntimeError):
    """Raised when a configured model backend cannot return a usable response."""


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
            return self._call_openai_compatible(self.config.llama_base_url, prompt)
        if self.config.model_backend in {"hosted_omni", "hosted_text_nemotron"}:
            endpoint = self.config.omni_endpoint_url or self.config.hf_endpoint_url
            if not endpoint:
                raise ModelClientError("hosted model backend requires OMNI_ENDPOINT_URL or HF_ENDPOINT_URL")
            return self._call_openai_compatible(endpoint, prompt)
        raise ModelClientError(f"unsupported model backend: {self.config.model_backend}")

    def _call_openai_compatible(self, base_url: str, prompt: str) -> dict[str, Any]:
        url = base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        body = {
            "model": self.config.active_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", **self._auth_headers()},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelClientError(f"model backend failed: {exc}") from exc
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        try:
            return _parse_json_object(content)
        except json.JSONDecodeError as exc:
            raise ModelClientError("model response was not valid JSON") from exc

    def _auth_headers(self) -> dict[str, str]:
        if not self.config.hf_token:
            return {}
        return {"Authorization": f"Bearer {self.config.hf_token}"}


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
