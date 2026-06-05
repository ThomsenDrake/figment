import importlib
from typing import Any


def test_protocol_guidance_returns_json_and_evidence_panel(monkeypatch: Any) -> None:
    app = importlib.import_module("app")
    retrieved_cards = [
        {
            "card_id": "PED-DEHYD-RED-FLAGS-v1",
            "title": "Pediatric dehydration red flags",
            "score": 8.5,
            "source": "json_fallback",
            "snippet": "Reported lethargy matched red flag wording.",
            "card": {
                "card_id": "PED-DEHYD-RED-FLAGS-v1",
                "title": "Pediatric dehydration red flags",
                "escalation_criteria": [
                    "Any listed red flag requires urgent or emergency escalation according to local protocol."
                ],
                "safety_boundary": (
                    "Prototype protocol navigation card for trained-responder review only."
                ),
            },
        },
        {
            "card_id": "SAFETY-BOUNDARIES-v1",
            "title": "Safety boundaries",
            "score": 3.0,
            "source": "json_fallback",
            "card": {
                "card_id": "SAFETY-BOUNDARIES-v1",
                "title": "Safety boundaries",
                "safety_boundary": "Prototype evidence only; not a medical device.",
            },
        },
    ]
    monkeypatch.setattr(app, "search_protocol_cards", lambda query: retrieved_cards)

    json_cards, evidence_panel = app._retrieve_with_evidence_ui(
        {
            "setting": "shelter clinic",
            "chief_concern": "vomiting and dehydration concern",
            "symptoms": "lethargic, very dry mouth",
        }
    )

    assert json_cards == retrieved_cards
    assert "Prototype evidence/source material" in evidence_panel
    assert "PED-DEHYD-RED-FLAGS-v1" in evidence_panel
    assert "Pediatric dehydration red flags" in evidence_panel
    assert "urgent or emergency escalation according to local protocol" in evidence_panel
    assert "score=8.50" in evidence_panel
    assert "Reported lethargy matched red flag wording." in evidence_panel
    assert "SAFETY-BOUNDARIES-v1" in evidence_panel
    assert "not a medical device" in evidence_panel
