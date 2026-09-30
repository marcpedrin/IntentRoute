"""API tests for app/main.py.

Most tests inject a deterministic fake classifier so they run in milliseconds
and do not need trained weights. The last test exercises the real fine-tuned
model and is skipped automatically when ./model (or INTENTROUTE_MODEL_DIR)
does not exist.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import CONFIDENCE_THRESHOLD, DEFAULT_MODEL_DIR, TOP_K, create_app


class FakeClassifier:
    """Returns canned, already-sorted probability lists keyed by input text."""

    RESPONSES = {
        # Clear-cut query: one dominant intent.
        "I still have not received my new card": [
            ("card_arrival", 0.91),
            ("card_delivery_estimate", 0.05),
            ("lost_or_stolen_card", 0.01),
        ],
        # Ambiguous query: probability mass spread across intents.
        "money": [
            ("transfer_not_received_by_recipient", 0.34),
            ("balance_not_updated_after_bank_transfer", 0.29),
            ("pending_transfer", 0.12),
        ],
        # Exactly at the threshold: must be auto-routed (rule is strictly "< 0.5").
        "boundary": [
            ("exchange_rate", CONFIDENCE_THRESHOLD),
            ("exchange_charge", 0.3),
            ("exchange_via_app", 0.1),
        ],
    }

    def predict(self, text: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
        return self.RESPONSES[text][:top_k]


@pytest.fixture
def client():
    # Using TestClient as a context manager runs the lifespan (startup) hook.
    with TestClient(create_app(classifier=FakeClassifier())) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_predict_confident_query_routes_auto(client):
    resp = client.post("/predict", json={"text": "I still have not received my new card"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["route"] == "auto"
    assert len(body["intents"]) == 3
    assert body["intents"][0] == {"intent": "card_arrival", "confidence": 0.91}
    confidences = [i["confidence"] for i in body["intents"]]
    assert confidences == sorted(confidences, reverse=True)


def test_predict_low_confidence_routes_to_human_review(client):
    resp = client.post("/predict", json={"text": "money"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["intents"][0]["confidence"] < CONFIDENCE_THRESHOLD
    assert body["route"] == "human_review"
    assert len(body["intents"]) == 3  # top-3 is still returned for the agent's context


def test_predict_threshold_boundary_routes_auto(client):
    resp = client.post("/predict", json={"text": "boundary"})
    assert resp.status_code == 200
    assert resp.json()["route"] == "auto"


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": 123}])
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "config.json").exists(),
    reason="fine-tuned model not found; run notebooks/02_distilbert.ipynb first",
)
def test_predict_with_real_model():
    with TestClient(create_app()) as c:
        body = c.post("/predict", json={"text": "How do I top up my account?"}).json()

    assert body["route"] in {"auto", "human_review"}
    assert len(body["intents"]) == 3
    confidences = [i["confidence"] for i in body["intents"]]
    assert all(0.0 <= p <= 1.0 for p in confidences)
    assert confidences == sorted(confidences, reverse=True)
    assert sum(confidences) <= 1.0 + 1e-3
    # Routing must be consistent with the threshold rule (skip if rounding hides the side).
    if abs(confidences[0] - CONFIDENCE_THRESHOLD) > 1e-4:
        expected = "human_review" if confidences[0] < CONFIDENCE_THRESHOLD else "auto"
        assert body["route"] == expected
