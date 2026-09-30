"""IntentRoute inference API.

Serves the fine-tuned DistilBERT Banking77 classifier produced by
notebooks/02_distilbert.ipynb.

Endpoints
---------
GET  /health   -> liveness + whether the model is loaded
POST /predict  -> top-3 intents with softmax confidences and a routing decision

Run locally:
    uvicorn app.main:app --reload

The model directory defaults to ./model and can be overridden with the
INTENTROUTE_MODEL_DIR environment variable.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

# Must match the max_len used during fine-tuning (notebooks/02_distilbert.ipynb).
MAX_LENGTH = 64
TOP_K = 3
# Predictions whose top confidence falls below this go to a human agent.
CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MODEL_DIR = Path(os.getenv("INTENTROUTE_MODEL_DIR", "model"))
STATIC_DIR = Path(__file__).parent / "static"


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, examples=["I still have not received my new card"])

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class IntentScore(BaseModel):
    intent: str
    confidence: float


class PredictResponse(BaseModel):
    intents: list[IntentScore]
    route: Literal["auto", "human_review"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class Classifier(Protocol):
    """Anything that maps a text to (intent, probability) pairs, best first."""

    def predict(self, text: str, top_k: int = TOP_K) -> list[tuple[str, float]]: ...


class HFIntentClassifier:
    """Wraps a Hugging Face sequence-classification checkpoint for CPU/GPU inference."""

    def __init__(self, model_dir: Path) -> None:
        # Heavy imports live here so tests that inject a fake classifier stay fast.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not model_dir.is_dir():
            raise RuntimeError(
                f"Model directory '{model_dir}' not found. Train it with "
                "notebooks/02_distilbert.ipynb or set INTENTROUTE_MODEL_DIR."
            )

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        # id2label is written into config.json by the training notebook.
        self.id2label: dict[int, str] = self.model.config.id2label

    def predict(self, text: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
        inputs = self.tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(self.device)
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits[0]
        probs = self._torch.softmax(logits, dim=-1)
        top = self._torch.topk(probs, k=min(top_k, probs.numel()))
        return [
            (self.id2label[int(idx)], float(p))
            for p, idx in zip(top.values.tolist(), top.indices.tolist())
        ]


def load_default_classifier(model_dir: Path) -> Classifier:
    """Prefer the fine-tuned DistilBERT; fall back to the TF-IDF baseline when its weights are absent."""
    if model_dir.is_dir():
        return HFIntentClassifier(model_dir)
    from app.baseline import BaselineIntentClassifier

    return BaselineIntentClassifier()


def route_for(top_confidence: float) -> Literal["auto", "human_review"]:
    """Low-confidence predictions are escalated to a human instead of auto-handled."""
    return "human_review" if top_confidence < CONFIDENCE_THRESHOLD else "auto"


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(classifier: Classifier | None = None, model_dir: Path = DEFAULT_MODEL_DIR) -> FastAPI:
    """Build the API.

    Pass `classifier` to inject a pre-built model (used by tests); otherwise the
    Hugging Face model is loaded from `model_dir` once, at startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier if classifier is not None else load_default_classifier(model_dir)
        yield
        app.state.classifier = None

    app = FastAPI(
        title="IntentRoute",
        description="Banking77 customer-support intent classifier with confidence-based routing.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    def demo() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded = getattr(request.app.state, "classifier", None) is not None
        return HealthResponse(status="ok", model_loaded=loaded)

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        # Sync endpoint: FastAPI runs it in a threadpool, so inference doesn't block the event loop.
        ranked = request.app.state.classifier.predict(body.text, top_k=TOP_K)
        intents = [IntentScore(intent=label, confidence=round(conf, 4)) for label, conf in ranked]
        # Route on the unrounded score so rounding can never flip the decision.
        return PredictResponse(intents=intents, route=route_for(ranked[0][1]))

    return app


# ASGI entrypoint for `uvicorn app.main:app`.
app = create_app()
