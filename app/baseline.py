"""TF-IDF + Logistic Regression baseline, packaged for serving.

Mirrors the pipeline in notebooks/01_baseline.ipynb so the API can run on a
small CPU box without the fine-tuned DistilBERT weights.

Train and save the model (used as the deploy build step):
    python -m app.baseline
"""

from __future__ import annotations

import csv
import io
import os
import urllib.request
from pathlib import Path

BANKING77_CSV = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
DEFAULT_BASELINE_PATH = Path(os.getenv("INTENTROUTE_BASELINE_PATH", "model_baseline.joblib"))
# Chosen by 5-fold CV on train in notebooks/01_baseline.ipynb.
BEST_C = 10.0


def load_split(split: str) -> tuple[list[str], list[str]]:
    with urllib.request.urlopen(f"{BANKING77_CSV}{split}.csv", timeout=60) as resp:
        rows = list(csv.DictReader(io.StringIO(resp.read().decode("utf-8"))))
    return [r["text"] for r in rows], [r["category"] for r in rows]


def train(path: Path = DEFAULT_BASELINE_PATH) -> float:
    """Fit on the official train split, save to `path`, return test accuracy."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=BEST_C)),
    ])
    pipeline.fit(x_train, y_train)
    accuracy = float(pipeline.score(x_test, y_test))
    joblib.dump(pipeline, path)
    return accuracy


class BaselineIntentClassifier:
    """Serves the saved scikit-learn pipeline behind the same interface as the HF model."""

    def __init__(self, path: Path = DEFAULT_BASELINE_PATH) -> None:
        import joblib

        if not path.is_file():
            raise RuntimeError(f"Baseline model '{path}' not found. Run `python -m app.baseline` first.")
        self.pipeline = joblib.load(path)
        self.classes = list(self.pipeline.classes_)

    def predict(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        probs = self.pipeline.predict_proba([text])[0]
        ranked = sorted(zip(self.classes, probs), key=lambda pair: pair[1], reverse=True)
        return [(label, float(p)) for label, p in ranked[:top_k]]


if __name__ == "__main__":
    acc = train()
    print(f"Saved {DEFAULT_BASELINE_PATH} (test accuracy {acc:.1%})")
