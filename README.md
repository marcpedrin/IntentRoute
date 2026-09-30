# IntentRoute

Customer-support **intent classification** with confidence-based routing, built on the Banking77 dataset.

## Problem

A bank's support inbox receives free-text messages such as *"my card still hasn't arrived"* or *"why was I charged a fee for this transfer?"*. IntentRoute classifies each message into one of **77 intents** so it can be sent to the right workflow automatically. When the model isn't confident, the message goes to a human agent instead of being guessed at:

```
             ┌──────────────┐   top-1 conf ≥ 0.5   ┌────────────────────┐
 message ──► │  DistilBERT  │ ───────────────────► │ route: auto        │
             │  classifier  │                      └────────────────────┘
             │  (top-3 +    │   top-1 conf < 0.5   ┌────────────────────┐
             │   softmax)   │ ───────────────────► │ route: human_review│
             └──────────────┘                      └────────────────────┘
```

## Dataset

[`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) on the Hugging Face Hub (Casanueva et al., 2020):

* 77 fine-grained intents from the online-banking domain
* 10,003 train / 3,080 test examples (official splits; all results below use the **test** split)
* Short, single-turn customer queries

> **Loading note:** the Hub repo `PolyAI/banking77` is a legacy loading script, and `datasets>=4` refuses to run scripts (`RuntimeError: Dataset scripts are no longer supported`). Both notebooks therefore use a small `load_banking77()` helper. It loads the same official CSVs the script points to with `datasets.load_dataset("csv", ...)`, and takes the canonical 77-label order from the repo's `dataset_infos.json` (pinned to a commit). Label ids and splits are identical to the Hub dataset.

## Results

Fill these in from the notebook outputs. The numbers come from `results/baseline_metrics.json`, `results/distilbert_metrics.json` and `results/comparison.csv`.

| Model | Accuracy (test) | Macro-F1 (test) | Notes |
|---|---|---|---|
| TF-IDF (word 1–2 grams) + Logistic Regression | 89.4% | 89.4% | `C=10.0` chosen by 5-fold CV on train |
| DistilBERT (`distilbert-base-uncased`, fine-tuned) | `TODO` | `TODO` | 3 epochs, lr 5e-5, batch 32, max_len 64 |

**Top-10 most-confused intent pairs (baseline):** see `results/baseline_confused_pairs.csv`, or paste it here:

| Rank | Intent A | Intent B | Total confusions |
|---|---|---|---|
| 1 | `verify_my_identity` | `why_verify_identity` | 12 |
| 2 | `balance_not_updated_after_bank_transfer` | `transfer_not_received_by_recipient` | 8 |
| 3 | `card_payment_wrong_exchange_rate` | `wrong_exchange_rate_for_cash_withdrawal` | 7 |
| 4 | `top_up_failed` | `top_up_reverted` | 7 |
| 5 | `unable_to_verify_identity` | `verify_my_identity` | 7 |
| 6 | `balance_not_updated_after_bank_transfer` | `pending_transfer` | 6 |
| 7 | `card_acceptance` | `card_not_working` | 6 |
| 8 | `get_physical_card` | `pin_blocked` | 6 |
| 9 | `card_payment_not_recognised` | `extra_charge_on_statement` | 5 |
| 10 | `exchange_via_app` | `fiat_currency_support` | 5 |

## Project layout

```
intentroute/
├── notebooks/
│   ├── 01_baseline.ipynb      # TF-IDF + LogReg baseline, metrics, confusion analysis
│   └── 02_distilbert.ipynb    # DistilBERT fine-tuning (Colab T4), comparison, saves ./model
├── app/
│   └── main.py                # FastAPI service: POST /predict, GET /health
├── tests/
│   └── test_api.py            # pytest suite for the API
├── results/                   # metrics JSON/CSV written by the notebooks
├── model/                     # fine-tuned weights (git-ignored)
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the results

1. **Baseline:** run `notebooks/01_baseline.ipynb` locally (CPU is fine). It writes `results/baseline_*.{json,csv}`.
2. **DistilBERT:** open `notebooks/02_distilbert.ipynb` in Google Colab, select a **T4 GPU** runtime, optionally upload `results/baseline_metrics.json` to `/content/results/`, and run all cells. The notebook:
   * fine-tunes the model and reports test accuracy and macro-F1,
   * prints the side-by-side comparison against the baseline,
   * saves the model to `./model` and, on Colab, downloads `intentroute_model.zip` and `intentroute_results.zip`.
3. Unzip both archives at the repo root so you have `./model/config.json` and friends.

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The model loads **once at startup** from `./model`. To use another directory, set `INTENTROUTE_MODEL_DIR=/path/to/model`. Interactive docs are served at http://localhost:8000/docs.

### Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | none | `{"status": "ok", "model_loaded": true}` |
| POST | `/predict` | `{"text": str}` | top-3 intents with softmax confidences and a `route` |

Routing rule: if the top-1 confidence is `< 0.5`, `route` is `"human_review"`. Otherwise it is `"auto"`.

### Example

```bash
curl -s http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true}
```

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I still have not received my new card"}'
```

The response has this shape. Intent names and confidences depend on your trained model, so the values below are placeholders:

```json
{
  "intents": [
    {"intent": "<intent_1>", "confidence": 0.0},
    {"intent": "<intent_2>", "confidence": 0.0},
    {"intent": "<intent_3>", "confidence": 0.0}
  ],
  "route": "auto"
}
```

`route` is `"human_review"` whenever `intents[0].confidence < 0.5`. Invalid input (a missing, empty or whitespace-only `text`) returns HTTP 422.

PowerShell equivalent:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType "application/json" -Body '{"text": "I still have not received my new card"}'
```

## Tests

```bash
pytest -q
```

The suite covers `/health`, a confident query (`route: auto`), the low-confidence fallback (`route: human_review`), the exact-threshold boundary, and input validation. These tests use an injected stub classifier, so they need no trained weights. One more test runs against the real model in `./model` and is skipped automatically when that model is absent.
