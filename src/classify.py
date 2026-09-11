"""
classify.py — Phase 3 + 5: Intent classifiers.

Implements three tiers:
1. TrivialClassifier  — always predicts majority class
2. TFIDFClassifier    — TF-IDF + logistic regression (simple baseline)
3. GrokClassifier     — Grok LLM with frozen taxonomy + few-shot (full system)

All expose the same interface:
    predict(text: str) -> dict[str, str | float]
    predict_batch(texts: list[str]) -> list[dict]

Decision log:
- Used grok-3-mini for classification (cheapest fast tier, confirmed from
  https://docs.x.ai/developers/models at time of writing).
- JSON structured output enforced via response_format to avoid parse errors.
- Confidence for Grok is self-reported in the JSON; for TFIDF it's max softmax prob.
- Majority-class baseline is computed from training data at fit() time.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import sys
from pathlib import Path

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from taxonomy import INTENT_NAMES, get_taxonomy_prompt_block, get_few_shot_examples

load_dotenv()
log = logging.getLogger(__name__)

# ── xAI model config (confirmed from docs.x.ai/developers/models) ─────────────
CLASSIFIER_MODEL = "grok-3-mini"     # cheapest / highest throughput tier
_OPENAI_BASE_URL = "https://api.x.ai/v1"

# Max retries for API calls
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


def _grok_client():
    from openai import OpenAI
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise EnvironmentError("XAI_API_KEY not set. Copy .env.example -> .env and fill in your key.")
    return OpenAI(api_key=api_key, base_url=_OPENAI_BASE_URL)


# ── 1. Trivial baseline ────────────────────────────────────────────────────────

class TrivialClassifier:
    """Always predicts the majority class seen during training."""

    name = "trivial_majority"

    def __init__(self):
        self.majority_class: str | None = None

    def fit(self, texts: list[str], labels: list[str]) -> "TrivialClassifier":
        from collections import Counter
        counts = Counter(labels)
        self.majority_class = counts.most_common(1)[0][0]
        log.info(f"[Trivial] Majority class = {self.majority_class}")
        return self

    def predict(self, text: str) -> dict:
        return {
            "intent": self.majority_class,
            "confidence": 1.0,
            "model": self.name,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]


# ── 2. TF-IDF + Logistic Regression baseline ──────────────────────────────────

class TFIDFClassifier:
    """TF-IDF features + logistic regression."""

    name = "tfidf_logreg"

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=30_000,
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=5.0,
                solver="lbfgs",
                multi_class="multinomial",
            )),
        ])
        self.label_encoder = LabelEncoder()
        self._fitted = False

    def fit(self, texts: list[str], labels: list[str]) -> "TFIDFClassifier":
        encoded = self.label_encoder.fit_transform(labels)
        self.pipeline.fit(texts, encoded)
        self._fitted = True
        log.info(f"[TFIDF] Trained on {len(texts):,} examples, "
                 f"{len(self.label_encoder.classes_)} classes.")
        return self

    def predict(self, text: str) -> dict:
        if not self._fitted:
            raise RuntimeError("TFIDFClassifier must be fit() before predict().")
        proba = self.pipeline.predict_proba([text])[0]
        pred_idx = int(np.argmax(proba))
        intent = self.label_encoder.inverse_transform([pred_idx])[0]
        return {
            "intent": intent,
            "confidence": float(proba[pred_idx]),
            "model": self.name,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        if not self._fitted:
            raise RuntimeError("TFIDFClassifier must be fit() before predict().")
        proba_matrix = self.pipeline.predict_proba(texts)
        results = []
        for proba in proba_matrix:
            pred_idx = int(np.argmax(proba))
            intent = self.label_encoder.inverse_transform([pred_idx])[0]
            results.append({
                "intent": intent,
                "confidence": float(proba[pred_idx]),
                "model": self.name,
            })
        return results


# ── 3. Grok-based classifier ───────────────────────────────────────────────────

_TAXONOMY_BLOCK = get_taxonomy_prompt_block()
_FEW_SHOT = get_few_shot_examples()

_SYSTEM_PROMPT = f"""You are an intent classifier for AmazonHelp customer support tweets.

{_TAXONOMY_BLOCK}

## Task
Given a customer message, output a JSON object with exactly two fields:
- "intent": one of the intent names listed above (exact snake_case string)
- "confidence": a float from 0.0 to 1.0 indicating your certainty

## Rules
- Choose the SINGLE most relevant intent.
- If the message fits no specific intent cleanly, use "general_complaint_other".
- Strip @mentions and URLs from your reasoning; they are routing artifacts.
- Output ONLY valid JSON. No markdown, no explanation.

## Few-shot examples
""" + "\n".join(
    f'Input: "{ex["text"]}"\nOutput: {{"intent": "{ex["intent"]}", "confidence": 0.95}}'
    for ex in _FEW_SHOT
)


def _call_grok_classify(text: str, client) -> dict:
    """Single Grok classify call with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=CLASSIFIER_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f'Input: "{text[:500]}"'},
                ],
                temperature=0.0,
                max_tokens=60,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            # Validate
            intent = parsed.get("intent", "general_complaint_other")
            if intent not in INTENT_NAMES:
                intent = "general_complaint_other"
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return {
                "intent": intent,
                "confidence": confidence,
                "model": CLASSIFIER_MODEL,
                "raw_response": raw,
            }
        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error on attempt {attempt+1}: {e}")
        except Exception as e:
            log.warning(f"API error on attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))

    # Fallback
    return {
        "intent": "general_complaint_other",
        "confidence": 0.0,
        "model": CLASSIFIER_MODEL,
        "raw_response": "ERROR",
    }


class GrokClassifier:
    """Grok LLM-based intent classifier."""

    name = "grok_classifier"

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = _grok_client()
        return self._client

    def fit(self, texts: list[str], labels: list[str]) -> "GrokClassifier":
        # LLM classifier requires no training — taxonomy is in the prompt.
        log.info("[Grok] No training required (zero/few-shot via taxonomy prompt).")
        return self

    def predict(self, text: str) -> dict:
        return _call_grok_classify(text, self._get_client())

    def predict_batch(self, texts: list[str], delay: float = 0.1) -> list[dict]:
        """Batch predict with optional throttle to avoid rate limits."""
        results = []
        client = self._get_client()
        for i, text in enumerate(texts):
            result = _call_grok_classify(text, client)
            results.append(result)
            if delay > 0 and i < len(texts) - 1:
                time.sleep(delay)
        return results


# ── Utility ────────────────────────────────────────────────────────────────────

def load_labeled_data(golden_csv: str | Path) -> tuple[list[str], list[str]]:
    """Load (text, intent) pairs from the golden set CSV."""
    df = pd.read_csv(golden_csv)
    texts = df["customer_text_clean"].tolist()
    labels = df["gold_intent"].tolist()
    return texts, labels


if __name__ == "__main__":
    # Quick smoke-test — requires XAI_API_KEY
    import sys
    test_text = "My package was supposed to arrive yesterday and I still haven't received it."
    print(f"Test text: {test_text}\n")

    gc = GrokClassifier()
    result = gc.predict(test_text)
    print(f"Grok -> {result}")
