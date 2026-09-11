"""
pipeline.py — End-to-end pipeline: message in → {intent, reply, escalate, reason}.

Wires together: classify → retrieve → escalate → draft_reply.

Supports all three system tiers:
  - "trivial"  : TrivialClassifier + CannedReply + AlwaysEscalate
  - "simple"   : TFIDFClassifier + VerbatimRetrieval + EscalationPolicy
  - "full"     : GrokClassifier + RetrievalIndex + EscalationPolicy + GrokReplyDrafter
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Add src/ to path so sub-modules import cleanly
sys.path.insert(0, str(Path(__file__).parent))

from taxonomy import INTENT_NAMES
from classify import TrivialClassifier, TFIDFClassifier, GrokClassifier
from retrieve import RetrievalIndex, load_index
from draft_reply import CannedReplyDrafter, VerbatimRetrievalDrafter, GrokReplyDrafter
from escalate import decide as escalation_decide, always_escalate, never_escalate


class AmazonHelpPipeline:
    """
    End-to-end pipeline for a single AmazonHelp support message.

    tier: "trivial" | "simple" | "full"
    """

    def __init__(self, tier: str = "full"):
        assert tier in ("trivial", "simple", "full"), f"Unknown tier: {tier}"
        self.tier = tier
        self._retrieval_index: Optional[RetrievalIndex] = None
        self._classifier = None
        self._drafter = None

    # ── Lazy initialization ───────────────────────────────────────────────────

    def _init_trivial(self):
        self._classifier = TrivialClassifier()
        self._drafter = CannedReplyDrafter()
        self._escalate_fn = always_escalate  # trivial escalation = always escalate
        log.info("[Pipeline] Trivial tier initialized.")

    def _init_simple(self, labeled_texts=None, labeled_labels=None):
        clf = TFIDFClassifier()
        if labeled_texts and labeled_labels:
            clf.fit(labeled_texts, labeled_labels)
        else:
            log.warning("[Pipeline] TFIDFClassifier not fitted — call pipeline.fit() first.")
        self._classifier = clf
        self._retrieval_index = self._load_index()
        self._drafter = VerbatimRetrievalDrafter(self._retrieval_index)
        self._escalate_fn = escalation_decide
        log.info("[Pipeline] Simple tier initialized.")

    def _init_full(self):
        self._classifier = GrokClassifier()
        self._retrieval_index = self._load_index()
        self._drafter = GrokReplyDrafter()
        self._escalate_fn = escalation_decide
        log.info("[Pipeline] Full (Grok) tier initialized.")

    def _load_index(self) -> RetrievalIndex:
        try:
            idx = load_index()
            return idx
        except Exception as e:
            log.warning(f"Could not load retrieval index: {e}. Retrieval will be skipped.")
            return RetrievalIndex()

    def setup(self, labeled_texts=None, labeled_labels=None):
        """Initialize all components. Call before running predict()."""
        if self.tier == "trivial":
            self._init_trivial()
        elif self.tier == "simple":
            self._init_simple(labeled_texts, labeled_labels)
        elif self.tier == "full":
            self._init_full()
        return self

    def fit(self, texts: list[str], labels: list[str]):
        """Fit the classifier (only meaningful for 'simple' tier)."""
        if self._classifier:
            self._classifier.fit(texts, labels)
        return self

    # ── Core predict ─────────────────────────────────────────────────────────

    def predict(self, customer_message: str) -> dict:
        """
        Run the full pipeline on a single customer message.

        Returns:
            {
              "intent": str,
              "confidence": float,
              "escalate": bool,
              "reason": str,
              "reason_detail": str,
              "reply": str,
              "grounded": bool,
              "retrieved_count": int,
              "classifier_model": str,
              "drafter_model": str,
              "tier": str,
            }
        """
        if self._classifier is None:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")

        # 1. Classify intent
        clf_result = self._classifier.predict(customer_message)
        intent = clf_result["intent"]
        confidence = clf_result["confidence"]

        # 2. Retrieve historical context
        retrieved_context = []
        if self._retrieval_index and self._retrieval_index.is_built():
            retrieved_context = self._retrieval_index.retrieve(
                customer_message, k=3, intent_filter=intent
            )

        # 3. Escalation decision
        if self.tier == "trivial":
            esc_result = always_escalate(customer_message)
        else:
            esc_result = escalation_decide(
                customer_text=customer_message,
                intent=intent,
                confidence=confidence,
                retrieved_context=retrieved_context if retrieved_context else None,
            )

        # 4. Draft reply (even if escalating, draft can be used as agent starting point)
        draft_result = self._drafter.draft(
            customer_message=customer_message,
            intent=intent,
            retrieved_context=retrieved_context,
        )

        return {
            "intent": intent,
            "confidence": confidence,
            "escalate": esc_result["escalate"],
            "reason": esc_result["reason"],
            "reason_detail": esc_result["reason_detail"],
            "reply": draft_result["reply"],
            "grounded": draft_result.get("grounded", False),
            "retrieved_count": draft_result.get("retrieved_count", 0),
            "classifier_model": clf_result.get("model", "unknown"),
            "drafter_model": draft_result.get("model", "unknown"),
            "tier": self.tier,
        }

    def predict_batch(self, messages: list[str]) -> list[dict]:
        return [self.predict(m) for m in messages]


# ── Quick CLI test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_messages = [
        "My order hasn't arrived yet and it's been 2 weeks. Where is it?!",
        "I received a damaged item, the box was completely crushed.",
        "Someone hacked my account and changed my email!",
        "I want to return this product, how do I get a refund?",
        "Can I speak to a manager please?",
    ]

    print("=== Testing trivial tier ===")
    trivial = AmazonHelpPipeline(tier="trivial").setup()
    for msg in test_messages[:2]:
        r = trivial.predict(msg)
        print(f"  ESCALATE={r['escalate']} | intent={r['intent']} | {msg[:50]}...")
        print(f"    reply: {r['reply'][:80]}...")

    print("\n=== Testing full tier (requires XAI_API_KEY) ===")
    try:
        full = AmazonHelpPipeline(tier="full").setup()
        r = full.predict(test_messages[0])
        print(f"  ESCALATE={r['escalate']} [{r['reason']}]")
        print(f"  intent={r['intent']} (conf={r['confidence']:.2f})")
        print(f"  reply: {r['reply'][:120]}...")
    except Exception as e:
        print(f"  Full pipeline error (expected if no API key): {e}")
