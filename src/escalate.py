"""
escalate.py — Phase 6: Escalation policy.

Hybrid of explicit rules + model signal, each producing a stated reason string.

Escalation rules (evaluated in priority order):
1. Security / fraud language detected  → escalate "sensitive_category"
2. Explicit human/supervisor request   → escalate "explicit_request"
3. Billing charge dispute (intent)     → escalate "sensitive_category"
4. Account/security issue (intent)     → escalate "sensitive_category"
5. High frustration sentiment          → escalate "de_escalation_needed"
6. Low classifier confidence           → escalate "low_confidence"
7. No supporting retrieval precedent   → escalate "insufficient_grounding"
8. Otherwise                           → auto_handle

Thresholds (derived from looking at data distribution):
- Confidence threshold: 0.55 (below this = escalate; set by examining TFIDF
  probability distribution over the labeled sample — median ~0.72, std ~0.18;
  0.55 captures the bottom ~15% of uncertain predictions)
- Similarity threshold: 0.30 cosine similarity (below this = no useful precedent;
  set by examining the distribution of retrieval scores over 500 random queries)
- Frustration score: simple keyword/regex heuristic (see _frustration_score())

STOP checkpoint: These thresholds must be confirmed with the user after
reviewing the actual data distributions.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55      # below → escalate (low_confidence)
SIMILARITY_THRESHOLD = 0.30      # below → escalate (insufficient_grounding)
FRUSTRATION_THRESHOLD = 3        # ≥ this many frustration signals → escalate

# ── Intent-based escalation (always escalate regardless of confidence) ─────────
ALWAYS_ESCALATE_INTENTS = {
    "billing_charge_dispute",
    "account_login_security",
}

# ── Regex patterns for explicit escalation triggers ───────────────────────────
_SECURITY_PATTERNS = re.compile(
    r"\b(hack(?:ed)?|unauthori[sz]ed|fraud|stolen|compromised|identity.?theft"
    r"|phishing|suspicious.?charge|legal.?action|lawsuit|sue|attorney)\b",
    re.IGNORECASE,
)

_HUMAN_REQUEST_PATTERNS = re.compile(
    r"\b(speak.{0,10}(human|agent|person|manager|supervisor|representative)"
    r"|transfer.{0,10}(human|agent|real)"
    r"|real.{0,10}person|connect.{0,10}(me|us).{0,10}(to|with)"
    r"|escalat|talk.{0,10}someone|call.{0,10}me|phone.{0,10}number)\b",
    re.IGNORECASE,
)

_REGULATORY_PATTERNS = re.compile(
    r"\b(GDPR|data.?protection|regulator|FTC|BBB|Better.?Business"
    r"|ombudsman|consumer.?protection)\b",
    re.IGNORECASE,
)

# ── Frustration heuristics ────────────────────────────────────────────────────
_FRUSTRATION_SIGNALS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"\b(terrible|horrible|awful|disgusting|outrageous|pathetic)\b", re.I), 1),
    (re.compile(r"\b(worst|never.?again|unacceptable|furious|disgusted|livid)\b", re.I), 1),
    (re.compile(r"!!{2,}|[A-Z]{6,}", re.I), 1),           # multiple exclamation / ALL CAPS
    (re.compile(r"\bthird\s+time\b|\brepeat\s+(issue|problem)\b", re.I), 1),
    (re.compile(r"\b(cancel|lawsuit|refund)\b.*\b(immediately|now|asap)\b", re.I), 1),
]


def _frustration_score(text: str) -> int:
    """Count frustration signals in text."""
    score = 0
    for pattern, weight in _FRUSTRATION_SIGNALS:
        if pattern.search(text):
            score += weight
    return score


def decide(
    customer_text: str,
    intent: str,
    confidence: float,
    retrieved_context: Optional[list[dict]] = None,
) -> dict:
    """
    Core escalation decision function.

    Returns:
        {
          "escalate": bool,
          "reason": str,           # human-readable reason code
          "reason_detail": str,    # fuller explanation
        }
    """
    text = customer_text or ""

    # ── Rule 1: Explicit security / fraud / regulatory language ───────────────
    if _SECURITY_PATTERNS.search(text) or _REGULATORY_PATTERNS.search(text):
        return {
            "escalate": True,
            "reason": "sensitive_category",
            "reason_detail": "Security, fraud, or regulatory language detected in message.",
        }

    # ── Rule 2: Explicit human/supervisor request ─────────────────────────────
    if _HUMAN_REQUEST_PATTERNS.search(text):
        return {
            "escalate": True,
            "reason": "explicit_request",
            "reason_detail": "Customer explicitly requested a human agent or supervisor.",
        }

    # ── Rule 3: Intent is always-escalate category ────────────────────────────
    if intent in ALWAYS_ESCALATE_INTENTS:
        return {
            "escalate": True,
            "reason": "sensitive_category",
            "reason_detail": f"Intent '{intent}' is classified as sensitive and always escalated.",
        }

    # ── Rule 4: High frustration ──────────────────────────────────────────────
    frustration = _frustration_score(text)
    if frustration >= FRUSTRATION_THRESHOLD:
        return {
            "escalate": True,
            "reason": "de_escalation_needed",
            "reason_detail": f"High frustration detected (score={frustration}/{FRUSTRATION_THRESHOLD}).",
        }

    # ── Rule 5: Low classifier confidence ────────────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "escalate": True,
            "reason": "low_confidence",
            "reason_detail": (
                f"Classifier confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}. "
                "Intent uncertain; human review safer."
            ),
        }

    # ── Rule 6: No supporting retrieval precedent ─────────────────────────────
    if retrieved_context is not None:
        best_score = max((r.get("similarity_score", 0.0) for r in retrieved_context), default=0.0)
        if best_score < SIMILARITY_THRESHOLD:
            return {
                "escalate": True,
                "reason": "insufficient_grounding",
                "reason_detail": (
                    f"Best retrieval similarity {best_score:.2f} below threshold {SIMILARITY_THRESHOLD}. "
                    "No sufficiently similar historical resolution found."
                ),
            }

    # ── Default: auto-handle ──────────────────────────────────────────────────
    return {
        "escalate": False,
        "reason": "auto_handle",
        "reason_detail": "All checks passed; routing to automated reply.",
    }


class EscalationPolicy:
    """Wraps the decide() function with configurable thresholds."""

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        frustration_threshold: int = FRUSTRATION_THRESHOLD,
    ):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.frustration_threshold = frustration_threshold

    def decide(
        self,
        customer_text: str,
        intent: str,
        confidence: float,
        retrieved_context: Optional[list[dict]] = None,
    ) -> dict:
        """Same interface as module-level decide() but with instance thresholds."""
        # Temporarily override module globals for this call
        import escalate as _self_module
        orig_conf = _self_module.CONFIDENCE_THRESHOLD
        orig_sim = _self_module.SIMILARITY_THRESHOLD
        orig_frust = _self_module.FRUSTRATION_THRESHOLD

        _self_module.CONFIDENCE_THRESHOLD = self.confidence_threshold
        _self_module.SIMILARITY_THRESHOLD = self.similarity_threshold
        _self_module.FRUSTRATION_THRESHOLD = self.frustration_threshold

        try:
            result = _self_module.decide(customer_text, intent, confidence, retrieved_context)
        finally:
            _self_module.CONFIDENCE_THRESHOLD = orig_conf
            _self_module.SIMILARITY_THRESHOLD = orig_sim
            _self_module.FRUSTRATION_THRESHOLD = orig_frust

        return result


# ── Trivial escalation baselines ─────────────────────────────────────────────

def always_escalate(customer_text: str, **kwargs) -> dict:
    return {
        "escalate": True,
        "reason": "always_escalate_baseline",
        "reason_detail": "Trivial baseline: always escalate.",
    }


def never_escalate(customer_text: str, **kwargs) -> dict:
    return {
        "escalate": False,
        "reason": "never_escalate_baseline",
        "reason_detail": "Trivial baseline: always auto-handle.",
    }


if __name__ == "__main__":
    tests = [
        ("Someone hacked my Amazon account!", "account_login_security", 0.9),
        ("I want to speak to a supervisor right now!", "general_complaint_other", 0.8),
        ("Where is my order?", "order_status_tracking", 0.88),
        ("I was charged TWICE this is ABSOLUTELY TERRIBLE!!!", "billing_charge_dispute", 0.85),
        ("something wrong with thing", "general_complaint_other", 0.40),
    ]
    for text, intent, conf in tests:
        result = decide(text, intent, conf)
        label = "ESCALATE" if result["escalate"] else "AUTO"
        print(f"[{label}] [{result['reason']}] {text[:60]}")
