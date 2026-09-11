"""
draft_reply.py — Phase 5: Grounded reply generation via Grok.

Decision log:
- Model: grok-3 (mid/balanced tier, better quality than grok-3-mini for generation).
  Using a different tier from the classifier (grok-3-mini) to vary quality/cost.
- Trivial baseline: a single canned template per intent (no generation).
- Simple baseline: verbatim return of nearest neighbor's agent reply.
- Full system: Grok with retrieved grounding context + AmazonHelp tone instructions.
- System prompt instructs the model to ONLY reference policies/actions seen in
  retrieved precedents — prevents hallucinated refund amounts/timelines.
- Reply length capped at ~150 words to match AmazonHelp's actual Twitter style.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# ── xAI model config ───────────────────────────────────────────────────────────
DRAFTING_MODEL = "grok-3"           # mid-tier: quality matters, fewer calls
_OPENAI_BASE_URL = "https://api.x.ai/v1"

MAX_RETRIES = 3
RETRY_DELAY = 2.0


def _grok_client():
    from openai import OpenAI
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise EnvironmentError("XAI_API_KEY not set.")
    return OpenAI(api_key=api_key, base_url=_OPENAI_BASE_URL)


# ── Trivial baseline: canned templates ───────────────────────────────────────

_CANNED_TEMPLATES: dict[str, str] = {
    "order_status_tracking": (
        "Hi there! Please check your order status at amazon.com/orders or via the Amazon app. "
        "If you need more help, DM us your order number and we'll look into it for you."
    ),
    "delivery_delay_lost": (
        "We're sorry to hear about the delay! Please DM us your order number so we can "
        "investigate the shipment status and help resolve this for you."
    ),
    "damaged_wrong_item": (
        "We're sorry you received the wrong or damaged item. Please visit "
        "amazon.com/returns to start a replacement or refund, or DM us your order number "
        "and we'll be happy to help."
    ),
    "return_refund": (
        "To start a return, please visit amazon.com/returns. Most items can be returned within "
        "30 days of delivery. DM us your order number if you need further assistance."
    ),
    "billing_charge_dispute": (
        "We're sorry for the billing concern! Please DM us your account info (no passwords) "
        "and our team will review the charge and follow up with you as soon as possible."
    ),
    "account_login_security": (
        "We take account security very seriously. Please visit amazon.com/help/hub/login "
        "to reset your password or secure your account. DM us if you need additional assistance "
        "— do NOT share your password here."
    ),
    "cancel_order": (
        "To cancel an order, please visit amazon.com/orders and select 'Cancel Items'. "
        "Orders must be cancelled before they ship. DM us your order number if you need help."
    ),
    "product_question": (
        "Great question! For product-specific details, please check the listing page on amazon.com "
        "or contact the seller directly. We're happy to help if you have other questions!"
    ),
    "general_complaint_other": (
        "We're sorry to hear about your experience! We'd love to help make things right. "
        "Please DM us the details and our team will follow up with you promptly."
    ),
}

DEFAULT_CANNED = "We're sorry for the trouble! Please DM us your order details and our team will help you right away."


class CannedReplyDrafter:
    """Trivial baseline: returns a fixed template per intent."""
    name = "canned_template"

    def draft(
        self,
        customer_message: str,
        intent: str,
        retrieved_context: Optional[list[dict]] = None,
        **kwargs,
    ) -> dict:
        reply = _CANNED_TEMPLATES.get(intent, DEFAULT_CANNED)
        return {
            "reply": reply,
            "model": self.name,
            "grounded": False,
            "retrieved_count": 0,
        }


class VerbatimRetrievalDrafter:
    """
    Simple baseline: return the agent_reply from the nearest neighbor verbatim.
    Requires a retrieval index.
    """
    name = "verbatim_retrieval"

    def __init__(self, retrieval_index):
        self.index = retrieval_index

    def draft(
        self,
        customer_message: str,
        intent: str,
        retrieved_context: Optional[list[dict]] = None,
        **kwargs,
    ) -> dict:
        # Retrieve if not pre-retrieved
        context = retrieved_context or self.index.retrieve(customer_message, k=1, intent_filter=intent)
        if context:
            reply = context[0]["agent_reply"]
            retrieved_count = 1
        else:
            reply = DEFAULT_CANNED
            retrieved_count = 0

        return {
            "reply": reply,
            "model": self.name,
            "grounded": retrieved_count > 0,
            "retrieved_count": retrieved_count,
        }


# ── Full system: Grok reply drafter ──────────────────────────────────────────

_BRAND_VOICE_INSTRUCTIONS = """
## AmazonHelp Brand Voice Guidelines
(Derived from observed AmazonHelp Twitter replies in the dataset)

- Tone: Warm, professional, apologetic when warranted, never defensive.
- Length: 1-3 sentences for Twitter (max ~140 characters for the core message; 
  can be slightly longer in DM context but stay concise).
- Always acknowledge the customer's issue before offering a solution.
- Offer a clear next step (link, DM request, or specific action).
- Use "we" not "I" — this is a brand account.
- Never promise specific refund amounts, timelines, or outcomes that aren't 
  supported by the retrieved precedents below.
- Never share or ask for passwords, full credit card numbers.
- Sign off with warmth but no need for sign-off text (the account name is visible).
- For sensitive issues (billing/security), always route to DM — never handle 
  publicly.
"""

_DRAFTING_SYSTEM_PROMPT = f"""You are a customer support reply writer for AmazonHelp (Amazon's official Twitter support account).

{_BRAND_VOICE_INSTRUCTIONS}

## Your Task
Draft a single reply to the customer message provided.

## Grounding Rule (CRITICAL)
You will be given 1-3 real historical (customer message, agent reply) pairs as context.
You MUST base your reply on the patterns, policies, and actions seen in those examples.
Do NOT invent:
- Specific dollar amounts or refund percentages
- Specific timelines (e.g., "within 24 hours") unless explicitly in a retrieved example
- Policy details not seen in retrieved examples

If no retrieved examples are relevant, respond conservatively with an acknowledgment and 
a DM request for more details.

## Output
Return ONLY the reply text. No JSON, no metadata, no quotes around the reply.
"""


def _format_context(retrieved: list[dict]) -> str:
    if not retrieved:
        return "(No historical precedents retrieved for this query.)"
    lines = ["## Retrieved Historical Precedents\n"]
    for i, rec in enumerate(retrieved, 1):
        lines.append(f"### Example {i} (similarity: {rec.get('similarity_score', 0):.2f})")
        lines.append(f"**Customer:** {rec['customer_text'][:200]}")
        lines.append(f"**AmazonHelp replied:** {rec['agent_reply'][:300]}")
        lines.append("")
    return "\n".join(lines)


class GrokReplyDrafter:
    """Full system: Grok-based reply drafter grounded in retrieved precedents."""

    name = "grok_reply_drafter"

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = _grok_client()
        return self._client

    def draft(
        self,
        customer_message: str,
        intent: str,
        retrieved_context: Optional[list[dict]] = None,
        **kwargs,
    ) -> dict:
        context_block = _format_context(retrieved_context or [])
        retrieved_count = len(retrieved_context) if retrieved_context else 0

        user_prompt = f"""{context_block}

## Customer Message
Intent: {intent}
Message: {customer_message[:600]}

## Your Reply
"""

        for attempt in range(MAX_RETRIES):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=DRAFTING_MODEL,
                    messages=[
                        {"role": "system", "content": _DRAFTING_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=200,
                )
                reply = response.choices[0].message.content.strip()
                return {
                    "reply": reply,
                    "model": DRAFTING_MODEL,
                    "grounded": retrieved_count > 0,
                    "retrieved_count": retrieved_count,
                }
            except Exception as e:
                log.warning(f"Drafting API error attempt {attempt+1}: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))

        return {
            "reply": DEFAULT_CANNED,
            "model": DRAFTING_MODEL,
            "grounded": False,
            "retrieved_count": 0,
            "error": "All retries failed",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    drafter = GrokReplyDrafter()

    test_msg = "My package shows delivered but I never got it. This is ridiculous."
    mock_context = [
        {
            "customer_text": "Package shows delivered but I didn't receive it.",
            "agent_reply": "We're sorry to hear that! Please DM us your order number and we'll investigate the delivery right away.",
            "similarity_score": 0.92,
        }
    ]
    result = drafter.draft(
        customer_message=test_msg,
        intent="delivery_delay_lost",
        retrieved_context=mock_context,
    )
    print(f"Reply: {result['reply']}")
    print(f"Model: {result['model']}, Grounded: {result['grounded']}")
