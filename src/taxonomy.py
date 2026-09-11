"""
taxonomy.py — Phase 2: Intent taxonomy definitions + labeling helpers.

The taxonomy was built by open-coding ~400 randomly sampled AmazonHelp
customer messages from the reconstructed thread dataset.

STOP checkpoint: This taxonomy must be confirmed with the user before
any classifier is built on top of it.

Decision log:
- 9 intents chosen (not 10+) because empirically the data split cleanly;
  a separate "general_complaint" catch-all captures residual noise without
  polluting clear intents.
- "Prime membership" collapsed into "account_issue" because the dataset
  shows very few dedicated Prime queries and they share resolution patterns.
- "product_question" kept even though agents often can't answer it, because
  it is a real escalation trigger ("I'll connect you with the product team").
"""

from __future__ import annotations

INTENTS: dict[str, dict] = {
    "order_status_tracking": {
        "definition": (
            "Customer asks where their order is, when it will arrive, "
            "or requests a tracking number/update."
        ),
        "examples": [
            "Where is my order? It's been 5 days.",
            "Can you give me the tracking number for order 123-456?",
            "My package says 'out for delivery' but nothing arrived.",
        ],
        "escalate_by_default": False,
    },
    "delivery_delay_lost": {
        "definition": (
            "Package is significantly late, presumed lost, or tracking shows "
            "no movement for an extended period."
        ),
        "examples": [
            "My order was supposed to arrive Monday, it's Friday and still nothing.",
            "Tracking hasn't updated in 10 days — is it lost?",
            "Package shows delivered but I never received it.",
        ],
        "escalate_by_default": False,
    },
    "damaged_wrong_item": {
        "definition": (
            "Customer received a broken, defective, or entirely wrong item."
        ),
        "examples": [
            "I got the wrong color. This is not what I ordered.",
            "The item arrived smashed, the box was destroyed.",
            "Received a completely different product.",
        ],
        "escalate_by_default": False,
    },
    "return_refund": {
        "definition": (
            "Customer wants to return an item and/or receive a refund, "
            "replacement, or exchange."
        ),
        "examples": [
            "I want to return this and get my money back.",
            "How do I start a return? I have 30 days right?",
            "Send me a prepaid label for the return.",
        ],
        "escalate_by_default": False,
    },
    "billing_charge_dispute": {
        "definition": (
            "Customer disputes a charge, was billed incorrectly, sees an "
            "unexpected charge, or reports a duplicate payment."
        ),
        "examples": [
            "I was charged twice for the same order.",
            "There's a $79 charge on my card I didn't authorize.",
            "Why was I charged more than the listed price?",
        ],
        "escalate_by_default": True,   # payment fraud risk
    },
    "account_login_security": {
        "definition": (
            "Customer can't log in, account was compromised/hacked, needs "
            "password reset, or has a Prime membership issue."
        ),
        "examples": [
            "Someone accessed my account without my permission.",
            "I can't reset my password, the email never arrives.",
            "My Prime membership keeps getting charged even after I cancelled.",
        ],
        "escalate_by_default": True,   # security category
    },
    "cancel_order": {
        "definition": (
            "Customer wants to cancel an order before it ships, or stop "
            "a subscription/auto-renewal."
        ),
        "examples": [
            "Please cancel my order immediately.",
            "I need to cancel before it ships out.",
            "How do I cancel my Subscribe & Save?",
        ],
        "escalate_by_default": False,
    },
    "product_question": {
        "definition": (
            "Customer asks about product features, compatibility, availability, "
            "or seller/listing details."
        ),
        "examples": [
            "Is this compatible with my 2019 MacBook Pro?",
            "Does this come in size XL?",
            "Is this sold by Amazon or a third-party seller?",
        ],
        "escalate_by_default": False,
    },
    "general_complaint_other": {
        "definition": (
            "Catch-all: high-frustration expression without a specific actionable "
            "ask, or a topic that doesn't fit the above categories."
        ),
        "examples": [
            "Amazon's customer service is absolutely terrible.",
            "This is the third time this has happened!",
            "Just wanted to say your app is broken.",
        ],
        "escalate_by_default": False,
    },
}

INTENT_NAMES = list(INTENTS.keys())
N_INTENTS = len(INTENT_NAMES)


def get_taxonomy_prompt_block() -> str:
    """Return a formatted string describing all intents for use in prompts."""
    lines = ["## AmazonHelp Intent Taxonomy\n"]
    for name, meta in INTENTS.items():
        lines.append(f"### {name}")
        lines.append(f"**Definition:** {meta['definition']}")
        lines.append("**Examples:**")
        for ex in meta["examples"]:
            lines.append(f"  - \"{ex}\"")
        lines.append("")
    return "\n".join(lines)


def get_few_shot_examples() -> list[dict]:
    """Return one example per intent for few-shot prompting."""
    examples = []
    for intent, meta in INTENTS.items():
        examples.append({
            "text": meta["examples"][0],
            "intent": intent,
        })
    return examples


if __name__ == "__main__":
    print(f"Taxonomy has {N_INTENTS} intents:\n")
    for name, meta in INTENTS.items():
        escalate = " [ESCALATE BY DEFAULT]" if meta["escalate_by_default"] else ""
        print(f"  {name}{escalate}")
        print(f"    → {meta['definition'][:80]}…")
    print(f"\n{get_taxonomy_prompt_block()}")
