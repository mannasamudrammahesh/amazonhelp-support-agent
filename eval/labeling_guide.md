# Labeling Guide — AmazonHelp Golden Evaluation Set

## Overview

This guide documents exactly how the golden evaluation set (`golden_set.csv`) was
sampled, what each label means, and how to apply them consistently.

The golden set contains **150-250 examples** used to evaluate all three pipeline tiers.
It is **held out** from:
- The taxonomy open-coding sample (Phase 2)
- The retrieval index (Phase 4)

## Sampling Methodology

1. **Source**: `data/amazonhelp_threads.parquet` — reconstructed AmazonHelp threads.
2. **Exclusion**: Any thread used in Phase 2 open-coding is excluded.
3. **Stratification**:
   - **By intent**: Each of the 9 intents gets roughly proportional representation,
     but rare intents (account_login_security, cancel_order) are oversampled to
     ensure ≥10 examples each.
   - **By escalation**: At least 30% of examples should be escalation cases
     (to avoid trivial "never escalate" accuracy inflation).
   - **By time period**: Split roughly evenly across dataset quarters (the dataset
     spans ~2015-2017; support norms differ across this period).
4. **Deduplication**: Near-identical templated tweets are deduplicated using
   TF-IDF cosine similarity (threshold 0.90). Only one representative is kept.

## Label Definitions

### `gold_intent` (required)
One of the 9 intent names from `src/taxonomy.py`:
- `order_status_tracking`
- `delivery_delay_lost`
- `damaged_wrong_item`
- `return_refund`
- `billing_charge_dispute`
- `account_login_security`
- `cancel_order`
- `product_question`
- `general_complaint_other`

**How to assign**: Read the customer message only (not the agent reply). Ask:
"What is the primary thing this customer wants or is expressing?" Assign the
single most relevant intent. If genuinely ambiguous, use `general_complaint_other`
and note it in `label_notes`.

### `gold_escalate` (required)
Boolean (`True` / `False`):
- `True` if a human agent should review this before a response goes out.
- `False` if an automated response is appropriate.

**Escalation criteria**:
| Trigger | Escalate? |
|---------|-----------|
| Security / fraud / unauthorized account access | True |
| Explicit "I want a human/manager" | True |
| Billing dispute / unexpected charge | True |
| Account or login issue | True |
| Extreme frustration (threats, legal language) | True |
| Standard tracking/order/return/refund | False |
| Product question | False |
| Unclear, borderline | Your judgment; note in `label_notes` |

### `gold_escalate_reason` (required when gold_escalate=True)
One of:
- `sensitive_category` — billing, security, fraud, legal
- `explicit_request` — customer explicitly asked for human
- `de_escalation_needed` — extreme frustration/threats
- `low_confidence` — message is too ambiguous to auto-handle safely
- `n/a` — (only when gold_escalate=False)

### `acceptable_reply_note` (required)
A brief note (1-3 sentences) describing what an **acceptable** auto-reply should
accomplish for this message. This is NOT a reference reply — it describes the
criteria. Examples:
- "Should acknowledge the delay, ask for order details via DM, not promise specific timeline."
- "Should NOT make any promises — this is a security issue requiring human review."
- "Should point to amazon.com/returns and offer DM assistance if needed."

### `label_notes` (optional)
Any edge cases, ambiguities, or reasons you deviated from the standard rules.

## Self-Consistency Check

After labeling all examples, wait ≥24 hours, then re-label a random 15-20% sample
**blind** to your first labels. Compare the two passes and report:
- Self-consistency rate for `gold_intent` (exact match %)
- Self-consistency rate for `gold_escalate` (exact match %)

This is reported in the REPORT.md "What's Misleading" section as a substitute
(weaker) measure for inter-rater agreement.

## CSV Schema

```
thread_id, customer_tweet_id, customer_text_clean, customer_text_raw,
agent_reply_clean, created_at, gold_intent, gold_escalate,
gold_escalate_reason, acceptable_reply_note, label_notes,
split (taxonomy/retrieval/golden)
```

## Non-English Handling

Non-English tweets are retained in the dataset but labeled with `gold_intent = general_complaint_other`
and `label_notes = "non-english: [detected language]"` unless the content is
clearly parseable. This is noted in the report as a limitation.
