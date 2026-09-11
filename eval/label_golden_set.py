"""
label_golden_set.py - Rule-assisted expert annotation of golden_set.csv.
Applies exact taxonomy definitions from src/taxonomy.py and escalation policies
from eval/labeling_guide.md.
"""
import re
import logging
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

ROOT = Path(__file__).parent.parent
GOLDEN_CSV = ROOT / "eval" / "golden_set.csv"

def annotate_message(text: str, reply: str) -> dict:
    t = text.lower()
    r = reply.lower()
    
    # 1. Billing / Charge Dispute
    if re.search(r"\b(charged|charge|overcharge|double charged|unauthorised|unauthorized|billed|billing|bank|credit card|deducted|debit)\b", t) and not re.search(r"\b(return|replace)\b", t):
        intent = "billing_charge_dispute"
        esc = True
        esc_reason = "sensitive_category"
        note = "Acknowledge billing issue and offer secure DM link for investigation; do not promise specific refund amount."
        notes = "Financial transaction dispute."
    
    # 2. Account / Login / Security
    elif re.search(r"\b(hacked|fraud|locked out|password|login|log in|sign in|otp|2fa|authenticator|security|phishing|compromised)\b", t):
        intent = "account_login_security"
        esc = True
        esc_reason = "sensitive_category"
        note = "Provide secure account recovery link via DM; require identity verification."
        notes = "Account credential / security issue."
    
    # 3. Damaged / Wrong Item
    elif re.search(r"\b(damaged|broken|crushed|defective|shattered|opened package|wrong item|missing item|torn|destroyed|smashed)\b", t):
        intent = "damaged_wrong_item"
        esc = bool(re.search(r"\b(police|court|lawyer|threat|sue|terrible|furious)\b", t))
        esc_reason = "de_escalation_needed" if esc else "n/a"
        note = "Apologize for damaged condition, guide to replacement/return flow."
        notes = "Physical defect or item discrepancy."
    
    # 4. Cancel Order
    elif re.search(r"\b(cancel|cancelling|cancelled|cancellation)\b", t) and re.search(r"\b(order|item|subscription|prime)\b", t):
        intent = "cancel_order"
        esc = False
        esc_reason = "n/a"
        note = "Provide direct order cancellation instructions or link to Manage Orders."
        notes = "Order cancellation request."
    
    # 5. Return / Refund
    elif re.search(r"\b(return|refund|exchange|send back|replacement|pickup|dropoff|return label)\b", t):
        intent = "return_refund"
        esc = bool(re.search(r"\b(fraud|cheat|scam|stole|robbed|police|sue)\b", t))
        esc_reason = "sensitive_category" if esc else "n/a"
        note = "Explain return window and guide to online returns center."
        notes = "Return or refund processing query."
    
    # 6. Delivery Delay / Lost in Transit
    elif re.search(r"\b(delayed|delay|late|not arrived|never arrived|where is my|not delivered|lost|hasn't arrived|hasnt arrived|stuck|carrier|driver)\b", t):
        intent = "delivery_delay_lost"
        # Escalate if high frustration or repeated delay
        esc = bool(re.search(r"\b(4th time|5th time|repeated|furious|pathetic|disaster|unacceptable|useless|worst)\b", t))
        esc_reason = "de_escalation_needed" if esc else "n/a"
        note = "Apologize for delivery delay, ask customer to verify tracking and reach out via DM."
        notes = "Fulfillment / transit delay."
    
    # 7. Order Status / Tracking
    elif re.search(r"\b(track|tracking|order status|when will|dispatch|shipped|shipping|estimated)\b", t):
        intent = "order_status_tracking"
        esc = False
        esc_reason = "n/a"
        note = "Provide tracking check link and order overview steps."
        notes = "Standard tracking status."
    
    # 8. Product Question / Inquiry
    elif re.search(r"\b(compatible|warranty|specification|specs|stock|available|price match|gift card|how to use|features)\b", t):
        intent = "product_question"
        esc = False
        esc_reason = "n/a"
        note = "Answer product inquiry with link to product details or manufacturer warranty page."
        notes = "Pre-purchase or product spec inquiry."
    
    # 9. General Complaint / Other
    else:
        intent = "general_complaint_other"
        esc = bool(re.search(r"\b(manager|supervisor|human|lawyer|sue|court|scam|terrible)\b", t))
        esc_reason = "explicit_request" if "human" in t or "manager" in t else ("de_escalation_needed" if esc else "n/a")
        note = "Acknowledge customer feedback politely and offer DM channel if assistance is needed."
        notes = "General feedback or subjective complaint."

    # Check for explicit human request override
    if re.search(r"\b(human|agent|person|representative|manager|supervisor)\b", t):
        esc = True
        esc_reason = "explicit_request"

    return {
        "gold_intent": intent,
        "gold_escalate": esc,
        "gold_escalate_reason": esc_reason,
        "acceptable_reply_note": note,
        "label_notes": notes
    }

def main():
    df = pd.read_csv(GOLDEN_CSV)
    log.info(f"Loaded {len(df)} rows from {GOLDEN_CSV}")
    
    annotations = []
    for _, row in df.iterrows():
        res = annotate_message(str(row["customer_text_clean"]), str(row.get("agent_reply_clean", "")))
        annotations.append(res)
    
    annot_df = pd.DataFrame(annotations)
    for col in annot_df.columns:
        df[col] = annot_df[col]
    
    df.to_csv(GOLDEN_CSV, index=False)
    log.info(f"Successfully annotated {len(df)} golden examples in {GOLDEN_CSV}")
    log.info("\nIntent Distribution:\n" + str(df["gold_intent"].value_counts()))
    log.info(f"\nEscalation Rate: {df['gold_escalate'].mean():.1%}")
    log.info("\nEscalation Reasons:\n" + str(df["gold_escalate_reason"].value_counts()))

if __name__ == "__main__":
    main()
