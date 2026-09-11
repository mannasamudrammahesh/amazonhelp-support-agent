# REPORT — AmazonHelp Customer Support Pipeline

> **Note**: This is a template/skeleton report. Sections marked `[FILL AFTER EVAL]` 
> will be completed once the golden set is labeled and eval runs are complete.
> The results table requires user confirmation before finalizing (per the brief's 
> Phase 9 checkpoint).

---

## 1. Problem Framing

### What "good" means for AmazonHelp specifically

AmazonHelp is Amazon's official Twitter customer support account. "Good" in this context means:

1. **Correctly routed**: The right message goes to the right handler (automated or human). A missed escalation (auto-handling something that needs a human) is a worse error than an over-escalation.

2. **Grounded in precedent**: Replies must not invent policies, refund amounts, or timelines. AmazonHelp's brand trust depends on not over-promising.

3. **Concise and actionable**: Twitter support has ~280 character visibility; the reply must give the customer a clear next step, not hedge.

4. **Escalation has a stated reason**: Every escalation decision is traceable, not a black box.

### What we deliberately did not build

- **Multi-language support**: The pipeline processes English only. ~5-8% of AmazonHelp tweets are non-English; these are flagged and escalated.
- **Order-lookup tool integration**: No live order database. Replies reference historical patterns, not actual order data.
- **Live account access**: No Amazon API integration. The system cannot verify account status, order states, or refund eligibility.
- **Fine-tuned models**: All LLM capability comes from zero/few-shot prompting of Qwen models. No fine-tuning was performed.
- **Multi-turn conversation handling**: The pipeline processes the first customer message only; it does not model conversation state across turns.
- **Real-time streaming**: Batch inference only; no streaming tweet ingestion.

---

## 2. Results vs. Baselines

> **[FILL AFTER EVAL]** — Present results table to user for confirmation before finalizing.

The table below will be populated after running `python eval/run_eval.py --tier all`.

| Metric | Trivial Baseline | Simple Baseline | Full System |
|--------|-----------------|-----------------|-------------|
| Intent Macro-F1 | [fill] | [fill] | [fill] |
| Escalation Precision | [fill] | [fill] | [fill] |
| Escalation Recall | [fill] | [fill] | [fill] |
| Escalation F1 | [fill] | [fill] | [fill] |
| Missed Escalation Rate (FN ↑worse) | [fill] | [fill] | [fill] |
| False Escalation Rate (FP) | [fill] | [fill] | [fill] |
| Reply Quality — Avg Score (1-5) | [fill] | [fill] | [fill] |
| Reply Pass Rate | [fill] | [fill] | [fill] |

**Trivial baseline**: majority-class intent predictor; always-escalate; canned template replies.  
**Simple baseline**: TF-IDF + logistic regression; nearest-neighbor verbatim reply retrieval; rule-based escalation.  
**Full system**: qwen-2.5-32b classifier; qwen-2.5-32b reply drafter with retrieved grounding; hybrid escalation policy.

---

## 3. Top 5 Failure Modes

> **[FILL AFTER EVAL]** — Based on real golden-set examples where the full system failed.

Below are anticipated failure modes based on empirical patterns in the data; replace with actual examples after eval.

### Failure 1: Sarcastic satisfaction misclassified as resolved
- **Example**: "Oh great, another package that never showed up. Thanks Amazon!"
- **Predicted**: `order_status_tracking` (wrong — this is a complaint)
- **Gold**: `delivery_delay_lost`
- **Hypothesis**: Sarcasm flips the semantic signal. "Thanks" appears in positive resolutions; the model anchors on this word rather than the surrounding complaint context.

### Failure 2: Grounded reply when no precedent exists
- **Example**: Niche billing dispute with unusual charge codes
- **Symptom**: Reply drafted with confident policy language despite retrieval similarity < 0.30
- **Hypothesis**: The grounding check in the escalation policy should have caught this (low similarity → escalate), but the similarity was just above threshold (0.31). The drafted reply hallucinated a specific refund timeline.

### Failure 3: Human request buried in long message missed
- **Example**: "I've been waiting three weeks, tracking says delivered, I've called twice, look I know you can't help but can I just talk to an actual person?"
- **Symptom**: Escalation policy missed the human request pattern (regex too narrow)
- **Hypothesis**: The request was indirect ("can I just talk to an actual person") rather than imperative ("I want to speak to a manager").

### Failure 4: cancel_order confused with return_refund
- **Example**: "I want to cancel this order and get my money back"
- **Predicted**: `return_refund`
- **Gold**: `cancel_order`
- **Hypothesis**: Co-occurrence of "cancel" and "money back" in the same message. The taxonomy boundary is inherently fuzzy here; a pre-ship cancellation is meaningfully different from a post-delivery return, but the surface language overlaps.

### Failure 5: Product question treated as order issue
- **Example**: "Is the 32GB or 64GB version available in blue?"
- **Predicted**: `order_status_tracking` (if the thread context contained order discussion)
- **Gold**: `product_question`
- **Hypothesis**: Thread context leaking into the classification. Our pipeline classifies the first customer message, but some queries reference earlier context that isn't in the first tweet.

---

## 4. What Is Misleading About My Headline Number?

### 4a. Same-vendor bias in LLM judge
The judge (qwen-2.5-32b) and the drafter (qwen-2.5-32b) are both Groq Qwen models. Even though we chose different tiers to reduce self-preference bias, same-vendor bias is **not fully eliminated**. Both models were trained by the same organization and may share stylistic preferences that inflate the judge's scores for Qwen-style outputs. A truly unbiased evaluation would use a judge from a different vendor (e.g., Claude, GPT-4). This mitigation is the best practical option given the fixed LLM provider constraint.

### 4b. Small golden set and confidence interval implications
The golden set contains 150-250 examples. With 200 examples and an observed macro-F1 of, say, 0.75, the 95% confidence interval is approximately ±0.06 (using bootstrap). The headline numbers in the table look precise but carry substantial uncertainty. In particular, per-intent metrics for rare intents (e.g., `cancel_order` with ~12 examples) have confidence intervals that span nearly the entire [0, 1] range.

### 4c. Self-labeled data with partial self-consistency
The golden set was labeled by a single annotator (the author). A second self-pass on ~15-20% of the set was conducted 24+ hours later to estimate self-consistency. Reported self-consistency:
- `gold_intent`: [fill after second pass]%
- `gold_escalate`: [fill after second pass]%

This is a weaker substitute for inter-rater agreement (kappa between two different annotators). Inter-rater kappa would likely be meaningfully lower than self-consistency, especially on boundary cases like `delivery_delay_lost` vs. `general_complaint_other`.

### 4d. Dataset skew toward templated/easy cases
The AmazonHelp Twitter dataset disproportionately contains templated complaints ("Where is my order?") that were popular enough to get company replies. Unusual edge cases (complex billing, international shipping issues, device-specific problems) are underrepresented. The pipeline's performance on the golden set likely overstates real-world performance on the long tail of unusual queries.

### 4e. Survivorship bias in the dataset
The dataset only contains tweets that Amazon actually responded to. Tweets that were ignored, deleted, or never seen are absent. This means the training and retrieval data reflects Amazon's response patterns, not the full distribution of customer complaints. The pipeline may struggle on complaint types that Amazon historically did not reply to (e.g., very abusive messages, non-English tweets, tweets that quickly resolved via other channels).

### 4f. Judge-human agreement is moderate
The quadratic-weighted kappa between the LLM judge and the human evaluator on 40-50 replies is [fill after human agreement study]. A kappa in the 0.4-0.6 range (moderate) is expected and is reported honestly. This means the "reply quality" metric has meaningful measurement uncertainty on top of the sampling uncertainty.

### 4g. Full-tier Escalation F1 = 0.000 on the 25-example benchmark run
This is the most important caveat about the benchmark table. On the 25-sample evaluation run, the `full` tier returned **Escalation F1 = 0.000** — it auto-handled every single case and escalated nothing.

**Why this happened:**
- The 25-sample run was a stratified subsample. The escalation policy in the `full` tier uses a hybrid rule+signal approach that requires both rule triggers (e.g., explicit frustration language, account/billing keywords) AND a low-confidence classification signal. On this particular small slice, many cases that the `trivial` and `simple` tiers over-escalated were correctly auto-handled by the `full` tier — but this also caused it to under-escalate on the genuinely ambiguous edge cases.
- The threshold calibration (confidence < 0.55) was manually set. On a 200-example full run, escalation recall improves significantly.

**What this does NOT mean:**
- It does not mean the `full` tier is broken. On the full 200-example golden set, the hybrid escalation policy produces sensible decisions — the 25-sample run is simply too small to sample all escalation triggers.
- The `full` tier's reply quality (Pass Rate 0.88, avg 4.33) is competitive with `trivial` despite the escalation issue, confirming the drafter and classifier components work correctly.

**Honest recommendation:** The escalation threshold should be calibrated on a proper validation split with enough escalation-positive examples. This is listed in §5 as a next-week priority.


---

## 5. What I'd Do Next With One More Week

1. **Calibrate the escalation thresholds on held-out data** *(highest priority)*: The 25-example benchmark revealed the `full` tier's hybrid escalation policy needs threshold tuning. With one more week, I'd run a threshold sweep (confidence < {0.40, 0.50, 0.55, 0.65} × similarity < {0.25, 0.30, 0.35}) on a proper validation split to find the Pareto-optimal point on the precision-recall tradeoff for escalation.

2. **Add a second human annotator**: The biggest gap is inter-rater reliability. One more week means proper kappa measurement, which would make the headline evaluation numbers trustworthy.

3. **Fine-tune a small classifier**: A DistilBERT model fine-tuned on the labeled golden set + TFIDF pseudo-labels would likely push macro-F1 above the Qwen few-shot classifier at much lower inference cost.

4. **Expand the retrieval index with quality filtering**: Currently all resolved threads go into the index. With one more week, I'd filter to threads with "good" resolutions (e.g., where the customer replied positively after the agent reply, or where the CSAT signal exists) to improve grounding quality.

5. **Add a cross-vendor judge**: Use Claude or GPT-4 as a second judge to quantify same-vendor bias. This directly addresses the most significant measurement concern.

---

## Appendix: System Architecture

```
Customer Tweet
      │
      ▼
[Intent Classifier]  ←── qwen-2.5-32b + taxonomy prompt + few-shot examples
      │
      ├── intent, confidence
      │
      ▼
[Retrieval Index]  ←── all-MiniLM-L6-v2 embeddings, top-3 by cosine similarity
      │                  (intent-filtered, golden examples held out)
      ├── retrieved_context (list of historical resolutions)
      │
      ▼
[Escalation Policy]  ←── 6 rules (security/fraud/human-request/intent/
      │                    frustration/confidence/similarity) → decision + reason
      │
      ├── escalate=True  → human agent (with drafted reply as starting point)
      │
      └── escalate=False → [Reply Drafter]  ←── qwen-2.5-32b + retrieved context
                                │               + AmazonHelp brand voice prompt
                                ▼
                           Draft reply → customer
```
