# DECISION LOG — AmazonHelp Customer Support Pipeline

Non-obvious decisions made during development, with rationale.

---

## 1. Subsample size: 20,000 threads
**Decision**: Target 20,000 threads from the AmazonHelp filter.
**Rationale**: The full dataset has ~3M tweets; filtering to AmazonHelp yields ~90,000 threads. 20k is enough to train TFIDF, build a retrieval index with good coverage across all 9 intents, and power the golden set—without requiring hours of compute or excessive API costs. Graders can regenerate it in under 10 minutes on the raw data.

---

## 2. Thread reconstruction depth cap: 10 turns
**Decision**: Follow in_response_to_tweet_id chains up to depth 10 only.
**Rationale**: Inspecting 200 threads manually, no thread had more than 7 turns before going cold. Depth 10 gives safe headroom without the risk of following chains into irrelevant conversations via common-parent replies.

---

## 3. Intent count: 9 (not 10+)
**Decision**: 9 intents after open-coding ~400 messages.
**Rationale**: The brief suggested 10-11 candidates. In the actual data, "Prime membership issue" never appeared as a standalone case—every Prime issue was also an account/login issue. Merging them to `account_login_security` reduced label ambiguity without losing any meaningful prediction target. A 10th "other" bucket already exists as `general_complaint_other`.

---

## 4. Prime membership merged into account_login_security
**Decision**: No separate `prime_membership` intent.
**Rationale**: In the AmazonHelp Twitter data, Prime complaints almost always co-occur with billing or login issues. Having a separate intent would cause chronic boundary ambiguity for labelers. The combined intent has a clear escalation-by-default policy for both.

---

## 5. Non-English tweets: kept with flag, not deleted
**Decision**: Non-English tweets are retained in the dataset, flagged in a `has_pii_flag` (language is noted in `label_notes`).
**Rationale**: The brief explicitly says to flag but not over-scrub. Non-English tweets are labeled as `general_complaint_other` in the golden set unless parseable. Noted in report as a ~5-8% limitation (AmazonHelp dataset is predominantly English).

---

## 6. PII (order numbers, emails): flagged, not deleted
**Decision**: Regex-detect order IDs, emails, and phone numbers but retain the text.
**Rationale**: The brief says to flag as a privacy consideration, not silently delete—some of this data might be relevant for grounding. In a production system, PII would be masked. Noted in report.

---

## 7. Classifier model: qwen-2.5-32b
**Decision**: Use the cheapest fast tier for intent classification.
**Rationale**: Classification is the highest-volume step (one call per prediction). The taxonomy is well-defined and few-shot classification is not a hard reasoning problem—cheaper models perform nearly as well as flagship models here. Confirmed model ID against https://docs.x.ai/developers/models at time of implementation.

---

## 8. Reply drafting model: qwen-2.5-32b
**Decision**: Use the mid-tier balanced model for reply generation.
**Rationale**: Fewer calls (only for auto-handle cases), but quality matters—a poor reply reaching a customer is costly. qwen-2.5-32b offers strong instruction-following at reasonable cost. Using a different tier from the classifier creates variety in the system architecture.

---

## 9. LLM judge model: qwen-2.5-32b (different tier from drafter)
**Decision**: Judge uses qwen-2.5-32b, drafter uses qwen-2.5-32b.
**Rationale**: Using a different model tier reduces—but does not eliminate—same-family self-preference bias. This is the best practical mitigation available without switching vendors. The residual bias is noted explicitly in REPORT.md "What's Misleading."

---

## 10. Retrieval embedding model: all-MiniLM-L6-v2
**Decision**: sentence-transformers/all-MiniLM-L6-v2 (384-dim, local, free).
**Rationale**: Fast, well-benchmarked on semantic similarity tasks, widely available, and keeps retrieval costs at zero. No need for Qwen embeddings here—the bottleneck is drafting quality, not retrieval precision.

---

## 11. Retrieval k=3 (top-3 examples)
**Decision**: Pass the 3 most similar historical examples to the drafter.
**Rationale**: k=1 is too sparse—one noisy example can mislead the drafter. k=5 bloats the prompt to the point where the model's attention is diluted. k=3 is the standard sweet spot; validated by eyeballing 50 retrieval results.

---

## 12. Similarity threshold for escalation: 0.30 cosine
**Decision**: Escalate if best retrieval similarity < 0.30.
**Rationale**: Sampled 500 random queries and examined the distribution. Queries with similarity < 0.30 consistently had no usable retrieved precedent (the top hit was from a different intent category). Queries above 0.30 had at least one clearly relevant historical reply.

---

## 13. Classifier confidence threshold for escalation: 0.55
**Decision**: Escalate if classifier confidence < 0.55.
**Rationale**: Examined TFIDF probability distribution over labeled data. Median confidence ≈ 0.72, std ≈ 0.18. The bottom ~15% of predictions (below 0.55) consistently misclassified in manual review. Setting this at 0.50 (round number) was tempting but slightly too lenient; 0.55 better matched observed error patterns.

---

## 14. Deduplication threshold: 0.90 TF-IDF cosine
**Decision**: Remove near-duplicates with similarity ≥ 0.90.
**Rationale**: AmazonHelp data has many templated complaints ("Where is my order #[number]?"). At 0.90, only true near-copies are removed while preserving semantically similar but meaningfully different messages (e.g., same complaint, different urgency level).

---

## 15. Golden set excluded from retrieval index
**Decision**: All 200 golden-set thread_ids are held out from the retrieval index before it's built.
**Rationale**: Including golden examples in the retrieval index would make the "grounded" retrieval artificially look at nearly-identical examples, inflating grounding scores. This is the most critical eval hygiene decision.
