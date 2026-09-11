# AmazonHelp Customer Support Pipeline

A take-home SDE Intern assignment submission: an end-to-end customer support automation pipeline for the AmazonHelp Twitter account.

## What This Does

Given an incoming customer tweet, the pipeline:
1. **Classifies** the intent (9 categories, from order tracking to billing disputes)
2. **Retrieves** the top-3 most similar historically-resolved cases as grounding context
3. **Decides** whether to escalate to a human agent (with a stated reason) or auto-handle
4. **Drafts** a reply grounded in real historical resolutions

Three tiers are implemented for comparison:

| Tier | Classifier | Reply | Escalation |
|------|-----------|-------|------------|
| **Trivial** | Majority class | Canned template | Always escalate |
| **Simple** | TF-IDF + LogReg | Verbatim NN retrieval | Rule-based |
| **Full** | Qwen 2.5 32B (LLM) | Qwen 2.5 32B (grounded gen) | Hybrid rule + signal |

---

## Reproduce the Headline Result in Under 15 Minutes

### Prerequisites

- Python 3.11+
- A `GROQ_API_KEY` from [console.groq.com](https://console.groq.com)
- The dataset subsample (see below)

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Configure secrets

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Step 3 — Get the data

**Option A**: Data subsample already included in `data/` (if the grader received a pre-packaged version):
```bash
# Skip this step — parquet files are already present
```

**Option B**: Regenerate from Kaggle (requires `~/.kaggle/kaggle.json`):
```bash
python src/ingest.py
# Takes ~5-10 minutes (downloads 177MB zip, processes 516MB CSV)
```

### Step 4 — Build the retrieval index

```bash
python src/retrieve.py
# Takes ~2-3 minutes (encodes 20,000 threads with sentence-transformers)
```

### Step 5 — Build + label the golden set

```bash
python eval/build_golden_set.py --n 25
# Then manually label eval/golden_set.csv per eval/labeling_guide.md
# A pre-labeled version is included if received as a complete package
```

### Step 6 — Run evaluation (headline result)

```bash
# Run all three tiers, print metrics table
python eval/run_eval.py --tier all

# Skip LLM judge to save API costs (faster iteration)
python eval/run_eval.py --tier full --skip-reply-judge
```

### Step 7 — See the metrics table

```
eval/results/metrics_table.md    # side-by-side comparison of all 3 tiers
eval/results/results_*.json      # detailed per-tier results
eval/results/confusion_*.csv     # intent confusion matrices
```

---

## Project Structure

```
/data/                          # Raw subsample + processed parquet (gitignored if large)
  amazonhelp_threads.parquet    # 20,000 reconstructed AmazonHelp threads
  amazonhelp_messages.parquet   # Flat customer-only messages

/src/
  ingest.py                     # Phase 1: Download, filter, thread reconstruction
  taxonomy.py                   # Phase 2: Intent definitions + prompting helpers
  classify.py                   # Phase 3+5: All three classifier tiers
  retrieve.py                   # Phase 4: Embedding retrieval index
  draft_reply.py                # Phase 5: All three reply drafter tiers
  escalate.py                   # Phase 6: Escalation policy (hybrid rules + signals)
  pipeline.py                   # End-to-end: message in → {intent, reply, escalate}

/eval/
  golden_set.csv                # 150-250 hand-labeled examples (held-out)
  labeling_guide.md             # Sampling methodology + label definitions
  judge_rubric.md               # LLM-as-judge rubric (5 dimensions, 1-5 scale)
  build_golden_set.py           # Phase 7: Golden set sampling script
  run_eval.py                   # Phase 8: Full automated evaluation harness
  human_agreement.py            # Phase 8: Judge-human agreement study

/report/
  REPORT.md                     # Full report (framing, results, failure modes, caveats)
  DECISION_LOG.md               # 15 non-obvious decisions with rationale

README.md
CITATIONS.md
.env.example
requirements.txt
```

---

## Quick Pipeline Test (Single Message)

```python
import sys; sys.path.insert(0, 'src')
from pipeline import AmazonHelpPipeline

pipeline = AmazonHelpPipeline(tier="full").setup()
result = pipeline.predict("My package shows delivered but I never received it!")

print(f"Intent: {result['intent']} (conf: {result['confidence']:.2f})")
print(f"Escalate: {result['escalate']} [{result['reason']}]")
print(f"Reply: {result['reply']}")
```

---

## Key Design Decisions

See [`report/DECISION_LOG.md`](report/DECISION_LOG.md) for 15 documented decisions. Highlights:

- **qwen-2.5-32b** for classification (high volume → cheapest tier)
- **qwen-2.5-32b** for reply drafting (quality matters → mid tier)
- **qwen-2.5-32b** for LLM judge (different tier than drafter → reduces self-preference bias; same-vendor bias not fully eliminated — noted in report)
- **all-MiniLM-L6-v2** for embeddings (local, fast, free — no inference cost)
- Golden set **held out** from retrieval index (critical eval hygiene)
- Escalation policy has **6 explicit rules** each producing a typed reason string

---

## Limitations

- English-only (non-English ~5-8% of data → escalated)
- No live order lookup or account access
- Same-vendor bias in LLM judge (Groq Qwen model used for both drafting and judging)
- Single annotator for golden set (self-consistency, not inter-rater agreement)
- Evaluation on ~200 examples → ±6% CI on macro-F1

See [`report/REPORT.md § 4`](report/REPORT.md) for full "what's misleading" section.
