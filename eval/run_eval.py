"""
run_eval.py — Phase 8: Full evaluation harness.

Runs all three pipeline tiers against the golden set and computes:
  - Intent: macro-F1, per-intent P/R/F1, confusion matrix
  - Escalation: precision/recall/F1 (separately for FP and FN)
  - Reply quality: LLM-as-judge aggregate + per-dimension scores + pass rate

Usage:
    python eval/run_eval.py --tier all          # run all three tiers
    python eval/run_eval.py --tier full         # run only the full Grok tier
    python eval/run_eval.py --tier trivial      # trivial baseline only
    python eval/run_eval.py --skip-reply-judge  # skip LLM judge (faster)

Outputs:
    eval/results/results_{tier}.json
    eval/results/metrics_table.md     (all tiers side-by-side)
    eval/results/confusion_{tier}.csv

Decision log:
- Judge model: grok-3-mini (different tier from drafting model grok-3).
  Intentional choice to reduce same-model self-preference bias.
- FN escalation (false auto-handle) is reported separately from FP escalation
  (false escalate) because missing a real escalation case is a worse error.
- Macro-F1 for intent (not weighted) because we care about rare-intent performance.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

sys.path.insert(0, str(SRC))

from pipeline import AmazonHelpPipeline
from taxonomy import INTENT_NAMES

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

JUDGE_MODEL = "grok-3-mini"   # deliberate: different tier from drafting model (grok-3)
_OPENAI_BASE_URL = "https://api.x.ai/v1"


# ── Golden set loading ────────────────────────────────────────────────────────

def load_golden(golden_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(golden_csv)
    required = ["customer_text_clean", "gold_intent", "gold_escalate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Golden set missing columns: {missing}")
    df["gold_escalate"] = df["gold_escalate"].astype(bool)
    log.info(f"Loaded {len(df)} golden examples.")
    return df


# ── Intent metrics ────────────────────────────────────────────────────────────

def compute_intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "macro_f1": macro_f1,
        "per_intent": {k: v for k, v in report.items() if k in labels},
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
    }


def save_confusion_matrix(metrics: dict, tier: str):
    labels = metrics["confusion_labels"]
    cm = np.array(metrics["confusion_matrix"])
    df = pd.DataFrame(cm, index=labels, columns=labels)
    out = RESULTS_DIR / f"confusion_{tier}.csv"
    df.to_csv(out)
    log.info(f"Confusion matrix saved → {out}")


# ── Escalation metrics ────────────────────────────────────────────────────────

def compute_escalation_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    """
    Reports escalation metrics with FP and FN separated.
    FN (predict auto-handle, truth=escalate) is the worse error.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tp = sum(1 for a, b in zip(y_true, y_pred) if a and b)
    fp = sum(1 for a, b in zip(y_true, y_pred) if not a and b)   # false escalate
    fn = sum(1 for a, b in zip(y_true, y_pred) if a and not b)   # false auto-handle (WORSE)
    tn = sum(1 for a, b in zip(y_true, y_pred) if not a and not b)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "false_escalate_rate": fp / max(fp + tn, 1),
        "missed_escalation_rate": fn / max(fn + tp, 1),
        "note": "FN (missed escalation) is the worse error type — reported separately.",
    }


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

def _judge_prompt(customer_message: str, reply: str, intent: str, retrieved_context: list) -> str:
    ctx_str = ""
    if retrieved_context:
        ctx_str = "\n".join(
            f"  Example {i+1}: Customer: {r['customer_text'][:100]} | Reply: {r['agent_reply'][:150]}"
            for i, r in enumerate(retrieved_context)
        )
    else:
        ctx_str = "  (No retrieved context — model had no grounding examples)"

    return f"""You are evaluating a customer support reply for AmazonHelp (Amazon's Twitter support account).

## Customer Message
Intent: {intent}
Message: {customer_message[:500]}

## Retrieved Historical Precedents (grounding context the system had access to)
{ctx_str}

## Draft Reply to Evaluate
{reply}

## Rubric (score each 1-5, then give pass/fail)
1. Groundedness: Does the reply avoid inventing specifics (amounts, timelines) not in retrieved examples?
   1=fabricated policies, 3=mix, 5=fully grounded in evidence
2. Relevance: Does it address the customer's actual ask?
   1=entirely off-topic, 3=partially relevant, 5=directly on-point
3. Factual_Consistency: No internal contradictions or impossible claims?
   1=contradicts itself, 3=one issue, 5=fully consistent
4. Tone: Matches AmazonHelp's warm, professional, empathetic brand voice?
   1=rude/defensive, 3=acceptable but stiff, 5=perfect brand tone
5. Actionability: Clear next step the customer can act on?
   1=no guidance, 3=vague, 5=specific and immediate

## Pass/Fail Gate
FAIL if ANY: promised refund amount/timeline not in retrieved examples, 
asks for passwords/card numbers, answers wrong question, any dimension=1.

Output ONLY valid JSON:
{{
  "groundedness": <1-5>,
  "relevance": <1-5>,
  "factual_consistency": <1-5>,
  "tone": <1-5>,
  "actionability": <1-5>,
  "pass_fail": "<PASS|FAIL>",
  "fail_reason": "<brief reason or null>",
  "aggregate": <float, average of 5 scores>
}}"""


def judge_reply(
    customer_message: str,
    reply: str,
    intent: str,
    retrieved_context: list,
    client,
    max_retries: int = 3,
) -> dict:
    prompt = _judge_prompt(customer_message, reply, intent, retrieved_context)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            # Ensure aggregate
            dims = ["groundedness", "relevance", "factual_consistency", "tone", "actionability"]
            scores = [parsed.get(d, 3) for d in dims]
            parsed["aggregate"] = round(sum(scores) / len(scores), 2)
            return parsed
        except json.JSONDecodeError as e:
            log.warning(f"Judge JSON error attempt {attempt+1}: {e}")
        except Exception as e:
            log.warning(f"Judge API error attempt {attempt+1}: {e}")
            time.sleep(2 * (attempt + 1))

    return {
        "groundedness": 0, "relevance": 0, "factual_consistency": 0,
        "tone": 0, "actionability": 0, "pass_fail": "ERROR",
        "fail_reason": "judge_api_error", "aggregate": 0.0,
    }


def run_judge_eval(predictions: list[dict], golden: pd.DataFrame, skip_judge: bool = False) -> dict:
    if skip_judge:
        return {"skipped": True}

    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        log.warning("XAI_API_KEY not set — skipping LLM judge.")
        return {"skipped": True, "reason": "no_api_key"}

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=_OPENAI_BASE_URL)

    judge_results = []
    for i, (pred, (_, row)) in enumerate(zip(predictions, golden.iterrows())):
        log.info(f"Judging reply {i+1}/{len(predictions)}…")
        result = judge_reply(
            customer_message=row["customer_text_clean"],
            reply=pred.get("reply", ""),
            intent=pred.get("intent", ""),
            retrieved_context=[],  # not stored in predictions — use empty
            client=client,
        )
        judge_results.append(result)
        time.sleep(0.2)  # light throttle

    # Aggregate
    valid = [r for r in judge_results if r.get("pass_fail") != "ERROR"]
    dims = ["groundedness", "relevance", "factual_consistency", "tone", "actionability"]
    avg_scores = {d: np.mean([r.get(d, 0) for r in valid]) for d in dims}
    pass_rate = sum(1 for r in valid if r.get("pass_fail") == "PASS") / max(len(valid), 1)

    return {
        "per_example": judge_results,
        "avg_scores": avg_scores,
        "aggregate_mean": round(float(np.mean([r["aggregate"] for r in valid])), 3),
        "pass_rate": round(pass_rate, 3),
        "n_valid": len(valid),
        "n_errors": len(judge_results) - len(valid),
    }


# ── Full eval run ─────────────────────────────────────────────────────────────

def run_tier(tier: str, golden: pd.DataFrame, skip_judge: bool = False) -> dict:
    log.info(f"\n{'='*60}")
    log.info(f"Running eval: tier={tier}")
    log.info(f"{'='*60}")

    # Initialize pipeline
    pipeline = AmazonHelpPipeline(tier=tier)

    # For simple tier, fit TFIDF on non-golden data (use thread data if available)
    if tier == "simple":
        thread_parquet = ROOT / "data" / "amazonhelp_threads.parquet"
        if thread_parquet.exists():
            df_train = pd.read_parquet(thread_parquet)
            # Exclude golden thread IDs
            golden_ids = set(golden.get("thread_id", pd.Series()).dropna())
            df_train = df_train[~df_train["thread_id"].isin(golden_ids)]
            # Use synthetic labels from intent column if it exists, else skip
            if "intent" in df_train.columns or "gold_intent" in df_train.columns:
                label_col = "gold_intent" if "gold_intent" in df_train.columns else "intent"
                df_train = df_train.dropna(subset=["customer_text_clean", label_col])
                pipeline.setup(
                    labeled_texts=df_train["customer_text_clean"].tolist(),
                    labeled_labels=df_train[label_col].tolist()
                )
            else:
                log.warning("No intent labels in thread data — TFIDF will use golden labels only (small training set).")
                pipeline.setup()
        else:
            log.warning("Thread parquet not found — TFIDF will be unfitted.")
            pipeline.setup()
    else:
        pipeline.setup()

    # Run predictions
    predictions = []
    texts = golden["customer_text_clean"].tolist()

    for i, text in enumerate(texts):
        if i % 20 == 0:
            log.info(f"  Predicting {i+1}/{len(texts)}…")
        pred = pipeline.predict(text)
        predictions.append(pred)
        if tier == "full":
            time.sleep(0.15)  # throttle for Grok API

    # Intent metrics
    y_intent_true = golden["gold_intent"].tolist()
    y_intent_pred = [p["intent"] for p in predictions]
    intent_metrics = compute_intent_metrics(y_intent_true, y_intent_pred)
    save_confusion_matrix(intent_metrics, tier)

    # Escalation metrics
    y_esc_true = golden["gold_escalate"].tolist()
    y_esc_pred = [p["escalate"] for p in predictions]
    esc_metrics = compute_escalation_metrics(y_esc_true, y_esc_pred)

    # LLM judge for reply quality
    judge_metrics = run_judge_eval(predictions, golden, skip_judge=skip_judge)

    results = {
        "tier": tier,
        "n_examples": len(golden),
        "intent": intent_metrics,
        "escalation": esc_metrics,
        "reply_quality": judge_metrics,
        "predictions": predictions,
    }

    # Save
    out_file = RESULTS_DIR / f"results_{tier}.json"
    # Don't serialize numpy arrays
    results_serializable = {k: v for k, v in results.items() if k != "predictions"}
    results_serializable["intent"]["confusion_matrix"] = intent_metrics["confusion_matrix"]
    with open(out_file, "w") as f:
        json.dump(results_serializable, f, indent=2, default=str)
    log.info(f"Results saved → {out_file}")

    return results


def print_metrics_table(all_results: dict[str, dict]):
    """Print a clean comparison table across all tiers."""
    tiers = list(all_results.keys())
    print("\n" + "=" * 80)
    print("RESULTS TABLE — AmazonHelp Customer Support Pipeline")
    print("=" * 80)

    header = f"{'Metric':<40}" + "".join(f"{t:<18}" for t in tiers)
    print(header)
    print("-" * (40 + 18 * len(tiers)))

    metrics_to_show = [
        ("Intent Macro-F1", lambda r: f"{r['intent']['macro_f1']:.3f}"),
        ("Escalation Precision", lambda r: f"{r['escalation']['precision']:.3f}"),
        ("Escalation Recall", lambda r: f"{r['escalation']['recall']:.3f}"),
        ("Escalation F1", lambda r: f"{r['escalation']['f1']:.3f}"),
        ("Missed Escalation Rate (FN)", lambda r: f"{r['escalation']['missed_escalation_rate']:.3f}"),
        ("False Escalation Rate (FP)", lambda r: f"{r['escalation']['false_escalate_rate']:.3f}"),
        ("Reply Quality (avg)", lambda r: f"{r['reply_quality'].get('aggregate_mean', 'N/A')}"),
        ("Reply Pass Rate", lambda r: f"{r['reply_quality'].get('pass_rate', 'N/A')}"),
    ]

    for label, fn in metrics_to_show:
        row = f"{label:<40}"
        for tier in tiers:
            try:
                row += f"{fn(all_results[tier]):<18}"
            except Exception:
                row += f"{'N/A':<18}"
        print(row)

    print("=" * 80)

    # Save as markdown
    md_lines = [
        "# Metrics Table — AmazonHelp Pipeline\n",
        "| Metric | " + " | ".join(tiers) + " |",
        "|--------|" + "|".join(["--------|"] * len(tiers)),
    ]
    for label, fn in metrics_to_show:
        cells = []
        for tier in tiers:
            try:
                cells.append(fn(all_results[tier]))
            except Exception:
                cells.append("N/A")
        md_lines.append(f"| {label} | " + " | ".join(cells) + " |")

    table_path = RESULTS_DIR / "metrics_table.md"
    table_path.write_text("\n".join(md_lines))
    log.info(f"Metrics table saved → {table_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run AmazonHelp pipeline evaluation.")
    parser.add_argument("--tier", default="all", choices=["trivial", "simple", "full", "all"])
    parser.add_argument("--golden", default=str(EVAL_DIR / "golden_set.csv"))
    parser.add_argument("--skip-reply-judge", action="store_true",
                        help="Skip LLM-as-judge step (saves API costs when iterating).")
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        log.error(f"Golden set not found at {golden_path}. Run build_golden_set.py first.")
        sys.exit(1)

    golden = load_golden(golden_path)

    tiers = ["trivial", "simple", "full"] if args.tier == "all" else [args.tier]
    all_results = {}

    for tier in tiers:
        results = run_tier(tier, golden, skip_judge=args.skip_reply_judge)
        all_results[tier] = results

    print_metrics_table(all_results)


if __name__ == "__main__":
    main()
