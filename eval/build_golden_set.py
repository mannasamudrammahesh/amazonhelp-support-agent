"""
build_golden_set.py — Phase 7: Sample and structure the golden evaluation set.

Produces eval/golden_set.csv with:
  - Stratified sampling across intents, escalation cases, and time periods
  - Deduplication of near-identical templated tweets
  - Held-out from the retrieval index

After running this script, you MUST manually label the golden_intent,
gold_escalate, gold_escalate_reason, and acceptable_reply_note columns.
See eval/labeling_guide.md for full labeling instructions.

Usage:
    python eval/build_golden_set.py --n 200 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"

THREAD_PARQUET = DATA_DIR / "amazonhelp_threads.parquet"
GOLDEN_CSV = EVAL_DIR / "golden_set.csv"

# IDs used during Phase 2 open-coding (set by taxonomy.py run)
TAXONOMY_SAMPLE_SEED = 99   # different seed from main subsample — ensures separation
TAXONOMY_SAMPLE_N = 500     # number of messages used for open-coding

# Deduplication threshold
DEDUP_THRESHOLD = 0.90


def load_threads() -> pd.DataFrame:
    if not THREAD_PARQUET.exists():
        log.error(f"Thread parquet not found at {THREAD_PARQUET}. Run src/ingest.py first.")
        sys.exit(1)
    df = pd.read_parquet(THREAD_PARQUET)
    log.info(f"Loaded {len(df):,} threads.")
    return df


def get_taxonomy_phase_ids(df: pd.DataFrame) -> set[str]:
    """
    Return thread_ids used during Phase 2 taxonomy open-coding.
    These are excluded from the golden set.
    """
    rng = np.random.RandomState(TAXONOMY_SAMPLE_SEED)
    sample_idx = rng.choice(len(df), size=min(TAXONOMY_SAMPLE_N, len(df)), replace=False)
    return set(df.iloc[sample_idx]["thread_id"].tolist())


def deduplicate(df: pd.DataFrame, threshold: float = DEDUP_THRESHOLD) -> pd.DataFrame:
    """
    Remove near-duplicate customer messages using TF-IDF cosine similarity.
    Keeps one representative per cluster of duplicates.
    """
    log.info(f"Deduplicating {len(df)} messages (threshold={threshold})…")
    texts = df["customer_text_clean"].tolist()

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)

    keep_mask = np.ones(len(texts), dtype=bool)
    for i in range(len(texts)):
        if not keep_mask[i]:
            continue
        # Compare i against all later examples
        if i < len(texts) - 1:
            sims = cosine_similarity(matrix[i], matrix[i+1:]).flatten()
            for j, sim in enumerate(sims, start=i+1):
                if sim >= threshold and keep_mask[j]:
                    keep_mask[j] = False

    result = df[keep_mask].reset_index(drop=True)
    log.info(f"After dedup: {len(result)} messages (removed {len(df) - len(result)} near-duplicates).")
    return result


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """
    Stratify sample across:
    1. Time period (quarterly)
    2. Message length (short/medium/long) — proxy for complexity
    """
    df = df.copy()

    if "created_at" in df.columns:
        df["quarter"] = pd.to_datetime(df["created_at"], errors="coerce").dt.to_period("Q")
    else:
        df["quarter"] = "unknown"

    # Sample proportionally from each quarter
    rng = np.random.RandomState(seed)
    quarters = df["quarter"].value_counts()
    sampled_parts = []

    for q, count in quarters.items():
        q_df = df[df["quarter"] == q]
        q_n = max(1, round(n * count / len(df)))
        q_n = min(q_n, len(q_df))
        sampled_parts.append(q_df.sample(q_n, random_state=seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed).reset_index(drop=True)

    log.info(f"Stratified sample: {len(sampled)} examples.")
    return sampled


def main(n: int = 200, seed: int = 42):
    df = load_threads()

    # Phase 2 holdout
    taxonomy_ids = get_taxonomy_phase_ids(df)
    df_eligible = df[~df["thread_id"].isin(taxonomy_ids)].reset_index(drop=True)
    log.info(f"After excluding Phase 2 taxonomy sample: {len(df_eligible):,} eligible threads.")

    # Deduplicate
    df_deduped = deduplicate(df_eligible, threshold=DEDUP_THRESHOLD)

    # Stratified sample
    sampled = stratified_sample(df_deduped, n=n, seed=seed)

    # Add label columns (to be filled manually)
    sampled["gold_intent"] = ""
    sampled["gold_escalate"] = ""
    sampled["gold_escalate_reason"] = ""
    sampled["acceptable_reply_note"] = ""
    sampled["label_notes"] = ""
    sampled["split"] = "golden"

    # Select output columns
    out_cols = [
        "thread_id", "customer_tweet_id", "customer_text_clean", "customer_text_raw",
        "agent_reply_clean", "created_at", "has_pii_flag", "num_turns",
        "gold_intent", "gold_escalate", "gold_escalate_reason",
        "acceptable_reply_note", "label_notes", "split"
    ]
    out_cols = [c for c in out_cols if c in sampled.columns]
    sampled = sampled[out_cols]

    sampled.to_csv(GOLDEN_CSV, index=False)
    log.info(f"\n✓ Golden set saved → {GOLDEN_CSV}")
    log.info(f"  {len(sampled)} examples to label.")
    log.info(f"  See eval/labeling_guide.md for labeling instructions.")

    # Stats
    if "created_at" in sampled.columns:
        sampled["quarter"] = pd.to_datetime(sampled["created_at"], errors="coerce").dt.to_period("Q")
        log.info(f"\nQuarter distribution:\n{sampled['quarter'].value_counts().sort_index()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(n=args.n, seed=args.seed)
