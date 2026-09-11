"""
human_agreement.py — Phase 8: Judge-human agreement study.

Loads the judge scores from run_eval.py output and prompts the evaluator
to score the same ~40 replies independently, then computes:
  - Quadratic-weighted Cohen's kappa per dimension
  - Pearson correlation per dimension
  - Exact-match % per dimension
  - Aggregate kappa across all dimensions

Usage:
    # Step 1: Create human scores file (interactive)
    python eval/human_agreement.py --create-human-scores --n 45

    # Step 2: Compute agreement
    python eval/human_agreement.py --compute --judge-results eval/results/results_full.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

ROOT = Path(__file__).parent.parent
EVAL_DIR = ROOT / "eval"
HUMAN_SCORES_PATH = EVAL_DIR / "human_scores.json"
RESULTS_DIR = EVAL_DIR / "results"

DIMENSIONS = ["groundedness", "relevance", "factual_consistency", "tone", "actionability"]


def quadratic_weighted_kappa(y1: list[int], y2: list[int], min_rating=1, max_rating=5) -> float:
    """Compute quadratic-weighted Cohen's kappa between two rating lists."""
    n = max_rating - min_rating + 1
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            weights[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

    hist1 = np.zeros(n)
    hist2 = np.zeros(n)
    conf_matrix = np.zeros((n, n))
    for a, b in zip(y1, y2):
        ai = a - min_rating
        bi = b - min_rating
        hist1[ai] += 1
        hist2[bi] += 1
        conf_matrix[ai, bi] += 1

    expected = np.outer(hist1, hist2) / len(y1)
    observed_error = np.sum(weights * conf_matrix) / len(y1)
    expected_error = np.sum(weights * expected) / len(y1)

    if expected_error == 0:
        return 1.0
    return 1.0 - (observed_error / expected_error)


def create_human_scores_interactive(judge_results_path: Path, n_samples: int = 45) -> None:
    """
    Interactive prompt: shows each reply and asks the evaluator to score it.
    Saves results to HUMAN_SCORES_PATH.
    """
    with open(judge_results_path) as f:
        results = json.load(f)

    judge_per_example = results.get("reply_quality", {}).get("per_example", [])
    if not judge_per_example:
        log.error("No per-example judge scores found. Run run_eval.py first.")
        sys.exit(1)

    # Also need the predictions to show the reply text
    pred_file = judge_results_path.parent / judge_results_path.name.replace("results_", "predictions_")
    # We'll use a sample index
    n_samples = min(n_samples, len(judge_per_example))
    sample_indices = sorted(random.sample(range(len(judge_per_example)), n_samples))

    # Load golden set for the customer messages
    golden_csv = EVAL_DIR / "golden_set.csv"
    golden = pd.read_csv(golden_csv) if golden_csv.exists() else None

    print("\n" + "=" * 70)
    print("HUMAN SCORING SESSION — AmazonHelp Reply Quality")
    print("=" * 70)
    print(f"You will score {n_samples} replies on 5 dimensions (1-5 each).")
    print("DO NOT look at the judge scores first — score independently.\n")
    print("Dimensions: groundedness, relevance, factual_consistency, tone, actionability")
    print("(See eval/judge_rubric.md for full scoring criteria)\n")
    input("Press Enter when ready to start...\n")

    human_scores = []
    for count, idx in enumerate(sample_indices, 1):
        judge_score = judge_per_example[idx]
        print(f"\n--- Example {count}/{n_samples} (index {idx}) ---")

        if golden is not None and idx < len(golden):
            row = golden.iloc[idx]
            print(f"Customer: {row['customer_text_clean'][:200]}")
            print(f"Intent: {row['gold_intent']}")
        else:
            print("(Customer message not available)")

        # Show reply if we have it
        # (In practice, predictions are in the results JSON)
        print("\nRate this reply (1=worst, 5=best) for each dimension:")

        scores = {"example_index": idx}
        for dim in DIMENSIONS:
            while True:
                try:
                    val = int(input(f"  {dim} (1-5): ").strip())
                    if 1 <= val <= 5:
                        scores[dim] = val
                        break
                    print("  → Please enter a number between 1 and 5.")
                except (ValueError, KeyboardInterrupt):
                    print("  → Invalid input. Try again.")

        pf = input("  pass_fail (P/F): ").strip().upper()
        scores["pass_fail"] = "PASS" if pf == "P" else "FAIL"

        human_scores.append(scores)
        print(f"  ✓ Recorded.")

    # Save
    with open(HUMAN_SCORES_PATH, "w") as f:
        json.dump({"sample_indices": sample_indices, "scores": human_scores}, f, indent=2)
    print(f"\n✓ Human scores saved to {HUMAN_SCORES_PATH}")
    print("Now run: python eval/human_agreement.py --compute")


def compute_agreement(judge_results_path: Path) -> dict:
    """Compute agreement metrics between human and judge scores."""
    with open(judge_results_path) as f:
        results = json.load(f)

    judge_per_example = results.get("reply_quality", {}).get("per_example", [])

    with open(HUMAN_SCORES_PATH) as f:
        human_data = json.load(f)

    sample_indices = human_data["sample_indices"]
    human_scores = human_data["scores"]

    agreement_results = {}

    for dim in DIMENSIONS:
        judge_vals = []
        human_vals = []
        for hi, (idx, hs) in enumerate(zip(sample_indices, human_scores)):
            if idx < len(judge_per_example):
                judge_score = judge_per_example[idx].get(dim, 0)
                human_score = hs.get(dim, 0)
                if 1 <= judge_score <= 5 and 1 <= human_score <= 5:
                    judge_vals.append(int(judge_score))
                    human_vals.append(int(human_score))

        if len(judge_vals) < 2:
            agreement_results[dim] = {"error": "insufficient_data"}
            continue

        kappa = quadratic_weighted_kappa(human_vals, judge_vals)
        corr, pval = pearsonr(human_vals, judge_vals)
        exact_match = sum(1 for a, b in zip(human_vals, judge_vals) if a == b) / len(human_vals)
        within_1 = sum(1 for a, b in zip(human_vals, judge_vals) if abs(a - b) <= 1) / len(human_vals)

        agreement_results[dim] = {
            "qwk": round(kappa, 3),
            "pearson_r": round(float(corr), 3),
            "pearson_p": round(float(pval), 4),
            "exact_match_pct": round(100 * exact_match, 1),
            "within_1_pct": round(100 * within_1, 1),
            "n": len(human_vals),
        }

    # Pass/fail agreement
    judge_pf = []
    human_pf = []
    for hi, (idx, hs) in enumerate(zip(sample_indices, human_scores)):
        if idx < len(judge_per_example):
            jp = judge_per_example[idx].get("pass_fail", "")
            hp = hs.get("pass_fail", "")
            if jp in ("PASS", "FAIL") and hp in ("PASS", "FAIL"):
                judge_pf.append(1 if jp == "PASS" else 0)
                human_pf.append(1 if hp == "PASS" else 0)

    pf_agreement = sum(1 for a, b in zip(judge_pf, human_pf) if a == b) / max(len(judge_pf), 1)

    # Print results
    print("\n" + "=" * 60)
    print("JUDGE-HUMAN AGREEMENT RESULTS")
    print("=" * 60)
    print(f"{'Dimension':<25} {'QWK':>6} {'Pearson':>8} {'Exact%':>7} {'Within1%':>9}")
    print("-" * 60)
    for dim, res in agreement_results.items():
        if "error" not in res:
            flag = " ⚠️ LOW" if res["qwk"] < 0.40 else ""
            print(f"{dim:<25} {res['qwk']:>6.3f} {res['pearson_r']:>8.3f} "
                  f"{res['exact_match_pct']:>7.1f} {res['within_1_pct']:>9.1f}{flag}")

    print(f"\nPass/Fail agreement: {100*pf_agreement:.1f}%")
    print(f"\nNote: QWK < 0.40 indicates low reliability for that dimension.")
    print("=" * 60)

    # Save
    output = {
        "per_dimension": agreement_results,
        "pass_fail_agreement_pct": round(100 * pf_agreement, 1),
        "n_samples": len(sample_indices),
    }
    out_path = RESULTS_DIR / "human_agreement.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Agreement results saved → {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-human-scores", action="store_true")
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--judge-results", default=str(RESULTS_DIR / "results_full.json"))
    parser.add_argument("--n", type=int, default=45, help="Number of examples to score")
    args = parser.parse_args()

    judge_path = Path(args.judge_results)

    if args.create_human_scores:
        create_human_scores_interactive(judge_path, n_samples=args.n)
    elif args.compute:
        if not HUMAN_SCORES_PATH.exists():
            log.error(f"Human scores not found at {HUMAN_SCORES_PATH}. Run --create-human-scores first.")
            sys.exit(1)
        compute_agreement(judge_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
