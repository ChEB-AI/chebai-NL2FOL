#!/usr/bin/env python3
"""
Script to calculate F1 score statistics from c3p_train_val_scores.json
"""

import argparse
import json
from pathlib import Path


def calculate_f1_score(tp, fp, fn):
    """Calculate F1 score from TP, FP, FN."""
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
    return (2 * tp) / denominator


def categorize_f1_scores(json_file_path, split="val"):
    """Load c3p_train_val_scores.json and categorize F1 scores."""
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Expected c3p_train_val_scores.json to contain a list of class records."
        )

    # Initialize bins
    bins = {
        "score == 1.0": 0,
        "0.8 <= score < 1.0": 0,
        "0.6 <= score < 0.8": 0,
        "0.4 <= score < 0.6": 0,
        "0.2 <= score < 0.4": 0,
        "0.0 < score < 0.2": 0,
        "score == 0.0": 0,
    }

    f1_scores = []
    failed_to_learn = 0

    # Calculate F1 scores and categorize

    for entry in data:
        metrics = entry.get(split)

        tp = metrics.get("num_true_positives")
        fp = metrics.get("num_false_positives")
        fn = metrics.get("num_false_negatives")

        f1 = metrics.get("f1")
        f1 = calculate_f1_score(tp, fp, fn)
        f1_scores.append(f1)

        # Categorize
        if f1 == 1.0:
            bins["score == 1.0"] += 1
        elif 0.8 <= f1 < 1.0:
            bins["0.8 <= score < 1.0"] += 1
        elif 0.6 <= f1 < 0.8:
            bins["0.6 <= score < 0.8"] += 1
        elif 0.4 <= f1 < 0.6:
            bins["0.4 <= score < 0.6"] += 1
        elif 0.2 <= f1 < 0.4:
            bins["0.2 <= score < 0.4"] += 1
        elif 0.0 < f1 < 0.2:
            bins["0.0 < score < 0.2"] += 1
        elif f1 == 0.0:
            bins["score == 0.0"] += 1
            # Count cases that failed to learn (tp=0, fn>0)
            if tp == 0 and fn > 0:
                failed_to_learn += 1

    total = len(data)

    # Print results
    print(f"F1 Score Distribution for '{split}' (Total: {total})")
    print("=" * 60)

    for bin_name, count in bins.items():
        percentage = (count / total) * 100
        if bin_name == "score == 0.0":
            print(
                f"\t{bin_name}: {count} (out of which {failed_to_learn} failed to learn) ({percentage:.2f}%)"
            )
        else:
            print(f"\t{bin_name}: {count} ({percentage:.2f}%)")

    print("=" * 60)

    print(f"Mean F1 Score: {sum(f1_scores) / len(f1_scores):.4f}")
    print(f"Min F1 Score: {min(f1_scores):.4f}")
    print(f"Max F1 Score: {max(f1_scores):.4f}")

    # Return results for potential further use
    return bins, total, failed_to_learn, f1_scores


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate F1 score distribution for c3p train/val metrics."
    )
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--json-path",
        type=Path,
        default=repo_root / "data" / "c3p_train_val_scores.json",
        help="Path to c3p_train_val_scores.json file.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Which metric split to analyze.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    json_path = args.json_path

    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        exit(1)

    categorize_f1_scores(json_path, split=args.split)
