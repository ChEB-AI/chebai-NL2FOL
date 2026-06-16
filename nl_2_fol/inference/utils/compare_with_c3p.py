import argparse
import csv
import json
import os
import pickle
from pathlib import Path

from nl_2_fol.inference.learner.definition_model import DefinitionLearningResults
from nl_2_fol.prompting.custom_api._test_inference import PROJECT_DIR


def _load_c3p_trust(c3p_path: Path) -> dict[int, dict[str, float]]:
    # Expecting a JSON array of objects produced by the c3p script
    # Each object contains a `chebi_id` like "CHEBI:12345" and a `val` block
    # where the validation F1 score is stored under the `f1` key.
    with open(c3p_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    parsed: dict[int, dict[str, float]] = {}
    for entry in raw:
        chebi_str = entry["chebi_id"]
        # chebi_id may be like "CHEBI:84948" — extract numeric part
        try:
            chebi_num = int(str(chebi_str).split(":")[-1])
        except Exception:
            # skip malformed ids
            raise ValueError(f"Malformed chebi_id: {chebi_str}")
        parsed[chebi_num] = entry["val"]
    return parsed


def _load_learned_definitions(
    learned_pickle_path: Path,
) -> DefinitionLearningResults:
    with open(learned_pickle_path, "rb") as f:
        return pickle.load(f)


def calculate_micro_macro_metrics(
    metric_records: list[dict],
) -> dict[str, float] | None:
    if not metric_records:
        return None

    total_tp = sum(record["num_true_positives"] for record in metric_records)
    total_fp = sum(record["num_false_positives"] for record in metric_records)
    total_fn = sum(record["num_false_negatives"] for record in metric_records)

    micro_precision_denom = total_tp + total_fp
    micro_recall_denom = total_tp + total_fn
    micro_precision = (
        total_tp / micro_precision_denom if micro_precision_denom > 0 else 0.0
    )
    micro_recall = total_tp / micro_recall_denom if micro_recall_denom > 0 else 0.0
    micro_f1_denom = 2 * total_tp + total_fp + total_fn
    micro_f1 = (2 * total_tp / micro_f1_denom) if micro_f1_denom > 0 else 0.0

    per_record_precisions = [
        (
            record["num_true_positives"]
            / (record["num_true_positives"] + record["num_false_positives"])
        )
        if (record["num_true_positives"] + record["num_false_positives"]) > 0
        else 0.0
        for record in metric_records
    ]
    per_record_recalls = [
        (
            record["num_true_positives"]
            / (record["num_true_positives"] + record["num_false_negatives"])
        )
        if (record["num_true_positives"] + record["num_false_negatives"]) > 0
        else 0.0
        for record in metric_records
    ]
    macro_precision = sum(per_record_precisions) / len(per_record_precisions)
    macro_recall = sum(per_record_recalls) / len(per_record_recalls)
    macro_f1 = sum(float(record["f1"]) for record in metric_records) / len(
        metric_records
    )

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "record_count": len(metric_records),
    }


def write_comparison_csv(
    ensemble_c3p_path: Path,
    o3_mini_c3p_path: Path,
    learned_pickle_path: Path,
    output_csv_path: Path,
) -> None:
    ensemble_c3p_metriccs = _load_c3p_trust(ensemble_c3p_path)
    o3_mini_c3p_metrics = _load_c3p_trust(o3_mini_c3p_path)

    learned_data = _load_learned_definitions(learned_pickle_path)

    not_learned_counter = 0
    not_learned_sucessfully = 0
    failed_to_validate = 0
    timeout_during_validation = 0
    learned_rows: dict[str, dict[str, dict[str, float | int]]] = {}
    for chebi_id, metrics in ensemble_c3p_metriccs.items():
        if chebi_id not in learned_data.learned_definitions:
            not_learned_counter += 1
            continue

        learned_def = learned_data.learned_definitions[chebi_id]

        if not learned_def.learn_success:
            not_learned_sucessfully += 1
            continue

        if learned_def.val_metrics is None:
            # Eg. [36835:3alphaHydroxySteroid:negative] processed 4975/35507
            # Validation pipeline is not completed with 48hrs time limit,
            failed_to_validate += 1
            continue

        if (
            learned_def.val_metrics.TP
            + learned_def.val_metrics.FN
            + learned_def.val_metrics.FP
            + learned_def.val_metrics.TN
        ) == 0:
            # ignore def for which all samples lead to timeouts during validation
            timeout_during_validation += 1
            continue

        recall = (
            learned_def.val_metrics.TP
            / (learned_def.val_metrics.TP + learned_def.val_metrics.FN)
            if (learned_def.val_metrics.TP + learned_def.val_metrics.FN) > 0
            else 0.0
        )

        o3_mini_metrics = o3_mini_c3p_metrics[chebi_id]
        learned_rows[str(chebi_id)] = {
            "ensemble_c3p_f1_score": metrics,
            "o3_mini_c3p_f1_score": o3_mini_metrics,
            "our_f1_score": {
                "num_true_positives": learned_def.val_metrics.TP,
                "num_true_negatives": learned_def.val_metrics.TN,
                "num_false_positives": learned_def.val_metrics.FP,
                "num_false_negatives": learned_def.val_metrics.FN,
                "f1": learned_def.val_metrics.F1,
                "precision": learned_def.val_metrics.PPV,
                "recall": recall,
            },
        }

    print("Total CHEBI ids in c3p:", len(ensemble_c3p_metriccs))
    print(
        "Total CHEBI ids in learned definitions:", len(learned_data.learned_definitions)
    )

    print("Not learned counter:", not_learned_counter)
    print("Not learned successfully:", not_learned_sucessfully)
    print("Failed to validate:", failed_to_validate)
    print("Timeout during validation:", timeout_during_validation)

    print("Finally number of selected chebi ids for comparison:", len(learned_rows))

    our_score_dict = calculate_micro_macro_metrics(
        [data["our_f1_score"] for data in learned_rows.values()]
    )
    print("Overall metrics for our learned definitions:")
    print(our_score_dict)
    ensemble_c3p_score_dict = calculate_micro_macro_metrics(
        [data["ensemble_c3p_f1_score"] for data in learned_rows.values()]
    )
    print("Overall metrics for ensemble c3p:")
    print(ensemble_c3p_score_dict)
    o3_mini_c3p_score_dict = calculate_micro_macro_metrics(
        [data["o3_mini_c3p_f1_score"] for data in learned_rows.values()]
    )
    print("Overall metrics for o3-mini c3p:")
    print(o3_mini_c3p_score_dict)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chebi_id",
                "ensemble_c3p_f1_score",
                "o3_mini_c3p_f1_score",
                "our_f1_score",
            ],
        )
        writer.writeheader()

        for chebi_id, data_dict in learned_rows.items():
            writer.writerow(
                {
                    "chebi_id": chebi_id,
                    "ensemble_c3p_f1_score": data_dict["ensemble_c3p_f1_score"],
                    "o3_mini_c3p_f1_score": data_dict["o3_mini_c3p_f1_score"],
                    "our_f1_score": data_dict["our_f1_score"],
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare c3p train/val metrics (validation F1 only) against learned definition scores and export CSV."
    )
    parser.add_argument(
        "--ensemble-c3p-json",
        type=Path,
        required=True,
        help="Path to json file of c3po containing val scores (See: https://github.com/chemkg/c3p/pull/23)",
    )
    parser.add_argument(
        "--o3-mini-c3p-json",
        type=Path,
        required=True,
        help="Path to json file of o3-mini containing val scores (See: https://github.com/chemkg/c3p/pull/23)",
    )

    claude_learned_fp = os.path.join(
        PROJECT_DIR,
        "inference",
        "learner",
        "learned",
        "claude-opus-4-6",
        "learned_definitions_a3_with_val.pkl",
    )
    parser.add_argument(
        "--learned-pickle",
        type=Path,
        required=True,
        # default=claude_learned_fp,
        help="Path to learned definitions pickle file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("comparison_with_c3p_ensemble_o3_mini.csv"),
        help="Path to write comparison CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    write_comparison_csv(
        ensemble_c3p_path=args.ensemble_c3p_json,
        o3_mini_c3p_path=args.o3_mini_c3p_json,
        learned_pickle_path=args.learned_pickle,
        output_csv_path=args.output_csv,
    )
    print(f"Wrote comparison rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
