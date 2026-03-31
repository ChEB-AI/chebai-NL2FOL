import argparse
import csv
import json
import pickle
from pathlib import Path

from nl_2_fol.inference.learner.definition_model import DefinitionLearningResults


def _load_c3p_trust(c3p_path: Path) -> dict[int, dict[str, float]]:
    with open(c3p_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    def _safe_div(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    parsed: dict[int, dict[str, float]] = {}
    for chebi_id, metrics in raw.items():
        tp = float(metrics["TP"])
        fp = float(metrics["FP"])
        fn = float(metrics["FN"])
        tn = float(metrics["TN"])
        ppv = float(metrics["PPV"])
        npv = float(metrics["NPV"])
        f1 = _f1_from_counts(tp=tp, fp=fp, fn=fn)
        parsed[int(chebi_id)] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "PPV": ppv,
            "NPV": npv,
            "F1": f1,
        }

    return parsed


def _load_learned_definitions(
    learned_pickle_path: Path,
) -> DefinitionLearningResults:
    with open(learned_pickle_path, "rb") as f:
        return pickle.load(f)


def write_comparison_csv(
    c3p_path: Path,
    learned_pickle_path: Path,
    output_csv_path: Path,
) -> None:
    c3p_metrics = _load_c3p_trust(c3p_path)
    learned_data = _load_learned_definitions(learned_pickle_path)

    learned_rows: dict[str, dict[str, float | bool]] = {}
    for chebi_id, metrics in c3p_metrics.items():
        learned_def = learned_data.learned_definitions[chebi_id]
        assert learned_def.val_metrics is not None
        learned_rows[str(chebi_id)] = {
            "c3p_f1_score": metrics["F1"],
            "model_train_f1": float(learned_def.train_metrics.F1),
            "model_val_f1": float(learned_def.val_metrics.F1),
        }

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chebi_id",
                "c3p_f1_score",
                "model_train_f1",
                "model_val_f1",
            ],
        )
        writer.writeheader()

        for chebi_id, data_dict in learned_rows.items():
            writer.writerow(
                {
                    "chebi_id": chebi_id,
                    "c3p_f1_score": data_dict["c3p_f1_score"],
                    "model_train_f1": data_dict["model_train_f1"],
                    "model_val_f1": data_dict["model_val_f1"],
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare c3p_trust metrics against learned definition scores and export CSV."
    )
    parser.add_argument(
        "--c3p-json",
        type=Path,
        required=True,
        help="Path to c3p_trust.json file.",
    )
    parser.add_argument(
        "--learned-pickle",
        type=Path,
        required=True,
        help="Path to learned definitions pickle file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("comparison_with_c3p.csv"),
        help="Path to write comparison CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    row_count = write_comparison_csv(
        c3p_path=args.c3p_json,
        learned_pickle_path=args.learned_pickle,
        output_csv_path=args.output_csv,
    )
    print(f"Wrote {row_count} comparison rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
