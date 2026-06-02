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
        chebi_str = entry.get("chebi_id")
        if not chebi_str:
            continue
        # chebi_id may be like "CHEBI:84948" — extract numeric part
        try:
            chebi_num = int(str(chebi_str).split(":")[-1])
        except Exception:
            # skip malformed ids
            raise ValueError(f"Malformed chebi_id: {chebi_str}")

        val_block = entry.get("val", {}) or {}
        # validation F1 is stored as `f1` (lowercase) in this file
        f1 = float(val_block.get("f1", 0.0))
        parsed[chebi_num] = {"F1": f1}

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

    learned_rows: dict[str, dict[str, float | str]] = {}
    for chebi_id, metrics in c3p_metrics.items():
        if chebi_id not in learned_data.learned_definitions:
            val_f1 = "not_learned"
        else:
            learned_def = learned_data.learned_definitions[chebi_id]

            if not learned_def.learn_success:
                val_f1 = "failed_to_learn"
            else:
                if learned_def.val_metrics is None:
                    # Eg. [36835:3alphaHydroxySteroid:negative] processed 4975/35507
                    # Validation pipeline is not completed with 48hrs time limit,
                    val_f1 = "failed_to_validate"
                else:
                    val_f1 = float(learned_def.val_metrics.F1)
        learned_rows[str(chebi_id)] = {
            "c3p_f1_score": metrics["F1"],
            "our_f1_score": val_f1,
        }

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chebi_id",
                "c3p_f1_score",
                "our_f1_score",
            ],
        )
        writer.writeheader()

        for chebi_id, data_dict in learned_rows.items():
            writer.writerow(
                {
                    "chebi_id": chebi_id,
                    "c3p_f1_score": data_dict["c3p_f1_score"],
                    "our_f1_score": data_dict["our_f1_score"],
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare c3p train/val metrics (validation F1 only) against learned definition scores and export CSV."
    )
    parser.add_argument(
        "--c3p-json",
        type=Path,
        default=Path("data") / "c3p_train_val_scores.json",
        help="Path to c3p_train_val_scores.json file (will use validation f1).",
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
        default=claude_learned_fp,
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
    write_comparison_csv(
        c3p_path=args.c3p_json,
        learned_pickle_path=args.learned_pickle,
        output_csv_path=args.output_csv,
    )
    print(f"Wrote comparison rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
