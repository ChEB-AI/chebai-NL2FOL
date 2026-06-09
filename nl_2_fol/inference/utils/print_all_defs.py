import pickle
from pathlib import Path

from nl_2_fol.inference.learner.definition_model import (
    DefinitionLearningResults,
)


def print_all_fols(pickle_file_path: Path, split="train"):
    with open(pickle_file_path, "rb") as f:
        data: DefinitionLearningResults = pickle.load(f)
    main_formulas_counter = 0
    if split == "train":
        for chebi_id, learned_def in data.learned_definitions.items():
            if learned_def.learn_success:
                main_formulas_counter += 1
                print(f"CHEBI:{chebi_id} - {learned_def.name}")
                print(
                    f"Pred variables: {[str(var) for var in learned_def.learned_FOL.pred_variables]}"
                )
                print("-" * 40)
                print(f"Formula: {learned_def.learned_FOL.formula}")
                print(f"Train Metrics: {learned_def.train_metrics}")
                if learned_def.val_metrics is not None:
                    print(f"Validation Metrics: {learned_def.val_metrics}")
                print("=" * 80)
    if split == "val":
        for chebi_id, learned_def in data.learned_definitions.items():
            if (
                learned_def.learn_success
                and learned_def.val_metrics is not None
                and (
                    # Filter out formulas for which all samples had timeouts
                    (
                        learned_def.val_metrics.TP
                        + learned_def.val_metrics.FN
                        + learned_def.val_metrics.FP
                        + learned_def.val_metrics.TN
                    )
                    != 0
                )
            ):
                main_formulas_counter += 1
                print(f"CHEBI:{chebi_id} - {learned_def.name}")
                print(
                    f"Pred variables: {[str(var) for var in learned_def.learned_FOL.pred_variables]}"
                )
                print("-" * 40)
                print(f"Formula: {learned_def.learned_FOL.formula}")
                print(f"Train Metrics: {learned_def.train_metrics}")
                if learned_def.val_metrics is not None:
                    print(f"Validation Metrics: {learned_def.val_metrics}")
                print("=" * 80)

    for name, add_def in data.additional_definitions.items():
        if add_def.learn_success:
            print(f"Additional definition for predicate: {name}")
            print(
                f"Pred variables: {[str(var) for var in add_def.fol_formula.pred_variables]}"
            )
            print(f"Formula: {add_def.fol_formula.formula}")
            print("---" * 10)

    print(f"Total main formulas printed for split '{split}': {main_formulas_counter}")


if __name__ == "__main__":
    pickle_file_path = Path(
        "nl_2_fol/inference/learner/learned/claude-opus-4-6/v1_learned_definitions_a3_with_val.pkl"
    )
    print_all_fols(pickle_file_path=pickle_file_path, split="val")
    # For Val: Total main formulas printed for split 'val': 198
