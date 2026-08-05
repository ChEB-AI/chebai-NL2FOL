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
                fol_definition = learned_def.learned_FOL.definition
                print(f"CHEBI:{chebi_id} - {learned_def.name}")
                print(
                    f"Pred variables: {[str(var) for var in fol_definition.variables]}"
                )
                print("-" * 40)
                print(f"Formula: {fol_definition}")
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
                fol_definition = learned_def.learned_FOL.definition
                print(f"CHEBI:{chebi_id} - {learned_def.name}")
                print(
                    f"Pred variables: {[str(var) for var in fol_definition.variables]}"
                )
                print("-" * 40)
                print(f"Formula: {fol_definition}")
                print(f"Train Metrics: {learned_def.train_metrics}")
                if learned_def.val_metrics is not None:
                    print(f"Validation Metrics: {learned_def.val_metrics}")
                print("=" * 80)

    for name, add_def in data.additional_definitions.items():
        if add_def.learn_success:
            add_fol_definition = add_def.fol_formula.definition
            print(f"Additional definition for predicate: {name}")
            print(
                f"Pred variables: {[str(var) for var in add_fol_definition.variables]}"
            )
            print(f"Formula: {add_fol_definition}")
            print("---" * 10)

    print(f"Total main formulas printed for split '{split}': {main_formulas_counter}")


def check_number_of_classes_to_validate(pickle_file_path: Path):
    with open(pickle_file_path, "rb") as f:
        data: DefinitionLearningResults = pickle.load(f)

    classes = []
    with open("classes_4.txt", "r") as f:
        classes = [line.strip() for line in f]

    classes_to_validate = [
        learned_def.name
        for _, learned_def in data.learned_definitions.items()
        if learned_def.learn_success
        and learned_def.val_metrics is None
        and learned_def.name in classes
    ]

    [print(f"{class_name}") for class_name in classes_to_validate]
    print(f"Classes to validate: {len(classes_to_validate)}")


if __name__ == "__main__":
    pickle_file_path = Path(
        "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3_with_val.pkl"
    )
    check_number_of_classes_to_validate(pickle_file_path=pickle_file_path)
    # print_all_fols(pickle_file_path=pickle_file_path, split="val")
    # For Val: Total main formulas printed for split 'val': 198
