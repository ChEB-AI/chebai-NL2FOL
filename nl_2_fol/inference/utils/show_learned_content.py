import pickle
from pathlib import Path

from nl_2_fol.inference.learner.definition_model import DefinitionLearningResults


def print_pickle_contents(pickle_file_path, class_name="all"):
    """Print the contents of a pickle file containing DefinitionLearningResults."""

    with open(pickle_file_path, "rb") as f:
        data: DefinitionLearningResults = pickle.load(f)

    print(f"Number of learned definitions: {len(data.learned_definitions)}")
    print(f"Number of additional definitions: {len(data.additional_definitions)}")

    for _, learned_def in data.learned_definitions.items():
        if class_name != "all" and learned_def.name != class_name:
            continue

        print(f"Learned definition for predicate: {learned_def.name}")
        print(f"Pred variables: {learned_def.learned_FOL.pred_variables}")
        print(f"Metrics: {learned_def.train_metrics}")
        print(f"Formula: {learned_def.learned_FOL.formula}")
        print(f"Learned success: {learned_def.learn_success}")
        [print(his) for his in learned_def.prompts_history]
        print("---" * 10)

    for name, add_def in data.additional_definitions.items():
        if class_name != "all" and name != class_name:
            continue
        print(f"Additional definition for predicate: {name}")
        print(f"Pred variables: {add_def.fol_formula.pred_variables}")
        print(f"Formula: {add_def.fol_formula.formula}")
        print(f"Learned success: {add_def.learn_success}")
        print("---" * 10)


if __name__ == "__main__":
    # Replace with your pickle file path
    pickle_file = Path(
        "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_new.pkl"
    )
    print_pickle_contents(pickle_file, class_name="all")
