import pickle
from pathlib import Path

from nl_2_fol.inference.learner.definition_model import DefinitionLearningResults


def print_pickle_contents(pickle_file_path):
    """Print the contents of a pickle file containing DefinitionLearningResults."""

    with open(pickle_file_path, "rb") as f:
        data: DefinitionLearningResults = pickle.load(f)

    for _, learned_def in data.learned_definitions.items():
        print(f"Learned definition for predicate: {learned_def.name}")
        print(f"Pred variables: {learned_def.learned_FOL.pred_variables}")
        print(f"Formula: {learned_def.learned_FOL.formula}")
        print("---" * 10)

    for name, add_def in data.additional_definitions.items():
        print(f"Additional definition for predicate: {name}")
        print(f"Pred variables: {add_def.pred_variables}")
        print(f"Formula: {add_def.formula}")
        print("---" * 10)


if __name__ == "__main__":
    # Replace with your pickle file path
    pickle_file = Path(
        "nl_2_fol/inference/learned/claude-opus-4-6/learned_definitions.pkl"
    )
    print_pickle_contents(pickle_file)
