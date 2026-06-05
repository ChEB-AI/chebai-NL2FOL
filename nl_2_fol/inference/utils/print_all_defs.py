import pickle
from pathlib import Path

from nl_2_fol.inference.learner.definition_model import (
    DefinitionLearningResults,
)

pickle_file_path = Path(
    "nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions_a3_with_val.pkl"
)
with open(pickle_file_path, "rb") as f:
    data: DefinitionLearningResults = pickle.load(f)

for chebi_id, learned_def in data.learned_definitions.items():
    if learned_def.learn_success:
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
