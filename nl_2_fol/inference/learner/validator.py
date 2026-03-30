import os
import pickle

from nl_2_fol.inference.fol_reasoner.model_check_molecule import GavelFOLReasoner
from nl_2_fol.inference.learner import definition_model as def_model
from nl_2_fol.inference.learner.base import BaseFOL


class PerformValidation(BaseFOL):
    def __init__(
        self,
        defs_file_path: str,
        slim_dataset_path: str,
        structures_path: str,
        chebi_version: int = 244,
    ):
        super().__init__(
            slim_dataset_path=slim_dataset_path,
            structures_path=structures_path,
            chebi_version=chebi_version,
            split="val",
        )
        self.defs_file_path = defs_file_path
        self._gavel = GavelFOLReasoner()
        self._loaded_defs = self._load_definitions(defs_file_path)

    def validate(self):
        for _, learned_def in self._loaded_defs.learned_definitions.items():
            if learned_def.learn_success and learned_def.val_metrics is None:
                self.validate_class(learned_def.name)

    def validate_class(self, class_name: str):
        print("-" * 10, f"Validating definition for {class_name}...", "-" * 10)
        chemical_class = self._c3po_slim_data.get_chemical_class_by_name(class_name)
        if chemical_class.id not in self._loaded_defs.learned_definitions:
            print(
                f"No learned definition found for class {class_name} with id "
                f"{chemical_class.id}. Skipping validation for this class."
            )
            return

        learned_def = self._loaded_defs.learned_definitions[chemical_class.id]
        if not learned_def.learn_success:
            print(
                f"Learned definition for class {class_name} with id {chemical_class.id} "
                f"was not successful learned during learning process. "
                f"Skipping validation for this class."
            )
            return

        val_metrics = self._score_definition(
            chemical_class=chemical_class,
            tptp_def=learned_def.learned_FOL.formula,
            sample_match_timeout_seconds=self._SAMPLE_MATCH_TIMEOUT_SECONDS,
            max_neg_samples=self._MAX_NEGATIVE_SAMPLES,
            temp_additional_defs=None,
        )[0]
        self._loaded_defs.learned_definitions[
            chemical_class.id
        ].val_metrics = val_metrics
        self._save_validated_definitions()

    def _save_validated_definitions(self):
        base_path, extension = os.path.splitext(self.defs_file_path)
        output_path = (
            f"{base_path}_with_val{extension}"
            if extension
            else f"{self.defs_file_path}_with_val"
        )
        with open(output_path, "wb") as f:
            pickle.dump(self._loaded_defs, f)

    def _load_definitions(
        self, definitions_file_path: str
    ) -> def_model.DefinitionLearningResults:
        print(f"Loading definitions from {definitions_file_path} if it exists...")
        if os.path.exists(definitions_file_path):
            with open(definitions_file_path, "rb") as f:
                definitions = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"No definitions file found at {definitions_file_path}. Please ensure the file exists."
            )
        return self._load_learned_definitions(definitions)

    def _load_learned_definitions(
        self, new_definitions: def_model.DefinitionLearningResults
    ) -> def_model.DefinitionLearningResults:
        """Load back the state from from learned definitions."""

        counter = 0
        for _, learned_def in new_definitions.learned_definitions.items():
            if learned_def.learn_success:
                self._gavel.add_background_definition(
                    learned_def.name,
                    learned_def.learned_FOL.pred_variables,
                    learned_def.learned_FOL.formula,
                )
                counter += 1

        print(f"Loaded {counter} definitions")

        counter = 0
        for name, add_def in new_definitions.additional_definitions.items():
            if add_def.learn_success:
                self._gavel.add_background_definition(
                    name,
                    add_def.fol_formula.pred_variables,
                    add_def.fol_formula.formula,
                )
                counter += 1
        print(f"Loaded {counter} additional definitions")

        return new_definitions
