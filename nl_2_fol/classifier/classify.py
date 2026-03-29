import os
import pickle

from rdkit import Chem

from nl_2_fol.inference.fol_reasoner.model_check_molecule import GavelFOLReasoner
from nl_2_fol.inference.learner import definition_model as def_model
from nl_2_fol.inference.preprocessing import CHEBI_ID


class NL2FOLChebiClassifier:
    def __init__(self, definitions_path: str):
        self.definitions_path = definitions_path
        self._gavel = GavelFOLReasoner()
        self.class_definitions = self._load_definitions(definitions_path)

    def classify_smiles(self, smiles: str) -> dict[str, list]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("")
        except Exception:
            return None  # TODO

        classification = []

        for chebi_id, learned_def in self.class_definitions.items():
            if self._gavel.does_mol_match_tptp_definition(
                mol, learned_def.learned_FOL.formula
            ):
                classification.append({"chebi_id": chebi_id, "name": learned_def.name})

        return {smiles: classification}

    def classify_smiles_list(self, smiles_list: list[str]) -> list[dict]:
        return [self.classify_smiles(smiles) for smiles in smiles_list]

    def _load_definitions(
        self, definitions_file_path: str
    ) -> dict[CHEBI_ID, def_model.LearnedDefinition]:
        print(f"Loading definitions from {definitions_file_path} if it exists...")
        if os.path.exists(definitions_file_path):
            with open(definitions_file_path, "rb") as f:
                definitions = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"No definitions file found at {definitions_file_path}. Please ensure the file exists."
            )
        return self._load_background_defs_from_pmodel(definitions)

    def _load_background_defs_from_pmodel(
        self, new_definitions: def_model.DefinitionLearningResults
    ) -> dict[CHEBI_ID, def_model.LearnedDefinition]:
        """Load back the state from learned definitions."""
        counter = 0
        successful_learned_definitions: dict[CHEBI_ID, def_model.LearnedDefinition] = {}
        for chebi_id, learned_def in new_definitions.learned_definitions.items():
            if learned_def.learn_success:
                self._gavel.add_background_definition(
                    learned_def.name,
                    learned_def.learned_FOL.pred_variables,
                    learned_def.learned_FOL.formula,
                )
                successful_learned_definitions[chebi_id] = learned_def
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

        return successful_learned_definitions
