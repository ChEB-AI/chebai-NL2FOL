from gavel.logic.logic import QuantifiedFormula
from nl_2_fol.inference.model_check_molecule import molecule_matches_definition
from rdkit import Chem


# TODO: rectify
class NL2FOLChebiClassifier:
    def __init__(self, definitions_path: str):
        self.definitions_path = definitions_path
        self.definitions_to_match: list[dict[str, QuantifiedFormula]] = []
        self.background_definitions: dict[str, tuple[list, QuantifiedFormula]] = {}

    def classify_smiles(self, smiles: str) -> dict[str, list]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("")
        except Exception:
            return {"error": ["Invalid SMILES string"]}

        classification = []

        # definitions_to_match = [{<name>: <formula>}, ...]
        for def_dict in self.definitions_to_match:
            definition_name, formula = list(def_dict.items())[0]
            if molecule_matches_definition(mol, formula, self.background_definitions):
                classification.append({"name": definition_name})

        return {smiles: classification}

    def classify_smiles_list(self, smiles_list: list[str]) -> list[dict]:
        return [self.classify_smiles(smiles) for smiles in smiles_list]
