class NL2FOLClassifier:
    def __init__(self, definitions_path: str):
        self.definitions_path = definitions_path
        # TODO: cache mechanism needs to be included here or can be
        # handled by chebifier (as it does currently)

    def classify_smiles(self, smiles: str) -> dict:
        return {}

    def classify_smiles_list(self, smiles_list: list[str]) -> list[dict]:
        return [self.classify_smiles(smiles) for smiles in smiles_list]
