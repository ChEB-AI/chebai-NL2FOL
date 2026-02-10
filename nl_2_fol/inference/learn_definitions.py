import os

from chemlog.preprocessing.chebi_data import ChEBIData
from gavel.logic.logic import QuantifiedFormula
from rdkit import Chem

from nl_2_fol.inference.custom_exceptions import LowF1ScoreException
from nl_2_fol.inference.data_model import (
    SMILES_STRING,
    ChemicalClass,
    ChemicalStructure,
    load_c3po_slim_dataset,
)
from nl_2_fol.inference.definition_model import (
    DefinitionLearningResults,
    DefinitionMetrics,
    LearnedDefinition,
)
from nl_2_fol.inference.model_check_molecule import GavelFOLReasoner
from nl_2_fol.prompting.chebai_prompt import ChebiPrompt
from nl_2_fol.prompting.models import CHEBIFOLOutput


# TODO: can langchain-graph be used here? or will it be an overkill?
class LearnDefinitions:
    _DEFINITION_JSON_FILE_NAME = "learned_definitions.json"

    def __init__(
        self,
        chebi_prompt_obj: ChebiPrompt,
        slim_dataset_path: str = "data/classes_slim.csv",  # https://huggingface.co/datasets/MonarchInit/C3PO/blob/main/slim_dataset.csv
        structures_path: str = "data/structures.csv",  # https://huggingface.co/datasets/MonarchInit/C3PO/blob/main/structures.csv
        max_attempts: int = 4,
        f1_threshold: float = 0.8,
        definitions_path: str | None = None,
    ):
        self.slim_dataset_path = slim_dataset_path
        self.structures_path = structures_path
        self.definitions_path = definitions_path
        self.chebi_prompt_obj = chebi_prompt_obj
        self.max_attempts = max_attempts
        self.f1_threshold = f1_threshold
        # load definitions from the path and store them in a suitable data structure
        # this will be used to learn new definitions based on the classified chemical classes
        self.definitions: DefinitionLearningResults = self._load_definitions(
            definitions_path
        )
        self._default_def_save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "learned",
            self.chebi_prompt_obj.model_name,
        )

        self._gavel = GavelFOLReasoner()
        (
            self._dataset,
            self.smiles_to_instance,
            self.validation_smiles,
            self.all_smiles,
        ) = load_c3po_slim_dataset(self.slim_dataset_path, self.structures_path)

        self._attempts: int = 0

    def learn_fol_definitions(self):

        for chemical_class in self._dataset.classes:
            self._attempts = 0
            if chemical_class.definition is None:
                continue
            self._learn(chemical_class)

    def _learn(self, chemical_class: ChemicalClass) -> None:

        # """CHEBI:16236 - ethanol: A primary alcohol that is ethane in which one
        # of the hydrogens is substituted by a hydroxy group."""
        input_text = (
            f"{chemical_class.id} - {chemical_class.name}: {chemical_class.definition}"
        )
        result: CHEBIFOLOutput = self.chebi_prompt_obj.invoke_llm_with_fs_prompt(
            input_text
        )
        output = self._parse_and_validate_generated_definition(result, chemical_class)
        if not isinstance(output, Exception):
            return None

        print(f"Failed to parse FOL definition for {chemical_class.id}: {output}")
        previous_fol_def = result.FOL_formula
        raised_exception = output
        while self._attempts < self.max_attempts:
            print(f"Attempt {self._attempts + 1} for {chemical_class.id}")
            result: CHEBIFOLOutput = (
                self.chebi_prompt_obj.invoke_llm_with_failure_prompt(
                    input_text,
                    previous_fol_def,
                    str(raised_exception),
                )
            )
            output = self._parse_and_validate_generated_definition(
                result, chemical_class
            )
            if not isinstance(output, Exception):
                return None
            print(f"Failed to parse FOL definition for {chemical_class.id}: {output}")
            self._attempts += 1
            previous_fol_def = result.FOL_formula
            raised_exception = output

    def _parse_and_validate_generated_definition(
        self, result: CHEBIFOLOutput, chemical_class: ChemicalClass
    ) -> bool | Exception:

        output = self._gavel.get_tptp_fol_definition(result.FOL_formula)

        pos_samples, neg_samples = self._get_positive_and_negative_samples(
            chemical_class
        )

        if isinstance(output, Exception):
            return output

        unmatched_pos_samples, matched_neg_samples = (
            self._check_if_definition_matches_samples(
                output,
                pos_samples,
                neg_samples,
            )
        )
        metrics = self._get_metrics(
            unmatched_pos_samples, matched_neg_samples, pos_samples, neg_samples
        )

        # Validate against the threshold
        # TODO:  adjust threshold wrt how many def meet it
        if metrics.F1 < self.f1_threshold:
            # TODO: check if mol definition is present
            raise LowF1ScoreException(
                list(pos_samples),
                list(neg_samples),
                list(matched_neg_samples),
                list(unmatched_pos_samples),
                max_examples=10,
            )

        self.definitions[chemical_class.id] = LearnedDefinition(
            metrics=metrics, definition=output
        )
        print(
            f"Learned definition for {chemical_class.id} with F1 score: {metrics.F1:.2f}"
        )
        return True

    def _get_positive_and_negative_samples(
        self, chemical_class: ChemicalClass
    ) -> tuple[set[ChemicalStructure], set[ChemicalStructure]]:
        all_positive = set(chemical_class.all_positive_examples)
        positive_examples = list(all_positive - self.validation_smiles)
        positive_instances = {
            self.smiles_to_instance[smiles] for smiles in positive_examples
        }
        negative_examples = list(
            (self.all_smiles - all_positive) - self.validation_smiles
        )
        negative_instances = {
            self.smiles_to_instance[smiles] for smiles in negative_examples
        }
        return positive_instances, negative_instances

    def _check_if_definition_matches_samples(
        self,
        tptp_def: QuantifiedFormula,
        pos_samples: set[ChemicalStructure],
        neg_samples: set[ChemicalStructure],
    ) -> tuple[set[SMILES_STRING], set[SMILES_STRING]]:

        def is_matched(smiles: SMILES_STRING) -> bool:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False
                matches = self._gavel.does_mol_match_tptp_definition(
                    mol,
                    tptp_def,
                )
            except Exception:
                return False
            return bool(matches)

        unmatched_pos_samples = set()
        for chemical in pos_samples:
            matches = is_matched(chemical.smiles)
            if not matches:
                unmatched_pos_samples.add(chemical.smiles)

        matched_neg_samples = set()
        for chemical in neg_samples:
            matches = is_matched(chemical.smiles)
            if matches:
                matched_neg_samples.add(chemical.smiles)

        return unmatched_pos_samples, matched_neg_samples

    def _get_metrics(
        self,
        unmatched_pos_samples: set[SMILES_STRING],
        matched_neg_samples: set[SMILES_STRING],
        pos_samples: set[ChemicalStructure],
        neg_samples: set[ChemicalStructure],
    ) -> DefinitionMetrics:

        num_true_positives = len(pos_samples) - len(unmatched_pos_samples)
        num_false_negatives = len(unmatched_pos_samples)
        num_false_positives = len(matched_neg_samples)
        num_true_negatives = len(neg_samples) - len(matched_neg_samples)
        f1 = (
            2
            * num_true_positives
            / (2 * num_true_positives + num_false_positives + num_false_negatives)
        )
        ppv = num_true_positives / (num_true_positives + num_false_positives)
        npv = num_true_negatives / (num_true_negatives + num_false_negatives)
        return DefinitionMetrics(
            F1=f1,
            PPV=ppv,
            NPV=npv,
            TP=num_true_positives,
            FP=num_false_positives,
            FN=num_false_negatives,
            TN=num_true_negatives,
        )

    def _load_definitions(self, path: str | None) -> DefinitionLearningResults:
        # load definitions from the given path and return as a dictionary
        # the key can be the chemical class and the value can be the FOL definition
        if path is not None:
            with open(path, "r") as f:
                definitions = DefinitionLearningResults.model_validate_json(f.read())
        elif os.path.exists(
            default_path := os.path.join(
                self._default_def_save_path, self._DEFINITION_JSON_FILE_NAME
            )
        ):
            with open(default_path, "r") as f:
                definitions = DefinitionLearningResults.model_validate_json(f.read())
        else:
            definitions = DefinitionLearningResults(root={})
        return definitions

    def _save_definitions(self, path: str | None) -> None:
        # save the learned definitions to the given path
        if path is None:
            path = self._default_def_save_path
            os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, self._DEFINITION_JSON_FILE_NAME), "w") as f:
            f.write(self.definitions.model_dump_json(indent=2))

        with open(os.path.join(path, "__metadata__.txt"), "w") as f:
            f.write(str(self.chebi_prompt_obj))


if __name__ == "__main__":
    chebi_data = ChEBIData(chebi_version=247)
