import os

from rdkit import Chem

from nl_2_fol.inference.data_model import Dataset
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
        dataset_path: str,  # https://huggingface.co/datasets/MonarchInit/C3PO
        chebi_prompt_obj: ChebiPrompt,
        max_attempts: int = 4,
        f1_threshold: float = 0.8,
        definitions_path: str | None = None,
    ):
        self.dataset_path = dataset_path
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

        self._gavel_fol_reasoner = GavelFOLReasoner()

    def learn_fol_definitions(
        self,
    ):
        dataset: Dataset = self._load_dataset()
        s2i = dataset.smiles_to_instance()
        all_validation = (
            set(dataset.validation_examples)
            if dataset.validation_examples is not None
            else set()
        )
        all_smiles = dataset.all_smiles()

        def get_positive_and_negative_samples(chemical_class):
            all_positive = set(chemical_class.all_positive_examples)
            positive_examples = list(all_positive - all_validation)
            positive_instances = [s2i[smiles] for smiles in positive_examples]
            negative_examples = list((all_smiles - all_positive) - all_validation)
            negative_instances = [s2i[smiles] for smiles in negative_examples]
            return positive_instances, negative_instances

        for chemical_class in dataset.classes:
            if chemical_class.definition is None:
                continue

            # """CHEBI:16236 - ethanol: A primary alcohol that is ethane in which one
            # of the hydrogens is substituted by a hydroxy group."""
            result: CHEBIFOLOutput = self.chebi_prompt_obj.invoke_llm_with_fs_prompt(
                f"{chemical_class.id} - {chemical_class.name}: {chemical_class.definition}"
            )
            try:
                tptp_def = self._gavel_fol_reasoner._get_tptp_fol_definition(
                    result.FOL_formula
                )
                # SMILES OF chemical class is not available in the dataset
                # TODO: Check whether we really need this to do the following
                # mol = Chem.MolFromSmiles(chemical_class.smiles)
                # if mol is None: continue
                # matches = self._gavel_fol_reasoner._molecule_matches_tptp_fol_definition(
                #     mol, tptp_def, {}
                # )
            except Exception as e:
                pass

            pos_samples, neg_samples = get_positive_and_negative_samples(chemical_class)

            unmatched_pos_samples, matched_neg_samples = (
                self._check_if_definition_matches_samples(
                    tptp_def,
                    pos_samples,
                    neg_samples,
                )
            )
            metrics = self._get_metrics(
                unmatched_pos_samples, matched_neg_samples, pos_samples, neg_samples
            )
            if metrics.F1 < self.f1_threshold:
                pass

            self.definitions[chemical_class.id] = LearnedDefinition(
                metrics=metrics, definition=tptp_def
            )

            print(
                f"Learned definition for {chemical_class.id} with F1 score: {metrics.F1:.2f}"
            )

    def _check_if_definition_matches_samples(
        self, tptp_def, pos_samples: list, neg_samples: list
    ) -> tuple[list[str], list[str]]:

        def is_matched(smiles: str) -> bool:
            try:
                mol = Chem.MolFromSmiles(smiles)
            except Exception:
                return False
            if mol is None:
                return False
            matches = self._gavel_fol_reasoner._molecule_matches_tptp_fol_definition(
                mol, tptp_def, {}
            )
            return bool(matches)

        unmatched_pos_samples = []
        for smiles in pos_samples:
            matches = is_matched(smiles)
            if not matches:
                unmatched_pos_samples.append(smiles)

        matched_neg_samples = []
        for smiles in neg_samples:
            matches = is_matched(smiles)
            if matches:
                matched_neg_samples.append(smiles)

        return unmatched_pos_samples, matched_neg_samples

    def _get_metrics(
        self, unmatched_pos_samples, matched_neg_samples, pos_samples, neg_samples
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

    def _load_dataset(self) -> Dataset:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset = Dataset.model_validate_json(f.read())
            print(
                f"Classes: {len(dataset.classes)} Instances: {len(dataset.structures)}"
            )
        return dataset

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
