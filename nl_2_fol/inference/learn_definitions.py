import os

from gavel.logic import logic
from gavel.logic.logic import QuantifiedFormula
from rdkit import Chem

from nl_2_fol.inference.chebi_data import ChEBIDataWrapper
from nl_2_fol.inference.custom_exceptions import (
    LearnOutOfBoxPredicateException,
    LowF1ScoreException,
    MissingPredicateException,
    RetryException,
)
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
        # ----- C3PO slim dataset loading -----
        (
            self._dataset,
            self.smiles_to_instance,
            self.validation_smiles,
            self.all_smiles,
        ) = load_c3po_slim_dataset(self.slim_dataset_path, self.structures_path)
        # ---------------------------------------

        self._attempts: int = 0
        self._prompts_history: list[str] = []

        # ------ Entire Chebi data loading -------
        entire_chebi_data = ChEBIDataWrapper(chebi_version=244)
        self._chebi_name_to_data_mapping = entire_chebi_data.get_name_to_data_mapping()

    def learn_fol_definitions(self):

        for chemical_class in self._dataset.classes:
            if chemical_class.definition is None:
                continue
            self._learn(chemical_class)

    def _learn(self, chemical_class: ChemicalClass) -> None:
        self._attempts = 0
        self._prompts_history = []

        # """CHEBI:16236 - ethanol: A primary alcohol that is ethane in which one
        # of the hydrogens is substituted by a hydroxy group."""
        input_text = (
            f"{chemical_class.id} - {chemical_class.name}: {chemical_class.definition}"
        )
        result, prompt_text = self.chebi_prompt_obj.invoke_llm_with_fs_prompt(
            input_text
        )
        raised_exception = None
        self._add_prompt_to_history(prompt_text, result)
        try:
            self._parse_and_validate_generated_definition(result, chemical_class)
            self._dataset.classes.remove(chemical_class)
            return
        except Exception as e:
            raised_exception = e
            if isinstance(e, MissingPredicateException):
                raised_exception = self._handle_missing_predicates_exception(e)

        print(
            f"Failed to parse FOL definition for {chemical_class.id}: {raised_exception}"
        )
        previous_fol_def = result.FOL_formula
        while self._attempts < self.max_attempts:
            print(f"Attempt {self._attempts + 1} for {chemical_class.id}")
            add_bck_def = None
            if isinstance(raised_exception, LearnOutOfBoxPredicateException):
                learned_predicates, prompt_to_learn_predicates = (
                    self.chebi_prompt_obj.invoke_llm_with_undef_failure_prompt(
                        input_text,
                        previous_fol_def,
                        raised_exception.predicates_to_learn,
                    )
                )
                prompt_text += "\n" + prompt_to_learn_predicates
                self._add_prompt_to_history(prompt_text, result)
                additional_def = learned_predicates.predicate_definitions
                try:
                    add_bck_def = self._gavel.convert_to_background_defintions(
                        additional_def
                    )
                except Exception as e:
                    # If additional generated definitions for the missing predicates
                    # are not parseable, we return the error to llm.
                    # This will lead generating new FOL formula input chemical class
                    # instead of trying to fix the additional missing predicates definitions
                    raised_exception = Exception(
                        f"Failed to parse FOL definition for the following predicate:"
                        f"{additional_def}. Error: {e}"
                    )
                    continue
            elif isinstance(raised_exception, RetryException):
                # Retries the result generated from previous attempt
                # This is because certain undefined predicated might be known by now
                pass
            else:
                result, prompt_text = (
                    self.chebi_prompt_obj.invoke_llm_with_err_failure_prompt(
                        input_text,
                        previous_fol_def,
                        str(raised_exception),
                    )
                )

                self._add_prompt_to_history(prompt_text, result)

            try:
                self._parse_and_validate_generated_definition(
                    result, chemical_class, background_definitions=add_bck_def
                )
                self._dataset.classes.remove(chemical_class)
                if add_bck_def:
                    self._gavel.merge_to_background_definitions(add_bck_def)
                return
            except Exception as e:
                raised_exception = e
                if isinstance(e, MissingPredicateException):
                    raised_exception = self._handle_missing_predicates_exception(e)
                print(
                    f"Failed to parse FOL definition for {chemical_class.id}: {raised_exception}"
                )
                self._attempts += 1
                previous_fol_def = result.FOL_formula

    def _handle_missing_predicates_exception(
        self, e: MissingPredicateException
    ) -> Exception:
        chemical_class_predicates = {
            predicate.lower().strip()
            for predicate in e.missing_predicates
            if predicate.lower().strip() in self._dataset.classes
        }
        for predicate in chemical_class_predicates:
            # NOTE: Recursively learn definition for the predicate if it's
            # in c3po slim dataset
            self._learn(self._dataset.get_chemical_class_by_name(predicate))

        raised_exception = RetryException()
        other_predicates = e.missing_predicates - chemical_class_predicates
        if other_predicates:
            predicates_to_learn: dict[str, str | None] = {}
            for predicate in other_predicates:
                if predicate in self._chebi_name_to_data_mapping:
                    chebi_data = self._chebi_name_to_data_mapping[predicate]
                    predicates_to_learn[predicate] = chebi_data["definition"]
                else:
                    predicates_to_learn[predicate] = None

            raised_exception = LearnOutOfBoxPredicateException(
                predicates_to_learn=predicates_to_learn
            )
        return raised_exception

    def _parse_and_validate_generated_definition(
        self,
        result: CHEBIFOLOutput,
        chemical_class: ChemicalClass,
        background_definitions: dict[
            str, tuple[list[logic.Variable], logic.QuantifiedFormula]
        ]
        | None = None,
    ) -> None:
        """
        Parses the generated FOL definition and validates it against the positive
        and negative samples of the chemical class.

        Raises an exception if parsing or validation fails, otherwise returns None.
        """

        tptp_def = self._gavel.get_tptp_fol_definition(result.FOL_formula)

        pos_samples, neg_samples = self._get_positive_and_negative_samples(
            chemical_class
        )

        unmatched_pos_samples, matched_neg_samples = (
            self._check_if_definition_matches_samples(
                tptp_def,
                pos_samples,
                neg_samples,
                background_definitions,
            )
        )
        metrics = self._get_metrics(
            unmatched_pos_samples, matched_neg_samples, pos_samples, neg_samples
        )

        # Validate against the threshold
        # TODO:  adjust threshold wrt how many def meet it
        if metrics.F1 < self.f1_threshold:
            raise LowF1ScoreException(
                list(pos_samples),
                list(neg_samples),
                list(matched_neg_samples),
                list(unmatched_pos_samples),
                max_examples=10,
                chebi_id_to_data_mapping=self._chebi_name_to_data_mapping,
            )

        self.definitions[chemical_class.id] = LearnedDefinition(
            metrics=metrics,
            learned_FOL=tptp_def,
            name=chemical_class.name,
            definition=chemical_class.definition if chemical_class.definition else "",
            prompts_history=self._prompts_history,
        )
        self._gavel.update_background_definition(chemical_class.name, tptp_def)
        print(
            f"Learned definition for {chemical_class.id} with F1 score: {metrics.F1:.2f}"
        )

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
        background_definitions: dict[
            str, tuple[list[logic.Variable], logic.QuantifiedFormula]
        ]
        | None = None,
    ) -> tuple[set[SMILES_STRING], set[SMILES_STRING]]:

        def is_matched(smiles: SMILES_STRING) -> bool:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            matches = self._gavel.does_mol_match_tptp_definition(
                mol,
                tptp_def,
                additional_background_definitions=background_definitions,
            )
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

    def _add_prompt_to_history(self, prompt: str, result: CHEBIFOLOutput) -> None:
        history_entry: str = (
            f"Prompt:\n{prompt}\n"
            f"{self.chebi_prompt_obj.model_name}(LLM) output:\n{result.FOL_formula}\n"
        )
        self._prompts_history.append(history_entry)
