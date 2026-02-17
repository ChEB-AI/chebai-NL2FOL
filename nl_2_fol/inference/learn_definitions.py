import os
import pickle

import tqdm
from gavel.logic import logic
from gavel.logic.logic import QuantifiedFormula

from nl_2_fol.inference import custom_exceptions as ce
from nl_2_fol.inference import data_model as dm
from nl_2_fol.inference import definition_model as def_model
from nl_2_fol.inference.chebi_data import ChEBIDataWrapper
from nl_2_fol.inference.model_check_molecule import GavelFOLReasoner
from nl_2_fol.prompting.chebai_prompt import ChebiPrompt
from nl_2_fol.prompting.prompt_models import CHEBIFOLOutput


# TODO: can langchain-graph be used here? or will it be an overkill?
class LearnDefinitions:
    _DEFINITION_FILE_NAME = "learned_definitions.pkl"

    def __init__(
        self,
        chebi_prompt_obj: ChebiPrompt,
        slim_dataset_path: str,
        structures_path: str,
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
        self._default_def_save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "learned",
            self.chebi_prompt_obj.model_name,
        )

        self._gavel = GavelFOLReasoner()

        self.definitions: def_model.DefinitionLearningResults = self._load_definitions(
            definitions_path
        )

        # ------ Entire Chebi data loading -------
        entire_chebi_data = ChEBIDataWrapper(chebi_version=244)
        self._chebi_name_to_data_mapping = entire_chebi_data.get_name_to_data_mapping()

        # ----- C3PO slim dataset loading -----
        (
            self._dataset,
            self.smiles_to_instance,
            self.validation_smiles,
            self.all_smiles,
        ) = dm.load_c3po_slim_dataset(
            entire_chebi_data, self.slim_dataset_path, self.structures_path
        )
        # ---------------------------------------

        self._attempts: int = 0
        self._prompts_history: list[str] = []

    def learn_fol_definitions(self):
        # Create a list copy to avoid "dictionary changed size during iteration" error
        # since self._dataset.classes.pop() is called during the loop
        for chemical_class_name in tqdm.tqdm(list(self._dataset.classes.keys())):
            if chemical_class_name not in self._dataset.classes:
                continue
            chemical_class = self._dataset.classes[chemical_class_name]
            if chemical_class.definition is None:
                continue
            if chemical_class.id in self.definitions.learned_definitions:
                continue
            self._learn(chemical_class)

    def _learn(self, chemical_class: dm.ChemicalClass) -> None:
        self._attempts = 0
        self._prompts_history = []

        raised_exception = None
        result, prompt_text = None, ""
        try:
            # """CHEBI:16236 - ethanol: A primary alcohol that is ethane in which one
            # of the hydrogens is substituted by a hydroxy group."""
            input_text = f"CHEBI:{chemical_class.id} - {chemical_class.name}: {chemical_class.definition}"
            result, prompt_text = self.chebi_prompt_obj.invoke_llm_with_fs_prompt(
                input_text
            )
            print(
                f"Initial attempt for CHEBI:{chemical_class.id}: {chemical_class.name}",
                f"\nInput text to LLM: {input_text}\n",
                f"Generated FOL definition: {result.FOL_formula}\n",
            )
            self._add_prompt_to_history(prompt_text, result)
            self._parse_and_validate_generated_definition(result, chemical_class)
            self._dataset.classes.pop(chemical_class.name)
            self._save_definitions()
            return
        except Exception as e:
            raised_exception = e
            if isinstance(e, ce.MissingPredicateException):
                raised_exception = self._handle_missing_predicates_exception(e)
            elif isinstance(e, ce.StopProgramException):
                raise e

        print(
            f"Failed to parse FOL definition for CHEBI:{chemical_class.id}: {chemical_class.name}:\n",
            f"\tRaised exception: {raised_exception}]\n",
        )
        previous_fol_def = result.FOL_formula if result else ""
        while self._attempts < self.max_attempts:
            print(
                f"Attempt {self._attempts + 1} for CHEBI:{chemical_class.id}: {chemical_class.name}"
            )
            add_bck_def = None
            if isinstance(raised_exception, ce.LearnOutOfBoxPredicateException):
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
                print(
                    f"Learned additional definitions for out-of-box predicates: {additional_def}\n"
                )
                try:
                    add_bck_def = self._gavel.convert_to_background_definitions(
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
            elif isinstance(raised_exception, ce.RetryException):
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
                print(f"\tGenerated FOL definition: {result.FOL_formula}\n")

            try:
                self._parse_and_validate_generated_definition(
                    result,  # pyright: ignore[reportArgumentType]
                    chemical_class,
                    add_background_defs=add_bck_def,
                )
                self._dataset.classes.pop(chemical_class.name)
                self._save_definitions()
                return
            except Exception as e:
                raised_exception = e
                if isinstance(e, ce.MissingPredicateException):
                    raised_exception = self._handle_missing_predicates_exception(e)
                elif isinstance(e, ce.StopProgramException):
                    raise e
                print(
                    f"Failed to parse FOL definition for CHEBI:{chemical_class.id}: {chemical_class.name}\n",
                    f"\tRaised exception: {raised_exception}]\n",
                )
                self._attempts += 1
                previous_fol_def = result.FOL_formula if result else previous_fol_def

    def _handle_missing_predicates_exception(
        self, e: ce.MissingPredicateException
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

        raised_exception = ce.RetryException()
        other_predicates = e.missing_predicates - chemical_class_predicates
        if other_predicates:
            predicates_to_learn: dict[str, str | None] = {}
            for predicate in other_predicates:
                if predicate in self._chebi_name_to_data_mapping:
                    chebi_data = self._chebi_name_to_data_mapping[predicate]
                    predicates_to_learn[predicate] = chebi_data["definition"]
                else:
                    predicates_to_learn[predicate] = None

            raised_exception = ce.LearnOutOfBoxPredicateException(
                predicates_to_learn=predicates_to_learn
            )
        return raised_exception

    def _parse_and_validate_generated_definition(
        self,
        result: CHEBIFOLOutput,
        chemical_class: dm.ChemicalClass,
        add_background_defs: dict[
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
                add_background_defs,
            )
        )
        metrics = self._get_metrics(
            unmatched_pos_samples, matched_neg_samples, pos_samples, neg_samples
        )

        # Validate against the threshold
        # TODO:  adjust threshold wrt how many def meet it
        if metrics.F1 < self.f1_threshold:
            raise ce.LowF1ScoreException(
                list(pos_samples),
                list(neg_samples),
                list(matched_neg_samples),
                list(unmatched_pos_samples),
                max_examples=10,
                chebi_id_to_data_mapping=self._chebi_name_to_data_mapping,
            )
        # TODO: What if the additonal defintions are changed in next attempt
        # and both are valid which to use? rn the earliest
        if add_background_defs:
            for def_name, (_, background_def) in add_background_defs.items():
                if def_name not in self.definitions.additional_definitions:
                    self.definitions.additional_definitions[def_name] = background_def
                    self.chebi_prompt_obj._generated_predicates_names.add(def_name)
                    self._gavel.update_background_definition(def_name, background_def)

        self.definitions.learned_definitions[chemical_class.id] = (
            def_model.LearnedDefinition(
                metrics=metrics,
                learned_FOL=tptp_def,
                name=chemical_class.name,
                definition=chemical_class.definition
                if chemical_class.definition
                else "",
                prompts_history=self._prompts_history,
            )
        )
        self._gavel.update_background_definition(chemical_class.name, tptp_def)
        self.chebi_prompt_obj._generated_predicates_names.add(chemical_class.name)
        print(
            f"Learned definition for {chemical_class.id} with F1 score: {metrics.F1:.2f}"
        )

    @ce.stop_program_exception
    def _get_positive_and_negative_samples(
        self, chemical_class: dm.ChemicalClass
    ) -> tuple[set[dm.ChemicalStructure], set[dm.ChemicalStructure]]:
        # validation examples already substracted during from positive examples
        positive_examples = chemical_class.all_positive_examples
        positive_instances = {
            self.smiles_to_instance[smiles]
            for smiles in positive_examples
            if smiles in self.smiles_to_instance
        }
        negative_examples = list(
            (self.all_smiles - positive_examples) - self.validation_smiles
        )
        negative_instances = {
            self.smiles_to_instance[smiles]
            for smiles in negative_examples
            if smiles in self.smiles_to_instance
        }
        assert len(positive_instances) > 0, (
            f"No positive samples found for {chemical_class.name}"
        )
        assert len(negative_instances) > 0, (
            f"No negative samples found for {chemical_class.name}"
        )

        return positive_instances, negative_instances

    def _check_if_definition_matches_samples(
        self,
        tptp_def: QuantifiedFormula,
        pos_samples: set[dm.ChemicalStructure],
        neg_samples: set[dm.ChemicalStructure],
        background_definitions: dict[
            str, tuple[list[logic.Variable], logic.QuantifiedFormula]
        ]
        | None = None,
    ) -> tuple[set[dm.SMILES_STRING], set[dm.SMILES_STRING]]:
        def is_matched(chemical: dm.ChemicalStructure) -> bool:
            return self._gavel.does_mol_match_tptp_definition(
                chemical.mol,
                tptp_def,
                additional_background_definitions=background_definitions,
            )

        unmatched_pos_samples = set()
        for chemical in tqdm.tqdm(
            pos_samples, desc="Checking definition for positive samples..."
        ):
            matches = is_matched(chemical)
            if not matches:
                unmatched_pos_samples.add(chemical.smiles)
        print(
            f"Unmatched positive samples: {len(unmatched_pos_samples)}/{len(pos_samples)}"
        )

        matched_neg_samples = set()
        for chemical in tqdm.tqdm(
            neg_samples, desc="Checking definition against negative samples..."
        ):
            matches = is_matched(chemical)
            if matches:
                matched_neg_samples.add(chemical.smiles)
        print(
            f"Matched negative samples: {len(matched_neg_samples)}/{len(neg_samples)}"
        )
        return unmatched_pos_samples, matched_neg_samples

    @ce.stop_program_exception
    def _get_metrics(
        self,
        unmatched_pos_samples: set[dm.SMILES_STRING],
        matched_neg_samples: set[dm.SMILES_STRING],
        pos_samples: set[dm.ChemicalStructure],
        neg_samples: set[dm.ChemicalStructure],
    ) -> def_model.DefinitionMetrics:
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
        return def_model.DefinitionMetrics(
            F1=f1,
            PPV=ppv,
            NPV=npv,
            TP=num_true_positives,
            FP=num_false_positives,
            FN=num_false_negatives,
            TN=num_true_negatives,
        )

    def _load_definitions(
        self, path: str | None
    ) -> def_model.DefinitionLearningResults:
        # load definitions from the given path and return as a dictionary
        # the key can be the chemical class and the value can be the FOL definition
        if path is not None:
            with open(path, "rb") as f:
                definitions = pickle.load(f)
        elif os.path.exists(
            default_path := os.path.join(
                self._default_def_save_path, self._DEFINITION_FILE_NAME
            )
        ):
            with open(default_path, "rb") as f:
                definitions = pickle.load(f)
        else:
            definitions = def_model.DefinitionLearningResults(
                learned_definitions={}, additional_definitions={}
            )
        self._load_background_defs_from_pmodel(definitions)
        return definitions

    def _load_background_defs_from_pmodel(
        self, new_definitions: def_model.DefinitionLearningResults
    ):
        """Load back the state from from learned definitions."""

        for _, learned_def in new_definitions.learned_definitions.items():
            self._gavel._background_definitions[learned_def.name] = (
                [],
                learned_def.learned_FOL,
            )
            self.chebi_prompt_obj._generated_predicates_names.add(learned_def.name)

        for name, add_def in new_definitions.additional_definitions.items():
            self._gavel._background_definitions[name] = ([], add_def)
            self.chebi_prompt_obj._generated_predicates_names.add(name)

    @ce.stop_program_exception
    def _save_definitions(self, path: str | None = None) -> None:
        # save the learned definitions to the given path
        if path is None:
            path = self._default_def_save_path
            os.makedirs(path, exist_ok=True)

        file_path = os.path.join(path, self._DEFINITION_FILE_NAME)
        meta_data_path = os.path.join(path, "__metadata__.txt")
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(meta_data_path):
            os.remove(meta_data_path)

        with open(file_path, "wb") as f:
            pickle.dump(self.definitions, f)

        with open(meta_data_path, "w") as f:
            f.write(str(self.chebi_prompt_obj))

    @ce.stop_program_exception
    def _add_prompt_to_history(
        self, prompt: str, result: CHEBIFOLOutput | None
    ) -> None:
        output = result.FOL_formula if result is not None else ""
        history_entry: str = (
            f"Prompt:\n{prompt}\n"
            f"{self.chebi_prompt_obj.model_name}(LLM) output:\n{output}\n"
        )
        self._prompts_history.append(history_entry)
