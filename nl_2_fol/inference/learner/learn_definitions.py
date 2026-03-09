import json
import os
import pickle
import queue
import traceback

import tqdm
from gavel.logic import logic
from gavel.logic.logic import QuantifiedFormula

from nl_2_fol.inference.fol_reasoner import GavelFOLReasoner
from nl_2_fol.inference.learner import custom_exceptions as ce
from nl_2_fol.inference.learner import definition_model as def_model
from nl_2_fol.inference.preprocessing import c3po_slim_data as dm
from nl_2_fol.inference.preprocessing.chebi_data import ChEBIDataWrapper
from nl_2_fol.prompting.chebai_prompt import ChebiPrompt
from nl_2_fol.prompting.prompt_models import CHEBIFOLOutput


class LearnDefinitions:
    _DEFINITION_FILE_NAME = "learned_definitions.pkl"
    _MAX_NEGATIVE_SAMPLES = 5000
    _MAX_SAMPLES_FOR_UNDEFINED_PREDICATE_EXCEPTION = 5

    def __init__(
        self,
        chebi_prompt_obj: ChebiPrompt,
        slim_dataset_path: str,
        structures_path: str,
        max_attempts: int = 4,
        f1_threshold: float = 0.8,
    ):
        self.slim_dataset_path = slim_dataset_path
        self.structures_path = structures_path
        self.chebi_prompt_obj = chebi_prompt_obj
        self.max_attempts = max_attempts
        self.f1_threshold = f1_threshold
        # load definitions from the path and store them in a suitable data structure
        # this will be used to learn new definitions based on the classified chemical classes

        self._gavel = GavelFOLReasoner()
        # Stores chemical classes which failed to learn FOL in previous programs calls,
        # so we can skip them in the main loop and avoid unnecessary attempts
        self._failed_classes: dict[str, list[str]] = {}

        # Tracks the classes for which definitions have been learned,
        # so we can avoid redundant learning when handling missing predicate exception
        self._learned_classes: set[str] = set()
        self.definitions: def_model.DefinitionLearningResults = self._load_definitions()

        # Load Entire Chebi data
        entire_chebi_data = ChEBIDataWrapper(chebi_version=244)
        self._chebi_name_to_data_mapping = entire_chebi_data.get_name_to_data_mapping()

        self._c3po_slim_dataset = dm.load_c3po_slim_dataset(
            entire_chebi_data, self.slim_dataset_path, self.structures_path
        )
        self.undirected_chebi_graph = entire_chebi_data.get_undirected_hierarchy_graph()
        self._attempts: int = 0

    def learn_fol_definitions(self):
        for chemical_class_name in tqdm.tqdm(self._c3po_slim_dataset.classes.keys()):
            if chemical_class_name in self._failed_classes:
                continue
            if chemical_class_name in self._learned_classes:
                # This class could have been learned duing recursive learning
                # when handling missing predicate exception, so we skip it in the main loop
                continue
            chemical_class = self._c3po_slim_dataset.classes[chemical_class_name]
            if chemical_class.definition is None:
                continue
            if chemical_class.id in self.definitions.learned_definitions:
                continue
            self._learn(chemical_class)

    def learn_class(self, class_name: str):
        if class_name not in self._c3po_slim_dataset.classes:
            print(f"{class_name} not found in the dataset.")
            return
        chemical_class = self._c3po_slim_dataset.classes[class_name]
        if chemical_class.definition is None:
            print(f"No definition available for {class_name}, skipping learning.")
            return
        if chemical_class.id in self.definitions.learned_definitions:
            print(f"Definition already learned for {class_name}, skipping learning.")
            return
        if class_name in self._failed_classes:
            print(
                f"{class_name} is in the list of classes which failed to learn in "
                "previous runs, skipping learning to avoid unnecessary attempts."
            )
            while True:
                retry_choice = (
                    input(f"Do you want to retry learning '{class_name}'? (yes/no): ")
                    .strip()
                    .lower()
                )
                if retry_choice in {"yes", "y"}:
                    break
                if retry_choice in {"no", "n"}:
                    print(f"Skipping retry for {class_name}.")
                    return
                print("Please answer with 'yes' or 'no'.")
        self._learn(chemical_class)

    def _learn(self, chemical_class: dm.ChemicalClass) -> None:
        self._attempts = 0
        attempt_failure_summary = []

        raised_exception = None
        result = None
        try:
            # """CHEBI:16236 - ethanol: A primary alcohol that is ethane in which one
            # of the hydrogens is substituted by a hydroxy group."""
            input_text = f"CHEBI:{chemical_class.id} - {chemical_class.name}: {chemical_class.definition}"
            result = self.chebi_prompt_obj.invoke_llm_first_call(
                input_text=input_text, session_id=chemical_class.name
            )
            print(
                f"Initial attempt for CHEBI:{chemical_class.id}: {chemical_class.name}",
                f"\nInput text to LLM: {input_text}\n",
                f"Generated FOL definition: {result.FOL_formula}\n",
            )
            self._parse_and_validate_generated_definition(result, chemical_class)
            self._on_successful_learning(chemical_class)
            return
        except Exception as e:
            error_trace = traceback.format_exc()
            raised_exception = e
            if isinstance(e, ce.MissingPredicateException):
                raised_exception = self._handle_missing_predicates_exception(e)
            elif isinstance(e, ce.StopProgramException):
                raise e
            attempt_failure_summary.append(
                f"Attempt {self._attempts} failed with exception: {raised_exception}\nStacktrace:\n{error_trace}"
            )

        print(
            f"Failed to parse FOL definition for CHEBI:{chemical_class.id}: {chemical_class.name}:\n",
            f"\tRaised exception: {raised_exception}]\n",
        )
        previous_fol_def = result.FOL_formula if result else ""
        while self._attempts < self.max_attempts:
            print(
                f"Attempt {self._attempts + 2} for CHEBI:{chemical_class.id}: {chemical_class.name}"
            )
            add_bck_def = None
            if isinstance(raised_exception, ce.LearnOutOfBoxPredicateException):
                learned_predicates = (
                    self.chebi_prompt_obj.invoke_llm_with_undef_failure_prompt(
                        raised_exception.predicates_to_learn,
                        session_id=chemical_class.name,
                    )
                )

                additional_def = learned_predicates.predicate_definitions
                formatted_additional_definitions = "\n\t".join(
                    f"{pred}: {defn}" for pred, defn in additional_def.items()
                )
                print(
                    f"[{chemical_class.name}]Learned additional definitions for out-of-box predicates:"
                    f"\n\t{formatted_additional_definitions}\n"
                )

                try:
                    add_bck_def = self._gavel.convert_to_background_definitions(
                        additional_def
                    )
                    self._validate_additional_predicates(add_bck_def)
                except Exception as e:
                    error_trace = traceback.format_exc()
                    # If additional generated definitions for the missing predicates
                    # are not parseable, we return the error to llm.
                    # This will lead generating new FOL formula input chemical class
                    # instead of trying to fix the additional missing predicates definitions
                    raised_exception = Exception(
                        f"Failed to parse FOL definition for the following predicate:"
                        f"{additional_def}. Error: {e}"
                    )
                    attempt_failure_summary.append(
                        f"Attempt {self._attempts + 2} failed with unparseable additional"
                        f"definition for out-of-box predicate: {additional_def}. Error: {e}\nStacktrace:\n{error_trace}"
                    )
                    continue
            elif isinstance(raised_exception, ce.RetryException):
                # Retries the result generated from previous attempt
                # This is because certain undefined predicated might be known by now
                pass
            else:
                result = self.chebi_prompt_obj.invoke_llm_with_error_failure_prompt(
                    error_message=str(raised_exception), session_id=chemical_class.name
                )
                print(f"\tGenerated FOL definition: {result.FOL_formula}\n")

            try:
                self._parse_and_validate_generated_definition(
                    result,  # pyright: ignore[reportArgumentType]
                    chemical_class,
                    temp_additional_defs=add_bck_def,
                )
                self._on_successful_learning(chemical_class)
                return
            except Exception as e:
                error_trace = traceback.format_exc()
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
                attempt_failure_summary.append(
                    f"Attempt {self._attempts} failed with exception: {raised_exception}\nStacktrace:\n{error_trace}"
                )
                previous_fol_def = result.FOL_formula if result else previous_fol_def

        self._failed_classes[chemical_class.name] = attempt_failure_summary
        self._save_not_learned_classes_list()
        self._post_cleanup(session_id=chemical_class.name)

    def _on_successful_learning(self, chemical_class: dm.ChemicalClass):
        self._learned_classes.add(chemical_class.name)
        if chemical_class.name in self._failed_classes:
            # some recursive calls might lead to re-learning of already failed classes
            del self._failed_classes[chemical_class.name]
        self._save_definitions()
        self._post_cleanup(session_id=chemical_class.name)

    def _post_cleanup(self, session_id: str):
        # This is to clean up the session history after learning a definition for a chemical
        # class or attempts are exhausted, so avoid uncessary runtime memory usage
        self.chebi_prompt_obj.delete_session_history(session_id=session_id)

    def _validate_additional_predicates(
        self, add_bck_def: dict[str, tuple[list[logic.Variable], QuantifiedFormula]]
    ):
        # Formula 1: class A -> class B & class C
        # Formula 2: class B -> class D
        # Formula 3: class C -> class E & class F
        print(
            "Validating additional background predicate definitions: "
            f"{len(add_bck_def)} candidate(s)."
        )
        for pred_name, (vars, defn) in list(add_bck_def.items()):
            print(f"[validate_additional] Checking predicate '{pred_name}'")
            # first extract the unknown predicates from the formula
            unknown_predicates = self._gavel.extract_unknown_predicates(defn)
            print(
                f"[validate_additional] Unknown predicates in '{pred_name}': "
                f"{unknown_predicates}"
            )
            for unknown_pred in unknown_predicates:
                if unknown_pred in self._failed_classes:
                    if unknown_pred not in add_bck_def:
                        print(
                            "[validate_additional] Failed class predicate has no "
                            f"definition provided: '{unknown_pred}'."
                        )
                        raise Exception(
                            f"Predicate {unknown_pred} which is part of the additional "
                            "background definition is also part of the failed classes list, "
                            "but no definition provided for it in the additional background "
                            "definitions. Hence we cannot validate the main predicate "
                            f"{pred_name} definition."
                        )
                elif unknown_pred in self._learned_classes:
                    if unknown_pred in add_bck_def:
                        # use learned definition instead of provided def
                        print(
                            "[validate_additional] Predicate already learned; removing "
                            f"redundant additional definition: '{unknown_pred}'."
                        )
                        add_bck_def.pop(unknown_pred)
                elif unknown_pred in self._c3po_slim_dataset.classes:
                    print(
                        "[validate_additional] Predicate is a slim dataset class; "
                        f"triggering recursive learning: '{unknown_pred}'."
                    )
                    self._learn(
                        self._c3po_slim_dataset.get_chemical_class_by_name(unknown_pred)
                    )
                elif unknown_pred not in add_bck_def:
                    # This means the definition provided for the missing predicate also contains unknown predicates which we don't have definitions for, hence we cannot validate it and we raise an exception to llm to generate a new definition for the main chemical class instead of trying to fix the additional background definition
                    print(
                        "[validate_additional] Unknown nested predicate is not provided "
                        f"in additional definitions: '{unknown_pred}'."
                    )
                    raise Exception(
                        "Additional background definition provided for missing predicate "
                        f"{unknown_pred} contains unknown predicates "
                        f"{self._gavel.extract_unknown_predicates(defn)} which we don't "
                        "have definitions for. Hence we cannot validate it."
                    )
                else:
                    print(
                        "[validate_additional] Unknown predicate has its own additional "
                        f"definition: '{unknown_pred}'."
                    )

            if pred_name in self._learned_classes:
                # Use definition which is already learned instead of provided def
                print(
                    "[validate_additional] Main predicate already learned; removing "
                    f"redundant additional definition: '{pred_name}'."
                )
                add_bck_def.pop(pred_name)
            elif pred_name in self._c3po_slim_dataset.classes:
                if pred_name in self._failed_classes:
                    # We still might want to attempt learning, now certain predicates
                    # which are part of the additional definition might be learned
                    # which can help learning the main chemical class definition
                    # [Placeholder] Might add logic trigger an error in future
                    print(
                        "[validate_additional] Main predicate is currently marked failed "
                        "but learning will be attempted again: "
                        f"'{pred_name}'."
                    )
                    pass
                print(
                    "[validate_additional] Triggering learning for main predicate: "
                    f"'{pred_name}'."
                )
                self._learn(
                    self._c3po_slim_dataset.get_chemical_class_by_name(pred_name)
                )
            else:
                print(
                    "[validate_additional] Main predicate is out-of-box and remains as "
                    f"an additional definition: '{pred_name}'."
                )

        print("[validate_additional] Validation pass complete.")

    def _handle_missing_predicates_exception(
        self, e: ce.MissingPredicateException
    ) -> Exception:
        # Predicates which are part of slim dataset classes and which are
        # not already learned or failed in previous attempts,
        # will be recursively learned first before handling other missing predicates
        chemical_class_predicates = {
            predicate
            for predicate in e.missing_predicates
            if predicate in self._c3po_slim_dataset.classes
            and predicate not in self._learned_classes
            and predicate not in self._failed_classes
        }
        for predicate in chemical_class_predicates:
            print(
                f"Missing predicate '{predicate}' is a chemical class in the slim dataset and not yet learned, "
                "hence triggering learning for it first before handling other missing predicates."
            )
            self._learn(self._c3po_slim_dataset.get_chemical_class_by_name(predicate))

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
        temp_additional_defs: dict[
            str, tuple[list[logic.Variable], logic.QuantifiedFormula]
        ]
        | None = None,
    ) -> None:
        """
        Parses the generated FOL definition and validates it against the positive
        and negative samples of the chemical class.

        Raises an exception if parsing or validation fails, otherwise returns None.
        """

        pred_variables, tptp_def = self._gavel.get_tptp_fol_definition(
            result.FOL_formula
        )

        pos_samples, neg_samples = self._get_positive_and_negative_samples(
            chemical_class,
            self._MAX_NEGATIVE_SAMPLES,
        )

        unmatched_pos_samples, matched_neg_samples = (
            self._check_if_definition_matches_samples(
                chemical_class,
                tptp_def,
                pos_samples,
                neg_samples,
                temp_additional_defs,
            )
        )
        metrics = self._get_metrics(
            unmatched_pos_samples, matched_neg_samples, pos_samples, neg_samples
        )

        # Validate against the threshold
        # TODO:  adjust threshold wrt how many def meet it
        if metrics.F1 < self.f1_threshold:
            print(
                f"F1 score {metrics.F1:.2f} is below the threshold of "
                f"{self.f1_threshold:.2f} for CHEBI:{chemical_class.id}: "
                f"{chemical_class.name}"
            )
            raise ce.LowF1ScoreException(
                current_f1_score=metrics.F1,
                pos_samples=pos_samples,
                neg_samples=neg_samples,
                matched_neg_samples=matched_neg_samples,
                unmatched_pos_samples=unmatched_pos_samples,
                max_examples=self._MAX_SAMPLES_FOR_UNDEFINED_PREDICATE_EXCEPTION,
                chebi_name_to_data_mapping=self._chebi_name_to_data_mapping,
            )
        # TODO: What if the additonal defintions are changed in next attempt
        # and both are valid which to use? rn the earliest

        if temp_additional_defs:
            # Make temp additional defs permanent, as the main FOL using them has passed the f1 threshold
            for def_name, (pred_vars, background_def) in temp_additional_defs.items():
                if def_name not in self.definitions.additional_definitions:
                    self.definitions.additional_definitions[def_name] = (
                        def_model.FOLFormula(
                            formula=background_def, pred_variables=pred_vars
                        )
                    )
                    self._add_generated_predicates_to_prompt_obj(def_name, pred_vars)
                    self._gavel.add_background_definition(
                        def_name, pred_vars, background_def
                    )

        prompts_history = self.chebi_prompt_obj.get_full_conversation_context(
            chemical_class.name
        )
        self.definitions.learned_definitions[chemical_class.id] = (
            def_model.LearnedDefinition(
                metrics=metrics,
                learned_FOL=def_model.FOLFormula(
                    formula=tptp_def, pred_variables=pred_variables
                ),
                name=chemical_class.name,
                definition=chemical_class.definition
                if chemical_class.definition
                else "",
                prompts_history=prompts_history,
            )
        )
        self._gavel.add_background_definition(
            chemical_class.name, pred_variables, tptp_def
        )
        self._add_generated_predicates_to_prompt_obj(
            chemical_class.name, pred_variables
        )
        print(
            f"Learned definition for {chemical_class.id} with F1 score: {metrics.F1:.2f}"
        )

    def _add_generated_predicates_to_prompt_obj(
        self,
        pred_name: str,
        vars: list[logic.Variable],
    ) -> None:
        """Add a predicate with its variables to the prompt object.

        Example: if pred_name='oligopeptide' and vars=[x0, x1],
                 this will add 'oligopeptide(x0, x1)' to generated_predicates_names

        If no variables, only the predicate name is added.
        """
        if len(vars) > 0:
            variables_str = ", ".join(str(var) for var in vars)
            predicate_with_vars = f"{pred_name}({variables_str})"
        else:
            predicate_with_vars = pred_name
        self.chebi_prompt_obj.generated_predicates_names.add(predicate_with_vars)

    @ce.stop_program_upon_failure
    def _get_closest_negatives(
        self, available_smiles: list[str], target_id, n_samples=100
    ) -> list[dm.SMILES_STRING]:
        # get closest samples in terms of distance in chebi
        if n_samples >= len(available_smiles):
            return available_smiles

        q = queue.Queue()
        q.put(int(target_id))
        visited = set()
        selected_smiles = set()

        # BFS until we get n_samples or exhaust the graph
        # select closest labels to target_id and choose SMILES from those labels until we have n_samples
        while not q.empty() and len(selected_smiles) < n_samples:
            current = q.get()
            for neighbor in self.undirected_chebi_graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.put(neighbor)
                    if str(neighbor) in self._c3po_slim_dataset.id_to_class_name:
                        for smiles in self._c3po_slim_dataset.classes[
                            self._c3po_slim_dataset.id_to_class_name[str(neighbor)]
                        ].all_positive_examples:
                            if smiles in available_smiles:
                                selected_smiles.add(smiles)
                            if len(selected_smiles) >= n_samples:
                                return list(selected_smiles)

        return list(selected_smiles)

    @ce.stop_program_upon_failure
    def _get_positive_and_negative_samples(
        self, chemical_class: dm.ChemicalClass, max_neg_samples: int = 5000
    ) -> tuple[set[dm.ChemicalStructure], set[dm.ChemicalStructure]]:
        # validation examples already substracted during from positive examples
        positive_examples = chemical_class.all_positive_examples
        positive_instances = {
            self._c3po_slim_dataset.smiles_to_instance[smiles]
            for smiles in positive_examples
            if smiles in self._c3po_slim_dataset.smiles_to_instance
        }
        negative_examples = list(
            (self._c3po_slim_dataset.all_smiles - positive_examples)
            - self._c3po_slim_dataset.validation_examples
        )
        negative_examples = self._get_closest_negatives(
            negative_examples, chemical_class.id, n_samples=max_neg_samples
        )
        negative_instances = {
            self._c3po_slim_dataset.smiles_to_instance[smiles]
            for smiles in negative_examples
            if smiles in self._c3po_slim_dataset.smiles_to_instance
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
        chemical_class: dm.ChemicalClass,
        tptp_def: QuantifiedFormula,
        pos_samples: set[dm.ChemicalStructure],
        neg_samples: set[dm.ChemicalStructure],
        temp_additional_defs: dict[
            str, tuple[list[logic.Variable], logic.QuantifiedFormula]
        ]
        | None = None,
    ) -> tuple[set[dm.SMILES_STRING], set[dm.SMILES_STRING]]:
        def is_matched(chemical: dm.ChemicalStructure) -> bool:
            return self._gavel.does_mol_match_tptp_definition(
                chemical.mol,
                tptp_def,
                temp_additional_defs=temp_additional_defs,
            )

        unmatched_pos_samples = set()
        for chemical in tqdm.tqdm(
            pos_samples,
            desc=f"Checking definition of {chemical_class.name} for positive samples...",
        ):
            matches = is_matched(chemical)
            if not matches:
                unmatched_pos_samples.add(chemical.smiles)
        print(
            f"\nUnmatched positive samples for {chemical_class.name}: {len(unmatched_pos_samples)}/{len(pos_samples)}"
        )

        matched_neg_samples = set()
        for chemical in tqdm.tqdm(
            neg_samples,
            desc=f"Checking definition of {chemical_class.name} against negative samples...",
        ):
            matches = is_matched(chemical)
            if matches:
                matched_neg_samples.add(chemical.smiles)
        print(
            f"\nMatched negative samples for {chemical_class.name}: {len(matched_neg_samples)}/{len(neg_samples)}"
        )
        return unmatched_pos_samples, matched_neg_samples

    @ce.stop_program_upon_failure
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
        self,
    ) -> def_model.DefinitionLearningResults:
        # load definitions from the given path and return as a dictionary
        # the key can be the chemical class and the value can be the FOL definition
        default_path = os.path.join(
            self.definitions_save_path, self._DEFINITION_FILE_NAME
        )
        print(f"Loading definitions from {default_path} if it exists...")
        if os.path.exists(default_path):
            with open(default_path, "rb") as f:
                definitions = pickle.load(f)
        else:
            definitions = def_model.DefinitionLearningResults(
                learned_definitions={}, additional_definitions={}
            )
        self._load_not_learned_classes_list()
        self._load_background_defs_from_pmodel(definitions)
        return definitions

    def _load_not_learned_classes_list(self):
        # This is to load the list of not learned classes from previous runs of the program, so we can skip them in the main loop and avoid unnecessary attempts
        unlearned_classes_path = os.path.join(
            self.definitions_save_path, "unlearned_classes.json"
        )
        if os.path.exists(unlearned_classes_path):
            with open(unlearned_classes_path, "r") as f:
                # Load class names from JSON file
                saved_classes_data = json.load(f)
                # Initialize the dict with loaded class names but empty failure lists for this run
                self._failed_classes = {
                    class_name: [] for class_name in saved_classes_data.keys()
                }
                print(
                    "Loaded not learned classes from previous runs:"
                    f"{set(saved_classes_data.keys())}.\n"
                    "Hence this run will skip learning definitions for these classes and "
                    "avoid unnecessary attempts."
                )

    def _save_not_learned_classes_list(self):
        # This is to save the list of not learned classes to a file, so we can load it in the next runs and skip them in the main loop and avoid unnecessary attempts
        unlearned_classes_path = os.path.join(
            self.definitions_save_path, "unlearned_classes.json"
        )

        # Load existing saved classes
        saved_classes_data = {}
        if os.path.exists(unlearned_classes_path):
            with open(unlearned_classes_path, "r") as f:
                saved_classes_data = json.load(f)

        # Update with new classes and their failures (only append new ones)
        for class_name, failure_list in self._failed_classes.items():
            if class_name not in saved_classes_data:
                # Only save if new class
                saved_classes_data[class_name] = failure_list if failure_list else []

        # Write the entire updated data to JSON file
        with open(unlearned_classes_path, "w") as f:
            json.dump(saved_classes_data, f, indent=2)

    def _load_background_defs_from_pmodel(
        self, new_definitions: def_model.DefinitionLearningResults
    ):
        """Load back the state from from learned definitions."""
        loaded_def_names = []
        for _, learned_def in new_definitions.learned_definitions.items():
            self._gavel.add_background_definition(
                learned_def.name,
                learned_def.learned_FOL.pred_variables,
                learned_def.learned_FOL.formula,
            )
            self._add_generated_predicates_to_prompt_obj(
                learned_def.name, learned_def.learned_FOL.pred_variables
            )
            loaded_def_names.append(learned_def.name)
            self._learned_classes.add(learned_def.name)
        print(f"Loaded definitions for the following classes: {loaded_def_names}")

        loaded_additional_def_names = []
        for name, add_def in new_definitions.additional_definitions.items():
            self._gavel.add_background_definition(
                name, add_def.pred_variables, add_def.formula
            )
            self._add_generated_predicates_to_prompt_obj(name, add_def.pred_variables)
            loaded_additional_def_names.append(name)
            self._learned_classes.add(name)
        print(
            f"Loaded the following additional definitions for out-of-box predicates: {loaded_additional_def_names}"
        )

    @ce.stop_program_upon_failure
    def _save_definitions(self, path: str | None = None) -> None:
        # save the learned definitions to the given path
        if path is None:
            path = self.definitions_save_path
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

    @property
    def definitions_save_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "learned",
            self.chebi_prompt_obj.model_name,
        )
