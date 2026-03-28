import os
import pickle

import tqdm
from gavel.logic import logic
from gavel.logic.logic import QuantifiedFormula

from nl_2_fol.inference.learner import custom_exceptions as ce
from nl_2_fol.inference.learner import definition_model as def_model
from nl_2_fol.inference.learner.base import BaseFOL
from nl_2_fol.inference.learner.tee_stream import TeeStream
from nl_2_fol.inference.preprocessing import c3po_slim_data as dm
from nl_2_fol.prompting.chebai_prompt import ChebiPrompt
from nl_2_fol.prompting.prompt_models import CHEBIFOLOutput


class LearnDefinitions(BaseFOL):
    _MAX_SAMPLES_FOR_LOW_THRESHOLD_EXCEPTION = 5
    _LEARNING_LOG_FILE_NAME = "learning_output_a{max_attempts}.txt"

    def __init__(
        self,
        chebi_prompt_obj: ChebiPrompt,
        slim_dataset_path: str,
        structures_path: str,
        max_attempts: int = 4,
        f1_threshold: float = 0.8,
        chebi_version: int = 244,
    ):
        super().__init__(
            slim_dataset_path=slim_dataset_path,
            structures_path=structures_path,
            chebi_version=chebi_version,
            split="train",
        )
        self.chebi_prompt_obj = chebi_prompt_obj
        self.max_attempts = max_attempts
        self.f1_threshold = f1_threshold

        # Stores chemical classes which failed to learn FOL in previous programs calls,
        # so we can skip them in the main loop and avoid unnecessary attempts
        self._failed_classes: set = set()

        # Tracks the classes for which definitions have been learned,
        self._learned_classes: set[str] = set()
        self.definitions: def_model.DefinitionLearningResults = self._load_definitions()

        self._chebi_name_to_data_map_train = (
            self._entire_chebi_data.get_name_to_data_mapping_train()
        )

    def learn_fol_definitions(self):
        with TeeStream.capture_learning_output(self.learning_log_path):
            for chemical_class_name in tqdm.tqdm(self._c3po_slim_data.classes.keys()):
                if chemical_class_name in self._failed_classes:
                    continue
                if chemical_class_name in self._learned_classes:
                    # This class could have been learned duing recursive learning
                    # when handling missing predicate exception, so we skip it in the main loop
                    continue
                chemical_class = self._c3po_slim_data.classes[chemical_class_name]
                if chemical_class.definition is None:
                    continue
                if chemical_class.id in self.definitions.learned_definitions:
                    continue
                self._learn(chemical_class)

    def learn_class(self, class_name: str):
        with TeeStream.capture_learning_output(self.learning_log_path):
            if class_name not in self._c3po_slim_data.classes:
                print(f"{class_name} not found in the dataset.")
                return
            chemical_class = self._c3po_slim_data.classes[class_name]
            if chemical_class.definition is None:
                print(f"No definition available for {class_name}, skipping learning.")
                return
            if chemical_class.id in self.definitions.learned_definitions:
                print(
                    f"Definition already learned for {class_name}, skipping learning."
                )
                return
            if class_name in self._failed_classes:
                print(
                    f"{class_name} is in the list of classes which failed to learn in "
                    "previous runs, skipping learning to avoid unnecessary attempts."
                )
                while True:
                    retry_choice = (
                        input(
                            f"Do you want to retry learning '{class_name}'? (yes/no): "
                        )
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
        attempts = 0

        # Tracks all the definitions which scored below threshold from all attempts
        # If no generated def pass the threshold, then we accept the one with best score
        low_score_defs_collector: dict[int, def_model.ScoredDefinition] = {}

        result = None
        raised_exception = None
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
            self._parse_and_validate_generated_definition(
                result=result,
                chemical_class=chemical_class,
                low_score_defs_collector=low_score_defs_collector,
            )
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
        outofbox_max_attempts, curr_outofbox = 1, 0
        undef_retry_context: str | None = None
        while attempts < self.max_attempts:
            print(
                f"Attempt {attempts + 2} for CHEBI:{chemical_class.id}: {chemical_class.name}"
            )
            add_bck_def = None
            if isinstance(raised_exception, ce.LearnOutOfBoxPredicateException):
                learned_predicates = (
                    self.chebi_prompt_obj.invoke_llm_with_undef_failure_prompt(
                        raised_exception.predicates_to_learn,
                        session_id=chemical_class.name,
                        retry_context=undef_retry_context,
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
                    class_def = additional_def.pop(chemical_class.name, None)
                    if class_def is not None and result is not None:
                        # If LLM also returns a new definition for the main chemical class
                        # along with the missing predicates definitions, we use it directly
                        # instead of retrying with the same definition, as the new
                        # definition might have changes
                        print(
                            f"LLM returned a new definition for the main chemical "
                            f"class along with the missing predicates definitions, "
                            f"using it directly instead of retrying with the same "
                            f"definition: {class_def}"
                        )
                        result.FOL_formula = class_def
                    add_bck_def = self._gavel.convert_to_background_definitions(
                        additional_def
                    )
                    self._validate_additional_predicates(
                        add_bck_def, current_class_name=chemical_class.name
                    )
                except Exception as e:
                    if curr_outofbox < outofbox_max_attempts:
                        # If the additional definitions generated for out-of-box predicates
                        # are not parseable, we retry generating them up one more time, as
                        # we want to give the model a chance to correct them before consuming
                        # the main attempt for learning the chemical class definition
                        curr_outofbox += 1
                        attempts += 1
                        print(
                            f"Retrying learning for {chemical_class.name} due to unparseable additional definition for out-of-box predicate. Attempt {attempts + 3}"
                        )
                        undef_retry_context = (
                            "[IMPORTANT] Previously generated additional definitions "
                            "for undefined predicates failed while parsing due to the below "
                            f"reason:\n{e}\n"
                            "Please analyze this error and return corrected definitions."
                        )
                        continue

                    # If additional generated definitions for the missing predicates
                    # are not parseable, we return the error to llm.
                    # This will lead generating new FOL formula input chemical class
                    # instead of trying to fix the additional missing predicates definitions
                    raised_exception = e
                    attempts += 1
                    print(
                        f"Failed to validate out-of-box predicate definitions for {chemical_class.name}. "
                        f"Consuming attempt {attempts + 1}/{self.max_attempts + 1} and retrying."
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
                    low_score_defs_collector=low_score_defs_collector,
                )
                return
            except Exception as e:
                raised_exception = e
                if isinstance(e, ce.MissingPredicateException):
                    raised_exception = self._handle_missing_predicates_exception(e)
                    undef_retry_context = None
                elif isinstance(e, ce.StopProgramException):
                    raise e
                print(
                    f"Failed to parse FOL definition for CHEBI:{chemical_class.id}: {chemical_class.name}\n",
                    f"\tRaised exception: {raised_exception}]\n",
                )
                attempts += 1
                previous_fol_def = result.FOL_formula if result else previous_fol_def

        self._accept_highest_scoring_def(chemical_class, low_score_defs_collector)

    def _accept_highest_scoring_def(
        self,
        chemical_class: dm.ChemicalClass,
        low_score_defs_collector: dict[int, def_model.ScoredDefinition],
    ):
        if not low_score_defs_collector:
            print(
                "No generated FOL definition could be accepted because no "
                "candidate definitions were collected."
            )
            # If no generated FOL could be parsed/scored, persist a safe parsed
            # placeholder definition along with prompt history for traceability.
            pred_variables, placeholder_tptp_def = self._gavel.get_tptp_fol_definition(
                "failed_placeholder_predicate(X) <=> (c(X) & ~c(X))"
            )
            best_scored_def = def_model.ScoredDefinition(
                pred_variables=pred_variables,
                tptp_def=placeholder_tptp_def,
                train_metrics=def_model.DefinitionMetrics(
                    F1=0.0,
                    TN=0,
                    FP=0,
                    FN=0,
                    TP=0,
                    NPV=0.0,
                    PPV=0.0,
                ),
                temp_additional_defs=None,
            )
        else:
            best_scored_def = max(
                low_score_defs_collector.values(),
                key=lambda x: x.train_metrics.F1,
            )
        print(
            "As no generated FOL definition was able to pass the F1 threshold of "
            f"{self.f1_threshold:.2f}, accepting the definition with the highest F1 "
            f"score {best_scored_def.train_metrics.F1:.2f} "
            f"among {len(low_score_defs_collector)} low threshold defintions generated in "
            f"all previous attempts."
        )
        learn_success = True
        if best_scored_def.train_metrics.F1 <= 0.0:
            # Either the generated FOL was not able to process/check even a single sample
            # due to timeout during model checking,
            # Or the FOL was not able to classify single sample correctly
            learn_success = False

        self._accept_learned_definition(
            chemical_class, scored_def=best_scored_def, learn_success=learn_success
        )

    def _post_cleanup(self, session_id: str):
        # This is to clean up the session history after learning a definition for a chemical
        # class or attempts are exhausted, so avoid uncessary runtime memory usage
        self.chebi_prompt_obj.delete_session_history(session_id=session_id)

    def _validate_additional_predicates(
        self,
        add_bck_def: dict[str, tuple[list[logic.Variable], QuantifiedFormula]],
        current_class_name: str,
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
                unknown_pred_name = str(unknown_pred)
                if unknown_pred_name == current_class_name:
                    raise Exception(
                        f"The definition of {pred_name} introduces a circular "
                        "dependency by referencing the predicate of class currently being "
                        f"learned ('{current_class_name}')."
                    )

                if unknown_pred_name in self._failed_classes:
                    if unknown_pred_name not in add_bck_def:
                        print(
                            "[validate_additional] Failed class predicate has no "
                            f"definition provided: '{unknown_pred_name}'."
                        )
                        raise Exception(
                            f"Predicate {unknown_pred_name} which is part of the additional "
                            "background definition is also part of the failed classes list, "
                            "but no definition provided for it in the additional background "
                            "definitions. Hence we cannot validate the main predicate "
                            f"{pred_name} definition."
                        )
                elif unknown_pred_name in self._learned_classes:
                    if unknown_pred_name in add_bck_def:
                        # use learned definition instead of provided def
                        print(
                            "[validate_additional] Predicate already learned; removing "
                            f"redundant additional definition: '{unknown_pred_name}'."
                        )
                        add_bck_def.pop(unknown_pred_name)
                elif unknown_pred_name in self._c3po_slim_data.classes:
                    print(
                        "[validate_additional] Predicate is a slim dataset class; "
                        f"triggering recursive learning: '{unknown_pred_name}'."
                    )
                    self._learn(
                        self._c3po_slim_data.get_chemical_class_by_name(
                            unknown_pred_name
                        )
                    )
                elif unknown_pred_name not in add_bck_def:
                    # This means the definition provided for the missing predicate also
                    # contains unknown predicates which we don't have definitions for,
                    # hence we cannot validate it and we raise an exception to llm to
                    # generate a new definition for the main chemical class instead of
                    # trying to fix the additional background definition
                    print(
                        "[validate_additional] Unknown nested predicate is not provided "
                        f"in additional definitions: '{unknown_pred_name}'."
                    )
                    raise Exception(
                        "Additional background definition provided for missing predicate "
                        f"{unknown_pred_name} contains unknown predicates "
                        f"{self._gavel.extract_unknown_predicates(defn)} which we don't "
                        "have definitions for. Hence we cannot validate it."
                    )
                else:
                    print(
                        "[validate_additional] Unknown predicate has its own additional "
                        f"definition: '{unknown_pred_name}'."
                    )

            if pred_name in self._learned_classes:
                # Use definition which is already learned instead of provided def
                print(
                    "[validate_additional] Main predicate already learned; removing "
                    f"redundant additional definition: '{pred_name}'."
                )
                add_bck_def.pop(pred_name)
            elif pred_name in self._c3po_slim_data.classes:
                if pred_name == current_class_name:
                    # This block is unreachable because we already remove and use the
                    # definition corresponding to the class being currently learned from
                    # the additional definitions before calling this function, but we keep
                    # this check for safety to avoid any possible circular dependency
                    # which can lead to infinite recursion.
                    raise ce.StopProgramException(
                        f"The main predicate '{pred_name}' is the same as the current class being learned, "
                        "which is not allowed as it will lead to circular dependency."
                    )

                if pred_name in self._failed_classes:
                    # We still might want to attempt learning, now certain predicates
                    # which are part of the additional definition might be learned
                    # which can help learning the main chemical class definition
                    # [Placeholder] Might add logic trigger an error in future
                    # print(
                    #     "[validate_additional] Main predicate is currently marked failed "
                    #     "but learning will be attempted again: "
                    #     f"'{pred_name}'."
                    # )
                    # raise Exception(
                    #     "Cannot use predicate `{pred_name}` as its not possible to learn"
                    #     "a reliable first order logic defintion at the moment"
                    # )
                    # TODO

                    # Let it learn as an additional definition
                    continue
                print(
                    "[validate_additional:recursive] Triggering recursive learning "
                    f"for main predicate: '{pred_name}'."
                    "\n--------------------------------------------------------------"
                )
                self._learn(self._c3po_slim_data.get_chemical_class_by_name(pred_name))
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
            if predicate in self._c3po_slim_data.classes
            and predicate not in self._learned_classes
            and predicate not in self._failed_classes
        }
        for predicate in chemical_class_predicates:
            print(
                f"Missing predicate '{predicate}' is a chemical class in the slim dataset and not yet learned, "
                "hence triggering learning for it first before handling other missing predicates."
            )
            self._learn(self._c3po_slim_data.get_chemical_class_by_name(predicate))

        raised_exception = ce.RetryException()
        other_predicates = e.missing_predicates - chemical_class_predicates
        if other_predicates:
            predicates_to_learn: dict[str, str | None] = {}
            for predicate in other_predicates:
                if predicate in self._chebi_name_to_data_map_train:
                    chebi_data = self._chebi_name_to_data_map_train[predicate]
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
        low_score_defs_collector: dict[int, def_model.ScoredDefinition],
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

        (
            train_metrics,
            unmatched_pos_samples,
            matched_neg_samples,
            processed_pos_samples,
            processed_neg_samples,
        ) = self._score_definition(
            chemical_class=chemical_class,
            tptp_def=tptp_def,
            sample_match_timeout_seconds=self._SAMPLE_MATCH_TIMEOUT_SECONDS,
            max_neg_samples=self._MAX_NEGATIVE_SAMPLES,
            temp_additional_defs=temp_additional_defs,
        )
        scored_def = def_model.ScoredDefinition(
            pred_variables=pred_variables,
            tptp_def=tptp_def,
            train_metrics=train_metrics,
            temp_additional_defs=temp_additional_defs,
        )

        if train_metrics.F1 < self.f1_threshold:
            low_score_defs_collector[len(low_score_defs_collector)] = scored_def

            print(
                f"F1 score {train_metrics.F1:.2f} is below the threshold of "
                f"{self.f1_threshold:.2f} for CHEBI:{chemical_class.id}: "
                f"{chemical_class.name}"
            )
            raise ce.LowF1ScoreException(
                current_f1_score=train_metrics.F1,
                pos_samples=processed_pos_samples,
                neg_samples=processed_neg_samples,
                matched_neg_samples=matched_neg_samples,
                unmatched_pos_samples=unmatched_pos_samples,
                max_examples=self._MAX_SAMPLES_FOR_LOW_THRESHOLD_EXCEPTION,
                chebi_name_to_data_mapping=self._chebi_name_to_data_map_train,
            )

        self._accept_learned_definition(
            chemical_class,
            scored_def,
            learn_success=True,
        )

    def _accept_learned_definition(
        self,
        chemical_class: dm.ChemicalClass,
        scored_def: def_model.ScoredDefinition,
        learn_success: bool,
    ):
        # TODO: What if a chemical class wants to use differnt additional definition
        # for the same predicate, than the one used by another chemical class?
        # Rn we only keep the earliest valid additional definition
        if scored_def.temp_additional_defs and learn_success:
            # Make temp additional defs permanent, as the main FOL using them has passed the f1 threshold
            for def_name, (
                pred_vars,
                background_def,
            ) in scored_def.temp_additional_defs.items():
                if def_name not in self.definitions.additional_definitions:
                    self.definitions.additional_definitions[def_name] = (
                        def_model.AdditionalDefinition(
                            used_for=[chemical_class.id],
                            fol_formula=def_model.FOLFormula(
                                formula=background_def, pred_variables=pred_vars
                            ),
                            learn_success=learn_success,
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
                train_metrics=scored_def.train_metrics,
                learned_FOL=def_model.FOLFormula(
                    formula=scored_def.tptp_def,
                    pred_variables=scored_def.pred_variables,
                ),
                # llm may rename `3OxoSteroid` to `threeOxoSteroid` hence use chemical.name
                # See:  https://github.com/ChEB-AI/chebai-NL2FOL/issues/13
                name=chemical_class.name,
                definition=chemical_class.definition
                if chemical_class.definition
                else "",
                prompts_history=prompts_history,
                learn_success=learn_success,
                additional_defs_used=scored_def.temp_additional_defs,
            )
        )
        if learn_success:
            self._gavel.add_background_definition(
                chemical_class.name, scored_def.pred_variables, scored_def.tptp_def
            )
            self._add_generated_predicates_to_prompt_obj(
                chemical_class.name, scored_def.pred_variables
            )
        print(
            f"Learned definition for {chemical_class.id}:{chemical_class.name} "
            f"with F1 score: {scored_def.train_metrics.F1:.2f}"
        )

        if not learn_success:
            self._failed_classes.add(chemical_class.name)
        else:
            self._learned_classes.add(chemical_class.name)
            if chemical_class.name in self._failed_classes:
                # some recursive calls might lead to re-learning of already failed classes
                self._failed_classes.remove(chemical_class.name)
        self._save_definitions()
        self._post_cleanup(session_id=chemical_class.name)

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

    def _load_definitions(
        self,
    ) -> def_model.DefinitionLearningResults:
        # load definitions from the given path and return as a dictionary
        # the key can be the chemical class and the value can be the FOL definition
        file_name = self._DEFINITION_FILE_NAME.format(max_attempts=self.max_attempts)
        default_path = os.path.join(self.definitions_save_path, file_name)
        print(f"Loading definitions from {default_path} if it exists...")
        if os.path.exists(default_path):
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
        loaded_def_names = []
        for _, learned_def in new_definitions.learned_definitions.items():
            if learned_def.learn_success:
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
            else:
                # If class has completely failed to learn, we ignore it
                self._failed_classes.add(learned_def.name)

        print(f"Loaded definitions for the following classes: {loaded_def_names}")

        loaded_additional_def_names = []
        for name, add_def in new_definitions.additional_definitions.items():
            if add_def.learn_success:
                self._gavel.add_background_definition(
                    name,
                    add_def.fol_formula.pred_variables,
                    add_def.fol_formula.formula,
                )
                self._add_generated_predicates_to_prompt_obj(
                    name, add_def.fol_formula.pred_variables
                )
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

        file_name = self._DEFINITION_FILE_NAME.format(max_attempts=self.max_attempts)
        file_path = os.path.join(path, file_name)
        meta_data_path = os.path.join(path, "__metadata__.txt")
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(meta_data_path):
            os.remove(meta_data_path)

        with open(file_path, "wb") as f:
            pickle.dump(self.definitions, f)

        with open(meta_data_path, "w") as f:
            f.write(str(self))

    @property
    def definitions_save_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "learned",
            self.chebi_prompt_obj.model_name,
        )

    def __repr__(self) -> str:
        return f"""
        DefinitionLearner(chebi_prompt_obj={self.chebi_prompt_obj}),
        max_attempts={self.max_attempts},
        f1_threshold={self.f1_threshold},
        slim_dataset_path={self.slim_dataset_path},
        structures_path={self.structures_path},
        chebi_version={self.chebi_version},
        Maximum_negative_samples={self._MAX_NEGATIVE_SAMPLES},
        Sample_match_timeout_seconds={self._SAMPLE_MATCH_TIMEOUT_SECONDS},
        Max_samples_for_low_threshold_exception={self._MAX_SAMPLES_FOR_LOW_THRESHOLD_EXCEPTION}
        """

    @property
    def learning_log_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "learned",
            self.chebi_prompt_obj.model_name,
            self._LEARNING_LOG_FILE_NAME.format(max_attempts=self.max_attempts),
        )
