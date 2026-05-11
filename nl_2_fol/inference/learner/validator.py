import multiprocessing
import os
import pickle

import tqdm

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
        self._loaded_defs = self._load_definitions(defs_file_path)
        self.counter = 0

    def validate(self):
        classes_to_validate = [
            learned_def.name
            for _, learned_def in self._loaded_defs.learned_definitions.items()
            if learned_def.learn_success and learned_def.val_metrics is None
        ]
        if len(classes_to_validate) == 0:
            return

        ctx = multiprocessing.get_context("fork")
        result_queue = ctx.Queue()
        processes = []

        for class_name in tqdm.tqdm(
            classes_to_validate, desc="Launching validation processes"
        ):
            process = ctx.Process(
                target=self._validate_class_worker,
                args=(class_name, result_queue),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        self._collect_validation_results(result_queue, len(classes_to_validate))
        if self.counter > 0:
            print(
                f"Validation completed. {self.counter} definitions could not be validated due to errors during validation."
            )

    def validate_class(self, class_name: str):
        result = self._validate_class_result(class_name)
        if result is None:
            return

        status, class_id, val_metrics = result
        if status != "ok" or class_id is None or val_metrics is None:
            return

        self._loaded_defs.learned_definitions[class_id].val_metrics = val_metrics
        self._save_validated_definitions()

    def _validate_class_worker(self, class_name: str, result_queue):
        result_queue.put((class_name, self._validate_class_result(class_name)))

    def _collect_validation_results(self, result_queue, expected_results: int):
        updated = False
        for _ in range(expected_results):
            _, result = result_queue.get()
            if result is None:
                continue

            status, class_id, val_metrics = result
            if status == "error":
                self.counter += 1
                continue

            if status == "ok" and class_id is not None and val_metrics is not None:
                self._loaded_defs.learned_definitions[
                    class_id
                ].val_metrics = val_metrics
                updated = True

        if updated:
            self._save_validated_definitions()

    def _validate_class_result(
        self, class_name: str
    ) -> tuple[str, int | None, def_model.DefinitionMetrics | None] | None:
        try:
            print("-" * 10, f"Validating definition for {class_name}...", "-" * 10)
            chemical_class = self._c3po_slim_data.get_chemical_class_by_name(class_name)
            if chemical_class.id not in self._loaded_defs.learned_definitions:
                print(
                    f"No learned definition found for class {class_name} with id "
                    f"{chemical_class.id}. Skipping validation for this class."
                )
                return ("skipped", None, None)

            learned_def = self._loaded_defs.learned_definitions[chemical_class.id]
            if not learned_def.learn_success:
                print(
                    f"Learned definition for class {class_name} with id {chemical_class.id} "
                    f"was not successful learned during learning process. "
                    f"Skipping validation for this class."
                )
                return ("skipped", None, None)

            val_metrics = self._score_definition(
                chemical_class=chemical_class,
                tptp_def=learned_def.learned_FOL.formula,
                sample_match_timeout_seconds=None,
                max_neg_samples=self._MAX_NEGATIVE_SAMPLES,
                temp_additional_defs=None,
            )[0]
        except Exception as e:
            self.counter += 1
            print(f"Parsed Definition: {learned_def.learned_FOL.formula}")
            print(
                "Variables:",
                [str(var) for var in learned_def.learned_FOL.pred_variables],
            )

            if learned_def.additional_defs_used:
                print("Additional definitions used during learning:")
                for add_def_name, (
                    def_vars,
                    add_def,
                ) in learned_def.additional_defs_used.items():
                    print(
                        f"\t{add_def_name}: {add_def} \n\tVariables: {[str(var) for var in def_vars]}"
                    )
            print(
                f"Error during validation of definition for class {class_name}: \n\t{e}"
            )
            return ("error", chemical_class.id, None)

        return ("ok", chemical_class.id, val_metrics)

    def _save_validated_definitions(self):
        if "_with_val" not in self.defs_file_path:
            base_path, extension = os.path.splitext(self.defs_file_path)
            output_path = (
                f"{base_path}_with_val{extension}"
                if extension
                else f"{self.defs_file_path}_with_val"
            )
        else:
            output_path = self.defs_file_path
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
                self._fol_reasoner.add_background_definition(
                    learned_def.name,
                    learned_def.learned_FOL.pred_variables,
                    learned_def.learned_FOL.formula,
                )
                counter += 1

        print(f"Loaded {counter} definitions")

        counter = 0
        for name, add_def in new_definitions.additional_definitions.items():
            if add_def.learn_success:
                self._fol_reasoner.add_background_definition(
                    name,
                    add_def.fol_formula.pred_variables,
                    add_def.fol_formula.formula,
                )
                counter += 1
        print(f"Loaded {counter} additional definitions")

        return new_definitions
