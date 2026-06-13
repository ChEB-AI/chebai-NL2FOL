import fcntl
import multiprocessing
import os
import pickle
import queue
import tempfile
import time
from collections import deque
from typing import Optional

import tqdm

from nl_2_fol.inference.learner import definition_model as def_model
from nl_2_fol.inference.learner.base import BaseFOL

# Allowing max_workers to equal the total CPU count is not optimal for this
# validation pipeline.
#
# Example:
# On a 120-core machine, launching 120 validation processes would assign
# roughly one core per validation task. Since many validation tasks are
# long-running and may exceed the cluster wall-time limit (48 hours),
# giving each task access to more CPU resources can improve completion time.
#
# Therefore, the number of concurrent validation processes is intentionally
# limited to 32. This reduces CPU and memory contention and leaves room for:
#   - internal multithreading used by libraries,
#   - parallelism inside validation routines,
#   - improved cache/memory efficiency,
#   - faster completion of individual validation jobs.
#
# As a starting point, using roughly 25–50% of available CPU cores as
# worker processes is recommended for large symbolic validation workloads.
MAX_PROCESSES = 32
# Also change `_MAX_VALIDATION_WORKERS` in `sample_matching_worker.py` for corresponding seperate progress bars


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
        self._file_save_idx: Optional[int] = None

    def validate(
        self,
        class_names: list[str] | None = None,
        file_save_index: Optional[int] = None,
    ):

        selected_classes = None
        if class_names is not None:
            if file_save_index is None:
                raise ValueError("Need file save index with clasess list")
            self._file_save_idx = file_save_index
            selected_classes = set()
            for class_name in class_names:
                resolved_class_name = self._validate_given_class_name(class_name)
                if resolved_class_name is not None:
                    selected_classes.add(resolved_class_name)

        classes_to_validate = [
            learned_def.name
            for _, learned_def in self._loaded_defs.learned_definitions.items()
            if learned_def.learn_success
            and learned_def.val_metrics is None
            and (selected_classes is None or learned_def.name in selected_classes)
        ]

        if len(classes_to_validate) == 0:
            print(
                "No classes to validate. All definitions are already validated or no valid class names provided."
            )
            return

        print(
            f"Starting validation for remaining {len(classes_to_validate)} classes..."
        )
        print("Classes to validate: ", classes_to_validate)

        ctx = multiprocessing.get_context("fork")
        result_queue = ctx.Queue()
        max_workers = max(
            1, min(os.cpu_count() or 1, MAX_PROCESSES, len(classes_to_validate))
        )
        print(f"Using up to {max_workers} parallel validation workers.")
        pending_classes = deque(classes_to_validate)
        active_processes = {}
        waiting_for_results = {}
        received_results = set()
        result_grace_seconds = 30.0

        with tqdm.tqdm(
            total=len(classes_to_validate), desc="Validating definitions"
        ) as progress:
            while pending_classes or active_processes or waiting_for_results:
                while pending_classes and len(active_processes) < max_workers:
                    class_name = pending_classes.popleft()
                    process = ctx.Process(
                        target=self._validate_class_worker,
                        args=(class_name, result_queue),
                    )
                    process.start()
                    active_processes[class_name] = process

                try:
                    class_name, result = result_queue.get(timeout=0.1)
                except queue.Empty:
                    pass
                else:
                    received_results.add(class_name)
                    waiting_for_results.pop(class_name, None)
                    # If the worker returned an error, terminate the worker process
                    # to avoid it continuing any background work unnecessarily.
                    status = result[0]

                    proc = active_processes.get(class_name)
                    if (
                        (status == "error" or status == "skipped")
                        and proc is not None
                        and proc.is_alive()
                    ):
                        try:
                            proc.terminate()
                            proc.join(timeout=1)
                            if proc.is_alive():
                                proc.kill()
                                proc.join()
                        except Exception:
                            pass

                    if self._apply_validation_result(result):
                        # Persist immediately so next run will skip this class
                        self._save_validated_definitions()
                    progress.update(1)

                finished_classes = []
                for class_name, process in active_processes.items():
                    process.join(timeout=0)
                    if process.is_alive():
                        continue

                    if (
                        process.exitcode not in (0, None)
                        and class_name not in received_results
                    ):
                        self.counter += 1
                        waiting_for_results.pop(class_name, None)
                        print(
                            f"Validation worker for {class_name} exited with code "
                            f"{process.exitcode} before returning a result."
                        )
                        progress.update(1)
                    elif process.exitcode == 0 and class_name not in received_results:
                        waiting_for_results.setdefault(class_name, time.monotonic())

                    finished_classes.append(class_name)

                for class_name in finished_classes:
                    active_processes.pop(class_name, None)

                for class_name, started_at in list(waiting_for_results.items()):
                    if time.monotonic() - started_at <= result_grace_seconds:
                        continue

                    self.counter += 1
                    waiting_for_results.pop(class_name, None)
                    print(
                        f"Validation worker for {class_name} did not return a result "
                        f"within {result_grace_seconds:.0f} seconds after finishing."
                    )
                    progress.update(1)

        self._save_validated_definitions()
        if self.counter > 0:
            print(
                f"Validation completed. {self.counter} definitions could not be validated due to errors during validation."
            )

    def validate_class(self, class_name: str):
        resolved_class_name = self._validate_given_class_name(class_name)
        if resolved_class_name is None:
            return

        result = self._validate_class_result(resolved_class_name)

        with open(f"{resolved_class_name}.pkl", "wb") as f:
            pickle.dump(result, f)

    def _apply_validation_result(
        self,
        result: tuple[str, int | None, def_model.DefinitionMetrics | None] | None,
    ) -> bool:
        if result is None:
            return False

        status, class_id, val_metrics = result
        if status == "error":
            return False

        if status != "ok" or class_id is None or val_metrics is None:
            return False

        self._loaded_defs.learned_definitions[class_id].val_metrics = val_metrics
        return True

    def _validate_class_worker(self, class_name: str, result_queue):
        result_queue.put((class_name, self._validate_class_result(class_name)))

    def _validate_class_result(
        self, class_name: str
    ) -> tuple[str, int | None, def_model.DefinitionMetrics | None] | None:
        try:
            chemical_class = self._c3po_slim_data.get_chemical_class_by_name(class_name)
            print(
                "-" * 10,
                f"Validating definition for ChEBI:{chemical_class.id} - {class_name}...",
                "-" * 10,
            )
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

            if (
                hasattr(learned_def, "additional_defs_used")
                and learned_def.additional_defs_used
            ):
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
        file_pattern_string = f"_with_val_file_idx_{self._file_save_idx}_"
        if file_pattern_string not in self.defs_file_path:
            base_path, extension = os.path.splitext(self.defs_file_path)
            output_path = (
                f"{base_path}{file_pattern_string}{extension}"
                if extension
                else f"{self.defs_file_path}{file_pattern_string}"
            )
        else:
            output_path = self.defs_file_path

        lock_path = f"{output_path}.lock"
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                merged_definitions = self._load_latest_validated_definitions(
                    output_path
                )
                self._merge_validation_metrics(
                    source_definitions=self._loaded_defs,
                    target_definitions=merged_definitions,
                )

                output_dir = os.path.dirname(output_path) or "."
                with tempfile.NamedTemporaryFile(
                    mode="wb", delete=False, dir=output_dir, prefix=".tmp-"
                ) as temp_file:
                    pickle.dump(merged_definitions, temp_file)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                    temp_path = temp_file.name

                os.replace(temp_path, output_path)
                self._loaded_defs = merged_definitions
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _load_latest_validated_definitions(
        self, output_path: str
    ) -> def_model.DefinitionLearningResults:
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                return pickle.load(f)
        return self._loaded_defs

    def _merge_validation_metrics(
        self,
        source_definitions: def_model.DefinitionLearningResults,
        target_definitions: def_model.DefinitionLearningResults,
    ) -> None:
        for class_id, learned_def in source_definitions.learned_definitions.items():
            if learned_def.val_metrics is None:
                continue

            if class_id not in target_definitions.learned_definitions:
                continue

            target_definitions.learned_definitions[
                class_id
            ].val_metrics = learned_def.val_metrics

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


if __name__ == "__main__":
    from nl_2_fol.inference.cli import DATA_DIR, PROJECT_DIR

    Validator = PerformValidation(
        defs_file_path=os.path.join(
            PROJECT_DIR,
            "inference",
            "learner",
            "learned",
            "claude-opus-4-6",
            "learned_definitions_a3_with_val.pkl",
        ),
        slim_dataset_path=os.path.join(DATA_DIR, "classes_slim.csv"),
        structures_path=os.path.join(DATA_DIR, "structures.csv"),
    )

    classes_to_validate = [
        learned_def.name
        for _, learned_def in Validator._loaded_defs.learned_definitions.items()
        if learned_def.learn_success and learned_def.val_metrics is None
    ]

    print(f"Classes to validate: {len(classes_to_validate)}")
    [print(f"{class_name}") for class_name in classes_to_validate]
