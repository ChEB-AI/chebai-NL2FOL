import queue

from gavel.logic import logic

from nl_2_fol.inference.fol_reasoner import GavelFOLReasoner
from nl_2_fol.inference.learner import custom_exceptions as ce
from nl_2_fol.inference.learner import definition_model as def_model
from nl_2_fol.inference.learner.sample_matching_worker import (
    check_if_definition_matches_samples,
    check_if_definition_matches_samples_clingo,
)
from nl_2_fol.inference.preprocessing import c3po_slim_data as dm
from nl_2_fol.inference.utils.to_camel_case import to_camel_case

from typing import Literal


class BaseFOL:
    _DEFINITION_FILE_NAME = "learned_definitions_a{max_attempts}.pkl"
    _MAX_NEGATIVE_SAMPLES = 1000
    _SAMPLE_MATCH_TIMEOUT_SECONDS = 10 * 60  # 10 minutes

    def __init__(
        self,
        slim_dataset_path: str,
        structures_path: str,
        chebi_version: int,
        split: Literal["train", "val"],
        fol_reasoner: Literal["gavel", "mistral", "asp"] = "gavel",
    ):
        self.slim_dataset_path = slim_dataset_path
        self.structures_path = structures_path
        self.chebi_version = chebi_version
        if split not in {"train", "val"}:
            raise ValueError(f"Invalid split: {split}")
        self.split = split
        self.fol_reasoner = fol_reasoner
        self._c3po_slim_data, self._entire_chebi_data = dm.load_c3po_slim_dataset(
            slim_dataset_path=self.slim_dataset_path,
            structures_path=self.structures_path,
            chebi_version=self.chebi_version,
            split=self.split,
        )
        self.undirected_chebi_graph = (
            self._entire_chebi_data.get_undirected_hierarchy_graph()
        )
        self._fol_reasoner = self.get_reasoner()

    def get_reasoner(self):
        if self.fol_reasoner == "gavel":
            print("Using `GavelFOLReasoner` as the FOL reasoner.")
            return GavelFOLReasoner()
        elif self.fol_reasoner == "mistral":
            print("Using `MistralCustomFOLReasoner` as the FOL reasoner.")
            # return MistralCustomFOLReasoner()
            raise ValueError("Support revoked")
        elif self.fol_reasoner == "asp":
            print("Using `ASPModelChecker` as the FOL reasoner.")
            from nl_2_fol.inference.fol_reasoner.asp_model_checker import (
                ASPModelChecker,
            )

            return ASPModelChecker()
        else:
            raise ValueError(f"Unsupported FOL reasoner: {self.fol_reasoner}")

    def _score_definition(
        self,
        *,
        chemical_class: dm.ChemicalClass,
        parsed_def: logic.QuantifiedFormula | str,
        sample_match_timeout_seconds: int | None,
        max_neg_samples: int,
        temp_additional_defs: dict[str, def_model.FOLDefinition] | None,
    ) -> tuple[
        def_model.DefinitionMetrics,
        set[dm.SMILES_STRING],
        set[dm.SMILES_STRING],
        set[dm.ChemicalStructure],
        set[dm.ChemicalStructure],
    ]:
        pos_samples, neg_samples = self._get_positive_and_negative_samples(
            chemical_class,
            max_neg_samples,
        )

        matching_func = (
            check_if_definition_matches_samples
            if self.fol_reasoner in ["gavel", "mistral"]
            else check_if_definition_matches_samples_clingo
        )

        match_result_dict, processed_samples_dict = matching_func(
            self._fol_reasoner,
            sample_match_timeout_seconds,
            chemical_class,
            parsed_def,
            pos_samples,
            neg_samples,
            temp_additional_defs,
            split=self.split,
        )

        metrics = self._get_metrics(match_result_dict)

        return (
            metrics,
            match_result_dict["unmatched_pos_samples"],
            match_result_dict["matched_neg_samples"],
            processed_samples_dict["processed_pos_samples"],
            processed_samples_dict["processed_neg_samples"],
        )

    @ce.stop_program_upon_failure
    def _get_positive_and_negative_samples(
        self, chemical_class: dm.ChemicalClass, max_neg_samples: int = 1000
    ) -> tuple[set[dm.ChemicalStructure], set[dm.ChemicalStructure]]:
        # validation examples already substracted during from positive examples
        positive_examples = chemical_class.all_positive_examples
        positive_instances = {
            self._c3po_slim_data.smiles_to_instance[smiles]
            for smiles in positive_examples
            if smiles in self._c3po_slim_data.smiles_to_instance
        }
        negative_examples = list(self._c3po_slim_data.all_smiles - positive_examples)

        if self.split == "train":
            # Closest 1000 negative samples are used for training to speed up training.
            # For validation, we use all negative samples to fairly compare against the c3po.
            negative_examples = self._get_closest_negatives(
                negative_examples, chemical_class.id, n_samples=max_neg_samples
            )
        negative_instances = {
            self._c3po_slim_data.smiles_to_instance[smiles]
            for smiles in negative_examples
            if smiles in self._c3po_slim_data.smiles_to_instance
        }
        assert len(positive_instances) > 0, (
            f"No positive samples found for {chemical_class.name}"
        )
        assert len(negative_instances) > 0, (
            f"No negative samples found for {chemical_class.name}"
        )

        return positive_instances, negative_instances

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
                    if str(neighbor) in self._c3po_slim_data.id_to_class_name:
                        for smiles in self._c3po_slim_data.classes[
                            self._c3po_slim_data.id_to_class_name[str(neighbor)]
                        ].all_positive_examples:
                            if smiles in available_smiles:
                                selected_smiles.add(smiles)
                            if len(selected_smiles) >= n_samples:
                                return list(selected_smiles)

        return list(selected_smiles)

    @ce.stop_program_upon_failure
    def _get_metrics(
        self,
        cm: dict[str, set[dm.SMILES_STRING]],
    ) -> def_model.DefinitionMetrics:
        num_true_positives = len(cm["matched_pos_samples"])  # TPs
        num_false_negatives = len(cm["unmatched_pos_samples"])  # FNs
        num_false_positives = len(cm["matched_neg_samples"])  # FPs
        num_true_negatives = len(cm["unmatched_neg_samples"])  # TNs

        def safe_divide(numerator: float, denominator: float) -> float:
            return numerator / denominator if denominator > 0 else 0.0

        # Guard against edge cases where no positive predictions are made.
        f1 = safe_divide(
            2 * num_true_positives,
            2 * num_true_positives + num_false_positives + num_false_negatives,
        )
        ppv = safe_divide(
            num_true_positives,
            num_true_positives + num_false_positives,
        )
        npv = safe_divide(
            num_true_negatives,
            num_true_negatives + num_false_negatives,
        )
        return def_model.DefinitionMetrics(
            F1=f1,
            PPV=ppv,
            NPV=npv,
            TP=num_true_positives,
            FP=num_false_positives,
            FN=num_false_negatives,
            TN=num_true_negatives,
            # Populate extended outcomes from confusion matrix
            inferred_match_pos=len(cm.get("inferred_match_pos", set())),
            inferred_match_neg=len(cm.get("inferred_match_neg", set())),
            inferred_no_match_pos=len(cm.get("inferred_no_match_pos", set())),
            inferred_no_match_neg=len(cm.get("inferred_no_match_neg", set())),
            timeout_pos=len(cm.get("timeout_pos", set())),
            timeout_neg=len(cm.get("timeout_neg", set())),
            error_pos=len(cm.get("error_pos", set())),
            error_neg=len(cm.get("error_neg", set())),
            unknown_pos=len(cm.get("unknown_pos", set())),
            unknown_neg=len(cm.get("unknown_neg", set())),
            fp_smiles=list(cm.get("matched_neg_samples", set())),
            fn_smiles=list(cm.get("unmatched_pos_samples", set())),
            timeout_smiles=list(
                cm.get("timeout_pos", set()).union(cm.get("timeout_neg", set()))
            ),
        )

    def _validate_given_class_name(self, class_name: str) -> None | str:
        resolved_class_name = class_name
        if resolved_class_name not in self._c3po_slim_data.classes:
            camel_cased_class_name = to_camel_case(class_name)
            print(
                f"Class name `{class_name}` was not found directly in the dataset. "
                f"Trying camel-cased variant `{camel_cased_class_name}`."
            )
            resolved_class_name = camel_cased_class_name
        if resolved_class_name not in self._c3po_slim_data.classes:
            print(f"{class_name} not found in the dataset.")
            return None
        return resolved_class_name
