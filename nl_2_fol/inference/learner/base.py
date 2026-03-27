import queue

from gavel.logic import logic

from nl_2_fol.inference.fol_reasoner.model_check_molecule import GavelFOLReasoner
from nl_2_fol.inference.learner import custom_exceptions as ce
from nl_2_fol.inference.learner import definition_model as def_model
from nl_2_fol.inference.learner.sample_matching_worker import (
    check_if_definition_matches_samples,
)
from nl_2_fol.inference.preprocessing import c3po_slim_data as dm


class BaseFOL:
    _DEFINITION_FILE_NAME = "learned_definitions_a{max_attempts}.pkl"
    _MAX_NEGATIVE_SAMPLES = 1000
    _SAMPLE_MATCH_TIMEOUT_SECONDS = 10 * 60  # 10 minutes

    def __init__(
        self,
        slim_dataset_path: str,
        structures_path: str,
        chebi_version: int,
        split: str,
    ):
        self.slim_dataset_path = slim_dataset_path
        self.structures_path = structures_path
        self.chebi_version = chebi_version
        self.split = split
        self._c3po_slim_data, self._entire_chebi_data = dm.load_c3po_slim_dataset(
            slim_dataset_path=self.slim_dataset_path,
            structures_path=self.structures_path,
            chebi_version=self.chebi_version,
            split=self.split,
        )
        self.undirected_chebi_graph = (
            self._entire_chebi_data.get_undirected_hierarchy_graph()
        )
        self._gavel = GavelFOLReasoner()

    def _score_definition(
        self,
        *,
        chemical_class: dm.ChemicalClass,
        tptp_def: logic.QuantifiedFormula,
        sample_match_timeout_seconds: int,
        max_neg_samples: int,
        temp_additional_defs: dict[
            str, tuple[list[logic.Variable], logic.QuantifiedFormula]
        ]
        | None,
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

        (
            unmatched_pos_samples,
            matched_neg_samples,
            processed_pos_samples,
            processed_neg_samples,
        ) = check_if_definition_matches_samples(
            self._gavel,
            sample_match_timeout_seconds,
            chemical_class,
            tptp_def,
            pos_samples,
            neg_samples,
            temp_additional_defs,
        )
        metrics = self._get_metrics(
            unmatched_pos_samples,
            matched_neg_samples,
            processed_pos_samples,
            processed_neg_samples,
        )
        return (
            metrics,
            unmatched_pos_samples,
            matched_neg_samples,
            processed_pos_samples,
            processed_neg_samples,
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
        unmatched_pos_samples: set[dm.SMILES_STRING],
        matched_neg_samples: set[dm.SMILES_STRING],
        processed_pos_samples: set[dm.ChemicalStructure],
        processed_neg_samples: set[dm.ChemicalStructure],
    ) -> def_model.DefinitionMetrics:
        num_true_positives = len(processed_pos_samples) - len(unmatched_pos_samples)
        num_false_negatives = len(unmatched_pos_samples)
        num_false_positives = len(matched_neg_samples)
        num_true_negatives = len(processed_neg_samples) - len(matched_neg_samples)

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
        )
