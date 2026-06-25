"""Tests for NL2FOLChebiClassifier."""

import pickle

import pytest

from nl_2_fol.classifier.classify import NL2FOLChebiClassifier
from nl_2_fol.inference.fol_reasoner import ChemlogModelChecker
from nl_2_fol.inference.learner import definition_model as def_model


@pytest.fixture(scope="module")
def sample_learning_results():
    """Create a sample DefinitionLearningResults object for testing."""
    gavel = ChemlogModelChecker()
    pred_vars_carbon, formula_carbon = gavel.get_tptp_fol_definition(
        "carbonCompound <=> ?[X]: (c(X))"
    )

    metrics = def_model.DefinitionMetrics(
        TP=10, FP=2, FN=3, TN=85, F1=0.83, PPV=0.83, NPV=0.97
    )

    learned_def = def_model.LearnedDefinition(
        train_metrics=metrics,
        learned_FOL=def_model.FOLFormula(
            formula=formula_carbon,
            pred_variables=pred_vars_carbon,
        ),
        prompts_history={},
        name="carbon compound",
        definition="A compound containing at least one carbon atom.",
        learn_success=True,
    )

    failed_def = def_model.LearnedDefinition(
        train_metrics=metrics,
        learned_FOL=def_model.FOLFormula(
            formula=formula_carbon,
            pred_variables=pred_vars_carbon,
        ),
        prompts_history={},
        name="failed compound",
        definition="A compound that failed to learn.",
        learn_success=False,
    )

    return def_model.DefinitionLearningResults(
        learned_definitions={
            12345: learned_def,
            99999: failed_def,
        },
        additional_definitions={},
    )


@pytest.fixture
def definitions_file(sample_learning_results, tmp_path):
    """Write sample_learning_results to a temporary pickle file."""
    defs_file = tmp_path / "test_definitions.pkl"
    with open(defs_file, "wb") as f:
        pickle.dump(sample_learning_results, f)
    return defs_file


@pytest.fixture
def classifier(definitions_file):
    """Create a NL2FOLChebiClassifier backed by the temporary definitions file."""
    return NL2FOLChebiClassifier(str(definitions_file))


class TestNL2FOLChebiClassifier:
    """Test suite for NL2FOLChebiClassifier."""

    def test_classify_smiles_invalid_smiles_returns_none(self, classifier):
        """Invalid SMILES should return {smiles: None}."""
        smiles = "not_a_valid_smiles!!!"
        result = classifier.classify_smiles(smiles)

        assert smiles in result
        assert result[smiles] is None

    def test_classify_smiles_invalid_smiles_result_schema(self, classifier):
        """Result for an invalid SMILES must use the SMILES string as the key."""
        smiles = "INVALID"
        result = classifier.classify_smiles(smiles)

        assert len(result) == 1
        assert list(result.keys())[0] == smiles

    def test_classify_smiles_valid_no_match(self, classifier):
        """Valid SMILES with no matching class should return {smiles: []}."""
        smiles = "O"  # Water — no carbon atoms
        result = classifier.classify_smiles(smiles)

        assert smiles in result
        assert result[smiles] == []

    def test_classify_smiles_valid_with_match(self, classifier):
        """Valid SMILES matching a class should return the correct classification."""
        smiles = "C"  # Methane — has a carbon atom
        result = classifier.classify_smiles(smiles)

        assert smiles in result
        classifications = result[smiles]
        assert isinstance(classifications, list)
        assert len(classifications) == 1
        assert classifications[0]["chebi_id"] == 12345
        assert classifications[0]["name"] == "carbon compound"

    def test_classify_smiles_result_always_has_smiles_key(self, classifier):
        """Result schema must always have the SMILES string as the top-level key."""
        for smiles in ["C", "O", "not_valid"]:
            result = classifier.classify_smiles(smiles)
            assert smiles in result
            assert len(result) == 1

    def test_classify_smiles_list_returns_list(self, classifier):
        """classify_smiles_list should return a list of per-SMILES result dicts."""
        smiles_list = ["C", "O", "invalid"]
        results = classifier.classify_smiles_list(smiles_list)

        assert isinstance(results, list)
        assert len(results) == len(smiles_list)

    def test_classify_smiles_list_schema(self, classifier):
        """Every entry in classify_smiles_list output has the SMILES as its key."""
        smiles_list = ["C", "O", "invalid"]
        results = classifier.classify_smiles_list(smiles_list)

        for smiles, result in zip(smiles_list, results):
            assert smiles in result

    def test_classify_smiles_list_invalid_entry(self, classifier):
        """Invalid SMILES inside a list should yield {smiles: None}."""
        smiles_list = ["invalid_smiles"]
        results = classifier.classify_smiles_list(smiles_list)

        assert len(results) == 1
        assert results[0]["invalid_smiles"] is None

    def test_failed_learn_success_not_loaded(self, sample_learning_results, tmp_path):
        """Definitions with learn_success=False must not appear in class_definitions."""
        defs_file = tmp_path / "test_defs_failed.pkl"
        with open(defs_file, "wb") as f:
            pickle.dump(sample_learning_results, f)

        c = NL2FOLChebiClassifier(str(defs_file))
        # 99999 has learn_success=False and must be excluded
        assert 99999 not in c.class_definitions
        # 12345 has learn_success=True and must be present
        assert 12345 in c.class_definitions

    def test_file_not_found_raises(self, tmp_path):
        """FileNotFoundError should be raised when the definitions file is missing."""
        non_existent = str(tmp_path / "nonexistent.pkl")

        with pytest.raises(FileNotFoundError):
            NL2FOLChebiClassifier(non_existent)
