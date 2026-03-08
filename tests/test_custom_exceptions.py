"""Tests for custom exceptions and decorators."""

import pytest
from rdkit import Chem

from nl_2_fol.inference.custom_exceptions import (
    LearnOutOfBoxPredicateException,
    LowF1ScoreException,
    MissingPredicateException,
)
from nl_2_fol.inference.preprocessing.c3po_slim_data import ChemicalStructure


class TestMissingPredicateException:
    """Test suite for MissingPredicateException."""

    def test_missing_predicate_exception_multiple_predicates(self, capsys):
        """Test MissingPredicateException with multiple missing predicates."""
        missing = {"pred1", "pred2", "pred3"}

        with pytest.raises(MissingPredicateException) as exc_info:
            raise MissingPredicateException(missing)

        error_message = str(exc_info.value)
        assert "pred1" in error_message
        assert "pred2" in error_message
        assert "pred3" in error_message

    def test_missing_predicate_exception_stores_predicates(self):
        """Test MissingPredicateException stores missing predicates."""
        missing = {"pred1", "pred2"}
        try:
            raise MissingPredicateException(missing)
        except MissingPredicateException as e:
            assert e.missing_predicates == missing


class TestLearnOutOfBoxPredicateException:
    """Test suite for LearnOutOfBoxPredicateException."""

    def test_learn_out_of_box_mixed_definitions(self, capsys):
        """Test LearnOutOfBoxPredicateException with mixed definitions."""
        predicates = {
            "predicate1": "Has definition",
            "predicate2": None,
        }

        with pytest.raises(LearnOutOfBoxPredicateException) as exc_info:
            raise LearnOutOfBoxPredicateException(predicates)

        error_message = str(exc_info.value)
        assert "predicate1" in error_message
        assert "Has definition" in error_message
        assert "predicate2" in error_message


class TestLowF1ScoreException:
    """Test suite for LowF1ScoreException."""

    @pytest.fixture
    def sample_chemicals(self):
        """Create sample chemical structures for testing."""
        pos_samples = {
            ChemicalStructure(
                name="Benzene",
                smiles="C1=CC=CC=C1",
                mol=Chem.MolFromSmiles("C1=CC=CC=C1"),
            ),
            ChemicalStructure(
                name="Phenol",
                smiles="C1=CC=CC=C1O",
                mol=Chem.MolFromSmiles("C1=CC=CC=C1O"),
            ),
            ChemicalStructure(
                name="Toluene",
                smiles="CC1=CC=CC=C1",
                mol=Chem.MolFromSmiles("CC1=CC=CC=C1"),
            ),
        }
        neg_samples = {
            ChemicalStructure(
                name="Aniline",
                smiles="C1=CC=CC=C1N",
                mol=Chem.MolFromSmiles("C1=CC=CC=C1N"),
            ),
            ChemicalStructure(
                name="Fluorobenzene",
                smiles="C1=CC=CC=C1F",
                mol=Chem.MolFromSmiles("C1=CC=CC=C1F"),
            ),
            ChemicalStructure(
                name="Chlorobenzene",
                smiles="C1=CC=CC=C1Cl",
                mol=Chem.MolFromSmiles("C1=CC=CC=C1Cl"),
            ),
        }
        mapping = {
            "benzene": {"definition": "A colorless aromatic hydrocarbon"},
            "phenol": {"definition": "An aromatic compound with a hydroxyl group"},
            "toluene": {"definition": ""},  # Empty definition
            "aniline": {"definition": "An aromatic amine"},
            "fluorobenzene": {"definition": "A substituted benzene with fluorine"},
            "chlorobenzene": {"definition": ""},  # Empty definition
        }
        return pos_samples, neg_samples, mapping

    def test_low_f1_prioritizes_false_positives(self, sample_chemicals):
        """Test LowF1ScoreException prioritizes FP when FN < 10% and FP > 10%."""
        pos_samples, neg_samples, mapping = sample_chemicals

        # 0% FN, 50% FP (1 out of 2 neg samples matched)
        matched_neg_samples = {"C1=CC=CC=C1N", "C1=CC=CC=C1F"}
        unmatched_pos_samples = set()

        with pytest.raises(LowF1ScoreException) as exc_info:
            raise LowF1ScoreException(
                current_f1_score=0.75,
                pos_samples=pos_samples,
                neg_samples=neg_samples,
                matched_neg_samples=matched_neg_samples,
                unmatched_pos_samples=unmatched_pos_samples,
                max_examples=2,
                chebi_name_to_data_mapping=mapping,
            )

        error_message = str(exc_info.value)
        assert "False Positives (FP)" in error_message
        assert "False Negatives (FN)" not in error_message
        # Should show chemicals with definitions first
        assert "Aniline" in error_message or "Fluorobenzene" in error_message

    def test_low_f1_prioritizes_false_negatives(self, sample_chemicals):
        """Test LowF1ScoreException prioritizes FN when FP < 10% and FN > 10%."""
        pos_samples, neg_samples, mapping = sample_chemicals

        # 50% FN, 0% FP
        matched_neg_samples = set()
        unmatched_pos_samples = {"C1=CC=CC=C1", "C1=CC=CC=C1O"}

        with pytest.raises(LowF1ScoreException) as exc_info:
            raise LowF1ScoreException(
                current_f1_score=0.75,
                pos_samples=pos_samples,
                neg_samples=neg_samples,
                matched_neg_samples=matched_neg_samples,
                unmatched_pos_samples=unmatched_pos_samples,
                max_examples=2,
                chebi_name_to_data_mapping=mapping,
            )

        error_message = str(exc_info.value)
        assert "False Negatives (FN)" in error_message
        assert "False Positives (FP)" not in error_message
        # Should show chemicals with definitions first
        assert "Benzene" in error_message or "Phenol" in error_message

    def test_low_f1_shows_both_errors(self, sample_chemicals):
        """Test LowF1ScoreException shows both FP and FN when both > 10%."""
        pos_samples, neg_samples, mapping = sample_chemicals

        # 33% FN, 33% FP
        matched_neg_samples = {"C1=CC=CC=C1N"}
        unmatched_pos_samples = {"C1=CC=CC=C1"}

        with pytest.raises(LowF1ScoreException) as exc_info:
            raise LowF1ScoreException(
                current_f1_score=0.75,
                pos_samples=pos_samples,
                neg_samples=neg_samples,
                matched_neg_samples=matched_neg_samples,
                unmatched_pos_samples=unmatched_pos_samples,
                max_examples=2,
                chebi_name_to_data_mapping=mapping,
            )

        error_message = str(exc_info.value)
        assert "False Positives (FP)" in error_message
        assert "False Negatives (FN)" in error_message

    def test_low_f1_prioritizes_chemicals_with_definitions(self, sample_chemicals):
        """Test that chemicals with definitions are shown first."""
        pos_samples, neg_samples, mapping = sample_chemicals

        # Create scenario where we have chemicals with and without definitions
        matched_neg_samples = {"C1=CC=CC=C1N", "C1=CC=CC=C1F", "C1=CC=CC=C1Cl"}
        unmatched_pos_samples = set()

        with pytest.raises(LowF1ScoreException) as exc_info:
            raise LowF1ScoreException(
                current_f1_score=0.75,
                pos_samples=pos_samples,
                neg_samples=neg_samples,
                matched_neg_samples=matched_neg_samples,
                unmatched_pos_samples=unmatched_pos_samples,
                max_examples=2,
                chebi_name_to_data_mapping=mapping,
            )

        error_message = str(exc_info.value)
        # Should show chemicals with definitions (Aniline, Fluorobenzene)
        # before those without (Chlorobenzene)
        assert "Aniline" in error_message or "Fluorobenzene" in error_message
        assert "An aromatic amine" in error_message
        assert "A substituted benzene with fluorine" in error_message

    def test_low_f1_respects_max_examples(self, sample_chemicals):
        """Test that max_examples limits the number of chemicals shown."""
        pos_samples, neg_samples, mapping = sample_chemicals

        matched_neg_samples = {"C1=CC=CC=C1N", "C1=CC=CC=C1F", "C1=CC=CC=C1Cl"}
        unmatched_pos_samples = set()

        with pytest.raises(LowF1ScoreException) as exc_info:
            raise LowF1ScoreException(
                current_f1_score=0.75,
                pos_samples=pos_samples,
                neg_samples=neg_samples,
                matched_neg_samples=matched_neg_samples,
                unmatched_pos_samples=unmatched_pos_samples,
                max_examples=1,
                chebi_name_to_data_mapping=mapping,
            )

        error_message = str(exc_info.value)
        # Count the number of "Chemical Name:" occurrences
        chemical_count = error_message.count("Chemical Name:")
        assert chemical_count == 1

    def test_low_f1_fills_with_chemicals_without_definitions(self):
        """Test that remaining slots are filled with chemicals without definitions."""
        pos_samples = {
            ChemicalStructure(
                name="Chem1",
                smiles="C",
                mol=Chem.MolFromSmiles("C"),
            ),
            ChemicalStructure(
                name="Chem2",
                smiles="CC",
                mol=Chem.MolFromSmiles("CC"),
            ),
            ChemicalStructure(
                name="Chem3",
                smiles="CCC",
                mol=Chem.MolFromSmiles("CCC"),
            ),
        }
        neg_samples = set()

        unmatched_pos_samples = {"C", "CC", "CCC"}
        matched_neg_samples = set()

        # Only one chemical has a definition
        mapping = {
            "chem1": {"definition": "First chemical"},
            "chem2": {"definition": ""},
            "chem3": {"definition": ""},
        }

        with pytest.raises(LowF1ScoreException) as exc_info:
            raise LowF1ScoreException(
                current_f1_score=0.75,
                pos_samples=pos_samples,
                neg_samples=neg_samples,
                matched_neg_samples=matched_neg_samples,
                unmatched_pos_samples=unmatched_pos_samples,
                max_examples=3,
                chebi_name_to_data_mapping=mapping,
            )

        error_message = str(exc_info.value)
        # Should show Chem1 first (has definition), then fill with Chem2/Chem3
        assert "Chem1" in error_message
        assert "First chemical" in error_message
        # At least 2 more chemicals should be shown
        chemical_count = error_message.count("Chemical Name:")
        assert chemical_count == 3
