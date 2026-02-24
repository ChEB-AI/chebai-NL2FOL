"""Tests for GavelFOLReasoner class."""

import pytest
from gavel.logic import logic
from rdkit import Chem

from nl_2_fol.inference.custom_exceptions import MissingPredicateException
from nl_2_fol.inference.model_check_molecule import GavelFOLReasoner


class TestGavelFOLReasoner:
    """Test suite for GavelFOLReasoner."""

    @pytest.fixture
    def reasoner(self):
        """Create a GavelFOLReasoner instance for testing."""
        return GavelFOLReasoner()

    def test_initialization(self, reasoner):
        """Test that GavelFOLReasoner initializes correctly."""
        assert reasoner._tptp_parser is not None
        assert reasoner._base_predicates is not None
        assert reasoner.background_definitions == {}

    def test_get_tptp_fol_definition_simple(self, reasoner: GavelFOLReasoner):
        """Test parsing a simple FOL definition."""
        formula_str = "simple_pred(x) <=> (p(x) & q(x))"
        result = reasoner.get_tptp_fol_definition(formula_str)

        assert isinstance(result, logic.QuantifiedFormula)
        assert result.quantifier == logic.Quantifier.EXISTENTIAL

    def test_extract_predicate_variables_single(self, reasoner: GavelFOLReasoner):
        """Test extracting a single variable from predicate definition."""
        formula_str = "new_predicate(X1) <=> ?[X2]: (has_bond(X1, X2) & o(X2))"
        variables = reasoner._extract_predicate_variables(formula_str)

        assert len(variables) == 1
        assert isinstance(variables[0], logic.Variable)
        assert str(variables[0]) == "X1"

    def test_extract_predicate_variables_multiple(self, reasoner: GavelFOLReasoner):
        """Test extracting multiple variables from predicate definition."""
        formula_str = "multi_pred(X1, X2, X3) <=> (p(X1) & q(X2, X3))"
        variables = reasoner._extract_predicate_variables(formula_str)

        assert len(variables) == 3
        assert all(isinstance(v, logic.Variable) for v in variables)
        assert [str(v) for v in variables] == ["X1", "X2", "X3"]

    def test_extract_predicate_variables_none(self, reasoner: GavelFOLReasoner):
        """Test extracting variables from a predicate with no arguments."""
        formula_str = "nullary_pred <=> (p & q)"
        variables = reasoner._extract_predicate_variables(formula_str)

        assert len(variables) == 0

    def test_convert_to_background_definitions(self, reasoner: GavelFOLReasoner):
        """Test converting string definitions to background definition format."""
        predicates = {
            "new_pred": "new_pred(X1) <=> ?[X2]: (has_bond(X1, X2) & o(X2))",
            "multi_pred": "multi_pred(X1, X2) <=> (p(X1) & q(X2))",
        }

        result = reasoner.convert_to_background_definitions(predicates)

        assert "new_pred" in result
        assert "multi_pred" in result

        # Check new_pred
        new_pred_vars, new_pred_formula = result["new_pred"]
        assert len(new_pred_vars) == 1
        assert str(new_pred_vars[0]) == "X1"
        assert isinstance(new_pred_formula, logic.QuantifiedFormula)

        # Check multi_pred
        multi_pred_vars, multi_pred_formula = result["multi_pred"]
        assert len(multi_pred_vars) == 2
        assert [str(v) for v in multi_pred_vars] == ["X1", "X2"]
        assert isinstance(multi_pred_formula, logic.QuantifiedFormula)

    def test_extract_predicates_from_formula(self, reasoner: GavelFOLReasoner):
        """Test extracting all predicates from a formula."""
        formula_str = "test_pred(X) <=> (p(X) & q(X) & r(X))"
        parsed_formula = reasoner.get_tptp_fol_definition(formula_str)

        predicates = reasoner._extract_predicates(parsed_formula)

        assert "p" in predicates
        assert "q" in predicates
        assert "r" in predicates

    def test_add_background_definition(self, reasoner: GavelFOLReasoner):
        """Test adding a background definition."""
        formula_str = "test_pred(X) <=> (p(X) & q(X))"
        parsed = reasoner.get_tptp_fol_definition(formula_str)

        reasoner.add_background_definition("test_pred", parsed)

        assert "test_pred" in reasoner.background_definitions
        assert reasoner.background_definitions["test_pred"][1] == parsed

    def test_missing_predicate_detection(self, reasoner: GavelFOLReasoner):
        """Test that missing predicates are detected."""
        # This formula references an undefined predicate
        formula_str = "test_pred(X) <=> undefined_pred(X)"
        parsed_formula = reasoner.get_tptp_fol_definition(formula_str)

        # Create a simple molecule
        mol = Chem.MolFromSmiles("C")

        # Should raise MissingPredicateException
        with pytest.raises(MissingPredicateException):
            reasoner.does_mol_match_tptp_definition(mol, parsed_formula)
