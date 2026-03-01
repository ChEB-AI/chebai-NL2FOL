"""Tests for GavelFOLReasoner class."""

import pytest
from gavel.dialects.tptp.parser import TPTPParser
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

    def test_tptp_parsing_error(self, reasoner):
        """Test that invalid TPTP formulas raise a parsing error."""
        invalid_formula = (
            "invalid_pred(X) <=> (p(X) & q(X)"  # Missing closing parenthesis
        )

        with pytest.raises(Exception):
            reasoner.get_tptp_fol_definition(invalid_formula)

    def test_get_tptp_fol_definition_simple(self, reasoner: GavelFOLReasoner):
        """Test parsing a simple FOL definition."""
        formula_str = "simple_pred(x) <=> (p(x) & q(x))"
        pred_vars, formula = reasoner.get_tptp_fol_definition(formula_str)

        assert len(pred_vars) == 1
        assert str(pred_vars[0]) == "x0"
        assert isinstance(formula, logic.QuantifiedFormula)
        assert formula.quantifier == logic.Quantifier.EXISTENTIAL

    def test_extract_predicate_variables_single(self, reasoner: GavelFOLReasoner):
        """Test extracting a single variable from predicate definition."""
        formula_str = "new_predicate(X1) <=> ?[X2]: (has_bond(X1, X2) & o(X2))"
        left_side = TPTPParser().parse(f"fof(temp, axiom, {formula_str}).").left
        variables = reasoner._extract_predicate_variables(left_side)

        assert len(variables) == 1
        assert isinstance(variables[0], logic.Variable)
        assert str(variables[0]) == "X1"

    def test_extract_predicate_variables_multiple(self, reasoner: GavelFOLReasoner):
        """Test extracting multiple variables from predicate definition."""
        formula_str = "multi_pred(X1, X2, X3) <=> (p(X1) & q(X2, X3))"
        left_side = TPTPParser().parse(f"fof(temp, axiom, {formula_str}).").left
        variables = reasoner._extract_predicate_variables(left_side)

        assert len(variables) == 3
        assert all(isinstance(v, logic.Variable) for v in variables)
        assert [str(v) for v in variables] == ["X1", "X2", "X3"]

    def test_extract_predicate_variables_none(self, reasoner: GavelFOLReasoner):
        """Test extracting variables from a predicate with no arguments."""
        formula_str = "nullary_pred <=> (p & q)"
        left_side = TPTPParser().parse(f"fof(temp, axiom, {formula_str}).").left
        variables = reasoner._extract_predicate_variables(left_side)

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
        _, parsed_formula = reasoner.get_tptp_fol_definition(formula_str)

        predicates = reasoner._extract_predicates(parsed_formula)

        assert "p" in predicates
        assert "q" in predicates
        assert "r" in predicates

    def test_missing_predicate_detection(self, reasoner: GavelFOLReasoner):
        """Test that missing predicates are detected."""
        # This formula references an undefined predicate
        formula_str = "test_pred(X) <=> undefined_pred(X)"
        _, parsed_formula = reasoner.get_tptp_fol_definition(formula_str)

        # Create a simple molecule
        mol = Chem.MolFromSmiles("C")

        # Should raise MissingPredicateException
        with pytest.raises(MissingPredicateException):
            reasoner.does_mol_match_tptp_definition(mol, parsed_formula)

    def test_parsing_step_2_quantified_formula_error(self, reasoner: GavelFOLReasoner):
        formula_str = "test_pred(X) <=> X"  # Bare variable (may or may not work)

        with pytest.raises(Exception) as exc_info:
            reasoner.get_tptp_fol_definition(formula_str)

        error_message = str(exc_info.value)
        assert "PARSING STEP 2/3 FAILED" in error_message, (
            f"Expected PARSING STEP 2/3 error, but got: {error_message[:200]}"
        )
        assert "Error wrapping parsed formula in QuantifiedFormula" in error_message

    def test_parsing_step_3_normalization_error(self, reasoner: GavelFOLReasoner):
        formula_str = (
            "test_pred(X) <=> ?[Y]: ![Z]: ?[W]: ![V]: (p(Y) & q(Z) & r(W) & s(V))"
        )

        with pytest.raises(Exception) as exc_info:
            reasoner.get_tptp_fol_definition(formula_str)

        error_message = str(exc_info.value)
        assert "PARSING STEP 3/3 FAILED" in error_message, (
            f"Expected PARSING STEP 3/3 error, but got: {error_message[:200]}"
        )
        assert "Error normalizing formula to PNF (Prenex Normal Form)" in error_message
        assert "Formula before normalization:" in error_message

    def test_missing_predicate_exception_raised(self, reasoner: GavelFOLReasoner):
        """Test that errors in does_mol_match_tptp_definition are properly raised."""
        formula_str = "test_pred(X) <=> (ptest(X) & qtest(X))"
        _, parsed_formula = reasoner.get_tptp_fol_definition(formula_str)

        # Create a simple molecule
        mol = Chem.MolFromSmiles("C")

        # Should raise MissingPredicateException since ptest and qtest are not defined
        with pytest.raises(MissingPredicateException) as exc_info:
            reasoner.does_mol_match_tptp_definition(mol, parsed_formula)

        assert exc_info.value.missing_predicates == {"ptest", "qtest"}
        error_message = str(exc_info.value)
        assert "ptest" in error_message
        assert "qtest" in error_message

    def test_does_mol_match_tptp_definition_exception(self, reasoner: GavelFOLReasoner):
        """Test that exceptions in does_mol_match_tptp_definition are properly raised."""
        cdef = reasoner.convert_to_background_definitions(
            {"ptest": "ptest <=> has_bond(X, Y)"}  # ptest has no variables
        )
        reasoner.add_background_definition("ptest", cdef["ptest"][0], cdef["ptest"][1])

        # Here, the formula reference ptest predicate with a variable,
        # but the background definition of ptest has no variables,
        # which should cause an error during model checking
        formula_str = "test_pred(X) <=> (ptest(X))"

        _, parsed_formula = reasoner.get_tptp_fol_definition(formula_str)

        # Create a simple molecule
        mol = Chem.MolFromSmiles("C")

        with pytest.raises(Exception) as exc_info:
            reasoner.does_mol_match_tptp_definition(mol, parsed_formula)

        error_message = str(exc_info.value)
        assert (
            "Predicate `ptest` is defined with arity 0 but called with 1 arguments"
            in error_message
        )
