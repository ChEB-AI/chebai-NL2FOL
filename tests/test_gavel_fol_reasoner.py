"""Tests for GavelFOLReasoner class."""

import pytest
from gavel.dialects.tptp.parser import TPTPParser
from gavel.logic import logic
from rdkit import Chem

import nl_2_fol.inference.model_check_molecule as model_check_molecule
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

    def test_tptp_parsing(self, reasoner):
        """Test that TPTP parsing handles invalid formulas gracefully."""
        # This test may fail in latest python versions
        # See https://github.com/gavel-tool/python-gavel/issues/25
        invalid_formula = "oligopeptide <=> peptide"
        reasoner.get_tptp_fol_definition(invalid_formula)

    def test_get_tptp_fol_definition_simple(self, reasoner: GavelFOLReasoner):
        """Test parsing a simple FOL definition."""
        formula_str = "simple_pred(x) <=> (p(x) & q(x))"
        pred_vars, formula = reasoner.get_tptp_fol_definition(formula_str)

        assert len(pred_vars) == 0
        assert isinstance(formula, logic.QuantifiedFormula)
        assert formula.quantifier == logic.Quantifier.EXISTENTIAL

    def test_extract_predicate_variables_single(self, reasoner: GavelFOLReasoner):
        """Test extracting a single variable from predicate definition."""
        formula_str = "new_predicate(X1) <=> ?[X2]: (has_bond(X1, X2) & o(X2))"
        tptp = TPTPParser().parse(f"fof(temp, axiom, {formula_str}).")[0].formula
        variables = reasoner._extract_predicate_variables(tptp.left)

        assert len(variables) == 1
        assert isinstance(variables[0], logic.Variable)
        assert str(variables[0]) == "X1"

    def test_extract_predicate_variables_multiple(self, reasoner: GavelFOLReasoner):
        """Test extracting multiple variables from predicate definition."""
        formula_str = "multi_pred(X1, X2, X3) <=> (p(X1) & q(X2, X3))"
        tptp = TPTPParser().parse(f"fof(temp, axiom, {formula_str}).")[0].formula
        variables = reasoner._extract_predicate_variables(tptp.left)

        assert len(variables) == 3
        assert all(isinstance(v, logic.Variable) for v in variables)
        assert [str(v) for v in variables] == ["X1", "X2", "X3"]

    def test_extract_predicate_variables_none(self, reasoner: GavelFOLReasoner):
        """Test extracting variables from a predicate with no arguments."""
        formula_str = "nullary_pred <=> (p & q)"
        tptp = TPTPParser().parse(f"fof(temp, axiom, {formula_str}).")[0].formula
        variables = reasoner._extract_predicate_variables(tptp.left)

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

    # TODO: FIXME
    @pytest.mark.xfail(
        reason="This test is expected to fail due to a known issue in the normalization step of the parser."
    )
    def test_parsing_normalization_error(self, reasoner: GavelFOLReasoner):
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

    def predicate_arity_exception(self, reasoner: GavelFOLReasoner):
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

    def test_model_checking(self, reasoner: GavelFOLReasoner):
        """Test that exceptions in does_mol_match_tptp_definition are properly raised."""
        formula_str = "cation <=> net_charge_positive"

        _, parsed_formula = reasoner.get_tptp_fol_definition(formula_str)

        mol = Chem.MolFromSmiles(
            "C(=O)(C1=CC=C(C=C1F)OCCCCCC[NH+](CC=C)C)C=2C=CC(=CC2)Br"
        )

        reasoner.does_mol_match_tptp_definition(mol, parsed_formula)

    def test_timeout_returns_false_before_threshold(
        self, reasoner: GavelFOLReasoner, monkeypatch
    ):
        """Test timeout handling returns False before max timeout threshold is reached."""

        class _TimeoutModelChecker:
            def __init__(self, *args, **kwargs):
                pass

            def find_model(self, *args, **kwargs):
                return model_check_molecule.ModelCheckerOutcome.TIMEOUT, None

        monkeypatch.setattr(
            "nl_2_fol.inference.model_check_molecule.ModelChecker",
            _TimeoutModelChecker,
        )
        monkeypatch.setattr(reasoner, "_mol_to_fol", lambda _mol: ([], {}))

        _, parsed_formula = reasoner.get_tptp_fol_definition("test_pred(X) <=> c(X)")
        mol = Chem.MolFromSmiles("C")

        calls_before_threshold = reasoner._MAX_ALLOWED_TIMEOUTS_PER_FORMULA - 1
        for _ in range(calls_before_threshold):
            assert reasoner.does_mol_match_tptp_definition(mol, parsed_formula) is False

        assert reasoner._timout_tracking[str(parsed_formula)] == calls_before_threshold

    def test_timeout_raises_at_threshold(self, reasoner: GavelFOLReasoner, monkeypatch):
        """Test timeout handling raises once max timeout threshold is reached."""

        class _TimeoutModelChecker:
            def __init__(self, *args, **kwargs):
                pass

            def find_model(self, *args, **kwargs):
                return model_check_molecule.ModelCheckerOutcome.TIMEOUT, None

        monkeypatch.setattr(
            "nl_2_fol.inference.model_check_molecule.ModelChecker",
            _TimeoutModelChecker,
        )
        monkeypatch.setattr(reasoner, "_mol_to_fol", lambda _mol: ([], {}))

        _, parsed_formula = reasoner.get_tptp_fol_definition("test_pred(X) <=> c(X)")
        mol = Chem.MolFromSmiles("C")

        for _ in range(reasoner._MAX_ALLOWED_TIMEOUTS_PER_FORMULA - 1):
            assert reasoner.does_mol_match_tptp_definition(mol, parsed_formula) is False

        with pytest.raises(Exception) as exc_info:
            reasoner.does_mol_match_tptp_definition(mol, parsed_formula)

        error_message = str(exc_info.value)
        assert "MODEL CHECKING FAILED" in error_message
        assert "Generated FOL formula took more than 30 seconds" in error_message

    def test_ambiguous_formula_without_brackets(self, reasoner: GavelFOLReasoner):
        """Test that ambiguous formulas without proper brackets raise an error."""
        # Formula without brackets: A <=> B & C is parsed as (A <=> B) & C, not A <=> (B & C)
        ambiguous_formula = "glycerolipid(x) <=> lipid(x) & ?[C1, C2]: (c(C1) & c(C2))"

        with pytest.raises(Exception) as exc_info:
            reasoner.get_tptp_fol_definition(ambiguous_formula)

        error_message = str(exc_info.value)
        assert "Invalid FOL formula structure" in error_message
        assert (
            "left-hand side of biimplication must be a predicate expression"
            in error_message
        )
        assert "ambiguous" in error_message.lower()

    def test_properly_bracketed_formula(self, reasoner: GavelFOLReasoner):
        """Test that properly bracketed formulas parse successfully."""
        # Formula with proper brackets: A <=> (B & C)
        properly_bracketed_formula = (
            "glycerolipid(x) <=> (lipid(x) & ?[C1, C2]: (c(C1) & c(C2)))"
        )

        pred_vars, parsed_formula = reasoner.get_tptp_fol_definition(
            properly_bracketed_formula
        )

        assert isinstance(parsed_formula, logic.QuantifiedFormula)
        assert len(pred_vars) == 0
