"""Tests for GavelFOLReasoner class."""

import pytest
from chemlog.fol_classification.model_checking import ModelCheckerOutcome
from gavel.dialects.tptp.parser import TPTPParser
from gavel.logic import logic
from rdkit import Chem

from nl_2_fol.inference.fol_reasoner.model_check_molecule import GavelFOLReasoner
from nl_2_fol.inference.learner.custom_exceptions import MissingPredicateException


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

    def test_tptp_parsing_numeric_predicate_name(self, reasoner):
        """Test that TPTP parsing handles numeric predicate names."""
        formula = "123predicate(X) <=> (p(X) & q(X))"
        # Raise error if predicate name start with a number
        with pytest.raises(Exception):
            reasoner.get_tptp_fol_definition(formula)

        formula = "predicate123(X) <=> (p(X) & q(X))"
        reasoner.get_tptp_fol_definition(formula)

        formula = "predicate(X) <=> (1p(X) & q(X))"
        with pytest.raises(Exception):
            reasoner.get_tptp_fol_definition(formula)

        formula = "predicate(X) <=> (p1(X) & q(X))"
        reasoner.get_tptp_fol_definition(formula)

    def test_tptp_parsing_success(self, reasoner):
        """Test that TPTP parsing handles invalid formulas gracefully."""
        # This test may fail in latest python versions
        # See https://github.com/gavel-tool/python-gavel/issues/25
        formula = "oligopeptide <=> peptide"
        reasoner.get_tptp_fol_definition(formula)

        formula = "failed_placeholder_predicate(X) <=> (c(X) & ~c(X))"
        reasoner.get_tptp_fol_definition(formula)

        formula = "failed_placeholder_predicate <=> failed_placeholder_predicate"
        reasoner.get_tptp_fol_definition(formula)

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

    def test_extract_unknown_predicates_respects_all_definition_sources(
        self, reasoner: GavelFOLReasoner
    ):
        """Test unknown predicates exclude base, background, and temporary defs."""
        bg_vars, bg_formula = reasoner.get_tptp_fol_definition("bg_pred(X) <=> c(X)")
        reasoner.add_background_definition("bg_pred", bg_vars, bg_formula)

        temp_vars, temp_formula = reasoner.get_tptp_fol_definition(
            "temp_pred(X) <=> o(X)"
        )
        temp_defs = {"temp_pred": (temp_vars, temp_formula)}

        _, parsed_formula = reasoner.get_tptp_fol_definition(
            "test_pred(X) <=> (c(X) & bg_pred(X) & temp_pred(X) & unknown_pred(X))"
        )

        missing = reasoner.extract_unknown_predicates(parsed_formula, temp_defs)

        assert missing == {"unknown_pred"}

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

    def test_predicate_arity_exception(self, reasoner: GavelFOLReasoner):
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

        formula_str = (
            "glycolipid <=> (glycerolipid & ?[O1, C1, O2, C2]: (o(O1) & "
            "has_0_hs(O1) & c(C1) & bSINGLE(O1, C1) & o(O2) & has_0_hs(O2) & bSINGLE(C1, O2) "
            "& c(C2) & bSINGLE(O2, C2) & has_1_hs(C1)))"
        )

        add_def = (
            "glycerolipid <=> ?[C1, C2, C3, O1, O2, O3]: (c(C1) & c(C2) & c(C3) & o(O1) "
            "& o(O2) & o(O3) & bSINGLE(C1, C2) & bSINGLE(C2, C3) & bSINGLE(C1, O1) & "
            "bSINGLE(C2, O2) & bSINGLE(C3, O3))"
        )
        _, parsed_add_def = reasoner.get_tptp_fol_definition(add_def)
        reasoner.add_background_definition("glycerolipid", [], parsed_add_def)

        mol = Chem.MolFromSmiles(
            "C([C@@H]([C@@H](/C=C/CCCCCCCCCCCCC)O)NC(CCCCCCC/C=C\\CCCCCCCC)=O)O[C@@H]1O[C@@H]([C@@H](O[C@@H]2O[C@@H]([C@H](O)[C@@H]([C@H]2O)O)CO)[C@@H]([C@H]1O)O)CO"
        )
        _, parsed_formula = reasoner.get_tptp_fol_definition(formula_str)
        reasoner.does_mol_match_tptp_definition(mol, parsed_formula)

    def test_model_checking_success(self, reasoner: GavelFOLReasoner):
        mol = Chem.MolFromSmiles(
            "C([C@@H]([C@@H](/C=C/CCCCCCCCCCCCC)O)NC(CCCCCCC/C=C\\CCCCCCCC)=O)O[C@@H]1O[C@@H]([C@@H](O[C@@H]2O[C@@H]([C@H](O)[C@@H]([C@H]2O)O)CO)[C@@H]([C@H]1O)O)CO"
        )

        formula_str = (
            "triterpenoidSaponin(X) <=> (terpeneGlycoside(X) & triterpenoid(X))"
        )
        add_defs_dict = {
            "terpeneGlycoside": "terpeneGlycoside(X) <=> (terpenoid(X) & glycoside(X))",
            "glycoside": "glycoside(X) <=> (organicMolecularEntity(X) & hasSugarMoietyAttachedByGlycosidicBond(X))",
            "organicMolecularEntity": "organicMolecularEntity(X) <=> (molecule(X) & c(X))",
            "molecule": "molecule <=> net_charge_neutral",
            "terpenoid": "terpenoid <=> molecule",
            "hasSugarMoietyAttachedByGlycosidicBond": "hasSugarMoietyAttachedByGlycosidicBond(X) <=> ?[A1, A2, A3]: (c(A1) & o(A2) & o(A3) & inRing(A1) & inRing(A2) & bSINGLE(A1, A2) & bSINGLE(A1, A3) & hasRingOxygen(A2))",
            "triterpenoid": "triterpenoid <=> (terpenoid & ?[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15]: (c(A1) & c(A2) & c(A3) & c(A4) & c(A5) & c(A6) & c(A7) & c(A8) & c(A9) & c(A10) & c(A11) & c(A12) & c(A13) & c(A14) & c(A15) & has_bond_to(A1, A2) & has_bond_to(A2, A3) & has_bond_to(A3, A4) & has_bond_to(A4, A5) & has_bond_to(A5, A6) & has_bond_to(A6, A7) & has_bond_to(A7, A8) & has_bond_to(A8, A9) & has_bond_to(A9, A10) & has_bond_to(A10, A11) & has_bond_to(A11, A12) & has_bond_to(A12, A13) & has_bond_to(A13, A14) & has_bond_to(A14, A15) & A1 != A2 & A1 != A3 & A1 != A4 & A1 != A5 & A1 != A6 & A1 != A7 & A1 != A8 & A1 != A9 & A1 != A10 & A1 != A11 & A1 != A12 & A1 != A13 & A1 != A14 & A1 != A15 & A2 != A3 & A2 != A4 & A2 != A5 & A2 != A6 & A2 != A7 & A2 != A8 & A2 != A9 & A2 != A10 & A2 != A11 & A2 != A12 & A2 != A13 & A2 != A14 & A2 != A15 & A3 != A4 & A3 != A5 & A3 != A6 & A3 != A7 & A3 != A8 & A3 != A9 & A3 != A10 & A3 != A11 & A3 != A12 & A3 != A13 & A3 != A14 & A3 != A15 & A4 != A5 & A4 != A6 & A4 != A7 & A4 != A8 & A4 != A9 & A4 != A10 & A4 != A11 & A4 != A12 & A4 != A13 & A4 != A14 & A4 != A15 & A5 != A6 & A5 != A7 & A5 != A8 & A5 != A9 & A5 != A10 & A5 != A11 & A5 != A12 & A5 != A13 & A5 != A14 & A5 != A15 & A6 != A7 & A6 != A8 & A6 != A9 & A6 != A10 & A6 != A11 & A6 != A12 & A6 != A13 & A6 != A14 & A6 != A15 & A7 != A8 & A7 != A9 & A7 != A10 & A7 != A11 & A7 != A12 & A7 != A13 & A7 != A14 & A7 != A15 & A8 != A9 & A8 != A10 & A8 != A11 & A8 != A12 & A8 != A13 & A8 != A14 & A8 != A15 & A9 != A10 & A9 != A11 & A9 != A12 & A9 != A13 & A9 != A14 & A9 != A15 & A10 != A11 & A10 != A12 & A10 != A13 & A10 != A14 & A10 != A15 & A11 != A12 & A11 != A13 & A11 != A14 & A11 != A15 & A12 != A13 & A12 != A14 & A12 != A15 & A13 != A14 & A13 != A15 & A14 != A15))",
        }
        parsed_add_def = reasoner.convert_to_background_definitions(add_defs_dict)
        for pred_name, (vars, formula) in parsed_add_def.items():
            reasoner.add_background_definition(pred_name, vars, formula)
        _, parsed_formula = reasoner.get_tptp_fol_definition(formula_str)

        with pytest.raises(
            Exception,
            match=(
                r"Predicate `triterpenoid` is defined with arity 0 but called with 1 arguments"
                r"[\s\S]*Predicate `terpenoid` is defined with arity 0 but called with 1 arguments"
                r"[\s\S]*Predicate `molecule` is defined with arity 0 but called with 1 arguments"
            ),
        ):
            reasoner.does_mol_match_tptp_definition(mol, parsed_formula)

        mol = Chem.MolFromSmiles(
            "C=1[C@@]2([C@]3(CC[C@]4([C@]([C@@]3(C=CC2=CC(C1)=O)[H])(CCC4=O)[H])C)[H])C"
        )

        formula_str = "threeOxoSteroid <=> (oxoSteroid & ?[A1, A2]: (c(A1) & o(A2) & bDOUBLE(A1, A2) & steroidPosition3(A1)))"

        add_def_dict = {
            "oxoSteroid": "oxoSteroid <=> (steroid & hasCarbonylGroup)",
            "steroidPosition3": "steroidPosition3(X) <=> (c(X) & inRing(X) & has_0_hs(X) & bDOUBLE(X, Y) & o(Y) & ?[A1, A2]: (c(A1) & c(A2) & bSINGLE(X, A1) & bSINGLE(X, A2) & inRing(A1) & inRing(A2) & A1 != A2))",
            "molecule": "molecule <=> net_charge_neutral",
            "hasCarbonylGroup": "hasCarbonylGroup <=> ?[C1, O1]: (c(C1) & o(O1) & bDOUBLE(C1, O1))",
            "steroid": "steroid <=> (molecule & ?[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17]: (c(A1) & c(A2) & c(A3) & c(A4) & c(A5) & c(A6) & c(A7) & c(A8) & c(A9) & c(A10) & c(A11) & c(A12) & c(A13) & c(A14) & c(A15) & c(A16) & c(A17) & has_bond_to(A1, A2) & has_bond_to(A2, A3) & has_bond_to(A3, A4) & has_bond_to(A4, A5) & has_bond_to(A5, A10) & has_bond_to(A10, A1) & has_bond_to(A5, A6) & has_bond_to(A6, A7) & has_bond_to(A7, A8) & has_bond_to(A8, A9) & has_bond_to(A9, A10) & has_bond_to(A8, A14) & has_bond_to(A14, A15) & has_bond_to(A15, A16) & has_bond_to(A16, A17) & has_bond_to(A17, A13) & has_bond_to(A13, A14) & has_bond_to(A9, A11) & has_bond_to(A11, A12) & has_bond_to(A12, A13)))",
        }
        cdef_dict = reasoner.convert_to_background_definitions(add_def_dict)
        for pred_name, (vars, formula) in cdef_dict.items():
            reasoner.add_background_definition(pred_name, vars, formula)

        _, parsed_formula = reasoner.get_tptp_fol_definition(formula_str)

        with pytest.raises(
            Exception,
            match=(
                r"Variable 'Y' is used in the definition of predicate 'steroidPosition3' but is not bound by predicate arguments or quantifiers"
            ),
        ):
            reasoner.does_mol_match_tptp_definition(mol, parsed_formula)

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

    def test_model_checking_success_consistency(self, reasoner: GavelFOLReasoner):
        carbonMonoxide = Chem.MolFromSmiles("[C-]#[O+]")  # CHEBI:17245
        ethanol = Chem.MolFromSmiles("CCO")
        thionitrousAcid = Chem.MolFromSmiles("SN=O")  # CHEBI:65308

        # Logical definition to match (I removed the `oneCarbonCompound` predicate for simplicity)
        definition_str = (
            "carbonMonoxide <=> ?[A1, A2]: (c(A1) & o(A2) & has_bond_to(A1,A2))"
        )
        definition_to_match = reasoner.get_tptp_fol_definition(definition_str)[1]
        matches = reasoner.does_mol_match_tptp_definition(
            carbonMonoxide, definition_to_match
        )
        assert matches == ModelCheckerOutcome.MODEL_FOUND, (
            "Expected carbon monoxide to match the definition, but it did not."
        )

        matches = reasoner.does_mol_match_tptp_definition(ethanol, definition_to_match)
        # returns model found (which contradicts the chemistry)
        assert matches == ModelCheckerOutcome.MODEL_FOUND, (
            "Expected ethanol to match the definition, but it did not. "
        )

        matches = reasoner.does_mol_match_tptp_definition(
            thionitrousAcid, definition_to_match
        )
        assert matches == ModelCheckerOutcome.NO_MODEL, (
            "Expected thionitrous acid to not match the definition, but it did."
        )

        # Logical definition to match (more accurate version - requires knowing what a oneCarbonCompound is)
        definition_str = "carbonMonoxide <=> ?[A1, A2]: (oneCarbonCompound & c(A1) & o(A2) & has_bond_to(A1,A2))"
        definition_to_match = reasoner.get_tptp_fol_definition(definition_str)[1]

        add_defs_dict = {
            "oneCarbonCompound": "oneCarbonCompound <=> ?[X]: (c(X) & ~twoPlusCarbonCompound)",
            "twoPlusCarbonCompound": "twoPlusCarbonCompound <=> ?[X, Y]: (c(X) & c(Y) & has_bond_to(X, Y) & X != Y)",
        }
        parsed_add_def = reasoner.convert_to_background_definitions(add_defs_dict)
        for pred_name, (vars, formula) in parsed_add_def.items():
            reasoner.add_background_definition(pred_name, vars, formula)

        matches = reasoner.does_mol_match_tptp_definition(
            carbonMonoxide, definition_to_match
        )
        assert matches == ModelCheckerOutcome.MODEL_FOUND, (
            "Expected carbon monoxide to match the definition, but it did not."
        )
        matches = reasoner.does_mol_match_tptp_definition(ethanol, definition_to_match)
        assert matches == ModelCheckerOutcome.NO_MODEL, (
            "Expected ethanol to not match the definition, but it did."
        )
        matches = reasoner.does_mol_match_tptp_definition(
            thionitrousAcid, definition_to_match
        )
        assert matches == ModelCheckerOutcome.NO_MODEL, (
            "Expected thionitrous acid to not match the definition, but it did."
        )

    def test_few_shots_examples_model_checking(self, reasoner: GavelFOLReasoner):
        """Test few-shot examples from nl_2_fol/prompting/prompt_templates/few_shots/with_DL_style.json"""
        # Test carboxylic acid formula
        few_shot_formula_1 = "carboxylicAcid <=> (carbonOxoacid & ?[A1, A2, A3]: (c(A1) & o(A2) & o(A3) & has_1_hs(A3) & bDOUBLE(A1, A2) & bSINGLE(A1, A3)))"

        # Add background definitions for carboxylic acid test
        add_defs_dict_1 = {
            "carbonOxoacid": "carbonOxoacid <=> ?[C1, O1]: (c(C1) & o(O1))"
        }
        parsed_add_def_1 = reasoner.convert_to_background_definitions(add_defs_dict_1)
        for pred_name, (vars, formula) in parsed_add_def_1.items():
            reasoner.add_background_definition(pred_name, vars, formula)

        # Create and test carboxylic acid molecule
        _, parsed_formula_1 = reasoner.get_tptp_fol_definition(few_shot_formula_1)

        # Test molecule that matches carboxylic acid pattern
        carboxylic_acid_mol = Chem.MolFromSmiles("CC(=O)O")  # Acetic acid
        result_1_match = reasoner.does_mol_match_tptp_definition(
            carboxylic_acid_mol, parsed_formula_1
        )
        # Verify model checking completes without error
        assert result_1_match == ModelCheckerOutcome.MODEL_FOUND, (
            "Acetic acid should match carboxylic acid formula"
        )

        # Test molecule that does NOT match carboxylic acid pattern
        ethanol_mol = Chem.MolFromSmiles(
            "CCO"
        )  # Ethanol - has C and O but not carboxylic acid structure
        result_1_no_match = reasoner.does_mol_match_tptp_definition(
            ethanol_mol, parsed_formula_1
        )
        assert result_1_no_match == ModelCheckerOutcome.NO_MODEL, (
            "Ethanol should not match carboxylic acid formula"
        )

        # Test azide formula
        few_shot_formula_2 = "azide <=> (nitrogenMolecularEntity & ?[A1, A2, A3]: (n(A1) & charge0(A1) & n(A2) & charge1(A2) & n(A3) & charge_m1(A3) & bDOUBLE(A1, A2) & bDOUBLE(A2, A3)))"

        # Add background definitions for azide test
        add_defs_dict_2 = {
            "nitrogenMolecularEntity": "nitrogenMolecularEntity <=> ?[N1]: (n(N1))"
        }
        parsed_add_def_2 = reasoner.convert_to_background_definitions(add_defs_dict_2)
        for pred_name, (vars, formula) in parsed_add_def_2.items():
            reasoner.add_background_definition(pred_name, vars, formula)

        # Create and test azide molecules
        _, parsed_formula_2 = reasoner.get_tptp_fol_definition(few_shot_formula_2)

        # Test molecule that matches azide pattern
        azide_mol = Chem.MolFromSmiles("[N-][N+]#N")  # Azide group
        result_2_match = reasoner.does_mol_match_tptp_definition(
            azide_mol, parsed_formula_2
        )
        # Verify model checking completes without error
        assert result_2_match in {
            ModelCheckerOutcome.MODEL_FOUND,
            ModelCheckerOutcome.NO_MODEL,
        }

        # Test molecule that does NOT match azide pattern
        aniline_mol = Chem.MolFromSmiles(
            "Nc1ccccc1"
        )  # Aniline - has nitrogen but not azide structure
        result_2_no_match = reasoner.does_mol_match_tptp_definition(
            aniline_mol, parsed_formula_2
        )
        assert result_2_no_match == ModelCheckerOutcome.NO_MODEL, (
            "Aniline should not match azide formula"
        )
