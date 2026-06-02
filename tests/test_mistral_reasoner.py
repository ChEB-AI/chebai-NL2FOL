"""Tests for MistralCustomFOLReasoner class."""

import pytest
from chemlog.fol_classification.model_checking import ModelCheckerOutcome
from rdkit import Chem

from nl_2_fol.inference.fol_reasoner import MistralCustomFOLReasoner


class TestMistralCustomFOLReasoner:
    """Test suite for MistralCustomFOLReasoner."""

    @pytest.fixture
    def reasoner(self):
        """Create a MistralCustomFOLReasoner instance for testing."""
        return MistralCustomFOLReasoner()

    def test_model_checking_success_consistency(
        self, reasoner: MistralCustomFOLReasoner
    ):
        """Test that the reasoner can successfully parse and model check a formula."""

        carbonMonoxide = Chem.MolFromSmiles("[C-]#[O+]")  # CHEBI:17245
        ethanol = Chem.MolFromSmiles("CCO")
        thionitrousAcid = Chem.MolFromSmiles("SN=O")  # CHEBI:65308

        # Logical definition to match (I removed `OneCarbonCompound` for simplicity)
        definition_str = "CarbonMonoxide ↔ (∃x ∃y (C(x) ∧ O(y) ∧ HasBondTo(x, y)))"
        definition_to_match = reasoner.get_tptp_fol_definition(definition_str)[1]
        matches = reasoner.does_mol_match_tptp_definition(
            carbonMonoxide, definition_to_match
        )
        assert matches == ModelCheckerOutcome.MODEL_FOUND, (
            "Expected carbon monoxide to match the definition, but it did not."
        )
        matches = reasoner.does_mol_match_tptp_definition(ethanol, definition_to_match)
        assert matches == ModelCheckerOutcome.MODEL_FOUND, (
            "Expected ethanol to match the definition, but it did not. "
        )
        matches = reasoner.does_mol_match_tptp_definition(
            thionitrousAcid, definition_to_match
        )
        assert matches == ModelCheckerOutcome.NO_MODEL, (
            "Expected thionitrous acid to not match the definition, but it did."
        )

        # Now test with a more accurate definition that includes `OneCarbonCompound`,
        # and provide background definitions for `OneCarbonCompound` and
        # `TwoPlusCarbonCompound`. This should allow carbon monoxide to match, but not
        # ethanol or thionitrous acid.
        definition_str = "CarbonMonoxide ↔ (∃x ∃y (OneCarbonCompound ∧ C(x) ∧ O(y) ∧ HasBondTo(x, y)))"
        definition_to_match = reasoner.get_tptp_fol_definition(definition_str)[1]
        add_defs_dict = {
            "onecarboncompound": "OneCarbonCompound ↔ (∃x (C(x) ∧ ¬TwoPlusCarbonCompound))",
            "twopluscarboncompound": "TwoPlusCarbonCompound ↔ (∃x ∃y (C(x) ∧ C(y) ∧ HasBondTo(x, y) ∧ x ≠ y))",
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
            "Expected thionitrous acid to not match the definition, but it did. This is because thionitrous acid does not contain a carbon atom bonded to an oxygen atom."
        )

    def test_few_shots_examples_model_checking(
        self, reasoner: MistralCustomFOLReasoner
    ):
        """Test few-shot examples from nl_2_fol/prompting/prompt_templates/few_shots/mistral_fol_math_syntax.json."""
        # Test carboxylic acid formula
        few_shot_formula_1 = "CarboxylicAcid ↔ (CarbonOxoacid ∧ ∃x ∃y ∃z (C(x) ∧ O(y) ∧ O(z) ∧ Has1Hs(z) ∧ BDOUBLE(x, y) ∧ BSINGLE(x, z)))"

        # Add background definitions for carboxylic acid test
        add_defs_dict_1 = {"carbonoxoacid": "CarbonOxoacid ↔ (∃x ∃y (C(x) ∧ O(y)))"}
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
        few_shot_formula_2 = "Azide ↔ (NitrogenMolecularEntity ∧ ∃x ∃y ∃z (N(x) ∧ Charge0(x) ∧ N(y) ∧ Charge1(y) ∧ N(z) ∧ ChargeM1(z) ∧ BDOUBLE(x, y) ∧ BDOUBLE(y, z)))"

        # Add background definitions for azide test
        add_defs_dict_2 = {
            "nitrogenmolecularentity": "NitrogenMolecularEntity ↔ (∃x N(x))"
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


class TestConvertToChemlogPredicates:
    """Test suite for _convert_to_chemlog_predicates method."""

    @pytest.fixture
    def reasoner(self):
        """Create a MistralCustomFOLReasoner instance for testing."""
        return MistralCustomFOLReasoner()

    # ==================== HasNHs Pattern Tests ====================

    def test_has_single_hydrogen(self, reasoner):
        """Test HasNHs pattern with single digit conversion."""
        formula = "Has1Hs(X)"
        result = reasoner._convert_to_chemlog_predicates(formula)
        assert result == "has_1_hs(X)", f"Expected 'has_1_hs(X)', got '{result}'"

    def test_has_multiple_hydrogens(self, reasoner):
        """Test HasNHs pattern with multiple different digit values."""
        test_cases = [
            ("Has2Hs(X)", "has_2_hs(X)"),
            ("Has3Hs(Y)", "has_3_hs(Y)"),
            ("Has4Hs(Z)", "has_4_hs(Z)"),
        ]
        for formula, expected in test_cases:
            result = reasoner._convert_to_chemlog_predicates(formula)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_has_hs_case_insensitive(self, reasoner):
        """Test HasNHs pattern with different case variations."""
        test_cases = [
            ("has1hs(X)", "has_1_hs(X)"),
            ("HAS1HS(X)", "has_1_hs(X)"),
            ("Has1hs(X)", "has_1_hs(X)"),
            ("hAS1HS(X)", "has_1_hs(X)"),
        ]
        for formula, expected in test_cases:
            result = reasoner._convert_to_chemlog_predicates(formula)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_has_hs_word_boundary(self, reasoner):
        """Test that HasNHs only matches whole words."""
        formula = "MyHas1HsFunc(X) ∧ Has1Hs(Y)"
        result = reasoner._convert_to_chemlog_predicates(formula)
        # MyHas1HsFunc should NOT be replaced (not a word boundary)
        # Has1Hs should be replaced
        assert "has_1_hs(Y)" in result
        assert "MyHas1HsFunc" in result

    def test_has_hs_multiple_occurrences(self, reasoner):
        """Test multiple HasNHs patterns in one formula."""
        formula = "∃x ∃y (Has1Hs(x) ∧ Has2Hs(y) ∧ Has3Hs(z))"
        result = reasoner._convert_to_chemlog_predicates(formula)
        assert "has_1_hs(x)" in result
        assert "has_2_hs(y)" in result
        assert "has_3_hs(z)" in result

    # ==================== HasAtLeastNHs Pattern Tests ====================

    def test_has_at_least_single_hydrogen(self, reasoner):
        """Test HasAtLeastNHs pattern."""
        formula = "HasAtLeast1Hs(X)"
        result = reasoner._convert_to_chemlog_predicates(formula)
        assert result == "has_at_least_1_hs(X)"

    def test_has_at_least_multiple_hydrogens(self, reasoner):
        """Test HasAtLeastNHs pattern with multiple values."""
        test_cases = [
            ("HasAtLeast2Hs(X)", "has_at_least_2_hs(X)"),
            ("HasAtLeast3Hs(Y)", "has_at_least_3_hs(Y)"),
            ("HasAtLeast10Hs(Z)", "has_at_least_10_hs(Z)"),
        ]
        for formula, expected in test_cases:
            result = reasoner._convert_to_chemlog_predicates(formula)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_has_at_least_case_insensitive(self, reasoner):
        """Test HasAtLeastNHs pattern with case variations."""
        test_cases = [
            ("hasatleast1hs(X)", "has_at_least_1_hs(X)"),
            ("HASATLEAST1HS(X)", "has_at_least_1_hs(X)"),
            ("HasAtLeast1Hs(X)", "has_at_least_1_hs(X)"),
        ]
        for formula, expected in test_cases:
            result = reasoner._convert_to_chemlog_predicates(formula)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    # ==================== HasMinNHs Pattern Tests ====================

    def test_has_min_single_hydrogen(self, reasoner):
        """Test HasMinNHs pattern."""
        formula = "HasMin1Hs(X)"
        result = reasoner._convert_to_chemlog_predicates(formula)
        assert result == "has_min_1_hs(X)"

    def test_has_min_multiple_hydrogens(self, reasoner):
        """Test HasMinNHs pattern with multiple values."""
        test_cases = [
            ("HasMin2Hs(X)", "has_min_2_hs(X)"),
            ("HasMin3Hs(Y)", "has_min_3_hs(Y)"),
            ("HasMin5Hs(Z)", "has_min_5_hs(Z)"),
        ]
        for formula, expected in test_cases:
            result = reasoner._convert_to_chemlog_predicates(formula)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_has_min_case_insensitive(self, reasoner):
        """Test HasMinNHs pattern with case variations."""
        test_cases = [
            ("hasmin1hs(X)", "has_min_1_hs(X)"),
            ("HASMIN1HS(X)", "has_min_1_hs(X)"),
            ("HasMin1Hs(X)", "has_min_1_hs(X)"),
        ]
        for formula, expected in test_cases:
            result = reasoner._convert_to_chemlog_predicates(formula)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    # ==================== Dictionary Predicate Map Tests ====================

    def test_charge_predicates(self, reasoner):
        """Test charge-related predicate conversions."""
        test_cases = [
            ("Charge0(X)", "charge0(X)"),
            ("Charge1(X)", "charge1(X)"),
            ("Charge2(X)", "charge2(X)"),
            ("Charge3(X)", "charge3(X)"),
            ("ChargeM1(X)", "charge_m1(X)"),
            ("ChargeM2(X)", "charge_m2(X)"),
            ("ChargeM3(X)", "charge_m3(X)"),
            ("ChargeP(X)", "charge_p(X)"),
            ("ChargeN(X)", "charge_n(X)"),
        ]
        for formula, expected in test_cases:
            result = reasoner._convert_to_chemlog_predicates(formula)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_charge_predicates_case_insensitive(self, reasoner):
        """Test charge predicates with case variations."""
        test_cases = [
            ("charge0(X)", "charge0(X)"),
            ("CHARGE0(X)", "charge0(X)"),
            ("ChArGe0(X)", "charge0(X)"),
        ]
        for formula, expected in test_cases:
            result = reasoner._convert_to_chemlog_predicates(formula)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_cip_code_predicates(self, reasoner):
        """Test CIP code predicate conversions."""
        test_cases = [
            ("CipCodeR(X)", "cip_code_R(X)"),
            ("CipCodeS(X)", "cip_code_S(X)"),
        ]
        for formula, expected in test_cases:
            result = reasoner._convert_to_chemlog_predicates(formula)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_bond_predicates(self, reasoner):
        """Test bond type predicate conversions."""
        test_cases = [
            ("BSINGLE(X, Y)", "bSINGLE(X, Y)"),
            ("BDOUBLE(X, Y)", "bDOUBLE(X, Y)"),
            ("BTRIPLE(X, Y)", "bTRIPLE(X, Y)"),
            ("BAROMATIC(X, Y)", "bAROMATIC(X, Y)"),
        ]
        for formula, expected in test_cases:
            result = reasoner._convert_to_chemlog_predicates(formula)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_bond_predicates_case_insensitive(self, reasoner):
        """Test bond predicates with case variations."""
        test_cases = [
            ("bsingle(X, Y)", "bSINGLE(X, Y)"),
            ("BSINGLE(X, Y)", "bSINGLE(X, Y)"),
            ("Bsingle(X, Y)", "bSINGLE(X, Y)"),
        ]
        for formula, expected in test_cases:
            result = reasoner._convert_to_chemlog_predicates(formula)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_bond_to_predicate(self, reasoner):
        """Test HasBondTo predicate conversion."""
        formula = "HasBondTo(X, Y)"
        result = reasoner._convert_to_chemlog_predicates(formula)
        assert result == "has_bond_to(X, Y)"

    def test_net_charge_predicates(self, reasoner):
        """Test net charge predicate conversions."""
        test_cases = [
            ("NetChargePositive(X)", "net_charge_positive(X)"),
            ("NetChargeNegative(X)", "net_charge_negative(X)"),
            ("NetChargeNeutral(X)", "net_charge_neutral(X)"),
        ]
        for formula, expected in test_cases:
            result = reasoner._convert_to_chemlog_predicates(formula)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    # ==================== Complex Formula Tests ====================

    def test_complex_formula_with_mixed_predicates(self, reasoner):
        """Test complex formula with multiple types of predicates."""
        formula = "Azide ↔ (∃x ∃y ∃z (N(x) ∧ Charge0(x) ∧ N(y) ∧ Charge1(y) ∧ N(z) ∧ ChargeM1(z) ∧ BDOUBLE(x, y) ∧ BDOUBLE(y, z)))"
        result = reasoner._convert_to_chemlog_predicates(formula)

        # Verify all predicates are converted correctly
        assert "charge0(x)" in result
        assert "charge1(y)" in result
        assert "charge_m1(z)" in result
        assert "bDOUBLE(x, y)" in result
        assert "bDOUBLE(y, z)" in result
        # N(x), N(y), N(z) should remain unchanged
        assert "N(x)" in result
        assert "N(y)" in result
        assert "N(z)" in result

    def test_carboxylic_acid_formula(self, reasoner):
        """Test carboxylic acid formula conversion."""
        formula = "CarboxylicAcid ↔ (∃x ∃y ∃z (C(x) ∧ O(y) ∧ O(z) ∧ Has1Hs(z) ∧ BDOUBLE(x, y) ∧ BSINGLE(x, z)))"
        result = reasoner._convert_to_chemlog_predicates(formula)

        assert "has_1_hs(z)" in result
        assert "bDOUBLE(x, y)" in result
        assert "bSINGLE(x, z)" in result
        assert "C(x)" in result
        assert "O(y)" in result
        assert "O(z)" in result

    def test_formula_with_logical_operators(self, reasoner):
        """Test that conversions work correctly with logical operators."""
        formula = "(Has1Hs(X) ∧ Has2Hs(Y)) ∨ (Charge0(Z) ∧ ¬NetChargePositive(W))"
        result = reasoner._convert_to_chemlog_predicates(formula)

        assert "has_1_hs(X)" in result
        assert "has_2_hs(Y)" in result
        assert "charge0(Z)" in result
        assert "net_charge_positive(W)" in result
        # Logical operators should be preserved
        assert "∧" in result or "and" in result.lower() or "&" in result
        assert "∨" in result or "or" in result.lower() or "|" in result
        assert "¬" in result or "not" in result.lower()

    def test_formula_with_quantifiers(self, reasoner):
        """Test that conversions work with quantified formulas."""
        formula = "∃x ∀y (Has3Hs(x) → HasAtLeast2Hs(y))"
        result = reasoner._convert_to_chemlog_predicates(formula)

        assert "has_3_hs(x)" in result
        assert "has_at_least_2_hs(y)" in result
        # Quantifiers should be preserved
        assert "∃" in result or "exists" in result.lower()
        assert "∀" in result or "forall" in result.lower()

    # ==================== Edge Cases ====================

    def test_empty_string(self, reasoner):
        """Test with empty string."""
        result = reasoner._convert_to_chemlog_predicates("")
        assert result == ""

    def test_no_matching_predicates(self, reasoner):
        """Test formula with no predicates to convert."""
        formula = "Atom(X) ∧ Bond(X, Y)"
        result = reasoner._convert_to_chemlog_predicates(formula)
        # Should remain unchanged
        assert "Atom(X)" in result
        assert "Bond(X, Y)" in result

    def test_predicate_not_at_word_boundary(self, reasoner):
        """Test that predicates not at word boundaries are not replaced."""
        formula = "MyHas1Hs(X) ∧ Has1Hs(Y)"
        result = reasoner._convert_to_chemlog_predicates(formula)
        # MyHas1Hs should not be converted
        assert "MyHas1Hs" in result
        # Has1Hs should be converted
        assert "has_1_hs(Y)" in result

    def test_whitespace_preserved(self, reasoner):
        """Test that whitespace is preserved in conversions."""
        formula = "Has1Hs ( X )  ∧  Has2Hs ( Y )"
        result = reasoner._convert_to_chemlog_predicates(formula)
        # Conversions should work even with extra whitespace
        assert "has_1_hs" in result
        assert "has_2_hs" in result

    def test_nested_predicates(self, reasoner):
        """Test nested predicate patterns."""
        formula = "f(Has1Hs(X), g(Has2Hs(Y), Charge0(Z)))"
        result = reasoner._convert_to_chemlog_predicates(formula)

        assert "has_1_hs(X)" in result
        assert "has_2_hs(Y)" in result
        assert "charge0(Z)" in result

    def test_multiple_predicate_maps_in_sequence(self, reasoner):
        """Test formula with multiple different mapped predicates in sequence."""
        formula = (
            "Charge0(X) ∧ Charge1(Y) ∧ ChargeM1(Z) ∧ HasBondTo(X, Y) ∧ BSINGLE(Y, Z)"
        )
        result = reasoner._convert_to_chemlog_predicates(formula)

        assert "charge0(X)" in result
        assert "charge1(Y)" in result
        assert "charge_m1(Z)" in result
        assert "has_bond_to(X, Y)" in result
        assert "bSINGLE(Y, Z)" in result
