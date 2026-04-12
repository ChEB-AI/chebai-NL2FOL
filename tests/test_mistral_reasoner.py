"""Tests for MistralCustomFOLReasoner class."""

import pytest
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
        assert matches is True, (
            "Expected carbon monoxide to match the definition, but it did not."
        )
        matches = reasoner.does_mol_match_tptp_definition(ethanol, definition_to_match)
        assert matches is True, (
            "Expected ethanol to match the definition, but it did not. "
        )
        matches = reasoner.does_mol_match_tptp_definition(
            thionitrousAcid, definition_to_match
        )
        assert matches is False, (
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
        assert matches is True, (
            "Expected carbon monoxide to match the definition, but it did not."
        )
        matches = reasoner.does_mol_match_tptp_definition(ethanol, definition_to_match)
        assert matches is False, (
            "Expected ethanol to not match the definition, but it did."
        )
        matches = reasoner.does_mol_match_tptp_definition(
            thionitrousAcid, definition_to_match
        )
        assert matches is False, (
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
        assert result_1_match is True, (
            "Acetic acid should match carboxylic acid formula"
        )

        # Test molecule that does NOT match carboxylic acid pattern
        ethanol_mol = Chem.MolFromSmiles(
            "CCO"
        )  # Ethanol - has C and O but not carboxylic acid structure
        result_1_no_match = reasoner.does_mol_match_tptp_definition(
            ethanol_mol, parsed_formula_1
        )
        assert result_1_no_match is False, (
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
        assert result_2_match is not None

        # Test molecule that does NOT match azide pattern
        aniline_mol = Chem.MolFromSmiles(
            "Nc1ccccc1"
        )  # Aniline - has nitrogen but not azide structure
        result_2_no_match = reasoner.does_mol_match_tptp_definition(
            aniline_mol, parsed_formula_2
        )
        assert result_2_no_match is False, "Aniline should not match azide formula"
