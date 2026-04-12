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

    def test_model_checking_success(self, reasoner: MistralCustomFOLReasoner):
        """Test that the reasoner can successfully parse and model check a formula."""

        carbonMonoxide = Chem.MolFromSmiles("[C-]#[O+]")  # CHEBI:17245
        ethanol = Chem.MolFromSmiles("CCO")
        thionitrousAcid = Chem.MolFromSmiles("SN=O")  # CHEBI:65308

        # Logical definition to match (I removed `OneCarbonCompound` for simplicity)
        definition_str = "CarbonMonoxide(1) ↔ (∃x ∃y (C(x) ∧ O(y) ∧ HasBondTo(x, y)))"
        definition_to_match = reasoner.get_tptp_fol_definition(definition_str)[1]
        matches = reasoner.does_mol_match_tptp_definition(
            carbonMonoxide, definition_to_match
        )
        assert matches is True, (
            "Expected carbon monoxide to match the definition, but it did not."
        )
        matches = reasoner.does_mol_match_tptp_definition(ethanol, definition_to_match)
        assert matches is True, (
            "Expected ethanol to match the definition, but it did not. This is because the definition is very broad and only requires the presence of a carbon atom bonded to an oxygen atom, which is true for ethanol as well."
        )
        matches = reasoner.does_mol_match_tptp_definition(
            thionitrousAcid, definition_to_match
        )
        assert matches is False, (
            "Expected thionitrous acid to not match the definition, but it did. This is because thionitrous acid does not contain a carbon atom bonded to an oxygen atom."
        )

        # Now test with a more accurate definition that includes `OneCarbonCompound`,
        # and provide background definitions for `OneCarbonCompound` and
        # `TwoPlusCarbonCompound`. This should allow carbon monoxide to match, but not
        # ethanol or thionitrous acid.
        definition_str = "CarbonMonoxide(1) ↔ (OneCarbonCompound(1) ∧ ∃x ∃y (C(x) ∧ O(y) ∧ HasBondTo(x, y)))"
        definition_to_match = reasoner.get_tptp_fol_definition(definition_str)[1]
        add_defs_dict = {
            "onecarboncompound": "OneCarbonCompound(1) ↔ (∃x C(x) ∧ ¬TwoPlusCarbonCompound(1))",
            "twopluscarboncompound": "TwoPlusCarbonCompound(1) ↔ (∃x ∃y (C(x) ∧ C(y) ∧ HasBondTo(x, y) ∧ x ≠ y))",
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
