from typing import Dict, List, Tuple

from chemlog.fol_classification.model_checking import ModelChecker
from gavel.dialects.tptp.parser import TPTPParser
from gavel.logic import logic
from gavel.logic.logic_utils import convert_to_cnf
from mol_to_fol import mol_to_fol
from rdkit import Chem


def parse_tptp_formula(formula: str) -> logic.QuantifiedFormula:
    """Parses a formula in TPTP format (as obtained from an LLM) into gavel's internal representation."""
    tptp_parser = TPTPParser()
    # wrap formula into an *annotated formula* for parsing
    formula_wrapped = f"fof(temp, axiom, {formula})."
    # unwrap the annotated formula after parsing, only take the right-hand side of the biimplication
    tptp_parsed = tptp_parser.parse(formula_wrapped)[0].formula.right
    # the model checker expects the matrix (the part after the quantifiers) to be in CNF (with N-ary conjunctions and disjunctions)
    tptp_parsed.formula = convert_to_cnf(tptp_parsed.formula)
    print(f"Input formula: {formula}\n\t Parsed as: {tptp_parsed}")
    return tptp_parsed

def molecule_matches_definition(molecule: Chem.Mol, definition_to_match: logic.QuantifiedFormula, background_definitions: Dict[str, Tuple[List[logic.Variable], logic.QuantifiedFormula]] | None) -> bool:
    """Checks if a given molecule matches the logical definition."""
    universe, extensions = mol_to_fol(molecule)
    model_checker = ModelChecker(universe, extensions, background_definitions)

    outcome, model = model_checker.find_model(definition_to_match)
    return outcome

if __name__ == "__main__":
    # Example usage
    carbonMonoxide = Chem.MolFromSmiles("[C-]#[O+]") # CHEBI:17245
    ethanol = Chem.MolFromSmiles("CCO")
    thionitrousAcid = Chem.MolFromSmiles("SN=O") # CHEBI:65308

    # Logical definition to match (I removed the oneCarbonCompound predicate for simplicity)
    definition_str = "carbonMonoxide <=> ?[A1, A2]: (c(A1) & o(A2) & has_bond_to(A1,A2))"
    definition_to_match = parse_tptp_formula(definition_str)
    # Background definitions (none needed here)
    background_definitions = {}
    matches = molecule_matches_definition(carbonMonoxide, definition_to_match, background_definitions)
    print(f"Carbon monoxide matches definition: {matches}")
    matches = molecule_matches_definition(ethanol, definition_to_match, background_definitions)
    print(f"Ethanol matches definition: {matches}") # returns model found (which contradicts the chemistry)
    matches = molecule_matches_definition(thionitrousAcid, definition_to_match, background_definitions)
    print(f"Thionitrous acid matches definition: {matches}")

    # Logical definition to match (more accurate version - requires knowing what a oneCarbonCompound is)
    definition_str = "carbonMonoxide <=> ?[A1, A2]: (oneCarbonCompound & c(A1) & o(A2) & has_bond_to(A1,A2))"
    definition_to_match = parse_tptp_formula(definition_str)
    background_definitions = {
        "oneCarbonCompound": ([], parse_tptp_formula("oneCarbonCompound <=> ?[X]: (c(X) & ~twoPlusCarbonCompound)")),
        "twoPlusCarbonCompound": ([], parse_tptp_formula("twoPlusCarbonCompound <=> ?[X, Y]: (c(X) & c(Y) & has_bond_to(X, Y) & X != Y)"))
    }
    matches = molecule_matches_definition(carbonMonoxide, definition_to_match, background_definitions)
    print(f"Carbon monoxide matches definition: {matches}")
    matches = molecule_matches_definition(ethanol, definition_to_match, background_definitions)
    print(f"Ethanol matches definition: {matches}") # now, no model found because we added the oneCarbonCompound definition
    matches = molecule_matches_definition(thionitrousAcid, definition_to_match, background_definitions)
    print(f"Thionitrous acid matches definition: {matches}")
