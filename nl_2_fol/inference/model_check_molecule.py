from chemlog.fol_classification.fol_utils import normalize_fol_formula
from chemlog.fol_classification.model_checking import ModelChecker, ModelCheckerOutcome
from chemlog.preprocessing.mol_to_fol import mol_to_fol_atoms
from gavel.dialects.tptp.parser import TPTPParser
from gavel.logic import logic
from rdkit import Chem

from nl_2_fol.inference.base_predicates import GAVEL_PREDICATES
from nl_2_fol.inference.custom_exceptions import (
    MissingPredicateException,
    model_check_exception,
    mol_to_fol_exception,
    tptp_parse_exception,
)


class GavelFOLReasoner:
    def __init__(self) -> None:
        self._tptp_parser = TPTPParser()
        self._base_predicates: dict[str, str] = GAVEL_PREDICATES
        self.background_definitions: dict[
            str, tuple[list[logic.Variable], logic.QuantifiedFormula]
        ] = {}

    @tptp_parse_exception
    def get_tptp_fol_definition(self, formula: str) -> logic.QuantifiedFormula:
        """Parses a formula in TPTP format into gavel's internal representation.

        Parsing Process:
        1. Wrap the input formula into TPTP annotated format: fof(temp, axiom, <formula>).
        2. Parse using TPTP parser and extract the right-hand side of the biimplication.
        3. Ensure the result is a QuantifiedFormula (wrap in existential quantifier if not).
        4. Normalize the formula to PNF (Prenex Normal Form) with matrix in CNF for model checking.
        """
        # wrap formula into an *annotated formula* for parsing
        formula_wrapped = f"fof(temp, axiom, {formula})."
        # unwrap the annotated formula after parsing, only take the right-hand side of the biimplication
        try:
            tptp_parsed = self._tptp_parser.parse(formula_wrapped)[0].formula.right
        except Exception as e:
            raise Exception(
                f"PARSING STEP 1/3 FAILED - Error parsing FOL formula to TPTP format.\n"
                f"Process: The formula is wrapped in TPTP annotated format and parsed.\n"
                f"More specifics: {e}"
            )
        # the model checker expects the matrix (the part after the quantifiers) to be in CNF (with N-ary conjunctions and disjunctions)
        try:
            # Eg. `oligopeptide(x) <=> (peptide(x) & has_few_amino_acid_residues(x))`
            # The above fol formula will be parsed and will have no formula attribute
            if not isinstance(tptp_parsed, logic.QuantifiedFormula):
                tptp_parsed = logic.QuantifiedFormula(
                    logic.Quantifier.EXISTENTIAL, [], tptp_parsed
                )
        except AssertionError as e:
            raise Exception(
                f"PARSING STEP 2/3 FAILED - Error wrapping parsed formula in QuantifiedFormula.\n"
                f"Parsed result: `{tptp_parsed}`\n"
                f"Process: If the parsed formula is not already quantified, it's wrapped in an existential quantifier.\n"
                f"More specifics: {e}"
            )

        try:
            tptp_parsed = normalize_fol_formula(tptp_parsed)
        except Exception as e:
            raise Exception(
                "PARSING STEP 3/3 FAILED - Error normalizing formula to PNF (Prenex Normal Form).\n"
                f"Formula before normalization: `{tptp_parsed}`\n"
                f"Process: The formula is converted to PNF (all quantifiers moved to the front)"
                "with the matrix in CNF (Conjunctive Normal Form with N-ary conjunctions and disjunctions).\n"
                f"More specifics: {e}"
            )
        print(f"Input formula: {formula}\n\t Parsed as: {tptp_parsed}")
        return tptp_parsed

    @model_check_exception
    def does_mol_match_tptp_definition(
        self,
        molecule: Chem.Mol,
        definition_to_match: logic.QuantifiedFormula,
        additional_background_definitions: dict[
            str, tuple[list[logic.Variable], logic.QuantifiedFormula]
        ]
        | None = None,
    ) -> bool:
        """Checks if a given molecule matches the logical definition."""
        predicates = self._extract_predicates(definition_to_match)
        missing_predicates = predicates - self._base_predicates.keys()
        missing_predicates = (
            missing_predicates - self.background_definitions.keys()
            if self.background_definitions
            else missing_predicates
        )
        if additional_background_definitions:
            missing_predicates = (
                missing_predicates - additional_background_definitions.keys()
            )
        if missing_predicates:
            raise MissingPredicateException(missing_predicates)

        universe, extensions = self._mol_to_fol(molecule)
        bck_def = {
            **self.background_definitions,
            **(additional_background_definitions or {}),
        }
        model_checker = ModelChecker(universe, extensions, bck_def)
        try:
            # Can fail for definitions like: `∃[]: ((peptide(x)))`
            outcome, _ = model_checker.find_model(definition_to_match)
        except Exception as e:
            raise Exception(
                f"MODEL CHECKING FAILED - Error during model checking for the formula.\n"
                f"Formula being checked: `{definition_to_match}`\n\n"
                f"Background: The formula was parsed through these steps:\n"
                f"  1. Parsed using TPTP parser and extracted right-hand side of biimplication.\n"
                f"  2. Wrapped in QuantifiedFormula if not already quantified.\n"
                f"  3. Normalized to PNF (all quantifiers at front) with matrix in CNF.\n\n"
                f"More specifics: {e}"
            )
        return outcome == ModelCheckerOutcome.MODEL_FOUND

    @mol_to_fol_exception
    def _mol_to_fol(self, mol: Chem.Mol):
        """Convert an RDKit molecule to a first-order logic representation."""
        universe, extensions = mol_to_fol_atoms(mol)
        # rename / add custom extensions if needed
        return universe, extensions

    def _extract_predicates(self, formula: logic.QuantifiedFormula) -> set[str]:
        """Extract all predicates from a parsed TPTP formula."""
        predicates = set()

        def traverse(node):
            if isinstance(node, logic.PredicateExpression):
                # node.predicate gives the predicate symbol
                predicates.add(node.predicate)
            elif isinstance(node, logic.BinaryFormula):
                traverse(node.left)
                traverse(node.right)
            elif isinstance(node, logic.NaryFormula):
                for formula_node in node.formulae:
                    traverse(formula_node)
            elif isinstance(node, logic.UnaryFormula):
                traverse(node.formula)
            elif isinstance(node, logic.QuantifiedFormula):
                traverse(node.formula)

        traverse(formula.formula)
        return predicates

    def add_background_definition(self, name: str, definition: logic.QuantifiedFormula):
        """Add a single background definition."""
        self.background_definitions[name] = ([], definition)

    def convert_to_background_definitions(
        self,
        predicates: dict[str, str],
    ) -> dict[str, tuple[list[logic.Variable], logic.QuantifiedFormula]]:
        """Convert a dictionary of predicate definitions (as strings) to the internal format."""
        converted = {}
        for name, def_str in predicates.items():
            converted[name] = (
                [],
                self.get_tptp_fol_definition(def_str),
            )
        return converted


if __name__ == "__main__":
    # Example usage
    fol_parser = GavelFOLReasoner()

    carbonMonoxide = Chem.MolFromSmiles("[C-]#[O+]")  # CHEBI:17245
    ethanol = Chem.MolFromSmiles("CCO")
    thionitrousAcid = Chem.MolFromSmiles("SN=O")  # CHEBI:65308

    # Logical definition to match (I removed the `oneCarbonCompound` predicate for simplicity)
    definition_str = (
        "carbonMonoxide <=> ?[A1, A2]: (c(A1) & o(A2) & has_bond_to(A1,A2))"
    )
    definition_to_match = fol_parser.get_tptp_fol_definition(definition_str)

    # Background definitions (none needed here)
    background_definitions = {}
    matches = fol_parser.does_mol_match_tptp_definition(
        carbonMonoxide, definition_to_match
    )
    print(f"Carbon monoxide matches definition: {matches}")
    matches = fol_parser.does_mol_match_tptp_definition(ethanol, definition_to_match)
    print(
        f"Ethanol matches definition: {matches}"
    )  # returns model found (which contradicts the chemistry)
    matches = fol_parser.does_mol_match_tptp_definition(
        thionitrousAcid, definition_to_match
    )
    print(f"Thionitrous acid matches definition: {matches}")

    # Logical definition to match (more accurate version - requires knowing what a oneCarbonCompound is)
    definition_str = "carbonMonoxide <=> ?[A1, A2]: (oneCarbonCompound & c(A1) & o(A2) & has_bond_to(A1,A2))"
    definition_to_match = fol_parser.get_tptp_fol_definition(definition_str)
    assert not isinstance(definition_to_match, Exception)
    fol_parser.background_definitions = {
        "oneCarbonCompound": (
            [],
            fol_parser.get_tptp_fol_definition(
                "oneCarbonCompound <=> ?[X]: (c(X) & ~twoPlusCarbonCompound)"
            ),
        ),
        "twoPlusCarbonCompound": (
            [],
            fol_parser.get_tptp_fol_definition(
                "twoPlusCarbonCompound <=> ?[X, Y]: (c(X) & c(Y) & has_bond_to(X, Y) & X != Y)"
            ),
        ),
    }
    matches = fol_parser.does_mol_match_tptp_definition(
        carbonMonoxide, definition_to_match
    )
    print(f"Carbon monoxide matches definition: {matches}")
    matches = fol_parser.does_mol_match_tptp_definition(ethanol, definition_to_match)
    print(
        f"Ethanol matches definition: {matches}"
    )  # now, no model found because we added the oneCarbonCompound definition
    matches = fol_parser.does_mol_match_tptp_definition(
        thionitrousAcid, definition_to_match
    )
    print(f"Thionitrous acid matches definition: {matches}")
