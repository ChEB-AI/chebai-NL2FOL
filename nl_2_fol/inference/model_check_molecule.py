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
        self._background_definitions: dict[
            str, tuple[list[logic.Variable], logic.QuantifiedFormula]
        ] = {}

    @tptp_parse_exception
    def get_tptp_fol_definition(self, formula: str) -> logic.QuantifiedFormula:
        """Parses a formula in TPTP format (as obtained from an LLM) into gavel's internal representation."""
        # wrap formula into an *annotated formula* for parsing
        formula_wrapped = f"fof(temp, axiom, {formula})."
        # unwrap the annotated formula after parsing, only take the right-hand side of the biimplication
        try:
            tptp_parsed = self._tptp_parser.parse(formula_wrapped)[0].formula.right
        except Exception as e:
            raise Exception(
                f"Error parsing FOL formula `{formula}` to TPTP formula: \n"
                f"More specifics on above error: {e}"
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
                f"The parsed TPTP formula `{tptp_parsed}` does not have the\n"
                f" expected structure (missing `.formula` attribute after parsing).\n"
                f"More specifics on above error: {e}"
            )

        try:
            tptp_parsed.formula = normalize_fol_formula(tptp_parsed.formula)
        except Exception as e:
            raise Exception(
                f"Error converting the parsed TPTP formula `{tptp_parsed.formula}` to CNF format\n"
                f"More specifics on above error: {e}"
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
            missing_predicates - self._background_definitions.keys()
            if self._background_definitions
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
            **self._background_definitions,
            **(additional_background_definitions or {}),
        }
        model_checker = ModelChecker(universe, extensions, bck_def)

        # Can fail for definitions like: `∃[]: ((peptide(x)))`
        outcome, _ = model_checker.find_model(definition_to_match)
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

    def update_background_definition(
        self, name: str, definition: logic.QuantifiedFormula
    ):
        """Add a single background definition."""
        self._background_definitions[name] = ([], definition)

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
    fol_parser._background_definitions = {
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
