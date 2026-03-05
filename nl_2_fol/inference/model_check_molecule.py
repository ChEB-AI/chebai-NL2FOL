import time

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
    _MODEL_CHECK_TIMEOUT_SECONDS = 30

    def __init__(self) -> None:
        self._tptp_parser = TPTPParser()
        self._base_predicates: dict[str, str] = GAVEL_PREDICATES
        self.background_definitions: dict[
            str, tuple[list[logic.Variable], logic.QuantifiedFormula]
        ] = {}

    @tptp_parse_exception
    def get_tptp_fol_definition(
        self, formula: str
    ) -> tuple[list[logic.Variable], logic.QuantifiedFormula]:
        """Parses a formula in TPTP format into gavel's internal representation.

        Parsing Process:
        1. Wrap the input formula into TPTP annotated format: fof(temp, axiom, <formula>).
        2. Parse using TPTP parser and extract the right-hand side of the biimplication.
        3. Ensure the result is a QuantifiedFormula (wrap in existential quantifier if not).
        4. Normalize the formula to PNF (Prenex Normal Form) with matrix in CNF for model checking.
        """
        # wrap formula into an *annotated formula* for parsing
        formula_wrapped = f"fof(temp, axiom, {formula})."
        try:
            tptp_parsed = self._tptp_parser.parse(formula_wrapped)[0].formula
        except Exception as e:
            raise Exception(
                f"Error parsing FOL formula to TPTP format.\n"
                f"Process: The formula is wrapped in TPTP annotated format and parsed.\n"
                f"[IMPORTANT] Critical error details for analysis:\n{e}"
            )

        # Ensure the left-hand side is a predicate expression (not ambiguous)
        if not isinstance(tptp_parsed.left, logic.PredicateExpression):
            raise Exception(
                f"Invalid FOL formula structure: left-hand side of biimplication must be a predicate expression.\n"
                f"Issue: The formula is ambiguous due to missing brackets. Operators like '&' and '|' should be "
                f"explicitly bracketed, e.g., 'predicate(x) <=> (condition1 & condition2)' instead of "
                f"'predicate(x) <=> condition1 & condition2'.\n"
                f"Parsed left side: {tptp_parsed.left}\n"
                f"Expected: A predicate expression like 'predicate', 'predicate(x)', 'predicate(x, y)', etc."
            )

        pred_variables = self._extract_predicate_variables(tptp_parsed.left)
        tptp_right_side = tptp_parsed.right

        # Eg. `oligopeptide(x) <=> (peptide(x) & has_few_amino_acid_residues(x))`
        # The above fol formula will be parsed and will have no formula attribute
        if not isinstance(tptp_right_side, logic.QuantifiedFormula):
            tptp_right_side = logic.QuantifiedFormula(
                logic.Quantifier.EXISTENTIAL, [], tptp_right_side
            )

        try:
            tptp_right_side = normalize_fol_formula(tptp_right_side)
        except Exception as e:
            raise Exception(
                "Error normalizing formula to PNF (Prenex Normal Form).\n"
                f"TPTP Parsed Formula before normalization: `{tptp_right_side}`\n"
                "Process: The formula is converted to PNF (all quantifiers moved to"
                "the front) with the matrix in CNF (Conjunctive Normal Form with N-ary"
                "conjunctions and disjunctions).\n"
                f"[IMPORTANT] Critical error details for analysis:\n{e}"
            )
        print(f"Input formula: {formula}\n\t Parsed Right Side as: {tptp_right_side}")
        return pred_variables, tptp_right_side

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
            model_check_start_time = time.monotonic()
            outcome, _ = model_checker.find_model(definition_to_match)
            elapsed_seconds = time.monotonic() - model_check_start_time
            if elapsed_seconds > self._MODEL_CHECK_TIMEOUT_SECONDS:
                raise TimeoutError(
                    "Generated FOL formula took more than 30 seconds during model checking. "
                    f"Elapsed: {elapsed_seconds:.2f}s. "
                    f"Formula being checked: `{definition_to_match}`"
                    "Reduce the complexity of the formula"
                )
        except Exception as e:
            raise Exception(
                f"MODEL CHECKING FAILED - Error during model checking for the formula.\n"
                f"Formula being checked: `{definition_to_match}`\n\n"
                f"Background: The formula was parsed through these steps:\n"
                f"  1. Parsed using TPTP parser and extracted right-hand side of biimplication.\n"
                f"  2. Wrapped in QuantifiedFormula if not already quantified.\n"
                f"  3. Normalized to PNF (all quantifiers at front) with matrix in CNF.\n\n"
                f"[IMPORTANT] Critical error details for analysis:\n{e}"
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

    def _extract_predicate_variables(
        self, formula_left_side: logic.PredicateExpression
    ) -> list[logic.Variable]:
        """Extract the variables from a predicate definition string.

        For a definition like `new_predicate(X1, X2) <=> ?[X3]: (...)`
        This extracts [X1, X2] from the predicate call on the left side of the biimplication.
        """
        # Extract variables from the predicate expression
        variables = []
        if isinstance(formula_left_side, logic.PredicateExpression):
            # The arguments should be Variable objects
            if hasattr(formula_left_side, "arguments") and formula_left_side.arguments:
                for arg in formula_left_side.arguments:
                    if isinstance(arg, logic.Variable):
                        variables.append(arg)

        return variables

    def add_background_definition(
        self,
        name: str,
        variables: list[logic.Variable],
        definition: logic.QuantifiedFormula,
    ):
        """Add a single background definition with extracted free variables."""
        self.background_definitions[name] = (variables, definition)

    def convert_to_background_definitions(
        self,
        predicates: dict[str, str],
    ) -> dict[str, tuple[list[logic.Variable], logic.QuantifiedFormula]]:
        """Convert a dictionary of predicate definitions (as strings) to the internal format."""
        converted = {}
        for name, def_str in predicates.items():
            pred_vars, fol_formula = self.get_tptp_fol_definition(def_str)
            converted[name] = (pred_vars, fol_formula)
        return converted


if __name__ == "__main__":
    # Example usage

    fol_parser = GavelFOLReasoner()
    # with open("nl_2_fol/inference/learned/learned_definitions.pkl", "rb") as f:
    #     new_definitions = pickle.load(f)
    # for _, learned_def in new_definitions.learned_definitions.items():
    #     print(
    #         f"Adding background definition for `{learned_def.name}`: {learned_def.learned_FOL}"
    #     )
    #     fol_parser.add_background_definition(
    #         learned_def.name,
    #         learned_def.learned_FOL.pred_variables,
    #         learned_def.learned_FOL.formula,
    #     )

    # for name, add_def in new_definitions.additional_definitions.items():
    #     print(f"Adding background definition for `{name}`: {add_def}")
    #     fol_parser.add_background_definition(
    #         name, add_def.pred_variables, add_def.formula
    #     )

    llm_for = "tripeptide <=> (oligopeptide & ?[C1, O1, N1, C2, O2, N2]: (c(C1) & o(O1) & bDOUBLE(C1, O1) & n(N1) & bSINGLE(C1, N1) & has_1_hs(N1) & c(C2) & o(O2) & bDOUBLE(C2, O2) & n(N2) & bSINGLE(C2, N2) & has_1_hs(N2) & C1 != C2 & O1 != O2 & N1 != N2 & ![C3, O3, N3]: ((c(C3) & o(O3) & bDOUBLE(C3, O3) & n(N3) & bSINGLE(C3, N3) & has_1_hs(N3) & peptide(C3, O3, N3)) => ((C3 = C1 & O3 = O1 & N3 = N1) | (C3 = C2 & O3 = O2 & N3 = N2)))))"
    fol_parser.get_tptp_fol_definition(llm_for)
    mol = Chem.MolFromSmiles(
        "C(=O)([C@@H](NC(=O)OCC=1C=CC=CC1)C(C)C)N[C@H](C(=O)N[C@H](C(=O)CF)CC(=O)OC)C"
    )
    matches = fol_parser.does_mol_match_tptp_definition(
        mol, fol_parser.get_tptp_fol_definition(llm_for)[1]
    )
    print(f"Tripeptide matches definition: {matches}")
    exit()

    carbonMonoxide = Chem.MolFromSmiles("[C-]#[O+]")  # CHEBI:17245
    ethanol = Chem.MolFromSmiles("CCO")
    thionitrousAcid = Chem.MolFromSmiles("SN=O")  # CHEBI:65308

    # Logical definition to match (I removed the `oneCarbonCompound` predicate for simplicity)
    definition_str = (
        "carbonMonoxide <=> ?[A1, A2]: (c(A1) & o(A2) & has_bond_to(A1,A2))"
    )
    definition_to_match = fol_parser.get_tptp_fol_definition(definition_str)[1]

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
    definition_to_match = fol_parser.get_tptp_fol_definition(definition_str)[1]
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
