from typing import List

from chemlog.fol_classification.fol_utils import normalize_fol_formula
from chemlog.fol_classification.model_checking import ModelChecker, ModelCheckerOutcome
from chemlog.preprocessing.mol_to_fol import mol_to_fol_atoms
from gavel.dialects.tptp.parser import TPTPParser
from gavel.logic import logic
from rdkit import Chem

from nl_2_fol.inference import PRINT_TRACES
from nl_2_fol.inference.fol_reasoner.base import (
    AbstractModelCheckerWrapper,
    FOLDefinition,
)
from nl_2_fol.inference.learner.custom_exceptions import (
    MissingPredicateException,
    model_check_exception,
    mol_to_fol_exception,
    parse_exception,
)


class ChemlogFOLDefinition(FOLDefinition):
    def __init__(
        self,
        predicate_name: str,
        variables: List[logic.Variable],
        definition: logic.QuantifiedFormula,
    ):
        super().__init__(predicate_name, variables, definition)

    def __str__(self):
        variables = self.variables
        if variables:
            head = f"{self.predicate_name}({', '.join(str(var) for var in variables)})"
        else:
            head = self.predicate_name
        return f"{head} <=> {self.definition}"


class ChemlogModelChecker(AbstractModelCheckerWrapper):
    def __init__(self) -> None:
        self._tptp_parser = TPTPParser()
        super().__init__()

    @parse_exception
    def parse_definition(self, formula: str) -> ChemlogFOLDefinition:
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
        if PRINT_TRACES:
            print(
                f"Input formula: {formula}\n\t Parsed Right Side as: {tptp_right_side}"
            )
        return ChemlogFOLDefinition(
            predicate_name=tptp_parsed.left.predicate,
            variables=pred_variables,
            definition=tptp_right_side,
        )

    @model_check_exception
    def does_mol_match_tptp_definition(
        self,
        molecule: Chem.Mol,
        definition_to_match: logic.QuantifiedFormula,
        temp_additional_defs: dict[str, ChemlogFOLDefinition] | None = None,
    ) -> ModelCheckerOutcome:
        """Checks if a given molecule matches a logical definition using model checking.

        Converts the molecule to a first-order logic representation and uses a model
        checker to determine if the molecule satisfies the given FOL formula.

        Args:
            molecule: RDKit molecule object to be checked.
            definition_to_match: Quantified FOL formula in TPTP format representing
                the chemical class definition to match against.
            temp_additional_defs: Optional temporary background definitions to use
                during this specific check, in addition to the instance's persistent
                background definitions. Useful for validating additional predicates
                before committing them permanently.

        Returns:
            ModelCheckerOutcome: The result of the model checking process.

        Raises:
            MissingPredicateException: If the formula contains predicates not defined
                in base predicates, background definitions, or temporary definitions.
            Exception: For various model checking failures such as predicate arity
                mismatches or normalization errors.
        """
        missing_predicates = self.extract_unknown_predicates(
            definition_to_match, temp_additional_defs
        )
        if missing_predicates:
            raise MissingPredicateException(missing_predicates)

        universe, extensions = self._mol_to_fol(molecule)
        bck_def = {
            **self.background_definitions,
            **(temp_additional_defs or {}),
        }
        bck_def = {name: (d.variables, d.definition) for name, d in bck_def.items()}
        model_checker = ModelChecker(universe, extensions, bck_def, all_different=True)

        exception_prefix = (
            "MODEL CHECKING FAILED - Error during model checking for the formula.\n"
            f"Background: The given formula was parsed through these steps:\n"
            f"  1. Parsed using TPTP parser and extracted right-hand side of biimplication.\n"
            f"  2. Wrapped in QuantifiedFormula if not already quantified.\n"
            f"  3. Normalized to PNF (all quantifiers at front) with matrix in CNF.\n\n"
            f"Parsed Formula being checked: `{definition_to_match}`\n\n"
            "[IMPORTANT] Critical error details for analysis: \n"
        )
        try:
            # Can fail for definitions like: `∃[]: ((peptide(x)))`
            outcome, _ = model_checker.find_model(definition_to_match)

        except ValueError as ve:
            print(f"Error occured for following smiles: {Chem.MolToSmiles(molecule)}")
            if "Predicate" in str(ve) and "is defined with arity" in str(ve):
                # If the raised error is https://github.com/sfluegel05/chemlog-peptides/pull/9/files
                # Extract predicate info from error message for better guidance
                error_msg = str(ve)
                logging_msg = (
                    f"Predicate arity mismatch detected: {error_msg}\n"
                    f"Example usage guidance:\n"
                    f"  - Predicate with arity 0 (no arguments): use as 'predicate' in formula\n"
                    f"  - Predicate with arity 1 (1 argument): use as 'predicate(x)'\n"
                    f"  - Predicate with arity 2 (2 arguments): use as 'predicate(x, y)'\n"
                    f"Ensure all predicate calls match their defined arity."
                )
                if PRINT_TRACES:
                    print(f"[WARNING] {logging_msg}")
                raise Exception(logging_msg)
            else:
                raise Exception(f"{exception_prefix}{ve}")
        except Exception as e:
            print(f"Error occured for following smiles: {Chem.MolToSmiles(molecule)}")
            raise Exception(f"{exception_prefix}{e}")
        return outcome

    @mol_to_fol_exception
    def _mol_to_fol(self, mol: Chem.Mol):
        """Convert an RDKit molecule to a first-order logic representation."""
        universe, extensions = mol_to_fol_atoms(mol)
        # rename / add custom extensions if needed
        return universe, extensions

    def _extract_predicate_names(self, formula: logic.QuantifiedFormula) -> set[str]:
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

    @property
    def dummy_formula(self) -> str:
        return "failed_placeholder_predicate(X) <=> (c(X) & ~c(X))"


if __name__ == "__main__":
    # Example usage

    fol_parser = ChemlogModelChecker()
    llm_for = "tripeptide <=> (oligopeptide & ?[C1, O1, N1, C2, O2, N2]: (c(C1) & o(O1) & bDOUBLE(C1, O1) & n(N1) & bSINGLE(C1, N1) & has_1_hs(N1) & c(C2) & o(O2) & bDOUBLE(C2, O2) & n(N2) & bSINGLE(C2, N2) & has_1_hs(N2) & C1 != C2 & O1 != O2 & N1 != N2 & ![C3, O3, N3]: ((c(C3) & o(O3) & bDOUBLE(C3, O3) & n(N3) & bSINGLE(C3, N3) & has_1_hs(N3) & peptide(C3, O3, N3)) => ((C3 = C1 & O3 = O1 & N3 = N1) | (C3 = C2 & O3 = O2 & N3 = N2)))))"
    fol_parser.parse_definition(llm_for)
    mol = Chem.MolFromSmiles(
        "C(=O)([C@@H](NC(=O)OCC=1C=CC=CC1)C(C)C)N[C@H](C(=O)N[C@H](C(=O)CF)CC(=O)OC)C"
    )
    matches = fol_parser.does_mol_match_tptp_definition(
        mol, fol_parser.parse_definition(llm_for)[1]
    )
    print(
        f"Tripeptide matches definition: {matches == ModelCheckerOutcome.MODEL_FOUND}"
    )
