from typing import List

import pandas as pd

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


class ASPDefinition(FOLDefinition):
    def __init__(self, predicate_name: str, variables: List[str], definition: str):
        super().__init__(predicate_name, variables, definition)


class ASPModelChecker(AbstractModelCheckerWrapper):
    def __init__(self):
        super().__init__()
        self._base_predicates["has_atom"] = "molecule has this atom"

    @parse_exception
    def parse_definition(self, definition: str) -> ASPDefinition:
        """
        Parse a definition string into an ASPDefinition object. Extract the predicate name and variables from the head of the rule.
        For the whole rule, check that it is a valid ASP rule (e.g., using clingo's parser).
        """
        # For simplicity, let's assume the definition is of the form:
        # "predicate_name(X1, X2) :- body."
        # We will extract the predicate name and variables from the head of the rule.
        head = definition.split(":-")[0]
        head = head.strip()
        predicate_name = head.split("(")[0].strip()
        variables = [
            var.strip() for var in head[head.find("(") + 1 : head.find(")")].split(",")
        ]

        from clingo.ast import parse_string

        try:
            parse_string(
                definition, lambda stm: None
            )  # We just want to check if it parses, not do anything with the AST
        except Exception as e:
            print(f"Error parsing formula with clingo: {e}")
            raise Exception(f"Error parsing formula with clingo:\n{e}")

        return ASPDefinition(predicate_name, variables, definition)

    @model_check_exception
    def do_molecules_match_asp_definition(
        self,
        molecules_df: pd.DataFrame,
        definition_to_match: str,
        temp_additional_defs: dict[str, ASPDefinition] | None = None,
        timeout: int | None = None,
    ) -> List:
        """Checks for each molecule if it matches a logical definition using model checking.

        Converts the molecule to a first-order logic representation and uses a model
        checker to determine if the molecule satisfies the given FOL formula.

        Args:
            molecules_df: DataFrame containing RDKit molecule objects to be checked (Index is ChEBI ID, needs a 'mol' column with RDKit Mol objects).
            definition_to_match: Logic Program representing
                the chemical class definition to match against.
            temp_additional_defs: Optional temporary background definitions to use
                during this specific check, in addition to the instance's persistent
                background definitions. Useful for validating additional predicates
                before committing them permanently.

        Returns:
            List: IDs of molecules that match the definition.

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

        background_facts = self._molecules_df_to_bk(molecules_df)
        bck_def = {
            **self.background_definitions,
            **(temp_additional_defs or {}),
        }
        rules = [definition_to_match] + [d.definition for _, d in bck_def.items()]

        exception_prefix = (
            "MODEL CHECKING FAILED - Error during model checking for the formula.\n"
            f"Parsed Formula being checked: `{definition_to_match}`\n\n"
            "[IMPORTANT] Critical error details for analysis: \n"
        )
        try:
            from chebILP.clingo_eval import evaluate_with_clingo

            asp_def = self.parse_definition(definition_to_match)
            positives = evaluate_with_clingo(
                rules,
                background_facts,
                [asp_def.predicate_name],
                molecules_df.index,
                timeout=timeout,
            )

        except Exception as e:
            print(f"Error occured for formula: {definition_to_match}")
            raise Exception(f"{exception_prefix}{e}")
        return (
            positives[asp_def.predicate_name]
            if asp_def.predicate_name in positives
            else []
        )

    @mol_to_fol_exception
    def _molecules_df_to_bk(self, molecules_df: pd.DataFrame) -> list[str]:
        """Convert a dataframe of RDKit molecules to ASP background facts."""
        from chebILP.ilp_problem_builder import build_background_chemlog

        return build_background_chemlog(molecules_df)[0]

    def _extract_predicate_names(self, formula: str) -> set[str]:
        """Extract all predicate names from an ASP formula."""
        predicates = set()

        from chebILP.utils import split_prolog_literals

        for rule in formula.split("."):
            rule = rule.strip()
            if not rule:
                continue
            body = rule.split(":-")[1] if ":-" in rule else ""
            literals = split_prolog_literals(body)
            for literal in literals:
                if "#" in literal:
                    # aggregators, e.g. #count{X : p(X), q(X)} = 2
                    # extract the inner part of the aggregator and recursively extract predicate names from it
                    literal_inner = (
                        literal[literal.find("{") + 1 : literal.rfind("}")]
                        .split(":")[1]
                        .strip()
                    )
                    inner_predicates = self._extract_predicate_names(literal_inner)
                    predicates.update(inner_predicates)
                if "=" in literal or ">" in literal or "<" in literal:
                    continue
                if literal.startswith("not "):
                    literal = literal[4:].strip()
                pred_name = literal.split("(")[0].strip()
                predicates.add(pred_name)

        return predicates

    @property
    def dummy_formula(self) -> str:
        return "failed_placeholder_predicate(M) :- c(M), not c(M)."


if __name__ == "__main__":
    # Example usage
    asp_checker = ASPModelChecker()
    definition = "diol(M) :- has_atom(M, O1), o(O1), has_1_hs(O1), has_bond_to(O1, C1), c(C1), has_atom(M, O2), o(O2), has_1_hs(O2), has_bond_to(O2, C2), c(C2), O1 != O2, C1 != C2."

    asp_checker.do_molecules_match_asp_definition(
        molecules_df=pd.DataFrame(),
        definition_to_match=definition,
        temp_additional_defs=None,
    )
