import re
from typing import List

import pandas as pd

from nl_2_fol.inference.fol_reasoner.abstract_model_checker import (
    AbstractModelCheckerWrapper,
    FOLDefinition,
)
from nl_2_fol.inference.fol_reasoner.clingo_utils import (
    ClingoTimeoutError,
    match_chunks_with_clingo,
    molecules_to_asp_facts,
    split_prolog_literals,
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
        # copy to avoid modifying the shared GAVEL_PREDICATES dict
        self._base_predicates = {
            **self._base_predicates,
            "has_atom": "molecule has this atom",
            # produced by chebi_utils and listed in the system prompt
            **{
                f"has_at_least_{n}_hs": f"has at least {n} hydrogens"
                for n in range(1, 5)
            },
        }

    @staticmethod
    def _head_names(formula: str) -> set[str]:
        """Names of the predicates defined (i.e. used in a rule head) in an ASP program.

        A definition can contain helper rules besides the rules for its main predicate.
        """
        heads = set()
        for rule in formula.split("."):
            match = re.match(r"\s*([a-z_]\w*)", rule.split(":-", 1)[0])
            if match:
                heads.add(match.group(1))
        return heads

    def _defining_definitions(
        self, temp_additional_defs: dict[str, ASPDefinition] | None = None
    ) -> dict[str, ASPDefinition]:
        """For each defined predicate (including helpers), the definition that defines it."""
        definitions = {
            **self.background_definitions,
            **(temp_additional_defs or {}),
        }
        index = dict(definitions)
        for d in definitions.values():
            for head in self._head_names(d.definition):
                index.setdefault(head, d)
        return index

    def _check_redefinitions(
        self,
        definition: str,
        target_predicate: str,
        temp_additional_defs: dict[str, ASPDefinition] | None = None,
    ) -> None:
        """Reject helper rules for predicates that are already defined elsewhere.

        In ASP, rules from different definitions with the same head would act as a
        disjunction once both definitions are used together.
        """
        index = self._defining_definitions(temp_additional_defs)
        for head in sorted(self._head_names(definition) - {target_predicate}):
            if head in self._base_predicates:
                raise Exception(
                    f"`{head}` is given as a fact and must not be defined by rules."
                )
            owner = index.get(head)
            if owner is not None and owner.predicate_name != target_predicate:
                raise Exception(
                    f"The predicate `{head}` is defined in your answer, but it is already "
                    f"defined as part of `{owner.predicate_name}`:\n{owner.definition}\n"
                    f"Use `{head}` without redefining it, or give your helper predicate "
                    "a different name."
                )

    def extract_unknown_predicates(
        self,
        formula: str,
        temp_additional_defs: dict[str, ASPDefinition] | None = None,
    ) -> set[str]:
        """Predicates used in the formula that are neither given, nor defined in the
        formula itself, nor defined by a (background or temporary) definition."""
        return (
            self._extract_predicate_names(formula)
            - self._base_predicates.keys()
            - self._head_names(formula)
            - self._defining_definitions(temp_additional_defs).keys()
        )

    def _get_required_background_rules(
        self,
        definition: str,
        target_predicate: str,
        temp_additional_defs: dict[str, ASPDefinition] | None = None,
    ) -> list[str]:
        """Background rules the definition (transitively) depends on.

        Only these rules are passed to clingo: grounding all background rules would make
        every check as expensive as the most expensive learned rule. Background rules for
        the target predicate itself are excluded - in ASP, several rules with the same
        head act as a disjunction, so an old definition would change the result.
        """
        index = self._defining_definitions(temp_additional_defs)
        required: dict[str, str] = {}  # main predicate -> definition
        visited: set[str] = set()
        frontier = list(
            self._extract_predicate_names(definition) - self._head_names(definition)
        )
        while frontier:
            predicate = frontier.pop()
            if predicate in visited or predicate == target_predicate:
                continue
            visited.add(predicate)
            owner = index.get(predicate)
            if owner is None or owner.predicate_name == target_predicate:
                continue
            if owner.predicate_name not in required:
                required[owner.predicate_name] = owner.definition
                frontier.extend(self._extract_predicate_names(owner.definition))
        return [required[p] for p in sorted(required)]

    @parse_exception
    def parse_definition(self, definition: str) -> ASPDefinition:
        """
        Parse a definition string into an ASPDefinition object. Extract the predicate name and variables from the head of the rule.
        For the whole rule, check that it is a valid ASP rule (e.g., using clingo's parser).
        """
        if ":" in definition and ":-" not in definition:
            raise Exception("Invalid rule implication: use ':-' instead of ':'")
        # For simplicity, let's assume the definition is of the form:
        # "predicate_name(X1, X2) :- body."
        # We will extract the predicate name and variables from the head of the rule.
        head = definition.split(":-")[0]
        head = head.strip()
        predicate_name = head.split("(")[0].strip()
        variables = []
        open_paren = head.find("(")
        if open_paren != -1:
            depth = 0
            close_paren = -1
            for idx in range(open_paren, len(head)):
                ch = head[idx]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        close_paren = idx
                        break
            if close_paren == -1:
                raise Exception("Malformed rule head: unbalanced parentheses")
            args = head[open_paren + 1 : close_paren].strip()
            if args:
                variables = [var.strip() for var in args.split(",")]

        from clingo.ast import parse_string

        # clingo only raises "syntax error"; the location and cause go to the logger
        messages = []
        try:
            parse_string(
                definition if definition.strip().endswith(".") else f"{definition}.",
                lambda stm: None,
                logger=lambda code, msg: messages.append(msg.strip()),
            )  # We just want to check if it parses, not do anything with the AST
        except Exception as e:
            details = "\n".join([str(e), *messages])
            print(f"Error parsing formula with clingo: {details}")
            raise Exception(f"Error parsing formula with clingo:\n{details}")

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
            ClingoTimeoutError: If grounding / solving exceeds ``timeout`` seconds.
            Exception: For various model checking failures such as predicate arity
                mismatches or normalization errors.
        """
        result = self.match_asp_definition_on_chunks(
            [molecules_df],
            definition_to_match,
            temp_additional_defs=temp_additional_defs,
            timeout=timeout,
        )[0]
        if isinstance(result, ClingoTimeoutError):
            raise result
        return result

    @model_check_exception
    def match_asp_definition_on_chunks(
        self,
        chunks: list[pd.DataFrame],
        definition_to_match: str,
        temp_additional_defs: dict[str, ASPDefinition] | None = None,
        timeout: float | None = None,
        n_jobs: int = 1,
    ) -> list[list | ClingoTimeoutError]:
        """Like :meth:`do_molecules_match_asp_definition`, for several chunks of molecules.

        The chunks are evaluated in parallel (``n_jobs`` processes), ``timeout`` applies
        to each chunk individually. Molecule IDs (the index) have to be unique across chunks.

        Returns:
            For each chunk, the IDs of matching molecules or a ``ClingoTimeoutError``.
        """
        missing_predicates = self.extract_unknown_predicates(
            definition_to_match, temp_additional_defs
        )
        if missing_predicates:
            raise MissingPredicateException(missing_predicates)
        self._check_redefinitions(
            definition_to_match,
            self.parse_definition(definition_to_match).predicate_name,
            temp_additional_defs,
        )

        exception_prefix = (
            "MODEL CHECKING FAILED - Error during model checking for the formula.\n"
            f"Parsed Formula being checked: `{definition_to_match}`\n\n"
            "[IMPORTANT] Critical error details for analysis: \n"
        )
        try:
            asp_def = self.parse_definition(definition_to_match)
            rules = [definition_to_match] + self._get_required_background_rules(
                definition_to_match, asp_def.predicate_name, temp_additional_defs
            )
            return match_chunks_with_clingo(
                rules,
                chunks,
                asp_def.predicate_name,
                allowed_predicates=set(self._base_predicates),
                timeout=timeout,
                n_jobs=n_jobs,
            )
        except Exception as e:
            print(f"Error occured for formula: {definition_to_match}")
            raise Exception(f"{exception_prefix}{e}")

    @mol_to_fol_exception
    def _molecules_df_to_bk(self, molecules_df: pd.DataFrame) -> list[str]:
        """Convert a dataframe of RDKit molecules to ASP background facts.

        Only predicates that are known to the model checker (and described in the
        prompt) are included.
        """
        return molecules_to_asp_facts(
            molecules_df, allowed_predicates=set(self._base_predicates)
        )

    def _extract_predicate_names(self, formula: str) -> set[str]:
        """Extract all predicate names from an ASP formula."""
        predicates = set()

        for rule in formula.split("."):
            rule = rule.strip()
            if not rule:
                continue
            body = ""
            if ":-" in rule:
                body = rule.split(":-", 1)[1]
            elif ":" in rule:
                body = rule.split(":", 1)[1]
            predicates.update(self._extract_predicate_names_from_body(body))

        return predicates

    def _extract_predicate_names_from_body(self, body: str) -> set[str]:
        predicates = set()
        for literal in split_prolog_literals(body):
            if not literal:
                continue
            if "#" in literal:
                # aggregates, e.g. #count{X : p(X), q(X)} = 2
                # extract the conditions of each aggregate element (separated by ";")
                inner = literal[literal.find("{") + 1 : literal.rfind("}")]
                for element in inner.split(";"):
                    if ":" in element:
                        predicates.update(
                            self._extract_predicate_names_from_body(
                                element.split(":", 1)[1]
                            )
                        )
                continue
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
