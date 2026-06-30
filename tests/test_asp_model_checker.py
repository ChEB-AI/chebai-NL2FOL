"""Tests for ASPModelChecker."""

import pytest

from nl_2_fol.inference.fol_reasoner.asp_model_checker import (
    ASPDefinition,
    ASPModelChecker,
)


class TestASPModelChecker:
    """Test suite for ASPModelChecker."""

    @pytest.fixture
    def reasoner(self):
        """Create an ASPModelChecker instance for testing."""
        pytest.importorskip("clingo")
        return ASPModelChecker()

    def test_parse_definition_success(self, reasoner: ASPModelChecker):
        """Valid ASP rules should parse into an ASPDefinition."""
        definition = (
            "diol(M) :- has_atom(M, O1), o(O1), has_atom(M, O2), o(O2), O1 != O2."
        )

        parsed_definition = reasoner.parse_definition(definition)

        assert isinstance(parsed_definition, ASPDefinition)
        assert parsed_definition.predicate_name == "diol"
        assert parsed_definition.variables == ["M"]
        assert parsed_definition.definition == definition

    def test_parse_definition_multiple_head_variables(self, reasoner: ASPModelChecker):
        """The parser should extract all variables from the rule head."""
        definition = "bonded_pair(M, A1, A2) :- has_atom(M, A1), has_bond_to(A1, A2)."

        parsed_definition = reasoner.parse_definition(definition)

        assert parsed_definition.predicate_name == "bonded_pair"
        assert parsed_definition.variables == ["M", "A1", "A2"]
        assert parsed_definition.definition == definition

    def test_parse_definition_invalid_rule_raises(self, reasoner: ASPModelChecker):
        """Invalid ASP syntax should raise a parsing error."""
        invalid_definition = "broken_pred(M) :- has_atom(M, O1), o(O1"

        with pytest.raises(Exception):
            reasoner.parse_definition(invalid_definition)

    def test_parse_definition_nullary_head_has_no_variables(
        self, reasoner: ASPModelChecker
    ):
        """Nullary ASP heads should not produce synthetic variables."""
        definition = "diol :- has_atom(m1, o1), o(o1)."

        parsed_definition = reasoner.parse_definition(definition)

        assert parsed_definition.predicate_name == "diol"
        assert parsed_definition.variables == []
        assert parsed_definition.definition == definition

    def test_parse_definition_nested_head_term_keeps_full_argument(
        self, reasoner: ASPModelChecker
    ):
        """Head argument extraction should not stop at nested closing parens."""
        definition = "wrapped(f(X)) :- has_atom(X, O1), o(O1)."

        parsed_definition = reasoner.parse_definition(definition)

        assert parsed_definition.predicate_name == "wrapped"
        assert parsed_definition.variables == ["f(X)"]
        assert parsed_definition.definition == definition

    def test_parse_definition_failure(self, reasoner: ASPModelChecker):
        """Valid ASP rules should parse into an ASPDefinition."""
        definition = (
            "diol(M) : has_atom(M, O1), o(O1), has_atom(M, O2), o(O2), O1 != O2."
        )

        parsed_definition = reasoner.parse_definition(definition)

        assert isinstance(parsed_definition, ASPDefinition)
        assert parsed_definition.predicate_name == "diol"
        assert parsed_definition.variables == ["M"]
        assert parsed_definition.definition == definition

    def test_parse_definition_without_period(self, reasoner: ASPModelChecker):
        # with "." at the end
        definition = (
            "diol(M) :- has_atom(M, O1), o(O1), has_atom(M, O2), o(O2), O1 != O2"
        )

        reasoner.parse_definition(definition)

    def test_parse_definition_with_colon(self, reasoner: ASPModelChecker):

        # ":" insteadd of ":-"
        definition = (
            "diol(M) : has_atom(M, O1), o(O1), has_atom(M, O2), o(O2), O1 != O2."
        )
        with pytest.raises(Exception):
            reasoner.parse_definition(definition)

    def test_parse_definition_different_implication_symbol(
        self, reasoner: ASPModelChecker
    ):
        with pytest.raises(Exception):
            definition = (
                "diol(M) => has_atom(M, O1), o(O1), has_atom(M, O2), o(O2), O1 != O2."
            )
            reasoner.parse_definition(definition)

    def test_extract_predicate_names_from_rule_body(self, reasoner: ASPModelChecker):
        """Predicate extraction should return body predicates only."""
        pytest.importorskip("chebILP.utils")
        formula = (
            "diol(M) :- has_atom(M, O1), o(O1), not charged(O1), "
            "has_atom(M, O2), o(O2), O1 != O2."
        )

        predicates = reasoner._extract_predicate_names(formula)

        assert predicates == {"has_atom", "o", "charged"}

    def test_list_predicates_with_other_implication_symbol(
        self, reasoner: ASPModelChecker
    ):
        """Predicate extraction should return body predicates only."""
        pytest.importorskip("chebILP.utils")
        formula = (
            "diol(M) : has_atom(M, O1), o(O1), not charged(O1), "
            "has_atom(M, O2), o(O2), O1 != O2."
        )

        predicates = reasoner._extract_predicate_names(formula)

        assert predicates == {"has_atom", "o", "charged"}
