"""Tests for the clingo-based evaluation of ASP definitions."""

import pandas as pd
import pytest
from rdkit import Chem

pytest.importorskip("clingo")

from nl_2_fol.inference.fol_reasoner.asp_model_checker import (  # noqa: E402
    ASPModelChecker,
)
from nl_2_fol.inference.fol_reasoner.clingo_utils import (  # noqa: E402
    ClingoTimeoutError,
    evaluate_with_clingo,
    ground_extensions,
    match_chunks_with_clingo,
    molecules_to_asp_facts,
    split_prolog_literals,
)
from nl_2_fol.inference.learner import sample_matching_worker  # noqa: E402
from nl_2_fol.inference.preprocessing.c3po_slim_data import (  # noqa: E402
    ChemicalClass,
    ChemicalStructure,
)

ETHANOL = "CCO"
ETHANE_DIOL = "OCCO"
ACETIC_ACID = "CC(=O)O"
METHANE = "C"


def _mols_df(smiles_list):
    return pd.DataFrame(
        {
            "mol": [Chem.MolFromSmiles(s) for s in smiles_list],
            "smiles": smiles_list,
        }
    )


def _structure(smiles):
    return ChemicalStructure(name=smiles, smiles=smiles, mol=Chem.MolFromSmiles(smiles))


class TestClingoUtils:
    def test_split_prolog_literals(self):
        body = "has_atom(M, A), c(A), #count{ X : has_atom(M, X), o(X) } = 2, A != B"
        assert split_prolog_literals(body) == [
            "has_atom(M, A)",
            "c(A)",
            "#count{ X : has_atom(M, X), o(X) } = 2",
            "A != B",
        ]

    def test_molecules_to_asp_facts_ethanol(self):
        facts = set(molecules_to_asp_facts(_mols_df([ETHANOL])))
        # atoms are numbered from 1, the molecule id is the index (0)
        for fact in [
            "has_atom(0,a0_1).",
            "has_atom(0,a0_2).",
            "has_atom(0,a0_3).",
            "c(a0_1).",
            "o(a0_3).",
            "bSINGLE(a0_2,a0_3).",
            "has_1_hs(a0_3).",
            "charge0(a0_1).",
            "net_charge_neutral(0).",
        ]:
            assert fact in facts, f"{fact} missing in {sorted(facts)}"

    def test_facts_are_restricted_to_prompt_predicates(self):
        # benzene ring: chebi_utils could produce ring / aromaticity predicates
        df = _mols_df(["c1ccccc1O"])
        predicates = {
            fact.split("(")[0] for fact in ASPModelChecker()._molecules_df_to_bk(df)
        }
        assert {"has_atom", "c", "o", "bAROMATIC", "has_1_hs"} <= predicates
        for unexpected in ["in_ring", "in_ring6", "ring6", "aromatic", "aliphatic"]:
            assert unexpected not in predicates
        assert predicates <= set(ASPModelChecker()._base_predicates)

    def test_evaluate_with_clingo(self):
        df = _mols_df([ETHANOL, METHANE, ETHANE_DIOL])
        rules = [
            "hasOxygen(M) :- has_atom(M, A), o(A).",
            "diol(M) :- net_charge_neutral(M), #count{ O : has_atom(M, O), o(O), has_1_hs(O) } = 2.",
        ]
        positives = evaluate_with_clingo(
            rules, molecules_to_asp_facts(df), ["hasOxygen", "diol"], df.index
        )
        assert positives == {"hasOxygen": [0, 2], "diol": [2]}

    def test_timeout_covers_grounding(self):
        # grounding this program takes much longer than the time limit
        rules = ["blowup(X) :- n(X), n(Y), n(Z), n(W), X != Y, Y != Z, Z != W."]
        facts = ["n(1..1000)."]
        with pytest.raises(ClingoTimeoutError):
            ground_extensions(rules, facts, ["blowup"], timeout=1)

    def test_clingo_errors_are_raised(self):
        with pytest.raises(RuntimeError):
            ground_extensions(["p(X) :- q(X"], ["q(1)."], ["p"], timeout=10)


class TestASPModelCheckerBackground:
    @pytest.fixture
    def reasoner(self):
        reasoner = ASPModelChecker()
        for definition in [
            "alcohol(M) :- has_atom(M, O), o(O), has_1_hs(O), bSINGLE(O, C), c(C).",
            "diol(M) :- alcohol(M), #count{ O : has_atom(M, O), hydroxy(O) } = 2.",
            "hydroxy(O) :- o(O), has_1_hs(O).",
            "unrelated(M) :- has_atom(M, A), cl(A).",
            # an old definition for the class that is being re-learned
            "carboxylicAcid(M) :- has_atom(M, A), c(A).",
        ]:
            reasoner.add_background_definition(reasoner.parse_definition(definition))
        return reasoner

    def test_does_not_modify_shared_base_predicates(self):
        from nl_2_fol.inference.fol_reasoner.base_predicates import GAVEL_PREDICATES

        ASPModelChecker()
        assert "has_atom" not in GAVEL_PREDICATES

    def test_required_background_rules_are_transitive(self, reasoner):
        rules = reasoner._get_required_background_rules(
            "vicinalDiol(M) :- diol(M).", "vicinalDiol"
        )
        assert rules == [
            reasoner.background_definitions["alcohol"].definition,
            reasoner.background_definitions["diol"].definition,
            reasoner.background_definitions["hydroxy"].definition,
        ]

    def test_undefined_predicate_in_aggregate_is_detected(self, reasoner):
        from nl_2_fol.inference.learner.custom_exceptions import (
            MissingPredicateException,
        )

        definition = (
            "triol(M) :- alcohol(M), #count{ O : has_atom(M, O), hydroxyGroup(O) } = 3."
        )
        assert reasoner.extract_unknown_predicates(definition) == {"hydroxyGroup"}
        with pytest.raises(MissingPredicateException):
            reasoner.do_molecules_match_asp_definition(_mols_df([ETHANOL]), definition)

    def test_target_predicate_background_rule_is_excluded(self, reasoner):
        # otherwise, clingo would evaluate the disjunction of old and new definition
        definition = (
            "carboxylicAcid(M) :- has_atom(M, C), c(C), has_atom(M, O1), o(O1), "
            "has_atom(M, O2), o(O2), has_1_hs(O2), bDOUBLE(C, O1), bSINGLE(C, O2)."
        )
        df = _mols_df([ETHANOL, ACETIC_ACID, METHANE])
        matched = reasoner.do_molecules_match_asp_definition(df, definition)
        assert list(matched) == [1]


class TestClingoSampleMatching:
    @pytest.fixture
    def chemical_class(self):
        return ChemicalClass(
            id="23824",
            name="diol",
            all_positive_examples={ETHANE_DIOL},
            num_of_members=1,
        )

    @pytest.mark.parametrize("n_jobs", [1, 3])
    def test_chunked_matching(self, monkeypatch, chemical_class, n_jobs):
        monkeypatch.setattr(sample_matching_worker, "CLINGO_CHUNK_SIZE", 2)
        pos = {_structure(ETHANE_DIOL), _structure("OCC(O)C")}
        neg = {_structure(ETHANOL), _structure(METHANE), _structure(ACETIC_ACID)}
        result, processed = (
            sample_matching_worker.check_if_definition_matches_samples_clingo(
                ASPModelChecker(),
                None,
                chemical_class,
                "diol(M) :- net_charge_neutral(M), #count{ O : has_atom(M, O), o(O), has_1_hs(O) } = 2.",
                pos,
                neg,
                n_jobs=n_jobs,
            )
        )
        assert result["matched_pos_samples"] == {ETHANE_DIOL, "OCC(O)C"}
        assert result["unmatched_neg_samples"] == {ETHANOL, METHANE, ACETIC_ACID}
        assert result["matched_neg_samples"] == set()
        assert processed["processed_pos_samples"] == pos
        assert processed["processed_neg_samples"] == neg

    def test_timed_out_chunks_are_counted_as_timeouts(
        self, monkeypatch, chemical_class
    ):
        monkeypatch.setattr(sample_matching_worker, "CLINGO_CHUNK_SIZE", 1)
        pos = {_structure(ETHANE_DIOL)}
        neg = {_structure(ETHANOL), _structure(METHANE)}

        def _second_chunk_times_out(self, chunks, *args, **kwargs):
            return [
                ClingoTimeoutError("timeout") if i == 1 else []
                for i, _ in enumerate(chunks)
            ]

        monkeypatch.setattr(
            ASPModelChecker, "match_asp_definition_on_chunks", _second_chunk_times_out
        )
        result, processed = (
            sample_matching_worker.check_if_definition_matches_samples_clingo(
                ASPModelChecker(),
                None,
                chemical_class,
                "diol(M) :- has_atom(M, O), o(O).",
                pos,
                neg,
            )
        )
        n_timeouts = len(result["timeout_pos"]) + len(result["timeout_neg"])
        n_processed = len(processed["processed_pos_samples"]) + len(
            processed["processed_neg_samples"]
        )
        assert n_timeouts == 1
        assert n_processed == 2
        # timed out molecules are neither TNs nor FNs
        timed_out = result["timeout_pos"] | result["timeout_neg"]
        assert not timed_out & (
            result["unmatched_neg_samples"] | result["unmatched_pos_samples"]
        )

    def test_all_chunks_timed_out_raises(self, monkeypatch, chemical_class):
        def _always_timeout(self, chunks, *args, **kwargs):
            return [ClingoTimeoutError("timeout") for _ in chunks]

        monkeypatch.setattr(
            ASPModelChecker, "match_asp_definition_on_chunks", _always_timeout
        )
        with pytest.raises(Exception, match="too expensive to evaluate"):
            sample_matching_worker.check_if_definition_matches_samples_clingo(
                ASPModelChecker(),
                None,
                chemical_class,
                "diol(M) :- has_atom(M, O), o(O).",
                {_structure(ETHANE_DIOL)},
                {_structure(ETHANOL)},
            )


class TestParallelChunks:
    def test_matches_sequential_result(self):
        smiles = [ETHANOL, METHANE, ETHANE_DIOL, ACETIC_ACID, "OCC(O)C"]
        df = _mols_df(smiles)
        chunks = [df.iloc[i : i + 2] for i in range(0, len(df), 2)]
        rules = ["hasOxygen(M) :- has_atom(M, A), o(A)."]
        results = match_chunks_with_clingo(rules, chunks, "hasOxygen", n_jobs=2)
        assert [sorted(r) for r in results] == [[0], [2, 3], [4]]

    def test_chunk_timeout(self):
        df = _mols_df([ETHANOL, METHANE])
        chunks = [df.iloc[:1], df.iloc[1:]]
        # grounding blows up for every chunk (t/3 has billions of instances)
        rules = [
            "slow(M) :- has_atom(M, _), t(1, 2, 3).",
            "t(X, Y, Z) :- n(X), n(Y), n(Z), X != Y, Y != Z.",
            "n(1..2000).",
        ]
        results = match_chunks_with_clingo(rules, chunks, "slow", timeout=1, n_jobs=2)
        assert all(isinstance(r, ClingoTimeoutError) for r in results)

    def test_clingo_error_is_raised(self):
        with pytest.raises(RuntimeError):
            match_chunks_with_clingo(
                ["p(M) :- q(M"], [_mols_df([ETHANOL])], "p", timeout=10, n_jobs=1
            )

    def test_clingo_error_details_are_included(self):
        # a common LLM mistake: the head variable only occurs inside the aggregate
        rule = "polyol(M) :- #count{ O : has_atom(M, O), o(O) } >= 2."
        with pytest.raises(RuntimeError) as exc_info:
            match_chunks_with_clingo(
                [rule], [_mols_df([ETHANE_DIOL])], "polyol", timeout=10, n_jobs=1
            )
        message = str(exc_info.value)
        assert "unsafe variables" in message
        assert "'M' is unsafe" in message
