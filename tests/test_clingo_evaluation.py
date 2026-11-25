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

    def test_cached_facts_are_identical(self, monkeypatch):
        from nl_2_fol.inference.fol_reasoner import clingo_utils

        monkeypatch.setattr(clingo_utils, "_FACTS_CACHE", {})
        monkeypatch.setattr(clingo_utils, "_FACTS_CACHE_PREDICATES", None)
        allowed = set(ASPModelChecker()._base_predicates)
        df = _mols_df([ETHANOL, ACETIC_ACID]).set_index(pd.Index([7, 8]))
        expected = molecules_to_asp_facts(df, allowed)
        clingo_utils.cache_molecule_facts(
            dict(zip(df["smiles"], df["mol"])), allowed, n_jobs=2
        )
        assert len(clingo_utils._FACTS_CACHE) == 2
        assert molecules_to_asp_facts(df, allowed) == expected

    def test_body_literals_are_reordered(self):
        from nl_2_fol.inference.fol_reasoner.asp_rule_order import (
            reorder_body_literals,
        )

        rule = (
            "cc(M) :- has_atom(M, A), has_atom(M, B), c(A), c(B), "
            "has_bond_to(A, B), A != B, #count{ X : has_atom(M, X) } > 1."
        )
        reordered = reorder_body_literals(rule)
        # each new atom is reached via a bond, not by enumerating the molecule
        assert reordered.index("has_bond_to(A,B)") < reordered.index("has_atom(M,B)")
        df = _mols_df([ETHANOL, METHANE, ETHANE_DIOL])
        facts = molecules_to_asp_facts(df)
        assert evaluate_with_clingo([rule], facts, ["cc"], df.index) == {"cc": [0, 2]}


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


class TestStoredResults:
    # the stored results deliberately differ from what the rules derive (ethanol and
    # ethane diol are alcohols), so that the tests show which one is used
    @pytest.fixture
    def reasoner(self):
        reasoner = ASPModelChecker()
        for definition in [
            "alcohol(M) :- has_atom(M, O), o(O), has_1_hs(O), bSINGLE(O, C), c(C).",
            "diol(M) :- alcohol(M), #count{ O : has_atom(M, O), hydroxy(O) } = 2.",
            "hydroxy(O) :- o(O), has_1_hs(O).",
        ]:
            reasoner.add_background_definition(reasoner.parse_definition(definition))
        return reasoner

    @pytest.fixture
    def df(self):
        return _mols_df([ETHANOL, METHANE, ETHANE_DIOL])

    def test_stored_results_replace_rules(self, reasoner, df):
        reasoner.store_definition_results(
            "alcohol", frozenset(df["smiles"]), {METHANE}, set()
        )
        assert reasoner.do_molecules_match_asp_definition(
            df, "x(M) :- alcohol(M)."
        ) == [1]

    def test_not_used_after_dependency_changed(self, reasoner, df):
        reasoner.store_definition_results(
            "diol", frozenset(df["smiles"]), {METHANE}, set()
        )
        assert reasoner.do_molecules_match_asp_definition(df, "x(M) :- diol(M).") == [1]
        reasoner.add_background_definition(
            reasoner.parse_definition("alcohol(M) :- has_atom(M, O), o(O).")
        )
        assert reasoner.do_molecules_match_asp_definition(df, "x(M) :- diol(M).") == [2]

    def test_not_used_for_other_molecules(self, reasoner, df):
        reasoner.store_definition_results(
            "alcohol", frozenset({ETHANOL, METHANE}), {METHANE}, set()
        )
        assert reasoner.do_molecules_match_asp_definition(
            df, "x(M) :- alcohol(M)."
        ) == [0, 2]

    def test_unknown_molecules(self, reasoner, df):
        reasoner.store_definition_results(
            "alcohol", frozenset(df["smiles"]), {METHANE}, {ETHANOL}
        )
        # without a set for unknown molecules, the rules are used
        assert reasoner.do_molecules_match_asp_definition(
            df, "x(M) :- alcohol(M)."
        ) == [0, 2]
        unknown_ids = set()
        results = reasoner.match_asp_definition_on_chunks(
            [df], "x(M) :- alcohol(M).", unknown_ids=unknown_ids
        )
        assert results == [[1]]
        assert unknown_ids == {0}

    def test_learner_derives_results_from_metrics(self):
        from types import SimpleNamespace

        from nl_2_fol.inference.learner.definition_model import DefinitionMetrics
        from nl_2_fol.inference.learner.learn_definitions import LearnDefinitions

        reasoner = ASPModelChecker()
        reasoner.add_background_definition(
            reasoner.parse_definition("chebi_1(M) :- has_atom(M, O), o(O).")
        )
        tp, fn, pos_timeout, fp, tn = "CO", "OCCO", "CCO", "OC=O", "C"
        stub = SimpleNamespace(
            fol_reasoner="asp",
            split="train",
            _fol_reasoner=reasoner,
            _get_positive_and_negative_samples=lambda _: (
                {_structure(s) for s in [tp, fn, pos_timeout]},
                {_structure(s) for s in [fp, tn]},
            ),
        )
        metrics = DefinitionMetrics(
            TP=1, FP=1, FN=1, TN=1, F1=0.5, PPV=0.5, NPV=0.5,
            fp_smiles=[fp], fn_smiles=[fn], timeout_smiles=[pos_timeout],
        )  # fmt: skip
        LearnDefinitions._store_definition_results(
            stub, SimpleNamespace(name="chebi_1"), metrics, None
        )
        stored = reasoner._stored_results["chebi_1"]
        assert stored.matched == {tp, fp}
        assert stored.unknown == {pos_timeout}
        assert stored.universe == {tp, fn, pos_timeout, fp, tn}


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
