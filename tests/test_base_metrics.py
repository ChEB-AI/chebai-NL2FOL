from unittest.mock import MagicMock

from rdkit import Chem

from nl_2_fol.inference.learner.base import BaseFOL
from nl_2_fol.inference.preprocessing.c3po_slim_data import ChemicalStructure


def _make_structure(name: str, smiles: str) -> ChemicalStructure:
    return ChemicalStructure(name=name, smiles=smiles, mol=Chem.MolFromSmiles(smiles))


def test_score_definition_uses_definite_outcomes_for_metrics(monkeypatch):
    base_fol = BaseFOL.__new__(BaseFOL)

    base_fol._fol_reasoner = MagicMock()
    base_fol.split = "train"

    chemical_class = MagicMock()
    chemical_class.name = "TestClass"
    chemical_class.id = 123

    pos_all = {
        _make_structure("pos-1", "CCO"),
        _make_structure("pos-2", "CCN"),
        _make_structure("pos-3", "CCC"),
    }
    neg_all = {
        _make_structure("neg-1", "C"),
        _make_structure("neg-2", "CO"),
        _make_structure("neg-3", "CN"),
    }

    unmatched_pos = {"CCN"}
    matched_neg = {"C"}

    monkeypatch.setattr(
        BaseFOL,
        "_get_positive_and_negative_samples",
        lambda self, chemical_class, max_neg_samples: (pos_all, neg_all),
    )
    monkeypatch.setattr(
        "nl_2_fol.inference.learner.base.check_if_definition_matches_samples",
        lambda *args, **kwargs: (
            {
                "matched_pos_samples": {
                    next(s.smiles for s in pos_all if s.smiles == "CCO")
                },
                "unmatched_pos_samples": {
                    next(s.smiles for s in pos_all if s.smiles == "CCN")
                },
                "matched_neg_samples": {
                    next(s.smiles for s in neg_all if s.smiles == "C")
                },
                "unmatched_neg_samples": {
                    next(s.smiles for s in neg_all if s.smiles == "CO")
                },
                "inferred_match_pos": set(),
                "inferred_match_neg": set(),
                "inferred_no_match_pos": set(),
                "inferred_no_match_neg": set(),
                "timeout_pos": set(),
                "timeout_neg": set(),
                "error_pos": set(),
                "error_neg": set(),
                "unknown_pos": set(),
                "unknown_neg": set(),
            },
            {"processed_pos_samples": pos_all, "processed_neg_samples": neg_all},
        ),
    )

    (
        metrics,
        returned_unmatched_pos,
        returned_matched_neg,
        returned_pos,
        returned_neg,
    ) = BaseFOL._score_definition(
        base_fol,
        chemical_class=chemical_class,
        tptp_def=MagicMock(),
        sample_match_timeout_seconds=None,
        max_neg_samples=1000,
        temp_additional_defs=None,
    )

    assert metrics.TP == 1
    assert metrics.FP == 1
    assert metrics.FN == 1
    assert metrics.TN == 1
    assert returned_unmatched_pos == unmatched_pos
    assert returned_matched_neg == matched_neg
    assert returned_pos == pos_all
    assert returned_neg == neg_all
