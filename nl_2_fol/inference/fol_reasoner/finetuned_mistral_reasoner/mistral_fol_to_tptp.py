import re

from gavel.logic import logic

from nl_2_fol.inference.fol_reasoner.finetuned_mistral_reasoner.cfg import (
    CFGParser,
)
from nl_2_fol.inference.fol_reasoner.model_check_molecule import GavelFOLReasoner
from nl_2_fol.inference.learner.custom_exceptions import parse_exception


class MistralCustomFOLReasoner(GavelFOLReasoner):
    def __init__(self):
        super().__init__()
        self.math_to_tptp_parser = CFGParser()

    @parse_exception
    def parse_definition(
        self, formula: str
    ) -> tuple[list[logic.Variable], logic.QuantifiedFormula]:
        try:
            tptp_formula = self.math_fol_to_tptp_fol(formula)
        except Exception as e:
            raise Exception(
                f"Error parsing formula '{formula}' to TPTP syntax: {str(e)}"
            )

        return super().parse_definition(tptp_formula)

    def math_fol_to_tptp_fol(self, math_fol_formula: str) -> str:
        ast_tree = self.math_to_tptp_parser.parse(math_fol_formula)
        tptp_formula = ast_tree.to_tptp()
        return self._convert_to_chemlog_predicates(tptp_formula)

    def _convert_to_chemlog_predicates(self, formula: str) -> str:
        """Map parser/TPTP-rendered predicate tokens back to canonical names."""

        fixed = formula

        # Has1Hs -> has_1_hs,  Has2Hs -> has_2_hs, etc.
        fixed = re.sub(r"\bHas(\d+)Hs\b", r"has_\1_hs", fixed, flags=re.IGNORECASE)

        # HasAtLeast1Hs -> has_at_least_1_hs,  HasAtLeast2Hs -> has_at_least_2_hs, etc.
        fixed = re.sub(
            r"\bHasAtLeast(\d+)Hs\b",
            r"has_at_least_\1_hs",
            fixed,
            flags=re.IGNORECASE,
        )
        # HasMin1Hs -> has_min_1_hs,  HasMin2Hs -> has_min_2_hs, etc.
        fixed = re.sub(
            r"\bHasMin(\d+)Hs\b",
            r"has_min_\1_hs",
            fixed,
            flags=re.IGNORECASE,
        )

        for src, dst in _CUSTOM_TO_CHEMLOG_PREDICATE_MAP.items():
            fixed = re.sub(rf"\b{re.escape(src)}\b", dst, fixed, flags=re.IGNORECASE)

        # If a predicate's lowercase form is an atom predicate, force lowercase.
        # The above is not needed as cfg parser should already have converted predicate tokens to lowercase,

        return fixed

    @property
    def dummy_formula(self) -> str:
        return "FailedPlaceholderPredicate ↔ (∃x (C(x) ∧ ¬C(x)))"


_CUSTOM_TO_CHEMLOG_PREDICATE_MAP = {
    "Charge0": "charge0",
    "Charge1": "charge1",
    "Charge2": "charge2",
    "Charge3": "charge3",
    "ChargeM1": "charge_m1",
    "ChargeM2": "charge_m2",
    "ChargeM3": "charge_m3",
    "ChargeP": "charge_p",
    "ChargeN": "charge_n",
    "CipCodeR": "cip_code_R",
    "CipCodeS": "cip_code_S",
    "HasBondTo": "has_bond_to",
    "BSINGLE": "bSINGLE",
    "BDOUBLE": "bDOUBLE",
    "BTRIPLE": "bTRIPLE",
    "BAROMATIC": "bAROMATIC",
    "NetChargePositive": "net_charge_positive",
    "NetChargeNegative": "net_charge_negative",
    "NetChargeNeutral": "net_charge_neutral",
}

if __name__ == "__main__":
    reasoner = MistralCustomFOLReasoner()
    formula = "Azide ↔ (NitrogenMolecularEntity ∧ ∃x ∃y ∃z (N(x) ∧ Charge0(x) ∧ N(y) ∧ Charge1(y) ∧ N(z) ∧ ChargeM1(z) ∧ BDOUBLE(x, y) ∧ BDOUBLE(y, z)))"
    print("Testing math_fol_to_tptp_fol:")
    tptp_formula = reasoner.math_fol_to_tptp_fol(formula)
    print(type(tptp_formula))
    print(tptp_formula)
    print("")

    print("\nTesting get_tptp_fol_definition:")
    variables, tptp_fol_formula = reasoner.parse_definition(formula)
    print("Variables:", variables)
    print("TPTP FOL Formula:", tptp_fol_formula)
