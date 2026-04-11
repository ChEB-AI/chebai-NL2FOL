from gavel.logic import logic

from cfg.cfgparser import CFGParser
from nl_2_fol.inference.fol_reasoner.model_check_molecule import GavelFOLReasoner
from nl_2_fol.inference.learner.custom_exceptions import tptp_parse_exception


class MistralCustomFOLReasoner(GavelFOLReasoner):
    def __init__(self):
        super().__init__()
        self.math_to_tptp_parser = CFGParser()

    @tptp_parse_exception
    def get_tptp_fol_definition(
        self, formula: str
    ) -> tuple[list[logic.Variable], logic.QuantifiedFormula]:
        try:
            tptp_formula = self.math_fol_to_tptp_fol(formula)
        except Exception as e:
            raise Exception(
                f"Error parsing formula '{formula}' to TPTP syntax: {str(e)}"
            )

        return super().get_tptp_fol_definition(tptp_formula)

    def math_fol_to_tptp_fol(self, math_fol_formula: str) -> str:
        ast_tree = self.math_to_tptp_parser.parse(math_fol_formula)
        return ast_tree.to_tptp()


if __name__ == "__main__":
    reasoner = MistralCustomFOLReasoner()
    formula = "Azide(1) ↔ (NitrogenMolecularEntity(1) ∧ ∃x ∃y ∃z (N(x) ∧ Charge0(x) ∧ N(y) ∧ Charge1(y) ∧ N(z) ∧ ChargeM1(z) ∧ BDOUBLE(x, y) ∧ BDOUBLE(y, z)))"
    print("Testing math_fol_to_tptp_fol:")
    tptp_formula = reasoner.math_fol_to_tptp_fol(formula)
    print(type(tptp_formula))
    print(tptp_formula)
    print("")

    print("\nTesting get_tptp_fol_definition:")
    variables, tptp_fol_formula = reasoner.get_tptp_fol_definition(formula)
    print("Variables:", variables)
    print("TPTP FOL Formula:", tptp_fol_formula)
