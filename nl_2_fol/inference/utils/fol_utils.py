from gavel.logic import logic

if __name__ == "__main__":
    from chemlog.fol_classification.fol_utils import normalize_fol_formula

    demo_formula = logic.QuantifiedFormula(
        logic.Quantifier.EXISTENTIAL,
        [logic.Variable("x")],
        logic.BinaryFormula(
            logic.PredicateExpression("my_pred", [logic.Variable("x")]),
            logic.BinaryConnective.IMPLICATION,
            logic.QuantifiedFormula(
                logic.Quantifier.EXISTENTIAL,
                [logic.Variable("x")],
                logic.BinaryFormula(
                    logic.PredicateExpression("other_pred", [logic.Variable("x")]),
                    logic.BinaryConnective.CONJUNCTION,
                    logic.PredicateExpression("third_pred", [logic.Variable("x")]),
                ),
            ),
        ),
    )
    print("Before normalization:")
    print(demo_formula)
    print("\nAfter normalization:")
    print(normalize_fol_formula(demo_formula))
