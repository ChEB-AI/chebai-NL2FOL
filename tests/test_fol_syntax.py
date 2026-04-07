from nl_2_fol.prompting.fol_syntax import (
    math_fol_to_tptp,
    should_use_math_fol_syntax,
    tptp_to_math_fol,
)


def test_should_use_math_fol_syntax_for_mistral() -> None:
    assert should_use_math_fol_syntax("custom", "Mistral-Small-24B-Instruct-nl-to-fol")
    assert should_use_math_fol_syntax("ollama", "my-mistral")
    assert not should_use_math_fol_syntax("openai", "gpt-4o")


def test_math_fol_to_tptp_converts_quantifiers_and_connectives() -> None:
    formula = "test(x) ↔ (p(x) ∧ ∀x,y (q(x, y) → ∃z (r(z) ∨ ¬s(x))))"
    converted = math_fol_to_tptp(formula)

    assert "<=>" in converted
    assert "&" in converted
    assert "=>" in converted
    assert "|" in converted
    assert "~" in converted
    assert "![X, Y]:" in converted
    assert "?[Z]:" in converted


def test_tptp_to_math_fol_converts_quantifiers_and_connectives() -> None:
    formula = "test(X) <=> (p(X) & ![Y]: (q(Y) => ?[Z]: (r(Z) | ~s(Y))))"
    converted = tptp_to_math_fol(formula)

    assert "↔" in converted
    assert "∧" in converted
    assert "→" in converted
    assert "∨" in converted
    assert "¬" in converted
    assert "∀Y" in converted
    assert "∃Z" in converted
