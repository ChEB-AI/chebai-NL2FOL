import re

_MATH_TO_TPTP_OPS = {
    "↔": "<=>",
    "⇔": "<=>",
    "→": "=>",
    "⇒": "=>",
    "∧": "&",
    "∨": "|",
    "¬": "~",
    "⊤": "$true",
    "⊥": "$false",
}


def should_use_math_fol_syntax(platform: str, model_name: str) -> bool:
    """Enable math-style FOL only for Mistral runs."""
    _ = platform  # reserved for future stricter routing
    return "mistral" in model_name.lower()


def get_math_fol_instruction_suffix() -> str:
    return (
        "\n\nWhen writing FOL formulas, use mathematical syntax instead of TPTP syntax.\n"
        "Use symbols like: ∀, ∃, ∧, ∨, ¬, →, ↔.\n"
        "For quantified expressions, always parenthesize the quantified scope.\n"
        "Example: ∀x (p(x) → q(x)) and ∃x,y (r(x) ∧ s(y))."
    )


def tptp_to_math_fol(formula: str) -> str:
    """Convert common TPTP surface syntax to mathematical symbols for prompting."""
    converted = formula

    converted = re.sub(
        r"!\[\s*([^\]]+?)\s*\]\s*:\s*\(",
        lambda m: f"∀{m.group(1).strip()} (",
        converted,
    )
    converted = re.sub(
        r"\?\[\s*([^\]]+?)\s*\]\s*:\s*\(",
        lambda m: f"∃{m.group(1).strip()} (",
        converted,
    )

    converted = converted.replace("<=>", "↔")
    converted = converted.replace("=>", "→")
    converted = converted.replace("~", "¬")
    converted = converted.replace("&", "∧")
    converted = converted.replace("|", "∨")
    return converted


def math_fol_to_tptp(formula: str) -> str:
    """Convert mathematical FOL notation (e.g., ∀x) into TPTP-compatible syntax."""
    converted = formula

    # Support ASCII quantifier words if a model emits them.
    converted = re.sub(r"\bforall\b", "∀", converted, flags=re.IGNORECASE)
    converted = re.sub(r"\bexists\b", "∃", converted, flags=re.IGNORECASE)

    for src, target in _MATH_TO_TPTP_OPS.items():
        converted = converted.replace(src, target)

    return _convert_math_quantifiers(converted)


def _convert_math_quantifiers(text: str) -> str:
    quantifier_map = {"∀": "!", "∃": "?"}
    idx = 0
    out: list[str] = []
    length = len(text)

    while idx < length:
        ch = text[idx]
        if ch not in quantifier_map:
            out.append(ch)
            idx += 1
            continue

        quant_symbol = quantifier_map[ch]
        idx += 1

        while idx < length and text[idx].isspace():
            idx += 1

        vars_start = idx
        while idx < length and (
            text[idx].isalnum() or text[idx] in {"_", ",", " ", "\t"}
        ):
            idx += 1
        raw_vars = text[vars_start:idx].strip()
        var_names = [v.strip() for v in raw_vars.split(",") if v.strip()]

        while idx < length and text[idx].isspace():
            idx += 1
        if idx < length and text[idx] in {":", "."}:
            idx += 1
        while idx < length and text[idx].isspace():
            idx += 1

        body_text: str
        if idx < length and text[idx] == "(":
            end_idx = _find_balanced_right_paren(text, idx)
            if end_idx is None:
                body_text = text[idx + 1 :]
                idx = length
            else:
                body_text = text[idx + 1 : end_idx]
                idx = end_idx + 1
        else:
            body_text = text[idx:]
            idx = length

        converted_body = _convert_math_quantifiers(body_text)
        converted_vars = [_as_tptp_variable(v) for v in var_names]
        converted_body = _replace_bound_variable_occurrences(
            converted_body,
            var_names,
            converted_vars,
        )

        out.append(f"{quant_symbol}[{', '.join(converted_vars)}]: ({converted_body})")

    return "".join(out)


def _find_balanced_right_paren(text: str, left_paren_idx: int) -> int | None:
    depth = 0
    for idx in range(left_paren_idx, len(text)):
        if text[idx] == "(":
            depth += 1
        elif text[idx] == ")":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _as_tptp_variable(name: str) -> str:
    return f"{name[:1].upper()}{name[1:]}"


def _replace_bound_variable_occurrences(
    body: str,
    source_vars: list[str],
    target_vars: list[str],
) -> str:
    updated = body
    for source, target in zip(source_vars, target_vars):
        if not source:
            continue
        updated = re.sub(rf"\b{re.escape(source)}\b", target, updated)
    return updated
