"""
Code written by https://github.com/fvossel.

@misc{vossel2025advancingnaturallanguageformalization,
    title={Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs},
    author={Felix Vossel and Till Mossakowski and Bj"orn Gehrke},
    year={2025},
    eprint={2509.22338},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2509.22338},
}
"""

from lark import Lark, UnexpectedCharacters


class NamingError(UnexpectedCharacters):
    """Enhanced UnexpectedCharacters with better error message"""

    # Mapping for better token names
    TOKEN_NAMES = {
        "__ANON_1": "junctor",
        "__ANON_2": "junctor",
        "__ANON_3": "junctor",
        "__ANON_4": "junctor",
        "__ANON_5": "junctor",
        "LPAR": "opening parenthesis",
        "RPAR": "closing parenthesis",
        "COMMA": "comma",
        "FORALL": "universal quantifier",
        "EXISTS": "existential quantifier",
        "PREDICATE": "predicate",
        "NAME": "name/constant",
        "VARIABLE": "variable",
        "__ANON_0": "junctor",
    }

    def __init__(
        self, parser: Lark, original_exception: UnexpectedCharacters, formula: str
    ):
        pos = original_exception.pos_in_stream
        tokens = list(parser.lex(formula[:pos]))
        last_token = tokens[-1]

        # Better token name for display
        token_display = self.TOKEN_NAMES.get(last_token.type, last_token.type)

        # Get terminal definition from grammar
        pattern = None
        for term in parser.terminals:
            if term.name == last_token.type:
                pattern = (
                    term.pattern.value
                    if hasattr(term.pattern, "value")
                    else str(term.pattern)
                )
                break

        # Format allowed tokens WITH their patterns
        allowed = original_exception.allowed or set()
        allowed_with_patterns = []
        for token_name in allowed:
            token_display_name = self.TOKEN_NAMES.get(token_name, token_name)

            # Get pattern for allowed token
            token_pattern = None
            for term in parser.terminals:
                if term.name == token_name:
                    token_pattern = (
                        term.pattern.value
                        if hasattr(term.pattern, "value")
                        else str(term.pattern)
                    )
                    break

            if token_pattern:
                allowed_with_patterns.append(f"{token_display_name} ({token_pattern})")
            else:
                allowed_with_patterns.append(token_display_name)

        allowed_str = (
            ", ".join(sorted(allowed_with_patterns))
            if allowed_with_patterns
            else "valid token"
        )

        # Special handling for structural tokens (parentheses, commas, operators)
        if last_token.type in ["LPAR", "RPAR", "COMMA"] or last_token.type.startswith(
            "__ANON"
        ):
            message = f"SYNTAX_ERROR: Unexpected character '{original_exception.char}' at position {original_exception.column} after {token_display} '{last_token.value}'. Expected: {allowed_str}"

            # Add hint for mixing operators after closing parenthesis
            if last_token.type == "RPAR" and original_exception.char in ["∧", "∨", "⊕"]:
                message += ". Hint: Cannot mix AND (∧), OR (∨), and XOR (⊕) operators without parentheses"
        else:
            # Naming problem (PREDICATE, NAME, VARIABLE)
            message = f"SYNTAX_ERROR: Invalid {token_display} '{last_token.value}' - unexpected character '{original_exception.char}' at position {original_exception.column}"
            if pattern:
                message += f". Expected pattern: {pattern}"

        # Copy all attributes from original exception
        self.__dict__.update(original_exception.__dict__)
        self.args = (message,)

    def __str__(self):
        return self.args[0]
