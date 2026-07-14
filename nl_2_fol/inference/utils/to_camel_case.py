def to_camel_case(name: str) -> str:
    raw_name = str(name).strip()
    while (
        len(raw_name) >= 2 and raw_name[0] == raw_name[-1] and raw_name[0] in {'"', "'"}
    ):
        raw_name = raw_name[1:-1].strip()

    tokens: list[str] = []
    current_token: list[str] = []
    bracket_depth = 0

    for char in raw_name:
        if char in "[({":
            bracket_depth += 1
            current_token.append(char)
            continue

        if char in ")]}":
            if bracket_depth > 0:
                bracket_depth -= 1
            current_token.append(char)
            continue

        if bracket_depth == 0 and char in {" ", "-"}:
            if current_token:
                tokens.append("".join(current_token))
                current_token = []
            continue

        current_token.append(char)

    if current_token:
        tokens.append("".join(current_token))

    if not tokens:
        return ""

    def normalize_token(token: str, uppercase_first: bool) -> str:
        normalized: list[str] = []
        in_brackets = 0
        first_alpha_done = False

        for char in token:
            if char in "[(":
                in_brackets += 1
                normalized.append(char)
                continue

            if char in ")]":
                if in_brackets > 0:
                    in_brackets -= 1
                normalized.append(char)
                continue

            if char.isalpha() and in_brackets == 0:
                if not first_alpha_done:
                    normalized.append(char.upper() if uppercase_first else char.lower())
                    first_alpha_done = True
                else:
                    normalized.append(char.lower())
                continue

            normalized.append(char)

        return "".join(normalized)

    first_token = tokens[0]
    first = (
        first_token
        if not first_token or not first_token[0].isalpha()
        else normalize_token(first_token, False)
    )
    rest = [
        normalize_token(token, True) if token and token[0].isalpha() else token
        for token in tokens[1:]
    ]
    return first + "".join(rest)
