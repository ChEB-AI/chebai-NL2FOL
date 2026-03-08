import re


def to_camel_case(name: str) -> str:
    raw_name = str(name).strip().strip('"').strip("'")
    tokens = re.findall(r"\d+(?:,\d+)+|[A-Za-z]\d+:\d+|[A-Za-z0-9]+", raw_name)
    if not tokens:
        return ""

    first = tokens[0].lower() if tokens[0][0].isalpha() else tokens[0]
    rest = [token.capitalize() if token[0].isalpha() else token for token in tokens[1:]]
    return first + "".join(rest)
