from typing import Any

from pydantic import BaseModel

from nl_2_fol.prompting.custom_api.base_chat_model import (
    _LocalModelPydanticOutputParser,
)


def test_local_model_pydantic_output_parser_parses_json_and_markdown() -> None:
    """Ensure valid JSON and markdown JSON are parsed into the Pydantic model."""
    parser = _LocalModelPydanticOutputParser(pydantic_object=_TestOutputModel)

    json_payload = (
        '{"FOL_formula": "P(x)", '
        '"explanation": "parsed", '
        '"items": ["a", "b"], '
        '"metadata": {"k": "v"}}'
    )

    # Plain JSON
    obj = parser.parse(json_payload)
    assert isinstance(obj, _TestOutputModel)
    assert obj.FOL_formula == "P(x)"
    assert obj.explanation == "parsed"
    assert obj.items == ["a", "b"]
    assert obj.metadata == {"k": "v"}

    # Fenced markdown JSON
    markdown_payload = f"```json\n{json_payload}\n```"
    obj_markdown = parser.parse(markdown_payload)
    assert isinstance(obj_markdown, _TestOutputModel)
    assert obj_markdown.FOL_formula == "P(x)"
    assert obj_markdown.explanation == "parsed"
    assert obj_markdown.items == ["a", "b"]
    assert obj_markdown.metadata == {"k": "v"}


def test_local_model_pydantic_output_parser_fallback_populates_fol_formula() -> None:
    """Ensure fallback from raw text produces a valid object with `FOL_formula` set."""
    parser = _LocalModelPydanticOutputParser(pydantic_object=_TestOutputModel)

    raw_text = "forall x. P(x) -> Q(x)"
    obj = parser.parse(raw_text)

    # Fallback should populate FOL_formula with the raw text.
    assert isinstance(obj, _TestOutputModel)
    assert obj.FOL_formula == raw_text

    # Other fields should follow the documented defaults from `_build_from_raw_text`.
    assert obj.explanation == ""
    assert obj.items == []
    assert obj.metadata == {}


class _TestOutputModel(BaseModel):
    """Internal test model for `_LocalModelPydanticOutputParser`.

    This mirrors the key patterns the parser is expected to handle:
    - a top-level `FOL_formula` field that should receive raw text on fallback
    - a string field
    - a list field
    - a dict field
    """

    FOL_formula: str | None = None
    explanation: str = ""
    items: list[str] = []
    metadata: dict[str, Any] = {}
