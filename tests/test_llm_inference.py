import pytest

from nl_2_fol.prompting.llm_inference import _anthropic_model_supports_temperature
from nl_2_fol.prompting.prompt_models import OutOfBoxPredicateDefinitions


@pytest.mark.parametrize(
    "model_name, expected",
    [
        ("claude-opus-4-6", True),
        ("claude-sonnet-4-5-20250929", True),
        ("claude-haiku-4-5", True),
        ("claude-opus-4-20250514", True),
        ("claude-3-5-sonnet-20241022", True),
        ("claude-opus-4-7", False),
        ("claude-opus-4-8", False),
        ("claude-opus-5-5", False),
        ("claude-haiku-5-5", False),
        ("claude-sonnet-5", False),
        ("claude-fable-5-2", False),
    ],
)
def test_anthropic_model_supports_temperature(model_name, expected):
    assert _anthropic_model_supports_temperature(model_name) is expected


@pytest.mark.parametrize(
    "few_shot_file, expected",
    [
        ("ASP_adaptive_thinking.json", "ChEBI_FOL_AT"),
        ("ASP_style.json", "CHEBIFOLOutput"),
        ("with_DL_style.json", "CHEBIFOLOutput"),
    ],
)
def test_structured_output_type_follows_few_shots(few_shot_file, expected):
    import os

    from nl_2_fol.prompting.chebai_prompt import ChebiPrompt

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "nl_2_fol",
        "prompting",
        "prompt_templates",
        "few_shots",
        few_shot_file,
    )
    assert ChebiPrompt._infer_structured_output_type(path) == expected


def test_out_of_box_definitions_list_format():
    defs = OutOfBoxPredicateDefinitions.model_validate(
        {
            "definitions": [
                {"predicate_name": "hydroxy", "formula": "hydroxy(O) :- o(O)."},
                {"predicate_name": "alcohol", "formula": "alcohol(M) :- c(M)."},
            ]
        }
    )
    assert defs.predicate_definitions == {
        "hydroxy": "hydroxy(O) :- o(O).",
        "alcohol": "alcohol(M) :- c(M).",
    }


def test_out_of_box_definitions_legacy_dict_format():
    defs = OutOfBoxPredicateDefinitions(
        predicate_definitions={"hydroxy": "hydroxy(O) :- o(O)."}
    )
    assert defs.predicate_definitions == {"hydroxy": "hydroxy(O) :- o(O)."}


def test_out_of_box_definitions_schema_has_no_free_form_objects():
    """Strict structured outputs do not support objects with arbitrary keys."""
    schema = OutOfBoxPredicateDefinitions.model_json_schema()

    def _check(node):
        if isinstance(node, dict):
            assert node.get("additionalProperties") not in ({"type": "string"},)
            for value in node.values():
                _check(value)
        elif isinstance(node, list):
            for value in node:
                _check(value)

    _check(schema)
