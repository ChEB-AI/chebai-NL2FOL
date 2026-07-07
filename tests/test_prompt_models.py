import pytest
from pydantic import ValidationError

from nl_2_fol.prompting.prompt_models import CHEBIFOLOutput


def test_chebi_fol_output_accepts_intermediate_output_dict_string():
    output = CHEBIFOLOutput.model_validate(
        {
            "intermediate_output": '{"relevant_definition": "Def", "superclasses": "Cls", "explanation": "Why"}',
            "FOL_formula": "testPredicate(x)",
        }
    )

    assert output.intermediate_output.relevant_definition == "Def"
    assert output.intermediate_output.superclasses == "Cls"
    assert output.intermediate_output.explanation == "Why"


def test_chebi_fol_output_accepts_intermediate_output_python_dict_string():
    output = CHEBIFOLOutput.model_validate(
        {
            "intermediate_output": "{'relevant_definition': 'Def', 'superclasses': 'Cls', 'explanation': 'Why'}",
            "FOL_formula": "testPredicate(x)",
        }
    )

    assert output.intermediate_output.relevant_definition == "Def"


def test_chebi_fol_output_normalizes_fol_formula_list():
    output = CHEBIFOLOutput.model_validate(
        {
            "intermediate_output": {
                "relevant_definition": "Def",
                "superclasses": "Cls",
                "explanation": "Why",
            },
            "FOL_formula": ["a(x)", "", "& b(x)"],
        }
    )

    assert output.FOL_formula == "a(x) & b(x)"


def test_chebi_fol_output_rejects_invalid_intermediate_output_string():
    with pytest.raises(ValidationError):
        CHEBIFOLOutput.model_validate(
            {
                "intermediate_output": "not a dict",
                "FOL_formula": "testPredicate(x)",
            }
        )
