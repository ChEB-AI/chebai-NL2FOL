import json

import pytest
from langchain_core.prompts import ChatPromptTemplate

pytest.importorskip("torch")
pytest.importorskip("transformers")

from nl_2_fol.prompting.custom_api.t5_model import T5_3B_NL2FOL
from nl_2_fol.prompting.prompt_models import CHEBIFOLOutput


def test_t5_parse_structured_response_accepts_fenced_json():
    raw_response = """```json
    {
      "intermediate_output": {
        "relevant_definition": "Def",
        "superclass": "Cls",
        "explanation": "Why"
      },
      "FOL_formula": "animal(x)"
    }
    ```"""

    parsed = T5_3B_NL2FOL._parse_structured_response(raw_response, CHEBIFOLOutput)

    assert isinstance(parsed, CHEBIFOLOutput)
    assert parsed.intermediate_output.superclass == "Cls"
    assert parsed.FOL_formula == "animal(x)"


def test_t5_with_structured_output_formats_and_parses(monkeypatch):
    llm = T5_3B_NL2FOL.model_construct()
    captured_prompt = {}

    def fake_invoke(self, prompt_text, **kwargs):
        captured_prompt["value"] = prompt_text
        return json.dumps(
            {
                "intermediate_output": {
                    "relevant_definition": "All dogs are animals.",
                    "superclass": "animal",
                    "explanation": "Dogs are a subset of animals.",
                },
                "FOL_formula": "forall x (dog(x) -> animal(x))",
            }
        )

    monkeypatch.setattr(T5_3B_NL2FOL, "invoke", fake_invoke)

    runnable = llm.with_structured_output(CHEBIFOLOutput)
    prompt_value = ChatPromptTemplate.from_messages(
        [("system", "Translate to FOL."), ("human", "{input}")]
    ).invoke({"input": "All dogs are animals."})

    result = runnable.invoke(prompt_value)

    assert isinstance(result, CHEBIFOLOutput)
    assert result.intermediate_output.relevant_definition == "All dogs are animals."
    assert result.FOL_formula == "forall x (dog(x) -> animal(x))"
    assert "Return only a valid JSON object" in captured_prompt["value"]
    assert "All dogs are animals." in captured_prompt["value"]
