"""
Model Hugging Face URL: https://huggingface.co/fvossel/t5-3b-nl-to-fol

Inference need to be run on a GPU-enabled machine.
You can use the below command to access one such machine on the cluster:

srun --partition=gpu --constraint="A100|H100.80gb" --ntasks=1 --cpus-per-task=8 --threads-per-core=1 --mem=64G --time=12:00:00 --gres=gpu:1 --pty bash
"""

import json
import re
from typing import Any, ClassVar, cast

import torch
from langchain_core.language_models import LLM
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, PrivateAttr
from transformers.models.t5 import T5ForConditionalGeneration, T5Tokenizer


class T5_3B_NL2FOL(LLM):
    HF_MODEL_URL: ClassVar[str] = "fvossel/t5-3b-nl-to-fol"

    # Declare private attributes
    _device: torch.device = PrivateAttr()
    _tokenizer: T5Tokenizer = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = T5Tokenizer.from_pretrained(self.HF_MODEL_URL)
        model = T5ForConditionalGeneration.from_pretrained(
            self.HF_MODEL_URL,
            torch_dtype=torch.float16,
        )
        self._model = cast(Any, model).to(self._device)
        self._model = torch.compile(self._model)
        self._model.eval()

    @property
    def _llm_type(self) -> str:
        return "t5-3b-nl-to-fol"

    def with_structured_output(self, schema: dict | type, **kwargs: Any) -> Runnable:
        include_raw = kwargs.pop("include_raw", False)
        if kwargs:
            unsupported_args = ", ".join(sorted(kwargs))
            raise ValueError(
                "Unsupported structured output options for T5_3B_NL2FOL: "
                f"{unsupported_args}"
            )

        def _invoke_structured(input_value: Any) -> dict | BaseModel:
            prompt = self._stringify_input(input_value)
            prompt_with_schema = self._append_schema_instructions(prompt, schema)
            raw_output = self.invoke(prompt_with_schema)
            parsed_output = self._parse_structured_response(raw_output, schema)

            if include_raw:
                return {
                    "raw": raw_output,
                    "parsed": parsed_output,
                    "parsing_error": None,
                }
            return parsed_output

        return RunnableLambda(_invoke_structured)

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        **kwargs,
    ) -> str:
        output = self._infer_model(prompt)
        return self._tokenizer.decode(output[0], skip_special_tokens=True)

    def _batch_call(self, prompts: list[str]) -> list[str]:
        outputs = self._infer_model(prompts)
        return [self._tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    @staticmethod
    def _stringify_input(input_value: Any) -> str:
        if isinstance(input_value, str):
            return input_value
        if isinstance(input_value, PromptValue):
            return input_value.to_string()
        if isinstance(input_value, BaseMessage):
            return get_buffer_string([input_value])
        if isinstance(input_value, list) and all(
            isinstance(message, BaseMessage) for message in input_value
        ):
            return get_buffer_string(input_value)
        return str(input_value)

    @staticmethod
    def _append_schema_instructions(prompt: str, schema: dict | type) -> str:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_payload = schema.model_json_schema()
        elif isinstance(schema, dict):
            schema_payload = schema
        else:
            raise TypeError(
                "schema must be a Pydantic model class or a JSON schema dictionary"
            )

        return (
            f"{prompt}\n\n"
            "Return only a valid JSON object that matches this schema. "
            "Do not include markdown fences or any explanatory text.\n"
            f"{json.dumps(schema_payload, indent=2, ensure_ascii=False)}"
        )

    @classmethod
    def _parse_structured_response(
        cls, response_text: str, schema: dict | type
    ) -> dict | BaseModel:
        payload = cls._extract_json_payload(response_text)

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(payload)
        if isinstance(schema, dict):
            if not isinstance(payload, dict):
                raise TypeError("Structured response must decode to a JSON object")
            return payload
        raise TypeError(
            "schema must be a Pydantic model class or a JSON schema dictionary"
        )

    @staticmethod
    def _extract_json_payload(response_text: str) -> Any:
        stripped = response_text.strip()
        candidates: list[str] = [stripped]

        fenced_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
        if fenced_match:
            candidates.append(fenced_match.group(1).strip())

        object_match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if object_match:
            candidates.append(object_match.group(0).strip())

        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Model did not return valid JSON: {response_text}")

    @torch.inference_mode()
    def _infer_model(self, input: list[str] | str) -> torch.Tensor:
        inputs = self._tokenizer(
            input,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        input_ids = cast(torch.Tensor, inputs["input_ids"])

        outputs = self._model.generate(
            input_ids,
            max_length=256,
            min_length=1,
            num_beams=5,
            length_penalty=2.0,
            early_stopping=True,
        )
        return outputs.sequences if hasattr(outputs, "sequences") else outputs


if __name__ == "__main__":
    llm = T5_3B_NL2FOL()
    # Example NL input
    nl_input = "All dogs are animals."
    # Preprocess prompt
    input_text = (
        "translate English natural language statements into first-order logic (FOL): "
        + nl_input
    )

    result = llm.invoke(input_text)
    print(result)
