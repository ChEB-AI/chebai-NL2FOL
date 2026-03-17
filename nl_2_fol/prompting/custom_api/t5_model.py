"""
Model Hugging Face URL: https://huggingface.co/fvossel/t5-3b-nl-to-fol

Inference need to be run on a GPU-enabled machine.
You can use the below command to access one such machine on the cluster:

srun --partition=gpu --constraint="A100|H100.80gb" --ntasks=1 --cpus-per-task=8 --threads-per-core=1 --mem=64G --time=12:00:00 --gres=gpu:1 --pty bash
"""

from typing import Any, ClassVar, cast

import torch
from langchain_core.language_models import LLM
from pydantic import PrivateAttr
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
