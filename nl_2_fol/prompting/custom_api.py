"""
Inference need to be run on a GPU-enabled machine.
You can use the below command to access one such machine on the cluster:

    srun --partition=gpu --constraint="A100|H100.80gb" --ntasks=1
    --cpus-per-task=8 --threads-per-core=1 --mem=64G
    --time=02:00:00 --gres=gpu:1 --pty bash
"""

from typing import ClassVar

import torch
from langchain_core.language_models import LLM
from pydantic import PrivateAttr
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5_3B_NL2FOL(LLM):
    HF_MODEL_URL: ClassVar[str] = "fvossel/t5-3b-nl-to-fol"
    PROMPT: ClassVar[str] = (
        "translate English natural language statements into first-order logic (FOL): "
    )

    # Declare private attributes
    _device: str = PrivateAttr()
    _tokenizer: T5Tokenizer = PrivateAttr()
    _model: T5ForConditionalGeneration = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = T5Tokenizer.from_pretrained(self.HF_MODEL_URL)
        self._model = T5ForConditionalGeneration.from_pretrained(
            self.HF_MODEL_URL,
            torch_dtype=torch.float16,
        ).to(self._device)
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
        output = self._infer_model(self.PROMPT + prompt)
        return self._tokenizer.decode(output[0], skip_special_tokens=True)

    def _batch_call(self, prompts: list[str]) -> list[str]:
        prompts = [self.PROMPT + p for p in prompts]
        outputs = self._infer_model(prompts)
        return [self._tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    @torch.inference_mode()
    def _infer_model(self, input: list[str] | str) -> torch.LongTensor:
        inputs = self._tokenizer(
            input,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        outputs = self._model.generate(
            inputs["input_ids"],
            max_length=256,
            min_length=1,
            num_beams=5,
            length_penalty=2.0,
            early_stopping=True,
        )
        return outputs


if __name__ == "__main__":
    llm = T5_3B_NL2FOL()
    result = llm.invoke("All dogs are animals.")
    print(result)
