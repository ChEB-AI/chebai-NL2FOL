"""
Model Hugging Face URL: https://huggingface.co/fvossel/Mistral-Small-24B-Instruct-2501-nl-to-fol

Inference needs to be run on a GPU-enabled machine.
You can use the below command to access one such machine on the cluster:

srun --partition=gpu --constraint="A100|H100.80gb" --ntasks=1 --cpus-per-task=8 --threads-per-core=1 --mem=64G --time=12:00:00 --gres=gpu:1 --pty bash
"""

from typing import Any, ClassVar

import torch
from langchain_core.language_models import LLM
from peft import PeftModel
from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer

_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that translates Natural Language (NL) text "
    "into First-Order Logic (FOL) using only the given quantors and junctors: "
    "\u2200 (for all), \u2203 (there exists), \u00ac (not), \u2227 (and), \u2228 (or), \u2192 (implies), "
    "\u2194 (if and only if), \u2295 (xor). "
    "Start your answer with '\U0001d719=' followed by the FOL-formula. Do not include any other text."
)


class Mistral_24B(LLM):
    HF_BASE_MODEL: ClassVar[str] = "mistralai/Mistral-Small-24B-Instruct-2501"
    HF_LORA_WEIGHTS: ClassVar[str] = "fvossel/Mistral-Small-24B-Instruct-2501-nl-to-fol"

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.HF_BASE_MODEL, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            self.HF_BASE_MODEL, trust_remote_code=True, device_map="auto"
        )
        model = PeftModel.from_pretrained(
            model, self.HF_LORA_WEIGHTS, device_map="auto"
        )
        self._model = torch.compile(model)
        self._model.eval()

    @property
    def _llm_type(self) -> str:
        return "Mistral-Small-24B-Instruct-nl-to-fol"

    def _format_prompt(self, text: str) -> str:
        return self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        **kwargs,
    ) -> str:
        formatted = self._format_prompt(prompt)
        output = self._infer_model(formatted)
        return self._tokenizer.decode(output[0], skip_special_tokens=True)

    def _batch_call(self, prompts: list[str]) -> list[str]:
        formatted = [self._format_prompt(p) for p in prompts]
        outputs = self._infer_model(formatted)
        return [self._tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    @torch.inference_mode()
    def _infer_model(self, input: list[str] | str) -> torch.Tensor:
        device = next(iter(self._model.parameters())).device
        inputs = self._tokenizer(
            input,
            return_tensors="pt",
            padding=True,
        ).to(device)

        outputs = self._model.generate(**inputs, max_new_tokens=100)
        return outputs


if __name__ == "__main__":
    llm = Mistral_24B()
    result = llm.invoke("All dogs are animals.")
    print(result)
