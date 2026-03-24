"""
Model Hugging Face URL: https://huggingface.co/fvossel/Mistral-Small-24B-Instruct-2501-nl-to-fol
See env requirements here: https://github.com/fvossel/NL2FOL/blob/main/requirements.txt

Inference needs to be run on a GPU-enabled machine.
You can use the below command to access one such machine on the cluster:

srun --partition=gpu --constraint="A100|H100.80gb" --ntasks=1 --cpus-per-task=8 --threads-per-core=1 --mem=80G --time=12:00:00 --gres=gpu:1 --pty bash
"""

import time
from typing import Any, ClassVar

import torch
from langchain_core.language_models import LLM
from peft import PeftModel
from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer


class Mistral_24B(LLM):
    HF_BASE_MODEL: ClassVar[str] = "mistralai/Mistral-Small-24B-Instruct-2501"
    HF_LORA_WEIGHTS: ClassVar[str] = "fvossel/Mistral-Small-24B-Instruct-2501-nl-to-fol"

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not torch.cuda.is_available():
            raise EnvironmentError(
                "CUDA-enabled GPU is required for inference with Mistral-24B. "
                "Please run this on a machine with a compatible GPU."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.HF_BASE_MODEL, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            self.HF_BASE_MODEL,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        self._model = PeftModel.from_pretrained(
            model,
            self.HF_LORA_WEIGHTS,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        # self._model = torch.compile(model)
        self._model.eval()

    @property
    def _llm_type(self) -> str:
        return "Mistral-Small-24B-Instruct-nl-to-fol"

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
    def _infer_model(self, input: list[str] | str):
        device = next(iter(self._model.parameters())).device
        inputs = self._tokenizer(
            input,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # print("Context length: ", self._model.config.max_position_embeddings)

        # Generate sequences that include both the input prompt and new tokens
        start_time = time.perf_counter()
        outputs = self._model.generate(**inputs, max_new_tokens=5000)
        elapsed_seconds = time.perf_counter() - start_time
        prompt_count = len(input) if isinstance(input, list) else 1
        print(
            f"[Mistral_24B] Inference took {elapsed_seconds:.2f}s for {prompt_count} prompt(s)."
        )

        # Get the length of the input sequence
        input_length = inputs["input_ids"].shape[1]

        # Slice the output tensor to keep ONLY the newly generated tokens
        generated_tokens = outputs[:, input_length:]

        return generated_tokens


if __name__ == "__main__":
    llm = Mistral_24B()
    # Example NL input
    nl_input = "All dogs are animals."
    # Preprocess prompt
    input_text = (
        "translate English natural language statements into first-order logic (FOL): "
        + nl_input
    )
    result = llm.invoke(input_text)
    print(result)
