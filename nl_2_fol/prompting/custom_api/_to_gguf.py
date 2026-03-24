def convert_mistral_to_gguf():
    # See: https://huggingface.co/fvossel/Mistral-Small-24B-Instruct-2501-nl-to-fol
    # See env requirements here: https://github.com/fvossel/NL2FOL/blob/main/requirements.txt
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model_name = "mistralai/Mistral-Small-24B-Instruct-2501"
    lora_weights = "fvossel/Mistral-Small-24B-Instruct-2501-nl-to-fol"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(
        model, lora_weights, device_map="auto", dtype=torch.bfloat16
    )

    model = model.merge_and_unload()  # type: ignore

    model.save_pretrained("mistral-merged")
    tokenizer.save_pretrained("mistral-merged")


if __name__ == "__main__":
    convert_mistral_to_gguf()
