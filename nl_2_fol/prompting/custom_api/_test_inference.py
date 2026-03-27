import os

from nl_2_fol.prompting.chebai_prompt import ChebiPrompt

WORKING_DIR = os.getcwd()

PROJECT_DIR = os.path.join(WORKING_DIR, "nl_2_fol")
DATA_DIR = os.path.join(WORKING_DIR, "data")
PROMPT_TEMPLATES_DIR = os.path.join(PROJECT_DIR, "prompting", "prompt_templates")
system_prompt_fp: str = os.path.join(
    PROMPT_TEMPLATES_DIR, "system_prompts", "with_predicates_list.yaml"
)
few_shot_prompt_fp: str = os.path.join(
    PROMPT_TEMPLATES_DIR, "few_shots", "with_DL_style.json"
)
err_failure_prompt_fp: str = os.path.join(
    PROMPT_TEMPLATES_DIR, "failure", "error_prompt.yaml"
)
undef_failure_prompt_fp: str = os.path.join(
    PROMPT_TEMPLATES_DIR, "failure", "predicates_undef_with_eg.yaml"
)


def infer_on_custom_model(model_name):
    custom_model = ["t5-3b-nl-to-fol", "Mistral-Small-24B-Instruct-nl-to-fol"]
    if model_name not in custom_model:
        raise ValueError(
            f"Invalid model name '{model_name}'. Valid options are: {custom_model}"
        )
    chebai_prompt = ChebiPrompt(
        platform="custom",
        model_name=model_name,
        system_prompt_fp=system_prompt_fp,
        few_shot_prompt_fp=few_shot_prompt_fp,
        err_failure_prompt_fp=err_failure_prompt_fp,
        undef_failure_prompt_fp=undef_failure_prompt_fp,
    )
    chebi_def = """CHEBI:16236 - ethanol: A primary alcohol that
    is ethane in which one of the hydrogens is substituted
    by a hydroxy group."""
    chebai_prompt.generated_predicates_names.add("primaryAlcohol")
    chebai_prompt.generated_predicates_names.add("hydroxyGroup")

    # Use the same session_id across all invocations to maintain conversation history
    test_session_id = "test_session"

    print("---" * 10, "FEW-SHOT PROMPT TEST", "---" * 10)
    # Test the few-shot prompt
    result = chebai_prompt.invoke_llm_first_call(
        input_text=chebi_def, session_id=test_session_id
    )
    print(f"Few-shot result:\n{result}\n\n")

    print(
        "---" * 10,
        "FOL DEFINITION UNDEFINED PREDICATES FAILURE PROMPT TEST",
        "---" * 10,
    )

    undefined_predicates = {
        "has_part": "A predicate indicating that a chemical entity has a certain part or component.",
        "SOME": None,
    }
    # This invocation now shares the same conversation history as the previous calls
    failure_prompt_text = chebai_prompt._get_undef_failure_prompt(undefined_predicates)
    failure_result = chebai_prompt.invoke_llm_with_undef_failure_prompt(
        undefined_predicates=undefined_predicates,
        session_id=test_session_id,  # Same session = shared memory
    )
    print(f"Undefined predicates failure prompt text: \n {failure_prompt_text} \n\n")
    print(f"Undefined predicates failure result:\n {failure_result}")

    print("\n" + "=" * 80)
    print("GET COMPLETE MEMORY (Returns structured data)")
    print("=" * 80)
    # Get the conversation context as a dictionary
    full_context = chebai_prompt.get_full_conversation_context("test_session")
    print(f"System prompt length: {len(full_context['system_prompt'])} characters")
    print(f"Number of few-shot examples: {len(full_context['few_shot_examples'])}")
    print(
        f"Number of conversation messages: {len(full_context['conversation_history'])}"
    )

    # You can also access individual parts:
    # print("\nFull system prompt:", full_context['system_prompt'])
    # print("\nFew-shot examples:", full_context['few_shot_examples'])
    # print("\nConversation history:", full_context['conversation_history'])

    print("\n" + "=" * 80)
    print("PRINT COMPLETE MEMORY (Pretty-printed view)")
    print("=" * 80)
    chebai_prompt.print_full_conversation_context("test_session")


if __name__ == "__main__":
    infer_on_custom_model("t5-3b-nl-to-fol")
