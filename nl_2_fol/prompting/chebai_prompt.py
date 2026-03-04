import json

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import Runnable

from nl_2_fol.inference import custom_exceptions as ce
from nl_2_fol.prompting.llm_inference import API_PLATFORM, get_llm_for_inference
from nl_2_fol.prompting.prompt_models import (
    CHEBIFOLOutput,
    OutOfBoxPredicateDefinitions,
)
from nl_2_fol.prompting.utils.read_configs import json_to_pyObj, load_yaml_sys_prompt


class ChebiPrompt:
    def __init__(
        self,
        platform: API_PLATFORM,
        model_name: str,
        system_prompt_fp: str,
        few_shot_prompt_fp: str,
        err_failure_prompt_fp: str,
        undef_failure_prompt_fp: str,
    ):
        self.platform: API_PLATFORM = platform
        self.model_name: str = model_name
        self.system_prompt_fp: str = system_prompt_fp
        self.few_shot_prompt_fp: str = few_shot_prompt_fp
        self.err_failure_prompt_fp: str = err_failure_prompt_fp
        self.undef_failure_prompt_fp: str = undef_failure_prompt_fp
        # To keep track of predicates generated across iterations, for prompting
        self.generated_predicates_names: set[str] = set()
        self._memory_store = {}

        self._llm = get_llm_for_inference(self.platform, self.model_name)

        self._conversation_chain = self._get_conversation_chain()
        self._undef_failure_chain = self._get_undef_failure_chain()

    # -------- Conversation Chain Construction --------------------- ##
    def _get_conversation_chain(self) -> Runnable:
        prompt = self._get_prompt_template()
        structured_llm = self._llm.with_structured_output(CHEBIFOLOutput)
        return prompt | structured_llm

    def _get_undef_failure_chain(self) -> Runnable:
        """Create a chain specifically for handling undefined predicates with shared memory."""
        prompt = self._get_prompt_template()
        structured_llm = self._llm.with_structured_output(OutOfBoxPredicateDefinitions)
        return prompt | structured_llm

    @ce.stop_program_upon_failure
    def get_session_history(self, session_id: str):
        """Shared session history for all chains to maintain conversation context."""
        if session_id not in self._memory_store:
            self._memory_store[session_id] = InMemoryChatMessageHistory()
        return self._memory_store[session_id]

    @ce.stop_program_upon_failure
    def delete_session_history(self, session_id: str):
        """Utility method to clear conversation history for a session."""
        if session_id in self._memory_store:
            del self._memory_store[session_id]

    def _get_prompt_template(self) -> ChatPromptTemplate:
        system_prompt = self._get_system_prompt()
        few_shot_prompt = self._get_few_shot_prompt()

        return ChatPromptTemplate.from_messages(
            [
                # System and few shots promots are static, they dont change or
                # are overwrittenm. They serve as context and behaviour guidance
                system_prompt,
                few_shot_prompt,
                # Memory inserted here and is dynamic, stores user previous conversations
                # Changes over time, can be summarized or truncated to fit in context window if needed
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

    # -------- System Prompt and Predicates List Management --------- ##
    def _get_system_prompt(self) -> SystemMessage:
        system_prompt_text = load_yaml_sys_prompt(self.system_prompt_fp)
        predicates_list_text = self._get_predicates_section()
        full_system_prompt_text = system_prompt_text + predicates_list_text
        return SystemMessage(content=full_system_prompt_text)

    def _get_predicates_section(self) -> str:
        """Generate the predicates list section dynamically."""
        # Only add predicates section if list is not empty
        # TODO: Check if needed, as list of predicates can be long and may overwhelm the prompt.
        # Maybe only add if there are less than N predicates?
        # Advanced models (Claude Sonnet 4.5/4.6, Claude Opus 4) support a 1,000,000-token
        # context window, other models handles 200,000 and 1,000,000 tokens
        if len(self.generated_predicates_names) > 0:
            return (
                "\nAlso, here is the list of predicates along with their arguments that "
                "were already defined in previous iterations for other CHEBI classes.\n"
                "If any predicate has no arguments, then just the predicate name is shown "
                "without parentheses. You can reuse these predicates if they are "
                "applicable to the current class definition.\n"
                f"Predicate List: {', '.join(sorted(self.generated_predicates_names))}"
            )
        return ""

    ## ------- Few-Shot Prompt Construction ------------------------- ##
    def _get_few_shot_prompt(self) -> FewShotChatMessagePromptTemplate:
        raw_examples = json_to_pyObj(self.few_shot_prompt_fp)

        def _normalize_fol_formula(ai_payload: dict) -> dict:
            fol_formula = ai_payload.get("FOL_formula")
            if isinstance(fol_formula, list):
                ai_payload = dict(ai_payload)
                ai_payload["FOL_formula"] = " ".join(
                    part.strip() for part in fol_formula if part.strip()
                )
            return ai_payload

        processed_examples = [
            {
                "human": ex["human"],
                "ai": json.dumps(
                    _normalize_fol_formula(ex["ai"]),
                    indent=4,
                    ensure_ascii=False,
                ),
            }
            for ex in raw_examples
        ]

        # Template variables must match keys in processed_examples
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{human}"),
                ("ai", "{ai}"),
            ]
        )

        return FewShotChatMessagePromptTemplate(
            examples=processed_examples,
            example_prompt=example_prompt,
        )

    ## ------- LLM Invocation for first call ------------------------ ##
    @ce.stop_program_upon_failure
    def invoke_llm_first_call(
        self, *, input_text: str, session_id: str
    ) -> CHEBIFOLOutput:
        try:
            input_text = self._normalize_input_text(input_text)

            # Get session history
            history = self.get_session_history(session_id)

            # Invoke chain with current history
            output = self._conversation_chain.invoke(
                {
                    "input": input_text,
                    "history": history.messages,
                }
            )

            # Manually add messages to history
            history.add_message(HumanMessage(content=input_text))
            history.add_message(AIMessage(content=output.model_dump_json(indent=2)))

            return output
        except Exception as e:
            print(f"Error during inference: {e}")
            raise e

    ## ------- LLM Invocation for error failure --------------------- ##
    @ce.stop_program_upon_failure
    def invoke_llm_with_error_failure_prompt(
        self, *, error_message: str, session_id: str
    ) -> CHEBIFOLOutput:
        try:
            error_text = self._normalize_input_text(error_message)
            error_prompt = self._get_err_failure_prompt(error_text)

            # Get session history
            history = self.get_session_history(session_id)

            # Invoke chain with current history
            output = self._conversation_chain.invoke(
                {
                    "input": error_prompt,
                    "history": history.messages,
                }
            )

            # Manually add messages to history
            history.add_message(HumanMessage(content=error_prompt))
            history.add_message(AIMessage(content=output.model_dump_json(indent=2)))

            return output
        except Exception as e:
            print(f"Error during inference: {e}")
            raise e

    def _get_err_failure_prompt(self, error_message: str) -> str:
        prompt_text = load_yaml_sys_prompt(
            self.err_failure_prompt_fp, key="failure_prompt"
        )
        prompt_text = prompt_text.format(error_message=error_message)
        return prompt_text

    ## ------- LLM Invocation for undefined additional predicates --- ##
    @ce.stop_program_upon_failure
    def invoke_llm_with_undef_failure_prompt(
        self,
        undefined_predicates: dict[str, str | None],
        session_id: str,
    ) -> OutOfBoxPredicateDefinitions:
        try:
            undefined_predicates_text = self._get_undef_failure_prompt(
                undefined_predicates
            )

            # Get session history
            history = self.get_session_history(session_id)

            # Invoke chain with current history
            output = self._undef_failure_chain.invoke(
                {
                    "input": undefined_predicates_text,
                    "history": history.messages,
                }
            )

            # Manually add messages to history
            history.add_message(HumanMessage(content=undefined_predicates_text))
            history.add_message(AIMessage(content=output.model_dump_json(indent=2)))

            return output
        except Exception as e:
            print(f"Error during failure prompt inference: {e}")
            raise e

    def _get_undef_failure_prompt(
        self,
        undefined_predicates_details: dict[str, str | None],
    ) -> str:
        prompt_text = load_yaml_sys_prompt(
            self.undef_failure_prompt_fp, key="failure_prompt"
        )
        undefined_predicates_txt = "\n".join(
            f"  - Predicate: {name}"
            + (f"\n    Chemical Definition: {definition}" if definition else "")
            for name, definition in undefined_predicates_details.items()
        )
        prompt_text = prompt_text.format(
            undefined_predicates_details=undefined_predicates_txt
        )
        return prompt_text

    def __repr__(self) -> str:
        return f"""
        ChebiPrompt(platform={self.platform},
        model_name={self.model_name},
        system_prompt_fp={self.system_prompt_fp},
        few_shot_prompt_fp={self.few_shot_prompt_fp},
        err_failure_prompt_fp={self.err_failure_prompt_fp},
        undef_failure_prompt_fp={self.undef_failure_prompt_fp})
        """

    @ce.stop_program_upon_failure
    def get_full_conversation_context(self, session_id: str = "default") -> dict:
        """
        Get the complete conversation context including system prompt, few-shots, and history.

        Returns:
            dict with keys:
                - 'system_prompt': str - The system prompt content
                - 'few_shot_examples': list[dict] - List of few-shot examples
                - 'conversation_history': list[dict] - List of conversation messages
        """
        # System Prompt
        system_prompt = self._get_system_prompt()

        # Few-Shot Examples
        few_shot_template = self._get_few_shot_prompt()
        few_shot_examples = []
        if few_shot_template.examples:
            few_shot_examples = [
                {"human": example["human"], "ai": example["ai"]}
                for example in few_shot_template.examples
            ]

        # Conversation History
        history = self.get_session_history(session_id)
        conversation_history = []
        if history.messages:
            conversation_history = [
                {"type": msg.__class__.__name__, "content": msg.content}
                for msg in history.messages
            ]

        return {
            "system_prompt": system_prompt.content,
            "few_shot_examples": few_shot_examples,
            "conversation_history": conversation_history,
        }

    @ce.stop_program_upon_failure
    def print_full_conversation_context(self, session_id: str = "default") -> None:
        """Print the complete conversation context including system prompt, few-shots, and history."""
        context = self.get_full_conversation_context(session_id)

        print("\n" + "=" * 80)
        print("COMPLETE CONVERSATION CONTEXT")
        print("=" * 80)

        # System Prompt
        print("\n[SYSTEM PROMPT]")
        print("-" * 80)
        print(context["system_prompt"])

        # Few-Shot Examples
        print("\n[FEW-SHOT EXAMPLES]")
        print("-" * 80)
        if context["few_shot_examples"]:
            for i, example in enumerate(context["few_shot_examples"], 1):
                print(f"\nExample {i}:")
                print(
                    f"  Human: {example['human'][:200]}..."
                    if len(example["human"]) > 200
                    else f"  Human: {example['human']}"
                )
                print(
                    f"  AI: {example['ai'][:200]}..."
                    if len(example["ai"]) > 200
                    else f"  AI: {example['ai']}"
                )
        else:
            print("(No few-shot examples)")

        # Conversation History
        print("\n[CONVERSATION HISTORY]")
        print("-" * 80)
        if context["conversation_history"]:
            for i, msg in enumerate(context["conversation_history"], 1):
                content_preview = (
                    msg["content"][:200] + "..."
                    if len(msg["content"]) > 200
                    else msg["content"]
                )
                print(f"\n{i}. {msg['type']}:")
                print(f"   {content_preview}")
        else:
            print("(No conversation history yet)")

        print("\n" + "=" * 80)

    @staticmethod
    def _normalize_input_text(input_text: str) -> str:
        """
        Normalize the input text by stripping leading/trailing whitespace
        and collapsing multiple spaces into a single space.
        """
        return " ".join(str(input_text).split())


if __name__ == "__main__":
    # ------------------- TESTING THE CLASS ------------------#
    # Example usage
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = os.path.join(base_dir, "prompt_templates")
    chebai_prompt = ChebiPrompt(
        platform="anthropic",
        model_name="claude-opus-4-6",
        system_prompt_fp=os.path.join(
            prompt_dir, "system_prompts", "with_predicates_list.yaml"
        ),
        few_shot_prompt_fp=os.path.join(prompt_dir, "few_shots", "with_DL_style.json"),
        err_failure_prompt_fp=os.path.join(prompt_dir, "failure", "error_prompt.yaml"),
        undef_failure_prompt_fp=os.path.join(
            prompt_dir, "failure", "predicates_undef_with_eg.yaml"
        ),
    )

    chebi_def = """CHEBI:16236 - ethanol: A primary alcohol that
        is ethane in which one of the hydrogens is substituted
        by a hydroxy group."""
    chebai_prompt.generated_predicates_names.add("primary_alcohol")
    chebai_prompt.generated_predicates_names.add("hydroxy_group")

    # Use the same session_id across all invocations to maintain conversation history
    test_session_id = "test_session"

    print("---" * 10, "FEW-SHOT PROMPT TEST", "---" * 10)
    # Test the few-shot prompt
    result = chebai_prompt.invoke_llm_first_call(chebi_def, session_id=test_session_id)
    print(f"Few-shot result:\n {result}")
    print("\n\n\n")

    print("---" * 10, "FOL DEFINITION ERROR FAILURE PROMPT TEST", "---" * 10)
    error_message = """ Unknow predicate 'has_part' used in the FOL formula,
    which is not defined in the system prompt.",
    """
    # Test the failure prompt - uses same session to maintain context
    failure_result = chebai_prompt.invoke_llm_with_error_failure_prompt(
        error_message=error_message, session_id=test_session_id
    )
    print(f"Failure result:\n {failure_result}")

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
    print(f"Undefined predicates failure prompt text: \n {failure_prompt_text} \n\n\n")
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
