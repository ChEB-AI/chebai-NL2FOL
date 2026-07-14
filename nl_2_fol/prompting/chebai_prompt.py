import json
import os
from typing import Callable, Literal, cast

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import Runnable

from nl_2_fol.inference.learner import custom_exceptions as ce
from nl_2_fol.prompting.llm_inference import API_PLATFORM, get_llm_for_inference
from nl_2_fol.prompting.prompt_models import (
    ChEBI_FOL_AT,
    CHEBIFOLOutput,
    OutOfBoxPredicateDefinitions,
)
from nl_2_fol.prompting.retrieve_relevant_predicates import (
    SemanticPredicateRetriever,
)
from nl_2_fol.prompting.utils.read_configs import json_to_pyObj, load_yaml_sys_prompt


class ChebiPrompt:
    MAX_INPUT_TOKENS = 15000

    def __init__(
        self,
        platform: API_PLATFORM,
        model_name: str,
        system_prompt_fp: str,
        few_shot_prompt_fp: str,
        err_failure_prompt_fp: str,
        undef_failure_prompt_fp: str,
        predicate_prompt_mode: Literal["relevant", "all"] = "relevant",
        structured_output_type: str = "ChEBI_FOL_AT",
    ):
        if predicate_prompt_mode not in {"relevant", "all"}:
            raise ValueError(
                "predicate_prompt_mode must be either 'relevant' or 'all'."
            )
        self.platform: API_PLATFORM = platform
        self.model_name: str = model_name
        self.system_prompt_fp: str = system_prompt_fp
        self.few_shot_prompt_fp: str = few_shot_prompt_fp
        self.err_failure_prompt_fp: str = err_failure_prompt_fp
        self.undef_failure_prompt_fp: str = undef_failure_prompt_fp
        self.predicate_prompt_mode = predicate_prompt_mode
        self.structured_output_type = structured_output_type
        # To keep track of predicates generated across iterations, for prompting
        self._memory_store = {}
        self.generated_predicates_names: set[str] = set()
        self._relevant_predicates: list[str] = []
        self._current_session_id: str | None = None  # Track current session

        self._llm = get_llm_for_inference(self.platform, self.model_name)

        self._conversation_chain = self._get_conversation_chain()
        self._undef_failure_chain = self._get_undef_failure_chain()

        if self.platform == "ollama" or self.platform == "custom":
            # For self hosted models, set no token limit as we don't incur a direct cost
            self.MAX_INPUT_TOKENS = None

        self._predicate_retriever = (
            SemanticPredicateRetriever()
            if self.predicate_prompt_mode == "relevant"
            else None
        )

    # -------- Conversation Chain Construction --------------------- ##
    def _get_conversation_chain(self) -> Runnable:
        prompt = self._get_prompt_template()
        if self.structured_output_type == "ChEBI_FOL_AT":
            structured_llm = self._llm.with_structured_output(ChEBI_FOL_AT)
        elif self.structured_output_type == "CHEBIFOLOutput":
            structured_llm = self._llm.with_structured_output(CHEBIFOLOutput)
        else:
            raise ValueError("Invalid structured_output_type")
        return prompt | structured_llm

    def _get_undef_failure_chain(self) -> Runnable:
        """Create a chain specifically for handling undefined predicates with shared memory."""
        prompt = self._get_prompt_template()
        structured_llm = self._llm.with_structured_output(OutOfBoxPredicateDefinitions)
        return prompt | structured_llm

    def _rebuild_chains(self) -> None:
        """Rebuild conversation chains with updated system prompt (including latest predicates)."""
        self._conversation_chain = self._get_conversation_chain()
        self._undef_failure_chain = self._get_undef_failure_chain()

    @ce.stop_program_upon_failure
    def get_session_history(self, session_id: str):
        """Shared session history for all chains to maintain conversation context."""
        # If this is a new session (different from current), rebuild chains with updated predicates
        if session_id != self._current_session_id:
            self._current_session_id = session_id
            self._rebuild_chains()

        if session_id not in self._memory_store:
            self._memory_store[session_id] = InMemoryChatMessageHistory()
        return self._memory_store[session_id]

    @ce.stop_program_upon_failure
    def delete_session_history(self, session_id: str):
        """Utility method to clear conversation history for a session."""
        if session_id in self._memory_store:
            del self._memory_store[session_id]
        self._relevant_predicates.clear()

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
        # Advanced models (Claude Sonnet 4.5/4.6, Claude Opus 4) support a 1,000,000-token
        # context window, other models handles from 200,000 to 1,000,000 tokens
        # There are around 367 c3p0 slim classes, so there can be at most 367 predicates,
        # And consider another 100 predicates as an additional predicates, hence managable
        predicates = self._get_predicates_for_system_prompt()
        if len(predicates) > 0:
            return (
                "\nAlso, here is the list of predicates along with their arguments that "
                "were already defined in previous iterations for other CHEBI classes.\n"
                "If any predicate has no arguments, then just the predicate name is shown "
                "without parentheses. You can reuse these predicates if they are "
                "applicable to the current class definition.\n"
                f"Predicate List: {', '.join(predicates)}"
            )
        return ""

    def _get_predicates_for_system_prompt(self) -> list[str]:
        if self.predicate_prompt_mode == "all":
            return sorted(self.generated_predicates_names)
        return self._relevant_predicates

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
    ) -> ChEBI_FOL_AT:
        try:
            self._refresh_relevant_predicates(input_text)
            # Get session history
            history = self.get_session_history(session_id)

            self._enforce_full_prompt_token_limit(
                input_text=input_text,
                history_messages=history.messages,
                call_name="invoke_llm_first_call",
            )

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
    ) -> ChEBI_FOL_AT:
        try:
            error_prompt = self._get_err_failure_prompt(error_message)
            # Get session history
            history = self.get_session_history(session_id)

            self._enforce_full_prompt_token_limit(
                input_text=error_prompt,
                history_messages=history.messages,
                call_name="invoke_llm_with_error_failure_prompt",
            )

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
        retry_context: str | None = None,
    ) -> OutOfBoxPredicateDefinitions:
        try:
            undefined_predicates_text = self._get_undef_failure_prompt(
                undefined_predicates,
                retry_context=retry_context,
            )
            # Get session history
            history = self.get_session_history(session_id)

            self._enforce_full_prompt_token_limit(
                input_text=undefined_predicates_text,
                history_messages=history.messages,
                call_name="invoke_llm_with_undef_failure_prompt",
            )

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
        retry_context: str | None = None,
    ) -> str:
        prompt_text = load_yaml_sys_prompt(
            self.undef_failure_prompt_fp, key="failure_prompt"
        )
        undefined_predicates_txt = "\n".join(
            f"  - Predicate: {name}"
            + (f"\n    Chemical Definition: {definition}" if definition else "")
            for name, definition in undefined_predicates_details.items()
        )
        retry_context_section = (
            f"\n\nAdditional context from the previous failed attempt:\n{retry_context}\n"
            if retry_context
            else ""
        )
        prompt_text = prompt_text.format(
            undefined_predicates_details=undefined_predicates_txt,
            retry_context_section=retry_context_section,
        )
        return prompt_text

    def __repr__(self) -> str:
        return f"""
        ChebiPrompt(platform={self.platform},
        model_name={self.model_name},
        system_prompt_fp={self.system_prompt_fp},
        few_shot_prompt_fp={self.few_shot_prompt_fp},
        err_failure_prompt_fp={self.err_failure_prompt_fp},
        undef_failure_prompt_fp={self.undef_failure_prompt_fp},
        predicate_prompt_mode={self.predicate_prompt_mode})
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

    def _enforce_full_prompt_token_limit(
        self,
        *,
        input_text: str,
        history_messages: list[BaseMessage],
        call_name: str,
    ) -> None:
        if self.MAX_INPUT_TOKENS is None:
            return None  # No token limit for this platform/model

        prompt_template = self._get_prompt_template()
        assembled_messages = prompt_template.format_messages(
            input=input_text,
            history=history_messages,
        )
        token_count = self._count_prompt_messages_tokens(assembled_messages)
        if token_count > self.MAX_INPUT_TOKENS:
            raise ValueError(
                "Prompt token limit exceeded before inference "
                f"in {call_name}: {token_count} > {self.MAX_INPUT_TOKENS}. "
                "Count includes system prompt, few-shot examples, conversation history, and input."
            )

    def _count_prompt_messages_tokens(self, messages: list[BaseMessage]) -> int:
        get_num_tokens_from_messages = getattr(
            self._llm, "get_num_tokens_from_messages", None
        )
        if callable(get_num_tokens_from_messages):
            token_counter = cast(
                Callable[[list[BaseMessage]], int], get_num_tokens_from_messages
            )
            try:
                return int(token_counter(messages))
            except Exception:
                pass

        # Conservative fallback for models that do not expose message-level token counting.
        serialized_prompt = "\n".join(
            f"{msg.type}: {msg.content if isinstance(msg.content, str) else str(msg.content)}"
            for msg in messages
        )
        return self._count_prompt_tokens(serialized_prompt)

    def _count_prompt_tokens(self, text: str) -> int:
        """Count tokens using model-specific tokenizer when available.

        We intentionally avoid LangChain's BaseLanguageModel.get_num_tokens fallback,
        which uses a GPT-2 tokenizer and emits a warning for non-GPT-2 models.
        """
        get_num_tokens = getattr(self._llm, "get_num_tokens", None)
        base_fallback = getattr(BaseLanguageModel, "get_num_tokens", None)
        bound_func = getattr(get_num_tokens, "__func__", None)

        if callable(get_num_tokens) and bound_func is not base_fallback:
            token_counter = cast(Callable[[str], int], get_num_tokens)
            return int(token_counter(text))

        # Conservative fallback used only when model tokenizer API is unavailable.
        # Typical English tokenization is roughly 1 token per 4 chars.
        return max(1, len(text) // 4)

    @ce.stop_program_upon_failure
    def add_predicates_to_memory(self, predicate_name: str, vars: list[str]) -> None:
        """Add predicates to the prompt predicate store.

        Example: if pred_name='oligopeptide' and vars=[x0, x1],
                 this will add 'oligopeptide(x0, x1)'.

        If no variables, only the predicate name is added.
        """
        if len(vars) > 0:
            variables_str = ", ".join(str(var) for var in vars)
            predicate_with_vars = f"{predicate_name}({variables_str})"
        else:
            predicate_with_vars = predicate_name

        if self.predicate_prompt_mode == "all":
            if predicate_with_vars in self.generated_predicates_names:
                return
            self.generated_predicates_names.add(predicate_with_vars)
        elif (
            self.predicate_prompt_mode == "relevant"
            and self._predicate_retriever is not None
        ):
            self._predicate_retriever.add_predicate(predicate_with_vars)
        else:
            raise ValueError(
                "Invalid predicate_prompt_mode or missing predicate retriever."
            )

    def _refresh_relevant_predicates(self, input_text: str) -> None:
        if self.predicate_prompt_mode != "relevant":
            return
        if self._predicate_retriever is None:
            return
        self._relevant_predicates = (
            self._predicate_retriever.retrieve_relevant_predicates(input_text)
        )


if __name__ == "__main__":
    # ------------------- TESTING THE CLASS ------------------#
    # Example usage
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
    chebai_prompt.add_predicates_to_memory("primary_alcohol", [])
    chebai_prompt.add_predicates_to_memory("hydroxy_group", [])

    # Use the same session_id across all invocations to maintain conversation history
    test_session_id = "test_session"

    print("---" * 10, "FEW-SHOT PROMPT TEST", "---" * 10)
    # Test the few-shot prompt
    result = chebai_prompt.invoke_llm_first_call(
        input_text=chebi_def, session_id=test_session_id
    )
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
