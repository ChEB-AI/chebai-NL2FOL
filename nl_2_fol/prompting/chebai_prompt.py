import json
from typing import cast

from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableWithMessageHistory

from nl_2_fol.inference import custom_exceptions as ce
from nl_2_fol.prompting.llm_inference import API_PLATFORM, get_llm_for_inference
from nl_2_fol.prompting.prompt_models import (
    CHEBIFOLOutput,
    OutOfBoxPredicateDefinitions,
)
from nl_2_fol.prompting.utils.read_configs import json_to_pyObj, load_yaml_sys_prompt

# TODO: Use memory in case of failure prompts
# TODO: llm with structured output


# --- Main Class ---
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

        self._undef_parser = PydanticOutputParser(
            pydantic_object=OutOfBoxPredicateDefinitions
        )

        self._conversation_chain = self._get_conversation_chain()
        self._undef_failure_prompt = self._get_undef_failure_prompt()
        self._undef_failure_chain = self._get_undef_failure_chain()

    def _get_session_history(self, session_id: str):
        """Shared session history for all chains to maintain conversation context."""
        if session_id not in self._memory_store:
            self._memory_store[session_id] = ConversationBufferMemory(
                memory_key="history",
                human_prefix="human",
                ai_prefix="ai",
                return_messages=True,
            )
        return self._memory_store[session_id]

    def _get_conversation_chain(self):
        system_prompt = self._get_system_prompt()
        few_shot_prompt = self._get_few_shot_prompt()

        prompt = ChatPromptTemplate.from_messages(
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

        llm = get_llm_for_inference(self.platform, self.model_name)
        structured_llm = llm.with_structured_output(CHEBIFOLOutput)

        chain = prompt | structured_llm

        chain_with_memory = RunnableWithMessageHistory(
            cast(Runnable, chain),
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return chain_with_memory

    # --------------------- System Prompt and Predicates List Management --------------------- #
    def _get_system_prompt(self) -> SystemMessagePromptTemplate:
        system_prompt_text = load_yaml_sys_prompt(self.system_prompt_fp)
        predicates_list_text = self._get_predicates_section()
        full_system_prompt_text = system_prompt_text + predicates_list_text
        return SystemMessagePromptTemplate.from_template(full_system_prompt_text)

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

    ## ---------------- Few-Shot Prompt Construction ---------------- ##
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

    ## ----------------- LLM Invocation ------------------ ##
    @ce.stop_program_upon_failure
    def invoke_llm_first_call(
        self, input_text: str, session_id: str = "default"
    ) -> CHEBIFOLOutput:
        try:
            input_text = self._normalize_input_text(input_text)

            output = self._conversation_chain.invoke(
                {
                    "input": input_text,
                },
                config={"configurable": {"session_id": session_id}},
            )
            return output
        except Exception as e:
            print(f"Error during inference: {e}")
            raise e

    ## ----------------- FOL Definition Error Failure Prompt ----------------- ##
    @ce.stop_program_upon_failure
    def invoke_llm_with_error_failure_prompt(
        self, error_message: str, session_id: str = "default"
    ) -> CHEBIFOLOutput:
        try:
            error_text = self._normalize_input_text(error_message)
            error_prompt = self._get_err_failure_prompt(error_text)
            output = self._conversation_chain.invoke(
                {
                    "input": error_prompt,
                },
                config={"configurable": {"session_id": session_id}},
            )
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

    ## ----------------- FOL Definition Undefined Predicates Failure Prompt ----------------- ##

    def _get_undef_failure_prompt(self) -> ChatPromptTemplate:
        # Load the raw string from YAML using the specific key
        prompt_text = load_yaml_sys_prompt(
            self.undef_failure_prompt_fp, key="failure_prompt"
        )

        # Create a template that includes format instructions
        format_instructions = self._undef_parser.get_format_instructions()
        full_prompt_text = prompt_text.replace(
            "{undef_predicates_format_instructions}", format_instructions
        )

        # Create the complete prompt template with history support
        # Expected variables: {input}, {previous_fol_definition},
        #  {undefined_predicates_details}, {learned_predicates_list}, {history}
        return ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                ("human", full_prompt_text),
            ]
        )

    def _get_undef_failure_chain(self) -> RunnableWithMessageHistory:
        """Create a chain specifically for handling undefined predicates with shared memory."""
        llm = get_llm_for_inference(self.platform, self.model_name)
        structured_llm = llm.with_structured_output(OutOfBoxPredicateDefinitions)

        chain = self._undef_failure_prompt | structured_llm

        # Wrap with same memory as main conversation chain
        chain_with_memory = RunnableWithMessageHistory(
            cast(Runnable, chain),
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return chain_with_memory

    @ce.stop_program_upon_failure
    def invoke_llm_with_undef_failure_prompt(
        self,
        input_text: str,
        previous_fol_definition: str,
        undefined_predicates: dict[str, str | None],
        session_id: str = "default",
    ) -> tuple[OutOfBoxPredicateDefinitions, str]:
        try:
            input_text = self._normalize_input_text(input_text)
            previous_fol_definition = self._normalize_input_text(
                previous_fol_definition
            )
            undefined_predicates_details = "\n".join(
                f"  - Predicate: {name}"
                + (f"\n    Chemical Definition: {definition}" if definition else "")
                for name, definition in undefined_predicates.items()
            )
            prompt_text = self.get_undef_failure_with_given_inputs(
                input_text, previous_fol_definition, undefined_predicates_details
            )
            # Invoke the chain with dynamic predicates list and shared session history
            output = self._undef_failure_chain.invoke(
                {
                    "input": input_text,
                    "previous_fol_definition": previous_fol_definition,
                    "undefined_predicates_details": undefined_predicates_details,
                    "learned_predicates_list": self._get_predicates_section(),
                },
                config={"configurable": {"session_id": session_id}},
            )
            return output, prompt_text
        except Exception as e:
            print(f"Error during failure prompt inference: {e}")
            raise e

    def get_undef_failure_with_given_inputs(
        self,
        input_text: str,
        previous_fol_definition: str,
        undefined_predicates_details: str,
    ) -> str:
        prompt_messages = self._undef_failure_prompt.format_messages(
            input=input_text,
            previous_fol_definition=previous_fol_definition,
            undefined_predicates_details=undefined_predicates_details,
            learned_predicates_list=self._get_predicates_section(),
        )
        prompt_text = "\n".join(
            [f"--- {m.type.upper()} MESSAGE ---\n{m.content}" for m in prompt_messages]
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
        platform="groq",
        model_name="openai/gpt-oss-120b",
        system_prompt_fp=os.path.join(
            prompt_dir, "system_prompts", "with_predicates_list.yaml"
        ),
        few_shot_prompt_fp=os.path.join(prompt_dir, "few_shots", "with_DL_style.json"),
        err_failure_prompt_fp=os.path.join(prompt_dir, "failure", "error_prompt.yaml"),
        undef_failure_prompt_fp=os.path.join(
            prompt_dir, "failure", "predicates_undefined.yaml"
        ),
    )

    chebi_def = """CHEBI:16236 - ethanol: A primary alcohol that
        is ethane in which one of the hydrogens is substituted
        by a hydroxy group."""
    chebai_prompt.generated_predicates_names.add("primary_alcohol")
    chebai_prompt.generated_predicates_names.add("hydroxy_group")

    # Use the same session_id across all invocations to maintain conversation history
    test_session_id = "test_session"

    print("---" * 50, "FEW-SHOT PROMPT TEST", "---" * 50)
    # Test the few-shot prompt
    result = chebai_prompt.invoke_llm_first_call(chebi_def, session_id=test_session_id)
    print(f"Few-shot result:\n {result}")
    print("\n\n\n")

    previous_fol_definition = """ethanol <=> (PrimaryAlcohol AND (is_a Ethane)
    AND (has_part SOME HydroxyGroup))"""
    error_message = """ Unknow predicate 'has_part' used in the FOL formula,
    which is not defined in the system prompt.",
    """

    print("---" * 50, "FOL DEFINITION ERROR FAILURE PROMPT TEST", "---" * 50)
    # Test the failure prompt - uses same session to maintain context
    failure_result = chebai_prompt.invoke_llm_with_error_failure_prompt(
        error_message=error_message, session_id=test_session_id
    )
    print(f"Failure result:\n {failure_result}")

    print(
        "---" * 50,
        "FOL DEFINITION UNDEFINED PREDICATES FAILURE PROMPT TEST",
        "---" * 50,
    )
    previous_fol_definition = """ethanol <=> (PrimaryAlcohol AND (is_a Ethane)
    AND (has_part SOME HydroxyGroup))"""
    undefined_predicates = {
        "has_part": "A predicate indicating that a chemical entity has a certain part or component.",
        "SOME": None,
    }
    # This invocation now shares the same conversation history as the previous calls
    failure_result, failure_prompt_text = (
        chebai_prompt.invoke_llm_with_undef_failure_prompt(
            input_text=chebi_def,
            previous_fol_definition=previous_fol_definition,
            undefined_predicates=undefined_predicates,
            session_id=test_session_id,  # Same session = shared memory
        )
    )
    print(f"Undefined predicates failure prompt text: \n {failure_prompt_text} \n\n\n")
    print(f"Undefined predicates failure result:\n {failure_result}")
