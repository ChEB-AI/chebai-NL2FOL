import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from nl_2_fol.prompting.llm_inference import get_llm_for_inference
from nl_2_fol.prompting.models import CHEBIFOLOutput
from nl_2_fol.utils.read_configs import json_to_pyObj, load_yaml_sys_prompt

# TODO: Use memory in case of failure prompts
# TODO: check langgraph for better prompt chaining, iteration func calling, especially for the failure prompts where we want to keep track of previous attempts and errors.
# TODO:
#   - IF THERE ARE certain predicates which reasoner does not understand, ask llm to define the predicate.
#   - if llm defines a predicate which is not in the list, ask it to define it, verify it and then add it to list


# --- Main Class ---
class ChebiPrompt:
    def __init__(
        self,
        platform: str,
        model_name: str,
        system_prompt_fp: str,
        few_shot_prompt_fp: str,
        failure_prompt_fp: str,
    ):
        self.platform: str = platform
        self.model_name: str = model_name
        self.system_prompt_fp: str = system_prompt_fp
        self.few_shot_prompt_fp: str = few_shot_prompt_fp
        self.failure_prompt_fp: str = failure_prompt_fp

        self._few_shot_parser = PydanticOutputParser(pydantic_object=CHEBIFOLOutput)
        self._fs_entire_prompt: ChatPromptTemplate = self._get_entire_few_shot_prompt()
        self._failure_prompt = self._get_failure_prompt()

        self._llm = get_llm_for_inference(self.platform, self.model_name)

        # Create the execution chain once
        # This pipes the Few shot prompt -> LLM -> Parser automatically
        self._few_shots_chain = (
            self._fs_entire_prompt | self._llm | self._few_shot_parser
        )
        self._failure_chain = self._failure_prompt | self._llm | self._few_shot_parser

    ## ---------------- Few-Shot Prompt Construction ---------------- ##
    def _get_entire_few_shot_prompt(self) -> ChatPromptTemplate:
        self._sys_promt = self._get_system_prompt_for_fs(self.system_prompt_fp)
        self._few_shot_promt = self._get_few_shot_prompts_examples(
            self.few_shot_prompt_fp
        )

        return ChatPromptTemplate.from_messages(
            [
                self._sys_promt,
                self._few_shot_promt,
                ("human", "{input}"),
            ]
        )

    @staticmethod
    def _normalize_input_text(input_text: str) -> str:
        """
        Normalize the input text by stripping leading/trailing whitespace
        and collapsing multiple spaces into a single space.
        """
        return " ".join(str(input_text).split())

    def _get_system_prompt_for_fs(
        self, system_prompt_fp: str
    ) -> SystemMessagePromptTemplate:
        system_prompt_text = load_yaml_sys_prompt(system_prompt_fp)

        # Escape curly braces that are not template variables
        # We need to double curly braces except for {format_instructions}
        # Replace all { and } with {{ and }}, then restore {format_instructions}
        escaped_text = system_prompt_text.replace("{", "{{").replace("}", "}}")
        escaped_text = escaped_text.replace(
            "{{format_instructions}}", "{format_instructions}"
        )

        format_instructions = self._few_shot_parser.get_format_instructions()

        # Create inner template and bind instructions
        prompt = PromptTemplate.from_template(escaped_text)
        partial_prompt = prompt.partial(format_instructions=format_instructions)

        return SystemMessagePromptTemplate(prompt=partial_prompt)  # pyright: ignore[reportArgumentType]

    def _get_few_shot_prompts_examples(
        self, few_shot_prompt_fp: str
    ) -> FewShotChatMessagePromptTemplate:
        raw_examples = json_to_pyObj(few_shot_prompt_fp)

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

    def invoke_llm_with_fs_prompt(self, input_text: str) -> tuple[CHEBIFOLOutput, str]:
        try:
            input_text = self._normalize_input_text(input_text)
            # Get the formatted prompt messages
            prompt_text = self.get_fs_prompt_with_given_input(input_text)
            # Invoke the chain
            output = self._few_shots_chain.invoke({"input": input_text})
            return output, prompt_text
        except Exception as e:
            print(f"Error during inference: {e}")
            raise e

    def get_fs_prompt_with_given_input(self, input_text) -> str:
        input_text = self._normalize_input_text(input_text)
        prompt_messages = self._fs_entire_prompt.format_messages(input=input_text)
        prompt_text = "\n".join(
            [f"--- {m.type.upper()} MESSAGE ---\n{m.content}" for m in prompt_messages]
        )
        return prompt_text

    ## ----------------- FOL Definition Failure Prompt ----------------- ##

    def _get_failure_prompt(self) -> ChatPromptTemplate:
        # 1. Get the basic text template for the failure message
        # Load the raw string from YAML using the specific key
        prompt_text = load_yaml_sys_prompt(self.failure_prompt_fp, key="failure_prompt")

        # Return a PromptTemplate that manages the variables {previous_fol_definition} and {error_message}
        failure_template = PromptTemplate.from_template(prompt_text)

        # 2. Wrap it as a Human Message.
        # This tells the LLM: "The user is saying this text."
        failure_message = HumanMessagePromptTemplate(prompt=failure_template)

        # 3. Combine: [Original Context] + [Failure Instruction]
        # The resulting chain expects variables: {input}, {previous_fol_definition}, {error_message}
        return ChatPromptTemplate.from_messages(
            [
                self._fs_entire_prompt,  # Contains: System -> FewShot -> Human: {input}
                failure_message,  # Appends: Human: Your last attempt...
            ]
        )

    def invoke_llm_with_failure_prompt(
        self, input_text: str, previous_fol_definition: str, error_message: str
    ) -> tuple[CHEBIFOLOutput, str]:
        try:
            input_text = self._normalize_input_text(input_text)
            previous_fol_definition = self._normalize_input_text(
                previous_fol_definition
            )
            prompt_text = self.get_failure_with_given_inputs(
                input_text, previous_fol_definition, error_message
            )
            # Invoke the chain
            output = self._failure_chain.invoke(
                {
                    "input": input_text,
                    "previous_fol_definition": previous_fol_definition,
                    "error_message": error_message,
                }
            )
            return output, prompt_text
        except Exception as e:
            print(f"Error during failure prompt inference: {e}")
            raise e

    def get_failure_with_given_inputs(
        self, input_text: str, previous_fol_definition: str, error_message: str
    ) -> str:
        prompt_messages = self._failure_prompt.format_messages(
            input=input_text,
            previous_fol_definition=previous_fol_definition,
            error_message=error_message,
        )
        prompt_text = "\n".join(
            [f"--- {m.type.upper()} MESSAGE ---\n{m.content}" for m in prompt_messages]
        )
        return prompt_text

    def __repr__(self) -> str:
        return f"""
        ChebiPrompt(platform={self.platform},\n 
        model_name={self.model_name},\n 
        few_shot_prompt={self._fs_entire_prompt}),\n 
        failure_prompt={self._failure_prompt})\n
        """


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
        failure_prompt_fp=os.path.join(prompt_dir, "failure", "prompt.yaml"),
    )

    chebi_def = """CHEBI:16236 - ethanol: A primary alcohol that
        is ethane in which one of the hydrogens is substituted
        by a hydroxy group."""

    # Test the few-shot prompt
    result, prompt_text = chebai_prompt.invoke_llm_with_fs_prompt(chebi_def)
    print(f"Few-shot prompt text: \n {prompt_text} \n\n\n")
    print(f"Few-shot result:\n {result}")
    print("\n\n\n")

    previous_fol_definition = """ethanol <=> (PrimaryAlcohol AND (is_a Ethane)
    AND (has_part SOME HydroxyGroup))"""
    error_message = """ Unknow predicate 'has_part' used in the FOL formula, 
    which is not defined in the system prompt.",
    """

    # Test the failure prompt
    failure_result, failure_prompt_text = chebai_prompt.invoke_llm_with_failure_prompt(
        input_text=chebi_def,
        previous_fol_definition=previous_fol_definition,
        error_message=error_message,
    )
    print(f"Failure prompt text: \n {failure_prompt_text} \n\n\n")
    print(f"Failure result:\n {failure_result}")
