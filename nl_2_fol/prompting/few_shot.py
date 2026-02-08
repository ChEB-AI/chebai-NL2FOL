import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, Field

from nl_2_fol.utils.read_configs import json_to_pyObj, load_yaml_sys_prompt

from .llm_inference import get_llm_for_inference


# --- Pydantic Models ---
class IntermediateOutput(BaseModel):
    relevant_definition: str = Field(
        ..., description="Relevant part of the CHEBI definition"
    )
    superclass: str = Field(..., description="Superclass of the CHEBI class")
    explanation: str = Field(..., description="How the class is defined")


class CHEBIFOLOutput(BaseModel):
    intermediate_output: IntermediateOutput
    FOL_formula: str = Field(..., description="First-order logic formula")


# --- Main Class ---
class ChebaiPrompt:
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

    def _get_system_prompt_for_fs(
        self, system_prompt_fp: str
    ) -> SystemMessagePromptTemplate:
        system_prompt_text = load_yaml_sys_prompt(system_prompt_fp)
        format_instructions = self._few_shot_parser.get_format_instructions()

        # Create inner template and bind instructions
        prompt = PromptTemplate.from_template(system_prompt_text)
        partial_prompt = prompt.partial(format_instructions=format_instructions)

        return SystemMessagePromptTemplate(prompt=partial_prompt)

    def _get_few_shot_prompts_examples(
        self, few_shot_prompt_fp: str
    ) -> FewShotChatMessagePromptTemplate:
        raw_examples = json_to_pyObj(few_shot_prompt_fp)

        processed_examples = [
            {
                "human": ex["human"],
                "ai": json.dumps(ex["ai"], indent=4, ensure_ascii=False),
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

    def invoke_llm_with_fs_prompt(self, input_text: str) -> CHEBIFOLOutput | Exception:
        try:
            return self._few_shots_chain.invoke({"input": input_text})
        except Exception as e:
            print(f"Error during inference: {e}")
            return e

    def print_fs_prompt_with_given_input(self, input_text):
        messages = self._fs_entire_prompt.format_messages(input=input_text)
        print("-" * 30, "ENTIRE PROMPT", "-" * 30)
        for m in messages:
            print(f"--- {m.type.upper()} MESSAGE ---")
            print(m.content)
        print("-" * 30, "END OF THE PROMPT", "-" * 30)

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
    ) -> CHEBIFOLOutput | Exception:
        try:
            return self._failure_chain.invoke(
                {
                    "input": input_text,
                    "previous_fol_definition": previous_fol_definition,
                    "error_message": error_message,
                }
            )
        except Exception as e:
            print(f"Error during failure prompt inference: {e}")
            return e

    def print_failure_prompt_with_given_inputs(
        self, input_text: str, previous_fol_definition: str, error_message: str
    ):
        messages = self._failure_prompt.format_messages(
            input=input_text,
            previous_fol_definition=previous_fol_definition,
            error_message=error_message,
        )
        print("-" * 30, "FAILURE PROMPT", "-" * 30)
        for m in messages:
            print(f"--- {m.type.upper()} MESSAGE ---")
            print(m.content)
        print("-" * 30, "END OF THE FAILURE PROMPT", "-" * 30)


if __name__ == "__main__":
    # ------------------- TESTING THE CLASS ------------------#
    # Example usage
    chebai_prompt = ChebaiPrompt(
        platform="groq",
        model_name="openai/gpt-oss-120b",
        system_prompt_fp="prompt_templates/system_prompts/with_predicates_list.yaml",
        few_shot_prompt_fp="prompt_templates/few_shots/with_DL_style.json",
        failure_prompt_fp="prompt_templates/failure/prompt.yaml",
    )

    chebi_def = """CHEBI:16236 - ethanol: A primary alcohol that
        is ethane in which one of the hydrogens is substituted
        by a hydroxy group."""

    chebai_prompt.print_fs_prompt_with_given_input(chebi_def)

    # Test the few-shot prompt
    output = chebai_prompt.invoke_llm_with_fs_prompt(chebi_def)
    print(output)

    previous_fol_definition = """ethanol <=> (PrimaryAlcohol AND (is_a Ethane)
    AND (has_part SOME HydroxyGroup))"""
    error_message = """ Unknow predicate 'has_part' used in the FOL formula, 
    which is not defined in the system prompt.",
    """

    chebai_prompt.print_failure_prompt_with_given_inputs(
        input_text=chebi_def,
        previous_fol_definition=previous_fol_definition,
        error_message=error_message,
    )

    # Test the failure prompt
    failure_output = chebai_prompt.invoke_llm_with_failure_prompt(
        input_text=chebi_def,
        previous_fol_definition=previous_fol_definition,
        error_message=error_message,
    )
    print(failure_output)
