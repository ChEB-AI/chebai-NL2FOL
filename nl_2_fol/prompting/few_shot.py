import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
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
class ChebaiFewShotPrompt:
    def __init__(
        self,
        platform: str,
        model_name: str,
        system_prompt_fp: str,
        few_shot_prompt_fp: str,
    ):
        self._parser = PydanticOutputParser(pydantic_object=CHEBIFOLOutput)
        self._sys_promt = self.get_system_prompt(system_prompt_fp)
        self._few_shot_promt = self.get_few_shot_prompt(few_shot_prompt_fp)

        self._final_prompt = ChatPromptTemplate.from_messages(
            [
                self._sys_promt,
                self._few_shot_promt,
                ("human", "{input}"),
            ]
        )
        self._llm = get_llm_for_inference(platform, model_name)

        # Create the execution chain once
        # This pipes the prompt -> LLM -> Parser automatically
        self._chain = self._final_prompt | self._llm | self._parser

    def get_system_prompt(self, system_prompt_fp: str) -> SystemMessagePromptTemplate:
        system_prompt_text = load_yaml_sys_prompt(system_prompt_fp)
        format_instructions = self._parser.get_format_instructions()

        # Create inner template and bind instructions
        prompt = PromptTemplate.from_template(system_prompt_text)
        partial_prompt = prompt.partial(format_instructions=format_instructions)

        return SystemMessagePromptTemplate(prompt=partial_prompt)

    def get_few_shot_prompt(
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

    def invoke_llm(self, input_text: str) -> CHEBIFOLOutput:
        try:
            return self._chain.invoke({"input": input_text})
        except Exception as e:
            print(f"Error during inference: {e}")
            raise e

    # --- Debugging Helpers ---
    def print_whole_prompt_with_given_input(self, input_text):
        messages = self._final_prompt.format_messages(input=input_text)
        print("-" * 15, "ENTIRE PROMPT", "-" * 15)
        for m in messages:
            print(f"--- {m.type.upper()} MESSAGE ---")
            print(m.content)
        print("-" * 30)
