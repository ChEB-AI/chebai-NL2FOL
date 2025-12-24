import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, Field

from nl_2_fol.utils.read_configs import json_to_pyObj, load_yaml_sys_prompt

from .llm_inference import get_llm_for_inference


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

    def get_system_prompt(self, system_prompt_fp: str) -> SystemMessagePromptTemplate:
        # 1. Load the text from YAML
        system_prompt_text = load_yaml_sys_prompt(system_prompt_fp)

        # 2. Get the raw instructions
        format_instructions = self._parser.get_format_instructions()

        # 3. Create the template object
        system_prompt_template = SystemMessagePromptTemplate.from_template(
            system_prompt_text
        )

        # 4. Bind the instructions using .partial()
        # Note: We must use the keyword argument 'format_instructions' to match
        # the placeholder "{format_instructions}" in your YAML file.
        system_prompt_template = system_prompt_template.partial(
            format_instructions=format_instructions
        )

        return system_prompt_template

    def print_system_prompt_template(self):
        pass

    def get_few_shot_prompt(
        self,
        few_shot_prompt_fp: str,
    ) -> FewShotChatMessagePromptTemplate:
        # 1. Load the raw examples
        raw_examples = json_to_pyObj(few_shot_prompt_fp)

        # 2. PRE-PROCESSING: Convert the 'AI' dict objects to JSON strings
        # This ensures the LLM sees valid JSON (double quotes) in string format
        processed_examples = [
            {
                "human": ex["human"],
                "ai": json.dumps(ex["ai"], indent=4, ensure_ascii=False),
            }
            for ex in raw_examples
        ]

        # 3. Define the 'example_prompt'
        # This tells LangChain: "HUMAN" key goes to HumanMessage, "AI" key goes to AIMessage
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{HUMAN}"),
                ("ai", "{AI}"),
            ]
        )

        # 4. Initialize the FewShot template with the example_prompt
        return FewShotChatMessagePromptTemplate(
            examples=processed_examples,
            example_prompt=example_prompt,
        )

    def print_few_shot_promt_template(self):
        pass

    def print_whole_prompt_with_given_input(self, input):
        messages = self._final_prompt.format_messages(input=input)

        for m in messages:
            print(f"{m.type}: {m.content}\n")

    def invoke_llm(self, input) -> "CHEBIFOLOutput":
        llm_output = self._llm.invoke(self._final_prompt.format(input=input))
        result: CHEBIFOLOutput = self._parser.parse(llm_output.content)

        return result


class IntermediateOutput(BaseModel):
    relevant_definition: str = Field(
        ..., description="Relevant part of the CHEBI definition"
    )
    superclass: str = Field(..., description="Superclass of the CHEBI class")
    explanation: str = Field(..., description="How the class is defined")


class CHEBIFOLOutput(BaseModel):
    intermediate_output: IntermediateOutput
    FOL_formula: str = Field(..., description="First-order logic formula")


if __name__ == "__main__":
    data_obj = ChebaiFewShotPrompt(
        system_prompt_fp="", few_shot_prompt_fp="./configs/few_shots_prompts.json"
    )
