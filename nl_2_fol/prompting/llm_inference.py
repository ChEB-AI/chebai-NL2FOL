import os
from typing import Literal

from nl_2_fol.prompting import ANTHROPIC_API_KEY, GROQ_API_KEY_NAME, OPENAI_API_KEY_NAME

API_PLATFORM = Literal["groq", "anthropic", "openai", "custom"]


def get_llm_for_inference(platform: API_PLATFORM, model_name):
    def _test_api_with_a_prompt(llm):
        result = llm.invoke("Hello LLM").content
        if result:
            print(
                f"Model `{model_name}` from platform `{platform}` is ready for inference."
            )
            return

        raise Exception(
            f"Didn't recieve any response from Model `{model_name}` from platform `{platform}`"
        )

    if platform == "groq":
        try:
            from langchain_groq import ChatGroq  # type: ignore
        except ImportError:
            raise ImportError(
                "Please install groq by using `pip install langchain-groq`"
            )

        assert GROQ_API_KEY_NAME in os.environ, (
            f"Please set the api key {GROQ_API_KEY_NAME} for groq in `api_keys.env` file."
        )

        llm = ChatGroq(
            model=model_name,  # "openai/gpt-oss-120b",
            temperature=0.0,
        )
        _test_api_with_a_prompt(llm)
        return llm

    elif platform == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
        except ImportError:
            raise ImportError(
                "Please install anthropic by using `pip install langchain-anthropic`"
            )

        assert ANTHROPIC_API_KEY in os.environ, (
            f"Please set the api key {ANTHROPIC_API_KEY} for anthropic in `api_keys.env` file."
        )

        llm = ChatAnthropic(
            model_name=model_name,  # "claude-opus-4-6",
            temperature=0.0,
        )  # pyright: ignore[reportCallIssue]
        _test_api_with_a_prompt(llm)
        return llm
    elif platform == "openai":
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError:
            raise ImportError(
                "Please install openai by using `pip install langchain-openai`"
            )
        assert OPENAI_API_KEY_NAME in os.environ, (
            f"Please set the api key {OPENAI_API_KEY_NAME} for openai in `api_keys.env` file."
        )

        llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
        )
        _test_api_with_a_prompt(llm)
        return llm

    elif platform == "custom":
        """
        Inference need to be run on a GPU-enabled machine.
        You can use the below command to access one such machine on the cluster:

            srun --partition=gpu --constraint="A100|H100.80gb" --ntasks=1
            --cpus-per-task=8 --threads-per-core=1 --mem=64G
            --time=02:00:00 --gres=gpu:1 --pty bash
        """

        if model_name == "t5-3b-nl-to-fol":
            from nl_2_fol.prompting.custom_api.t5_model import T5_3B_NL2FOL

            custom_llm = T5_3B_NL2FOL()
        else:
            raise ValueError(f"Unknown custom model name `{model_name}`")

        from nl_2_fol.prompting.custom_api.base_chat_model import LocalModelChat

        chat_model = LocalModelChat(llm=custom_llm)

        result = chat_model.invoke("All dogs are animals.")
        if result is None:
            raise Exception(
                f"Didn't recieve any response from Model `{model_name}` from platform `{platform}`"
            )
        return chat_model

    raise ValueError("Unknown platform")


if __name__ == "__main__":
    # import anthropic
    # client = anthropic.Anthropic()
    # models = client.models.list()
    # for model in models.data:
    #     print(model.id)

    # Example usage:
    llm = get_llm_for_inference(platform="anthropic", model_name="claude-opus-4-6")
