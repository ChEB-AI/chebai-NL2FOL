import os

from . import GROQ_API_KEY_NAME, OPENAI_API_KEY_NAME


def get_llm_for_inference(platform, model_name):
    def _test_api_with_a_prompt(llm):
        result = llm.invoke("Hello LLM").content
        if result:
            print(
                "Model `{model_name}` from platform `{platform}` is ready for inference."
            )
            return

        raise Exception(
            "Didn't recieve any response from Model `{model_name}` from platform `{platform}`"
        )

    if platform == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "Please install groq by using `pip install langchain-groq`"
            )

        assert GROQ_API_KEY_NAME in os.environ, (
            f"Please set the api key {GROQ_API_KEY_NAME} for groq in `api_keys.env` file."
        )

        llm = ChatGroq(
            model_name=model_name,  # "openai/gpt-oss-120b",
            temperature=0,
        )
        _test_api_with_a_prompt(llm)
        return llm

    elif platform == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "Please install groq by using `pip install langchain-openai`"
            )
        assert OPENAI_API_KEY_NAME in os.environ, (
            f"Please set the api key {OPENAI_API_KEY_NAME} for groq in `api_keys.env` file."
        )

        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
        )
        _test_api_with_a_prompt(llm)
        return llm

    raise ValueError("Unknown platform")
