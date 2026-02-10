import os

from nl_2_fol.prompting import GROQ_API_KEY_NAME, OPENAI_API_KEY_NAME


def get_llm_for_inference(platform, model_name):
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
                "Please install openai by using `pip install langchain-openai`"
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

    elif model_name == "t5-3b-nl-to-fol":
        """
        Inference need to be run on a GPU-enabled machine.
        You can use the below command to access one such machine on the cluster:

            srun --partition=gpu --constraint="A100|H100.80gb" --ntasks=1
            --cpus-per-task=8 --threads-per-core=1 --mem=64G
            --time=02:00:00 --gres=gpu:1 --pty bash
        """
        from nl_2_fol.prompting.custom_api import T5_3B_NL2FOL

        platform = "custom"
        llm = T5_3B_NL2FOL()

        result = llm.invoke("All dogs are animals.")
        if result is None:
            raise Exception(
                f"Didn't recieve any response from Model `{model_name}` from platform `{platform}`"
            )
        return llm

    raise ValueError("Unknown platform")


if __name__ == "__main__":
    # Example usage:
    llm = get_llm_for_inference(platform="custom", model_name="t5-3b-nl-to-fol")
