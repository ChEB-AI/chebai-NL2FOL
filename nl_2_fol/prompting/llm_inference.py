import os
import re
from typing import Literal

from nl_2_fol.prompting import ANTHROPIC_API_KEY, GROQ_API_KEY_NAME, OPENAI_API_KEY_NAME

# claude_cli: Claude via the logged-in `claude` CLI (Claude subscription, no API key)
# anthropic: Claude via the API (API key)
API_PLATFORM = Literal["claude_cli", "groq", "anthropic", "openai", "ollama", "custom"]
EFFORT = Literal["low", "medium", "high", "xhigh", "max"]

ANTHROPIC_MAX_TOKENS = 16000

_CLAUDE_VERSION = re.compile(r"^claude-(?:(?:opus|sonnet|haiku)-)?(\d+)(?:-(\d+))?")


def _anthropic_model_supports_temperature(model_name: str) -> bool:
    """Whether a Claude model accepts sampling parameters (`temperature`).

    Claude models up to version 4.6 accept them; newer models (Opus 4.7+, Opus 5.x,
    Haiku 5.5, Fable, ...) reject them with a 400 error.
    """
    match = _CLAUDE_VERSION.match(model_name)
    if match is None:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    if minor >= 100:  # date suffix, e.g. claude-opus-4-20250514
        minor = 0
    return major < 4 or (major == 4 and minor <= 6)


def get_llm_for_inference(
    platform: API_PLATFORM, model_name, effort: EFFORT | None = None
):
    """Create a chat model.

    Args:
        effort: Reasoning effort (only used for Anthropic models that support it,
            e.g. Claude Opus 5.5 / Haiku 5.5). If None, the API default is used.
    """

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

    if platform == "claude_cli":
        from nl_2_fol.prompting.claude_cli import ClaudeCLIChat

        return ClaudeCLIChat(model_name=model_name, effort=effort)

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

        llm_kwargs = {}
        if _anthropic_model_supports_temperature(model_name):
            llm_kwargs["temperature"] = 0.0
        if effort is not None:
            llm_kwargs["effort"] = effort
        llm = ChatAnthropic(
            model_name=model_name,  # "claude-opus-5-5",
            # explicit limit: the default for models without a langchain profile may be
            # too small once thinking tokens are included
            max_tokens=ANTHROPIC_MAX_TOKENS,
            **llm_kwargs,
        )  # pyright: ignore[reportCallIssue]
        # _test_api_with_a_prompt(llm)
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

    elif platform == "ollama":
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except ImportError:
            raise ImportError(
                "Please install ollama support by using `pip install langchain-ollama`"
            )

        assert "OLLAMA_HOST" in os.environ, (
            "Please set the OLLAMA_HOST environment variable to point to your Ollama server. "
            "For example: `export OLLAMA_HOST=http://localhost:11435`"
            "If on cluster, prefer to start ollama on custom port to avoid conflicts with system ollama "
        )
        assert "OLLAMA_TIMEOUT" in os.environ, (
            "Please set the OLLAMA_TIMEOUT environment variable to specify the timeout for API calls to your Ollama server. "
            "For example: `export OLLAMA_TIMEOUT=600` (timeout in seconds)"
        )
        base_url = os.environ["OLLAMA_HOST"]
        timeout = float(os.environ["OLLAMA_TIMEOUT"])

        llm = ChatOllama(
            model=model_name,
            temperature=0.0,
            base_url=base_url,
            client_kwargs={"timeout": timeout},
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
            from nl_2_fol.prompting.custom_api.models.t5 import T5_3B_NL2FOL

            custom_llm = T5_3B_NL2FOL()
        elif model_name == "Mistral-Small-24B-Instruct-nl-to-fol":
            from nl_2_fol.prompting.custom_api.models.mistral import Mistral_24B

            custom_llm = Mistral_24B()
        else:
            raise ValueError(f"Unknown custom model name `{model_name}`")

        from nl_2_fol.prompting.custom_api.base_chat_model import LocalModelChat

        chat_model = LocalModelChat(llm=custom_llm)

        _test_api_with_a_prompt(chat_model)
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
