import os

from dotenv import load_dotenv

GROQ_API_KEY_NAME = "GROQ_API_KEY"
OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
API_KEYS_NAME_LIST = [GROQ_API_KEY_NAME, OPENAI_API_KEY_NAME, ANTHROPIC_API_KEY]


load_dotenv("./api_keys.env")  # This loads the .env file
# Get groq api key from : https://console.groq.com/keys
# Get openai api key from: https://openai.com/api/


def validate_any_api_key_present() -> None:
    """Validate that at least one supported API key is available.

    Keep this as an explicit runtime check instead of running on import,
    so unit tests and CI can import modules without API credentials.
    """
    assert any(os.getenv(key) for key in API_KEYS_NAME_LIST), (
        f"Detailed Error: None of the required API keys ({API_KEYS_NAME_LIST}) "
        "were found in the environment variables. Please check your .env file."
    )


tracing = False  # Set to True in debug mode
if tracing:
    validate_any_api_key_present()
    # Refer Tracibility docs: https://docs.langchain.com/langsmith/observability-quickstart
    # View on trace of the prompting here:
    # https://smith.langchain.com/public/5dddca10-2d31-4be8-b4c3-caca98504868/r/019b55af-4679-7e31-96f0-516ffad9499d

    # LANGSMITH_TRACING=true
    # LANGSMITH_PROJECT="nl2fol"
    # LANGSMITH_API_KEY=<your-api-key>
    # LANGSMITH_ENDPOINT=https://api.smith.langchain.com

    assert (
        "LANGSMITH_TRACING" in os.environ and os.environ["LANGSMITH_TRACING"] == "true"
    )
    # assert "LANGSMITH_WORKSPACE_ID" in os.environ  # Not necessary, but good to have
    assert "LANGSMITH_API_KEY" in os.environ
    assert (
        "LANGSMITH_ENDPOINT" in os.environ
        and os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
    )
    assert (
        "LANGSMITH_PROJECT" in os.environ
        and os.environ["LANGSMITH_PROJECT"] == "nl2fol"
    )
