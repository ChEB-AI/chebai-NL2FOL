import os

from dotenv import load_dotenv

GROQ_API_KEY_NAME = "GROQ_API_KEY"
OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
API_KEYS_NAME_LIST = [GROQ_API_KEY_NAME, OPENAI_API_KEY_NAME]


load_dotenv("./api_keys.env")  # This loads the .env file
# Get groq api key from : https://console.groq.com/keys
# Get openai api key from: https://openai.com/api/


assert any(os.getenv(key) for key in API_KEYS_NAME_LIST), (
    f"Detailed Error: None of the required API keys ({API_KEYS_NAME_LIST}) "
    "were found in the environment variables. Please check your .env file."
)


tracing = False  # Set to True in debug mode
if tracing:
    # Refer Tracibility docs: https://docs.langchain.com/langsmith/observability-quickstart
    # View on trace of the prompting here:
    # https://smith.langchain.com/public/5dddca10-2d31-4be8-b4c3-caca98504868/r/019b55af-4679-7e31-96f0-516ffad9499d
    assert (
        "LANGSMITH_TRACING" in os.environ and os.environ["LANGSMITH_TRACING"] == "true"
    )
    # assert "LANGSMITH_WORKSPACE_ID" in os.environ  # Not necessary, but good to have
    assert "LANGSMITH_API_KEY" in os.environ
