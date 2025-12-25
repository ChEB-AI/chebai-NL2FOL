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


tracing = True
if tracing:
    # Refer Tracibility docs: https://docs.langchain.com/langsmith/observability-quickstart
    assert (
        "LANGSMITH_TRACING" in os.environ and os.environ["LANGSMITH_TRACING"] == "true"
    )
    assert "LANGSMITH_WORKSPACE_ID" in os.environ  # Not necessary, but good to have
    assert "LANGSMITH_API_KEY" in os.environ
