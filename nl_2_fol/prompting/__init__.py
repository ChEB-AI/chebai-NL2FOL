import os

from dotenv import load_dotenv

GROQ_API_KEY_NAME = "GROQ_API_KEY"
OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
API_KEYS_NAME_LIST = [GROQ_API_KEY_NAME, OPENAI_API_KEY_NAME, ANTHROPIC_API_KEY]


load_dotenv("./api_keys.env")  # This loads the .env file
# Get groq api key from : https://console.groq.com/keys
# Get openai api key from: https://openai.com/api/

# Avoid tokenizers parallelism + fork warning/deadlock risk unless user explicitly overrides.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _validate_langsmith_tracing_config() -> None:
    """Validate LangSmith settings only when tracing is explicitly enabled."""
    # Refer Tracibility docs: https://docs.langchain.com/langsmith/observability-quickstart
    # LANGSMITH_TRACING=true
    # LANGSMITH_PROJECT="nl2fol"
    # LANGSMITH_API_KEY=<your-api-key>
    # LANGSMITH_ENDPOINT=https://api.smith.langchain.com

    required_env = ["LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT"]
    missing = [name for name in required_env if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Detailed Error: LangSmith tracing is enabled but required settings are missing: "
            f"{missing}."
        )

    endpoint = os.getenv("LANGSMITH_ENDPOINT", "")
    if endpoint != "https://api.smith.langchain.com":
        raise RuntimeError(
            "Detailed Error: LANGSMITH_ENDPOINT must be set to "
            "'https://api.smith.langchain.com'."
        )


tracing = _is_truthy(os.getenv("LANGSMITH_TRACING"))

if tracing:
    print("LangSmith tracing is enabled. Validating configuration...")
    _validate_langsmith_tracing_config()
else:
    print(
        "LangSmith tracing is disabled. To enable, set `LANGSMITH_TRACING`"
        " to a truthy value and provide necessary configuration in environment variables."
    )
    # Disable LangSmith tracing by removing its environment variables
    # This prevents automatic client initialization when tracing is not explicitly enabled
    for key in [
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_TRACING",
        "LANGCHAIN_TRACING_V2",
    ]:
        os.environ.pop(key, None)
