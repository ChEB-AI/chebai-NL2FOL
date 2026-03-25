import os
import warnings

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
    if not any(os.getenv(key) for key in API_KEYS_NAME_LIST):
        raise RuntimeError(
            f"Detailed Error: None of the required API keys ({API_KEYS_NAME_LIST}) "
            "were found in the environment variables. Please check your .env file."
        )


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _validate_langsmith_tracing_config() -> None:
    """Validate LangSmith settings only when tracing is explicitly enabled."""
    # Refer Tracibility docs: https://docs.langchain.com/langsmith/observability-quickstart
    # LANGSMITH_TRACING=true
    # LANGSMITH_PROJECT="nl2fol"
    # LANGSMITH_API_KEY=<your-api-key>
    # LANGSMITH_ENDPOINT=https://api.smith.langchain.com
    validate_any_api_key_present()

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

# Set LangSmith timeout to 30 seconds (applies when tracing is enabled)
# This prevents connection timeouts from blocking execution in offline/slow networks
os.environ.setdefault("LANGSMITH_TIMEOUT", "30")

if tracing:
    print("LangSmith tracing is enabled. Validating configuration...")
    try:
        _validate_langsmith_tracing_config()
    except RuntimeError as exc:
        # Fail open for local/offline runs while making misconfiguration obvious.
        tracing = False
        warnings.warn(f"LangSmith tracing disabled: {exc}", stacklevel=2)
else:
    # Disable LangSmith tracing by removing its environment variables
    # This prevents automatic client initialization when tracing is not explicitly enabled
    for key in [
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
    ]:
        os.environ.pop(key, None)
