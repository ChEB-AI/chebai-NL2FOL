import os

GROQ_API_KEY_NAME = "GROQ_API_KEY"
OPENAI_API_KEY_NAME = "OPENAI_API_KEY"

API_KEYS_NAME_LIST = [GROQ_API_KEY_NAME, OPENAI_API_KEY_NAME]


def set_api_env_var(file_path="./api_keys.env"):
    # Get groq api key from : https://console.groq.com/keys
    # Get openai api key from: https://openai.com/api/

    # if tracing needed in future
    # os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    # os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    # assert "LANGCHAIN_API_KEY" in os.environ

    flag_api_key_set = False
    # Read the file and set environment variables
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            # Split key and value
            key, value = line.strip().split("=", 1)
            if value is None or value.strip() == "":
                raise ValueError(f"Invalid line in {file_path}: {line}")
            # Set environment variable
            if key in API_KEYS_NAME_LIST:
                os.environ[key] = value
                print(f"Environment variable set: {key}")
                flag_api_key_set = True

    if not flag_api_key_set:
        raise ValueError(
            f"No valid api key found in {file_path}\n",
            f"Use of the following api keys: {API_KEYS_NAME_LIST}",
        )


set_api_env_var()
