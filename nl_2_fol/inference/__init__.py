import os

STACK_TRACE_ENV_VAR = "NL2FOL_PRINT_STACK_TRACES"


def _should_print_traces() -> bool:
    env_value = os.getenv(STACK_TRACE_ENV_VAR, "")
    return env_value.lower() in {"1", "true", "yes", "on"}


PRINT_TRACES = _should_print_traces()
print(
    f"PRINT_TRACES is set to {PRINT_TRACES} based on environment variable '{STACK_TRACE_ENV_VAR}'."
)
