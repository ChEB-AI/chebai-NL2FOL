"""Claude via the local ``claude`` CLI (Claude Agent SDK), billed to the Claude subscription.

Adapted from ``chebILP.predicate_generation.llm_client``. The SDK drives the logged-in
``claude`` CLI as a subprocess (``claude``, then ``/login``) and returns JSON matching a
schema (``output_format`` / ``structured_output``). Set ``NL2FOL_CLAUDE_CLI`` to the CLI
binary if it is not on ``PATH``.

Each LLM call is one CLI query without tools or user / project settings. The conversation
built by :class:`~nl_2_fol.prompting.chebai_prompt.ChebiPrompt` (system prompt, few-shot
examples, history, new input) is passed as the system prompt and one user prompt that
contains the previous turns.

Usage limits: before each query, the client pauses until the 5-hour subscription window
resets if the previous query reported it at least ``NL2FOL_SESSION_LIMIT`` (default 90%)
full. Hitting a weekly limit raises :class:`UsageLimitReached`.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

_CLI_PATH = os.environ.get("NL2FOL_CLAUDE_CLI")
# A wedged CLI subprocess (auth prompt, network stall, ...) would otherwise block forever
_CLI_TIMEOUT = float(os.environ.get("NL2FOL_CLAUDE_CLI_TIMEOUT", "900"))
_SESSION_LIMIT = float(os.environ.get("NL2FOL_SESSION_LIMIT", "0.9"))
_RESET_MARGIN = 120  # seconds past the reported reset before querying again
_MAX_RETRIES = 5


class ClaudeCLIError(RuntimeError):
    """No usable answer from the model after all retries."""


class ModelRefusal(ClaudeCLIError):
    """The model's safety classifier declined the request.

    Reasking is pointless (the classifier is deterministic per input), so this is not retried.
    """


class UsageLimitReached(ClaudeCLIError):
    """A (weekly) subscription usage limit was reached; the run has to be resumed later."""


def _drop_api_key_auth() -> None:
    """Remove API-key auth from the environment so that the CLI uses its subscription login.

    The SDK passes ``os.environ`` to the CLI subprocess; an ``ANTHROPIC_API_KEY`` there (e.g.
    from ``api_keys.env``) would take precedence over the login and bill the API key instead.
    """
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        os.environ.pop(var, None)


def _message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in message.content
    )


def render_messages(messages: list[BaseMessage]) -> tuple[str, str]:
    """Turn a chat (system message, previous turns, final user message) into
    ``(system_prompt, prompt)`` for a single CLI query.

    Previous turns (few-shot examples and conversation history) are included in the prompt
    as a transcript.
    """
    system = "\n\n".join(
        _message_text(m) for m in messages if isinstance(m, SystemMessage)
    )
    turns = [m for m in messages if not isinstance(m, SystemMessage)]
    if not turns or not isinstance(turns[-1], HumanMessage):
        raise ValueError("The last message has to be a user message")
    *previous, last = turns
    if not previous:
        return system, _message_text(last)

    transcript = []
    for message in previous:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        transcript.append(f"<{role}>\n{_message_text(message)}\n</{role}>")
    prompt = (
        "<conversation_history>\n"
        + "\n".join(transcript)
        + "\n</conversation_history>\n\n"
        "The conversation so far (including the examples) is shown above. "
        "Reply to the following message:\n\n" + _message_text(last)
    )
    return system, prompt


class _SessionWindow:
    """Rate-limit status reported by the latest CLI query."""

    info: Any = None

    @classmethod
    def wait_if_needed(cls) -> None:
        info = cls.info
        if info is None:
            return
        window = (info.raw.get("unifiedWindows") or {}).get("five_hour") or {}
        used, resets_at = window.get("utilization"), window.get("resetsAt")
        if info.rate_limit_type == "five_hour":
            resets_at = resets_at or info.resets_at
            if info.status == "rejected":
                used = 1.0
            elif used is None:
                used = info.utilization
        if used is None or used < _SESSION_LIMIT or not resets_at:
            return
        wait = resets_at + _RESET_MARGIN - time.time()
        if wait > 0:
            resume = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(resets_at + _RESET_MARGIN)
            )
            print(
                f"  5-hour session window at {used:.0%} (limit {_SESSION_LIMIT:.0%}); "
                f"pausing until {resume}",
                flush=True,
            )
            time.sleep(wait)
        cls.info = None


class ClaudeCLIChat:
    """Minimal chat-model replacement for the parts of langchain's ``ChatAnthropic`` used
    by ``ChebiPrompt``: ``with_structured_output(schema)``.

    Also accumulates the token usage of all queries (``usage_totals``).
    """

    def __init__(self, model_name: str, effort: str | None = None):
        self.model_name = model_name
        self.effort = effort
        self.usage_totals: dict[str, int] = {}
        self.n_queries = 0

    def with_structured_output(self, schema: type[BaseModel], **kwargs) -> Runnable:
        def _invoke(prompt_input) -> BaseModel:
            if isinstance(prompt_input, PromptValue):
                messages = prompt_input.to_messages()
            else:
                messages = list(prompt_input)
            system, prompt = render_messages(messages)
            return self.structured_completion(system, prompt, schema)

        return RunnableLambda(_invoke)

    async def _run_query(self, system: str, prompt: str, schema: type[BaseModel]):
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            RateLimitEvent,
            ResultMessage,
            query,
        )

        _SessionWindow.info = None
        _drop_api_key_auth()
        options = ClaudeAgentOptions(
            model=self.model_name,
            system_prompt=system,
            tools=[],
            setting_sources=[],
            output_format={"type": "json_schema", "schema": schema.model_json_schema()},
            effort=self.effort,
            # skips the session title request the CLI otherwise sends to a Haiku model
            env={"CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1"},
            **({"cli_path": _CLI_PATH} if _CLI_PATH else {}),
        )
        result = None
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, RateLimitEvent):
                _SessionWindow.info = message.rate_limit_info
            elif isinstance(message, ResultMessage):
                result = message
        return result

    def _record_usage(self, result) -> dict[str, int]:
        tokens: dict[str, int] = {}
        for name, usage in (result.model_usage or {}).items():
            if not (usage.get("canonicalModel") or name).startswith(self.model_name):
                continue  # e.g. small side calls of the CLI on another model
            for key, field in [
                ("input", "inputTokens"),
                ("output", "outputTokens"),
                ("thinking", "thinkingTokens"),
                ("cache_write", "cacheCreationInputTokens"),
                ("cache_read", "cacheReadInputTokens"),
            ]:
                tokens[key] = tokens.get(key, 0) + (usage.get(field) or 0)
        for key, value in tokens.items():
            self.usage_totals[key] = self.usage_totals.get(key, 0) + value
        self.n_queries += 1
        return tokens

    def structured_completion(
        self, system: str, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        """One structured answer. Transient CLI errors are retried with backoff; the SDK
        itself re-prompts on schema mismatches."""
        from claude_agent_sdk import CLIConnectionError, ProcessError

        last_exc: BaseException | None = None
        attempt = 0
        while attempt < _MAX_RETRIES:
            _SessionWindow.wait_if_needed()
            attempt += 1
            backoff = min(30 * 2 ** (attempt - 1), 300)
            started = time.monotonic()
            try:
                result = asyncio.run(
                    asyncio.wait_for(
                        self._run_query(system, prompt, schema), timeout=_CLI_TIMEOUT
                    )
                )
            except asyncio.TimeoutError:
                last_exc = TimeoutError(
                    f"CLI query timed out after {_CLI_TIMEOUT:.0f}s"
                )
            except (CLIConnectionError, ProcessError) as e:
                last_exc = e
            else:
                if result is None:
                    last_exc = RuntimeError(
                        "Agent SDK query produced no result message"
                    )
                else:
                    tokens = self._record_usage(result)
                    print(
                        f"  [{self.model_name}] response in {time.monotonic() - started:.0f}s, "
                        f"tokens: {tokens}, total ({self.n_queries} queries): {self.usage_totals}",
                        flush=True,
                    )
                    info = _SessionWindow.info
                    if (
                        info is not None
                        and info.status == "rejected"
                        and not (
                            result.subtype == "success" and result.structured_output
                        )
                    ):
                        if info.rate_limit_type == "five_hour":
                            attempt -= 1  # waited out at the top of the loop
                            continue
                        raise UsageLimitReached(
                            f"{info.rate_limit_type} usage limit of the Claude subscription "
                            f"reached (resets at {info.resets_at})"
                        )
                    if result.stop_reason == "refusal":
                        raise ModelRefusal(
                            f"{self.model_name} declined this request (safety classifier)"
                        )
                    if result.subtype == "success" and result.structured_output:
                        return schema.model_validate(result.structured_output)
                    # no structured output despite the SDK's own re-prompting: terminal
                    raise ClaudeCLIError(
                        f"No valid structured output (subtype={result.subtype}, "
                        f"errors={result.errors}): "
                        f"{json.dumps(result.structured_output) if result.structured_output else result.result}"
                    )
            if attempt < _MAX_RETRIES:
                print(
                    f"  CLI error (attempt {attempt}/{_MAX_RETRIES}), retrying in {backoff}s: {last_exc}",
                    flush=True,
                )
                time.sleep(backoff)
        raise ClaudeCLIError(
            f"Claude CLI query failed after {_MAX_RETRIES} attempts: {last_exc}"
        )
