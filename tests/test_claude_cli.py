"""Tests for the Claude CLI (subscription) backend, without calling the CLI."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from nl_2_fol.inference.learner.custom_exceptions import StopProgramException
from nl_2_fol.inference.learner.learn_definitions import _caused_by_refusal
from nl_2_fol.prompting import claude_cli
from nl_2_fol.prompting.claude_cli import (
    ClaudeCLIChat,
    ModelRefusal,
    UsageLimitReached,
    render_messages,
)
from nl_2_fol.prompting.prompt_models import ChEBI_FOL_AT


def test_render_single_message():
    system, prompt = render_messages(
        [SystemMessage(content="sys"), HumanMessage(content="define diol")]
    )
    assert system == "sys"
    assert prompt == "define diol"


def test_render_conversation():
    system, prompt = render_messages(
        [
            SystemMessage(content="sys"),
            HumanMessage(content="example input"),
            AIMessage(content='{"FOL_formula": "x"}'),
            HumanMessage(content="define diol"),
        ]
    )
    assert system == "sys"
    assert "<user>\nexample input\n</user>" in prompt
    assert '<assistant>\n{"FOL_formula": "x"}\n</assistant>' in prompt
    assert prompt.endswith("define diol")
    assert prompt.index("example input") < prompt.index("define diol")


def test_render_requires_final_user_message():
    with pytest.raises(ValueError):
        render_messages([SystemMessage(content="sys"), AIMessage(content="x")])


def _result(**kwargs):
    defaults = dict(
        subtype="success",
        structured_output={"FOL_formula": "diol(M) :- polyol(M)."},
        stop_reason="end_turn",
        result="",
        errors=None,
        model_usage={
            "claude-haiku-5-5": {"inputTokens": 100, "outputTokens": 20},
            "claude-haiku-4-5": {"inputTokens": 5, "outputTokens": 1},  # side call
        },
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture
def chat(monkeypatch):
    monkeypatch.setattr(claude_cli._SessionWindow, "info", None)
    return ClaudeCLIChat("claude-haiku-5-5", effort="high")


def _fake_query(monkeypatch, result, rate_limit_info=None):
    async def _run_query(self, system, prompt, schema):
        self.last_call = (system, prompt, schema)
        claude_cli._SessionWindow.info = rate_limit_info
        return result

    monkeypatch.setattr(ClaudeCLIChat, "_run_query", _run_query)


def test_structured_output_through_prompt_template(monkeypatch, chat):
    _fake_query(monkeypatch, _result())
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="sys"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    chain = template | chat.with_structured_output(ChEBI_FOL_AT)
    output = chain.invoke(
        {
            "input": "define diol",
            "history": [HumanMessage(content="q"), AIMessage(content="a")],
        }
    )
    assert isinstance(output, ChEBI_FOL_AT)
    assert output.FOL_formula == "diol(M) :- polyol(M)."
    system, prompt, schema = chat.last_call
    assert system == "sys" and prompt.endswith("define diol") and schema is ChEBI_FOL_AT
    # only tokens of the requested model are counted
    assert chat.usage_totals["input"] == 100
    assert chat.usage_totals["output"] == 20


def test_refusal_is_not_retried(monkeypatch, chat):
    _fake_query(monkeypatch, _result(stop_reason="refusal", structured_output=None))
    with pytest.raises(ModelRefusal):
        chat.structured_completion("sys", "prompt", ChEBI_FOL_AT)
    assert chat.n_queries == 1


def test_weekly_limit_raises(monkeypatch, chat):
    info = SimpleNamespace(
        status="rejected",
        rate_limit_type="seven_day",
        resets_at=0,
        raw={},
        utilization=1.0,
    )
    _fake_query(
        monkeypatch,
        _result(subtype="error", structured_output=None),
        rate_limit_info=info,
    )
    with pytest.raises(UsageLimitReached):
        chat.structured_completion("sys", "prompt", ChEBI_FOL_AT)


def test_api_key_is_removed_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    claude_cli._drop_api_key_auth()
    import os

    assert "ANTHROPIC_API_KEY" not in os.environ


def test_caused_by_refusal():
    try:
        try:
            raise ModelRefusal("declined")
        except ModelRefusal as e:
            raise StopProgramException(str(e)) from e
    except StopProgramException as wrapped:
        assert _caused_by_refusal(wrapped)
    assert not _caused_by_refusal(StopProgramException("other"))
