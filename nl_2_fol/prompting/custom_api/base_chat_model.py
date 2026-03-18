"""
Refer Langchain Docs:
https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel
"""

from operator import itemgetter
from typing import Any, List, Union, get_args, get_origin

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import LLM, BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.utils.json import parse_json_markdown
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel


class _LocalModelPydanticOutputParser(BaseOutputParser):
    """Output parser for local models that may output raw text instead of JSON.

    First tries JSON / JSON-markdown parsing.  On failure, uses schema
    introspection to build a Pydantic instance with sensible defaults, placing
    the raw model output in the ``FOL_formula`` field when it exists.
    """

    pydantic_object: (
        Any  # type[BaseModel] — stored without pydantic structural validation
    )

    @property
    def _type(self) -> str:
        return "local_model_pydantic_output_parser"

    def parse(self, text: str) -> BaseModel:
        text = text.strip()

        # 1. Try standard JSON / fenced-code-block JSON parsing first.
        try:
            data = parse_json_markdown(text)
            return self.pydantic_object.model_validate(data)
        except Exception:
            pass

        # 2. Fallback: build a minimal valid object from the raw text.
        try:
            return self._build_from_raw_text(self.pydantic_object, text, is_root=True)
        except Exception as exc:
            raise OutputParserException(
                f"Cannot parse output for schema "
                f"{self.pydantic_object.__name__}: {text}"
            ) from exc

    def _resolve_annotation(self, annotation: Any) -> Any:
        """Unwrap Optional[X] / Union[X, None] to the inner type X."""
        origin = get_origin(annotation)
        if origin is Union:
            non_none = [a for a in get_args(annotation) if a is not type(None)]
            return non_none[0] if non_none else str
        return annotation

    def _build_from_raw_text(
        self, cls: type[BaseModel], text: str, *, is_root: bool
    ) -> BaseModel:
        """Recursively construct *cls* from *text*.

        * ``FOL_formula`` at the root level receives the full raw text.
        * Nested ``BaseModel`` fields are filled recursively.
        * ``str`` fields default to ``""``.
        * ``dict`` / ``list`` fields default to ``{}`` / ``[]``.
        * Everything else defaults to ``None``.
        """
        result: dict[str, Any] = {}
        for name, field_info in cls.model_fields.items():
            if name == "FOL_formula" and is_root:
                result[name] = text
                continue

            annotation = self._resolve_annotation(field_info.annotation)
            origin = get_origin(annotation)

            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                result[name] = self._build_from_raw_text(
                    annotation, text, is_root=False
                )
            elif annotation is str:
                result[name] = ""
            elif origin is dict or annotation is dict:
                result[name] = {}
            elif origin is list or annotation is list:
                result[name] = []
            else:
                result[name] = None

        return cls.model_validate(result)


class _TestOutputModel(BaseModel):
    """Internal test model for `_LocalModelPydanticOutputParser`.

    This mirrors the key patterns the parser is expected to handle:
    - a top-level `FOL_formula` field that should receive raw text on fallback
    - a string field
    - a list field
    - a dict field
    """

    FOL_formula: str | None = None
    explanation: str = ""
    items: list[str] = []
    metadata: dict[str, Any] = {}


def test_local_model_pydantic_output_parser_parses_json_and_markdown() -> None:
    """Ensure valid JSON and markdown JSON are parsed into the Pydantic model."""
    parser = _LocalModelPydanticOutputParser(pydantic_object=_TestOutputModel)

    json_payload = (
        '{"FOL_formula": "P(x)", '
        '"explanation": "parsed", '
        '"items": ["a", "b"], '
        '"metadata": {"k": "v"}}'
    )

    # Plain JSON
    obj = parser.parse(json_payload)
    assert isinstance(obj, _TestOutputModel)
    assert obj.FOL_formula == "P(x)"
    assert obj.explanation == "parsed"
    assert obj.items == ["a", "b"]
    assert obj.metadata == {"k": "v"}

    # Fenced markdown JSON
    markdown_payload = f"```json\n{json_payload}\n```"
    obj_markdown = parser.parse(markdown_payload)
    assert isinstance(obj_markdown, _TestOutputModel)
    assert obj_markdown.FOL_formula == "P(x)"
    assert obj_markdown.explanation == "parsed"
    assert obj_markdown.items == ["a", "b"]
    assert obj_markdown.metadata == {"k": "v"}


def test_local_model_pydantic_output_parser_fallback_populates_fol_formula() -> None:
    """Ensure fallback from raw text produces a valid object with `FOL_formula` set."""
    parser = _LocalModelPydanticOutputParser(pydantic_object=_TestOutputModel)

    raw_text = "forall x. P(x) -> Q(x)"
    obj = parser.parse(raw_text)

    # Fallback should populate FOL_formula with the raw text.
    assert isinstance(obj, _TestOutputModel)
    assert obj.FOL_formula == raw_text

    # Other fields should follow the documented defaults from `_build_from_raw_text`.
    assert obj.explanation == ""
    assert obj.items == []
    assert obj.metadata == {}


class LocalModelChat(BaseChatModel):
    llm: LLM

    @property
    def _llm_type(self) -> str:
        return self.llm._llm_type

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        _ = kwargs.pop("method", None)
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)

        if isinstance(schema, type) and is_basemodel_subclass(schema):
            output_parser = _LocalModelPydanticOutputParser(pydantic_object=schema)
        else:
            output_parser = JsonOutputParser()

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser,
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=self) | parser_with_fallback

        return self | output_parser

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate the result.

        Args:
            messages: The messages to generate from.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager to use for this call.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            The chat result.
        """
        prompt = self._convert_messages_to_prompt(messages)

        response = self.llm._call(prompt, stop=stop, run_manager=run_manager, **kwargs)

        message = AIMessage(content=response)

        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        parts = []
        for m in messages:
            parts.append(f"{m.type}: {m.content}")
        return "\n".join(parts)
