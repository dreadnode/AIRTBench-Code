import re
import typing as t
from dataclasses import dataclass

import litellm

Message = dict[str, str]


class XmlAction:
    tag: t.ClassVar[str]

    @classmethod
    def xml_start_tag(cls) -> str:
        return f"<{cls.tag}>"

    @classmethod
    def xml_end_tag(cls) -> str:
        return f"</{cls.tag}>"

    @classmethod
    def xml_example(cls) -> str:
        return f"<{cls.tag}>\n{cls.example_content()}\n</{cls.tag}>"

    @classmethod
    def example_content(cls) -> str:
        return ""


@dataclass(frozen=True)
class ExecuteCode(XmlAction):
    tag: t.ClassVar[str] = "execute_code"
    code: str

    @classmethod
    def example_content(cls) -> str:
        return "print('Hello world')"


@dataclass(frozen=True)
class RestartKernel(XmlAction):
    tag: t.ClassVar[str] = "restart_kernel"
    not_used: str = ""

    @classmethod
    def example_content(cls) -> str:
        return "restart"


@dataclass(frozen=True)
class GiveUp(XmlAction):
    tag: t.ClassVar[str] = "give_up"
    summary: str

    @classmethod
    def example_content(cls) -> str:
        return "Summarize why the challenge cannot be completed."


@dataclass
class ChatSession:
    messages: list[Message]

    def add_user(self, content: str) -> "ChatSession":
        self.messages.append({"role": "user", "content": content})
        return self

    def add_assistant(self, content: str) -> "ChatSession":
        self.messages.append({"role": "assistant", "content": content})
        return self

    def drop_last_message(self) -> None:
        if self.messages:
            self.messages.pop()


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    stop_reason: str | None = None
    usage: Usage | None = None


def _parse_tag(content: str, tag: str) -> list[str]:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL)
    return [match.group(1).strip() for match in pattern.finditer(content)]


def parse_execute_code(content: str) -> list[ExecuteCode]:
    return [ExecuteCode(code=code) for code in _parse_tag(content, ExecuteCode.tag)]


def parse_restart_kernel(content: str) -> RestartKernel | None:
    matches = _parse_tag(content, RestartKernel.tag)
    return RestartKernel(matches[0]) if matches else None


def parse_give_up(content: str) -> GiveUp | None:
    matches = _parse_tag(content, GiveUp.tag)
    return GiveUp(matches[0]) if matches else None


async def complete_chat(
    *,
    model: str,
    messages: list[Message],
    timeout: int,
    stop: list[str] | None = None,
) -> ChatCompletion:
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        timeout=timeout,
        stop=stop,
    )
    choice = response.choices[0]
    content = choice.message.content or ""
    usage = response.usage
    return ChatCompletion(
        content=content,
        stop_reason=choice.finish_reason,
        usage=Usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )
        if usage
        else None,
    )
