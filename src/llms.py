import database
import storage
from utils import EnumABCMeta

import gigachat.context
import gigachat.models
from gigachat import GigaChat as GigaChatModel
import numpy.typing as npt
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Self, Any
import enum


ClassLabel = np.int64
ExampleIndex = np.int64


@storage.Stringifiable.make_stringifiable()
class LLMType(storage.Stringifiable, enum.StrEnum, metaclass=EnumABCMeta):
    GIGACHAT = enum.auto()

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_str(s: str, db: 'database.DataDatabase') -> 'LLMType':
        for member in LLMType:
            if member.value == s:
                return member
        raise ValueError(f'Invalid LLMType string: {s}')


@dataclass(frozen=True)
class Chat(ABC):
    llm: 'LLM'
    inner_chat: Any
    session_id: str | None = None

    def query(self, prompt: str) -> str:
        return self.llm.query(self, prompt)

    @abstractmethod
    def add_system_message(self, msg: str) -> None: ...

    @abstractmethod
    def add_user_message(self, msg: str) -> None: ...

    @abstractmethod
    def add_assistant_message(self, msg: str) -> None: ...

    @classmethod
    def from_precached(cls, llm: 'LLM', inner_chat: Any, chat_id: str, system_prompt: str) -> Self:
        chat = cls(llm, inner_chat, chat_id)
        chat.add_system_message(system_prompt)
        return chat


class LLM(ABC):
    def query_examples(
        self, chat: Chat, prompt: str, extraction: Callable[[str], npt.NDArray[ExampleIndex]]
    ) -> npt.NDArray[ExampleIndex]:
        return extraction(self.query(chat, prompt))

    def annotate_examples(
        self, chat: Chat, prompt: str, extraction: Callable[[str], dict[ExampleIndex, ClassLabel]]
    ) -> dict[ExampleIndex, ClassLabel]:
        return extraction(self.query(chat, prompt))

    @abstractmethod
    def query(self, chat: Chat, prompt: str) -> str: ...

    @classmethod
    @abstractmethod
    def connect(cls: type[Self], api_key: str, *params: Any) -> Self: ...

    @classmethod
    @abstractmethod
    def get_type(cls) -> LLMType: ...

    @abstractmethod
    def make_chat(self, system_prompt: str, chat_id: str) -> Chat: ...


class GigachatChat(Chat):
    def add_system_message(self, msg: str) -> None:
        self.inner_chat.messages.append(gigachat.models.Messages(role=gigachat.models.MessagesRole.SYSTEM, content=msg))

    def add_user_message(self, msg: str) -> None:
        self.inner_chat.messages.append(gigachat.models.Messages(role=gigachat.models.MessagesRole.USER, content=msg))

    def add_assistant_message(self, msg: str) -> None:
        self.inner_chat.messages.append(
            gigachat.models.Messages(role=gigachat.models.MessagesRole.ASSISTANT, content=msg)
        )


class GigachatLLM(LLM):
    def __init__(self, client: GigaChatModel):
        self.client = client

    @classmethod
    def connect(cls, api_key: str) -> Self:
        return GigachatLLM(
            GigaChatModel(
                credentials=api_key,
                model='GigaChat',
                scope='GIGACHAT_API_PERS',
                verify_ssl_certs=False,
            ),
        )

    def make_chat(self, system_prompt: str, chat_id: str) -> GigachatChat:
        return GigachatChat.from_precached(
            self, gigachat.models.Chat(model='GigaChat', messages=[]), chat_id, system_prompt
        )

    def query(self, chat: Chat, prompt: str) -> str:
        if not ENABLED:
            raise RuntimeError('LLMs are disabled, to query them pass --allow-llm')
        token = None
        if chat.session_id is not None:
            token = gigachat.context.session_id_cvar.set(chat.session_id)

        chat.add_user_message(prompt)
        response = self.client.chat(chat.inner_chat)
        print(response.usage)
        result = response.choices[0].message.content
        chat.add_assistant_message(result)
        if token is not None:
            gigachat.context.session_id_cvar.reset(token)
        return result

    @classmethod
    def get_type(cls) -> LLMType:
        return LLMType.GIGACHAT


ENABLED: bool = False


def enable():
    global ENABLED
    ENABLED = True
