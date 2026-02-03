from abc import abstractmethod
from collections.abc import Generator

import ollama
import openai

from .api import ChatMessages
from .tracing import tracer


class LLMClient:
    def __init__(self):
        pass

    @abstractmethod
    def streaming_chat(
        self,
        chat_messages: ChatMessages,
        model: str,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def chat(
        self,
        chat_messages: ChatMessages,
        model: str,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> str:
        pass

    @abstractmethod
    def embed(
        self,
        *args,
        **kwargs,
    ) -> list[float]:
        pass


class OllamaClient(LLMClient):
    def __init__(self, base_url: str):
        self.url = base_url
        self.client = ollama.Client(host=base_url)

    def _get_options(self, temperature: float, seed: int | None) -> dict:
        options = {
            "temperature": temperature,
        }
        if seed is not None:
            options["seed"] = seed
        return options

    def streaming_chat(
        self,
        chat_messages: ChatMessages,
        model: str,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> Generator[str, None, None]:
        response = self.client.chat(
            model=model,
            messages=chat_messages.messages,
            stream=True,
            options=self._get_options(temperature, seed),
        )

        for chunk in response:
            if chunk["message"]["content"] is not None:
                yield chunk["message"]["content"]

    def chat(
        self,
        chat_messages: ChatMessages,
        model: str,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> str:
        response = self.client.chat(
            model=model,
            messages=chat_messages.messages,
            stream=False,
            options=self._get_options(temperature, seed),
        )
        return response["message"]["content"]

    def embed(self, *args, **kwargs):
        return self.client.embed(*args, **kwargs)


# TODO: Consider responses API
class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, base_url: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def _get_tracing_metadata(
        self, chat_messages: ChatMessages | None = None, **extra_metadata
    ) -> dict:
        metadata = {
            "metadata": {
                "existing_trace_id": tracer.get_current_trace_id(),
                "parent_observation_id": tracer.get_current_observation_id(),
                "trace_release": tracer.release,
                **extra_metadata,
            }
        }
        if chat_messages is not None and chat_messages.prompt is not None:
            metadata["metadata"]["prompt"] = chat_messages.prompt_to_metadata_dict()
        return metadata

    def streaming_chat(
        self,
        chat_messages: ChatMessages,
        model: str,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> Generator[str, None, None]:
        response = self.client.chat.completions.create(
            model=model,
            messages=chat_messages.messages,
            stream=True,
            temperature=temperature,
            seed=seed,
            extra_body=self._get_tracing_metadata(
                generation_name="streaming_chat", chat_messages=chat_messages
            ),
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def chat(
        self,
        chat_messages: ChatMessages,
        model: str,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=chat_messages.messages,
            stream=False,
            temperature=temperature,
            seed=seed,
            extra_body=self._get_tracing_metadata(
                generation_name="chat", chat_messages=chat_messages
            ),
        )
        return response.choices[0].message.content

    def embed(self, *args, **kwargs):
        response = self.client.embeddings.create(
            *args,
            **kwargs,
            encoding_format=None,  # specific to ollama
            extra_body=self._get_tracing_metadata(generation_name="embed"),
        )
        return {"embeddings": [data.embedding for data in response.data]}
