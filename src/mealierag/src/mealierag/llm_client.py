from collections.abc import Generator

import ollama


class LLMClient:
    def __init__(self):
        pass


class OllamaClient(LLMClient):
    def __init__(self, url: str):
        self.url = url
        self.client = ollama.Client(host=url)

    def streaming_chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        seed: int = None,
    ) -> Generator[str, None, None]:
        options = {
            "temperature": temperature,
        }
        if seed is not None:
            options["seed"] = seed
        response = self.client.chat(
            model=model,
            messages=messages,
            stream=True,
            options=options,
        )

        return response

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        seed: int = None,
    ) -> str:
        options = {
            "temperature": temperature,
        }
        if seed is not None:
            options["seed"] = seed
        response = self.client.chat(
            model=model,
            messages=messages,
            stream=False,
            options=options,
        )
        return response["message"]["content"]

    def embed(self, *args, **kwargs):
        return self.client.embed(*args, **kwargs)
