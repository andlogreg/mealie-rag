from dataclasses import dataclass
from typing import Any

from langfuse.model import Prompt_Chat


@dataclass
class ChatMessages:
    messages: list[dict[str, str]]
    prompt: Any | None = None

    def prompt_to_metadata_dict(self) -> dict[str, Any]:
        """Convert prompt to metadata dictionary"""
        if self.prompt is None:
            return {}
        prompt = Prompt_Chat(
            prompt=[
                {
                    "role": message["role"],
                    "content": message["content"],
                    "type": "chatmessage",
                }
                for message in self.prompt.prompt
            ],
            name=self.prompt.name,
            version=self.prompt.version,
            config=self.prompt.config,
            labels=self.prompt.labels,
            tags=self.prompt.tags,
            commit_message=self.prompt.commit_message,
        )
        return prompt.dict()

    @property
    def messages_count(self) -> int:
        return len(self.messages)
