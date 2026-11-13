"""
Prompt template registry.

A ``PromptTemplate`` turns a ``CaptionRequest`` into OpenAI-style chat
messages. Each image is an ``{"type": "image_url", "image_url": {"url":
IMAGE_PLACEHOLDER}}`` part; the backend replaces placeholders with the
request's images, in order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

from .schema import CaptionRequest

Message = Dict[str, Any]
ChatMessages = List[Message]

IMAGE_PLACEHOLDER = "<image>"


class PromptTemplate(ABC):
    """Base class for prompt templates."""

    @abstractmethod
    def build_messages(self, request: CaptionRequest) -> ChatMessages:
        """Convert a CaptionRequest into a list of chat messages."""
        ...

    def __call__(self, request: CaptionRequest) -> ChatMessages:
        return self.build_messages(request)


class PromptRegistry:
    """Global registry of named prompt templates.

    Usage::

        @PromptRegistry.register("my_task")
        class MyPrompt(PromptTemplate):
            ...

        msgs = PromptRegistry.get("my_task")(request)
    """

    _registry: Dict[str, PromptTemplate] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator: register a PromptTemplate class under *name*."""
        def decorator(template_cls: Type[PromptTemplate]):
            cls._registry[name] = template_cls()
            return template_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> PromptTemplate:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "(none)"
            raise KeyError(
                f"Prompt template '{name}' not found. Available: {available}"
            )
        return cls._registry[name]
