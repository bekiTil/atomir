"""Provider factories: turn a `{provider, config}` block into an implementation.

This is the seam that makes the product accessible to ANYONE, not just Groq +
Jina users. The engine calls `LLMFactory.create(settings.llm)` /
`EmbedderFactory.create(settings.embedder)` and gets back an `LLM` / `Embedder`
without ever naming a vendor. Adding a new provider = write one class with a
`from_config` classmethod, then add one line to the registry here. Zero engine
changes (DECISION #6).
"""

from __future__ import annotations

from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM

# The concrete provider classes are imported LAZILY inside the registry builders
# below. That keeps this module's top level free of `atomir.llm` /
# `atomir.embeddings`, avoiding a circular import when those packages are
# imported before `atomir.providers` is fully initialized.


class LLMFactory:
    @staticmethod
    def _registry() -> dict[str, type[LLM]]:
        from atomir.llm import FakeLLM, GroqLLM  # Groq is the FIRST impl, not the only one
        from atomir.llm.anthropic import AnthropicLLM
        from atomir.llm.azure_openai import AzureOpenAILLM
        from atomir.llm.gemini import GeminiLLM
        from atomir.llm.ollama import OllamaLLM
        from atomir.llm.openai import OpenAILLM

        return {
            "fake": FakeLLM,
            "groq": GroqLLM,
            "openai": OpenAILLM,
            "anthropic": AnthropicLLM,
            "gemini": GeminiLLM,
            "azure_openai": AzureOpenAILLM,
            "ollama": OllamaLLM,
        }

    @classmethod
    def create(cls, cfg: dict) -> LLM:
        registry = cls._registry()
        provider = cfg.get("provider")
        if provider not in registry:
            raise ValueError(
                f"Unknown LLM provider {provider!r}. "
                f"Valid providers: {sorted(registry)}"
            )
        return registry[provider].from_config(cfg.get("config", {}))


class EmbedderFactory:
    @staticmethod
    def _registry() -> dict[str, type[Embedder]]:
        from atomir.embeddings import FakeEmbedder, JinaEmbedder
        from atomir.embeddings.azure_openai import AzureOpenAIEmbedder
        from atomir.embeddings.gemini import GeminiEmbedder
        from atomir.embeddings.ollama import OllamaEmbedder
        from atomir.embeddings.openai import OpenAIEmbedder
        from atomir.embeddings.voyage import VoyageEmbedder

        return {
            "fake": FakeEmbedder,
            "jina": JinaEmbedder,
            "voyage": VoyageEmbedder,
            "openai": OpenAIEmbedder,
            "gemini": GeminiEmbedder,
            "azure_openai": AzureOpenAIEmbedder,
            "ollama": OllamaEmbedder,
        }

    @classmethod
    def create(cls, cfg: dict) -> Embedder:
        registry = cls._registry()
        provider = cfg.get("provider")
        if provider not in registry:
            raise ValueError(
                f"Unknown embedder provider {provider!r}. "
                f"Valid providers: {sorted(registry)}"
            )
        return registry[provider].from_config(cfg.get("config", {}))
