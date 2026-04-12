from __future__ import annotations

import math

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import AppConfig


class SimpleHashEmbeddings:
    def __init__(self, size: int = 256) -> None:
        self.size = size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        tokens = text.lower().split()
        for token in tokens:
            vector[hash(token) % self.size] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def build_chat_model(config: AppConfig):
    provider = config.llm_provider.strip().lower()
    base_url = config.base_url

    if provider == "openrouter" and not base_url:
        base_url = "https://openrouter.ai/api/v1"

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.chat_model,
            base_url=config.base_url or "http://localhost:11434",
            temperature=config.temperature,
            reasoning=False,
        )

    if provider in {"openai", "openrouter", "custom"}:
        if not config.api_key:
            raise ValueError("未找到 API_KEY，请在 .env 中配置 API_KEY 或 OPENAI_API_KEY。")
        return ChatOpenAI(
            model=config.chat_model,
            api_key=config.api_key,
            base_url=base_url,
            temperature=config.temperature,
        )

    raise ValueError(f"不支持的 LLM_PROVIDER: {config.llm_provider}")


def build_embeddings(config: AppConfig):
    provider = config.embedding_provider.strip().lower()

    if provider in {"simple", "local", "hash"}:
        return SimpleHashEmbeddings()

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.base_url or "http://localhost:11434",
        )

    if provider in {"openai", "custom"}:
        if not config.api_key:
            raise ValueError("未找到 Embedding 所需 API_KEY，请在 .env 中补充配置。")
        return OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    raise ValueError(
        f"不支持的 EMBEDDING_PROVIDER: {config.embedding_provider}，"
        "当前仅支持 simple / huggingface / ollama / openai / custom。"
    )
