from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def slugify(text: str) -> str:
    sanitized = "".join(character.lower() if character.isalnum() else "_" for character in text.strip())
    sanitized = "_".join(part for part in sanitized.split("_") if part)
    return sanitized[:80] or "untitled"


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    llm_provider: str = "openai"
    chat_model: str = "gpt-4.1-mini"
    api_key: str | None = None
    base_url: str | None = None
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    temperature: float = 0.2
    rag_top_k: int = 4
    translation_chunk_size: int = 3500
    translation_chunk_overlap: int = 300
    rag_collection_name: str = "novel_translation"

    @classmethod
    def from_env(cls, project_root: str | Path | None = None) -> "AppConfig":
        root = Path(project_root or Path.cwd()).resolve()
        load_dotenv(root / ".env")

        return cls(
            project_root=root,
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
            api_key=(
                os.getenv("API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("OPENROUTER_API_KEY")
            ),
            base_url=(
                os.getenv("OLLAMA_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or None
            ),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "huggingface"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            rag_top_k=int(os.getenv("RAG_TOP_K", "4")),
            translation_chunk_size=int(os.getenv("TRANSLATION_CHUNK_SIZE", "3500")),
            translation_chunk_overlap=int(os.getenv("TRANSLATION_CHUNK_OVERLAP", "300")),
            rag_collection_name=os.getenv("RAG_COLLECTION_NAME", "novel_translation"),
        )

    @property
    def input_dir(self) -> Path:
        return self.project_root / "data" / "input"

    @property
    def output_root(self) -> Path:
        return self.project_root / "data" / "output"

    @property
    def rag_dir(self) -> Path:
        return self.project_root / "data" / "rag_db"

    @property
    def source_archive_dir(self) -> Path:
        return self.output_root / "source"

    @property
    def chapters_source_dir(self) -> Path:
        return self.output_root / "chapters" / "source"

    @property
    def translations_dir(self) -> Path:
        return self.output_root / "chapters" / "zh"

    @property
    def metadata_dir(self) -> Path:
        return self.output_root / "metadata"

    @property
    def outline_path(self) -> Path:
        return self.metadata_dir / "outline.md"

    @property
    def characters_path(self) -> Path:
        return self.metadata_dir / "characters.md"

    @property
    def original_copy_path(self) -> Path:
        return self.source_archive_dir / "novel_original.txt"

    @property
    def default_input_file(self) -> Path:
        return self.input_dir / "novel.txt"

    def chapter_source_path(self, chapter_number: int, title: str) -> Path:
        return self.chapters_source_dir / f"chapter_{chapter_number:03d}_{slugify(title)}.txt"

    def chapter_translation_path(self, chapter_number: int, title: str) -> Path:
        return self.translations_dir / f"chapter_{chapter_number:03d}_{slugify(title)}_zh.md"

    def ensure_directories(self) -> None:
        for path in (
            self.input_dir,
            self.output_root,
            self.rag_dir,
            self.source_archive_dir,
            self.chapters_source_dir,
            self.translations_dir,
            self.metadata_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
