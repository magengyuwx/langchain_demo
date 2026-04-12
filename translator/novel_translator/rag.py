from __future__ import annotations

from typing import Sequence
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import AppConfig
from .llm_factory import build_embeddings
from .splitter import Chapter


class NovelRAG:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embeddings = build_embeddings(config)
        self.vectorstore = self._create_vectorstore()

    def _create_vectorstore(self) -> Chroma:
        return Chroma(
            collection_name=self.config.rag_collection_name,
            persist_directory=str(self.config.rag_dir),
            embedding_function=self.embeddings,
        )

    def reset_store(self) -> None:
        try:
            self.vectorstore.delete_collection()
        except Exception:
            pass
        self.vectorstore = self._create_vectorstore()

    def index_full_novel(self, full_text: str) -> None:
        documents = self._chunk_text(
            full_text,
            metadata={"scope": "novel_full"},
            chunk_size=1800,
            chunk_overlap=200,
        )
        self._add_documents(documents)

    def index_chapters(self, chapters: Sequence[Chapter]) -> None:
        documents: list[Document] = []
        for chapter in chapters:
            documents.extend(
                self._chunk_text(
                    chapter.content,
                    metadata={
                        "scope": "chapter_source",
                        "chapter_number": chapter.number,
                        "chapter_title": chapter.title,
                    },
                    chunk_size=1400,
                    chunk_overlap=150,
                )
            )
        self._add_documents(documents)

    def index_story_metadata(self, outline: str, characters: str) -> None:
        documents = [
            *self._chunk_text(
                outline,
                metadata={"scope": "outline"},
                chunk_size=1800,
                chunk_overlap=100,
            ),
            *self._chunk_text(
                characters,
                metadata={"scope": "characters"},
                chunk_size=1800,
                chunk_overlap=100,
            ),
        ]
        self._add_documents(documents)

    def index_translation(self, chapter: Chapter, translated_text: str) -> None:
        documents = self._chunk_text(
            translated_text,
            metadata={
                "scope": "chapter_translation",
                "chapter_number": chapter.number,
                "chapter_title": chapter.title,
                "language": "zh",
            },
            chunk_size=1400,
            chunk_overlap=150,
        )
        self._add_documents(documents)

    def search(
        self,
        query: str,
        *,
        scope: str | None = None,
        chapter_number: int | None = None,
        k: int | None = None,
    ) -> list[Document]:
        filter_payload: dict[str, object] = {}
        if scope:
            filter_payload["scope"] = scope
        if chapter_number is not None:
            filter_payload["chapter_number"] = chapter_number

        return self.vectorstore.similarity_search(
            query,
            k=k or self.config.rag_top_k,
            filter=filter_payload or None,
        )

    def build_translation_context(self, chapter: Chapter) -> dict[str, str]:
        query = f"{chapter.title}\n\n{chapter.content[:1500]}"
        outline_docs = self.search("小说主线 剧情 大纲", scope="outline", k=1)
        character_docs = self.search("人物 设定 关系 称谓", scope="characters", k=1)
        source_docs = self.search(
            query,
            scope="chapter_source",
            k=max(2, self.config.rag_top_k // 2),
        )
        previous_docs = (
            self.search(
                chapter.title,
                scope="chapter_translation",
                chapter_number=chapter.number - 1,
                k=1,
            )
            if chapter.number > 1
            else []
        )

        return {
            "outline": self._format_documents(outline_docs),
            "characters": self._format_documents(character_docs),
            "related_source": self._format_documents(source_docs),
            "previous_chapter": self._format_documents(previous_docs),
        }

    def _chunk_text(
        self,
        text: str,
        *,
        metadata: dict[str, object],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", ". ", " "],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_text(text)
        return [
            Document(
                page_content=chunk,
                metadata={**metadata, "chunk_index": index},
            )
            for index, chunk in enumerate(chunks, start=1)
            if chunk.strip()
        ]

    def _add_documents(self, documents: Sequence[Document]) -> None:
        if not documents:
            return
        self.vectorstore.add_documents(
            list(documents),
            ids=[str(uuid4()) for _ in documents],
        )

    @staticmethod
    def _format_documents(documents: Sequence[Document]) -> str:
        if not documents:
            return ""
        return "\n\n".join(document.page_content.strip() for document in documents if document.page_content.strip())
