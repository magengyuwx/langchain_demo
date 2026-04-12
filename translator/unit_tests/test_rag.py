from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from novel_translator.config import AppConfig
from novel_translator.rag import NovelRAG
from novel_translator.splitter import Chapter


class _FakeVectorStore:
    def __init__(self) -> None:
        self.deleted = False
        self.added_docs: list[Document] = []
        self.last_search_args: tuple | None = None

    def delete_collection(self) -> None:
        self.deleted = True

    def add_documents(self, documents, ids):
        self.added_docs.extend(documents)
        self.last_ids = ids

    def similarity_search(self, query, k=4, filter=None):
        self.last_search_args = (query, k, filter)
        return [Document(page_content="hit", metadata={"scope": "x"})]


class TestNovelRAG(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config = AppConfig(
            project_root=root,
            embedding_provider="simple",
            embedding_model="dummy",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_search_builds_filter_payload(self) -> None:
        fake_store = _FakeVectorStore()
        with patch("novel_translator.rag.build_embeddings", return_value=MagicMock()):
            with patch.object(NovelRAG, "_create_vectorstore", return_value=fake_store):
                rag = NovelRAG(self.config)

        docs = rag.search("query", scope="outline", chapter_number=2, k=3)

        self.assertEqual(len(docs), 1)
        self.assertEqual(fake_store.last_search_args, ("query", 3, {"scope": "outline", "chapter_number": 2}))

    def test_reset_store_recreates_vectorstore(self) -> None:
        store1 = _FakeVectorStore()
        store2 = _FakeVectorStore()
        with patch("novel_translator.rag.build_embeddings", return_value=MagicMock()):
            with patch.object(NovelRAG, "_create_vectorstore", side_effect=[store1, store2]):
                rag = NovelRAG(self.config)
                rag.reset_store()

        self.assertTrue(store1.deleted)
        self.assertIs(rag.vectorstore, store2)

    def test_build_translation_context_formats_documents(self) -> None:
        with patch("novel_translator.rag.build_embeddings", return_value=MagicMock()):
            with patch.object(NovelRAG, "_create_vectorstore", return_value=_FakeVectorStore()):
                rag = NovelRAG(self.config)

        chapter = Chapter(number=2, title="Chapter 2", content="content")

        def fake_search(query, *, scope=None, chapter_number=None, k=None):
            if scope == "outline":
                return [Document(page_content="OUTLINE", metadata={})]
            if scope == "characters":
                return [Document(page_content="CHARACTERS", metadata={})]
            if scope == "chapter_source":
                return [Document(page_content="SOURCE", metadata={})]
            if scope == "chapter_translation":
                return [Document(page_content="PREV", metadata={})]
            return []

        with patch.object(rag, "search", side_effect=fake_search):
            context = rag.build_translation_context(chapter)

        self.assertEqual(context["outline"], "OUTLINE")
        self.assertEqual(context["characters"], "CHARACTERS")
        self.assertEqual(context["related_source"], "SOURCE")
        self.assertEqual(context["previous_chapter"], "PREV")

    def test_index_chapters_pushes_documents(self) -> None:
        fake_store = _FakeVectorStore()
        with patch("novel_translator.rag.build_embeddings", return_value=MagicMock()):
            with patch.object(NovelRAG, "_create_vectorstore", return_value=fake_store):
                rag = NovelRAG(self.config)

        chapters = [Chapter(number=1, title="Chapter 1", content="hello world " * 60)]
        rag.index_chapters(chapters)

        self.assertGreaterEqual(len(fake_store.added_docs), 1)
        self.assertTrue(all(doc.metadata.get("scope") == "chapter_source" for doc in fake_store.added_docs))


if __name__ == "__main__":
    unittest.main()
