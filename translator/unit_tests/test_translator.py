from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_translator.config import AppConfig
from novel_translator.splitter import Chapter
from novel_translator.translator import ChapterTranslator


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, _):
        if self.responses:
            return self.responses.pop(0)
        return _FakeResponse("兜底段")


class _FakeRAG:
    def build_translation_context(self, _):
        return {
            "outline": "RAG_OUTLINE",
            "characters": "RAG_CHARACTERS",
            "related_source": "RAG_SOURCE",
            "previous_chapter": "RAG_PREV",
        }


class TestTranslator(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            project_root=PROJECT_ROOT,
            translation_chunk_size=80,
            translation_chunk_overlap=10,
        )

    def test_translate_returns_normalized_text(self) -> None:
        chapter = Chapter(
            number=1,
            title="Chapter 1",
            content=("This is a long sentence. " * 20).strip(),
        )
        llm = _FakeLLM([
            _FakeResponse("```markdown\n第一段\n```"),
            _FakeResponse("第二段"),
            _FakeResponse("第三段"),
            _FakeResponse("第四段"),
        ])
        translator = ChapterTranslator(self.config, llm, _FakeRAG())

        output = translator.translate(chapter, outline="O", characters="C")

        self.assertIn("第一段", output)
        self.assertNotIn("```", output)

    def test_message_to_text_handles_list(self) -> None:
        text = ChapterTranslator._message_to_text(_FakeResponse(["x", "y"]))
        self.assertEqual(text, "x\ny")


if __name__ == "__main__":
    unittest.main()
