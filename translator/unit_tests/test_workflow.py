from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_translator.config import AppConfig
from novel_translator.splitter import Chapter
from novel_translator.workflow import NovelTranslationWorkflow


class TestWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config = AppConfig(
            project_root=self.root,
            llm_provider="ollama",
            chat_model="kimi-k2.5:cloud",
            embedding_provider="simple",
            embedding_model="dummy",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _build_workflow(self) -> NovelTranslationWorkflow:
        with patch("novel_translator.workflow.build_chat_model", return_value=MagicMock()):
            with patch("novel_translator.workflow.NovelRAG", return_value=MagicMock()):
                with patch("novel_translator.workflow.StoryMetadataExtractor", return_value=MagicMock()):
                    with patch("novel_translator.workflow.ChapterTranslator", return_value=MagicMock()):
                        return NovelTranslationWorkflow(self.config)

    def test_prepare_calls_pipeline_and_returns_chapters(self) -> None:
        workflow = self._build_workflow()
        input_file = self.root / "data" / "input" / "novel.txt"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        input_file.write_text("source", encoding="utf-8")
        chapters = [
            Chapter(number=1, title="Chapter 1", content="A"),
            Chapter(number=2, title="Chapter 2", content="B"),
        ]
        workflow.extractor.extract.return_value = ("outline", "characters")

        with patch("novel_translator.workflow.load_text", return_value="full text"):
            with patch("novel_translator.workflow.copy_text") as mock_copy:
                with patch("novel_translator.workflow.split_novel_into_chapters", return_value=chapters):
                    with patch("novel_translator.workflow.save_chapters") as mock_save_chapters:
                        with patch("novel_translator.workflow.save_text") as mock_save_text:
                            result = workflow.prepare("data/input/novel.txt", max_chapters=1)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].number, 1)
        mock_copy.assert_called_once()
        mock_save_chapters.assert_called_once()
        self.assertEqual(mock_save_text.call_count, 2)
        workflow.rag.reset_store.assert_called_once()
        workflow.rag.index_full_novel.assert_called_once()
        workflow.rag.index_chapters.assert_called_once()
        workflow.rag.index_story_metadata.assert_called_once()

    def test_translate_skips_existing_when_not_force(self) -> None:
        workflow = self._build_workflow()
        chapter = Chapter(number=1, title="Chapter 1", content="content")
        target = self.config.chapter_translation_path(chapter.number, chapter.title)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("existing", encoding="utf-8")

        with patch.object(workflow, "_ensure_prepared", return_value=[chapter]):
            with patch("novel_translator.workflow.load_text", side_effect=["outline", "characters"]):
                with patch("novel_translator.workflow.save_text") as mock_save_text:
                    outputs = workflow.translate(force=False)

        self.assertEqual(outputs, [])
        mock_save_text.assert_not_called()
        workflow.translator.translate.assert_not_called()
        workflow.rag.index_translation.assert_not_called()

    def test_translate_writes_output_when_force_true(self) -> None:
        workflow = self._build_workflow()
        chapter = Chapter(number=1, title="Chapter 1", content="content")
        workflow.translator.translate.return_value = "译文"

        with patch.object(workflow, "_ensure_prepared", return_value=[chapter]):
            with patch("novel_translator.workflow.load_text", side_effect=["outline", "characters"]):
                with patch("novel_translator.workflow.save_text") as mock_save_text:
                    outputs = workflow.translate(force=True)

        self.assertEqual(len(outputs), 1)
        mock_save_text.assert_called_once()
        workflow.rag.index_translation.assert_called_once_with(chapter, "译文")

    def test_resolve_input_path_raises_for_missing_file(self) -> None:
        workflow = self._build_workflow()
        with self.assertRaises(FileNotFoundError):
            workflow._resolve_input_path("data/input/not_exists.txt")


if __name__ == "__main__":
    unittest.main()
