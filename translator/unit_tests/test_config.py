from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_translator.config import AppConfig, slugify


class TestConfig(unittest.TestCase):
    def test_slugify_basic(self) -> None:
        self.assertEqual(slugify("Chapter 1: Hello, World!"), "chapter_1_hello_world")
        self.assertEqual(slugify("   "), "untitled")

    def test_from_env_reads_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env = {
                "LLM_PROVIDER": "ollama",
                "CHAT_MODEL": "kimi-k2.5:cloud",
                "OLLAMA_BASE_URL": "http://localhost:11434",
                "EMBEDDING_PROVIDER": "simple",
                "EMBEDDING_MODEL": "dummy-embed",
                "TEMPERATURE": "0.3",
                "RAG_TOP_K": "6",
            }
            with patch.dict(os.environ, env, clear=False):
                config = AppConfig.from_env(root)

            self.assertEqual(config.project_root, root.resolve())
            self.assertEqual(config.llm_provider, "ollama")
            self.assertEqual(config.chat_model, "kimi-k2.5:cloud")
            self.assertEqual(config.base_url, "http://localhost:11434")
            self.assertEqual(config.embedding_provider, "simple")
            self.assertEqual(config.embedding_model, "dummy-embed")
            self.assertAlmostEqual(config.temperature, 0.3)
            self.assertEqual(config.rag_top_k, 6)

    def test_paths_and_ensure_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = AppConfig(project_root=root)
            config.ensure_directories()

            self.assertTrue(config.input_dir.exists())
            self.assertTrue(config.chapters_source_dir.exists())
            self.assertTrue(config.translations_dir.exists())
            self.assertEqual(
                config.chapter_source_path(1, "Chapter 1"),
                root / "data" / "output" / "chapters" / "source" / "chapter_001_chapter_1.txt",
            )
            self.assertEqual(
                config.chapter_translation_path(1, "Chapter 1"),
                root / "data" / "output" / "chapters" / "zh" / "chapter_001_chapter_1_zh.md",
            )


if __name__ == "__main__":
    unittest.main()
