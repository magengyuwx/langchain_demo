from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_translator.splitter import Chapter, load_saved_chapters, save_chapters, split_novel_into_chapters


class TestSplitter(unittest.TestCase):
    def test_split_novel_with_chapter_headers(self) -> None:
        text = """Chapter 1 First\nA\n\nChapter 2 Second\nB"""
        chapters = split_novel_into_chapters(text)

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0].number, 1)
        self.assertIn("Chapter 1", chapters[0].title)
        self.assertIn("A", chapters[0].content)
        self.assertEqual(chapters[1].number, 2)

    def test_split_novel_fallback(self) -> None:
        text = ("no chapter header. " * 400).strip()
        chapters = split_novel_into_chapters(text, fallback_chunk_size=300)

        self.assertGreaterEqual(len(chapters), 2)
        self.assertTrue(chapters[0].title.startswith("Chunk"))

    def test_save_and_load_saved_chapters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            chapters = [
                Chapter(number=1, title="Chapter 1", content="Chapter 1\nBody 1"),
                Chapter(number=2, title="Chapter 2", content="Chapter 2\nBody 2"),
            ]
            saved = save_chapters(chapters, output_dir)

            self.assertEqual(len(saved), 2)
            loaded = load_saved_chapters(output_dir)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].number, 1)
            self.assertIn("Body 1", loaded[0].content)


if __name__ == "__main__":
    unittest.main()
