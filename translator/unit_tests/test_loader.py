from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_translator.loader import copy_text, load_text, save_text


class TestLoader(unittest.TestCase):
    def test_load_and_save_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            file_path = root / "input.txt"
            file_path.write_text("  hello world  ", encoding="utf-8")

            loaded = load_text(file_path)
            self.assertEqual(loaded, "hello world")

            output_path = root / "nested" / "output.txt"
            save_text(output_path, "  content  ")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "content\n")

    def test_copy_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.md"
            target = root / "copied" / "target.md"
            source.write_text("copy me", encoding="utf-8")

            copy_text(source, target)
            self.assertEqual(target.read_text(encoding="utf-8"), "copy me\n")

    def test_load_text_raises_for_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_text("missing_file.txt")

    def test_load_text_raises_for_unsupported_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "a.docx"
            file_path.write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_text(file_path)


if __name__ == "__main__":
    unittest.main()
