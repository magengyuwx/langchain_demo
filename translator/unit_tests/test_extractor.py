from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_translator.config import AppConfig
from novel_translator.extractor import StoryMetadataExtractor


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, _):
        return self._responses.pop(0)


class TestExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(project_root=PROJECT_ROOT)

    def test_extract_returns_outline_and_characters_sections(self) -> None:
        llm = _FakeLLM(
            [
                _FakeResponse("剧情要点: A\n人物要点: B"),
                _FakeResponse("# 小说大纲\n\n主线剧情\n\n# 人物设定\n\n角色设定"),
            ]
        )
        extractor = StoryMetadataExtractor(self.config, llm)

        outline, characters = extractor.extract("短文本")

        self.assertIn("主线剧情", outline)
        self.assertIn("角色设定", characters)

    def test_message_to_text_supports_list(self) -> None:
        text = StoryMetadataExtractor._message_to_text(_FakeResponse(["a", "b"]))
        self.assertEqual(text, "a\nb")

    def test_extract_section(self) -> None:
        source = "# 小说大纲\n\nA\n\n# 人物设定\n\nB"
        section = StoryMetadataExtractor._extract_section(source, "人物设定")
        self.assertEqual(section, "B")


if __name__ == "__main__":
    unittest.main()
