from __future__ import annotations

import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_translator.config import AppConfig
from novel_translator.llm_factory import build_chat_model


class TestLLMFactory(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig.from_env(PROJECT_ROOT)
        self.config.llm_provider = "ollama"
        self.config.chat_model = self.config.chat_model or "kimi-k2.5:cloud"
        self.config.base_url = self.config.base_url or "http://localhost:11434"

    def test_real_llm_invoke_returns_text(self) -> None:
        model = build_chat_model(self.config)
        result = model.invoke("请只回复：测试通过")
        content = getattr(result, "content", result)
        text = str(content).strip()

        print(f"真实llm返回结果: {text}")
        self.assertTrue(text)


if __name__ == "__main__":
    unittest.main()
