from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import AppConfig


PARTIAL_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是长篇小说分析助手，擅长从长文本中提炼主线剧情、人物设定和角色关系。",
        ),
        (
            "human",
            "请分析以下小说片段，并输出两部分内容：\n"
            "1. 剧情要点：按时间顺序列出关键事件\n"
            "2. 人物要点：记录新出现或重要人物的身份、关系、称谓、目标、性格\n\n"
            "片段编号：{chunk_index}\n"
            "小说片段：\n{chunk}",
        ),
    ]
)

MERGE_METADATA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是小说编辑，请将零散笔记整理成稳定可复用的翻译资料。",
        ),
        (
            "human",
            "请基于以下分段分析结果，合并输出：\n"
            "# 小说大纲\n"
            "- 按故事推进顺序总结主要剧情、矛盾冲突和阶段目标\n\n"
            "# 人物设定\n"
            "- 按人物列出身份、关系网、称谓、性格特征、重要经历\n\n"
            "要求：去重、统一名字写法、避免遗漏关键人物。\n\n"
            "分析材料：\n{chunk_notes}",
        ),
    ]
)


class StoryMetadataExtractor:
    def __init__(self, config: AppConfig, llm) -> None:
        self.config = config
        self.llm = llm

    def extract(self, full_text: str) -> tuple[str, str]:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", ". ", " "],
            chunk_size=6000,
            chunk_overlap=400,
        )
        chunks = splitter.split_text(full_text)

        chunk_notes: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            response = self.llm.invoke(
                PARTIAL_ANALYSIS_PROMPT.format_messages(
                    chunk_index=index,
                    chunk=chunk,
                )
            )
            chunk_notes.append(f"## 片段 {index}\n{self._message_to_text(response)}")

        merged_response = self.llm.invoke(
            MERGE_METADATA_PROMPT.format_messages(
                chunk_notes="\n\n".join(chunk_notes),
            )
        )
        merged_text = self._message_to_text(merged_response)

        outline = self._extract_section(merged_text, "小说大纲") or merged_text
        characters = self._extract_section(merged_text, "人物设定") or merged_text
        return outline.strip(), characters.strip()

    @staticmethod
    def _message_to_text(response) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(str(item) for item in content).strip()
        return str(content).strip()

    @staticmethod
    def _extract_section(text: str, heading: str) -> str:
        pattern = rf"(?is)(?:^|\n)#+\s*{re.escape(heading)}\s*(.*?)(?=\n#+\s*[^\n]+|\Z)"
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""
