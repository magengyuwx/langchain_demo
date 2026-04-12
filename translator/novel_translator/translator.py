from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import AppConfig
from .rag import NovelRAG
from .splitter import Chapter


TRANSLATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是资深文学翻译，请将英文小说翻译成自然、连贯、风格统一的简体中文。"
            "翻译时必须保证人名、地名、称谓、时间线和角色口吻前后一致。",
        ),
        (
            "human",
            "请翻译当前章节片段。\n\n"
            "【全局小说大纲】\n{outline}\n\n"
            "【人物设定】\n{characters}\n\n"
            "【上一章中文译文（如有）】\n{previous_chapter}\n\n"
            "【RAG 检索出的相关原文上下文】\n{related_source}\n\n"
            "【当前章节信息】\n"
            "标题：{title}\n"
            "当前片段：第 {segment_index}/{segment_total} 段\n\n"
            "【同章已完成译文尾部（帮助衔接）】\n{previous_segment_tail}\n\n"
            "【待翻译原文】\n{segment}\n\n"
            "要求：\n"
            "1. 只输出中文译文，不要解释。\n"
            "2. 不遗漏剧情与对白。\n"
            "3. 若当前是章节的中间片段，不要重复输出前文内容。\n"
            "4. 保持自然分段和文学性。",
        ),
    ]
)


class ChapterTranslator:
    def __init__(self, config: AppConfig, llm, rag: NovelRAG) -> None:
        self.config = config
        self.llm = llm
        self.rag = rag

    def translate(self, chapter: Chapter, outline: str, characters: str) -> str:
        context = self.rag.build_translation_context(chapter)
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", ". ", " "],
            chunk_size=self.config.translation_chunk_size,
            chunk_overlap=self.config.translation_chunk_overlap,
        )
        segments = splitter.split_text(chapter.content)

        translated_segments: list[str] = []
        for index, segment in enumerate(segments, start=1):
            previous_segment_tail = "\n".join(translated_segments)[-1200:]
            response = self.llm.invoke(
                TRANSLATION_PROMPT.format_messages(
                    outline=context["outline"] or outline,
                    characters=context["characters"] or characters,
                    previous_chapter=context["previous_chapter"],
                    related_source=context["related_source"],
                    title=chapter.title,
                    segment_index=index,
                    segment_total=len(segments),
                    previous_segment_tail=previous_segment_tail,
                    segment=segment,
                )
            )
            translated_segments.append(self._normalize(self._message_to_text(response)))

        return "\n\n".join(segment for segment in translated_segments if segment.strip()).strip()

    @staticmethod
    def _message_to_text(response) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(str(item) for item in content).strip()
        return str(content).strip()

    @staticmethod
    def _normalize(text: str) -> str:
        cleaned = text.replace("```markdown", "").replace("```", "").strip()
        return cleaned
