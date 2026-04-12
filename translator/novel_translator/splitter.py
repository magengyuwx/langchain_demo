from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import slugify


CHAPTER_HEADER_PATTERNS = (
    r"(?mi)^(chapter\s+\d+[^\n]*)$",
    r"(?mi)^(chapter\s+[ivxlcdm]+[^\n]*)$",
    r"(?mi)^(第[零〇一二三四五六七八九十百千0-9]+章[^\n]*)$",
    r"(?mi)^((prologue|epilogue)[^\n]*)$",
)


@dataclass(slots=True)
class Chapter:
    number: int
    title: str
    content: str


def split_novel_into_chapters(text: str, fallback_chunk_size: int = 15000) -> list[Chapter]:
    normalized = text.replace("\r\n", "\n").strip()
    matches = _detect_chapter_headers(normalized)

    if not matches:
        return _fallback_split(normalized, fallback_chunk_size)

    chapters: list[Chapter] = []
    for index, match in enumerate(matches, start=1):
        start = match.start()
        end = matches[index].start() if index < len(matches) else len(normalized)
        block = normalized[start:end].strip()
        title = match.group(0).strip() or f"Chapter {index}"
        chapters.append(Chapter(number=index, title=title, content=block))

    return chapters


def save_chapters(chapters: list[Chapter], output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    for chapter in chapters:
        file_path = output_path / f"chapter_{chapter.number:03d}_{slugify(chapter.title)}.txt"
        file_path.write_text(f"{chapter.content.strip()}\n", encoding="utf-8")
        saved_files.append(file_path)

    return saved_files


def load_saved_chapters(chapters_dir: str | Path) -> list[Chapter]:
    directory = Path(chapters_dir)
    chapter_files = sorted(directory.glob("chapter_*.txt"))

    chapters: list[Chapter] = []
    for fallback_index, chapter_file in enumerate(chapter_files, start=1):
        match = re.search(r"chapter_(\d+)_", chapter_file.name)
        number = int(match.group(1)) if match else fallback_index
        content = chapter_file.read_text(encoding="utf-8").strip()
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        title = lines[0] if lines else f"Chapter {number}"
        chapters.append(Chapter(number=number, title=title, content=content))

    return chapters


def _detect_chapter_headers(text: str) -> list[re.Match[str]]:
    candidates: list[list[re.Match[str]]] = []
    for pattern in CHAPTER_HEADER_PATTERNS:
        matches = list(re.finditer(pattern, text))
        if matches:
            candidates.append(matches)

    if not candidates:
        return []

    return max(candidates, key=len)


def _fallback_split(text: str, fallback_chunk_size: int) -> list[Chapter]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", ". ", " "],
        chunk_size=fallback_chunk_size,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)

    return [
        Chapter(number=index, title=f"Chunk {index}", content=chunk.strip())
        for index, chunk in enumerate(chunks, start=1)
        if chunk.strip()
    ]
