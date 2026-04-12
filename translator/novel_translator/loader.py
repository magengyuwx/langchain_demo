from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from pypdf import PdfReader


def load_text(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8").strip()
    if suffix == ".pdf":
        return _load_pdf_text(path)
    if suffix == ".epub":
        return _load_epub_text(path)

    raise ValueError(f"暂不支持的文件类型: {path.suffix}，当前支持 txt / md / pdf / epub。")


def save_text(file_path: str | Path, content: str) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = content.strip()
    path.write_text(f"{text}\n" if text else "", encoding="utf-8")
    return path


def copy_text(source_path: str | Path, target_path: str | Path) -> Path:
    return save_text(target_path, load_text(source_path))


def _load_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(text)
    return "\n\n".join(pages).strip()


def _load_epub_text(path: Path) -> str:
    book = epub.read_epub(str(path))
    sections: list[str] = []

    for item in book.get_items():
        if item.get_type() != ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text("\n", strip=True)
        if text:
            sections.append(text)

    return "\n\n".join(sections).strip()
