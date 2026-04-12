from __future__ import annotations

import argparse
from pathlib import Path

from novel_translator import AppConfig, NovelTranslationWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="长篇小说翻译工作流（LangChain）")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("prepare", "translate", "all"):
        subparser = subparsers.add_parser(command_name)
        subparser.add_argument(
            "--input",
            default="data/input/novel.txt",
            help="原始小说 txt 文件路径，默认是 data/input/novel.txt",
        )
        subparser.add_argument(
            "--max-chapters",
            type=int,
            default=None,
            help="仅处理前 N 章，便于调试或断点续跑。",
        )
        if command_name in {"translate", "all"}:
            subparser.add_argument(
                "--force",
                action="store_true",
                help="即使已有中文译文，也强制重新翻译。",
            )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config = AppConfig.from_env(project_root)
    workflow = NovelTranslationWorkflow(config)

    if args.command == "prepare":
        chapters = workflow.prepare(
            args.input,
            max_chapters=args.max_chapters,
        )
        print(f"预处理完成，共识别 {len(chapters)} 个章节。")
        return

    if args.command == "translate":
        outputs = workflow.translate(
            input_path=args.input,
            force=args.force,
            max_chapters=args.max_chapters,
        )
        print(f"翻译完成，本次输出 {len(outputs)} 个章节文件。")
        return

    chapters, outputs = workflow.run_all(
        input_path=args.input,
        force=args.force,
        max_chapters=args.max_chapters,
    )
    print(
        f"完整流程完成：共 {len(chapters)} 个章节，"
        f"本次新增/覆盖 {len(outputs)} 个中文章节文件。"
    )


if __name__ == "__main__":
    main()
