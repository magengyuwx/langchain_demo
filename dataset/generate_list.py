import json
import os


products_dir = r"D:\git\LV-Dataset\products"
output_file = "annotation_subdirs.json"


def collect_subdirs_with_annotation(root_dir: str) -> list[str]:
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"products_dir 不存在: {root_dir}")

    result = []
    subdirs = sorted(
        (entry for entry in os.scandir(root_dir) if entry.is_dir()),
        key=lambda entry: entry.name.lower(),
    )
    for entry in subdirs:
        annotation_path = os.path.join(entry.path, "annotation.json")
        if os.path.isfile(annotation_path):
            result.append(entry.name)
    return result


def main() -> None:
    subdir_names = collect_subdirs_with_annotation(products_dir)
    output_path = os.path.join(products_dir, output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subdir_names, f, ensure_ascii=False, indent=4)

    print(f"已写入 {len(subdir_names)} 个子目录到: {output_path}")


if __name__ == "__main__":
    main()
