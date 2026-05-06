import json
import os
from typing import Any
from preprocess.txt2json import ALLOWED_IMAGE_PARTS
from tqdm import tqdm


products_dir = r"C:\magengyu\商品库\LouisVuitton"
output_dir = r"C:\magengyu\商品库\LouisVuitton"
output_file = "image_label.jsonl"

allowed_parts = tuple(ALLOWED_IMAGE_PARTS)

USER_PROMPT_TEMPLATE = (
    "请识别这张商品图像并输出 JSON。"
    "part 候选包括：{parts}。"
    "description 需要简洁描述图中关键可见内容。"
    "只输出一个 JSON 对象，格式为 "
    '{{"part":"...","description":"..."}}'
)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _build_sample(image_path: str, part: str, description: str) -> dict[str, Any]:
    user_text = f"<image>{USER_PROMPT_TEMPLATE.format(parts=', '.join(allowed_parts))}"
    assistant_output = json.dumps(
        {
            "part": part,
            "description": description,
        },
        ensure_ascii=False,
    )
    sample = {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_output},
        ],
        "images": [_normalize_path(image_path)],
    }

    if user_text.count("<image>") != len(sample["images"]):
        raise ValueError("样本中的 <image> 数量与 images 列数量不一致。")

    return {
        "messages": sample["messages"],
        "images": sample["images"],
    }


def generate_vlm_dataset(
    products_root: str, images_output_dir: str, image_label_file_name: str
) -> dict[str, Any]:
    products_root = os.path.abspath(products_root)
    images_output_dir = os.path.abspath(images_output_dir)

    if not os.path.isdir(products_root):
        raise FileNotFoundError(f"products_dir 不存在: {products_root}")

    os.makedirs(images_output_dir, exist_ok=True)
    image_label_path = os.path.join(images_output_dir, image_label_file_name)

    samples: list[dict[str, Any]] = []
    total_images = 0
    skipped_images = 0

    product_paths = [
        entry.path for entry in os.scandir(products_root) if entry.is_dir(follow_symlinks=False)
    ]
    for product_path in tqdm(product_paths, desc="处理商品", unit="商品"):

        annotation_path = os.path.join(product_path, "annotation.json")
        if not os.path.isfile(annotation_path):
            continue

        with open(annotation_path, "r", encoding="utf-8") as f:
            annotation = json.load(f)

        images = annotation.get("images", [])
        if not isinstance(images, list):
            continue

        for image_item in images:
            if not isinstance(image_item, dict):
                skipped_images += 1
                continue

            file_name = str(image_item.get("filename", "")).strip()
            part = str(image_item.get("part", "")).strip()
            description = str(image_item.get("description", "")).strip()
            if not file_name or not part:
                skipped_images += 1
                continue
            if part not in allowed_parts:
                skipped_images += 1
                continue

            image_abs_path = os.path.join(product_path, file_name)
            if not os.path.isfile(image_abs_path):
                skipped_images += 1
                continue

            image_rel_path = os.path.relpath(image_abs_path, products_root)
            samples.append(_build_sample(image_rel_path, part, description))
            total_images += 1

    with open(image_label_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write("\n")

    stats = {
        "products_root": products_root,
        "output_dir": images_output_dir,
        "image_label_file": image_label_path,
        "samples": total_images,
        "skipped_images": skipped_images,
    }
    return stats


def main() -> None:
    stats = generate_vlm_dataset(products_dir, output_dir, output_file)
    print(f"已生成 {stats['samples']} 条样本，跳过 {stats['skipped_images']} 条。")
    print(f"图片目录: {stats['output_dir']}")
    print(f"图像标注文件: {stats['image_label_file']}")


if __name__ == "__main__":
    main()
