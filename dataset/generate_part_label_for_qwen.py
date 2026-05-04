import json
import os
import shutil
from typing import Any
from preprocess.txt2json import ALLOWED_IMAGE_PARTS


products_dir = r"D:\git\LV-Dataset\products"
output_dir = r"D:\git\LV-Dataset\vlm_dataset"
output_file = "data.jsonl"
subdirs_file = "annotation_subdirs.json"

allowed_parts = tuple(ALLOWED_IMAGE_PARTS)

SYSTEM_PROMPT = "你是一个奢侈品商品图像部位识别助手。请只输出部位名称。"
USER_PROMPT_TEMPLATE = "这个图像是商品的哪个部位？候选包括：{parts}"


def _load_subdir_names(root_dir: str, file_name: str) -> list[str]:
    file_path = os.path.join(root_dir, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"未找到子目录清单文件: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{file_path} 格式错误，必须是字符串数组。")

    names = []
    for item in data:
        if isinstance(item, str) and item.strip():
            names.append(item.strip())
    return names


def _safe_copy_image(src_path: str, dst_dir: str, product_name: str) -> str:
    original_name = os.path.basename(src_path)
    base, ext = os.path.splitext(original_name)
    candidate = f"{product_name}__{base}{ext}"
    dst_path = os.path.join(dst_dir, candidate)

    shutil.copy2(src_path, dst_path)
    return candidate


def _build_sample(image_name: str, part: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": [{"text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"text": USER_PROMPT_TEMPLATE.format(parts=", ".join(allowed_parts))},
                    {"image": image_name},
                ],
            },
            {"role": "assistant", "content": [{"text": part}]},
        ]
    }


def generate_vlm_dataset(
    products_root: str, images_output_dir: str, label_file_name: str, subdir_list_file: str
) -> dict[str, Any]:
    products_root = os.path.abspath(products_root)
    images_output_dir = os.path.abspath(images_output_dir)

    if not os.path.isdir(products_root):
        raise FileNotFoundError(f"products_dir 不存在: {products_root}")

    os.makedirs(images_output_dir, exist_ok=True)
    label_path = os.path.join(images_output_dir, label_file_name)

    samples: list[dict[str, Any]] = []
    total_images = 0
    skipped_images = 0

    subdir_names = _load_subdir_names(products_root, subdir_list_file)
    for subdir_name in subdir_names:
        product_path = os.path.join(products_root, subdir_name)
        if not os.path.isdir(product_path):
            skipped_images += 1
            continue

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
            if not file_name or not part:
                skipped_images += 1
                continue
            if part not in allowed_parts:
                skipped_images += 1
                continue

            src_image = os.path.join(product_path, file_name)
            if not os.path.isfile(src_image):
                skipped_images += 1
                continue

            copied_name = _safe_copy_image(src_image, images_output_dir, subdir_name)
            samples.append(_build_sample(copied_name, part))
            total_images += 1

    with open(label_path, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "products_root": products_root,
        "output_dir": images_output_dir,
        "label_file": label_path,
        "samples": total_images,
        "skipped_images": skipped_images,
    }
    return stats


def main() -> None:
    stats = generate_vlm_dataset(products_dir, output_dir, output_file, subdirs_file)
    print(f"已生成 {stats['samples']} 条样本，跳过 {stats['skipped_images']} 条。")
    print(f"图片目录: {stats['output_dir']}")
    print(f"标注文件: {stats['label_file']}")


if __name__ == "__main__":
    main()


