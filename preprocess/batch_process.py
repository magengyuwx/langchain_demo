import json
import os
from typing import Any

from txt2json import build_default_llm, process_product_dir

products_dir = r"D:\git\LV-Dataset\products"

def batch_process_products(products_dir: str) -> list[dict[str, str]]:
    products_dir = os.path.abspath(products_dir)
    if not os.path.isdir(products_dir):
        raise FileNotFoundError(f"products_dir 不存在: {products_dir}")

    llm_model = build_default_llm()
    product_entries: list[dict[str, str]] = []
    failed: list[dict[str, Any]] = []

    subdirs = sorted(
        (entry for entry in os.scandir(products_dir) if entry.is_dir()),
        key=lambda entry: entry.name.lower(),
    )
    for entry in subdirs:
        product_dir = os.path.abspath(entry.path)
        product_entries.append({"name": entry.name, "path": product_dir})

        print(f"开始处理: {entry.name}")
        try:
            process_product_dir(product_dir, llm_model)
        except Exception as exc:
            failed.append({"name": entry.name, "path": product_dir, "error": str(exc)})
            print(f"处理失败: {entry.name} -> {exc}")

    product_list_path = os.path.join(products_dir, "product_list.json")
    with open(product_list_path, "w", encoding="utf-8") as f:
        json.dump(product_entries, f, ensure_ascii=False, indent=2)
    print(f"商品列表已保存: {product_list_path}")

    if failed:
        failed_path = os.path.join(products_dir, "product_failed.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        print(f"失败列表已保存: {failed_path}")

    return product_entries


def main() -> None:
    batch_process_products(products_dir)


if __name__ == "__main__":
    main()
