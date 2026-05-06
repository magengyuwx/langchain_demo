import json
import logging
import os
from typing import Any, Iterable
from tqdm import tqdm


products_dir = r"C:\magengyu\商品库\LouisVuitton"
output_dir = r"C:\magengyu\商品库\LouisVuitton"
output_file = "part_grade_label.jsonl"

PART_DISPLAY_NAMES = {
	"exterior": "外部",
	"hardware": "五金",
	"interior": "内部",
	"handle": "手柄/肩带",
	"strap": "肩带",
	"sewing": "缝线",
}

PART_IMAGE_HINTS = {
	"exterior": {
		"preferred_parts": ("正面/整体", "侧面/整体", "底面/整体", "穿搭", "局部细节"),
		"keywords": ("正面", "整体", "侧面", "外侧", "包身", "外观", "纹理", "边角", "细节"),
	},
	"hardware": {
		"preferred_parts": ("五金", "局部细节", "正面/整体", "侧面/整体"),
		"keywords": ("五金", "金属", "拉链", "扣", "链条", "logo", "挂钩", "配件"),
	},
	"interior": {
		"preferred_parts": ("内部", "内侧logo", "局部细节"),
		"keywords": ("内部", "内里", "内衬", "口袋", "打开后", "标签", "里侧"),
	},
	"handle": {
		"preferred_parts": ("局部细节", "正面/整体", "穿搭", "侧面/整体"),
		"keywords": ("手柄", "提手", "肩带", "背带", "挎带", "可调节", "顶部手柄"),
	},
	"strap": {
		"preferred_parts": ("局部细节", "穿搭", "正面/整体", "侧面/整体"),
		"keywords": ("肩带", "背带", "挎带", "可调节", "strap"),
	},
	"sewing": {
		"preferred_parts": ("局部细节", "正面/整体", "侧面/整体", "底面/整体"),
		"keywords": ("缝线", "走线", "车线", "线迹", "针脚", "缝边"),
	},
}

SUPPORTED_PART_KEYS = frozenset(PART_DISPLAY_NAMES.keys())

USER_PROMPT_TEMPLATE = "{images}请判断这个商品的{part_name}状况。只输出简短中文描述，例如：轻微磨损、划痕、失去光泽、边角磨损。只根据这部分对应的图像判断。"

logger = logging.getLogger(__name__)


def _normalize_path(path: str) -> str:
	return path.replace("\\", "/")


def _normalize_text(value: Any) -> str:
	return str(value or "").strip()


def _display_part_name(part_key: str) -> str:
	return PART_DISPLAY_NAMES.get(part_key, part_key)


def _extract_product_name(annotation: dict[str, Any], product_path: str) -> str:
	product_name = _normalize_text(annotation.get("product_name"))
	if product_name:
		return product_name
	return os.path.basename(product_path)


def _extract_part_conditions(annotation: dict[str, Any]) -> list[tuple[str, str]]:
	condition = annotation.get("condition")
	if not isinstance(condition, dict):
		return []

	items: list[tuple[str, str]] = []
	for key, value in condition.items():
		if key in {"grade", "accessories"}:
			continue
		if key not in SUPPORTED_PART_KEYS:
			continue

		text = _normalize_text(value)
		if text:
			items.append((str(key).strip(), text))

	return items


def _score_image_match(image_item: dict[str, Any], part_key: str) -> int:
	hints = PART_IMAGE_HINTS.get(part_key, {"preferred_parts": (), "keywords": ()})
	image_part = _normalize_text(image_item.get("part"))
	description = _normalize_text(image_item.get("description")).lower()
	score = 0

	preferred_parts = hints.get("preferred_parts", ())
	for index, candidate_part in enumerate(preferred_parts):
		if image_part == candidate_part:
			score += 100 - index * 10
			break

	for keyword in hints.get("keywords", ()): 
		if keyword.lower() in description:
			score += 20

	part_name = _display_part_name(part_key).lower()
	if part_name and part_name in description:
		score += 15

	if part_key.lower() in description:
		score += 10

	return score


def _select_part_image_paths(
	annotation: dict[str, Any],
	product_path: str,
	products_root: str,
	part_key: str,
) -> list[str]:
	images = annotation.get("images", [])
	if not isinstance(images, list):
		return []

	ranked_images: list[tuple[int, int, str]] = []
	for index, image_item in enumerate(images):
		if not isinstance(image_item, dict):
			continue

		file_name = _normalize_text(image_item.get("filename"))
		if not file_name:
			continue

		image_path = os.path.join(product_path, file_name)
		if not os.path.isfile(image_path):
			continue

		score = _score_image_match(image_item, part_key)
		ranked_images.append((score, -index, os.path.relpath(image_path, products_root)))

	if not ranked_images:
		return []

	ranked_images.sort(reverse=True)
	positive_matches = [path for score, _, path in ranked_images if score > 0]
	if positive_matches:
		return positive_matches[:3]

	return [path for _, _, path in ranked_images[:1]]


def _build_sample(
	product_name: str,
	image_paths: list[str],
	part_key: str,
	condition_value: str,
) -> dict[str, Any]:
	part_name = _display_part_name(part_key)
	image_tokens = "<image>" * len(image_paths)
	base_prompt = USER_PROMPT_TEMPLATE.format(images=image_tokens, part_name=part_name)
	user_text = f"商品名称：{product_name}\n{base_prompt}"

	sample = {
		"messages": [
			{"role": "user", "content": user_text},
			{"role": "assistant", "content": condition_value},
		],
		"images": [_normalize_path(image_path) for image_path in image_paths],
	}

	if user_text.count("<image>") != len(sample["images"]):
		raise ValueError("样本中的 <image> 数量与 images 列数量不一致。")

	return {
		"messages": sample["messages"],
		"images": sample["images"],
	}


def generate_vlm_dataset(
	products_root: str,
	images_output_dir: str,
	label_file_name: str,
) -> dict[str, Any]:
	products_root = os.path.abspath(products_root)
	images_output_dir = os.path.abspath(images_output_dir)

	if not os.path.isdir(products_root):
		raise FileNotFoundError(f"products_dir 不存在: {products_root}")

	os.makedirs(images_output_dir, exist_ok=True)
	label_path = os.path.join(images_output_dir, label_file_name)

	samples: list[dict[str, Any]] = []
	total_products = 0
	total_part_samples = 0
	skipped_products = 0
	skipped_parts = 0
	skipped_images = 0
	part_counts: dict[str, int] = {}
	skip_reasons: dict[str, int] = {
		"missing_annotation": 0,
		"missing_part_conditions": 0,
		"missing_or_invalid_images": 0,
		"missing_part_images": 0,
		"invalid_image_item": 0,
		"missing_image_filename": 0,
		"missing_image_file": 0,
	}

	product_paths = [
		entry.path for entry in os.scandir(products_root) if entry.is_dir(follow_symlinks=False)
	]
	for product_path in tqdm(product_paths, desc="处理商品", unit="商品"):

		annotation_path = os.path.join(product_path, "annotation.json")
		if not os.path.isfile(annotation_path):
			skipped_products += 1
			skip_reasons["missing_annotation"] += 1
			continue

		with open(annotation_path, "r", encoding="utf-8") as f:
			annotation = json.load(f)

		images = annotation.get("images", [])
		if not isinstance(images, list) or not images:
			skipped_products += 1
			skip_reasons["missing_or_invalid_images"] += 1
			continue

		valid_image_count = 0
		for image_item in images:
			if not isinstance(image_item, dict):
				skipped_images += 1
				skip_reasons["invalid_image_item"] += 1
				continue

			file_name = str(image_item.get("filename", "")).strip()
			if not file_name:
				skipped_images += 1
				skip_reasons["missing_image_filename"] += 1
				continue

			image_abs_path = os.path.join(product_path, file_name)
			if not os.path.isfile(image_abs_path):
				skipped_images += 1
				skip_reasons["missing_image_file"] += 1
				continue

			valid_image_count += 1

		if valid_image_count == 0:
			skipped_products += 1
			skip_reasons["missing_or_invalid_images"] += 1
			continue

		part_conditions = _extract_part_conditions(annotation)
		if not part_conditions:
			skipped_products += 1
			skip_reasons["missing_part_conditions"] += 1
			continue

		product_name = _extract_product_name(annotation, product_path)

		product_sample_count = 0
		for part_key, condition_value in part_conditions:
			image_rel_paths = _select_part_image_paths(annotation, product_path, products_root, part_key)
			if not image_rel_paths:
				skipped_parts += 1
				skip_reasons["missing_part_images"] += 1
				continue

			samples.append(_build_sample(product_name, image_rel_paths, part_key, condition_value))
			part_counts[part_key] = part_counts.get(part_key, 0) + 1
			total_part_samples += 1
			product_sample_count += 1

		if product_sample_count == 0:
			skipped_products += 1
			continue

		total_products += 1

	with open(label_path, "w", encoding="utf-8") as f:
		for sample in samples:
			f.write(json.dumps(sample, ensure_ascii=False))
			f.write("\n")

	return {
		"products_root": products_root,
		"output_dir": images_output_dir,
		"label_file": label_path,
		"samples": total_part_samples,
		"products": total_products,
		"skipped_products": skipped_products,
		"skipped_parts": skipped_parts,
		"skipped_images": skipped_images,
		"part_counts": dict(sorted(part_counts.items())),
		"skip_reasons": skip_reasons,
	}


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	stats = generate_vlm_dataset(products_dir, output_dir, output_file)
	print(f"已生成 {stats['samples']} 条部件样本，覆盖 {stats['products']} 个商品，跳过 {stats['skipped_products']} 个商品。")
	print(f"跳过部件数量: {stats['skipped_parts']}")
	print(f"缺失或无效图片数量: {stats['skipped_images']}")
	print(f"图片目录: {stats['output_dir']}")
	print(f"标注文件: {stats['label_file']}")
	logger.info("跳过原因统计:")
	for reason, count in stats["skip_reasons"].items():
		if count > 0:
			logger.info("  %s: %d", reason, count)
	logger.info("部件样本数量统计:")
	if stats["part_counts"]:
		for part, count in stats["part_counts"].items():
			logger.info("  %s: %d", part, count)
	else:
		logger.info("  (无)")


if __name__ == "__main__":
	main()
