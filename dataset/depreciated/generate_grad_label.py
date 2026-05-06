import json
import logging
import os
from typing import Any, Iterable


products_dir = r"D:\git\LV-Dataset\LouisVuitton"
output_dir = r"D:\git\LV-Dataset\LouisVuitton"
output_file = "grade_label.jsonl"

DEFAULT_GRADE_CANDIDATES = (
	"优秀",
	"轻微磨损",
	"明显磨损",
	"有缺陷",
	"假冒商品",
)

# Available, Shows Wear, Shows Wear (显示磨损), Snag(s), 优秀, 可赠送, 新, 显示磨损, 有缺陷, 磨损, 等级：显示磨损
GRADE_MAP = {
	"available": "优秀",
	"shows wear": "显示磨损",
	"shows wear (显示磨损)": "轻微磨损",
	"snag(s)": "有缺陷",
	"优秀": "优秀",
	"可赠送": "优秀",
	"新": "优秀",
	"显示磨损": "轻微磨损",	
	"有缺陷": "有缺陷",
	"磨损": "明显磨损",
	"等级：显示磨损": "轻微磨损",
}

USER_PROMPT_TEMPLATE = "请综合判断这个商品的整体成色等级。候选包括：{grades}"

logger = logging.getLogger(__name__)


def _normalize_path(path: str) -> str:
	return path.replace("\\", "/")


def _normalize_grade(grade: Any) -> str:
	return str(grade or "").strip()


def _extract_grade(annotation: dict[str, Any]) -> str:
	condition = annotation.get("condition")
	if isinstance(condition, dict):
		grade = _normalize_grade(condition.get("grade"))
		if grade:
			return grade

	# Fallback for data that stores grade at the top level.
	return _normalize_grade(annotation.get("grade"))


def _map_grade(grade: str) -> str:
	mapped_grade = _normalize_grade(grade)
	if not mapped_grade:
		return ""

	# Support alias chains, e.g. A -> B and B -> C.
	for _ in range(3):
		next_grade = GRADE_MAP.get(mapped_grade)
		if next_grade is None:
			next_grade = GRADE_MAP.get(mapped_grade.lower())
		if next_grade is None:
			break

		normalized_next = _normalize_grade(next_grade)
		if not normalized_next or normalized_next == mapped_grade:
			break
		mapped_grade = normalized_next

	return mapped_grade


def _normalize_grade_candidates(grade_candidates: Iterable[str] | None) -> tuple[str, ...]:
	if grade_candidates is None:
		grade_candidates = DEFAULT_GRADE_CANDIDATES

	normalized: list[str] = []
	for item in grade_candidates:
		grade = _normalize_grade(item)
		if grade and grade not in normalized:
			normalized.append(grade)
	return tuple(normalized)


def _build_user_prompt(image_count: int, grade_candidates: tuple[str, ...]) -> str:
	image_tokens = "<image>" * image_count
	if grade_candidates:
		return f"{image_tokens}{USER_PROMPT_TEMPLATE.format(grades=', '.join(grade_candidates))}"
	return f"{image_tokens}请综合判断这个商品的整体成色等级。"


def _build_sample(image_paths: list[str], grade: str, grade_candidates: tuple[str, ...]) -> dict[str, Any]:
	user_text = _build_user_prompt(len(image_paths), grade_candidates)
	sample = {
		"messages": [
			{"role": "user", "content": user_text},
			{"role": "assistant", "content": grade},
		],
		"images": [_normalize_path(image_path) for image_path in image_paths],
	}

	if user_text.count("<image>") != len(sample["images"]):
		raise ValueError("样本中的 <image> 数量与 images 列数量不一致。")

	return sample


def generate_vlm_dataset(
	products_root: str,
	images_output_dir: str,
	label_file_name: str,
	grade_candidates: Iterable[str] | None = None,
) -> dict[str, Any]:
	products_root = os.path.abspath(products_root)
	images_output_dir = os.path.abspath(images_output_dir)
	normalized_grade_candidates = _normalize_grade_candidates(grade_candidates)

	if not os.path.isdir(products_root):
		raise FileNotFoundError(f"products_dir 不存在: {products_root}")

	os.makedirs(images_output_dir, exist_ok=True)
	label_path = os.path.join(images_output_dir, label_file_name)

	samples: list[dict[str, Any]] = []
	total_products = 0
	skipped_products = 0
	skipped_images = 0
	grade_counts: dict[str, int] = {}
	skip_reasons: dict[str, int] = {
		"missing_annotation": 0,
		"missing_grade": 0,
		"invalid_grade": 0,
		"missing_or_invalid_images": 0,
		"invalid_image_item": 0,
		"missing_image_filename": 0,
	}

	product_paths = [
		entry.path for entry in os.scandir(products_root) if entry.is_dir(follow_symlinks=False)
	]
	for product_path in product_paths:

		annotation_path = os.path.join(product_path, "annotation.json")
		if not os.path.isfile(annotation_path):
			skipped_products += 1
			skip_reasons["missing_annotation"] += 1
			continue

		with open(annotation_path, "r", encoding="utf-8") as f:
			annotation = json.load(f)

		raw_grade = _extract_grade(annotation)
		grade = _map_grade(raw_grade)
		if not grade:
			skipped_products += 1
			skip_reasons["missing_grade"] += 1
			continue

		grade_counts[grade] = grade_counts.get(grade, 0) + 1

		if normalized_grade_candidates and grade not in normalized_grade_candidates:
			skipped_products += 1
			skip_reasons["invalid_grade"] += 1
			continue

		images = annotation.get("images", [])
		if not isinstance(images, list) or not images:
			skipped_products += 1
			skip_reasons["missing_or_invalid_images"] += 1
			continue

		image_rel_paths: list[str] = []
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

			image_rel_paths.append(os.path.relpath(os.path.join(product_path, file_name), products_root))

		if not image_rel_paths:
			skipped_products += 1
			skip_reasons["missing_or_invalid_images"] += 1
			continue

		samples.append(_build_sample(image_rel_paths, grade, normalized_grade_candidates))
		total_products += 1

	with open(label_path, "w", encoding="utf-8") as f:
		for sample in samples:
			f.write(json.dumps(sample, ensure_ascii=False))
			f.write("\n")

	return {
		"products_root": products_root,
		"output_dir": images_output_dir,
		"label_file": label_path,
		"samples": total_products,
		"skipped_products": skipped_products,
		"skipped_images": skipped_images,
		"grade_counts": dict(sorted(grade_counts.items())),
		"skip_reasons": skip_reasons,
		"grade_candidates": list(normalized_grade_candidates),
	}


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	stats = generate_vlm_dataset(products_dir, output_dir, output_file)
	print(f"已生成 {stats['samples']} 条样本，跳过 {stats['skipped_products']} 个商品。")
	print(f"缺失或无效图片数量: {stats['skipped_images']}")
	print(f"图片目录: {stats['output_dir']}")
	print(f"标注文件: {stats['label_file']}")
	logger.info("跳过原因统计:")
	for reason, count in stats["skip_reasons"].items():
		if count > 0:
			logger.info("  %s: %d", reason, count)
	logger.info("grade 出现次数统计:")
	if stats["grade_counts"]:
		for grade, count in stats["grade_counts"].items():
			logger.info("  %s: %d", grade, count)
	else:
		logger.info("  (无)")


if __name__ == "__main__":
	main()
