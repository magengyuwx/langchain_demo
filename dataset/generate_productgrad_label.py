import json
import logging
import os
from typing import Any

from tqdm import tqdm


products_dir = r"C:\magengyu\商品库\LouisVuitton"
output_dir = r"C:\magengyu\商品库\LouisVuitton"
output_file = "product_grade_label.jsonl"

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

USER_PROMPT_TEMPLATE = (
	"你是奢侈品商品成色评估助手。"
	"请根据商品名称、部件痕迹信息（含附件）和图片描述，输出商品整体描述与整体grade。"
	"grade 候选包括：{grades}。"
	"只输出一个 JSON 对象，格式为 "
	'{{"overall_description":"...","grade":"..."}}。'
)

logger = logging.getLogger(__name__)


def _normalize_text(value: Any) -> str:
	return str(value or "").strip()


def _normalize_grade(grade: Any) -> str:
	return _normalize_text(grade)


def _map_grade(grade: str) -> str:
	mapped_grade = _normalize_grade(grade)
	if not mapped_grade:
		return ""

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


def _extract_grade(annotation: dict[str, Any]) -> str:
	condition = annotation.get("condition")
	if isinstance(condition, dict):
		grade = _normalize_grade(condition.get("grade"))
		if grade:
			return grade
	return _normalize_grade(annotation.get("grade"))


def _extract_product_name(annotation: dict[str, Any], product_path: str) -> str:
	product_name = _normalize_text(annotation.get("product_name"))
	if product_name:
		return product_name
	return os.path.basename(product_path)


def _extract_condition_without_grade(annotation: dict[str, Any]) -> dict[str, str]:
	condition = annotation.get("condition")
	if not isinstance(condition, dict):
		return {}

	result: dict[str, str] = {}
	for key, value in condition.items():
		if key == "grade":
			continue

		if isinstance(value, list):
			normalized_items = [_normalize_text(item) for item in value if _normalize_text(item)]
			normalized_value = "、".join(normalized_items)
		else:
			normalized_value = _normalize_text(value)

		if normalized_value:
			result[_normalize_text(key)] = normalized_value

	return result


def _extract_image_descriptions(annotation: dict[str, Any]) -> list[str]:
	images = annotation.get("images", [])
	if not isinstance(images, list):
		return []

	descriptions: list[str] = []
	for image_item in images:
		if not isinstance(image_item, dict):
			continue

		description = _normalize_text(image_item.get("description"))
		if not description:
			continue

		file_name = _normalize_text(image_item.get("filename"))
		part = _normalize_text(image_item.get("part"))
		prefix_parts = [item for item in [file_name, part] if item]
		if prefix_parts:
			descriptions.append(f"{' | '.join(prefix_parts)}: {description}")
		else:
			descriptions.append(description)

	return descriptions


def _extract_overall_description(annotation: dict[str, Any], condition_items: dict[str, str]) -> str:
	description_obj = annotation.get("description")
	if isinstance(description_obj, dict):
		details = _normalize_text(description_obj.get("details"))
		if details:
			return details

	if condition_items:
		parts = [f"{key}：{value}" for key, value in condition_items.items()]
		return "；".join(parts)

	return ""


def _format_condition_for_prompt(condition_items: dict[str, str]) -> str:
	if not condition_items:
		return "(无)"
	return "\n".join(f"- {key}: {value}" for key, value in condition_items.items())


def _format_image_desc_for_prompt(image_descriptions: list[str]) -> str:
	if not image_descriptions:
		return "(无)"
	return "\n".join(f"- {text}" for text in image_descriptions)


def _build_user_prompt(
	product_name: str,
	condition_items: dict[str, str],
	image_descriptions: list[str],
	grade_candidates: tuple[str, ...],
) -> str:
	return (
		f"{USER_PROMPT_TEMPLATE.format(grades=', '.join(grade_candidates))}\n\n"
		f"商品名称:\n{product_name}\n\n"
		f"part_condition:\n{_format_condition_for_prompt(condition_items)}\n\n"
		f"每张图片的描述:\n{_format_image_desc_for_prompt(image_descriptions)}"
	)


def _build_sample(
	product_name: str,
	condition_items: dict[str, str],
	image_descriptions: list[str],
	overall_description: str,
	grade: str,
	grade_candidates: tuple[str, ...],
) -> dict[str, Any]:
	user_text = _build_user_prompt(product_name, condition_items, image_descriptions, grade_candidates)
	assistant_text = json.dumps(
		{
			"overall_description": overall_description,
			"grade": grade,
		},
		ensure_ascii=False,
	)
	return {
		"messages": [
			{"role": "user", "content": user_text},
			{"role": "assistant", "content": assistant_text},
		]
	}


def generate_dataset(
	products_root: str,
	output_root: str,
	label_file_name: str,
) -> dict[str, Any]:
	products_root = os.path.abspath(products_root)
	output_root = os.path.abspath(output_root)

	if not os.path.isdir(products_root):
		raise FileNotFoundError(f"products_dir 不存在: {products_root}")

	os.makedirs(output_root, exist_ok=True)
	label_path = os.path.join(output_root, label_file_name)

	samples: list[dict[str, Any]] = []
	total_products = 0
	skipped_products = 0
	grade_counts: dict[str, int] = {}
	skip_reasons: dict[str, int] = {
		"missing_annotation": 0,
		"missing_grade": 0,
		"missing_condition_items": 0,
		"missing_image_descriptions": 0,
		"missing_overall_description": 0,
	}

	product_paths = [
		entry.path for entry in os.scandir(products_root) if entry.is_dir(follow_symlinks=False)
	]
	grade_candidates = DEFAULT_GRADE_CANDIDATES

	for product_path in tqdm(product_paths, desc="处理商品", unit="商品"):
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

		condition_items = _extract_condition_without_grade(annotation)
		if not condition_items:
			skipped_products += 1
			skip_reasons["missing_condition_items"] += 1
			continue

		image_descriptions = _extract_image_descriptions(annotation)
		if not image_descriptions:
			skipped_products += 1
			skip_reasons["missing_image_descriptions"] += 1
			continue

		overall_description = _extract_overall_description(annotation, condition_items)
		if not overall_description:
			skipped_products += 1
			skip_reasons["missing_overall_description"] += 1
			continue

		product_name = _extract_product_name(annotation, product_path)
		sample = _build_sample(
			product_name=product_name,
			condition_items=condition_items,
			image_descriptions=image_descriptions,
			overall_description=overall_description,
			grade=grade,
			grade_candidates=grade_candidates,
		)
		samples.append(sample)
		grade_counts[grade] = grade_counts.get(grade, 0) + 1
		total_products += 1

	with open(label_path, "w", encoding="utf-8") as f:
		for sample in samples:
			f.write(json.dumps(sample, ensure_ascii=False))
			f.write("\n")

	return {
		"products_root": products_root,
		"output_dir": output_root,
		"label_file": label_path,
		"samples": total_products,
		"skipped_products": skipped_products,
		"grade_counts": dict(sorted(grade_counts.items())),
		"skip_reasons": skip_reasons,
	}


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	stats = generate_dataset(products_dir, output_dir, output_file)
	print(f"已生成 {stats['samples']} 条样本，跳过 {stats['skipped_products']} 个商品。")
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
