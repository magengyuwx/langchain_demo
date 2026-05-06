import base64
import json
import mimetypes
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

try:
    from validate_annotation import Annotation, get_annotation_schema, validate_annotation
except ImportError:
    from .validate_annotation import Annotation, get_annotation_schema, validate_annotation


load_dotenv()

qwen_api_key = os.getenv("QWEN_API_KEY")
qwen_model = os.getenv("QWEN_MODEL", "qwen-plus")
qwen_url = os.getenv("QWEN_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

base_dir = os.path.dirname(os.path.abspath(__file__))
sample_dir = os.path.join(base_dir, "sample")
info_cn_file = os.path.join(sample_dir, "info_cn.txt")
info_en_file = os.path.join(sample_dir, "info_en.txt")
output_file = os.path.join(sample_dir, "annotation.json")

ALLOWED_IMAGE_PARTS = (
    "正面/整体",
    "侧面/整体",
    "底面/整体",
    "穿搭",
    "局部细节",
    "外侧logo",
    "内侧logo",
    "内部",
)


class ImageDescriptionResult(BaseModel):
    part: Literal["正面/整体", "侧面/整体", "底面/整体", "穿搭", "局部细节", "外侧logo", "内侧logo", "内部"]
    description: str


class DetailsCorrectionResult(BaseModel):
    details: str


def read_info_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_product_info_texts(product_dir: str) -> tuple[str, str]:
    info_cn_path = os.path.join(product_dir, "info_cn.txt")
    info_en_path = os.path.join(product_dir, "info_en.txt")
    legacy_info_path = os.path.join(product_dir, "info.txt")

    if os.path.exists(info_cn_path) and os.path.exists(info_en_path):
        info_cn_text = read_info_file(info_cn_path)
        info_en_text = read_info_file(info_en_path)
        print(f"已读取中文商品信息: {info_cn_path}")
        print(f"已读取英文商品信息: {info_en_path}")
        return info_cn_text, info_en_text

    if os.path.exists(legacy_info_path):
        info_cn_text = read_info_file(legacy_info_path)
        print(f"已读取兼容商品信息: {legacy_info_path}")
        return info_cn_text, ""

    raise FileNotFoundError(
        f"未找到商品信息文件，期望存在 {info_cn_path} 与 {info_en_path}，或兼容文件 {legacy_info_path}。"
    )


def image_to_data_url(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def normalize_image_part(part_text: str, description_text: str = "") -> str:
    text = f"{part_text} {description_text}".strip()
    if not text:
        return "正面/整体"

    rules = [
        ("外侧logo", ["外侧logo", "外部logo", "外侧标志", "外侧标识", "外部标志"]),
        ("内侧logo", ["内侧logo", "内部logo", "内里logo", "内侧标志", "内侧标识"]),
        ("底面/整体", ["底面", "包底", "鞋底"]),
        ("内部", ["内部", "内里", "包内", "内衬", "里侧", "打开后"]),
        ("穿搭", ["穿搭", "上身", "模特", "肩背", "斜挎", "手提展示", "背上效果"]),
        ("侧面/整体", ["侧面", "侧边", "侧视"]),
        ("正面/整体", ["正面", "前面", "包身正面", "整体", "全貌", "背面", "背部", "后面", "反面"]),
        ("局部细节", ["细节", "特写", "近景", "五金", "拉链", "边角", "纹理", "刻印", "logo"]),
    ]

    for standard_part, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return standard_part

    return "正面/整体"


def build_default_llm() -> ChatOpenAI:
    if not qwen_api_key or not qwen_api_key.strip():
        raise ValueError("未读取到 QWEN_API_KEY，请先在 .env 中配置。")

    return ChatOpenAI(
        api_key=qwen_api_key,
        model=qwen_model,
        base_url=qwen_url,
        temperature=0,
        extra_body={"enable_thinking": False},
    )


def generate_image_description(image_path: str, llm_model) -> dict:
    structured_image_llm = llm_model.with_structured_output(ImageDescriptionResult, method="json_schema")
    image_data_url = image_to_data_url(image_path)
    messages = [
        SystemMessage(
            content=(
                "你是一个商品图片分析助手。"
                "请基于图片内容识别商品展示部位，并生成简洁准确的中文描述。"
                f"part 只能从以下集合中选择：{', '.join(ALLOWED_IMAGE_PARTS)}。"
                "必须只返回一个 JSON 对象，格式为 "
                '{"part": "...", "description": "..."}。'
            )
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": "请分析这张商品图片，输出标准部位和详细描述。"},
                {"type": "image_url", "image_url": image_data_url},
            ]
        ),
    ]

    try:
        result = structured_image_llm.invoke(messages)
        if isinstance(result, ImageDescriptionResult):
            part = normalize_image_part(result.part, result.description)
            description = result.description.strip()
        elif isinstance(result, dict):
            description = str(result.get("description", "")).strip()
            part = normalize_image_part(str(result.get("part", "")), description)
        else:
            raw_content = result.content if hasattr(result, "content") else str(result)
            description = raw_content.strip()
            part = normalize_image_part("", description)
    except Exception:
        fallback_result = llm_model.invoke(messages)
        description = (
            fallback_result.content.strip()
            if hasattr(fallback_result, "content")
            else str(fallback_result).strip()
        )
        part = normalize_image_part("", description)

    print(f"分析结果 - 部位: {part}, 描述: {description}")
    return {"part": part, "description": description}


def correct_description_details(info_cn_text: str, info_en_text: str, content: dict, llm_model) -> str:
    structured_llm = llm_model.with_structured_output(DetailsCorrectionResult, method="json_schema")
    product_name = str(content.get("product_name", "")).strip()
    current_details = str(content.get("description", {}).get("details", "")).strip()

    messages = [
        SystemMessage(
            content=(
                "你是奢侈品商品文案校对助手。"
                "你的任务是只修正 description.details 的中文翻译错误。"
                "不要修改product_name字段，也不要修改 description 中的其他字段。"
                "必须优先参考 product_name 中的原文英文名称，不要把关键商品名误译成别的中文。"
                "请同时参考中文与英文信息，出现冲突时以两者一致信息为准，避免臆造。"
                "输出要自然、准确、简洁。"
                '只返回一个 JSON 对象，格式为 {"details": "..."}。'
            )
        ),
        HumanMessage(
            content=f"""
请根据以下信息，修正商品描述 details 中的翻译错误。

要求：
1. 只修改 description.details
2. 优先使用 product_name 的原文来纠正商品名称、系列名称
3. 输出中文，但商品名要用"中文（原文）"格式，例如"古驰玛蒙特迷你链条包（Gucci Marmont Matelassé）"
4. 不要补充 info_cn.txt 或 info_en.txt 中没有出现的事实

product_name:
{product_name}

当前 details（中文）:
{current_details}

info_cn.txt 原文:
{info_cn_text}

info_en.txt 原文:
{info_en_text}
""".strip()
        ),
    ]

    try:
        result = structured_llm.invoke(messages)
        if isinstance(result, DetailsCorrectionResult):
            return result.details.strip()
        if isinstance(result, dict):
            return str(result.get("details", "")).strip() or current_details
        raw_content = result.content if hasattr(result, "content") else str(result)
        return DetailsCorrectionResult(**json.loads(raw_content)).details.strip()
    except Exception:
        return current_details


def text_to_json(info_cn_text: str, info_en_text: str, image_descriptions: dict, llm_model) -> dict:
    schema = get_annotation_schema()
    structured_llm = llm_model.with_structured_output(Annotation, method="json_schema")

    messages = [
        SystemMessage(
            content=(
                "你是一个商品标注助手。"
                "请严格按照给定的 JSON Schema 生成结果。"
                "输入里会同时提供中文与英文商品信息，请综合两者进行理解。"
                "输出的所有文本字段必须为中文（品牌原文可保留在括号中）。"
                "必须只返回一个 JSON 对象，不能返回解释、Markdown、代码块或额外文本。"
            )
        ),
        HumanMessage(
            content=f"""
请根据以下商品信息和图片描述生成标注数据。

要求：
1. 输出必须严格符合给定 schema
2. 所有字段都必须返回
3. 字符串字段缺失时填写空字符串 ""
4. 数组字段缺失时填写 []
5. 数值字段缺失时填写 0
6. `images` 字段中需要包含文件名、部位和对应描述，且 `part` 只能从以下集合中选择：{", ".join(ALLOWED_IMAGE_PARTS)}
7. 只返回 JSON 对象

JSON Schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

中文商品信息（info_cn.txt）:
{info_cn_text}

英文商品信息（info_en.txt）:
{info_en_text}

图片描述:
{json.dumps(image_descriptions, ensure_ascii=False, indent=2)}
""".strip()
        ),
    ]

    try:
        result = structured_llm.invoke(messages)
        if isinstance(result, Annotation):
            content = result.model_dump()
        elif isinstance(result, dict):
            content = Annotation(**result).model_dump()
        else:
            raw_content = result.content if hasattr(result, "content") else str(result)
            content = Annotation(**json.loads(raw_content)).model_dump()
    except Exception as exc:
        raise ValueError(f"LLM 输出不是符合 schema 的 JSON: {exc}") from exc

    for image in content.get("images", []):
        image["part"] = normalize_image_part(image.get("part", ""), image.get("description", ""))

    if "description" in content:
        content["description"]["details"] = correct_description_details(info_cn_text, info_en_text, content, llm_model)

    if not validate_annotation(content):
        raise ValueError("LLM 输出格式错误，需要重新生成。")

    print("LLM 输出格式正确")
    return content


def process_product_dir(product_dir: str, llm_model) -> dict:
    product_dir = os.path.abspath(product_dir)
    output_path = os.path.join(product_dir, "annotation.json")

    if not os.path.isdir(product_dir):
        raise FileNotFoundError(f"商品目录不存在: {product_dir}")

    info_cn_text, info_en_text = read_product_info_texts(product_dir)

    image_files = sorted(
        f for f in os.listdir(product_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    )
    print(f"找到 {len(image_files)} 张图片")

    image_descriptions = {}
    for image_file in image_files:
        image_path = os.path.join(product_dir, image_file)
        print(f"正在分析图片: {image_file}")
        image_descriptions[image_file] = generate_image_description(image_path, llm_model)

    print("正在生成 JSON...")
    json_output = text_to_json(info_cn_text, info_en_text, image_descriptions, llm_model)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    print(f"JSON 标注文件已保存到: {output_path}")
    return json_output


def main() -> None:
    process_product_dir(sample_dir, build_default_llm())


if __name__ == "__main__":
    main()
