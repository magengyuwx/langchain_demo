import os
import json
import base64
import mimetypes
from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

try:
    from validate_annotation import Annotation, get_annotation_schema, validate_annotation
except ImportError:
    from .validate_annotation import Annotation, get_annotation_schema, validate_annotation

# 配置本地 Ollama 模型
llm = ChatOllama(
    model="kimi-k2.5:cloud",
    base_url="http://localhost:11434",
    temperature=0,
)

# 定义路径
base_dir = os.path.dirname(os.path.abspath(__file__))
sample_dir = os.path.join(base_dir, "sample")
info_file = os.path.join(sample_dir, "info.txt")
output_file = os.path.join(sample_dir, "annotation.json")

ALLOWED_IMAGE_PARTS = ("正面", "背面", "穿搭", "细节", "外侧logo", "内侧logo", "内部", "底部")


class ImageDescriptionResult(BaseModel):
    part: Literal["正面", "背面", "穿搭", "细节", "外侧logo", "内侧logo", "内部", "底部"]
    description: str


# 读取info.txt文件
def read_info_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def image_to_data_url(image_path: str) -> str:
    """将本地图片转成 base64 data URL，供多模态模型直接读取。"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def normalize_image_part(part_text: str, description_text: str = "") -> str:
    """将图片部位归一化到 schema 允许的固定枚举值。"""
    text = f"{part_text} {description_text}".strip()
    if not text:
        return "细节"

    rules = [
        ("外侧logo", ["外侧logo", "外部logo", "外侧标志", "外侧标识", "外部标志"]),
        ("内侧logo", ["内侧logo", "内部logo", "内里logo", "内侧标志", "内侧标识"]),
        ("底部", ["底部", "底面", "包底", "鞋底"]),
        ("内部", ["内部", "内里", "包内", "内衬", "里侧", "打开后"]),
        ("穿搭", ["穿搭", "上身", "模特", "肩背", "斜挎", "手提展示", "背上效果"]),
        ("背面", ["背面", "背部", "后面", "反面"]),
        ("正面", ["正面", "前面", "包身正面", "整体", "全貌"]),
        ("细节", ["细节", "特写", "近景", "五金", "拉链", "边角", "纹理", "刻印", "logo"]),
    ]

    for standard_part, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return standard_part

    return "细节"


# 使用VLM生成图像描述
def generate_image_description(image_path):
    structured_image_llm = llm.with_structured_output(ImageDescriptionResult, method="json_schema")
    image_data_url = image_to_data_url(image_path)
    messages = [
        SystemMessage(content=(
            "你是一个商品图片分析助手。"
            "我已经直接提供了图片内容，不是只给你路径。"
            "请基于图片内容识别商品展示部位并生成详细描述。"
            "part 只能从以下集合中选值：正面、背面、穿搭、细节、外侧logo、内侧logo、内部、底部。"
            "必须只返回一个 JSON 对象，格式为 {\"part\": \"...\", \"description\": \"...\"}。"
        )),
        HumanMessage(content=[
            {"type": "text", "text": "请分析这张商品图片，输出标准部位和详细描述。"},
            {"type": "image_url", "image_url": image_data_url},
        ])
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
        fallback_result = llm.invoke(messages)
        description = fallback_result.content.strip() if hasattr(fallback_result, "content") else str(fallback_result).strip()
        part = normalize_image_part("", description)

    print(f"分析结果 - 部位: {part}, 描述: {description}")
    return {
        "part": part,
        "description": description
    }


# 使用LLM将文本转换为JSON
def text_to_json(text, image_descriptions):
    schema = get_annotation_schema()
    structured_llm = llm.with_structured_output(Annotation, method="json_schema")

    messages = [
        SystemMessage(content=(
            "你是一个商品标注助手。"
            "请严格根据给定的 JSON Schema 生成结果。"
            "必须只返回一个 JSON 对象，不能返回解释、Markdown、代码块或额外文本。"
        )),
        HumanMessage(content=f"""
请根据以下商品信息和图片描述生成标注数据。

要求：
1. 输出必须严格符合给定 schema
2. 所有字段都必须返回
3. 字符串字段缺失时填写空字符串 ""
4. 数组字段缺失时填写 []
5. 数值字段缺失时填写 0
6. `images` 字段中需要包含文件名、部位和对应描述，且 `part` 只能从以下集合中选值："正面"、"背面"、"穿搭"、"细节"、"外侧logo"、"内侧logo"、"内部"、"底部"
7. 只返回 JSON 对象

JSON Schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

商品信息:
{text}

图片描述:
{json.dumps(image_descriptions, ensure_ascii=False, indent=2)}
""".strip())
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
        raise ValueError(f"LLM 输出不是符合 schema 的 JSON：{exc}") from exc

    for image in content.get("images", []):
        image["part"] = normalize_image_part(image.get("part", ""), image.get("description", ""))

    if not validate_annotation(content):
        raise ValueError("LLM输出格式错误，需要重新生成！")

    print("LLM输出格式正确！")
    return content


# 主函数
def main():
    # 读取商品信息
    info_text = read_info_file(info_file)
    print("已读取商品信息")

    # 收集图片文件
    image_files = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')]
    print(f"找到 {len(image_files)} 张图片")

    # 生成图片描述
    image_descriptions = {}
    for image_file in image_files:
        image_path = os.path.join(sample_dir, image_file)
        print(f"正在分析图片: {image_file}")
        description = generate_image_description(image_path)
        image_descriptions[image_file] = description
    
    # 生成JSON
    print("正在生成JSON...")
    json_output = text_to_json(info_text, image_descriptions)
    
    # 保存JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    print(f"JSON标注文件已保存到: {output_file}")


if __name__ == "__main__":
    main()