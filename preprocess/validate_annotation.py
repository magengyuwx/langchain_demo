from pydantic import BaseModel, Field
from typing import List, Literal


class Price(BaseModel):
    USD: float
    CNY: float


class Condition(BaseModel):
    grade: str
    exterior: str
    hardware: str
    interior: str
    handle: str
    accessories: List[str]


class Description(BaseModel):
    item_number: str
    material: str
    details: str
    authenticity: str


class Dimensions(BaseModel):
    base_length: str
    max_length: str
    height: str
    width: str
    drop_length: str


class Image(BaseModel):
    filename: str
    part: Literal["正面", "背面", "穿搭", "细节", "外侧logo", "内侧logo", "内部", "底部"] = Field(
        description="图片部位，必须从固定集合中选择"
    )
    description: str


class Annotation(BaseModel):
    brand: str
    product_name: str
    price: Price
    retail_price: str
    condition: Condition
    description: Description
    dimensions: Dimensions
    images: List[Image]


def validate_annotation(data: dict) -> bool:
    """
    验证数据是否符合Annotation格式
    
    Args:
        data: 要验证的字典数据
        
    Returns:
        bool: 验证是否成功
    """
    try:
        # 尝试解析数据
        annotation = Annotation(**data)
        print("✓ 数据验证成功！")
        return True
    except Exception as e:
        print(f"✗ 数据验证失败: {e}")
        return False


def get_annotation_schema() -> dict:
    """
    获取Annotation的JSON Schema描述，用于约束LLM输出
    
    Returns:
        dict: JSON Schema描述
    """
    schema = Annotation.model_json_schema()
    return schema


if __name__ == "__main__":
    # 示例使用
    import json
    
    # 从文件加载数据
    with open('annotation.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 验证数据
    validate_annotation(data)
    
    # 打印Schema（可用于提示LLM）
    print("\nAnnotation JSON Schema:")
    print(json.dumps(get_annotation_schema(), ensure_ascii=False, indent=2))