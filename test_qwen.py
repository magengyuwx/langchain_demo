import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()

api_key = os.getenv("QWEN_API_KEY")
has_api_key = bool(api_key and api_key.strip())
print("QWEN_API_KEY 已配置：", has_api_key)

if not has_api_key:
    raise ValueError("未读取到 QWEN_API_KEY，请先在 .env 中配置。")

model = ChatOpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.2,
)

messages = [
    SystemMessage(content="你是一个简洁的助手。"),
    HumanMessage(content="请只回复“Qwen API 调用成功”。"),
]

response = model.invoke(messages)
print("模型返回：", response.content)
