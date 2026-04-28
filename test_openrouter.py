from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
print("API 密钥读取成功：", api_key is not None)

# Initialize the OpenRouter LLM
llm = ChatOpenRouter(
    model="z-ai/glm-4.5-air:free",
    temperature=0.7,
)

# Simple message
response = llm.invoke("Hello, how are you?")
print(response.content)

# 输出json格式的响应
response = llm.invoke("What is the capital of France?", response_format="json")
print(response.content)


# 多轮问答
messages = [
    HumanMessage(content="你好，今天天气怎么样？"), 
    AIMessage(content="今天天气晴朗，适合外出。"), 
    HumanMessage(content="那我应该穿什么？")
]
response = llm.invoke(messages)
print(response.content)

# 使用streaming功能
print("\nStreaming output:")
for chunk in llm.stream("Tell me a short story."):
    print(chunk.content, end="")
print()  # 最后添加一个换行
