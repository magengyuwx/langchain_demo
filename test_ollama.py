import os

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()

ollama_model = os.getenv("OLLAMA_MODEL", "kimi-k2.5:cloud")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 初始化本地 Ollama 模型
llm = OllamaLLM(
    model=ollama_model,
    base_url=ollama_base_url,
)

# 调用
for chunk in llm.stream("请用中文简单介绍AI"):
    print(chunk, end="", flush=True)
print()  # 输出换行