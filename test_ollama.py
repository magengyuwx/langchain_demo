from langchain_ollama import OllamaLLM

# 初始化本地 Ollama 模型
llm = OllamaLLM(
    model="kimi-k2.5:cloud",   # 你本地有的模型
    base_url="http://localhost:11434"  # 默认地址
)

# 调用
for chunk in llm.stream("请用中文简单介绍AI"):
    print(chunk, end="", flush=True)
print()  # 输出换行