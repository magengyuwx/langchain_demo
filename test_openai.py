from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import HumanMessage, SystemMessage  

api_key = os.getenv("OPENROUTER_API_KEY")
print("API 密钥读取成功：", api_key is not None)

model = ChatOpenAI(  
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="z-ai/glm-4.5-air:free",
)


messages = [
    SystemMessage(content="Translate the following from English into Chinese"),
    HumanMessage(content="hi!"),
]  
  
from langchain_core.output_parsers import StrOutputParser  
  
parser = StrOutputParser()  
result = model.invoke(messages)  
output = parser.invoke(result)  
print("翻译结果：", output)