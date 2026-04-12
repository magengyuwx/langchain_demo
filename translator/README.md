# 长篇小说翻译工作流（LangChain）

这是一个面向**长篇小说**的翻译脚手架，核心流程包括：

1. **原文切章**：将整本小说按章节拆分并保存。
2. **信息提取**：从原文提取`小说大纲`与`人物设定`。
3. **RAG 向量检索**：为以下内容建立向量索引：
   - 原文全文
   - 原文各章节
   - 小说大纲
   - 人物设定
   - 每章中文翻译
4. **逐章翻译**：翻译每一章时，自动召回`上一章中文译文 + 大纲 + 人物设定 + 相关原文片段`，提升全局一致性。

---

## 目录结构

```text
translator/
├─ main.py
├─ requirements.txt
├─ .env.example
├─ novel_translator/
│  ├─ config.py
│  ├─ splitter.py
│  ├─ rag.py
│  ├─ extractor.py
│  ├─ translator.py
│  └─ workflow.py
└─ data/
   ├─ input/               # 放原始小说 txt
   ├─ output/
   │  ├─ source/           # 原文归档
   │  ├─ chapters/source/  # 拆分后的英文章节
   │  ├─ chapters/zh/      # 中文译文
   │  └─ metadata/         # 大纲、人物设定
   └─ rag_db/              # Chroma 向量库
```

---

## 环境准备

```bash
conda activate langchain
pip install -r requirements.txt
copy .env.example .env
```

在 `.env` 中至少补充。若使用 Ollama，可直接写成：

```env
LLM_PROVIDER=ollama
CHAT_MODEL=kimi-k2.5:cloud
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

> 如果你使用 `OpenAI` / `OpenRouter`，再配置对应的 `API_KEY` 与 `OPENAI_BASE_URL` 即可。

---

## 使用方式

### 1) 放入原文

将原始小说文本放到：

```text
data/input/novel.txt
```

或者运行时通过 `--input` 指定路径。当前支持：`txt`、`md`、`pdf`、`epub`。
例如仓库内样例：

```bash
python main.py all --input "samples/Code Name Verity.epub" --max-chapters 1
```

### 2) 先做预处理

```bash
python main.py prepare --input data/input/novel.txt
```

会完成：
- 原文归档
- 章节拆分
- 大纲提取
- 人物设定提取
- RAG 向量化

### 3) 再逐章翻译

```bash
python main.py translate --input data/input/novel.txt
```

调试时建议先限制章节数：

```bash
python main.py translate --input data/input/novel.txt --max-chapters 3
```

### 4) 一键全流程

```bash
python main.py all --input data/input/novel.txt
```

---

## 模块说明

| 模块 | 作用 |
| --- | --- |
| `splitter.py` | 识别章节标题并拆分长篇原文 |
| `extractor.py` | 从原文中抽取小说大纲与人物设定 |
| `rag.py` | 构建 Chroma 向量库，管理检索上下文 |
| `translator.py` | 基于 RAG 上下文逐章翻译为中文 |
| `workflow.py` | 串联预处理、向量化与翻译全过程 |

---

## 测试运行

### 统一执行命令（discover）

```bash
conda activate langchain
python -m unittest discover -s unit_tests -p "test_*.py" -v
```

### 测试运行清单

- `unit_tests/test_config.py`：配置读取、路径构造与目录创建
- `unit_tests/test_loader.py`：文件读取/保存/复制与异常分支
- `unit_tests/test_splitter.py`：章节拆分、fallback、章节落盘回读
- `unit_tests/test_extractor.py`：分段抽取与大纲/人物设定解析（mock）
- `unit_tests/test_translator.py`：翻译分段拼装与文本规范化（mock）
- `unit_tests/test_rag.py`：向量检索过滤与上下文拼装（mock）
- `unit_tests/test_workflow.py`：prepare/translate 主流程（mock）
- `unit_tests/test_llm_factory.py`：真实 Ollama 调用验证（integration 风格）

### 说明

- 运行 `test_llm_factory.py` 前，请确认本地 Ollama 服务已启动，且模型已可用。
- 如果仅想执行纯 mock 测试，可临时排除 `test_llm_factory.py` 单独运行。

---

## 建议

- 长篇小说建议先使用 `--max-chapters` 做小规模验证。
- 若你想断点续跑，已有中文章节会默认跳过；加 `--force` 可重译。
- 如果不同章节存在固定称谓、人名译法偏好，可以把这些规则补充到 `data/output/metadata/characters.md` 后再次运行翻译。
