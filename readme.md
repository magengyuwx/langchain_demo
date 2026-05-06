# langchain_demo

本项目用于奢侈品商品数据处理，核心能力包括：

1. 将单个商品目录中的 `info_cn.txt + info_en.txt + 图片` 转为结构化 `annotation.json`
2. 批量处理多个商品目录
3. 提供本地网页工具查看和编辑 `annotation.json`
4. 将标注数据导出为 VLM 训练格式（`jsonl + 图片`）

---

## 1. 项目结构

```text
langchain_demo/
├─ preprocess/
│  ├─ txt2json.py              # 单商品：info_cn.txt + info_en.txt + 图片 -> annotation.json（兼容旧 info.txt）
│  ├─ batch_process.py         # 批量处理商品目录
│  ├─ web_viewer.py            # 本地 Web 标注查看/编辑工具
│  ├─ validate_annotation.py   # annotation schema 与校验
│  └─ sample/                  # 示例数据
├─ dataset/
│  ├─ generate_list.py         # 生成包含 annotation.json 的子目录清单
│  ├─ generate_image_label.py  # 生成图像部位识别数据集（jsonl）
│  ├─ generate_price_label.py  # 预留（当前为空）
│  └─ generate_quality_label.py# 预留（当前为空）
├─ test_qwen.py
├─ test_openai.py
├─ test_openrouter.py
├─ test_ollama.py
└─ pyproject.toml
```

---

## 2. 环境准备

### 2.1 Python 版本

- Python `>=3.10`

### 2.2 安装依赖

当前 `pyproject.toml` 的 `dependencies` 为空，请手动安装运行所需包：

```bash
pip install -U pydantic python-dotenv langchain-core langchain-openai langchain-openrouter langchain-ollama
```

---

## 3. `.env` 配置

在项目根目录创建 `.env`。当前 `preprocess/txt2json.py` 默认走 Qwen：

```env
QWEN_API_KEY=你的key
QWEN_MODEL=qwen-plus
QWEN_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

如果需要测试本地 Ollama，可额外配置：

```env
OLLAMA_BASE_URL=http://192.168.18.213:11434
OLLAMA_MODEL=qwen3-vl
```

---

## 4. 数据目录约定

单个商品目录格式：

```text
某商品目录/
├─ info_cn.txt
├─ info_en.txt
├─ 01.jpg
├─ 02.jpg
├─ ...
└─ annotation.json  # 运行后生成
```

兼容说明：

- 若目录中只有旧版 `info.txt`，`txt2json.py` 仍可处理。
- 若存在 `info_cn.txt` 与 `info_en.txt`，会同时送给模型分析，并输出中文版 `annotation.json`。

`txt2json.py` 会读取目录中的图片扩展名：

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

---

## 5. 使用方式

### 5.1 单商品生成 annotation

脚本入口：`preprocess/txt2json.py`

默认会处理：`preprocess/sample/`

```bash
python preprocess/txt2json.py
```

输出文件：`preprocess/sample/annotation.json`

### 5.2 批量处理商品目录

脚本入口：`preprocess/batch_process.py`

默认目录变量（需按你机器路径修改）：

```python
products_dir = r"D:\git\LV-Dataset\products"
```

运行：

```bash
python preprocess/batch_process.py
```

处理完成后会在 `products_dir` 下生成：

- `product_list.json`：全部子目录清单
- `product_failed.json`：失败目录与错误信息（若有）

### 5.3 启动本地 Web 标注工具

脚本入口：`preprocess/web_viewer.py`

```bash
python preprocess/web_viewer.py --root D:\git\LV-Dataset\products --host 127.0.0.1 --port 8899
```

浏览器打开：`http://127.0.0.1:8899`

功能说明：

1. 浏览商品目录
2. 展示 `info.txt`、图片和 `annotation.json`（当前 Web 工具仍读取 `info.txt`）
3. 修改图片部位/描述后自动保存
4. 保存时自动备份旧文件为 `annotation.json.bak.YYYYMMDD_HHMMSS`

### 5.4 生成 VLM 图像部位数据集

流程：

1. 先生成子目录清单

```bash
python dataset/generate_list.py
```

2. 再导出训练数据

```bash
python dataset/generate_image_label.py
```

`generate_image_label.py` 默认读取：

- `products_dir = D:\git\LV-Dataset\products`
- `subdirs_file = annotation_subdirs.json`

输出到：

- `output_dir = D:\git\LV-Dataset\vlm_dataset`
- `data.jsonl`
- 复制后的图片文件

---

## 6. annotation 关键字段

`annotation.json` 由 `preprocess/validate_annotation.py` 定义，包含以下主要字段：

- `brand`
- `product_name`
- `price` (`USD`, `CNY`)
- `retail_price`
- `condition`
- `description`
- `dimensions`
- `images`（每张图包含 `filename`, `part`, `description`）

图片部位 `part` 允许值：

- `正面/整体`
- `侧面/整体`
- `底面/整体`
- `穿搭`
- `局部细节`
- `外侧logo`
- `内侧logo`
- `内部`

---

## 7. 测试脚本

根目录测试脚本用于快速验证不同模型连接：

- `python test_qwen.py`
- `python test_openai.py`
- `python test_openrouter.py`
- `python test_ollama.py`

说明：

- 这些脚本偏向连通性验证，不是完整单元测试。
- `test_openai.py` 目前也在使用 OpenRouter 的 `base_url`。

---

## 8. 已知事项

1. `dataset/generate_price_label.py` 与 `dataset/generate_quality_label.py` 当前为空实现。
2. 多个脚本中存在硬编码路径（如 `D:\git\LV-Dataset\products`），迁移环境时需要先修改。
3. 当前项目未集中维护依赖列表（`pyproject.toml` 中 `dependencies` 为空），建议后续补齐。

---

## 9. 建议的最小工作流

1. 配置 `.env`（至少 `QWEN_API_KEY`）
2. 准备商品目录（推荐 `info_cn.txt + info_en.txt + 图片`，兼容旧 `info.txt`）
3. 执行 `python preprocess/txt2json.py` 或 `python preprocess/batch_process.py`
4. 用 `python preprocess/web_viewer.py --root ...` 人工校对
5. 执行 `python dataset/generate_list.py` 与 `python dataset/generate_image_label.py` 导出训练数据
