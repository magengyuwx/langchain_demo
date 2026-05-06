# 数据集生成流程说明

本项目的数据集生成分为 3 个阶段：

1. 先生成每张图像的描述
2. 再生成每个部件的等级/痕迹描述
3. 最终生成整个商品的整体等级

---

## 阶段一：图像级标注（部位 + 图像描述）

脚本：`preprocess/txt2json.py`

目标：
- 对商品目录中的每张图片进行分析
- 生成该图片对应的 `part`（部位）和 `description`（图像描述）

输出：
- 每个商品目录下的 `annotation.json`
- 其中 `images` 字段包含：
	- `filename`
	- `part`
	- `description`

这一阶段的产物是后续两个阶段的基础输入。

---

## 阶段二：部件级样本（每个部件的痕迹）

脚本：`dataset/generate_partgrad_label.py`

输入来源：
- `annotation.json`
- 重点使用：
	- `product_name`
	- `condition`（排除 `grade` 与 `accessories` 后的部件项）
	- `images` 中的 `part` 与 `description`

处理逻辑：
- 对每个商品的每个部件（如 exterior/hardware/interior 等）分别构造样本
- 按图片 `part` 与 `description` 对该部件做匹配打分
- 选择最匹配的 1~3 张图用于该部件样本
- 用户输入中包含商品名称
- 助手输出为该部件在 `condition` 中的痕迹描述

输出：
- `part_grade_label.jsonl`
- 每条样本为 LLaMA-Factory 友好格式（仅 `messages` 与 `images`）

---

## 阶段三：商品级样本（整体描述 + grade）

脚本：`dataset/generate_productgrad_label.py`

输入来源：
- `annotation.json`
- 包含：
	- 商品名称
	- `part_condition`（即 `condition` 中除 `grade` 外所有项目，包含附件）
	- 每张图片的描述

处理逻辑：
- 将上述信息拼接为纯文本 user message（纯 LLM，不使用图片输入）
- assistant 输出结构化 JSON：
	- `overall_description`
	- `grade`

输出：
- `product_grade_label.jsonl`

---

## 总结

完整链路为：

1. `preprocess/txt2json.py`：先把每张图像转成可训练的图像描述信息
2. `dataset/generate_partgrad_label.py`：再把图像信息聚合到部件级痕迹样本
3. `dataset/generate_productgrad_label.py`：最后汇总为整商品的整体等级样本

即：图像级 -> 部件级 -> 商品级。
