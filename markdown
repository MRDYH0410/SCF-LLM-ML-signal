data/                       ← 存放数据
src/
├── preprocess.py           ← 执行文本清洗 + 实体识别（英文）
├── embedder.py             ← 提取句子语义向量（Huggingface模型）
├── scorer.py               ← 基于嵌入和关键词设计评分函数（如 reputation score）
├── aggregator.py           ← 句子或段落得分聚合为 R_{i,t}^{(k)}
├── run_pipeline.py         ← 主程序，组织以上各模块运行我标注了我现在的构架，这样的构架是否满足3.1的要求


| 3.1方法论环节            | 你的模块              | 说明                                     |
| ------------------- | ----------------- | -------------------------------------- |
| 文本清洗                | `preprocess.py`   | 可包含清理 HTML、分词前处理、保留目标句                 |
| 实体识别（NER）           | `preprocess.py`   | 使用 Huggingface 英文NER模型识别 ORG/PER       |
| 语义表示（Transformer嵌入） | `embedder.py`     | 使用BERT系列模型提取 `[CLS]` 或 mean vector     |
| 语义得分（按能力维度）         | `scorer.py`       | 针对每个 signal 维度提取句子分数，如情感极性、承诺意图        |
| 能力信号聚合              | `aggregator.py`   | 将多个句子的得分聚合为 firm-level $R^{(k)}_{i,t}$ |
| 跨模块运行流程             | `run_pipeline.py` | 控制输入 → 清洗 → 嵌入 → 评分 → 聚合 → 输出全过程       |
