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


example:
| 维度             | 核心内容（anchor覆盖）               | 关键词/表达类型                                                               | 情绪特征                       | 示例摘要                                                                                                            |
| -------------- | ---------------------------- | ---------------------------------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Patent**     | 技术创新、专利申请、知识产权保护、研发成果披露      | patent, innovation, filing, intellectual property, proprietary         | 中性为主，部分积极（突破、授权）           | "The company filed 15 new battery-related patents this year."；"Their proprietary process is now protected."     |
| **Reputation** | 公共情绪、市场信任、品牌评价、投资人反应         | sentiment, trust, confidence, criticism, praise                        | 正负情绪强                      | "Investor confidence has fallen."；"Public sentiment toward the brand is increasingly favorable."                |
| **Executive**  | 高管行动、战略决策、组织调整、未来承诺          | CEO, launch, commit, initiative, leadership, transformation            | 多为正向（计划、承诺），但也涵盖负面（离职等）    | "The company committed to digital transformation."；"Leadership transition raised concern."                      |
| **Crypto**     | 区块链应用、token发布、智能合约、DeFi/NFT等 | crypto, blockchain, token, Ethereum, wallet, custody, decentralization | 多为技术中性或正面（兴奋、创新），也有监管担忧等负面 | "They launched a tokenized loyalty platform."；"The firm received regulatory clearance for its crypto exchange." |
| **Brand**      | 品牌认知、可见性、社交媒体声量、视觉识别、定位策略    | brand awareness, Google Trends, visual identity, loyalty, influencer   | 情绪多为正向（提升、认同），也有危机与波动      | "Brand loyalty metrics have improved."；"The refreshed brand identity received praise."                          |

| 维度             | 典型语义边界     | 易重叠维度                        | 重叠情形举例                                                                   |
| -------------- | ---------- | ---------------------------- | ------------------------------------------------------------------------ |
| **Patent**     | 技术性、法律保护导向 | executive, reputation        | "The CEO announced a new patented process."（executive + patent）          |
| **Reputation** | 舆论导向、市场观感  | brand, executive             | "Public confidence rose after the CEO’s speech."（executive + reputation） |
| **Executive**  | 主体为决策者行为   | patent, reputation           | "Committed to innovation."（可能也匹配 patent）                                 |
| **Crypto**     | 区块链技术行为表达  | executive (决策引导), brand (营销) | "They launched a crypto brand experience."                               |
| **Brand**      | 市场感知与身份    | reputation, executive        | "The rebranding improved public perception."                             |


               Reputation
                /     \
         Brand         Executive
            \         /
             \       /
              Patent      Crypto
