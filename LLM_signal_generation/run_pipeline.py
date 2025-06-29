'''
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
'''

import os
from typing import List, Dict
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from LLM_signal_generation.preprocess import classify_with_emotion_and_semantics
from LLM_signal_generation.embedder import SentenceEmbedder
from LLM_signal_generation.aggregator import aggregate_scores_by_firm_time
from LLM_signal_generation.scorer import Scorer
from util import load_dimension_examples_from_files, smooth_minmax_scaling_symmetric

# ✅ 初始化嵌入器与嵌入模型
embedder = SentenceEmbedder()
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ 加载样例 & 初始化 Scorer
base_dir = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(base_dir, "two-side emotional examples")
DIMENSION_EXAMPLES = load_dimension_examples_from_files(example_dir)
scorer = Scorer(model, DIMENSION_EXAMPLES)

# ✅ 核心处理函数
def run_pipeline(records: List[Dict]) -> pd.DataFrame:
    sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    firm_date_scores = {}

    for record in records:
        firm_id = record["firm_id"]
        raw_date = record["date"]
        # date = raw_date[:7]
        date = raw_date.strip()
        text = record["text"]

        firm_date_key = (firm_id, date)
        if firm_date_key not in firm_date_scores:
            firm_date_scores[firm_date_key] = {}

        print("\n📝 Input Text:", text)
        embedding = embedder.encode([text])[0]
        print("📐 Embedding vector (shape):", embedding.shape)

        score_all_result = scorer.score_all(text)
        score_ranked_result = scorer.score_ranked(text)

        for dim in score_all_result:
            val_all = score_all_result[dim][0][1]
            val_ranked = score_ranked_result[dim][0][1]
            blended_score = 0.6 * val_all + 0.4 * val_ranked

            if dim not in firm_date_scores[firm_date_key]:
                firm_date_scores[firm_date_key][dim] = {"total_score": 0.0, "count": 0}

            firm_date_scores[firm_date_key][dim]["total_score"] += blended_score
            firm_date_scores[firm_date_key][dim]["count"] += 1

    # ✅ 新聚合逻辑，仅输出5个维度（正 - 负 → 平滑）
    output_data = []
    paired_dimensions = [
        ("brand_positive", "brand_negative", "brand"),
        ("reputation_positive", "reputation_negative", "reputation"),
        ("crypto_positive", "crypto_negative", "crypto"),
        ("patent_positive", "patent_negative", "patent"),
        ("executive_positive", "executive_negative", "executive"),
    ]

    for (firm_id, date), dim_scores in firm_date_scores.items():
        row = {"firm_id": firm_id, "date": date}
        raw_diffs = []

        for pos_dim, neg_dim, final_dim in paired_dimensions:
            pos_score = dim_scores.get(pos_dim, {"total_score": 0.0})["total_score"]
            neg_score = dim_scores.get(neg_dim, {"total_score": 0.0})["total_score"]
            raw_diffs.append(pos_score - neg_score)

        # 平滑到 [-1, 1]
        scaled_diffs = smooth_minmax_scaling_symmetric(raw_diffs)

        for (_, _, final_dim), score in zip(paired_dimensions, scaled_diffs):
            row[final_dim] = score

        output_data.append(row)

    df = pd.DataFrame(output_data)
    print("\n📊 Aggregated Result (5维度):")
    print(df)
    return df

# ✅ 追加或更新已有 CSV（按 firm_id + date 去重）
def update_csv_with_result(new_df: pd.DataFrame, csv_path: str = "output/aggregated_case_month.csv"):
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        merged_df = pd.concat([existing_df, new_df])
        merged_df = merged_df.drop_duplicates(subset=["firm_id", "date"], keep="last")
    else:
        merged_df = new_df

    merged_df.to_csv(csv_path, index=False)
    print(f"✅ 已保存并更新: {csv_path}")

# ✅ 测试运行主程序
if __name__ == "__main__":
    # folder_path = os.path.join(base_dir, "grasp_data/output/aggregate")
    folder_path = os.path.join(base_dir, "../case/output/days")

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            print(f"\n📄 正在处理文件: {filename}")
            filepath = os.path.join(folder_path, filename)

            try:
                df = pd.read_csv(filepath)
                example_input = df.to_dict(orient="records")

                df_result = run_pipeline(example_input)
                update_csv_with_result(df_result)

            except Exception as e:
                print(f"❌ 文件处理失败: {filename}, 错误：{e}")

    # example_df = pd.read_csv("grasp_data/output/walmart/Walmart 12 2017.txt")  # ⚠️ 只含一个公司一个月的数据
    # example_input = example_df.to_dict(orient="records")
    #
    # df_result = run_pipeline(example_input)
    # update_csv_with_result(df_result)

    '''
    自定义权重比例是关于不同scorer算法的结果接受比例，这调节这个来得到不同的语句得分呈现效果！！！
    根据数量的结果比例可以调节来改变不同维度间的差距！！！！
    '''