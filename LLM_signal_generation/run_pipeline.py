# 功能：组织全流程，输入原始记录，输出 R_{i,t}^{(k)} 聚合结果

from LLM_signal_generation.preprocess import classify_with_emotion_and_semantics

from LLM_signal_generation.embedder import SentenceEmbedder
from LLM_signal_generation.aggregator import aggregate_scores_by_firm_time
from LLM_signal_generation.scorer import Scorer

from LLM_signal_generation.generate_test_cases import generate_test_records
from util import load_dimension_examples_from_files

from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
import os
from typing import List, Dict

# ✅ 初始化嵌入器（用于句子向量提取）
embedder = SentenceEmbedder()

# ✅ 加载维度样例（按文本文件管理）
base_dir = os.path.dirname(os.path.abspath(__file__))  # 获取 run_valuation.py 所在目录
example_dir = os.path.join(base_dir, "scorer_examples")
DIMENSION_EXAMPLES = load_dimension_examples_from_files(example_dir)

# ✅ 初始化嵌入模型（MiniLM 或可换 FinBERT）
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ 初始化 Scorer（一次性编码 + 缓存）
scorer = Scorer(model, DIMENSION_EXAMPLES)

def run_pipeline(records: List[Dict]) -> pd.DataFrame:
    """
    核心执行函数：
    输入原始文本数据，按能力维度分类并调用评分函数 → 汇总打分 → 聚合输出能力表

    参数：
        records: List of dicts，结构如下：
            {
                'firm_id': str,
                'date': str,
                'text': str
            }

    返回：
        pd.DataFrame: 聚合结果，每行是公司/时间，列是各能力维度的评分值
    """
    scored_items = []

    for record in records:
        firm_id = record["firm_id"]
        date = record["date"]
        text = record["text"]

        print("\n📝 Input Text:", text)

        # Step 1: 获取语义向量表示（可选）
        embedding = embedder.encode([text])[0]
        print("📐 Embedding vector (shape):", embedding.shape)

        # Step 2: 分类该句子属于哪种能力维度（分类器输出）
        # dimension = classify_capability_dimension(text)
        sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        dimension = classify_with_emotion_and_semantics(text, scorer, sentiment_model)
        print("🏷️  Classified Dimension:", dimension)

        # Step 3: 直接调用 scorer 评分（按分类维度匹配）
        score_result = scorer.score_all(text, top_k=1)

        # Step 4: 获取该维度下的相似度
        score = score_result.get(dimension, [(None, 0.0)])[0][1]
        print(f"🎯 Score for {dimension}: {score:.4f}")

        # Step 5: 整理为结构化记录
        scored_items.append({
            "firm_id": firm_id,
            "date": date,
            "score_name": dimension,
            "score_value": score
        })

    # Step 6: 聚合结果
    df = aggregate_scores_by_firm_time(scored_items)
    print("\n📊 Aggregated Result:")
    print(df)

    return df


# ✅ 示例测试
if __name__ == "__main__":
    # example_input = generate_test_records(n=100)
    # df_result = run_pipeline(example_input)
    # print("\n✅ Pipeline finished.")

    example_df = pd.read_csv("input/example.csv")
    example_input = example_df.to_dict(orient="records")
    df_result = run_pipeline(example_input)
    df_result.to_csv("output/aggregated_result.csv", index=False)

    print("\n✅ Pipeline finished.")