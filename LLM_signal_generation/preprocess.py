# 提供两个主要功能：
# 1. extract_named_entities：使用 BERT 模型进行英文命名实体识别（NER）
# 2. classify_capability_dimension：根据文本内容推断其所属的数字能力维度（如声誉、专利、品牌等）

from transformers import pipeline
from typing import List, Union

# ✅ 设置英文NER任务的预训练模型
MODEL_NAME = "dslim/bert-base-NER"
'''
这是 Huggingface 上的一个英文 BERT 模型，专门用于 NER（命名实体识别），它能识别 4 类实体：
ORG（组织）、PER（人物）、LOC（地点）、MISC（其他）
'''

# ✅ 初始化NER pipeline（使用 GPU 可加速，device=0 表示 GPU；若使用 CPU，请设为 device=-1）
ner_pipeline = pipeline(
    "ner",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    aggregation_strategy="simple",  # 将连续子词聚合为单个实体，例如 Apple + Inc → Apple Inc
    device=0  # 如果没有 GPU，请将其设置为 -1
)


def extract_named_entities(texts: Union[str, List[str]]) -> Union[List[tuple], List[List[tuple]]]:
    """
    使用英文 BERT NER 模型提取命名实体（如组织、人物、地点）。

    参数：
        texts (str 或 List[str]): 一句话或多个句子

    返回：
        若输入为 str，则返回 List[Tuple[str, str]]
        若输入为 List[str]，则返回 List[List[Tuple[str, str]]]
    """
    is_single = False
    if isinstance(texts, str):
        texts = [texts]  # 将单句包装为列表，便于统一处理
        is_single = True

    raw_results = ner_pipeline(texts)  # 批量处理所有句子
    structured_results = []
    for sentence_results in raw_results:
        entities = []
        for r in sentence_results:
            entity_text = r["word"]           # 实体文本，例如 Apple Inc
            entity_label = r["entity_group"]   # 实体类别，例如 ORG
            entities.append((entity_text, entity_label))
        structured_results.append(entities)

    return structured_results[0] if is_single else structured_results


def classify_with_emotion_and_semantics(text: str, scorer, sentiment_model, emotion_weight=0.3, threshold=0.4) -> str:
    """
    根据语义相似度 + 情绪 polarity 共同判断文本属于哪个数字能力维度。

    语义打分仍然是第一要义，情绪是作用的放大器，用于分析这句话的重要程度并给予一定的权重

    参数:
        text: 输入文本
        scorer: 已初始化的 Scorer 对象
        sentiment_model: Huggingface pipeline, 支持情绪分析（如 FinBERT）
        emotion_weight: 情绪影响在最终打分中的权重（建议 0.2~0.5）
        threshold: 如果最大加权分数低于此，则返回 'uncertain'

    返回:
        str: 最终推断维度名（如 'brand', 'patent'）
    """

    # Step 1: 获取情绪极性得分
    sentiment = sentiment_model(text)[0]
    sentiment_score = sentiment["score"]
    polarity = 1.0 if sentiment["label"].lower() == "positive" else -1.0
    emotion_value = sentiment_score * polarity  # 范围 [-1, +1]

    # Step 2: 获取语义相似度分数
    similarity_scores = scorer.score_all(text, top_k=1)

    # Step 3: 综合加权评分（语义 + 情绪）
    final_scores = {}
    for dim, match_list in similarity_scores.items():
        semantic_score = match_list[0][1]  # 相似度
        combined = (1 - emotion_weight) * semantic_score + emotion_weight * abs(emotion_value)
        final_scores[dim] = combined

    # Step 4: 选择最大分数维度（超过阈值）
    best_dim, best_score = max(final_scores.items(), key=lambda x: x[1])
    if best_score < threshold:
        return "uncertain"

    return best_dim


# ✅ 示例测试：展示 NER 和分类器效果
if __name__ == "__main__":
    text = "Apple announced a major brand.txt expansion on Ethereum."

    print("Entities:", extract_named_entities(text))
    print("Classified as:", classify_with_emotion_and_semantics(text))