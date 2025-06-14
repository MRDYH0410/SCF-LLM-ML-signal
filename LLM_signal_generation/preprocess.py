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


def classify_capability_dimension(text: str) -> str:
    """
    根据输入文本内容判断该句子属于哪类数字能力维度。
    用于后续评分时路由到正确的评分器。

    参数：
        text (str): 一段英文句子或段落

    返回：
        str: 所属维度标签（reputation, executive.txt, patent.txt, brand.txt, crypto.txt）
    """
    lowered = text.lower()  # 转为小写，便于匹配关键词

    if any(keyword in lowered for keyword in ["patent", "uspto", "intellectual property"]):
        return "patent"
    elif any(keyword in lowered for keyword in ["search volume", "brand", "google trends"]):
        return "brand"
    elif any(keyword in lowered for keyword in ["token", "ethereum", "etherscan", "crypto"]):
        return "crypto"
    elif any(keyword in lowered for keyword in ["announce", "launch", "initiate", "expand", "commit"]):
        return "executive"
    else:
        return "reputation"  # 默认为声誉类文本


# ✅ 示例测试：展示 NER 和分类器效果
if __name__ == "__main__":
    text = "Apple announced a major brand.txt expansion on Ethereum."

    print("Entities:", extract_named_entities(text))
    print("Classified as:", classify_capability_dimension(text))