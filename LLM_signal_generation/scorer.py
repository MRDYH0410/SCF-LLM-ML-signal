import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from LLM_signal_generation.embedder import SentenceEmbedder

embedder = SentenceEmbedder()

# ✅ 示例句子（用于生成方向向量）
DIMENSION_EXAMPLES = {
    "reputation_pos": [
        "The company received high praise for innovation.",
        "Customer satisfaction is very high.",
        "Market sentiment is optimistic about the firm."
    ],
    "reputation_neg": [
        "The company faced severe criticism.",
        "Investor confidence has fallen.",
        "There is a loss of trust in the brand."
    ],
    "executive": [
        "The CEO announced a new initiative.",
        "They launched a strategic business plan.",
        "The company committed to digital transformation."
    ],
    "patent": [
        "The company filed a new patent.",
        "They received a technology patent.",
        "Innovation resulted in multiple patents."
    ],
    "brand": [
        "Brand awareness has increased.",
        "Google Trends show strong interest in the brand.",
        "The company is leading in brand recognition."
    ],
    "crypto": [
        "They launched their own token.",
        "Blockchain payment system was adopted.",
        "Ethereum smart contracts are integrated."
    ]
}

# ✅ 构建方向向量
DIRECTION_VECTORS = {
    dim: embedder.encode(examples).mean(axis=0)
    for dim, examples in DIMENSION_EXAMPLES.items()
}

# ✅ 通用余弦相似度
def cosine_similarity(vec1, vec2):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return float(np.dot(vec1, vec2) / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0

# ✅ sigmoid 缩放函数（控制梯度）
def scaled_score(cos_sim, temp=8, baseline=0.7, max_scale=0.8):
    raw = 1 / (1 + np.exp(-temp * (cos_sim - baseline)))
    return float(max_scale * raw)

# ✅ 关键词集
KEYWORDS = {
    "executive": ["announce", "launch", "commit", "initiate", "expand"],
    "patent": ["patent", "filed", "intellectual property", "USPTO"],
    "crypto": ["token", "blockchain", "ethereum", "crypto"],
    "brand": ["brand", "reputation", "google trends", "search volume"]
}

def contains_keywords(text, dimension):
    keyword_list = KEYWORDS.get(dimension, [])
    return any(k in text.lower() for k in keyword_list)

# ✅ FinBERT 模型（用于情绪得分）
FINBERT_MODEL = None
FINBERT_TOKENIZER = None
def load_finbert():
    global FINBERT_MODEL, FINBERT_TOKENIZER
    if FINBERT_MODEL is None or FINBERT_TOKENIZER is None:
        model_name = "yiyanghkust/finbert-tone"
        FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
        FINBERT_MODEL.eval()
        FINBERT_MODEL.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def score_reputation_finbert(text):
    load_finbert()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = FINBERT_TOKENIZER(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = FINBERT_MODEL(**inputs).logits[0].cpu().numpy()
    probs = np.exp(logits) / np.sum(np.exp(logits))
    neg, neu, pos = probs.tolist()
    return float(np.clip(pos - neg, -1.0, 1.0))

# # ✅ 向量方向打分 + 关键词放大机制
# def score_by_embedding_direction(embedding, dimension, text=None):
#     vector = DIRECTION_VECTORS.get(dimension)
#     if vector is None:
#         return 0.0
#     cos_sim = cosine_similarity(embedding, vector)
#     score = scaled_score(cos_sim, temp=5)
#     if text and contains_keywords(text, dimension):
#         score *= 1.2  # 增强语义+结构双重命中
#     return float(np.clip(score, 0, 1))

def score_by_embedding_direction(embedding, dimension, text=None):
    vector = DIRECTION_VECTORS.get(dimension)
    if vector is None:
        return 0.0
    cos_sim = cosine_similarity(embedding, vector)
    score = scaled_score(cos_sim)

    if text and contains_keywords(text, dimension):
        score *= 1.05

    # DEBUG 输出
    print(f"🔍 [{dimension}] cosine_sim = {cos_sim:.4f}, scaled = {score:.4f}")
    return float(np.clip(score, 0, 1))

# ✅ 总调度器
def score_by_dimension(dimension, embedding, text):
    if dimension == "reputation":
        return score_reputation_finbert(text)
    elif dimension in DIRECTION_VECTORS:
        return score_by_embedding_direction(embedding, dimension, text)
    else:
        return 0.0

# ✅ 示例测试
if __name__ == "__main__":
    sample_texts = [
        "BYD launched a strategic expansion plan.",
        "Google Trends show strong interest in BYD brand.",
        "Tesla received a new crypto patent on Ethereum.",
        "The company filed 12 blockchain-related patents.",
        "Customer sentiment is very positive about BYD."
    ]

    for text in sample_texts:
        emb = embedder.encode([text])[0]
        for dim in ["executive", "brand", "patent", "crypto", "reputation"]:
            score = score_by_dimension(dim, emb, text)
            print(f"[{dim}] Score for: {text[:40]}... → {score:.4f}")
