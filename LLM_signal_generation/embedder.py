# embedder.py
# 功能：使用 BERT 提取英文句子的语义嵌入向量（默认 [CLS] 表示）
# 可选择启用 mean pooling

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# ✅ 默认使用 BERT-base 模型（通用语义表示）
MODEL_NAME = "bert-base-uncased"

class SentenceEmbedder:
    def __init__(self, model_name=MODEL_NAME, use_mean_pooling=False):
        """
        初始化嵌入器，加载 BERT 模型和 tokenizer，自动使用 GPU（如可用）。

        参数：
            model_name (str): Huggingface 上的模型名称
            use_mean_pooling (bool): 是否使用 mean pooling 替代 [CLS] 向量
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # 关闭 dropout，提高稳定性

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # print(f"✅ SentenceEmbedder loaded on: {self.device}")

        self.use_mean_pooling = use_mean_pooling

    def encode(self, sentences):
        """
        提取输入句子的嵌入表示。

        参数：
            sentences (str 或 List[str]): 单句或多句输入文本

        返回：
            np.ndarray: 每个句子的向量嵌入，形状为 [batch_size, hidden_dim]
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        # ✅ 编码输入文本，自动 padding 和截断
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        # ✅ 执行模型前向传播，获取隐藏状态
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        if self.use_mean_pooling:
            # mean pooling：对所有 token 向量取平均值（可忽略 padding）
            embeddings = last_hidden_state.mean(dim=1)
        else:
            # 默认使用第一个 token（[CLS]）作为句子表示
            embeddings = last_hidden_state[:, 0, :]

        return embeddings.cpu().numpy()


# ✅ 示例测试
if __name__ == "__main__":
    embedder = SentenceEmbedder()
    sample_sentences = [
        "Apple signed a new strategic agreement with CATL.",
        "Tesla launched a new token-based rewards program."
    ]
    vectors = embedder.encode(sample_sentences)
    print("向量 shape:", vectors.shape)  # 应输出: (2, 768)