import os                      # 用于处理文件路径和目录
import joblib                  # 高效的Python对象持久化工具，用于保存和加载嵌入
import numpy as np            # 数值处理库
from sklearn.metrics.pairwise import cosine_similarity  # 用于计算余弦相似度
from sentence_transformers import SentenceTransformer
from util import load_dimension_examples_from_files

class Scorer:
    """
    针对多个能力维度的文本相似性评分模块。
    该类会为每个维度的 example 生成嵌入向量并缓存，后续输入文本通过余弦相似度与之匹配。
    """

    def __init__(self, model, dimension_examples, cache_dir="scorer_cache"):
        """
        初始化函数，加载模型并处理所有维度的嵌入
        :param model: 支持 encode() 接口的嵌入模型（如 SentenceTransformer）
        :param dimension_examples: 字典，每个维度映射到若干个示例语句
        :param cache_dir: 嵌入缓存存储目录
        """
        self.model = model                                # 嵌入模型（如 BERT/FinBERT）
        self.dimension_examples = dimension_examples      # 存储每个能力维度的示例文本
        self.cache_dir = cache_dir                        # 嵌入缓存路径
        self.dimension_embeddings = {}                    # 存放各维度的嵌入结果

        os.makedirs(cache_dir, exist_ok=True)             # 若缓存目录不存在，则创建之

        # 遍历每个能力维度，对示例进行编码或读取缓存
        for dim, examples in self.dimension_examples.items():
            cache_path = os.path.join(cache_dir, f"{dim}.pkl")  # 为每个维度构建独立缓存路径

            if os.path.exists(cache_path):
                # 如果缓存文件存在，直接从中加载嵌入向量
                print(f"[Scorer] Loading {dim} from cache.")
                data = joblib.load(cache_path)  # 加载字典：包含嵌入向量
                self.dimension_embeddings[dim] = {
                    "examples": examples,       # 示例文本
                    "embeddings": data['embeddings']  # 缓存的嵌入矩阵
                }
            else:
                # 如果没有缓存，则进行编码并保存
                print(f"[Scorer] Encoding {dim} examples.")
                embeddings = self.model.encode(
                    examples,                   # 待嵌入的文本列表
                    batch_size=8,               # 批量大小，避免显存溢出
                    convert_to_numpy=True       # 转为numpy格式便于后续计算
                )
                joblib.dump({'embeddings': embeddings}, cache_path)  # 保存至磁盘
                self.dimension_embeddings[dim] = {
                    "examples": examples,       # 原始示例
                    "embeddings": embeddings    # 刚刚生成的嵌入
                }


    def score_all(self, input_text, top_k=1):
        """
        对给定文本在所有维度上计算余弦相似度，并返回 top_k 匹配结果。
        :param input_text: 输入待判断的文本
        :param top_k: 每个维度返回的最相似样本数
        :return: dict，key为维度，value为(top_k 示例及其相似度)的列表
        """
        # 对输入文本进行编码为嵌入向量
        input_embedding = self.model.encode(
            [input_text],           # 单条输入文本作为列表
            convert_to_numpy=True   # 输出为numpy向量
        )[0]                        # 提取嵌入向量本身（维度 D）

        results = {}                # 存储最终每个维度的得分结果

        # 遍历每个维度，计算相似度
        for dim, data in self.dimension_embeddings.items():
            similarities = cosine_similarity(
                [input_embedding],  # 输入文本嵌入（1xD）
                data['embeddings']  # 当前维度的所有示例嵌入（NxD）
            )[0]                    # 得到 1xN 的相似度数组

            # 获取相似度最高的 top_k 条目索引（按相似度降序排列）
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # 构建返回值：[(示例文本, 相似度)]
            results[dim] = [(data['examples'][i], similarities[i]) for i in top_indices]

        return results  # 返回包含所有维度得分的字典

    def add_examples(self, dimension, new_examples):
        """
        添加新的示例语句到某个能力维度，并更新其嵌入缓存
        :param dimension: 目标能力维度
        :param new_examples: 新增示例文本列表
        """
        if dimension not in self.dimension_examples:
            # 若该维度尚不存在，则新建空列表
            self.dimension_examples[dimension] = []

        # 合并新旧示例
        self.dimension_examples[dimension].extend(new_examples)

        # 重新编码所有示例
        embeddings = self.model.encode(
            self.dimension_examples[dimension],
            batch_size=8,
            convert_to_numpy=True
        )

        # 缓存路径更新
        cache_path = os.path.join(self.cache_dir, f"{dimension}.pkl")

        # 将新的嵌入向量保存到磁盘
        joblib.dump({'embeddings': embeddings}, cache_path)

        # 更新内存中的嵌入字典
        self.dimension_embeddings[dimension] = {
            "examples": self.dimension_examples[dimension],
            "embeddings": embeddings
        }

        print(f"[Scorer] Updated and re-cached {dimension} examples.")


if __name__ == "__main__":
    # 路径指向包含所有 .txt 示例的文件夹
    example_dir = "scorer_examples"

    # 自动加载所有示例
    DIMENSION_EXAMPLES = load_dimension_examples_from_files(example_dir)

    # 初始化嵌入模型（可换成 FinBERT）
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 初始化评分器
    scorer = Scorer(model, DIMENSION_EXAMPLES)

    # 输入一条文本进行匹配
    text = "The company announced a blockchain-based fundraising platform"
    scores = scorer.score_all(text, top_k=2)

    # 输出结果
    for dim, matches in scores.items():
        print(f"\n📌 {dim.upper()}")
        for example, sim in matches:
            print(f" → {example} | 相似度: {sim:.3f}")