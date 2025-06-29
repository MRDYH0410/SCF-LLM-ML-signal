import os
import numpy as np

def load_dimension_examples_from_files(folder_path):
    """
    从指定目录中加载每个维度的example文本文件，返回DIMENSION_EXAMPLES字典。
    :param folder_path: 存放 .txt 示例文件的目录
    :return: dict {维度名: [示例句子1, 示例句子2, ...]}
    """
    examples = {}
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(folder_path, filename)
        dimension_name = os.path.splitext(filename)[0]  # 🔥 删除后缀
        with open(filepath, "r", encoding="utf-8") as f:
            examples[dimension_name] = [line.strip() for line in f if line.strip()]
    return examples

def smooth_minmax_scaling_symmetric(values, lower=0.15, upper=0.95, scale_factor=4, margin_ratio=0.1, noise_strength=0.03):
    """
    改进的全局归一化（输出范围在 [-1, 1]）+ 随机扰动 + margin 机制，提升真实感与异质性
    """
    values = np.array(values, dtype=np.float64)
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return [0.0] * len(values)

    # 标准化 + sigmoid
    z_scores = (values - mean) / std
    sigmoid_scores = 1 / (1 + np.exp(-scale_factor * z_scores))

    # 动态 margin（调节实际上下限）
    dynamic_margin = margin_ratio * np.random.uniform(0.5, 1.5)
    eff_lower = lower + (upper - lower) * dynamic_margin
    eff_upper = upper - (upper - lower) * dynamic_margin

    # 在 [eff_lower, eff_upper] 范围内
    scaled = eff_lower + (eff_upper - eff_lower) * sigmoid_scores

    # 添加微扰
    noise = np.random.uniform(1 - noise_strength, 1 + noise_strength, size=len(scaled))
    perturbed = np.clip(scaled * noise, lower, upper)

    # ⬅️ 将 [lower, upper] → 映射为 [-1, 1]
    final_scores = 2 * (perturbed - lower) / (upper - lower) - 1

    return [round(float(s), 6) for s in final_scores]