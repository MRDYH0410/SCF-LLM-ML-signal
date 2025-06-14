import os

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
