# extract_sentences_from_txt.py

import os
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from langdetect import detect, DetectorFactory, LangDetectException

# 设置 langdetect 可复现
DetectorFactory.seed = 0
nltk.download('punkt')

def is_english(text: str) -> bool:
    """
    判断文本是否为英文句子
    """
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

# def extract_firm_sentences(company_name: str, news_text: str, month_str: str, company_list=None):
#     """
#     提取英文新闻文本中提及公司名的英文句子（限制为 512 字符以内）
#     参数:
#         company_name: 公司名
#         news_text (str): 原始新闻文本
#         month_str (str): 指定月份（格式 "YYYY-MM"）
#         company_list (List[str]): 可选，指定公司名称列表
#     返回:
#         pd.DataFrame: 包含 firm_id, date, text 三列
#     """
#     if company_list is None:
#         company_list = [company_name]
#
#     MAX_LENGTH = 512
#     sentences = sent_tokenize(news_text)
#     results = []
#
#     for sent in sentences:
#         sent_clean = sent.strip().replace("\n", " ")
#
#         # 跳过过长句子
#         if len(sent_clean) > MAX_LENGTH:
#             continue
#
#         # 跳过非英文句子
#         if not is_english(sent_clean):
#             continue
#
#         # 判断是否提及公司名
#         for firm in company_list:
#             if re.search(rf"\b{firm}\b", sent_clean, re.IGNORECASE):
#                 results.append({
#                     "firm_id": firm,
#                     "date": month_str,
#                     "text": sent_clean
#                 })
#                 break  # 每句只保留一次公司匹配
#     return pd.DataFrame(results)


# def extract_firm_sentences(company_name: str, news_text: str, month_str: str, company_list=None):
#     """
#     提取英文新闻文本中提及公司名的英文句子（限制为 512 字符以内）
#     参数:
#         company_name: 公司名
#         news_text (str): 原始新闻文本
#         month_str (str): 指定月份（格式 "YYYY-MM"）
#         company_list (List[str]): 可选，指定公司名称列表
#     返回:
#         pd.DataFrame: 包含 firm_id, date, text 三列
#     """
#     if company_list is None:
#         company_list = {
#             company_name: [company_name]
#         }
#
#     def is_english(text: str) -> bool:
#         """
#         判断文本是否为英文句子
#         """
#         try:
#             return detect(text) == 'en'
#         except LangDetectException:
#             return False
#
#     MAX_LENGTH = 512
#     results = []
#     paragraphs = news_text.split('\n\n')  # 粗略地以段落为单位
#
#     for para in paragraphs:
#         para_clean = para.strip().replace("\n", " ")
#         matched_firm = None
#
#         for firm, aliases in company_list.items():
#             if any(re.search(rf"\b{alias}\b", para_clean, re.IGNORECASE) for alias in aliases):
#                 matched_firm = firm
#                 break
#
#         if matched_firm:
#             sentences = sent_tokenize(para_clean)
#             for sent in sentences:
#                 sent_clean = sent.strip()
#                 if len(sent_clean) <= MAX_LENGTH and is_english(sent_clean):
#                     results.append({
#                         "firm_id": matched_firm,
#                         "date": month_str,
#                         "text": sent_clean
#                     })
#
#     return pd.DataFrame(results)
def extract_firm_sentences(company_name: str, news_text: str, month_str: str) -> pd.DataFrame:
    """
    提取新闻段落（严格英文 + 合理长度 + 清洗元信息），统一标注为某公司名。

    输入：
        company_name: 公司名，用于 firm_id 列
        news_text: 原始新闻全文文本（可能包含多个段落）
        month_str: 日期（格式 'YYYY-MM'）

    返回：
        pd.DataFrame，列为 ['firm_id', 'date', 'text']
    """

    MAX_LENGTH = 512      # 最大字符长度（防止过长段落）
    MIN_WORDS = 10        # 最小英文单词数量（避免保留垃圾段）
    results = []

    # 清洗空段落 + 去除多余空行
    paragraphs = [p.strip() for p in news_text.split("\n") if p.strip()]

    def is_english_and_valid(text: str) -> bool:
        """
        判定文本是否为【严格英文】且符合清洗要求的有效段落：
        - 是英文语言（langdetect）
        - 不包含非ASCII字符（如中文、emoji、全角标点）
        - 不包含元数据或编辑信息
        - 英文单词数 ≥ MIN_WORDS
        """
        # 1. 判定语言为英文（langdetect）
        try:
            if detect(text) != 'en':
                return False
        except LangDetectException:
            return False

        # 2. 严格字符检查：仅允许 ASCII 范围（去掉任何非英文字符）
        if re.search(r"[^\x00-\x7F]", text):
            return False

        # 3. 过滤典型的元信息/垃圾内容（邮箱、版权、编辑信息等）
        metadata_patterns = [
            r'@[\w\.]+',                             # email 地址
            r'Copyright', r'©',                      # 版权声明
            r'Editing by',                           # 编辑信息
            r'All rights reserved',
            r'https?://\S+', r'\bwww\.\S+\b',         # URL链接
            r'\b(index\.html|/news/|urn:|articleId=)',# 网页路径
            r'[\^\*\-=~]{5,}',                        # 分隔线（例如 "*****"）
            r'\(\(.*\)\)',                            # ((元信息))
        ]
        for pattern in metadata_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return False

        # 4. 至少包含 MIN_WORDS 个英文单词
        word_tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        if len(word_tokens) < MIN_WORDS:
            return False

        return True

    # 遍历每个段落进行筛选
    for para in paragraphs:
        if len(para) > MAX_LENGTH:
            continue
        if not is_english_and_valid(para):
            continue
        results.append({
            "firm_id": company_name,
            "date": month_str,
            "text": para
        })

    return pd.DataFrame(results)

def append_to_csv(df: pd.DataFrame, csv_path: str):
    """
    将提取结果写入或追加到 CSV 文件，自动去重
    """
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True).drop_duplicates()
    else:
        df_combined = df
    df_combined.to_csv(csv_path, index=False)
    print(f"✅ 写入成功：{csv_path}，当前总条数: {len(df_combined)}")

def process_single_txt(company_name: str, txt_path: str, month_str: str, output_csv: str):
    """
    主函数：从 TXT 文件中提取结构化句子，并写入 CSV
    参数:
        txt_path: 输入新闻 TXT 文件路径
        month_str: 日期字符串（格式 '2025-06'）
        output_csv: 输出 CSV 文件路径
        company_name: 输入公司名
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    df_result = extract_firm_sentences(company_name, text, month_str)
    if not df_result.empty:
        append_to_csv(df_result, output_csv)
    else:
        print("⚠️ 未提取到符合条件的英文句子。")

# ✅ 示例用法（只需修改以下参数）
if __name__ == "__main__":
    txt_path = "input/Tesla Elon Musk 29-30 5 2025.txt"   # 输入文件
    month = "2025-5(29-30)"   # 设置月份
    output_csv = "output/Tesla Elon Musk 29-30 5 2025.txt"  # 输出路径
    company_name = 'Tesla'
    process_single_txt(company_name, txt_path, month, output_csv)