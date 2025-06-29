# 功能：将打分结果（公司、日期、能力维度）聚合为 R_{i,t}^{(k)} 表格结构

import pandas as pd
import numpy as np
from typing import List, Dict

def aggregate_scores_by_firm_time(
    records: List[Dict],
    method: str = "mean"
) -> pd.DataFrame:
    """
    将句子级别的打分记录聚合为公司-月份-能力维度的评分表。

    参数：
        records (List[Dict]): 每条记录结构如下：
            {
                'firm_id': str,
                'date': str or pd.Timestamp,  # 支持 yyyy-mm
                'score_name': str (如 'reputation_finbert'),
                'score_value': float
            }
        method (str): 聚合方式，可选 'mean', 'max', 'weighted'

    返回：
        pd.DataFrame: 聚合结果，列为 score_name，各行对应 (firm_id, month)
    """
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # 支持 yyyy-mm 格式
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()  # 转换为月起始时间
    grouped = df.groupby(["firm_id", "date", "score_name"])

    if method == "mean":
        agg = grouped["score_value"].mean().reset_index()
    elif method == "max":
        agg = grouped["score_value"].max().reset_index()
    elif method == "weighted":
        df["weight"] = 1.0  # 可根据置信度设置自定义权重
        agg = grouped.apply(lambda g: np.average(g["score_value"], weights=g["weight"]))
        agg = agg.reset_index(name="score_value")
    else:
        raise ValueError(f"Unsupported method: {method}")

    # 转换为宽表，每个能力维度一列
    result = agg.pivot(index=["firm_id", "date"], columns="score_name", values="score_value")
    result = result.reset_index()

    # ✅ 将 date 从 Timestamp 转为 yyyy-mm 字符串
    result["date"] = result["date"].dt.strftime("%Y-%m")

    return result

# ✅ 示例测试
if __name__ == "__main__":
    sample_data = [
        {"firm_id": "BYD", "date": "2025-05", "score_name": "reputation_finbert", "score_value": 0.2},
        {"firm_id": "BYD", "date": "2025-05", "score_name": "reputation_finbert", "score_value": -0.4},
        {"firm_id": "BYD", "date": "2025-05", "score_name": "executive_signal", "score_value": 0.6},
        {"firm_id": "BYD", "date": "2025-05", "score_name": "executive_signal", "score_value": 0.8},
        {"firm_id": "TSLA", "date": "2025-05", "score_name": "reputation_finbert", "score_value": -0.1},
    ]

    df = aggregate_scores_by_firm_time(sample_data, method="mean")
    print(df)