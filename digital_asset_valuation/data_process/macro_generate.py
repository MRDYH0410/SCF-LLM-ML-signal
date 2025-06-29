import pandas as pd
import numpy as np
import os

# 设置随机种子确保复现性
np.random.seed(42)

# 路径配置（修改为你实际的路径）
input_path = "cleaned_input/economic_indicators.csv"  # 原始文件应含 firm_id 和 date 字段
output_path = "cleaned_input/macro_indicators.csv"

# 读取经济指标文件（用于获取 firm_id 和 date 对应结构）
df = pd.read_csv(input_path)

# 保留主键列（公司+月份）
df_macro = df[["firm_id", "date"]].copy()

# 仿真宏观经济变量
n = len(df_macro)
df_macro["interest_rate"] = np.random.normal(loc=0.03, scale=0.005, size=n)
df_macro["inflation_rate"] = np.random.normal(loc=0.02, scale=0.003, size=n)
df_macro["policy_uncertainty_index"] = np.random.normal(loc=120, scale=25, size=n)
df_macro["sector_sentiment_index"] = np.random.normal(loc=50, scale=10, size=n)
df_macro["commodity_index"] = np.random.normal(loc=300, scale=40, size=n)

# 保存为新的 CSV 文件
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_macro.to_csv(output_path, index=False)

print("✅ 宏观指标仿真成功，保存至:", output_path)
print(df_macro.head())