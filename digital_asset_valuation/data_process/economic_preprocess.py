import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

# 读取原始数据
df_raw = pd.read_excel("input/Financial parameters.xlsx")  # 修改路径

# 字段重命名
df_cleaned = df_raw.rename(columns={
    "ROE": "roe",
    "Total Debt Percentage of Total Equity": "debt_ratio",
    "Gross margin": "gross_margin",
    "current ratio(Liudong bilv)": "current_ratio",
    "Cash and short-term investments": "cash_shortterm",
    "inventory turnover": "inventory_turnover",
    "Total Assets": "total_assets"
})

# 对量纲大的字段做 log 变换
df_cleaned["log_cash_shortterm"] = np.log1p(df_cleaned["cash_shortterm"])
df_cleaned["log_total_assets"] = np.log1p(df_cleaned["total_assets"])

# 删除核心字段缺失的行
df_cleaned = df_cleaned.dropna(subset=["roe", "debt_ratio"])

# 提取起始日期
df_cleaned["date_start"] = pd.to_datetime(
    df_cleaned["time"].str.split('-').str[0].str.strip(),
    format="%d.%m.%Y",
    errors='coerce'
)

# 初始化结果表
expanded_rows = []

# 遍历每行，扩展为3个月
for _, row in df_cleaned.iterrows():
    for i in range(3):  # 当前月 + 下2月
        expanded_date = (row["date_start"] + relativedelta(months=i)).strftime("%Y-%m")
        new_row = row.copy()
        new_row["date"] = expanded_date
        expanded_rows.append(new_row)

# 构建长表格式
df_expanded = pd.DataFrame(expanded_rows)

# 最终字段选择并重命名
df_expanded = df_expanded.rename(columns={"company": "firm_id"})
columns_final = [
    "firm_id", "date", "roe", "debt_ratio",
    "gross_margin", "inventory_turnover", "current_ratio",
    "cash_shortterm", "total_assets"
]
df_final = df_expanded[columns_final]

# 保存结果
df_final.to_csv("cleaned_input/economic_indicators.csv", index=False)

# 打印前几行确认
print("✅ 时间拆分完成，每季度三个月扩展成功，输出示例如下：")
print(df_final.head(6))