import pandas as pd

# 读取 Excel 文件
file_path = "500.xlsx"  # 替换为实际路径
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 转换日期为 datetime 类型
df['date'] = pd.to_datetime(df['date'])

# 只保留数值列进行统计（排除公司名、字符串时间等）
numeric_cols = df.select_dtypes(include='number').columns

# 定义聚合函数并命名
agg_funcs = [
    ('mean', 'mean'),
    ('std', 'std'),
    ('median', 'median'),
    ('q25', lambda x: x.quantile(0.25)),
    ('q75', lambda x: x.quantile(0.75))
]

# ========== 1. 公司级别聚合 ==========
company_agg = df.groupby('firm_id')[numeric_cols].agg(agg_funcs)
company_agg.columns = ['{}_{}'.format(col, stat) for col, stat in company_agg.columns]
company_agg = company_agg.reset_index()

# ========== 2. 月份聚合 ==========
df['month'] = df['date'].dt.to_period('M')
month_agg = df.groupby('month')[numeric_cols].agg(agg_funcs)
month_agg.columns = ['{}_{}'.format(col, stat) for col, stat in month_agg.columns]
month_agg = month_agg.reset_index()

# ========== 导出 ==========
company_agg.to_excel("company_aggregated_stats.xlsx", index=False)
month_agg.to_excel("monthly_aggregated_stats.xlsx", index=False)

# ========== 3. 整体数据统计（不分公司、不分时间） ==========
overall_agg = df[numeric_cols].agg([
    'mean',
    'std',
    'median',
    lambda x: x.quantile(0.25),
    lambda x: x.quantile(0.75)
])

# 重命名 index
overall_agg.index = ['mean', 'std', 'median', 'q25', 'q75']
overall_agg = overall_agg.T  # 转置：每行是一个变量，每列是一个统计量
overall_agg = overall_agg.reset_index().rename(columns={'index': 'variable'})

# 可选：输出为 Excel 或 CSV
overall_agg.to_excel("overall_aggregated_stats.xlsx", index=False)