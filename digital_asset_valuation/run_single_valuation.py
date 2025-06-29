# run_full_pipeline.py
"""
├──LLM_signal_generation/
    ├── preprocess.py
    ├── embedder.py
    ├── scorer.py
    ├── aggregator.py
    └── run_pipeline.py

├──digital_state_estimator.py
└──valuation.py

└──run_full_pipeline.py

主运行文件：将整个3.1 → 3.2 → 3.3流程串联为一体化流程
包含以下步骤：
    1. 文本嵌入 → 语义评分 → 维度打分 (R^{(k)}_{i,t})
    2. 状态估计 (θ^{(k)}_{i,t}) via Kalman filter
    3. 与财务/宏观拼接 → 模型估值 → SHAP解释
"""

from digital_asset_valuation.digital_state_estimator import run_state_estimation_with_trace
from digital_asset_valuation.valuation import compute_valuation_model_with_shap

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_model_performance_comparison(metrics_list, model_names):
    """
    绘制三个模型的 MSE 和 R² 对比柱状图（双子图）
    参数:
        - metrics_list: 包含每个模型的 {'mse': float, 'r2': float} 字典列表
        - model_names: 模型名称列表，如 ["Digital", "Financial", "Macro"]
    """
    mse_values = [m['mse'] for m in metrics_list]
    r2_values = [m['r2'] for m in metrics_list]

    # 设置画布尺寸和风格
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")

    # 子图1: MSE
    plt.subplot(1, 2, 1)
    bars1 = sns.barplot(x=model_names, y=mse_values, palette="Blues_d")
    plt.title("MSE Comparison", fontsize=14, weight='bold')
    plt.ylabel("Mean Squared Error", fontsize=12)
    plt.xlabel("Model Type", fontsize=12)
    plt.ylim(0, max(mse_values) * 1.2)

    # 添加注释
    for i, v in enumerate(mse_values):
        bars1.text(i, v + max(mse_values) * 0.03, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

    # 子图2: R²
    plt.subplot(1, 2, 2)
    bars2 = sns.barplot(x=model_names, y=r2_values, palette="Greens_d")
    plt.title("R² Comparison", fontsize=14, weight='bold')
    plt.ylabel("R-squared", fontsize=12)
    plt.xlabel("Model Type", fontsize=12)
    plt.ylim(min(-0.3, min(r2_values) * 1.2), 1.05)

    # 添加注释（考虑负值）
    for i, v in enumerate(r2_values):
        offset = 0.03 if v >= 0 else -0.05
        bars2.text(i, v + offset, f"{v:.2f}", ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)

    plt.suptitle("Model Performance Comparison", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("output/single_valuation/Model_Performance_Comparison.png", dpi=300)
    plt.show()




def generate_top_shap_table(shap_values_list, X_train_list, model_names, top_n=3):
    """
    生成每个模型最重要的 top_n 个特征及其平均 SHAP 值表格
    参数:
        - shap_values_list: shap_values 对象列表
        - X_train_list: 对应的训练集 DataFrame 列表
        - model_names: 模型名称列表
        - top_n: 每个模型取前几个特征
    返回:
        - DataFrame: 每个模型 top_n 个特征及其平均 SHAP 值
    """
    summary = []

    for shap_vals, X, name in zip(shap_values_list, X_train_list, model_names):
        shap_df = pd.DataFrame(shap_vals, columns=X.columns)
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False).head(top_n)
        for feature, value in mean_abs_shap.items():
            summary.append({
                "Model": name,
                "Feature": feature,
                "Mean(|SHAP|)": round(value, 4)
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('output/single_valuation/top_shap_table.csv', index=False)


if __name__ == "__main__":
    print("\n📥 Step 0: 读取 LLM 语义维度得分")
    df_R = pd.read_csv("../LLM_signal_generation/output/aggregated_result_v2.csv")
    print(df_R.head())

    score_cols = ['leadership', 'brand', 'patent', 'crypto', 'reputation']
    score_cols = [col for col in score_cols if col in df_R.columns]

    firm_cols = ['Brunswick', 'Chevron', 'Civmec', 'ConocoPhillips', 'Exxon', 'Ford Motor', 'General Electric',
                 'General Motors', 'HubSpot', 'ServiceNow', 'walmart', 'tesla']

    print("\n🔁 Step 1: 状态估计 - Kalman Filter")
    # df_theta = run_state_estimation(df_R, score_cols)
    # print(df_theta.head())

    # 如果你想可视化 trace：
    df_theta, trace_records = run_state_estimation_with_trace(df_R, score_cols)
    # print(df_theta.head())
    # print(trace_records)

    print("\n📎 Step 2: 拼接经济指标（财务 + 宏观）")
    df_econ = pd.read_csv("data_process/final_input/economic_indicators.csv")
    df_macro = pd.read_csv("data_process/final_input/macro_indicators.csv")

    # 合并经济指标
    df_external = df_econ.merge(df_macro, on=["firm_id", "date"], how="outer")
    df_theta = df_theta.merge(df_external, on=["firm_id", "date"], how="left")

    print("\n📐 Step 3: 引入真实 market_value（股价收盘）")
    df_price = pd.read_csv("y_value/input/economic_indicators.csv", encoding="utf-16")

    # 👉 保持 date 字段为原始字符串（如 2017-1），避免 datetime 格式丢失匹配
    df_price["date"] = df_price["date"].astype(str).str.strip()

    # 转为 long format
    df_price_long = df_price.melt(id_vars="date", var_name="firm_id", value_name="market_value")
    df_price_long["firm_id"] = df_price_long["firm_id"].astype(str).str.strip()

    df_theta["date"] = df_theta["date"].astype(str).str.strip()

    # merge to df_theta
    df_theta = df_theta.merge(df_price_long, on=["firm_id", "date"], how="left")
    print("🧪 缺失的 market_value 数量：", df_theta["market_value"].isna().sum())
    df_theta = df_theta.dropna(subset=["market_value"])  # 可选：避免后续 MSE 报错

    print("\n🧪 检查真实 market_value 缺失值:")
    print(df_theta["market_value"].isnull().sum())

    print("\n📎 Step 4: 构建估值模型输入数据（拼接财务 + 宏观）")
    # ✅ 选择所有维度 + 财务 + 宏观作为特征

    digital_cols = ["theta_brand", "theta_patent", "theta_crypto",
        "theta_reputation", "theta_leadership"]
    financial_cols = ["roe", "debt_ratio", "gross_margin", "inventory_turnover",
        "current_ratio", "cash_shortterm", "total_assets"]
    macro_cols = ["interest_rate", "inflation_rate", "policy_uncertainty_index",
            "commodity_index"]
    feature_cols = [
        "theta_brand", "theta_patent", "theta_crypto",
        "theta_reputation", "theta_leadership",
        "roe", "debt_ratio", "gross_margin", "inventory_turnover",
        "current_ratio", "cash_shortterm", "total_assets",
        "interest_rate", "inflation_rate", "policy_uncertainty_index",
        "commodity_index"
    ]
    combine_cols = ["roe", "debt_ratio", "gross_margin", "inventory_turnover",
        "current_ratio", "cash_shortterm", "total_assets", "interest_rate", "inflation_rate", "policy_uncertainty_index",
            "commodity_index"]
    target_col = "market_value"

    print("\n🎯 Step 5: 模型训练 + SHAP 分析")
    model_1, shap_values_1, explainer_1, metrics_1, X_train_1 = compute_valuation_model_with_shap(
        df_theta, digital_cols, target_col
    )
    print(f"\n📈 模型性能:\n - MSE: {metrics_1['mse']:.2f}\n - R²:  {metrics_1['r2']:.2f}")

    model_2, shap_values_2, explainer_2, metrics_2, X_train_2 = compute_valuation_model_with_shap(
        df_theta, financial_cols, target_col
    )
    print(f"\n📈 模型性能:\n - MSE: {metrics_2['mse']:.2f}\n - R²:  {metrics_2['r2']:.2f}")

    model_3, shap_values_3, explainer_3, metrics_3, X_train_3 = compute_valuation_model_with_shap(
        df_theta, macro_cols, target_col
    )
    print(f"\n📈 模型性能:\n - MSE: {metrics_3['mse']:.2f}\n - R²:  {metrics_3['r2']:.2f}")
    model_4, shap_values_4, explainer_4, metrics_4, X_train_4 = compute_valuation_model_with_shap(
        df_theta, feature_cols, target_col
    )
    print(f"\n📈 模型性能:\n - MSE: {metrics_4['mse']:.2f}\n - R²:  {metrics_4['r2']:.2f}")

    model_5, shap_values_5, explainer_5, metrics_5, X_train_5 = compute_valuation_model_with_shap(
        df_theta, combine_cols, target_col
    )
    print(f"\n📈 模型性能:\n - MSE: {metrics_5['mse']:.2f}\n - R²:  {metrics_5['r2']:.2f}")

    # ✅ 新增验证图
    plot_model_performance_comparison(
        [metrics_1, metrics_2, metrics_3, metrics_4, metrics_5],
        ["Digital", "Financial", "Macro", "Total", "Combined"]
    )

    df_shap_top = generate_top_shap_table(
        [shap_values_1, shap_values_2, shap_values_3, shap_values_4, shap_values_5],
        [X_train_1, X_train_2, X_train_3, X_train_4, X_train_5],
        ["Digital", "Financial", "Macro", "Total", "Combined"]
    )
    print("\n✅ 全流程执行完成。估值模型与 SHAP 解释结果已输出。")