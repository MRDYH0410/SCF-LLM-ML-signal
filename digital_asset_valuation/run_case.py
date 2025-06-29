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
import copy

from digital_asset_valuation.digital_state_estimator import run_state_estimation, run_state_estimation_with_trace, \
    plot_state_traces, plot_kalman_gain_heatmap, plot_kalman_process_graph, generate_table_state_summary, \
    generate_table_dim_summary
from digital_asset_valuation.valuation import compute_valuation_model_with_shap, \
    plot_feature_importance, generate_prediction_gap_table, save_model_to_disk

from digital_asset_valuation.valuation import plot_prediction_vs_actual, plot_residuals_vs_actual, \
    generate_shap_dependence_plot, generate_shap_summary_bar, \
    generate_shap_group_table, plot_shap_force_plot

import pandas as pd

pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt

def clean_date_str(date_str):
    return (
        str(date_str)
        .strip()                  # 去前后空格
        .replace('\u200b', '')    # 去零宽空格
        .replace('／', '/')       # 中文斜杠 → 英文
        .replace('－', '-')       # 中文连字符 → 英文
        .replace('\r', '')
        .replace('\n', '')
    )

def plot_asset_margin_price(df, firm_id, time_col="date"):
    """
    为指定企业绘制5个维度图，每图包含：
        - 左轴：theta_x + gross_margin
        - 右轴：market_value
    """
    score_cols = ["theta_brand", "theta_patent", "theta_crypto", "theta_reputation", "theta_leadership"]

    # 筛选公司数据并排序
    df_firm = df[df["firm_id"] == firm_id].copy()
    df_firm[time_col] = pd.to_datetime(df_firm[time_col], errors="coerce")
    df_firm = df_firm.dropna(subset=[time_col])
    df_firm = df_firm.sort_values(by=time_col)

    # 创建子图
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 20), sharex=True)

    for i, score_col in enumerate(score_cols):
        ax1 = axes[i]
        ax2 = ax1.twinx()  # 创建右轴

        # 左轴：theta + gross_margin
        ax1.plot(df_firm[time_col], df_firm[score_col], label=score_col, color="tab:blue", marker='o')
        ax1.plot(df_firm[time_col], df_firm["gross_margin"], label="gross_margin", color="tab:green", linestyle='--', marker='x')
        ax1.set_ylabel("Score / Margin", color="tab:blue")
        ax1.tick_params(axis='y', labelcolor="tab:blue")

        # 右轴：market_value
        ax2.plot(df_firm[time_col], df_firm["market_value"], label="market_value", color="tab:red", linestyle=':', marker='s')
        ax2.set_ylabel("Market Value", color="tab:red")
        ax2.tick_params(axis='y', labelcolor="tab:red")

        ax1.set_title(f"{score_col} vs Gross Margin & Market Value")
        ax1.grid(True)

        # 图例（合并左右轴）
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.xlabel("Time")
    plt.suptitle(f"{firm_id}: Digital Asset Dimensions with Market Value (Dual Axis)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def plot_leadership_impact(df, model, feature_cols, firm_id="tesla", time_col="date", event_date="2025-06-25"):
    # 筛选数据
    df_plot = df[df["firm_id"] == firm_id].copy()
    df_plot[time_col] = pd.to_datetime(df_plot[time_col])
    df_plot = df_plot.sort_values(by=time_col)

    # 预测估值
    X = df_plot[feature_cols]
    df_plot["market_value_pred"] = model.predict(X)

    # 绘图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # 左轴：theta_leadership
    ax1.plot(df_plot[time_col], df_plot["theta_leadership"], label="theta_leadership", color="tab:blue", marker='o')
    ax1.set_ylabel("Leadership Score", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    # 右轴：model 估值预测
    ax2.plot(df_plot[time_col], df_plot["market_value_pred"], label="Predicted Market Value", color="tab:red", linestyle="--", marker='x')
    ax2.set_ylabel("Predicted Value", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    # 标注关键节点
    event = pd.to_datetime(event_date)
    ax1.axvline(event, color="gray", linestyle=":", linewidth=1.5)
    ax1.text(event, ax1.get_ylim()[1]*0.95, "June 25", rotation=90, color="gray")

    # 图例和美化
    ax1.set_title(f"{firm_id}: Leadership Decline and Valuation Impact around {event_date}")
    fig.tight_layout()
    plt.show()

    return df_plot[["date", "theta_leadership", "market_value_pred"]]

if __name__ == "__main__":
    df_M = pd.read_csv("../LLM_signal_generation/output/case_study/aggregated_case_month.csv")
    df_D = pd.read_csv("../LLM_signal_generation/output/case_study/aggregated_case_day.csv")

    score_cols_1 = ['leadership', 'brand', 'patent', 'crypto', 'reputation']
    score_cols_1 = [col for col in score_cols_1 if col in df_M.columns]

    score_cols_2 = ['leadership', 'brand', 'patent', 'crypto', 'reputation']
    score_cols_2 = [col for col in score_cols_2 if col in df_D.columns]

    firm_cols = ['tesla']

    df_theta_1, trace_records_1 = run_state_estimation_with_trace(df_M, score_cols_1)
    df_theta_2, trace_records_2 = run_state_estimation_with_trace(df_D, score_cols_2)

    df_theta_1['date'] = df_theta_1['date'].apply(clean_date_str)
    df_theta_2['date'] = df_theta_2['date'].apply(clean_date_str)

    # print("\n📎 1. month")
    # df_econ = pd.read_csv("data_process/final_input/case study/case_month/economic_indicators.csv", encoding='gbk')
    # df_macro = pd.read_csv("data_process/final_input/case study/case_month/macro_indicators.csv", encoding='gbk')
    #
    # # 合并经济指标
    # df_external = df_econ.merge(df_macro, on=["firm_id", "date"], how="outer")
    # df_theta_1 = df_theta_1.merge(df_external, on=["firm_id", "date"], how="left")
    #
    # # 检查缺失
    # print("\n🔍 合并后缺失值预览:")
    # # df_theta.to_csv("merged_with_missing.csv", index=False)
    # print(df_theta_1.isnull().sum())
    #
    # df_price = pd.read_csv("y_value/input/case/months_indicators.csv", encoding='gbk')
    #
    # # 👉 保持 date 字段为原始字符串（如 2017-1），避免 datetime 格式丢失匹配
    # df_price["date"] = df_price["date"].astype(str).str.strip()
    #
    # # 转为 long format
    # df_price_long = df_price.melt(id_vars="date", var_name="firm_id", value_name="market_value")
    # df_price_long["firm_id"] = df_price_long["firm_id"].astype(str).str.strip()
    #
    # df_theta_1["date"] = df_theta_1["date"].astype(str).str.strip()
    #
    # # merge to df_theta
    # df_theta_1 = df_theta_1.merge(df_price_long, on=["firm_id", "date"], how="left")
    # print("🧪 缺失的 market_value 数量：", df_theta_1["market_value"].isna().sum())
    # df_theta_1 = df_theta_1.dropna(subset=["market_value"])  # 可选：避免后续 MSE 报错
    #
    # # ✅ 选择所有维度 + 财务 + 宏观作为特征
    # feature_cols = [
    #     "theta_brand", "theta_patent", "theta_crypto",
    #     "theta_reputation", "theta_leadership",
    #     "roe", "debt_ratio", "gross_margin", "inventory_turnover",
    #     "current_ratio", "cash_shortterm", "total_assets",
    #     "interest_rate", "inflation_rate", "policy_uncertainty_index",
    #     "commodity_index"
    # ]
    # target_col = "market_value"
    #
    # model_1, shap_values_1, explainer_1, metrics_1, X_train_1 = compute_valuation_model_with_shap(
    #     df_theta_1, feature_cols, target_col, "smooth"
    # )

    print("=============================================================================")
    print("\n📎 2. day")
    df_econ = pd.read_csv("data_process/final_input/case study/case_day/economic_indicators.csv", encoding='gbk')
    df_macro = pd.read_csv("data_process/final_input/case study/case_day/macro_indicators.csv", encoding='gbk')

    # 合并经济指标
    df_external = df_econ.merge(df_macro, on=["firm_id", "date"], how="outer")
    df_theta_2 = df_theta_2.merge(df_external, on=["firm_id", "date"], how="left")

    # 检查缺失
    print("\n🔍 合并后缺失值预览:")
    # df_theta.to_csv("merged_with_missing.csv", index=False)
    print(df_theta_2.isnull().sum())

    df_price = pd.read_csv("y_value/input/case/days_indicators.csv", encoding='gbk')

    # 👉 保持 date 字段为原始字符串（如 2017-1），避免 datetime 格式丢失匹配
    df_price["date"] = df_price["date"].astype(str).str.strip()

    # 转为 long format
    df_price_long = df_price.melt(id_vars="date", var_name="firm_id", value_name="market_value")
    df_price_long["firm_id"] = df_price_long["firm_id"].astype(str).str.strip()

    df_theta_2["date"] = df_theta_2["date"].astype(str).str.strip()

    # merge to df_theta
    df_theta_2 = df_theta_2.merge(df_price_long, on=["firm_id", "date"], how="left")
    print("🧪 缺失的 market_value 数量：", df_theta_2["market_value"].isna().sum())
    df_theta_2 = df_theta_2.dropna(subset=["market_value"])  # 可选：避免后续 MSE 报错

    # ✅ 选择所有维度 + 财务 + 宏观作为特征
    feature_cols = [
        "theta_brand", "theta_patent", "theta_crypto",
        "theta_reputation", "theta_leadership",
        "roe", "debt_ratio", "gross_margin", "inventory_turnover",
        "current_ratio", "cash_shortterm", "total_assets",
        "interest_rate", "inflation_rate", "policy_uncertainty_index",
        "commodity_index"
    ]
    target_col = "market_value"

    model_2, shap_values_2, explainer_2, metrics_2, X_train_2 = compute_valuation_model_with_shap(
        df_theta_2, feature_cols, target_col, "smooth"
    )

    # 可视化
    # plot_asset_margin_price(df_theta_1, firm_id='tesla')
    # plot_asset_margin_price(df_theta_2, firm_id='tesla')

    plot_leadership_impact(
        df=df_theta_2,
        model=model_2,
        feature_cols=feature_cols,
        firm_id="tesla",
        event_date="2025-06-25"
    )

    # generate_shap_dependence_plot(shap_values_2, X_train_2, feature_name="theta_reputation", interaction_index="roe")
    # generate_shap_summary_bar(shap_values_2, X_train_2)
    # generate_shap_group_table(shap_values_2, X_train_2)