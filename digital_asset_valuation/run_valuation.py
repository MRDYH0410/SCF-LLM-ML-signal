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

from LLM_signal_generation.run_pipeline import run_pipeline
from digital_asset_valuation.digital_state_estimator import run_state_estimation, run_state_estimation_with_trace, \
    plot_state_traces, plot_kalman_gain_heatmap, plot_kalman_process_graph
from digital_asset_valuation.valuation import compute_valuation_model_with_shap, \
    plot_feature_importance, save_model_to_disk

from digital_asset_valuation.valuation import plot_prediction_vs_actual, plot_residuals_vs_actual, \
    generate_shap_dependence_plot, generate_shap_summary_bar, \
    generate_shap_group_table, plot_shap_force_plot

import pandas as pd

pd.set_option('display.max_columns', None)

import numpy as np

if __name__ == "__main__":
    print("\n📥 Step 0: 读取 LLM 语义维度得分")
    df_R = pd.read_csv("../LLM_signal_generation/output/aggregated_result_v2.csv")
    print(df_R.head())

    score_cols = ['executive', 'brand', 'patent', 'crypto', 'reputation']
    score_cols = [col for col in score_cols if col in df_R.columns]

    firm_cols = ['Brunswick', 'Chevron', 'Civmec', 'ConocoPhillips', 'Exxon', 'Ford Motor', 'General Electric',
                 'General Motors', 'HubSpot', 'ServiceNow', 'Sinopec Group', 'walmart', 'tesla']

    print("\n🔁 Step 1: 状态估计 - Kalman Filter")
    # df_theta = run_state_estimation(df_R, score_cols)
    # print(df_theta.head())

    # 如果你想可视化 trace：
    df_theta, trace_records = run_state_estimation_with_trace(df_R, score_cols)
    # print(df_theta.head())
    # print(trace_records)

    # 画估计状态轨迹
    for firm in firm_cols:
        plot_state_traces(trace_records, score_cols, firm)

    # 可选：展示 Kalman Gain
    plot_kalman_gain_heatmap(trace_records, score_cols, step=-1)

    # 可选：展示因果结构图
    plot_kalman_process_graph(score_cols)

    print("\n📎 Step 2: 拼接经济指标（财务 + 宏观）")
    df_econ = pd.read_csv("data_process/cleaned_input/economic_indicators.csv")
    df_macro = pd.read_csv("data_process/cleaned_input/macro_indicators.csv")

    # 合并经济指标
    df_external = df_econ.merge(df_macro, on=["firm_id", "date"], how="outer")
    df_theta = df_theta.merge(df_external, on=["firm_id", "date"], how="left")

    # 检查缺失
    print("\n🔍 合并后缺失值预览:")
    print(df_theta.isnull().sum())

    print("\n📐 Step 3: 构造 market_value (用于训练目标变量，可替换为真实估值)")
    df_theta["market_value"] = (
            40 + 22 * df_theta["theta_brand"] +
            18 * df_theta["theta_patent"] +
            12 * df_theta["theta_crypto"] +
            14 * df_theta["theta_reputation"] +
            16 * df_theta["theta_executive"] +
            60 * df_theta["roe"] +
            20 * df_theta["gross_margin"] -
            25 * df_theta["debt_ratio"] -
            200 * df_theta["interest_rate"] -
            0.15 * df_theta["policy_uncertainty_index"] +
            np.random.normal(0, 3, len(df_theta))
    )

    print("\n📎 Step 2: 构建估值模型输入数据（拼接财务 + 宏观）")

    # ✅ 选择所有维度 + 财务 + 宏观作为特征
    feature_cols = [
        "theta_brand", "theta_patent", "theta_crypto",
        "theta_reputation", "theta_executive",
        "roe", "debt_ratio", "gross_margin", "inventory_turnover",
        "current_ratio", "cash_shortterm", "total_assets",
        "interest_rate", "inflation_rate", "policy_uncertainty_index",
        "sector_sentiment_index", "commodity_index"
    ]
    target_col = "market_value"

    print("\n🎯 Step 4: 模型训练 + SHAP 分析")
    model, shap_values, explainer, metrics, X_train = compute_valuation_model_with_shap(
        df_theta, feature_cols, target_col
    )
    print(f"\n📈 模型性能:\n - MSE: {metrics['mse']:.2f}\n - R²:  {metrics['r2']:.2f}")

    # ✅ 新增验证图
    plot_feature_importance(model, X_train.columns)

    y_pred = model.predict(X_train)
    y_true = df_theta.loc[X_train.index, target_col].reset_index(drop=True)

    plot_prediction_vs_actual(y_true, y_pred)  # 图 2
    plot_residuals_vs_actual(y_true, y_pred)  # 图 3

    generate_shap_dependence_plot(shap_values, X_train, feature_name="theta_reputation", interaction_index="roe")  # 图 4
    generate_shap_summary_bar(shap_values, X_train)  # 图 5

    generate_shap_group_table(shap_values, X_train)  # 表 4
    plot_shap_force_plot(model, X_train, sample_index=0)  # 图 6

    save_model_to_disk(model)

    print("\n✅ 全流程执行完成。估值模型与 SHAP 解释结果已输出。")

    # # 合成财务 + 宏观指标 + 目标变量（此处为示例数据）
    # np.random.seed(42)
    # df_theta["roe"] = np.random.normal(0.12, 0.05, len(df_theta))
    # df_theta["debt_ratio"] = np.random.normal(0.4, 0.05, len(df_theta))
    # df_theta["interest_rate"] = np.random.normal(0.03, 0.005, len(df_theta))
    # df_theta["epu_index"] = np.random.normal(130, 30, len(df_theta))
    #
    # # 构造 market value（作为训练目标）用于示例验证
    # df_theta["market_value"] = (
    #         50 + 20 * df_theta["theta_brand"] +
    #         25 * df_theta["theta_patent"] +
    #         10 * df_theta["theta_crypto"] +
    #         15 * df_theta["theta_reputation"] +
    #         18 * df_theta["theta_executive"] +
    #         80 * df_theta["roe"] -
    #         30 * df_theta["debt_ratio"] -
    #         200 * df_theta["interest_rate"] -
    #         0.1 * df_theta["epu_index"] +
    #         np.random.normal(0, 3, len(df_theta))
    # )
    #
    # # 选择特征列和目标列
    # feature_cols = [
    #     "theta_brand", "theta_patent", "theta_crypto",
    #     "theta_reputation", "theta_executive",
    #     "roe", "debt_ratio", "interest_rate", "epu_index"
    # ]
    # target_col = "market_value"
