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
from digital_asset_valuation.digital_state_estimator import run_state_estimation
from digital_asset_valuation.valuation import compute_valuation_model_with_shap, generate_shap_visualizations, \
    plot_feature_importance, save_model_to_disk

from digital_asset_valuation.valuation import plot_prediction_vs_actual, plot_residuals_vs_actual, \
                                                generate_shap_dependence_plot, generate_shap_summary_bar,\
                                                generate_shap_group_table, plot_shap_force_plot


import pandas as pd

pd.set_option('display.max_columns', None)

import numpy as np



if __name__ == "__main__":
    df_R = pd.read_csv("../LLM_signal_generation/output/aggregated_result.csv")
    print(df_R)

    score_cols = ['executive', 'brand', 'patent', 'crypto', 'reputation']
    score_cols = [col for col in score_cols if col in df_R.columns]

    print("\n🔁 Step 1: 状态估计 - Kalman Filter 滤波")
    df_theta = run_state_estimation(df_R, score_cols)
    print(df_theta)

    print("\n📎 Step 2: 构建估值模型输入数据（拼接财务 + 宏观）")

    # 合成财务 + 宏观指标 + 目标变量（此处为示例数据）
    np.random.seed(42)
    df_theta["roe"] = np.random.normal(0.12, 0.05, len(df_theta))
    df_theta["debt_ratio"] = np.random.normal(0.4, 0.05, len(df_theta))
    df_theta["interest_rate"] = np.random.normal(0.03, 0.005, len(df_theta))
    df_theta["epu_index"] = np.random.normal(130, 30, len(df_theta))

    # 构造 market value（作为训练目标）用于示例验证
    df_theta["market_value"] = (
            50 + 20 * df_theta["theta_brand"] +
            25 * df_theta["theta_patent"] +
            10 * df_theta["theta_crypto"] +
            15 * df_theta["theta_reputation"] +
            18 * df_theta["theta_executive"] +
            80 * df_theta["roe"] -
            30 * df_theta["debt_ratio"] -
            200 * df_theta["interest_rate"] -
            0.1 * df_theta["epu_index"] +
            np.random.normal(0, 3, len(df_theta))
    )

    # 选择特征列和目标列
    feature_cols = [
        "theta_brand", "theta_patent", "theta_crypto",
        "theta_reputation", "theta_executive",
        "roe", "debt_ratio", "interest_rate", "epu_index"
    ]
    target_col = "market_value"

    print("\n🎯 Step 3: 模型训练 + SHAP 分析 + 保存模型")
    model, shap_values, explainer, metrics, X_train = compute_valuation_model_with_shap(
        df_theta, feature_cols, target_col
    )

    print(f"\n📈 模型性能:\n - MSE: {metrics['mse']:.2f}\n - R²:  {metrics['r2']:.2f}")

    # ✅ 新增验证图
    y_pred = model.predict(X_train)
    y_true = df_theta.loc[X_train.index, target_col].reset_index(drop=True)

    plot_prediction_vs_actual(y_true, y_pred)  # 图 2
    plot_residuals_vs_actual(y_true, y_pred)  # 图 3

    generate_shap_dependence_plot(shap_values, X_train, feature_name="theta_reputation", interaction_index="roe") # 图 4
    generate_shap_summary_bar(shap_values, X_train) # 图 5

    generate_shap_group_table(shap_values, X_train) # 表 4
    plot_shap_force_plot(model, X_train, sample_index=0) # 图 6

    save_model_to_disk(model)


    print("\n✅ 全流程执行完成。估值模型与 SHAP 解释结果已输出。")
