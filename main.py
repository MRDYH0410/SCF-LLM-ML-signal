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

import pandas as pd
import numpy as np

# ✅ 示例原始输入（可替换为真实文本数据）
example_input = [
    {"firm_id": "BYD", "date": "2025-05-01", "text": "BYD announced a strategic expansion plan."},
    {"firm_id": "BYD", "date": "2025-05-01", "text": "The brand reputation of BYD surged on Google Trends."},
    {"firm_id": "BYD", "date": "2025-05-01", "text": "The company filed 12 new battery patents."},
    {"firm_id": "TSLA", "date": "2025-05-01", "text": "Tesla launched its tokenized loyalty platform."},
    {"firm_id": "TSLA", "date": "2025-05-01",
     "text": "Market perception of Tesla fell sharply after leadership changes."}
]

if __name__ == "__main__":
    print("🚀 Step 1: LLM 语义打分 Pipeline 开始...")
    df_R = run_pipeline(example_input)

    score_cols = ['executive', 'brand', 'patent', 'crypto', 'reputation']

    print("\n🔁 Step 2: 状态估计 - Kalman Filter 滤波")
    df_theta = run_state_estimation(df_R, score_cols)

    print("\n📎 Step 3: 构建估值模型输入数据（拼接财务 + 宏观）")

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

    print("\n🎯 Step 4: 模型训练 + SHAP 分析 + 保存模型")
    model, shap_values, explainer, metrics, X_train = compute_valuation_model_with_shap(
        df_theta, feature_cols, target_col
    )

    print(f"\n📈 模型性能:\n - MSE: {metrics['mse']:.2f}\n - R²:  {metrics['r2']:.2f}")

    generate_shap_visualizations(explainer, shap_values, X_train)
    plot_feature_importance(model, X_train.columns)
    save_model_to_disk(model)

    print("\n✅ 全流程执行完成。估值模型与 SHAP 解释结果已输出。")
