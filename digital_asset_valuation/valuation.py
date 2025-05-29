"""
valuation_model.py
├── prepare_features(df_theta, df_financial, df_macro)
├── train_model(X, y)
├── evaluate_model(model, X_test, y_test)
├── shap_summary_plot(model, X)
├── shap_dependence_plot(model, X, feature)
├── shap_force_plot(model, X, index)

| 类型                | 数据   | 来源说明                      |
| ----------------- | ---- | ------------------------- |
| $\mu_{i,t}^{(k)}$ | 后验能力 | 你的 `state_df` 输出          |
| $Z_{i,t}$         | 财务指标 | 企业季度财务数据，如 `ROE`、负债率      |
| $M_t$             | 宏观指标 | 通胀、利率、政策不确定性              |
| $P_{i,t}$         | 目标市值 | 来自真实市值（如 Compustat, Wind） |
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import joblib

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import joblib

# ⬇️ 模型训练 + SHAP解释函数
def compute_valuation_model_with_shap(df: pd.DataFrame, feature_cols: list, target_col: str):
    X = df[feature_cols].copy()
    y = df[target_col]
    nunique = X.nunique()
    valid_cols = nunique[nunique > 1].index.tolist()
    if not valid_cols:
        raise ValueError("❌ 所有特征列均为恒定值或无效，无法训练模型")
    X = X[valid_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    return model, shap_values, explainer, {"mse": mse, "r2": r2}, X_train

# ⬇️ 生成 SHAP 可视化
def generate_shap_visualizations(explainer, shap_values, X_train):
    shap.summary_plot(shap_values, X_train)
    if "theta_brand" in X_train.columns:
        shap.dependence_plot("theta_brand", shap_values, X_train)

# ⬇️ 画 LightGBM 的特征重要性图
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importances from LightGBM")
    plt.tight_layout()
    plt.show()

# ⬇️ 保存模型函数
def save_model_to_disk(model, path="valuation_model_lgbm.pkl"):
    joblib.dump(model, path)
    print(f"✅ 模型已保存至: {path}")

# ⬇️ 示例运行入口
if __name__ == "__main__":
    # 合成数据生成
    np.random.seed(42)
    firms = [f"Firm_{i}" for i in range(20)]
    dates = pd.date_range("2022-01-01", periods=8, freq="Q")
    rows = []

    for firm in firms:
        for date in dates:
            row = {
                "firm_id": firm,
                "date": date,
                "theta_brand": np.random.normal(0.6, 0.15),
                "theta_patent": np.random.normal(0.7, 0.05),
                "theta_crypto": np.random.normal(0.5, 0.05),
                "theta_reputation": np.random.normal(0.65, 0.05),
                "theta_exec": np.random.normal(0.55, 0.05),
                "roe": np.random.normal(0.12, 0.05),
                "debt_ratio": np.random.normal(0.4, 0.05),
                "interest_rate": np.random.normal(0.03, 0.005),
                "epu_index": np.random.normal(130, 30)
            }
            row["market_value"] = (
                50 + 20 * row["theta_brand"] + 25 * row["theta_patent"] +
                10 * row["theta_crypto"] + 15 * row["theta_reputation"] +
                18 * row["theta_exec"] + 80 * row["roe"] -
                30 * row["debt_ratio"] - 200 * row["interest_rate"] -
                0.1 * row["epu_index"] + np.random.normal(0, 3)
            )
            rows.append(row)

    df = pd.DataFrame(rows)

    feature_cols = [
        "theta_brand", "theta_patent", "theta_crypto",
        "theta_reputation", "theta_exec",
        "roe", "debt_ratio", "interest_rate", "epu_index"
    ]
    target_col = "market_value"

    # 模型训练与解释
    model, shap_values, explainer, metrics, X_train = compute_valuation_model_with_shap(df, feature_cols, target_col)

    print("\n🎯 模型性能:")
    print(f" - MSE: {metrics['mse']:.2f}")
    print(f" - R²:  {metrics['r2']:.2f}")

    generate_shap_visualizations(explainer, shap_values, X_train)
    plot_feature_importance(model, X_train.columns)
    save_model_to_disk(model)

    # 单个样本预测
    print("\n📌 单个企业估值预测:")

    test_input = {
        "theta_brand": 0.65,
        "theta_patent": 0.72,
        "theta_crypto": 0.48,
        "theta_reputation": 0.68,
        "theta_exec": 0.58,
        "roe": 0.15,
        "debt_ratio": 0.38,
        "interest_rate": 0.028,
        "epu_index": 135
    }

    X_sample = pd.DataFrame([test_input])
    predicted_value = model.predict(X_sample)[0]

    print(f"🧮 预测估值结果: {predicted_value:.2f}")
    print("📊 输入特征:")
    for k, v in test_input.items():
        print(f"  - {k}: {v}")

    try:
        shap_sample = explainer(X_sample)
        print("\n📈 特征贡献 (SHAP):")
        for feature, val in zip(X_sample.columns, shap_sample.values[0]):
            print(f"  - {feature}: {val:.4f}")
    except Exception as e:
        print(f"⚠️ SHAP 分析失败: {e}")




    # np.random.seed(42)
    # firms = [f"Firm_{i}" for i in range(20)]
    # dates = pd.date_range("2022-01-01", periods=8, freq="Q")
    # rows = []
    # for firm in firms:
    #     for date in dates:
    #         theta_brand = np.random.normal(0.6, 0.15)
    #         theta_patent = np.random.normal(0.7, 0.05)
    #         theta_crypto = np.random.normal(0.5, 0.05)
    #         theta_reputation = np.random.normal(0.65, 0.05)
    #         theta_exec = np.random.normal(0.55, 0.05)
    #         roe = np.random.normal(0.12, 0.05)
    #         debt_ratio = np.random.normal(0.4, 0.05)
    #         interest_rate = np.random.normal(0.03, 0.005)
    #         epu_index = np.random.normal(130, 30)
    #         market_value = (
    #             50 + 20 * theta_brand + 25 * theta_patent + 10 * theta_crypto +
    #             15 * theta_reputation + 18 * theta_exec + 80 * roe -
    #             30 * debt_ratio - 200 * interest_rate - 0.1 * epu_index +
    #             np.random.normal(0, 3)
    #         )
    #         rows.append({
    #             "firm_id": firm,
    #             "date": date,
    #             "theta_brand": theta_brand,
    #             "theta_patent": theta_patent,
    #             "theta_crypto": theta_crypto,
    #             "theta_reputation": theta_reputation,
    #             "theta_exec": theta_exec,
    #             "roe": roe,
    #             "debt_ratio": debt_ratio,
    #             "interest_rate": interest_rate,
    #             "epu_index": epu_index,
    #             "market_value": market_value
    #         })
    # df = pd.DataFrame(rows)
    # feature_cols = [
    #     "theta_brand", "theta_patent", "theta_crypto",
    #     "theta_reputation", "theta_exec",
    #     "roe", "debt_ratio", "interest_rate", "epu_index"
    # ]
    # model, shap_values, explainer, metrics, X_train = compute_valuation_model_with_shap(
    #     df, feature_cols, "market_value"
    # )
    # print("🎯 模型表现:")
    # print(f" - MSE: {metrics['mse']:.2f}")
    # print(f" - R²:  {metrics['r2']:.2f}")
    # generate_shap_visualizations(explainer, shap_values, X_train)
    # plot_feature_importance(model, X_train.columns)
    # save_model_to_disk(model)