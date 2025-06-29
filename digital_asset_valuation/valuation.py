"""
valuation.py
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
import seaborn as sns

# ⬇️ 模型训练 + SHAP解释函数
def compute_valuation_model_with_shap(df: pd.DataFrame, feature_cols: list, target_col: str, strict_signal):
    X = df[feature_cols].copy()
    y = df[target_col]
    nunique = X.nunique()
    valid_cols = nunique[nunique > 1].index.tolist()
    if not valid_cols:
        raise ValueError("❌ 所有特征列均为恒定值或无效，无法训练模型")
    X = X[valid_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if strict_signal == "smooth":
        model = LGBMRegressor(min_data_in_bin=1,
        min_data_in_leaf=1,
        n_estimators=100,
        learning_rate=0.1)
    elif strict_signal == "hard":
        model = LGBMRegressor(random_state=42, n_estimators=150, min_child_samples=15, num_leaves=11, learning_rate=0.1)
    else:
        raise "Wrong learner strict requirement"
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    return model, shap_values, explainer, {"mse": mse, "r2": r2}, X_train

# ⬇️ 画 LightGBM 的特征重要性图
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importances from LightGBM")
    plt.tight_layout()
    plt.savefig(f"output/valuation/Feature Importances.png")
    plt.close()

# 预测值与真实值对比分析
def plot_prediction_vs_actual(y_test, y_pred):
    plt.figure(figsize=(6, 6))
    sns.set(style="whitegrid")
    sns.scatterplot(x=y_test, y=y_pred, s=50, color="steelblue", alpha=0.6, edgecolor="black", linewidth=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title("Predicted vs Actual Market Value", fontsize=13, fontweight='bold')
    plt.xlabel("Actual Market Value", fontsize=11)
    plt.ylabel("Predicted Market Value", fontsize=11)
    min_val = min(y_test.min(), y_pred.min()) - 1
    max_val = max(y_test.max(), y_pred.max()) + 1
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"output/valuation/pred_vs_actual.png")
    plt.close()

# 误差分析
def plot_residuals_vs_actual(y_test, y_pred):
    residuals = y_pred - y_test
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Actual Market Value")
    plt.ylabel("Prediction Residuals")
    plt.title("Residuals vs Actual Market Value")
    plt.tight_layout()
    plt.savefig(f"output/valuation/Residuals vs Actual Market Value.png")
    plt.close()


# SHAP图
def generate_shap_dependence_plot(shap_values, X_train, feature_name="theta_reputation", interaction_index="roe"):
    x = X_train[feature_name].values
    shap_val = pd.DataFrame(shap_values, columns=X_train.columns)[feature_name].values
    color_feat = X_train[interaction_index].values
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(x, shap_val, c=color_feat, cmap="coolwarm", edgecolor="k", alpha=0.8)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    cbar = plt.colorbar(scatter)
    cbar.set_label(interaction_index)
    plt.xlabel("Digital Asset Score: Reputation")
    plt.ylabel("SHAP Value on Valuation")
    plt.title("SHAP Dependence Plot")
    plt.tight_layout()
    plt.savefig(f"output/valuation/SHAP Dependence Plot.png")
    plt.close()



# SHAP summary
def generate_shap_summary_bar(shap_values, X_train):
    mean_shap = np.abs(shap_values).mean(axis=0)
    summary_df = pd.DataFrame({"Feature": X_train.columns, "MeanSHAP": mean_shap}).sort_values("MeanSHAP", ascending=True)
    palette = sns.color_palette("viridis", len(summary_df))
    plt.figure(figsize=(8, 5))
    sns.barplot(x="MeanSHAP", y="Feature", data=summary_df, palette=palette)
    plt.title("SHAP Summary Plot", fontsize=14)
    plt.xlabel("Mean(|SHAP Value|)", fontsize=12)
    plt.ylabel("")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"output/valuation/SHAP Summary.png")
    plt.close()


# 生成group
def generate_shap_group_table(shap_values, X_train):
    shap_df = pd.DataFrame(shap_values, columns=X_train.columns)
    group_map = {
        "Digital Capabilities ($R_{i,t}$)": ["theta_brand", "theta_patent", "theta_crypto", "theta_reputation", "theta_leadership"],
        "Financial Fundamentals ($Z_{i,t}$)": ["roe", "debt_ratio"],
        "Macroeconomic Variables ($M_t$)": ["interest_rate", "policy_uncertainty_index"],
        "Interaction Terms": []
    }
    group_data = []
    total_value = 0
    for group, features in group_map.items():
        mean_shap = shap_df[features].abs().mean().sum() if features else 0.0
        group_data.append((group, mean_shap))
        total_value += mean_shap
    df_group = pd.DataFrame(group_data, columns=["Feature Group", "Mean SHAP Value"])
    df_group["Share of Total (%)"] = df_group["Mean SHAP Value"] / total_value * 100
    df_group["Share of Total (%)"] = df_group["Share of Total (%)"].map(lambda x: f"{x:.1f}%")
    df_group["Ranking"] = df_group["Mean SHAP Value"].rank(ascending=False).astype(int)
    df_group.sort_values("Ranking", inplace=True)
    df_group.to_csv("output/valuation/shap_group_table.csv", index=False)
    return df_group

def plot_shap_force_plot(model, X_train, sample_index):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    sample_shap = shap_values[sample_index]
    contributions = pd.Series(sample_shap, index=X_train.columns).sort_values()
    colors = ['red' if val < 0 else 'green' for val in contributions]
    plt.figure(figsize=(8, 4))
    contributions.plot(kind='barh', color=colors)
    plt.title("SHAP Force-like Plot (Single Firm Observation)")
    plt.xlabel("Contribution to Predicted Valuation")
    plt.tight_layout()
    plt.savefig(f"output/valuation/SHAP Force-like Plot.png")
    plt.close()

def generate_prediction_gap_table(df_full: pd.DataFrame, model, feature_cols: list):
    df = df_full.copy()
    X = df[feature_cols]
    y_true = df["market_value"]
    y_pred = model.predict(X)

    gap_df = df[["firm_id", "date"]].copy()
    gap_df["true_value"] = y_true
    gap_df["predicted_value"] = y_pred
    gap_df["absolute_error"] = np.abs(y_pred - y_true)
    gap_df["relative_error (%)"] = 100 * gap_df["absolute_error"] / (y_true.replace(0, np.nan))

    gap_df = gap_df.sort_values("absolute_error", ascending=False)  # 可选排序
    gap_df.to_csv('output/valuation/prediction_gap_table.csv', index=False)


# ⬇️ 保存模型函数
def save_model_to_disk(model, path="valuation_model_lgbm.pkl"):
    joblib.dump(model, path)
    print(f"✅ 模型已保存至: {path}")

# ⬇️ 示例运行入口
if __name__ == "__main__":
    # 合成数据生成
    np.random.seed(42)
    firms = [f"Firm_{i}" for i in range(80)]
    dates = pd.date_range("2022-01-01", periods=6, freq="Q")
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
                0.1 * row["epu_index"] + np.random.normal(0, 8)
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
    model, shap_values, explainer, metrics, X_train = compute_valuation_model_with_shap(df, feature_cols, target_col,"hard")

    print("\n🎯 模型性能:")
    print(f" - MSE: {metrics['mse']:.2f}")
    print(f" - R²:  {metrics['r2']:.2f}")

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