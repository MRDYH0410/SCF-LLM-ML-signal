# digital_state_estimator.py

import pandas as pd
import numpy as np
from LLM_signal_generation.run_pipeline import run_pipeline

def initialize_prior(df: pd.DataFrame, score_cols: list):
    """
    初始化状态 μ_0 和 Σ_0，确保所有维度非 NaN。
    - 若某列全为 NaN，则设为默认值 0.5
    - 协方差中的 NaN 替换为默认小值
    """
    sub_df = df[score_cols]

    # 均值处理：NaN → 0.5
    mu0 = sub_df.mean(skipna=True).fillna(0.5).values

    # 协方差处理
    sigma0 = sub_df.cov(min_periods=1)
    sigma0 = sigma0.fillna(0.01).values

    return mu0, sigma0

def kalman_update(mu_pred, sigma_pred, R_obs, H, R_cov):
    """
    执行 Kalman Filter 的观测更新
    参数：
        mu_pred: 预测状态 μ_{t|t-1}
        sigma_pred: 预测协方差 Σ_{t|t-1}
        R_obs: 观测向量 R_{i,t}^{(k)}，缺失值 NaN
        H: 观测矩阵（通常为单位阵，尺寸根据观测值而变）
        R_cov: 观测噪声协方差
    返回：
        mu_post, sigma_post: 后验估计
    """
    mask = ~np.isnan(R_obs)
    if not np.any(mask):
        return mu_pred, sigma_pred  # 无观测，不更新

    R_obs = R_obs[mask]
    H_t = H[mask]
    R_t = R_cov[np.ix_(mask, mask)]

    y = R_obs - H_t @ mu_pred
    S = H_t @ sigma_pred @ H_t.T + R_t
    K = sigma_pred @ H_t.T @ np.linalg.inv(S)

    mu_post = mu_pred + K @ y
    sigma_post = (np.eye(len(mu_pred)) - K @ H_t) @ sigma_pred
    return mu_post, sigma_post

def run_state_estimation(df: pd.DataFrame, score_cols: list, rho=0.95, q_scale=0.01, r_scale=0.05):
    """
    对多个公司进行状态空间估计，输出 θ_hat（均值估计）
    """
    firms = df['firm_id'].unique()
    results = []

    for firm in firms:
        df_firm = df[df['firm_id'] == firm].sort_values("date")
        T = len(df_firm)

        mu0, sigma0 = initialize_prior(df_firm, score_cols)
        Q = q_scale * np.eye(len(score_cols))
        R_cov = r_scale * np.eye(len(score_cols))
        H = np.eye(len(score_cols))

        mu_hist = []
        mu_t = mu0
        sigma_t = sigma0

        for _, row in df_firm.iterrows():
            R_obs = row[score_cols].astype(float).values  # 👈 强制转为 float

            # 状态预测
            mu_pred = rho * mu_t
            sigma_pred = rho ** 2 * sigma_t + Q

            # Kalman 更新
            mu_t, sigma_t = kalman_update(mu_pred, sigma_pred, R_obs, H, R_cov)

            result_row = {
                "firm_id": row["firm_id"],
                "date": row["date"]
            }
            for i, k in enumerate(score_cols):
                result_row[f"theta_{k}"] = mu_t[i]
            mu_hist.append(result_row)

        results.extend(mu_hist)

    return pd.DataFrame(results)

# ✅ 示例
if __name__ == "__main__":
    example_input = [
        {"firm_id": "BYD", "date": "2025-05-01", "text": "BYD announced a strategic expansion plan."},
        {"firm_id": "BYD", "date": "2025-05-01", "text": "The brand.txt reputation of BYD surged on Google Trends."},
        {"firm_id": "BYD", "date": "2025-05-01", "text": "The company filed 12 new battery patents."},
        {"firm_id": "TSLA", "date": "2025-05-01", "text": "Tesla launched its tokenized loyalty platform."},
        {"firm_id": "TSLA", "date": "2025-05-01",
         "text": "Market perception of Tesla fell sharply after leadership changes."}
    ]

    # df = pd.read_csv("aggregated_result.csv")  # 或 run_pipeline 输出
    df = run_pipeline(example_input)
    score_cols = ['executive.txt', 'brand.txt', 'patent.txt', 'crypto.txt', 'reputation']

    state_df = run_state_estimation(df, score_cols)
    print(state_df.head())