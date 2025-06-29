# digital_state_estimator.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
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

# ===================== 新增可视化支持函数 =====================

def run_state_estimation_with_trace(df: pd.DataFrame, score_cols: list,
                                     rho=0.95, q_scale=0.01, r_scale=0.05):
    """
    和 run_state_estimation 类似，但返回 trace 日志供可视化。
    """
    firms = df['firm_id'].unique()
    results = []
    trace_records = []

    for firm in firms:
        df_firm = df[df['firm_id'] == firm].sort_values("date")
        T = len(df_firm)

        mu0, sigma0 = initialize_prior(df_firm, score_cols)
        Q = q_scale * np.eye(len(score_cols))
        R_cov = r_scale * np.eye(len(score_cols))
        H = np.eye(len(score_cols))

        mu_t = mu0
        sigma_t = sigma0

        for _, row in df_firm.iterrows():
            R_obs = row[score_cols].astype(float).values

            mu_pred = rho * mu_t
            sigma_pred = rho ** 2 * sigma_t + Q

            mu_post, sigma_post = kalman_update(mu_pred, sigma_pred, R_obs, H, R_cov)

            result_row = {
                "firm_id": row["firm_id"],
                "date": row["date"]
            }
            for i, k in enumerate(score_cols):
                result_row[f"theta_{k}"] = mu_post[i]

            results.append(result_row)
            trace_records.append({
                "firm_id": row["firm_id"],
                "date": row["date"],
                "mu_pred": mu_pred.copy(),
                "mu_post": mu_post.copy(),
                "K_gain": sigma_pred @ H.T @ np.linalg.inv(H @ sigma_pred @ H.T + R_cov),
                "R_obs": R_obs.copy()
            })

            mu_t, sigma_t = mu_post, sigma_post

    return pd.DataFrame(results), trace_records

def plot_state_traces(trace_records, score_cols, firm_name):
    """
    可视化 Kalman 状态估计过程（Prior vs Posterior）
    - trace_records: run_state_estimation_with_trace 中返回的 trace 列表
    - score_cols: 所有维度名
    - firm_id: 可选，只画某公司
    """

    # 👉 打印调试信息
    # print("✅ trace_records 示例：", trace_records[:1])
    # print("✅ 维度名 score_cols：", score_cols)

    # 1. 构造 DataFrame
    df_trace = pd.DataFrame(trace_records)

    # 2. 修复日期格式
    df_trace["date"] = pd.to_datetime(
        df_trace["date"].astype(str).str.strip().str.replace(r"-([1-9])$", r"-0\1", regex=True),
        format="%Y-%m",
        errors="coerce"
    )
    df_trace = df_trace.dropna(subset=["date"])  # 避免画图失败

    # 3. 过滤指定公司
    df_trace = df_trace[df_trace["firm_id"] == firm_name]
    if df_trace.empty:
        all_firms = pd.DataFrame(trace_records)["firm_id"].unique()
        print(f"⚠️ 公司 '{firm_name}' 不存在 trace 中。可选公司名包括：{list(all_firms)}")
        return

    # 4. 检查 trace 是否有效
    if df_trace.empty or not {"mu_pred", "mu_post"}.issubset(df_trace.columns):
        print("⚠️ No valid data to plot.")
        return

    # 5. 准备画布
    fig, axs = plt.subplots(len(score_cols), 1, figsize=(10, 3 * len(score_cols)), sharex=True)

    if len(score_cols) == 1:
        axs = [axs]  # 统一处理

    # 6. 分别绘图
    for i, dim in enumerate(score_cols):
        mu_preds = df_trace["mu_pred"].apply(lambda x: x[i] if isinstance(x, (list, np.ndarray)) and len(x) > i else np.nan)
        mu_posts = df_trace["mu_post"].apply(lambda x: x[i] if isinstance(x, (list, np.ndarray)) and len(x) > i else np.nan)

        axs[i].plot(df_trace["date"], mu_preds, linestyle="--", label="Prior (μ_pred)")
        axs[i].plot(df_trace["date"], mu_posts, linestyle="-", label="Posterior (μ_post)")

        axs[i].set_title(f"{firm_name}'s Kalman State θ_{dim}")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    os.makedirs("output/bayesion", exist_ok=True)
    filepath = f"output/bayesion/{firm_name}_Kalman_State.png"
    plt.savefig(filepath, dpi=300)
    # plt.show()

def plot_kalman_gain_heatmap(trace_records, score_cols, step=-1):
    K = trace_records[step]["K_gain"]
    if K is not None:
        plt.figure(figsize=(6, 4))
        sns.heatmap(K, annot=True, fmt=".2f", xticklabels=score_cols, yticklabels=score_cols, cmap="Blues")
        plt.title(f"Kalman Gain Matrix (Step {step})")
        os.makedirs("output/bayesion", exist_ok=True)
        filepath = f"output/bayesion/Kalman Gain Matrix.png"
        plt.savefig(filepath)
        # plt.show()

def plot_kalman_process_graph(score_cols):
    """
        绘制 Kalman 状态传播图：θ_k(t-1) → θ_k(t) ← R_k(t)
        """
    G = nx.DiGraph()

    for col in score_cols:
        G.add_node(f"θ_{col}(t-1)", layer="past_state")
        G.add_node(f"θ_{col}(t)", layer="current_state")
        G.add_node(f"R_{col}(t)", layer="observation")

        # 状态传播 & 观测更新
        G.add_edge(f"θ_{col}(t-1)", f"θ_{col}(t)", label="ρ")
        G.add_edge(f"R_{col}(t)", f"θ_{col}(t)", label="K")

    # 手动 layout 排布
    pos = {}
    for i, col in enumerate(score_cols):
        pos[f"θ_{col}(t-1)"] = (0, -i)
        pos[f"θ_{col}(t)"] = (1.5, -i)
        pos[f"R_{col}(t)"] = (3.0, -i)

    plt.figure(figsize=(10, 1.2 * len(score_cols)))
    nx.draw_networkx_nodes(G, pos, node_size=2200, node_color="#bbdefb")
    nx.draw_networkx_labels(G, pos, font_size=11)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', edge_color='gray', width=1.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_size=10)

    plt.title("Kalman State Transition & Observation Update (per dimension)", fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs("output/bayesion", exist_ok=True)
    filepath = f"output/bayesion/Kalman State Transition.png"
    plt.savefig(filepath)
    # plt.show()



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