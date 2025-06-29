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


def plot_state_traces(trace_records, score_cols, firm_name, firm_cols):
    """
    可视化 Kalman 状态估计过程（Prior vs Posterior）
    使用点 + 误差棒方式展示 Posterior，Prior 为平滑虚线趋势
    - trace_records: run_state_estimation_with_trace 中返回的 trace 列表
    - score_cols: 所有维度名
    - firm_name: 只画某公司
    - firm_cols: 公司集合
    """
    # 构造 DataFrame
    df_trace = pd.DataFrame(trace_records)

    firm_index = firm_cols.index(firm_name)

    # 修复日期格式
    df_trace["date"] = pd.to_datetime(
        df_trace["date"].astype(str).str.strip().str.replace(r"-([1-9])$", r"-0\1", regex=True),
        format="%Y-%m",
        errors="coerce"
    )
    df_trace = df_trace.dropna(subset=["date"])

    # 过滤指定公司
    df_trace = df_trace[df_trace["firm_id"] == firm_name]
    if df_trace.empty:
        all_firms = pd.DataFrame(trace_records)["firm_id"].unique()
        print(f"⚠️ 公司 '{firm_name}' 不存在 trace 中。可选公司名包括：{list(all_firms)}")
        return

    # 每月仅保留最后一个记录
    df_trace = df_trace.sort_values("date")
    df_trace = df_trace.groupby("date").tail(1).reset_index(drop=True)

    # 检查必要字段
    required_fields = {"mu_pred", "mu_post"}
    if df_trace.empty or not required_fields.issubset(df_trace.columns):
        print("⚠️ No valid data to plot or missing required fields (mu_pred, mu_post).")
        return

    # 若无 sigma_post 则补默认值
    if "sigma_post" not in df_trace.columns:
        dim_len = len(df_trace["mu_post"].iloc[0]) if len(df_trace) > 0 else len(score_cols)
        df_trace["sigma_post"] = df_trace["mu_post"].apply(lambda x: [0.01] * dim_len)

    # 准备画布
    fig, axs = plt.subplots(len(score_cols), 1, figsize=(10, 3 * len(score_cols)), sharex=True)
    if len(score_cols) == 1:
        axs = [axs]

    # 绘图
    for i, dim in enumerate(score_cols):
        mu_preds = df_trace["mu_pred"].apply(lambda x: x[i] if isinstance(x, (list, np.ndarray)) and len(x) > i else np.nan)
        mu_posts = df_trace["mu_post"].apply(lambda x: x[i] if isinstance(x, (list, np.ndarray)) and len(x) > i else np.nan)
        sigmas = df_trace["sigma_post"].apply(lambda x: x[i] if isinstance(x, (list, np.ndarray)) and len(x) > i else np.nan)

        axs[i].plot(df_trace["date"], mu_preds, linestyle="--", color="tab:blue", label="Prior (μ_pred)")
        axs[i].errorbar(df_trace["date"], mu_posts, yerr=sigmas, fmt='o-', color="tab:orange",
                        label="Posterior (μ_post)", capsize=3)

        axs[i].set_title(f"Firm {firm_index}'s Kalman State θ_{dim}")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    os.makedirs("output/bayesion", exist_ok=True)
    filepath = f"output/bayesion/{firm_name}_Kalman_State.png"
    plt.savefig(filepath, dpi=300)
#     # plt.show()

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

def generate_table_state_summary(state_df: pd.DataFrame, score_cols: list, output_path="output/bayesion/table4_summary.csv"):
    """
    生成 Table 4：每个公司在每个时间点的 θ 均值和标准差摘要
    """
    state_df["time_index"] = state_df.groupby("firm_id").cumcount()
    summary_records = []

    for firm_id, group in state_df.groupby("firm_id"):
        for t, row in group.iterrows():
            theta_values = [row[f"theta_{col}"] for col in score_cols]
            mean_theta = np.mean(theta_values)
            std_theta = np.std(theta_values)
            summary_records.append({
                "firm_id": firm_id,
                "time_index": row["time_index"],
                "mean_theta": round(mean_theta, 3),
                "std_theta": round(std_theta, 3)
            })

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv("output/bayesion/table_statesummary.csv", index=False)
    return df_summary

def generate_table_dim_summary(trace_records: list, dim_name="reputation", score_cols=None):
    """
    生成 Table 5：对特定维度，跨公司、时间点汇总 μ_t 和协方差 trace
    """
    if score_cols is None:
        raise ValueError("score_cols is required")

    dim_index = score_cols.index(dim_name)
    df_trace = pd.DataFrame(trace_records)

    df_trace["time_index"] = df_trace.groupby("firm_id").cumcount()

    # 聚合：按 time_index
    grouped = df_trace.groupby("time_index")
    table5_data = []
    for t, group in grouped:
        mus = group["mu_post"].apply(
            lambda x: x[dim_index] if isinstance(x, (list, np.ndarray)) else np.nan).dropna()
        sigmas = group["K_gain"].apply(
            lambda K: K[dim_index, dim_index] if isinstance(K, np.ndarray) else np.nan).dropna()
        # 此处 trace_Σ ≈ 对应对角线元素近似
        table5_data.append({
            "time_index": t,
            "mean_mu": round(mus.mean(), 3),
            "std_mu": round(mus.std(), 3),
            "trace_sigma": round(sigmas.mean(), 4)  # 简化近似
        })

    df_table5 = pd.DataFrame(table5_data)
    df_table5.to_csv("output/bayesion/table_dim_summary.csv", index=False)
    return df_table5

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