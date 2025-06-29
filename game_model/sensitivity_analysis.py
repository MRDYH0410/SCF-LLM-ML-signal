import numpy as np
import pandas as pd
import joblib
from game_model.digital_firm import DigitalAssetFirm
import matplotlib.pyplot as plt

def heatmap(df_sa, a_grid, grid_mu, tau):
    # 可视化：收益随μ_0和投资/抑制变化（分抑制/不抑制热力图）
    pivot0 = df_sa[df_sa['s']==0].pivot(index="mu_0", columns="a", values="payoff")
    pivot1 = df_sa[df_sa['s']==1].pivot(index="mu_0", columns="a", values="payoff")
    fig, axs = plt.subplots(1, 2, figsize=(13,6))
    im0 = axs[0].imshow(pivot0.values, origin="lower", aspect="auto", cmap="coolwarm",
                        extent=[a_grid[0], a_grid[-1], grid_mu[0], grid_mu[-1]])
    axs[0].set_title("No Suppression (s=0)")
    axs[0].set_xlabel("Investment a")
    axs[0].set_ylabel("Initial μ")
    fig.colorbar(im0, ax=axs[0], label="Payoff")
    im1 = axs[1].imshow(pivot1.values, origin="lower", aspect="auto", cmap="coolwarm",
                        extent=[a_grid[0], a_grid[-1], grid_mu[0], grid_mu[-1]])
    axs[1].set_title("Suppression (s=1)")
    axs[1].set_xlabel("Investment a")
    axs[1].set_ylabel("Initial μ")
    fig.colorbar(im1, ax=axs[1], label="Payoff")
    plt.suptitle(f"Sensitivity Analysis: Payoff vs μ₀ & Investment a (τ={tau})")
    plt.tight_layout()
    plt.savefig("output/sensitivity_analysis/sa_heatmap.png")




    return df_sa


def optimal_strategy_comparison_table(df_sa, tau=0.5, mu0_critical=None):
    """
    自动生成Table 7风格的最优策略对比表
    :param df_sa: 敏感性分析结果DataFrame
    :param tau: 声誉临界点
    :param mu0_critical: 如果有指定，则选取此初始mu，否则自动取mu0最接近tau的
    :return: DataFrame
    """
    # 选择mu_0最接近tau的点（最敏感点），也可以人工指定
    if mu0_critical is None:
        mu0_critical = df_sa['mu_0'].iloc[(df_sa['mu_0']-tau).abs().argsort()[:1]].values[0]
    subset = df_sa[df_sa['mu_0'] == mu0_critical]
    result = []
    for s in [0, 1]:
        row = subset[subset['s']==s].sort_values("payoff", ascending=False).iloc[0]
        s_j = s
        s_i = s  # 单步分析，Leader与Follower同步
        a_star = round(row['a'], 2)
        pi_star = round(row['payoff'], 3)
        if s==0:
            interp = "No suppression, high investment"
        else:
            interp = "Retaliated, moderate investment"
        result.append([s_i, s_j, a_star, pi_star, interp])
    df_table = pd.DataFrame(result, columns=["$s_i^*$", "$s_j^*$", "$a_i^*$", "$\\pi_i^*$", "Interpretation"])
    df_table.to_csv("output/sensitivity_analysis/Optimal Strategy Comparison Under Grid Search.csv", index=False)

def sensitivity_analysis_grid(lgb_model_path, tau=0.5, grid_mu=None, a_grid=None):
    # 参数与网格配置
    lgb_model = joblib.load(lgb_model_path)
    params = {
        "rho": 0.82,
        "b": 1.15,
        "gamma": 0.42,
        "eta": 0.11,
        "lambda": 10.0,
        "tau": tau,
        "M": 200.0,
        "c": 0.08,
        "kappa": 0.1,
        "base": 0.05,
        "sigma": 0.00
    }
    fixed_features = {
        "theta_brand": 0.6,
        "theta_patent": 0.6,
        "theta_crypto": 0.5,
        "theta_reputation": 0.5,
        "theta_leadership": 0.6,
        "roe": 0.13,
        "debt_ratio": 0.42,
        "gross_margin": 0.35,
        "inventory_turnover": 2.8,
        "current_ratio": 1.5,
        "cash_shortterm": 2.1,
        "total_assets": 50.0,
        "interest_rate": 0.03,
        "inflation_rate": 0.025,
        "policy_uncertainty_index": 110,
        "commodity_index": 95.0
    }
    # μ和a的遍历区间
    if grid_mu is None:
        grid_mu = np.linspace(0, 1, 21)
    if a_grid is None:
        a_grid = np.linspace(0, 2.5, 21)
    s_grid = [0, 1]  # 抑制/不抑制

    records = []
    for mu_0 in grid_mu:
        for s in s_grid:
            for a in a_grid:
                # 只模拟单轮决策，不考虑马尔可夫历史（与5.2节数值分析一致）
                firm = DigitalAssetFirm("Leader", mu_init=mu_0, lgb_model=lgb_model, fixed_features=fixed_features, params=params)
                firm.a = a
                firm.s = s
                # 假设对手μ为τ（最敏感点，最贴合文中情景）；如需更真实可扩展
                mu_opponent = tau
                # 单步能力更新
                firm.update_mu(opponent_suppression=1 if s else 0, noise_sigma=0)
                V = firm.compute_valuation()
                payoff = firm.compute_payoff()
                records.append({
                    "mu_0": mu_0, "a": a, "s": s,
                    "mu_next": firm.mu, "valuation": V, "payoff": payoff
                })
    df_sa = pd.DataFrame(records)
    df_sa.to_csv("output/sensitivity_analysis/sa_results", index=False)

    heatmap(df_sa, a_grid, grid_mu, tau)
    optimal_strategy_comparison_table(df_sa)

