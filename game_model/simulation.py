import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from game_model.digital_firm import DigitalAssetFirm
from game_model.sensitivity_analysis import sensitivity_analysis_grid

def Investment_Game(df_hist):
    # 可视化
    fig, axs = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    axs[0].plot(df_hist["mu_A"], label="Firm A: μ", color="blue")
    axs[0].plot(df_hist["mu_B"], label="Firm B: μ", color="orange")
    axs[0].fill_between(df_hist.index, df_hist["mu_A"], df_hist["mu_B"], color="gray", alpha=0.1)
    axs[0].set_ylabel("Digital Capability (μ)")
    axs[0].legend()
    axs[0].grid(True)
    axs[1].plot(df_hist["V_A"], label="Firm A: Valuation", color="blue")
    axs[1].plot(df_hist["V_B"], label="Firm B: Valuation", color="orange")
    axs[1].set_ylabel("Valuation")
    axs[1].legend()
    axs[1].grid(True)
    axs[2].plot(df_hist["a_A"], label="Firm A: Investment", linestyle="-", color="blue")
    axs[2].plot(df_hist["a_B"], label="Firm B: Investment", linestyle="-", color="orange")
    axs[2].set_ylabel("Investment (aₜ)")
    axs[2].legend()
    axs[2].grid(True)
    axs[3].step(df_hist.index, df_hist["s_A"], label="Firm A: Suppression", linestyle="--", color="blue", where="mid")
    axs[3].step(df_hist.index, df_hist["s_B"], label="Firm B: Suppression", linestyle="--", color="orange", where="mid")
    axs[3].set_ylabel("Suppression (sₜ)")
    axs[3].set_xlabel("Time")
    axs[3].legend()
    axs[3].grid(True)
    plt.suptitle("Stackelberg Digital Investment Game (LGBM Driven)", fontsize=16)
    plt.tight_layout()
    plt.savefig("output/game_theory/Stackelberg Digital Investment Game.png")
    plt.show()
    return df_hist

def behavior_payoff_table(df_hist, rounds=20):
    """
    可视化博弈行为与收益表格
    :param df_hist: 仿真历史DataFrame
    :return: 表格DataFrame
    """
    # 只显示前rounds轮
    table = df_hist.loc[:rounds-1, [
        "mu_A", "a_A", "s_A", "V_A", "payoff_A",
        "mu_B", "a_B", "s_B", "V_B", "payoff_B"
    ]].copy()
    # 更直观列名
    table.columns = [
        "μ_A", "investment_A", "suppression_A", "valuation_A", "payoff_A",
        "μ_B", "investment", "suppression_B", "valuation_B", "payoff_B"
    ]
    # 输出
    table.to_csv('output/game_theory/behavior_payoff_table.csv', index=False)

def simulate_stackelberg_game(T=30):
    """
    模拟 Stackelberg 博弈中两家企业在数字资产投资与压制行为下的动态演化
    """

    lgb_model = joblib.load("../digital_asset_valuation/valuation_model_lgbm.pkl")

    # 固定特征（支持多公司配置不同）
    fixed_features_A = {
        "theta_brand": 0.7,
        "theta_patent": 0.6,
        "theta_crypto": 0.55,
        "theta_reputation": 0.6,  # 起点
        "theta_leadership": 0.6,
        "roe": 0.15,
        "debt_ratio": 0.45,
        "gross_margin": 0.32,
        "inventory_turnover": 2.8,
        "current_ratio": 1.5,
        "cash_shortterm": 2.2,
        "total_assets": 55.0,
        "interest_rate": 0.035,
        "inflation_rate": 0.024,
        "policy_uncertainty_index": 110,
        "commodity_index": 98.0
    }
    fixed_features_B = fixed_features_A.copy()
    fixed_features_B['theta_brand'] = 0.5
    fixed_features_B['theta_reputation'] = 0.2

    # 参数配置
    params = {
        "rho": 0.82,
        "b": 1.15,
        "gamma": 0.42,
        "eta": 0.11,
        "lambda": 10.0,
        "tau": 0.5,
        "M": 200.0,
        "c": 0.08,
        "kappa": 0.1,
        "base": 0.05,  # 能力自然增长
        "sigma": 0.01  # 演化噪声
    }

    firm_A = DigitalAssetFirm("A", mu_init=0.65, lgb_model=lgb_model, fixed_features=fixed_features_A, params=params)
    firm_B = DigitalAssetFirm("B", mu_init=0.25, lgb_model=lgb_model, fixed_features=fixed_features_B, params=params)

    history = {k: [] for k in [
        "mu_A", "mu_B", "a_A", "a_B", "s_A", "s_B", "V_A", "V_B", "payoff_A", "payoff_B"
    ]}

    for t in range(T):
        # 决策投资
        a_A = firm_A.optimal_investment()
        a_B = firm_B.optimal_investment()
        # 决策抑制（顺序Stackelberg可A先B后）
        s_A = firm_A.decide_suppression(firm_B.mu)
        s_B = firm_B.decide_suppression(firm_A.mu)
        # 状态更新
        firm_A.update_mu(opponent_suppression=s_B, noise_sigma=params["sigma"])
        firm_B.update_mu(opponent_suppression=s_A, noise_sigma=params["sigma"])
        # 估值与收益
        V_A = firm_A.compute_valuation()
        V_B = firm_B.compute_valuation()
        payoff_A = firm_A.compute_payoff()
        payoff_B = firm_B.compute_payoff()
        # 记录
        for k, v in zip(history.keys(), [
            firm_A.mu, firm_B.mu,
            firm_A.a, firm_B.a,
            firm_A.s, firm_B.s,
            V_A, V_B,
            payoff_A, payoff_B
        ]):
            history[k].append(v)
        print(
            f"[t={t}] μ_A={firm_A.mu:.4f}, a_A={firm_A.a:.3f}, s_A={firm_A.s}, V_A={V_A:.2f}, payoff_A={payoff_A:.2f} | μ_B={firm_B.mu:.4f}, a_B={firm_B.a:.3f}, s_B={firm_B.s}, V_B={V_B:.2f}, payoff_B={payoff_B:.2f}"
        )

    df_hist = pd.DataFrame(history)
    Investment_Game(df_hist)
    behavior_payoff_table(df_hist)


if __name__ == "__main__":
    mode = "sa"  # 改成 "sa" 即可做敏感性分析
    if mode == "game":
        simulate_stackelberg_game()
    elif mode == "sa":
        sensitivity_analysis_grid(
            lgb_model_path="../digital_asset_valuation/valuation_model_lgbm.pkl",
            tau=0.5,
            grid_mu=np.linspace(0, 1, 41),
            a_grid=np.linspace(0, 2.5, 41),
        )