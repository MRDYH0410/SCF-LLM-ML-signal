import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from game_model.digital_firm import DigitalAssetFirm

def strategic_tradeoff_analysis(
        lgb_model_path, tau=0.5, n_followers=2,
        mu_grid=None, invest_grid=None,
        params=None, fixed_features=None,
    ):
    """
    对比自我投资与抑制的最优收益，适用于5.3节战略权衡曲线
    :param lgb_model_path: LightGBM估值模型路径
    :param tau: 估值函数拐点
    :param n_followers: 追随者数目（反击模拟用）
    :param mu_grid: reputation遍历区间
    :param invest_grid: 投资网格
    :param params: 动态参数字典
    :param fixed_features: 公司特征字典
    :param save_csv: 数据导出路径
    :param save_png: 图像导出路径
    :param show_plot: 是否显示图像
    """
    # 默认参数（如未给定）
    if params is None:
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
    if fixed_features is None:
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
    # 网格配置
    if mu_grid is None:
        mu_grid = np.linspace(0, 1, 41)
    if invest_grid is None:
        invest_grid = np.linspace(0, 2.5, 51)

    lgb_model = joblib.load(lgb_model_path)
    results = []
    # 遍历所有belief区间
    for mu in mu_grid:
        # s=0（无抑制），最优投资
        best_pi0, best_a0 = -np.inf, None
        for a in invest_grid:
            firm = DigitalAssetFirm("Leader", mu_init=mu, lgb_model=lgb_model, fixed_features=fixed_features, params=params)
            firm.a = a
            firm.s = 0
            firm.update_mu(opponent_suppression=0, noise_sigma=0)
            payoff = firm.compute_payoff()
            if payoff > best_pi0:
                best_pi0, best_a0 = payoff, a
        # s=1（抑制），最优投资+模拟n个follower反击
        best_pi1, best_a1 = -np.inf, None
        for a in invest_grid:
            firm = DigitalAssetFirm("Leader", mu_init=mu, lgb_model=lgb_model, fixed_features=fixed_features, params=params)
            firm.a = a
            firm.s = 1
            # 反击总抑制数：gamma * n_followers
            firm.update_mu(opponent_suppression=n_followers, noise_sigma=0)
            payoff = firm.compute_payoff()
            if payoff > best_pi1:
                best_pi1, best_a1 = payoff, a
        # 记录
        results.append({
            "mu": mu, "pi_s0": best_pi0, "a_s0": best_a0,
            "pi_s1": best_pi1, "a_s1": best_a1,
            "tau": tau
        })

    df = pd.DataFrame(results)
    df.to_csv("output/strategic_tradeoff_analysis/strategy_tradeoff_results.csv", index=False)

    # 绘图
    plt.figure(figsize=(8,5))
    plt.plot(df["mu"], df["pi_s0"], label="No Suppression ($s_i=0$)", color="orange", linewidth=2)
    plt.plot(df["mu"], df["pi_s1"], label="With Suppression ($s_i=1$)", color="red", linestyle="--", linewidth=2)
    plt.axvline(tau, color="gray", linestyle=":", label="Inflection ($\\tau$)")
    plt.xlabel("Belief $\\tilde{\\mu}_i$")
    plt.ylabel("Leader Payoff $\\pi_i^*$")
    plt.title("Leader Strategy Payoff under Varying Belief States")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/strategic_tradeoff_analysis/strategy_tradeoff_curve.png")
    plt.close()


    return df
