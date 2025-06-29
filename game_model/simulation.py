import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from game_model.stackelberg_game import DigitalAssetFirm

def simulate_stackelberg_game_with_model(T=20, use_model=True, save_path=None):
    """
    主函数：模拟 Stackelberg 博弈中两家企业在数字资产投资与压制行为下的动态演化
    增加调试信息，排查 μ 恒为 0 的问题
    """

    lgb_model = joblib.load("../digital_asset_valuation/valuation_model_lgbm.pkl") if use_model else None

    fixed_features = {
        "theta_brand": 0.6,
        "theta_patent": 0.65,
        "theta_crypto": 0.5,
        "theta_executive": 0.55,
        "roe": 0.12,
        "debt_ratio": 0.4,
        "interest_rate": 0.03,
        "epu_index": 130
    }

    params = {
        "rho": 0.9,
        "b": 0.6,
        "gamma": 0.3,
        "eta": 0.1,
        "lambda": 6.0,
        "tau": 0.5,
        "M": 200.0,
        "c": 1.0,
        "kappa": 3.0
    }

    firm_A = DigitalAssetFirm("A", mu_init=0.4, params=params, lgb_model=lgb_model, fixed_features=fixed_features)
    firm_B = DigitalAssetFirm("B", mu_init=0.3, params=params, lgb_model=lgb_model, fixed_features=fixed_features)

    history = {k: [] for k in [
        "mu_A", "mu_B", "a_A", "a_B", "s_A", "s_B",
        "V_A", "V_B", "payoff_A", "payoff_B"]}

    for t in range(T):
        firm_A.optimal_investment()
        firm_B.optimal_investment()

        firm_A.decide_suppression(firm_B.mu)
        firm_B.decide_suppression(firm_A.mu)

        avg_invest = (firm_A.a + firm_B.a) / 2
        firm_A.update_mu(avg_a=avg_invest, opponent_suppression=firm_B.s)
        firm_B.update_mu(avg_a=avg_invest, opponent_suppression=firm_A.s)

        V_A = firm_A.compute_valuation()
        V_B = firm_B.compute_valuation()
        payoff_A = firm_A.compute_payoff()
        payoff_B = firm_B.compute_payoff()

        # 调试信息打印
        print(f"[t={t}] μ_A={firm_A.mu:.4f}, a_A={firm_A.a:.4f}, s_A={firm_A.s}, V_A={V_A:.2f}, payoff_A={payoff_A:.2f}")
        print(f"[t={t}] μ_B={firm_B.mu:.4f}, a_B={firm_B.a:.4f}, s_B={firm_B.s}, V_B={V_B:.2f}, payoff_B={payoff_B:.2f}")

        for k, v in zip(history.keys(), [
            firm_A.mu, firm_B.mu,
            firm_A.a, firm_B.a,
            firm_A.s, firm_B.s,
            V_A, V_B,
            payoff_A, payoff_B
        ]):
            history[k].append(v)

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(history["mu_A"], label="Firm A: μ")
    plt.plot(history["mu_B"], label="Firm B: μ")
    plt.plot(history["V_A"], label="Firm A: Valuation")
    plt.plot(history["V_B"], label="Firm B: Valuation")
    plt.title("Digital Capability and Valuation over Time")
    plt.xlabel("Time Period")
    plt.ylabel("Value / Capability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return pd.DataFrame(history)


if __name__ == "__main__":
    simulate_stackelberg_game_with_model()