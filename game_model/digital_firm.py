import numpy as np
import pandas as pd

class DigitalAssetFirm:
    def __init__(self, name, mu_init, lgb_model, fixed_features, params):
        self.name = name
        self.mu = mu_init
        self.mu_history = [mu_init]
        self.lgb_model = lgb_model
        self.fixed_features = fixed_features.copy()
        self.params = params
        self.a = 0.0      # 当前投资
        self.s = 0        # 当前抑制
        self.payoff = 0.0
        self.valuation = 0.0
        self.cooldown = 0 # 抑制冷却期

    def construct_features(self, mu_value=None):
        # 使用 mu_value 替代 reputation 特征
        x = self.fixed_features.copy()
        x['theta_reputation'] = mu_value if mu_value is not None else self.mu
        return pd.DataFrame([x])

    def update_mu(self, opponent_suppression, noise_sigma=0.01):
        # 数字能力的演化模型（可调公式）
        p = self.params
        noise = np.random.normal(0, noise_sigma)
        base = p.get('base', 0.04)
        self.mu = (
            base
            + p['rho'] * self.mu
            + p['b'] * self.a
            - p['gamma'] * opponent_suppression
            - p['eta'] * self.a**2
            + noise
        )
        self.mu = np.clip(self.mu, 0, 1)
        self.mu_history.append(self.mu)

    def compute_valuation(self):
        X = self.construct_features()
        self.valuation = float(self.lgb_model.predict(X)[0])
        return self.valuation

    def compute_payoff(self):
        # payoff = 估值 - 投资成本 - 抑制成本
        p = self.params
        self.payoff = self.valuation - p['c'] * self.a ** 2 - p['kappa'] * self.s
        return self.payoff

    def optimal_investment(self, explore_sigma=0.1):
        """
        动态投资决策（有限差分逼近 + 随机探索）
        """
        eps = 0.01
        X_plus = self.construct_features(mu_value=self.mu + eps)
        X_minus = self.construct_features(mu_value=self.mu - eps)
        V_plus = float(self.lgb_model.predict(X_plus)[0])
        V_minus = float(self.lgb_model.predict(X_minus)[0])
        V_prime = (V_plus - V_minus) / (2 * eps)
        # 动态规划式/最优控制思想（贴合博弈论文思路）:
        p = self.params
        a_star = max(0, min(2.5, (p['b'] * V_prime) / (2 * p['c'] + 1e-6)))  # 加max/min保证合理
        a_star += np.random.normal(0, explore_sigma)  # 探索扰动
        a_star = np.clip(a_star, 0, 2.5)
        self.a = a_star
        return self.a

    def decide_suppression(self, mu_opponent):
        # 抑制决策采用概率性+冷却机制，模拟市场博弈不确定性
        p = self.params
        if self.cooldown > 0:
            self.s = 0
            self.cooldown -= 1
            return self.s
        # 计算抑制收益
        X_no = self.construct_features(mu_value=mu_opponent)
        X_attack = self.construct_features(mu_value=max(0, mu_opponent - p['gamma']))
        V_no = float(self.lgb_model.predict(X_no)[0])
        V_attack = float(self.lgb_model.predict(X_attack)[0])
        delta_V = V_no - V_attack
        min_threat = 0.2
        # 概率决策 + cool-down防刷
        threshold = max(p['kappa'], min_threat)
        prob = min(1.0, max(0.0, (delta_V - threshold) / (threshold * 2)))
        if np.random.rand() < prob:
            self.s = 1
            self.cooldown = 2
        else:
            self.s = 0
        return self.s
