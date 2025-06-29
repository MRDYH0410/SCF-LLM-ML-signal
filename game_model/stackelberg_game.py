# game_model/stackelberg_game.py

import numpy as np
import pandas as pd

class DigitalAssetFirm:
    def __init__(self, firm_id, mu_init, params, lgb_model=None, fixed_features=None):
        self.firm_id = firm_id
        self.mu = mu_init
        self.params = params
        self.a = 0.0
        self.s = 0
        self.lgb_model = lgb_model
        self.fixed_features = fixed_features or {}

    def _construct_full_feature_vector(self, mu_override=None):
        x = self.fixed_features.copy()
        x_full = {
            "theta_brand": x.get("theta_brand", 0.5),
            "theta_patent": x.get("theta_patent", 0.5),
            "theta_crypto": x.get("theta_crypto", 0.5),
            "theta_reputation": mu_override if mu_override is not None else self.mu,
            "theta_executive": x.get("theta_executive", 0.5),
            "roe": x.get("roe", 0.12),
            "debt_ratio": x.get("debt_ratio", 0.4),
            "interest_rate": x.get("interest_rate", 0.03),
            "epu_index": x.get("epu_index", 130)
        }
        return pd.DataFrame([x_full])

    def update_mu(self, avg_a, opponent_suppression):
        rho = self.params['rho']
        b = self.params['b']
        gamma = self.params['gamma']
        eta = self.params['eta']
        self.mu = rho * self.mu + b * self.a - gamma * opponent_suppression - eta * self.a ** 2

    def compute_valuation(self):
        if self.lgb_model:
            X = self._construct_full_feature_vector()
            return self.lgb_model.predict(X)[0]
        else:
            M = self.params['M']
            lam = self.params['lambda']
            tau = self.params['tau']
            return M / (1 + np.exp(-lam * (self.mu - tau)))

    def compute_payoff(self):
        V = self.compute_valuation()
        c = self.params['c']
        kappa = self.params['kappa']
        return V - c * self.a ** 2 - kappa * self.s

    def optimal_investment(self):
        if self.lgb_model:
            eps = 0.01
            V_plus = self.lgb_model.predict(self._construct_full_feature_vector(mu_override=self.mu + eps))[0]
            V_minus = self.lgb_model.predict(self._construct_full_feature_vector(mu_override=self.mu - eps))[0]
            V_prime = (V_plus - V_minus) / (2 * eps)
            b = self.params['b']
            c = self.params['c']
            self.a = max((b / (2 * c)) * V_prime, 0.0)
        else:
            lam = self.params['lambda']
            tau = self.params['tau']
            M = self.params['M']
            c = self.params['c']
            b = self.params['b']
            exp_term = np.exp(-lam * (self.mu - tau))
            V_prime = (M * lam * exp_term) / ((1 + exp_term) ** 2)
            self.a = (b / (2 * c)) * V_prime
        return self.a

    def decide_suppression(self, mu_opponent):
        if self.lgb_model:
            V_no_attack = self.lgb_model.predict(self._construct_full_feature_vector(mu_override=mu_opponent))[0]
            V_attack = \
            self.lgb_model.predict(self._construct_full_feature_vector(mu_override=mu_opponent - self.params['gamma']))[
                0]
        else:
            lam = self.params['lambda']
            tau = self.params['tau']
            M = self.params['M']
            exp_no = np.exp(-lam * (mu_opponent - tau))
            exp_att = np.exp(-lam * (mu_opponent - self.params['gamma'] - tau))
            V_no_attack = M / (1 + exp_no)
            V_attack = M / (1 + exp_att)
        delta_V = V_no_attack - V_attack
        self.s = int(delta_V >= self.params['kappa'])
        return self.s
