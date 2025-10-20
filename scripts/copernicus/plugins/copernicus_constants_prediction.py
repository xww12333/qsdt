#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
常数预测模块（概念验证）

目标：
- 基于附录7的β方程，从普朗克→电弱→低能关键能标，输出 g, J, E, Γ。
- 预留“物理常数映射”接口，将 (g, J, E, Γ) → (α, α_s, G_F, …)。
- 当前默认仅打印与单位一致性检查；不声称数值等同实验，需后续标定。
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
# 复用已有耦合常数与θ_W实现
try:
    from scripts.copernicus.models import run_alpha_qed, run_alpha_s  # when run from repo root
except Exception:
    from ..models import run_alpha_qed, run_alpha_s  # when run inside scripts/copernicus
try:
    from scripts.copernicus.plugins.strict_formulas import _calculate_weinberg_angle as calc_theta_w
except Exception:
    from .strict_formulas import _calculate_weinberg_angle as calc_theta_w


class ConstantsPredictor:
    def __init__(self):
        # 关键能标（GeV）
        self.mu_Pl = 1.22e19
        self.mu_EW = 246.0
        self.mu_L = 85.0
        self.mu_hadron = 1.0

        # β函数参数（附录7）
        self.A = 1.0
        self.b_J = 0.1
        self.c_J = 0.1

        # 低能边界（附录7 QED）
        self.J_low = 9.78e8
        self.E_low = 1.956e9

    def beta_equations(self, t, y):
        # t = ln(mu)
        g, J, E = y
        mu = math.exp(t)

        g_c = max(0.0, min(1.0, g))
        Gamma = g_c * J

        dg_dt = self.A * g_c * (1.0 - g_c)
        dJ_dt = (-self.b_J * J + self.c_J * (Gamma**2) / J) if J > 0 else 0.0
        dE_dt = (-self.b_J * E + self.c_J * g_c * J)
        return [dg_dt, dJ_dt, dE_dt]

    def run(self, mu_start, mu_end, y0):
        t_span = [math.log(mu_start), math.log(mu_end)]
        sol = solve_ivp(
            self.beta_equations, t_span, y0, method="RK45", rtol=1e-6, atol=1e-8, max_step=0.1
        )
        if not sol.success:
            return {
                "mu": np.array([mu_start, mu_end]),
                "g": np.array([y0[0], y0[0]]),
                "J": np.array([y0[1], y0[1]]),
                "E": np.array([y0[2], y0[2]]),
                "Gamma": np.array([y0[0] * y0[1], y0[0] * y0[1]]),
                "note": f"integrator failed: {sol.message}",
            }

        mu_vals = np.exp(sol.t)
        g_vals = sol.y[0]
        J_vals = sol.y[1]
        E_vals = sol.y[2]
        Gamma_vals = g_vals * J_vals
        return {"mu": mu_vals, "g": g_vals, "J": J_vals, "E": E_vals, "Gamma": Gamma_vals}

    def map_constants_placeholder(self, mu_GeV, g, J, E, Gamma):
        """
        物理常数映射占位：
        - α(μ) ~ f(g, J, E, Γ)
        - α_s(μ) ~ f(...)
        - G_F, θ_W 等
        说明：需后续通过附录与实验数值标定；此处仅返回结构化占位。
        """
        # α_em(μ), α_s(μ) 使用已有一回路近似（带阈值），不绑定Higgs
        try:
            alpha_em = run_alpha_qed(float(mu_GeV), None)
        except Exception:
            alpha_em = None
        try:
            alpha_s = run_alpha_s(float(mu_GeV), None)
        except Exception:
            alpha_s = None

        # θ_W 使用现有公式 sin^2θ_W = g'^2/(g^2+g'^2) 的实现
        try:
            theta_w = calc_theta_w({}, float(mu_GeV), {})  # 该实现内部采用固定g, g'
        except Exception:
            theta_w = None

        # G_F = 1/(√2 v^2)，标准模型关系，v≈246 GeV（与Higgs质量无直接依赖）
        try:
            v_GeV = 246.0
            G_F_GeV = 1.0 / (math.sqrt(2.0) * (v_GeV ** 2))
        except Exception:
            G_F_GeV = None

        return {
            "alpha_em": alpha_em,
            "alpha_s": alpha_s,
            "G_F_GeV^-2": G_F_GeV,
            "theta_W_sin2": theta_w,
            "notes": "由现有一回路与标准关系计算；需要与附录数值对表标定。",
        }

    def run_report(self):
        print("=== 常数预测（概念验证） ===")
        print("基于附录7 β函数，输出关键能标处(g, J, E, Γ)，并预留常数映射接口\n")

        y0 = [1e-6, self.J_low, self.E_low]

        segments = [
            (self.mu_Pl, self.mu_EW, "Planck → EW"),
            (self.mu_EW, self.mu_L, "EW → Lepton scale"),
            (self.mu_EW, self.mu_hadron, "EW → Hadron scale"),
        ]

        last = y0
        for mu_s, mu_e, tag in segments:
            print(f"段: {tag}")
            res = self.run(mu_s, mu_e, last)
            g, J, E, Gamma = res["g"][-1], res["J"][-1], res["E"][-1], (res["g"][-1] * res["J"][-1])
            print(f"  终点μ = {mu_e:.3e} GeV")
            print(f"    g = {g:.6e}")
            print(f"    J = {J:.6e} J")
            print(f"    E = {E:.6e} J")
            print(f"    Γ = {Gamma:.6e} J")
            mapping = self.map_constants_placeholder(mu_e, g, J, E, Gamma)
            print("    常数映射：")
            print(f"      α_em(μ) = {mapping['alpha_em']}")
            print(f"      α_s(μ)  = {mapping['alpha_s']}")
            print(f"      sin²θ_W = {mapping['theta_W_sin2']}")
            print(f"      G_F     = {mapping['G_F_GeV^-2']} GeV^-2")
            print(f"      说明    = {mapping['notes']}")
            print()
            last = [g, J, E]


def main():
    cp = ConstantsPredictor()
    cp.run_report()


if __name__ == "__main__":
    main()


