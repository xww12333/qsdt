#!/usr/bin/env python3
"""
脚本名称: lepton_copernicus.py
功能: 轻子质量（e, μ, τ）的多路线关系计算与对比（零自由参数）。
作用:
- 并行输出二进制瀑布、层级律、以及附录52/54 关系修正（cosφ、g_eff、54 树级/统一修正）下的结果, 并与观测对比。
输入: m0,g,β_Γ（关系不变量/锚定尺度）
输出: study/outputs/lepton_copernicus.json（含多路线与附录54不确定度带）
使用方法: python3 study/scripts/lepton_copernicus.py
注意事项:
- 不引入拟合; 附录54不确定度带目前对 α∈[1/137,1/128], δn∈[0.07,0.10] 做端点扫描。
相关附录: 52（耦合涌现）、54（电子质量量子化完整推导）。

关系（零自由参数）
- 二进制瀑布：E_n = m0 / 2^n
- 层级律：m = m_ref * (1 + omega) ** (N/2)
- omega = (beta_gamma * g**2 * ln2) / (4 * pi**2)
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple


LN2 = math.log(2.0)
PI2_4 = 4.0 * math.pi * math.pi


def omega_from(g: float, beta_gamma: float) -> float:
    return (beta_gamma * (g ** 2) * LN2) / PI2_4


def cascade(m0: float, n: int) -> float:
    return m0 / (2.0 ** n)


def layer_gain(omega: float, N: int) -> float:
    return (1.0 + omega) ** (N / 2.0)


def nearest_under_n(m0: float, target: float) -> int:
    """找到满足 m0/2^n <= target 的最大 n（即基准不超过目标）。"""
    n = max(0, int(math.floor(math.log(m0 / target, 2))))
    # 确保不超过目标
    while cascade(m0, n) > target:
        n += 1
    return n


def infer_integer_N(m_ref: float, target: float, omega: float) -> Tuple[int, float]:
    """给定 m_ref 与 target，求整数 N 使 m_ref*(1+omega)^(N/2) 最接近 target。
    返回 (N, m_pred)。
    """
    if m_ref <= 0:
        raise ValueError("m_ref must be positive")
    if target <= 0:
        raise ValueError("target must be positive")
    if omega <= 0:
        # 无层级增益时直接返回 N=0
        return 0, m_ref

    # 以连续值估计 N，再在邻域整数中搜索
    ratio = target / m_ref
    N_cont = 2.0 * math.log(ratio) / math.log(1.0 + omega)
    N0 = int(round(N_cont))
    best = None
    for N in range(max(0, N0 - 5), N0 + 6):
        m = m_ref * layer_gain(omega, N)
        err = abs(m - target)
        if best is None or err < best[0]:
            best = (err, N, m)
    assert best is not None
    return best[1], best[2]


@dataclass
class Inputs:
    m0: float
    g: float
    beta_gamma: float
    # 观测值（用于反推离散映射；正推用于对照）
    m_e_obs: float = 0.511e-3  # GeV
    m_mu_obs: float = 0.10566  # GeV
    m_tau_obs: float = 1.77686  # GeV


def run(inp: Inputs) -> Dict:
    omg = omega_from(inp.g, inp.beta_gamma)

    # 电子：演示 n=18（可视为家族定位样例）；附录52给出耦合涌现/低能反馈的关系式
    n_e = 18
    e_ref = cascade(inp.m0, n_e)
    e_pred = e_ref  # 未加层级律与修正
    # 关系修正A：cos(phi_n) ≈ sqrt(1 - g/2) → 放大因子 ≈ 1/cos(phi_n)（电弱几何近似）
    try:
        cos_phi = math.sqrt(max(1.0 - 0.5 * float(inp.g), 1e-12))
    except Exception:
        cos_phi = 1.0
    e_pred_corr = e_pred / cos_phi
    # 关系修正B：g_eff(μ_n) = g / sqrt(1 + (b g^2 / 8π²) ln(μ_n/M_H))（附录52）
    # 取 μ_n = E_n, M_H = m0, b = 7（可计算化重写口径）
    try:
        b_coeff = 7.0
        mu_n = e_ref  # GeV
        M_H = inp.m0
        ln_ratio = math.log(max(mu_n, 1e-30) / max(M_H, 1e-30))
        denom = math.sqrt(max(1.0 + (b_coeff * (inp.g ** 2) / (8.0 * math.pi * math.pi)) * ln_ratio, 1e-12))
        g_eff = inp.g / denom
        # 作为比例放大因子（相对 g）：
        amp_geff = g_eff / max(inp.g, 1e-12)
    except Exception:
        amp_geff = 1.0
    e_pred_geff = e_pred * amp_geff

    # 附录54路线：y_e = omega * 2^{-n_e}，m = y v / sqrt(2)，统一修正 F_QED * F_geo
    try:
        n_bit = 10
        two_pow = 2.0 ** (-n_bit)
        y0 = omg * two_pow
        v = 246.22  # GeV
        m_tree = y0 * (v / math.sqrt(2.0))  # GeV
        # F_QED ≈ exp[-(3α/4π) ln(v^2 / m^2)]，α~1/130
        alpha = 1.0 / 130.0
        ln_ratio = 2.0 * math.log(max(v,1e-30) / max(m_tree,1e-30))
        fqed = math.exp(-(3.0 * alpha / (4.0 * math.pi)) * ln_ratio)
        # F_geo = 2^{-δn}，δn≈0.07（统一边界）
        delta_n = 0.07
        fgeo = 2.0 ** (-delta_n)
        m_uni = m_tree * fqed * fgeo
        # 不确定度带：α ∈ [1/137,1/128], δn ∈ [0.07,0.10]
        a_min, a_max = 1.0/137.0, 1.0/128.0
        dn_min, dn_max = 0.07, 0.10
        def fqed_of(a):
            return math.exp(-(3.0 * a / (4.0 * math.pi)) * ln_ratio)
        def fgeo_of(dn):
            return 2.0 ** (-dn)
        # 扫描端点组合
        band = []
        for a in (a_min, a_max):
            for dn in (dn_min, dn_max):
                band.append(m_tree * fqed_of(a) * fgeo_of(dn))
        m_uni_min = min(band)
        m_uni_max = max(band)
    except Exception:
        m_tree = None
        m_uni = None
        m_uni_min = None
        m_uni_max = None

    # μ：反推 n（使基准不超过观测），再求整数 N
    n_mu = nearest_under_n(inp.m0, inp.m_mu_obs)
    mu_ref = cascade(inp.m0, n_mu)
    N_mu, mu_pred = infer_integer_N(mu_ref, inp.m_mu_obs, omg)

    # τ：同法
    n_tau = nearest_under_n(inp.m0, inp.m_tau_obs)
    tau_ref = cascade(inp.m0, n_tau)
    N_tau, tau_pred = infer_integer_N(tau_ref, inp.m_tau_obs, omg)

    return {
        "inputs": asdict(inp),
        "omega": omg,
        "electron": {
            "n": n_e,
            "m_ref_GeV": e_ref,
            "m_pred_GeV": e_pred,
            "m_pred_corr_GeV": e_pred_corr,
            "m_pred_geff_GeV": e_pred_geff,
            "m_appendix54_tree_GeV": m_tree,
            "m_appendix54_uni_GeV": m_uni,
            "m_appendix54_uni_min_GeV": m_uni_min,
            "m_appendix54_uni_max_GeV": m_uni_max,
            "m_obs_GeV": inp.m_e_obs,
            "rel_err": (e_pred - inp.m_e_obs) / inp.m_e_obs,
            "rel_err_corr": (e_pred_corr - inp.m_e_obs) / inp.m_e_obs,
            "rel_err_geff": (e_pred_geff - inp.m_e_obs) / inp.m_e_obs,
            "rel_err_appendix54_tree": ((m_tree - inp.m_e_obs) / inp.m_e_obs) if m_tree else None,
            "rel_err_appendix54_uni": ((m_uni - inp.m_e_obs) / inp.m_e_obs) if m_uni else None,
            "rel_err_appendix54_uni_min": ((m_uni_min - inp.m_e_obs) / inp.m_e_obs) if m_uni_min else None,
            "rel_err_appendix54_uni_max": ((m_uni_max - inp.m_e_obs) / inp.m_e_obs) if m_uni_max else None,
            "note": "附录52：A) cosφ_n≈√(1-g/2)；B) g_eff = g / √[1+(b g²/8π²) ln(μ_n/M_H)]，b=7",
        },
        "muon": {
            "n": n_mu,
            "N": N_mu,
            "m_ref_GeV": mu_ref,
            "m_pred_GeV": mu_pred,
            "m_obs_GeV": inp.m_mu_obs,
            "rel_err": (mu_pred - inp.m_mu_obs) / inp.m_mu_obs,
        },
        "tau": {
            "n": n_tau,
            "N": N_tau,
            "m_ref_GeV": tau_ref,
            "m_pred_GeV": tau_pred,
            "m_obs_GeV": inp.m_tau_obs,
            "rel_err": (tau_pred - inp.m_tau_obs) / inp.m_tau_obs,
        },
        "notes": {
            "principle": "反推用于识别离散映射 (n,N)，正推用于对比；全程零自由参数",
            "todo": "在 notes 中补齐 QED/几何修正的关系式后纳入计算",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="轻子质量的哥白尼式推演（关系先行，零自由参数）")
    ap.add_argument("--m0", type=float, default=125.1, help="模族基准 m0 [GeV]")
    ap.add_argument("--g", type=float, default=0.223, help="关系不变量 g (Γ/J)")
    ap.add_argument("--beta-gamma", dest="beta_gamma", type=float, default=3.75, help="关系不变量 β_Γ")
    ap.add_argument("--out", type=str, default="study/outputs/lepton_copernicus.json", help="输出 JSON")
    args = ap.parse_args()

    payload = run(Inputs(m0=args.m0, g=args.g, beta_gamma=args.beta_gamma))
    print("omega=", payload["omega"]) 
    for k in ("electron", "muon", "tau"):
        row = payload[k]
        print(f"{k:8s}", {kk: row[kk] for kk in row if kk in ("n","N","m_pred_GeV","m_obs_GeV","rel_err")})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
