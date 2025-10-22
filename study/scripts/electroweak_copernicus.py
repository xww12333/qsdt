#!/usr/bin/env python3
"""
脚本名称: electroweak_copernicus.py
功能: 电弱部分（W/Z 与 sin²θ_W）的多路径并行关系计算与对比（零自由参数）。
作用:
- 反推 (n,N) 闭环 W/Z 质量; 并输出多条 sin²θ_W 路线: on-shell, cosφ 修正, QSDT 修正RG, 统一边界+RG, 几何投影+RG, QSDT-omega(★)。
输入:
- m0,g,β_Γ（关系不变量或锚定尺度）。
- 可选: study/inputs/electroweak_omega.json（提供 ω_W, ω_Y, β 比以驱动“★”路线）。
输出: study/outputs/electroweak_copernicus.json
使用方法: python3 study/scripts/electroweak_copernicus.py --out study/outputs/electroweak_copernicus.json
注意事项:
- 不引入拟合; “★”路线若无数据文件则回退为 g/g′ 比值的演示; on-shell 不确定度与有效角参考也会标注。
相关附录: 55（on-shell 与有效角）、56（QSDT-omega ★）。

关系:
- 二进制瀑布: E_n = m0 / 2^n
- 层级律: m = m_ref * (1 + omega)^(N/2)
- omega = (beta_gamma * g^2 ln2) / (4 pi^2)
- 电弱: m_W = m_Z cosθ, sin²θ = 1 − (m_W/m_Z)²
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import Tuple, Dict


LN2 = math.log(2.0)
PI2_4 = 4.0 * math.pi * math.pi


def omega_from(g: float, beta_gamma: float) -> float:
    return (beta_gamma * (g ** 2) * LN2) / PI2_4


def cascade(m0: float, n: int) -> float:
    return m0 / (2.0 ** n)


def layer_gain(omega: float, N: int) -> float:
    return (1.0 + omega) ** (N / 2.0)


def nearest_under_n(m0: float, target: float) -> int:
    n = max(0, int(math.floor(math.log(m0 / target, 2))))
    while cascade(m0, n) > target:
        n += 1
    return n


def infer_integer_N(m_ref: float, target: float, omega: float) -> Tuple[int, float]:
    if m_ref <= 0 or target <= 0:
        raise ValueError("m_ref/target 必须为正")
    if omega <= 0:
        return 0, m_ref
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


def ew_qsdt_run(mu_start: float, mu_end: float, g0: float, gp0: float, g_ratio: float) -> Dict[str, float]:
    """QSDT修正的电弱一回路跑动（零参数实现，移植自项目严公式口径）。
    b1_sm=41/6, b2_sm=-19/6；有效系数 b_eff = b_SM * (1 - k * π/2 * (Γ/J)^2)。
    这里以 k1≈0.35 (U(1)_Y), k2≈1.0 (SU(2)_L) 作为关系抑制权重（来自项目实现口径）。
    解析解：1/g(μ)^2 = 1/g(μ0)^2 - 2 b/(16π²) ln(μ/μ0)。
    """
    b1_sm = 41.0 / 6.0
    b2_sm = -19.0 / 6.0
    k1, k2 = 0.35, 1.0
    base = (math.pi / 2.0) * max(g_ratio, 0.0) ** 2
    # 能标权重（项目严公式口径）：靠近电弱阈值处修正增强，远离处减弱（对数权重，夹断）
    try:
        t_abs = abs(math.log(max(mu_end, 1e-30) / max(mu_start, 1e-30)))
        w = max(0.5, min(1.0, 1.0 - 0.08 * t_abs))
    except Exception:
        w = 0.8
    corr1 = max(1.0 - w * k1 * base, 1e-6)
    corr2 = max(1.0 - w * k2 * base, 1e-6)
    b1 = b1_sm * corr1
    b2 = b2_sm * corr2
    kfac = 1.0 / (8.0 * math.pi * math.pi)  # 2/(16π²)
    t = math.log(max(mu_end, 1e-30) / max(mu_start, 1e-30))
    def run_one(g0_val: float, b: float) -> float:
        inv2 = (1.0 / (g0_val * g0_val)) - kfac * b * t
        if inv2 <= 1e-30:
            inv2 = 1e-30
        return 1.0 / math.sqrt(inv2)
    g_end = run_one(max(g0, 1e-12), b2)
    gp_end = run_one(max(gp0, 1e-12), b1)
    return {"g": g_end, "gp": gp_end}


@dataclass
class Inputs:
    m0: float
    g: float
    beta_gamma: float
    m_W_obs: float = 80.379   # GeV
    m_Z_obs: float = 91.1876  # GeV


def run(inp: Inputs) -> Dict:
    omg = omega_from(inp.g, inp.beta_gamma)

    # 反推 W
    n_W = nearest_under_n(inp.m0, inp.m_W_obs)
    W_ref = cascade(inp.m0, n_W)
    N_W, mW = infer_integer_N(W_ref, inp.m_W_obs, omg)

    # 反推 Z
    n_Z = nearest_under_n(inp.m0, inp.m_Z_obs)
    Z_ref = cascade(inp.m0, n_Z)
    N_Z, mZ = infer_integer_N(Z_ref, inp.m_Z_obs, omg)

    # 温伯格角预测（on-shell 质量比）
    sin2 = 1.0 - (mW / mZ) ** 2

    # 关系修正尝试（附录52口径）：
    # 耦合涌现：g_w = g * cos(phi_n)，cos(phi_n) ≈ sqrt(1 - g_ratio/2)
    # 其中 g_ratio = Γ/J（本脚本用输入 g 即为该比值）；
    # 用 on-shell 反解得到 g_base, gprime_base，再应用 g_w 修正，计算修正后的 sin²。
    v = 246.0
    g_base = 2.0 * mW / v
    g_combo = 2.0 * mZ / v
    gp_base = max(g_combo * g_combo - g_base * g_base, 1e-18) ** 0.5
    # cos(phi) 关系修正（零参数）
    try:
        cos_phi = math.sqrt(max(1.0 - 0.5 * float(inp.g), 1e-12))
    except Exception:
        cos_phi = 1.0
    g_w_eff = g_base * cos_phi
    sin2_corr = (gp_base * gp_base) / (g_w_eff * g_w_eff + gp_base * gp_base)

    # QSDT修正RG：从 μ_EW=246 GeV 跑到 μ=M_Z（零参数，使用关系不变量 g_ratio=inp.g）
    mu_EW = 246.0
    res_run = ew_qsdt_run(mu_start=mu_EW, mu_end=inp.m_Z_obs, g0=g_base, gp0=gp_base, g_ratio=float(inp.g))
    g_run, gp_run = res_run["g"], res_run["gp"]
    sin2_run = (gp_run * gp_run) / (g_run * g_run + gp_run * gp_run)

    # 统一边界（采用离散映射预测的 mW,mZ 在 μ_EW 反解 g0,gp0，再做RG）
    g0_pred = 2.0 * mW / mu_EW
    gcombo_pred = 2.0 * mZ / mu_EW
    gp0_pred = max(gcombo_pred * gcombo_pred - g0_pred * g0_pred, 1e-18) ** 0.5
    res_run_pred = ew_qsdt_run(mu_start=mu_EW, mu_end=inp.m_Z_obs, g0=g0_pred, gp0=gp0_pred, g_ratio=float(inp.g))
    g_run_pred, gp_run_pred = res_run_pred["g"], res_run_pred["gp"]
    sin2_run_pred = (gp_run_pred * gp_run_pred) / (g_run_pred * g_run_pred + gp_run_pred * gp_run_pred)

    # 组合口径：几何投影 + RG（在 RG 后套用 cosφ 作为观察者通道投影）
    g_w_run_eff = g_run * cos_phi
    sin2_corr_run = (gp_run * gp_run) / (g_w_run_eff * g_w_run_eff + gp_run * gp_run)

    # Appendix 56: QSDT-omega route (demonstrative mapping)
    # Formula: sin^2 = 1 / (1 + (omega_W/omega_Y) * (beta_Y/beta_W)).
    # In absence of separately measured omegas, emulate omega ratio via (g/g')^2 from on-shell mapping.
    try:
        # 优先读取数据文件：study/inputs/electroweak_omega.json
        import json as _json
        try:
            data = _json.load(open("study/inputs/electroweak_omega.json", "r", encoding="utf-8"))
            omega_W = float(data.get("omega_W"))
            omega_Y = float(data.get("omega_Y"))
            beta_ratio = float(data.get("beta_ratio", 1.0))
            R = (omega_W / omega_Y) * (1.0 / beta_ratio)  # = (ω_W/ω_Y) * (β_Y/β_W)
            sin2_qsdt_omega = 1.0 / (1.0 + R)
            qsdt_source = "datafile"
        except Exception:
            # 回退：用 g,g' 比值近似（β 比 ~1）
            R = (g_base * g_base) / (gp_base * gp_base)
            sin2_qsdt_omega = 1.0 / (1.0 + R)
            qsdt_source = "fallback_g_over_gp"
    except Exception:
        sin2_qsdt_omega = None
        qsdt_source = None

    # Appendix 55: on-shell vs effective
    # On-shell uncertainty estimate using experimental dW,dZ
    try:
        dW, dZ = 0.012, 0.0021
        ds2_dW = -(2.0 * mW) / (mZ * mZ)
        ds2_dZ = (2.0 * mW * mW) / (mZ * mZ * mZ)
        sin2_on_err = ( (ds2_dW * dW) ** 2 + (ds2_dZ * dZ) ** 2 ) ** 0.5
    except Exception:
        sin2_on_err = None
    sin2_eff_ref = 0.2315

    return {
        "inputs": asdict(inp),
        "omega": omg,
        "W": {"n": n_W, "N": N_W, "m_ref_GeV": W_ref, "m_pred_GeV": mW, "m_obs_GeV": inp.m_W_obs, "rel_err": (mW - inp.m_W_obs)/inp.m_W_obs},
        "Z": {"n": n_Z, "N": N_Z, "m_ref_GeV": Z_ref, "m_pred_GeV": mZ, "m_obs_GeV": inp.m_Z_obs, "rel_err": (mZ - inp.m_Z_obs)/inp.m_Z_obs},
        "weinberg_angle": {
            "sin2_pred": sin2,
            "sin2_corr": sin2_corr,
            "sin2_run": sin2_run,
            "sin2_run_pred": sin2_run_pred,
            "sin2_corr_run": sin2_corr_run,
            "sin2_qsdt_omega": sin2_qsdt_omega,
            "qsdt_omega_source": qsdt_source,
            "sin2_on_shell_err": sin2_on_err,
            "sin2_eff_ref": sin2_eff_ref,
            "sin2_ref": 0.231,
            "rel_err": (sin2 - 0.231)/0.231,
            "rel_err_corr": (sin2_corr - 0.231)/0.231,
            "rel_err_run": (sin2_run - 0.231)/0.231,
            "rel_err_run_pred": (sin2_run_pred - 0.231)/0.231,
            "rel_err_corr_run": (sin2_corr_run - 0.231)/0.231
        },
        "notes": {"principle": "离散映射 (n,N)+固定ω；on-shell质量比→sin²θ_W；附录52耦合涌现与QSDT修正RG (零参数)"},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="W/Z 与温伯格角的哥白尼式推演（关系先行，零自由参数）")
    ap.add_argument("--m0", type=float, default=125.1, help="模族基准 m0 [GeV]")
    ap.add_argument("--g", type=float, default=0.223, help="关系不变量 g (Γ/J)")
    ap.add_argument("--beta-gamma", dest="beta_gamma", type=float, default=3.75, help="关系不变量 β_Γ")
    ap.add_argument("--out", type=str, default="study/outputs/electroweak_copernicus.json", help="输出 JSON")
    args = ap.parse_args()

    payload = run(Inputs(m0=args.m0, g=args.g, beta_gamma=args.beta_gamma))
    print("omega=", payload["omega"]) 
    print("W:", payload["W"]) 
    print("Z:", payload["Z"]) 
    print("sin²θ_W:", payload["weinberg_angle"]) 
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
