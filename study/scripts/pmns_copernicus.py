#!/usr/bin/env python3
"""
脚本名称: pmns_copernicus.py
功能: PMNS 中微子混合的多路线关系计算（几何/相位/复合/正交/能标权重）。
作用:
- 以 (n_e,n_μ,n_τ) 与 g,β_Γ 为输入, 输出多路线的 U 与 (θ12,θ23,θ13) 角度（无参数, 教学探索）。
输入: 固定 (n_e,n_μ,n_τ) 与 g,β_Γ（脚本内常量）
输出: study/outputs/pmns_copernicus.json, study/outputs/讲义_PMNS_教学注解.md
使用方法: python3 study/scripts/pmns_copernicus.py
注意事项: 此模块为教学/探索, 非精准拟合；后续可替换为附录重叠函数。
相关附录: 13/15（PMNS）。
"""
from __future__ import annotations
import json, os

OUT_JSON = "study/outputs/pmns_copernicus.json"
OUT_MD = "study/outputs/讲义_PMNS_教学注解.md"

import math

# 简化：使用已确定的家族层级指数（来自轻子闭环）
N_E, N_MU, N_TAU = 18, 11, 7

def omega(g: float, beta_gamma: float) -> float:
    return (beta_gamma * (g**2) * math.log(2.0)) / (4.0 * math.pi * math.pi)

def route_geom(g: float, beta_gamma: float):
    # 几何重叠：w_ij = exp(-ω · |Δn|)，行归一（零参数）
    om = omega(g, beta_gamma)
    n = [N_E, N_MU, N_TAU]
    W = [[math.exp(-om * abs(n[i]-n[j])) for j in range(3)] for i in range(3)]
    # 行归一
    U = []
    for i in range(3):
        s = sum(W[i])
        U.append([W[i][j]/s for j in range(3)])
    return U

def route_phase(g: float):
    # 相位重叠：w_ij = |cos(φ_i - φ_j)|，φ_k = g · ln(2^n_k)；行归一
    ln2 = math.log(2.0)
    n = [N_E, N_MU, N_TAU]
    phi = [g * n_k * ln2 for n_k in n]
    W = [[abs(math.cos(phi[i]-phi[j])) for j in range(3)] for i in range(3)]
    U = []
    for i in range(3):
        s = sum(W[i])
        U.append([W[i][j]/s for j in range(3)])
    return U

def route_mixed(g: float, beta_gamma: float):
    # 复合重叠：U_mix = w_g * U_geom + (1-w_g) * U_phase，w_g = cosφ ≈ √(1 - g/2)
    try:
        w_g = math.sqrt(max(1.0 - 0.5 * float(g), 1e-12))
    except Exception:
        w_g = 0.9
    U_g = route_geom(g, beta_gamma)
    U_p = route_phase(g)
    U = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(w_g * U_g[i][j] + (1.0 - w_g) * U_p[i][j])
        s = sum(row)
        U.append([x / s for x in row])
    return U

def route_energy_weighted(g: float, beta_gamma: float):
    # 能标权重复合：用 ω 与 cosφ 混合权重（零参数）
    ln2 = math.log(2.0)
    om = omega(g, beta_gamma)
    # 归一权重 w1:w2 = om : (1-om) 与 cosφ : (1-cosφ)
    try:
        cosphi = math.sqrt(max(1.0 - 0.5 * float(g), 1e-12))
    except Exception:
        cosphi = 0.9
    w_g1, w_p1 = om, max(0.0, 1.0 - om)
    w_g2, w_p2 = cosphi, max(0.0, 1.0 - cosphi)
    wg = (w_g1 + w_g2) / 2.0
    wp = (w_p1 + w_p2) / 2.0
    s = wg + wp or 1.0
    wg, wp = wg/s, wp/s
    Ug = route_geom(g, beta_gamma)
    Up = route_phase(g)
    U = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(wg*Ug[i][j] + wp*Up[i][j])
        srow = sum(row) or 1.0
        U.append([x/srow for x in row])
    return U
def angles_from_U(U):
    # 近似提取角：s13 = |U_e3|；s12 = |U_e2|/sqrt(1-s13^2)；s23 = |U_μ3|/sqrt(1-s13^2)
    Ue3 = abs(U[0][2])
    s13 = max(0.0, min(1.0, Ue3))
    c13 = math.sqrt(max(0.0, 1.0 - s13*s13))
    Ue2 = abs(U[0][1])
    Um3 = abs(U[1][2])
    s12 = max(0.0, min(1.0, Ue2 / (c13 if c13>1e-12 else 1.0)))
    s23 = max(0.0, min(1.0, Um3 / (c13 if c13>1e-12 else 1.0)))
    th12 = math.degrees(math.asin(s12))
    th23 = math.degrees(math.asin(s23))
    th13 = math.degrees(math.asin(s13))
    return {"theta12_deg": th12, "theta23_deg": th23, "theta13_deg": th13, "delta_CP_deg": None}

def route_mixed_ortho(g: float, beta_gamma: float):
    # 对 mixed 矩阵做简单列Gram–Schmidt以近似酉化
    U = route_mixed(g, beta_gamma)
    # 提取列向量
    import math as _m
    cols = [[U[i][j] for i in range(3)] for j in range(3)]
    # 归一
    def norm(v):
        return _m.sqrt(sum(x*x for x in v))
    def dot(a,b):
        return sum(x*y for x,y in zip(a,b))
    def sub(a,b,coef):
        return [x - coef*y for x,y in zip(a,b)]
    v1 = cols[0]
    n1 = norm(v1) or 1.0
    e1 = [x/n1 for x in v1]
    v2 = cols[1]
    v2p = sub(v2, e1, dot(v2,e1))
    n2 = norm(v2p) or 1.0
    e2 = [x/n2 for x in v2p]
    v3 = cols[2]
    v3p = sub(sub(v3, e1, dot(v3,e1)), e2, dot(v3,e2))
    n3 = norm(v3p) or 1.0
    e3 = [x/n3 for x in v3p]
    # 回组 U，取绝对值作为重叠强度
    Uo = [[abs(e1[i]), abs(e2[i]), abs(e3[i])] for i in range(3)]
    # 每行归一为概率分布
    for i in range(3):
        s = sum(Uo[i]) or 1.0
        Uo[i] = [x/s for x in Uo[i]]
    return Uo

def run():
    # 使用关系不变量（与其它 study 脚本一致）
    g = 0.223
    beta_gamma = 3.75
    U_geom = route_geom(g, beta_gamma)
    U_phase = route_phase(g)
    U_mixed = route_mixed(g, beta_gamma)
    U_mixed_ortho = route_mixed_ortho(g, beta_gamma)
    U_energy = route_energy_weighted(g, beta_gamma)
    ang_geom = angles_from_U(U_geom)
    ang_phase = angles_from_U(U_phase)
    ang_mixed = angles_from_U(U_mixed)
    ang_mixed_ortho = angles_from_U(U_mixed_ortho)
    ang_energy = angles_from_U(U_energy)
    payload = {
        "routes": {
            "geom_exp_omega": {"U": U_geom, "angles": ang_geom},
            "phase_cos": {"U": U_phase, "angles": ang_phase},
            "mixed": {"U": U_mixed, "angles": ang_mixed},
            "mixed_ortho": {"U": U_mixed_ortho, "angles": ang_mixed_ortho},
            "energy_weighted": {"U": U_energy, "angles": ang_energy}
        },
        "notes": {
            "principle": "多路关系：几何重叠与相位重叠（零参数）；由 (n_e,n_μ,n_τ) 与 g,β_Γ 决定",
            "status": "experimental",
        }
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    md = []
    md.append("# 讲义：PMNS 中微子混合\n\n")
    md.append("## 观察者视角\n- 弱通道下，轻子味态与中微子质量态的关系重叠显化为 U_PMNS。\n\n")
    md.append("## 核心关系（零参数）\n- 路线A：几何重叠 U_ij ∝ e^{-ω|Δn|}（ω=g^2β_Γ ln2 / 4π²）\n- 路线B：相位重叠 U_ij ∝ |cos(φ_i-φ_j)|，φ_k=g·ln 2^{n_k}\n\n")
    md.append("## 航海步骤\n- 多路并行：A/B 各自给出 θ_ij，自动对比最优闭环；不调参，仅改关系。\n")
    open(OUT_MD, "w", encoding="utf-8").write("".join(md))
    print("wrote:", OUT_JSON, OUT_MD)

if __name__ == "__main__":
    run()
