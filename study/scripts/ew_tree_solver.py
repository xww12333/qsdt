#!/usr/bin/env python3
"""
脚本名称: ew_tree_solver.py
功能: 用树级/含 Δr 的标准关系, 由 (α, G_F, m_Z) 联立求解 (sin²θ_W, m_W)。
作用:
- 作为电弱的“标准对照”模块，与航海多路径结果做横向比较（教学/审查）。
输入: 常量 α、G_F、m_Z（脚本内默认, 可修改）
输出: study/outputs/ew_tree_solutions.json/.md
使用方法: python3 study/scripts/ew_tree_solver.py
注意事项: 本模块为对照, 不参与最优路线选择; Δr 仅演示 0/0.02 两个示例。
相关附录: 56 标准推导。
"""
from __future__ import annotations
import math, json, os

OUT_JSON = "study/outputs/ew_tree_solutions.json"
OUT_MD = "study/outputs/ew_tree_solutions.md"

def solve_tree(alpha: float, Gf: float, mZ: float):
    # s^2 c^2 = A^2, A^2 = pi*alpha /(sqrt(2) Gf mZ^2)
    A2 = math.pi * alpha / (math.sqrt(2.0) * Gf * (mZ**2))
    disc = max(0.0, 1.0 - 4.0 * A2)
    s2_1 = 0.5 * (1.0 - math.sqrt(disc))
    s2_2 = 0.5 * (1.0 + math.sqrt(disc))
    # physical root: s2 ~ 0.23
    s2 = s2_1 if s2_1 < 0.5 else s2_2
    c2 = 1.0 - s2
    mW = mZ * math.sqrt(c2)
    return s2, mW

def solve_with_delta_r(alpha: float, Gf: float, mZ: float, delta_r: float):
    # mW^2 (1 - mW^2/mZ^2) = pi alpha /(sqrt(2) Gf) * 1/(1-Delta r)
    RHS = math.pi * alpha / (math.sqrt(2.0) * Gf) / max(1e-12, (1.0 - delta_r))
    # solve for x = mW^2: x (1 - x/mZ^2) - RHS = 0 -> -x^2/mZ^2 + x - RHS = 0
    a = -1.0 / (mZ*mZ)
    b = 1.0
    c = -RHS
    disc = b*b - 4*a*c
    if disc < 0: disc = 0.0
    x1 = (-b + math.sqrt(disc)) / (2*a)
    x2 = (-b - math.sqrt(disc)) / (2*a)
    x = max(x1, x2) if max(x1, x2) > 0 else min(x1, x2)
    if x <= 0: x = RHS  # fallback
    mW = math.sqrt(x)
    s2 = 1.0 - (mW*mW)/(mZ*mZ)
    return s2, mW

def main():
    # default inputs
    alpha = 1.0/128.0
    Gf = 1.1663787e-5  # GeV^-2
    mZ = 91.1876       # GeV
    s2_tree, mW_tree = solve_tree(alpha, Gf, mZ)
    # with delta r examples
    s2_dr0, mW_dr0 = solve_with_delta_r(alpha, Gf, mZ, 0.0)
    s2_dr02, mW_dr02 = solve_with_delta_r(alpha, Gf, mZ, 0.02)

    out = {
        "inputs": {"alpha": alpha, "G_F": Gf, "m_Z_GeV": mZ},
        "tree": {"sin2": s2_tree, "m_W_GeV": mW_tree},
        "with_delta_r": {
            "delta_r=0.0": {"sin2": s2_dr0, "m_W_GeV": mW_dr0},
            "delta_r=0.02": {"sin2": s2_dr02, "m_W_GeV": mW_dr02}
        }
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    md = []
    md.append("# 电弱树级/含Δr 联立求解\n\n")
    md.append(f"输入：α≈{alpha}, G_F={Gf}, m_Z={mZ} GeV\n\n")
    md.append("| 口径 | sin²θ_W | m_W [GeV] |\n| --- | ---: | ---: |\n")
    md.append(f"| 树级 | {s2_tree:.6f} | {mW_tree:.6f} |\n")
    md.append(f"| Δr=0.0 | {s2_dr0:.6f} | {mW_dr0:.6f} |\n")
    md.append(f"| Δr=0.02 | {s2_dr02:.6f} | {mW_dr02:.6f} |\n")
    open(OUT_MD, "w", encoding="utf-8").write("".join(md))
    print("wrote:", OUT_JSON, OUT_MD)

if __name__ == "__main__":
    main()
