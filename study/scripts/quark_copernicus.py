#!/usr/bin/env python3
"""
脚本名称: quark_copernicus.py
功能: 六夸克质量的离散映射闭环（零自由参数）。
作用:
- 对每个 m_obs 反推 (n), 通过层级律整数 (N) 放大, 使 m_ref*(1+ω)^(N/2) 逼近观测；输出 JSON 与统一报告使用的信息。
输入: m0,g,β_Γ（关系不变量/锚定尺度）
输出: study/outputs/quark_copernicus.json
使用方法: python3 study/scripts/quark_copernicus.py --out study/outputs/quark_copernicus.json
注意事项: 本脚本不加入 QCD 等修正；待附录关系式补齐后以显式函数加入。
相关附录: 质量量子化/强子章节。
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List


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


@dataclass
class Inputs:
    m0: float
    g: float
    beta_gamma: float
    # 观测质量（GeV）
    m_u_obs: float = 0.0022e0  # 2.2 MeV
    m_d_obs: float = 0.0047e0  # 4.7 MeV
    m_s_obs: float = 0.095e0   # 95 MeV
    m_c_obs: float = 1.27
    m_b_obs: float = 4.18
    m_t_obs: float = 172.76


def run(inp: Inputs) -> Dict:
    omg = omega_from(inp.g, inp.beta_gamma)
    data: List[Tuple[str, float]] = [
        ("u", inp.m_u_obs),
        ("d", inp.m_d_obs),
        ("s", inp.m_s_obs),
        ("c", inp.m_c_obs),
        ("b", inp.m_b_obs),
        ("t", inp.m_t_obs),
    ]

    out: Dict[str, Dict] = {}
    for name, m_obs in data:
        n = nearest_under_n(inp.m0, m_obs)
        m_ref = cascade(inp.m0, n)
        N, m_pred = infer_integer_N(m_ref, m_obs, omg)
        out[name] = {
            "n": n,
            "N": N,
            "m_ref_GeV": m_ref,
            "m_pred_GeV": m_pred,
            "m_obs_GeV": m_obs,
            "rel_err": (m_pred - m_obs) / m_obs,
        }

    return {
        "inputs": asdict(inp),
        "omega": omg,
        "quarks": out,
        "notes": {
            "principle": "反推离散映射 (n,N)，正推对比；零自由参数",
            "todo": "待加入基于关系式的 QCD/几何修正",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="夸克质量的哥白尼式推演（关系先行，零自由参数）")
    ap.add_argument("--m0", type=float, default=125.1, help="模族基准 m0 [GeV]")
    ap.add_argument("--g", type=float, default=0.223, help="关系不变量 g (Γ/J)")
    ap.add_argument("--beta-gamma", dest="beta_gamma", type=float, default=3.75, help="关系不变量 β_Γ")
    ap.add_argument("--out", type=str, default="study/outputs/quark_copernicus.json", help="输出 JSON")
    args = ap.parse_args()

    payload = run(Inputs(m0=args.m0, g=args.g, beta_gamma=args.beta_gamma))
    print("omega=", payload["omega"]) 
    for k, row in payload["quarks"].items():
        print(f"{k:2s}", {kk: row[kk] for kk in row if kk in ("n","N","m_pred_GeV","m_obs_GeV","rel_err")})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
