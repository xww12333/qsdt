#!/usr/bin/env python3
"""
脚本名称: mass_spectrum.py
功能: 质量谱计算器（关系先行, 零自由参数）。
作用:
- 提供 omega/层级律/二进制瀑布/复合模式的质量谱计算, 支持范围/单点与导出 JSON/CSV。
输入: 命令行参数（g, β_Γ, m0, units, N/N_min/N_max/cascade_n）。
输出: 控制台摘要, 可选 JSON/CSV（study/outputs）。
使用方法: python3 study/scripts/mass_spectrum.py --help（或在 lepton/electroweak 模块中被动使用）
注意事项: 不含任何拟合; 修正项须另行定义显式关系函数。
相关附录: 质量量子化总述。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Optional


LN2 = math.log(2.0)
PI2_4 = 4.0 * math.pi * math.pi


@dataclass
class Inputs:
    g: float
    beta_gamma: float
    m0: float  # 基准质量（由 --units 指定单位）
    units: str = "GeV"  # m0 的基准单位
    N: Optional[int] = None  # 计算单个层级指数
    N_min: Optional[int] = None  # 层级范围起（含）
    N_max: Optional[int] = None  # 层级范围止（含）
    cascade_n: Optional[int] = None  # 二进制瀑布深度 n（可选）


@dataclass
class SpectrumPoint:
    N: int
    m_in_units: float  # 按基准单位表示
    m_GeV: float
    m_MeV: float


def compute_omega(g: float, beta_gamma: float) -> float:
    return (beta_gamma * (g ** 2) * LN2) / PI2_4


def m_layer(m0: float, omega: float, N: int) -> float:
    return m0 * ((1.0 + omega) ** (N / 2.0))


def to_GeV(value: float, units: str) -> float:
    if units.lower() == "gev":
        return value
    if units.lower() == "mev":
        return value / 1000.0
    raise ValueError(f"Unsupported units: {units}")


def to_MeV(value: float, units: str) -> float:
    if units.lower() == "gev":
        return value * 1000.0
    if units.lower() == "mev":
        return value
    raise ValueError(f"Unsupported units: {units}")


def build_points(inp: Inputs) -> List[SpectrumPoint]:
    omega = compute_omega(inp.g, inp.beta_gamma) if (inp.g is not None and inp.beta_gamma is not None) else None
    gevs = []
    # 若提供二进制瀑布，则先计算模族参考质量 m_ref
    m_ref = inp.m0
    if inp.cascade_n is not None:
        m_ref = m_ref / (2.0 ** int(inp.cascade_n))

    if omega is not None and (inp.N is not None or (inp.N_min is not None and inp.N_max is not None)):
        if inp.N is not None:
            Ns = [inp.N]
        else:
            Ns = list(range(int(inp.N_min), int(inp.N_max) + 1))
        for N in Ns:
            mN = m_layer(m_ref, omega, N)
            mG = to_GeV(mN, inp.units)
            mM = to_MeV(mN, inp.units)
            gevs.append(SpectrumPoint(N=N, m_in_units=mN, m_GeV=mG, m_MeV=mM))
    else:
        # 未提供层级律参数：仅输出一个点（N 取瀑布深度或 0）
        N = int(inp.cascade_n) if inp.cascade_n is not None else 0
        mG = to_GeV(m_ref, inp.units)
        mM = to_MeV(m_ref, inp.units)
        gevs.append(SpectrumPoint(N=N, m_in_units=m_ref, m_GeV=mG, m_MeV=mM))
    return gevs


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(path: str, rows: List[SpectrumPoint]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("N,m_in_units,m_GeV,m_MeV\n")
        for r in rows:
            f.write(f"{r.N},{r.m_in_units:.12g},{r.m_GeV:.12g},{r.m_MeV:.12g}\n")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="基于关系（零自由参数）的质量谱计算")
    p.add_argument("--g", type=float, help="关系不变量 g (Γ/J)")
    p.add_argument("--beta-gamma", type=float, dest="beta_gamma", help="关系不变量 β_Γ")
    p.add_argument("--m0", type=float, required=True, help="模族基准尺度 m0（单位由 --units 指定）")
    p.add_argument("--units", default="GeV", choices=["GeV", "MeV"], help="m0 与基准输出单位")
    p.add_argument("--N", type=int, help="单个层级指数 N")
    p.add_argument("--N-min", dest="N_min", type=int, help="层级范围最小值（含）")
    p.add_argument("--N-max", dest="N_max", type=int, help="层级范围最大值（含）")
    p.add_argument("--cascade-n", dest="cascade_n", type=int, help="二进制瀑布深度 n（E_n = m0/2^n）")
    p.add_argument("--out-json", help="可选：JSON 输出路径")
    p.add_argument("--out-csv", help="可选：CSV 输出路径")

    args = p.parse_args(argv)

    inp = Inputs(
        g=args.g,
        beta_gamma=args.beta_gamma,
        m0=args.m0,
        units=args.units,
        N=args.N,
        N_min=args.N_min,
        N_max=args.N_max,
        cascade_n=args.cascade_n,
    )
    omega = compute_omega(inp.g, inp.beta_gamma) if (inp.g is not None and inp.beta_gamma is not None) else None
    pts = build_points(inp)

    # Print concise summary
    print("关系先行质量谱")
    if omega is not None:
        print(f"  g={inp.g}, beta_gamma={inp.beta_gamma}, omega={omega:.8g}")
    if inp.cascade_n is not None:
        print(f"  二进制瀑布：n={inp.cascade_n} (E_n = m0/2^n)")
    if inp.N is not None:
        r = pts[0]
        print(f"  N={r.N}: m={r.m_in_units:.9g} {inp.units}  |  {r.m_GeV:.9g} GeV  |  {r.m_MeV:.9g} MeV")
    elif inp.N_min is not None and inp.N_max is not None:
        print(f"  N ∈ [{inp.N_min}, {inp.N_max}], m0={inp.m0} {inp.units}")
    else:
        r = pts[0]
        print(f"  （未使用层级律）N={r.N}: m={r.m_in_units:.9g} {inp.units} | {r.m_GeV:.9g} GeV | {r.m_MeV:.9g} MeV")

    payload = {
        "inputs": asdict(inp),
        "omega": omega,
        "spectrum": [asdict(r) for r in pts],
        "notes": {
            "principle": "零自由参数；所有输入均为关系不变量或关系锚定尺度。",
            "formula": "omega=(beta_gamma*g^2*ln2)/(4*pi^2); 层级律: m=m_ref*(1+omega)^(N/2); 二进制瀑布: E_n=m0/2^n",
        },
    }

    if args.out_json:
        save_json(args.out_json, payload)
        print(f"  wrote JSON -> {args.out_json}")
    if args.out_csv:
        save_csv(args.out_csv, pts)
        print(f"  wrote CSV  -> {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
