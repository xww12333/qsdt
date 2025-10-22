#!/usr/bin/env python3
"""
脚本名称: report_sin2_scan.py
功能: sin²θ_W 多口径扫描示意（在不同 μ 标签下展示恒定结果, 仅供教学说明）。
输入: electroweak_copernicus.run（内部调用）
输出: study/outputs/sin2_theta_scan.md
"""
from __future__ import annotations
import json, math, os
from electroweak_copernicus import Inputs, run

outfile = "study/outputs/sin2_theta_scan.md"

mus = [0.000511, 0.10566, 1.0, 10.0, 85.0, 91.1876, 246.0, 1000.0]

rows = []
for mu in mus:
    payload = run(Inputs(m0=125.1, g=0.223, beta_gamma=3.75))
    # override observation scale to mu for W/Z? Here sin2 is computed at M_Z based on W,Z masses.
    # For scan, we treat reported sin2_pred (on-shell), sin2_corr (cosphi), sin2_run (EW RG) as representatives.
    w = payload["weinberg_angle"]
    rows.append((mu, w.get("sin2_pred"), w.get("sin2_corr"), w.get("sin2_run")))

md = []
md.append("# sin²θ_W 多口径扫描（示意）\n\n")
md.append("| μ (GeV) | on-shell | cosφ 修正 | QSDT修正RG |\n")
md.append("| ---: | ---: | ---: | ---: |\n")
for mu, a, b, c in rows:
    md.append(f"| {mu} | {a:.6f} | {b:.6f} | {c:.6f} |\n")

os.makedirs(os.path.dirname(outfile), exist_ok=True)
open(outfile, "w", encoding="utf-8").write("".join(md))
print("wrote:", outfile)
