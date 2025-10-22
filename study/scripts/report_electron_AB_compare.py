#!/usr/bin/env python3
"""
脚本名称: report_electron_AB_compare.py
功能: 电子 A/B 修正（cosφ 与 g_eff）路线对比报告。
输入: study/outputs/lepton_copernicus.json
输出: study/outputs/electron_AB_compare.md
"""
from __future__ import annotations
import json, os

infile = "study/outputs/lepton_copernicus.json"
outfile = "study/outputs/electron_AB_compare.md"

data = json.load(open(infile, "r", encoding="utf-8"))
e = data["electron"]
base = e["m_pred_GeV"]*1e3
corA = e.get("m_pred_corr_GeV", 0.0)*1e3
corB = e.get("m_pred_geff_GeV", 0.0)*1e3
obs  = e["m_obs_GeV"]*1e3

def relerr(x):
    return (x-obs)/obs*100.0

md = []
md.append("# 电子质量：A/B 修正对比\n")
md.append("输入：m0=125.1 GeV, g=0.223, β_Γ=3.75 → ω=%.10f\n" % data["omega"]) 
md.append("\n")
md.append("| 口径 | 数值 [MeV] | 相对误差 |\n")
md.append("| --- | ---: | ---: |\n")
md.append(f"| 基准（二进制瀑布） | {base:.6f} | {relerr(base):+.3f}% |\n")
md.append(f"| 修正A（cosφ≈√(1−g/2)） | {corA:.6f} | {relerr(corA):+.3f}% |\n")
md.append(f"| 修正B（g_eff 关系式） | {corB:.6f} | {relerr(corB):+.3f}% |\n")
md.append(f"| 观测 | {obs:.6f} | — |\n")

os.makedirs(os.path.dirname(outfile), exist_ok=True)
open(outfile, "w", encoding="utf-8").write("".join(md))
print("wrote:", outfile)
