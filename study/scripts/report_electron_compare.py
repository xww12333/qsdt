#!/usr/bin/env python3
"""
脚本名称: report_electron_compare.py
功能: 电子质量多路线对比报告（表格 + Mermaid 饼图 + 附录54不确定度带）。
输入: study/outputs/lepton_copernicus.json
输出: study/outputs/electron_routes_compare.md
"""
from __future__ import annotations
import json, os

IN = "study/outputs/lepton_copernicus.json"
OUT = "study/outputs/electron_routes_compare.md"

d = json.load(open(IN, "r", encoding="utf-8"))
e = d["electron"]
obs = e["m_obs_GeV"]*1e3
rows = [
    ("cascade", e["m_pred_GeV"]*1e3, e["rel_err"]),
    ("cosφ 修正", e.get("m_pred_corr_GeV", 0.0)*1e3, e.get("rel_err_corr")),
    ("g_eff 关系式", e.get("m_pred_geff_GeV", 0.0)*1e3, e.get("rel_err_geff")),
    ("附录54 树级", e.get("m_appendix54_tree_GeV", 0.0)*1e3, e.get("rel_err_appendix54_tree")),
    ("附录54 统一修正", e.get("m_appendix54_uni_GeV", 0.0)*1e3, e.get("rel_err_appendix54_uni")),
]

md = []
md.append("# 电子质量多路线对比\n\n")
md.append(f"参考：{obs:.6f} MeV\n\n")
md.append("| 路线 | 数值 [MeV] | 相对误差 |\n")
md.append("| --- | ---: | ---: |\n")
for name, val, err in rows:
    md.append(f"| {name} | {val:.6f} | {err*100.0:+.3f}% |\n")

# Mermaid：误差占比饼图
md.append("\n```mermaid\n")
md.append("pie showData\n  title 电子质量各路线误差占比\n")
for name, val, err in rows:
    md.append(f"  \"{name}\" : {abs(err or 0.0):.6f}\n")
md.append("```\n")

# 附：附录54 统一修正不确定度带
try:
    uni_min = e.get("m_appendix54_uni_min_GeV")*1e3
    uni_max = e.get("m_appendix54_uni_max_GeV")*1e3
    md.append(f"\n> 附录54 统一修正带：[{uni_min:.6f}, {uni_max:.6f}] MeV\n")
except Exception:
    pass

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("".join(md))
print("wrote:", OUT)
