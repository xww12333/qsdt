#!/usr/bin/env python3
"""
脚本名称: report_electroweak_compare.py
功能: 电弱 sin²θ_W 多路线对比报告（表格 + Mermaid 饼图 + 流程图）。
输入: study/outputs/electroweak_copernicus.json
输出: study/outputs/electroweak_routes_compare.md
"""
from __future__ import annotations
import json, os

IN = "study/outputs/electroweak_copernicus.json"
OUT = "study/outputs/electroweak_routes_compare.md"

ew = json.load(open(IN, "r", encoding="utf-8"))
w = ew.get("weinberg_angle", {})

rows = [
    ("on-shell", w.get("sin2_pred"), w.get("rel_err")),
    ("cosφ 修正", w.get("sin2_corr"), w.get("rel_err_corr")),
    ("QSDT 修正RG", w.get("sin2_run"), w.get("rel_err_run")),
    ("统一边界+RG", w.get("sin2_run_pred"), w.get("rel_err_run_pred")),
    ("几何投影+RG", w.get("sin2_corr_run"), w.get("rel_err_corr_run")),
    ("QSDT-omega(★)", w.get("sin2_qsdt_omega"), (w.get("sin2_qsdt_omega")-w.get("sin2_ref"))/w.get("sin2_ref") if w.get("sin2_qsdt_omega") and w.get("sin2_ref") else None),
]

md = []
md.append("# 电弱 sin²θ_W 多路线对比\n\n")
md.append("| 路线 | sin²θ_W | 相对误差 |\n")
md.append("| --- | ---: | ---: |\n")
for name, val, err in rows:
    if val is None: continue
    md.append(f"| {name} | {val:.6f} | {err*100.0:+.3f}% |\n")
try:
    sin2_on_err = w.get("sin2_on_shell_err")
    sin2_eff = w.get("sin2_eff_ref")
    md.append(f"\n> on-shell 不确定度（附录55）：±{sin2_on_err:.6f}，有效角参考：{sin2_eff}\n\n")
except Exception:
    pass

# Mermaid 图像化：误差占比饼图
errs = [(name, abs(err or 0.0)) for name, _, err in rows if err is not None]
md.append("\n```mermaid\n")
md.append("pie showData\n  title sin²θ_W 各路线误差占比\n")
for name, v in errs:
    md.append(f"  \"{name}\" : {v:.6f}\n")
md.append("```\n\n")

# Mermaid 流程图：从 W/Z 质量到各路线 sin²θ_W
md.append("```mermaid\n")
md.append("flowchart LR\n  A[W/Z 质量闭环] --> B(on-shell)\n  A --> C(cosφ 修正)\n  A --> D(QSDT 修正RG)\n  A --> E(统一边界+RG)\n  A --> F(几何投影+RG)\n")
def lab(v):
    return f"{v:.6f}" if v is not None else "—"
md.append(f"  B:::n -->|{lab(rows[0][1])}| G[sin²θ_W]\n")
md.append(f"  C:::n -->|{lab(rows[1][1])}| G\n")
md.append(f"  D:::n -->|{lab(rows[2][1])}| G\n")
md.append(f"  E:::n -->|{lab(rows[3][1])}| G\n")
md.append(f"  F:::n -->|{lab(rows[4][1])}| G\n")
md.append("  classDef n fill:#eef,stroke:#88f,stroke-width:1px;\n")
md.append("```\n")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("".join(md))
print("wrote:", OUT)
