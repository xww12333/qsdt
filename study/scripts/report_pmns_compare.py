#!/usr/bin/env python3
"""
脚本名称: report_pmns_compare.py
功能: PMNS 多路线角度对比报告（表格 + Mermaid 节点图）。
输入: study/outputs/pmns_copernicus.json
输出: study/outputs/pmns_routes_compare.md
"""
from __future__ import annotations
import json, os

IN = "study/outputs/pmns_copernicus.json"
OUT = "study/outputs/pmns_routes_compare.md"

pmns = json.load(open(IN, "r", encoding="utf-8"))
md = []
md.append("# PMNS 多路线对比\n\n")
md.append("| 路线 | θ12 (°) | θ23 (°) | θ13 (°) |\n")
md.append("| --- | ---: | ---: | ---: |\n")
for label, blk in pmns.get("routes", {}).items():
    ang = blk.get("angles", {})
    md.append(f"| {label} | {ang.get('theta12_deg'):.3f} | {ang.get('theta23_deg'):.3f} | {ang.get('theta13_deg'):.3f} |\n")

# Mermaid：路线节点图
md.append("\n```mermaid\n")
md.append("flowchart LR\n  U[PMNS 关系] --> A(geom)\n  U --> B(phase)\n  U --> C(mixed)\n  U --> D(mixed_ortho)\n  U --> E(energy_weighted)\n")
def lab(x):
    return f"{x:.2f}°"
for label, blk in pmns.get("routes", {}).items():
    ang = blk.get("angles", {})
    t12, t23, t13 = ang.get('theta12_deg'), ang.get('theta23_deg'), ang.get('theta13_deg')
    if t12 is None: continue
    md.append(f"  {label[0].upper()}:::n -->|θ12={lab(t12)}, θ23={lab(t23)}, θ13={lab(t13)}| F[展示] \n")
md.append("  classDef n fill:#efe,stroke:#484,stroke-width:1px;\n")
md.append("```\n")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("".join(md))
print("wrote:", OUT)
