#!/usr/bin/env python3
"""
脚本名称: report_amu_compare.py
功能: μ子 g−2 多路线对比报告（含 Mermaid 图表）。
输入: study/outputs/amu_copernicus.json
输出: study/outputs/amu_routes_compare.md
使用: python3 study/scripts/report_amu_compare.py
注意: 仅负责可视化与对比，不做数值计算。
"""
from __future__ import annotations
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "study/outputs/amu_copernicus.json"
OUT = ROOT / "study/outputs/amu_routes_compare.md"

amu = json.load(open(IN, "r", encoding="utf-8"))
md = []
md.append("# μ子 g−2 多路线对比\n\n")
md.append("| 路线 | a_μ | 备注 |\n")
md.append("| --- | ---: | --- |")
for label, blk in amu.get("routes", {}).items():
    val = blk.get("a_mu")
    note = ", ".join(f"{k}={v}" for k,v in (blk.get("weights") or {}).items())
    md.append(f"\n| {label} | {val} | {note} |")

# Mermaid：数值占比饼图（归一到 Schwinger 基准）
base = amu.get("routes", {}).get("schwinger", {}).get("a_mu") or 1.0
md.append("\n\n```mermaid\npie showData\n  title a_μ 各路线相对基准占比\n")
for label, blk in amu.get("routes", {}).items():
    val = (blk.get("a_mu") or 0.0) / base
    md.append(f"  \"{label}\" : {val:.6f}\n")
md.append("```\n")

os.makedirs(OUT.parent, exist_ok=True)
OUT.write_text("".join(md), encoding="utf-8")
print("wrote:", str(OUT))
