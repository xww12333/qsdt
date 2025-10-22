#!/usr/bin/env python3
"""
脚本名称: report_quark_compare.py
功能: 夸克质量闭环对比报告（表格 + Mermaid 误差占比）。
输入: study/outputs/quark_copernicus.json
输出: study/outputs/quark_compare.md
"""
from __future__ import annotations
import json, os

IN = "study/outputs/quark_copernicus.json"
OUT = "study/outputs/quark_compare.md"

d = json.load(open(IN, "r", encoding="utf-8"))
rows = []
for q in ["u","d","s","c","b","t"]:
    blk = d["quarks"][q]
    rows.append((q, blk["m_pred_GeV"], blk["m_obs_GeV"], blk["rel_err"]))

md = []
md.append("# 夸克质量对比（离散映射闭环）\n\n")
md.append("| 夸克 | 预测 [GeV] | 参考 [GeV] | 相对误差 |\n")
md.append("| --- | ---: | ---: | ---: |\n")
for q, pred, ref, err in rows:
    md.append(f"| {q} | {pred:.9g} | {ref:.9g} | {err*100.0:+.3f}% |\n")

# Mermaid：误差占比（绝对值）
md.append("\n```mermaid\n")
md.append("pie showData\n  title 夸克质量误差占比\n")
for q, _, _, err in rows:
    md.append(f"  \"{q}\" : {abs(err or 0.0):.6f}\n")
md.append("```\n")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("".join(md))
print("wrote:", OUT)
