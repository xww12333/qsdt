#!/usr/bin/env python3
"""
脚本名称: report_delta_mnp_compare.py
功能: Δm_np 分解对比报告（表格 + Mermaid 流程图）。
输入: study/outputs/np_split_copernicus.json
输出: study/outputs/delta_mnp_compare.md
"""
from __future__ import annotations
import json, os
from pathlib import Path

IN = Path(__file__).resolve().parents[2]/"study/outputs/np_split_copernicus.json"
OUT = Path(__file__).resolve().parents[2]/"study/outputs/delta_mnp_compare.md"

d = json.load(open(IN, "r", encoding="utf-8"))
blk = d.get("decomposition", {})

md = []
md.append("# Δm_np 分解与闭环\n\n")
md.append("| 项 | 数值 [MeV] |\n")
md.append("| --- | ---: |\n")
md.append(f"| ΔE_quark | {blk.get('Delta_E_quark_MeV'):.6f} |\n")
md.append(f"| ΔE_EM | {blk.get('Delta_E_EM_MeV'):.6f} |\n")
md.append(f"| ΔE_QCD | {blk.get('Delta_E_QCD_MeV'):.6f} |\n")
md.append(f"| Δm_np(pred) | {blk.get('Delta_m_np_pred_MeV'):.3f} |\n")

# Mermaid：分解流程图
md.append("\n```mermaid\n")
md.append("flowchart LR\n  A[ΔE_quark]-->D[Δm_np]\n  B[ΔE_EM]-->D\n  C[ΔE_QCD]-->D\n")
md.append(f"  A:::n -- {blk.get('Delta_E_quark_MeV'):.3f} MeV --> D\n")
md.append(f"  B:::n -- {blk.get('Delta_E_EM_MeV'):.3f} MeV --> D\n")
md.append(f"  C:::n -- {blk.get('Delta_E_QCD_MeV'):.3f} MeV --> D\n")
md.append("  classDef n fill:#ffe,stroke:#cc8,stroke-width:1px;\n")
md.append("```\n")

os.makedirs(OUT.parent, exist_ok=True)
OUT.write_text("".join(md), encoding="utf-8")
print("wrote:", OUT)
