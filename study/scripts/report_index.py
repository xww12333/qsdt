#!/usr/bin/env python3
"""
脚本名称: report_index.py
功能: 多路径对比教学索引页生成。
输入: 已生成的各 compare.md 文件
输出: study/outputs/讲义_多路径对比索引.md
"""
from __future__ import annotations
import os

OUT = "study/outputs/讲义_多路径对比索引.md"

files = [
    ("电弱 sin²θ_W 多路线", "electroweak_routes_compare.md"),
    ("PMNS 多路线", "pmns_routes_compare.md"),
    ("μ子 g−2 多路线", "amu_routes_compare.md"),
    ("电子质量多路线", "electron_routes_compare.md"),
    ("夸克质量对比", "quark_compare.md"),
    ("Δm_np 分解闭环", "delta_mnp_compare.md"),
]

md = []
md.append("# 多路径对比教学索引\n\n")
for title, path in files:
    md.append(f"- {title}：{path}\n")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("".join(md))
print("wrote:", OUT)
