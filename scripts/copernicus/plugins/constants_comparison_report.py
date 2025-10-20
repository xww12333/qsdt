#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
常数预测对比报告脚本（不涉及Higgs）

功能：
- 在关键能标下计算 α_em(μ)、α_s(μ)、sin²θ_W、G_F
- 与文档/配置中的参考值进行比对
- 生成 Markdown 报告输出到 scripts/copernicus/常数预测对比报告.md

说明：
- α_em/α_s 采用已有一回路近似与阈值；可通过pipeline校准（若启用）
- sin²θ_W 使用现有公式实现（内部固定g、g'于M_Z）
- G_F 使用标准关系 G_F = 1/(√2 v²), v=246 GeV
"""

from __future__ import annotations

import os
import math
from typing import Dict, List

try:
    from scripts.copernicus.models import run_alpha_qed, run_alpha_s
except Exception:
    from ..models import run_alpha_qed, run_alpha_s

try:
    from scripts.copernicus.plugins.strict_formulas import _calculate_weinberg_angle as calc_theta_w
except Exception:
    from .strict_formulas import _calculate_weinberg_angle as calc_theta_w


def compute_constants(mu_GeV: float) -> Dict[str, float | None]:
    try:
        alpha_em = run_alpha_qed(mu_GeV, None)
    except Exception:
        alpha_em = None
    try:
        alpha_s = run_alpha_s(mu_GeV, None)
    except Exception:
        alpha_s = None
    try:
        sin2_theta_w = calc_theta_w({}, mu_GeV, {})
    except Exception:
        sin2_theta_w = None
    try:
        v = 246.0
        G_F = 1.0 / (math.sqrt(2.0) * v * v)
    except Exception:
        G_F = None
    return {
        "alpha_em": alpha_em,
        "alpha_s": alpha_s,
        "sin2_theta_w": sin2_theta_w,
        "G_F_GeV^-2": G_F,
    }


def format_row(mu: float, pred: Dict[str, float | None], ref: Dict[str, float | None]) -> str:
    def rel_err(p: float | None, r: float | None) -> str:
        if p is None or r is None or r == 0:
            return "-"
        return f"{abs(p - r) / abs(r):.3%}"

    return (
        f"| {mu:.6g} | "
        f"{pred['alpha_em'] if pred['alpha_em'] is not None else '-'} | "
        f"{ref.get('alpha_em','-')} | {rel_err(pred['alpha_em'], ref.get('alpha_em'))} | "
        f"{pred['alpha_s'] if pred['alpha_s'] is not None else '-'} | "
        f"{ref.get('alpha_s','-')} | {rel_err(pred['alpha_s'], ref.get('alpha_s'))} | "
        f"{pred['sin2_theta_w'] if pred['sin2_theta_w'] is not None else '-'} | "
        f"{ref.get('sin2_theta_w','-')} | {rel_err(pred['sin2_theta_w'], ref.get('sin2_theta_w'))} | "
        f"{pred['G_F_GeV^-2'] if pred['G_F_GeV^-2'] is not None else '-'} | "
        f"{ref.get('G_F_GeV^-2','-')} | {rel_err(pred['G_F_GeV^-2'], ref.get('G_F_GeV^-2'))} |"
    )


def generate_report() -> str:
    mus = [246.0, 91.1876, 85.0, 1.0]

    # 参考值汇总（来自文档/测试配置）
    refs: Dict[float, Dict[str, float]] = {
        91.1876: {
            "alpha_em": 0.0072973525693,  # 1/127.955
            "alpha_s": 0.1181,
            "sin2_theta_w": 0.23142,      # 文档给出的QSDT理论预测
            "G_F_GeV^-2": 1.1663787e-5,   # 实验精确值
        },
        246.0: {
            # 若无权威参考，留空，仅做展示
        },
        85.0: {},
        1.0: {},
    }

    lines: List[str] = []
    lines.append("### 常数预测对比报告（自动生成）")
    lines.append("")
    lines.append("- 目标：检验文档中物理常数预测是否与脚本输出一致（不含Higgs）")
    lines.append("- 参考来源：`scripts/copernicus/tests/test_data.yaml`、`量子空间动力学-附录21.md`、`量子空间动力学-附录26.md`")
    lines.append("")
    lines.append("| μ (GeV) | α_em(μ) 预测 | α_em 参考 | 相对误差 | α_s(μ) 预测 | α_s 参考 | 相对误差 | sin²θ_W 预测 | sin²θ_W 参考 | 相对误差 | G_F 预测 (GeV^-2) | G_F 参考 | 相对误差 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for mu in mus:
        pred = compute_constants(mu)
        ref = refs.get(mu, {})
        lines.append(format_row(mu, pred, ref))

    lines.append("")
    lines.append("说明：参考值缺失处仅展示预测，不做误差评估；完整标定需后续补充。")
    return "\n".join(lines)


def main():
    report_md = generate_report()
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "常数预测对比报告.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print("生成报告:", out_path)
    print()
    print(report_md)


if __name__ == "__main__":
    main()


