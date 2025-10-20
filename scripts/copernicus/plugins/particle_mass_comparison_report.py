#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粒子质量预测对比报告（不含Higgs）

目标：统一校验文档中粒子物理相关预测（轻子质量、夸克质量、弱玻色子、质子-中子质量差）与脚本实现是否一致。

来源实现：
- strict_formulas._calculate_lepton_masses
- strict_formulas._calculate_quark_masses
- strict_formulas._calculate_weinberg_angle
- strict_formulas._calculate_weak_boson_masses（若存在）
- strict_formulas._calculate_delta_mnp

输出：生成 scripts/copernicus/粒子质量预测对比报告.md 并打印摘要。
"""

from __future__ import annotations

import os
from typing import Dict, Any, List

try:
    from scripts.copernicus.models import predict_at_mu
except Exception:
    from ..models import predict_at_mu
try:
    from scripts.copernicus.plugins.strict_formulas import (
        _calculate_weinberg_angle,
        _calculate_delta_mnp,
    )
except Exception:
    from .strict_formulas import (
        _calculate_weinberg_angle,
        _calculate_delta_mnp,
    )


def rel_err(pred: float | None, ref: float | None) -> str:
    if pred is None or ref is None or ref == 0:
        return "-"
    return f"{abs(pred - ref) / abs(ref):.3%}"


def generate_report() -> str:
    # 关键能标选择（与文档一致）
    mu_L = 85.0       # 轻子映射标尺
    mu_MZ = 91.1876   # Z 玻色子标尺
    mu_hadron = 1.0   # 强子标尺
    mu_top = 173.0    # 顶夸克标尺

    params: Dict[str, float] = {}
    cfg: Dict[str, Any] = {}

    # 使用models.predict_at_mu以避免外部数据依赖（内部有fallback常数）
    lep_out = predict_at_mu(params, mu_L, ['m_e_MeV','m_mu_MeV','m_tau_MeV'], cfg)
    m_e = lep_out.get('m_e_MeV')
    m_mu = lep_out.get('m_mu_MeV')
    m_tau = lep_out.get('m_tau_MeV')

    top_out = predict_at_mu(params, mu_top, ['m_t_GeV'], cfg)
    m_t = top_out.get('m_t_GeV')
    dm_np = _calculate_delta_mnp(params, mu_hadron, cfg)  # MeV，总差

    # Weinberg角（sin²θ_W）
    sin2_w = _calculate_weinberg_angle(params, mu_MZ, cfg)

    # 参考值（来自 scripts/copernicus/tests/test_data.yaml 与相关文档）
    refs = {
        'm_e_MeV': 0.511,
        'm_mu_MeV': 105.66,
        'm_tau_MeV': 1776.86,
        'delta_mnp_MeV': 1.293,
        'sin2_theta_W': 0.231,
        'm_t_GeV': 172.76,
    }

    lines: List[str] = []
    lines.append("### 粒子质量预测对比报告（自动生成，不含Higgs）")
    lines.append("")
    lines.append("- 目的：检验文档中粒子物理常量（质量与相关量）的预测是否与脚本实现一致")
    lines.append("- 参考来源：`scripts/copernicus/tests/test_data.yaml` 与相关附录（7/8/9/10/21/27）")
    lines.append("")

    # 轻子质量
    lines.append("#### 轻子质量 (μ_L=85 GeV)")
    lines.append("| 量 | 预测 | 参考 | 相对误差 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| m_e (MeV)  | {m_e} | {refs['m_e_MeV']} | {rel_err(m_e, refs['m_e_MeV'])} |")
    lines.append(f"| m_μ (MeV)  | {m_mu} | {refs['m_mu_MeV']} | {rel_err(m_mu, refs['m_mu_MeV'])} |")
    lines.append(f"| m_τ (MeV)  | {m_tau} | {refs['m_tau_MeV']} | {rel_err(m_tau, refs['m_tau_MeV'])} |")
    lines.append("")

    # 质子-中子质量差
    lines.append("#### 质子-中子质量差 (μ≈1 GeV)")
    lines.append("| 量 | 预测 | 参考 | 相对误差 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Δm_np (MeV) | {dm_np} | {refs['delta_mnp_MeV']} | {rel_err(dm_np, refs['delta_mnp_MeV'])} |")
    lines.append("")

    # 顶夸克与Weinberg角
    lines.append("#### 顶夸克与温伯格角")
    lines.append("| 量 | 预测 | 参考 | 相对误差 | 标尺 |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| m_t (GeV) | {m_t} | {refs['m_t_GeV']} | {rel_err(m_t, refs['m_t_GeV'])} | {mu_top} |")
    lines.append(f"| sin²θ_W   | {sin2_w} | {refs['sin2_theta_W']} | {rel_err(sin2_w, refs['sin2_theta_W'])} | {mu_MZ} |")
    lines.append("")

    lines.append("说明：上述实现不涉猎Higgs质量；所有参考值来自项目文档与测试配置。")
    return "\n".join(lines)


def main():
    report = generate_report()
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "粒子质量预测对比报告.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("生成报告:", out_path)
    print()
    print(report)


if __name__ == "__main__":
    main()


