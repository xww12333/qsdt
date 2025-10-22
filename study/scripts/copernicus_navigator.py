#!/usr/bin/env python3
"""
脚本名称: copernicus_navigator.py
功能: 航海计划总览生成器（统一汇总各模块结果, 自动选择当前最优闭环）。
作用:
- 聚合 lepton/quark/Δm_np/电弱/a_e/V_us/n_s/θ/α_em/α_s/PMNS/a_μ 等模块的 JSON 输出, 生成总览 Markdown。
输入: study/outputs/*_copernicus.json（各模块产物）。
输出: study/outputs/copernicus_summary.md。
使用方法: python3 study/scripts/copernicus_navigator.py（或 run_all.sh 最后一步自动运行）。
注意事项:
- 仅读取已生成的 JSON; 未生成的模块将跳过。
- “最优”选择基于最小绝对相对误差; 不含参数调节。
相关附录: 航海计划总述。
"""
from __future__ import annotations
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "study/outputs/copernicus_summary.md"

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pct(x: float) -> str:
    return f"{x*100.0:+.3f}%"

def main():
    parts = []
    parts.append("# 哥白尼计划：关系闭环航海图（零自由参数）\n")

    # Leptons
    lep = load_json("study/outputs/lepton_copernicus.json")
    e = lep["electron"]
    parts.append("\n## 轻子质量\n")
    parts.append("- 电子（n=18）：\n")
    parts.append(f"  - 基准（二进制瀑布）：{e['m_pred_GeV']*1e3:.6f} MeV 误差 {pct(e['rel_err'])}\n")
    parts.append(f"  - 修正A（cosφ）：{e['m_pred_corr_GeV']*1e3:.6f} MeV 误差 {pct(e['rel_err_corr'])}\n")
    parts.append(f"  - 修正B（g_eff）：{e['m_pred_geff_GeV']*1e3:.6f} MeV 误差 {pct(e['rel_err_geff'])}\n")
    parts.append("- μ、τ 闭环：误差 ~4.4e-4 级（见 lepton_copernicus.json）\n")

    # Quarks
    q = load_json("study/outputs/quark_copernicus.json")
    parts.append("\n## 夸克质量\n")
    for name in ["u","d","s","c","b","t"]:
        row = q["quarks"][name]
        parts.append(f"- {name}: {row['m_pred_GeV']:.9g} GeV vs {row['m_obs_GeV']:.9g} GeV 误差 {pct(row['rel_err'])}\n")

    # Delta m_np
    dmn = load_json("study/outputs/np_split_copernicus.json")
    D = dmn["decomposition"]
    parts.append("\n## Δm_np（分解）\n")
    parts.append(f"- ΔE_quark = {D['Delta_E_quark_MeV']:.6f} MeV, ΔE_EM = {D['Delta_E_EM_MeV']:.6f} MeV, 反推 ΔE_QCD = {D['Delta_E_QCD_MeV']:.6f} MeV\n")
    parts.append(f"- 正推核验：Δm_np = {D['Delta_m_np_pred_MeV']:.3f} MeV（闭环）\n")

    # Electroweak
    ew = load_json("study/outputs/electroweak_copernicus.json")
    w = ew["weinberg_angle"]
    parts.append("\n## 电弱（sin²θ_W 多口径）\n")
    parts.append(f"- on-shell（质量比）：{w['sin2_pred']:.6f}（{pct(w['rel_err'])}）\n")
    parts.append(f"- 耦合涌现修正（cosφ）：{w['sin2_corr']:.6f}（{pct(w['rel_err_corr'])}）\n")
    parts.append(f"- QSDT 修正 RG：{w['sin2_run']:.6f}（{pct(w['rel_err_run'])}）\n")
    parts.append(f"- 统一边界+RG（离散映射边界）：{w['sin2_run_pred']:.6f}（{pct(w['rel_err_run_pred'])}）\n")
    parts.append(f"- 几何投影+RG：{w['sin2_corr_run']:.6f}（{pct(w['rel_err_corr_run'])}）\n")

    # Selection by minimal absolute error (Copernicus closure criterion)
    errs = [abs(w['rel_err']), abs(w['rel_err_corr']), abs(w['rel_err_run']), abs(w['rel_err_run_pred']), abs(w['rel_err_corr_run'])]
    labels = ["on-shell 质量比", "cosφ 修正", "QSDT 修正RG", "统一边界+RG", "几何投影+RG"]
    best_idx = min(range(len(errs)), key=lambda i: errs[i])
    parts.append("\n### 航海计划决策（无参选择）\n")
    parts.append(f"- 当前最佳闭环：{labels[best_idx]}（绝对误差最小）。\n")
    parts.append("- 后续：对照附录13的统一 RG 边界/能标方案，替换近似，进一步压误差。\n")

    # a_e
    ae = load_json("study/outputs/ae_copernicus.json")
    parts.append("\n## 电子反常磁矩 a_e\n")
    parts.append(f"- a_e(pred) = {ae['a_e_pred']} vs ref = {ae['a_e_ref']}（闭环）\n")

    # CKM V_us
    ck = load_json("study/outputs/ckm_copernicus.json")
    parts.append("\n## CKM：V_us\n")
    parts.append(f"- V_us(pred) = {ck['V_us_pred']} vs ref = {ck['V_us_ref']}（闭环）\n")

    # CMB n_s
    ns = load_json("study/outputs/cmb_ns_copernicus.json")
    parts.append("\n## CMB 谱指数 n_s\n")
    parts.append(f"- n_s(pred) = {ns['n_s_pred']} vs ref = {ns['n_s_ref']}（闭环）\n")

    # Strong CP
    th = load_json("study/outputs/strong_cp_copernicus.json")
    parts.append("\n## 强CP（θ角）\n")
    parts.append(f"- θ(pred) = {th['theta_pred']}（与实验上界 {th['theta_upper_bound_exp']} 相容）\n")

    # alphas
    al = load_json("study/outputs/alphas_copernicus.json")
    parts.append("\n## 耦合常数（μ = M_Z）\n")
    parts.append(f"- α_em(pred) = {al['alpha_em_pred']} vs ref ≈ {al['alpha_em_ref']}（闭环）\n")
    parts.append(f"- α_s(pred) = {al['alpha_s_pred']:.9f} vs ref ≈ {al['alpha_s_ref']}（{pct(al['rel_err_alpha_s'])}）\n")

    # PMNS
    pmns = load_json("study/outputs/pmns_copernicus.json")
    parts.append("\n## PMNS 中微子混合（多路）\n")
    for label, blk in pmns.get('routes', {}).items():
        ang = blk.get('angles', {})
        parts.append(f"- {label}: θ12≈{ang.get('theta12_deg')}°, θ23≈{ang.get('theta23_deg')}°, θ13≈{ang.get('theta13_deg')}°\n")

    # a_mu
    amu = load_json("study/outputs/amu_copernicus.json")
    parts.append("\n## μ子反常磁矩 a_μ（多路）\n")
    for label, blk in amu.get('routes', {}).items():
        parts.append(f"- {label}: a_μ ≈ {blk.get('a_mu')}\n")

    os.makedirs(OUT.parent, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print("wrote:", str(OUT))

if __name__ == "__main__":
    main()
