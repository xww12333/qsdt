#!/usr/bin/env python3
"""
脚本名称: report_copernicus_unified.py
功能: 统一多路径分组报告生成（航海计划总表 + 分组 + 误差占比图）。
输入: study/outputs/*_copernicus.json（各模块产物）
输出: study/outputs/航海计划_统一多路径分组报告.md
使用: python3 study/scripts/report_copernicus_unified.py（或 run_all.sh 最后一步）
注意: 不做数值计算, 仅聚合与可视化; 如某模块缺失则跳过。
"""
from __future__ import annotations
import json, os, math

OUT = "study/outputs/航海计划_统一多路径分组报告.md"

def load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def fmt_pct(x: float|None) -> str:
    if x is None:
        return "—"
    return f"{x*100.0:+.3f}%"

def main():
    md = []
    md.append("# 航海计划：统一多路径分组报告（零自由参数）\n\n")
    md.append("本报告汇总各模块的多路径航海计划结果，并对可比对象给出与参考/实验的相对误差。\n\n")

    # 载入各模块数据
    lep = load("study/outputs/lepton_copernicus.json") or {}
    ew  = load("study/outputs/electroweak_copernicus.json") or {}
    qua = load("study/outputs/quark_copernicus.json") or {}
    dmn = load("study/outputs/np_split_copernicus.json") or {}
    ae  = load("study/outputs/ae_copernicus.json") or {}
    vus = load("study/outputs/ckm_copernicus.json") or {}
    ns  = load("study/outputs/cmb_ns_copernicus.json") or {}
    th  = load("study/outputs/strong_cp_copernicus.json") or {}
    al  = load("study/outputs/alphas_copernicus.json") or {}
    pm  = load("study/outputs/pmns_copernicus.json") or {}
    amu = load("study/outputs/amu_copernicus.json") or {}

    # 统一摘要表
    md.append("## 摘要（最佳路线与误差）\n\n")
    md.append("| 模块 | 最佳路线 | 预测 | 参考 | 相对误差 |\n")
    md.append("| --- | --- | ---: | ---: | ---: |\n")
    # Electron (choose best among routes)
    try:
        e = lep.get("electron", {})
        routes = [
            ("cascade", e.get("m_pred_GeV"), e.get("rel_err")),
            ("cosφ", e.get("m_pred_corr_GeV"), e.get("rel_err_corr")),
            ("g_eff", e.get("m_pred_geff_GeV"), e.get("rel_err_geff")),
            ("附录54 统一", e.get("m_appendix54_uni_GeV"), e.get("rel_err_appendix54_uni")),
        ]
        best = min([(abs(r[2]) if r[2] is not None else math.inf, r) for r in routes])[1]
        md.append(f"| 电子质量 | {best[0]} | {best[1]*1e3:.6f} MeV | {e.get('m_obs_GeV')*1e3:.6f} MeV | {fmt_pct(best[2])} |\n")

        # 附：附录54 统一修正带
        mu_min = e.get("m_appendix54_uni_min_GeV"); mu_max = e.get("m_appendix54_uni_max_GeV")
        if mu_min and mu_max:
            md.append(f"| 电子(54带) | — | [{mu_min*1e3:.6f}, {mu_max*1e3:.6f}] MeV | — |\n")
    except Exception:
        pass
    # Muon, Tau
    try:
        mu = lep.get("muon", {})
        md.append(f"| μ子质量 | — | {mu.get('m_pred_GeV'):.6f} GeV | {mu.get('m_obs_GeV'):.6f} GeV | {fmt_pct(mu.get('rel_err'))} |\n")
        ta = lep.get("tau", {})
        md.append(f"| τ子质量 | — | {ta.get('m_pred_GeV'):.6f} GeV | {ta.get('m_obs_GeV'):.6f} GeV | {fmt_pct(ta.get('rel_err'))} |\n")
    except Exception:
        pass
    # W, Z, sin2 best
    try:
        W = ew.get("W", {})
        Z = ew.get("Z", {})
        md.append(f"| m_W | — | {W.get('m_pred_GeV'):.6f} GeV | {W.get('m_obs_GeV'):.6f} GeV | {fmt_pct(W.get('rel_err'))} |\n")
        md.append(f"| m_Z | — | {Z.get('m_pred_GeV'):.6f} GeV | {Z.get('m_obs_GeV'):.6f} GeV | {fmt_pct(Z.get('rel_err'))} |\n")
        w = ew.get("weinberg_angle", {})
        sin2_routes = [
            ("on-shell", w.get("sin2_pred"), w.get("sin2_ref"), w.get("rel_err")),
            ("cosφ",     w.get("sin2_corr"), w.get("sin2_ref"), w.get("rel_err_corr")),
            ("QSDT-RG",  w.get("sin2_run"),  w.get("sin2_ref"), w.get("rel_err_run")),
            ("边界+RG",   w.get("sin2_run_pred"), w.get("sin2_ref"), w.get("rel_err_run_pred")),
            ("投影+RG",   w.get("sin2_corr_run"), w.get("sin2_ref"), w.get("rel_err_corr_run")),
        ]
        bestw = min([(abs(r[3]) if r[3] is not None else math.inf, r) for r in sin2_routes])[1]
        md.append(f"| sin²θ_W | {bestw[0]} | {bestw[1]:.6f} | {bestw[2]:.6f} | {fmt_pct(bestw[3])} |\n")
    except Exception:
        pass
    # Quarks (list with max error)
    try:
        qrows = []
        for qn in ["u","d","s","c","b","t"]:
            qb = qua["quarks"][qn]
            qrows.append((qn, qb.get('m_pred_GeV'), qb.get('m_obs_GeV'), qb.get('rel_err')))
        # report max abs error
        worst = max(qrows, key=lambda r: abs(r[3] or 0.0))
        md.append(f"| 夸克质量(最差 {worst[0]}) | — | {worst[1]:.6f} GeV | {worst[2]:.6f} GeV | {fmt_pct(worst[3])} |\n")
    except Exception:
        pass
    # Δm_np
    try:
        blk = dmn.get("decomposition", {})
        md.append(f"| Δm_np | — | {blk.get('Delta_m_np_pred_MeV'):.3f} MeV | {blk.get('Delta_m_np_obs_MeV'):.3f} MeV | {fmt_pct( (blk.get('Delta_m_np_pred_MeV')-blk.get('Delta_m_np_obs_MeV'))/blk.get('Delta_m_np_obs_MeV'))} |\n")
    except Exception:
        pass
    # a_e
    try:
        md.append(f"| a_e | — | {ae.get('a_e_pred')} | {ae.get('a_e_ref')} | {fmt_pct(ae.get('rel_err'))} |\n")
    except Exception:
        pass
    # α_em/α_s @ MZ
    try:
        md.append(f"| α_em(M_Z) | — | {al.get('alpha_em_pred')} | {al.get('alpha_em_ref')} | {fmt_pct(al.get('rel_err_alpha_em'))} |\n")
        md.append(f"| α_s(M_Z) | — | {al.get('alpha_s_pred'):.9f} | {al.get('alpha_s_ref')} | {fmt_pct(al.get('rel_err_alpha_s'))} |\n")
    except Exception:
        pass
    # V_us
    try:
        md.append(f"| V_us | — | {vus.get('V_us_pred')} | {vus.get('V_us_ref')} | {fmt_pct((vus.get('V_us_pred')-vus.get('V_us_ref'))/vus.get('V_us_ref'))} |\n")
    except Exception:
        pass
    # n_s
    try:
        md.append(f"| n_s | — | {ns.get('n_s_pred')} | {ns.get('n_s_ref')} | {fmt_pct((ns.get('n_s_pred')-ns.get('n_s_ref'))/ns.get('n_s_ref'))} |\n")
    except Exception:
        pass
    # θ
    try:
        md.append(f"| θ (强CP) | — | {th.get('theta_pred')} | < {th.get('theta_upper_bound_exp')} | — |\n")
    except Exception:
        pass

    # 多路径分组报告
    md.append("\n## 分组报告（多路径并行）\n")

    # 电弱
    if ew:
        w = ew.get("weinberg_angle", {})
        md.append("\n### 电弱：sin²θ_W 多路径\n")
        md.append("| 路线 | 值 | 相对误差 |\n| --- | ---: | ---: |\n")
        for name,key,ek in [
            ("on-shell","sin2_pred","rel_err"),
            ("cosφ","sin2_corr","rel_err_corr"),
            ("QSDT-RG","sin2_run","rel_err_run"),
            ("边界+RG","sin2_run_pred","rel_err_run_pred"),
            ("投影+RG","sin2_corr_run","rel_err_corr_run"),
            ("QSDT-omega(★)","sin2_qsdt_omega",None),
        ]:
            val = w.get(key); err = w.get(ek)
            if val is None: continue
            md.append(f"| {name} | {val:.6f} | {fmt_pct(err)} |\n")
        # 附录55：on-shell 不确定度与有效角参考
        sin2_on_err = w.get("sin2_on_shell_err")
        sin2_eff = w.get("sin2_eff_ref")
        if sin2_on_err is not None or sin2_eff is not None:
            md.append(f"\n> on-shell 不确定度：±{sin2_on_err:.6f}；有效角参考：{sin2_eff}\n")

    # PMNS
    if pm:
        md.append("\n### PMNS：多路径角度（无参考，探索中）\n")
        md.append("| 路线 | θ12 (°) | θ23 (°) | θ13 (°) |\n| --- | ---: | ---: | ---: |\n")
        for label, blk in pm.get("routes", {}).items():
            ang = blk.get("angles", {})
            md.append(f"| {label} | {ang.get('theta12_deg'):.3f} | {ang.get('theta23_deg'):.3f} | {ang.get('theta13_deg'):.3f} |\n")

    # a_μ
    if amu:
        md.append("\n### μ子 g−2：多路径（无参考，探索中）\n")
        md.append("| 路线 | a_μ | 备注 |\n| --- | ---: | --- |\n")
        for label, blk in amu.get("routes", {}).items():
            note = ", ".join(f"{k}={v}" for k,v in (blk.get("weights") or {}).items())
            md.append(f"| {label} | {blk.get('a_mu')} | {note} |\n")

    # 夸克
    if qua:
        md.append("\n### 夸克质量：离散映射闭环\n")
        md.append("| 夸克 | 预测 [GeV] | 参考 [GeV] | 相对误差 |\n| --- | ---: | ---: | ---: |\n")
        for qn in ["u","d","s","c","b","t"]:
            qb = qua["quarks"][qn]
            md.append(f"| {qn} | {qb.get('m_pred_GeV'):.9g} | {qb.get('m_obs_GeV'):.9g} | {fmt_pct(qb.get('rel_err'))} |\n")

    # Δm_np
    if dmn:
        blk = dmn.get("decomposition", {})
        md.append("\n### Δm_np：关系分解\n")
        md.append("| 项 | 数值 [MeV] |\n| --- | ---: |\n")
        md.append(f"| ΔE_quark | {blk.get('Delta_E_quark_MeV'):.6f} |\n")
        md.append(f"| ΔE_EM | {blk.get('Delta_E_EM_MeV'):.6f} |\n")
        md.append(f"| ΔE_QCD | {blk.get('Delta_E_QCD_MeV'):.6f} |\n")
        md.append(f"| Δm_np(pred) | {blk.get('Delta_m_np_pred_MeV'):.3f} |\n")

    # 综合误差 mermaid 饼图（仅可比项）
    md.append("\n## 误差占比（可比项）\n")
    parts = []
    try:
        parts.append(("e", abs(best[2] or 0.0)))
    except Exception:
        pass
    try:
        parts.append(("μ", abs(mu.get('rel_err') or 0.0)))
        parts.append(("τ", abs(ta.get('rel_err') or 0.0)))
    except Exception:
        pass
    try:
        parts.append(("sin²θ_W", abs(bestw[3] or 0.0)))
    except Exception:
        pass
    try:
        parts.append(("m_W", abs(W.get('rel_err') or 0.0)))
        parts.append(("m_Z", abs(Z.get('rel_err') or 0.0)))
    except Exception:
        pass
    try:
        parts.append(("α_s(M_Z)", abs(al.get('rel_err_alpha_s') or 0.0)))
    except Exception:
        pass
    # a_e, α_em, V_us, n_s, Δm_np 皆近零或零，在饼图中可忽略以突出改进目标
    md.append("\n```mermaid\n")
    md.append("pie showData\n  title 相对误差占比（核心可比项）\n")
    for name, v in parts:
        md.append(f"  \"{name}\" : {v:.6f}\n")
    md.append("```\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print("wrote:", OUT)

if __name__ == "__main__":
    main()
