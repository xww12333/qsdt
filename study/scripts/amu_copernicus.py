#!/usr/bin/env python3
"""
脚本名称: amu_copernicus.py
功能: μ子反常磁矩 a_μ 的多路线（多口径）关系计算与对比输出（零自由参数）。
作用:
- 按航海计划并行生成多条无参关系路线（Schwinger、一阶QSDT反馈、几何投影、层级增益、通道加权）。
输入:
- scripts/copernicus/outputs/copernicus_predictions.standard.json（读取 α_em(μ)）。
- study/outputs/electroweak_copernicus.json（读取 sin²θ_W 作为弱通道权重）。
输出:
- study/outputs/amu_copernicus.json（多路线数值）
- study/outputs/讲义_a_mu_教学注解.md（讲义说明）
使用方法: python3 study/scripts/amu_copernicus.py
注意事项:
- 本脚本仅展示多路线的关系并列, 不引入拟合参数; “通道加权”使用 α_em、α_s、sin²θ_W 归一权重的零参数合成。
相关附录: a_μ 关系修正口径（QED/弱/强子）, 航海计划总述。
"""
from __future__ import annotations
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "study/outputs/amu_copernicus.json"
OUT_MD = ROOT / "study/outputs/讲义_a_mu_教学注解.md"
SRC = ROOT / "scripts/copernicus/outputs/copernicus_predictions.standard.json"

def run():
    data = json.load(open(SRC, "r", encoding="utf-8"))
    mu_key = "0.10566"  # μ at 0.10566 GeV
    row = data.get(mu_key, {})
    alpha_em = row.get("alpha_em@mu")
    alpha_s = row.get("alpha_s@mu", 0.1181)
    # 路线A：Schwinger 1-loop（a = α/2π）
    a_A = (alpha_em / (2.0 * 3.141592653589793)) if alpha_em else None
    # 路线B：QSDT 低能反馈修正（b=7，δ = - (b g^2 / 8π²) ln(μ/M_H)）
    g = 0.223
    b = 7.0
    mu = 0.10566
    M_H = 125.1
    import math
    ln_ratio = math.log(max(mu,1e-30)/max(M_H,1e-30))
    delta = - (b * g*g / (8.0 * math.pi * math.pi)) * ln_ratio
    a_B = (a_A * (1.0 + delta)) if a_A else None
    # 路线C：几何投影修正（cosφ≈√(1−g/2)）
    cosphi = (1.0 - 0.5 * g) ** 0.5
    a_C = (a_A / cosphi) if a_A else None
    # 路线D：层级小增益（乘以 (1+ω)）
    def omega(g: float, beta_gamma: float) -> float:
        import math
        return (beta_gamma * (g**2) * math.log(2.0)) / (4.0 * math.pi * math.pi)
    omg = omega(g, 3.75)
    a_D = (a_A * (1.0 + omg)) if a_A else None
    # 路线E：通道加权 δ（用四路线的相对增量映射到 EM/weak/strong），权重由 α_em, α_s, sin²θ_W 归一
    # 从已生成的电弱报告读取 sin²θ_W（若无则用 0.231）
    try:
        ew = json.load(open(ROOT / "study/outputs/electroweak_copernicus.json", "r", encoding="utf-8"))
        sin2 = ew.get("weinberg_angle", {}).get("sin2_pred", 0.231)
    except Exception:
        sin2 = 0.231
    w_em_raw = alpha_em or 0.0
    w_s_raw = alpha_s or 0.0
    w_w_raw = (alpha_em or 0.0) * (sin2 or 0.231)
    tot = (w_em_raw + w_s_raw + w_w_raw) or 1.0
    w_em, w_s, w_w = w_em_raw/tot, w_s_raw/tot, w_w_raw/tot
    # 相对增量：
    d_em = (a_C/a_A - 1.0) if (a_A and a_C) else 0.0
    d_s  = (a_D/a_A - 1.0) if (a_A and a_D) else 0.0
    d_w  = (a_B/a_A - 1.0) if (a_A and a_B) else 0.0
    delta_wsum = w_em*d_em + w_s*d_s + w_w*d_w
    a_E = a_A * (1.0 + delta_wsum) if a_A else None
    payload = {
        "routes": {
            "schwinger": {"a_mu": a_A},
            "qwdt_feedback_ln": {"a_mu": a_B, "delta": delta},
            "cosphi": {"a_mu": a_C},
            "omega_gain": {"a_mu": a_D, "omega": omg},
            "channel_weighted": {"a_mu": a_E, "weights": {"em": w_em, "weak": w_w, "strong": w_s}, "deltas": {"em": d_em, "weak": d_w, "strong": d_s}}
        },
        "notes": {
            "principle": "多路关系：A) Schwinger 基本项；B) QSDT 低能反馈修正（零参数）",
            "status": "experimental"
        }
    }
    os.makedirs(OUT_JSON.parent, exist_ok=True)
    json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    md = []
    md.append("# 讲义：μ子反常磁矩 a_μ\n\n")
    md.append("## 观察者视角\n- a_μ 是多通道（QED/弱/强子）叠加的通道量，离散网络对高能通道的关系修正显化为 δa_μ。\n\n")
    md.append("## 核心关系（零参数）\n- 路线A：a_μ ≈ α(μ)/(2π)；路线B：乘以 QSDT 反馈因子 (1+δ)，δ=−(b g²/8π²) ln(μ/M_H)。\n\n")
    md.append("## 航海步骤\n- 多路并行→对比闭环；不调参，仅改关系。\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print("wrote:", str(OUT_JSON), str(OUT_MD))

if __name__ == "__main__":
    run()
