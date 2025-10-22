#!/usr/bin/env python3
"""
脚本名称: alphas_copernicus.py
功能: 读取项目标准输出中的 α_em(μ)、α_s(μ)（默认 μ=M_Z）, 生成对比 JSON/讲义。
作用:
- 展示电磁与强相互耦合在 Z 标尺处的闭环情况, 并给出相对误差用于统一报告。
输入:
- scripts/copernicus/outputs/copernicus_predictions.standard.json（项目产物）。
输出:
- study/outputs/alphas_copernicus.json
- study/outputs/讲义_耦合常数_教学注解.md
使用方法: python3 study/scripts/alphas_copernicus.py
注意事项:
- 仅读取既有文件, 不上网; 若文件缺失则需先运行项目管线生成。
相关附录: 电弱与 QCD 跑动口径说明。
"""
from __future__ import annotations
import json, os, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "scripts/copernicus/outputs/copernicus_predictions.standard.json"
OUT_JSON = ROOT / "study/outputs/alphas_copernicus.json"
OUT_MD = ROOT / "study/outputs/讲义_耦合常数_教学注解.md"

def run():
    data = json.load(open(SRC, "r", encoding="utf-8"))
    mu_key = "91.1876"  # Z 标尺
    row = data.get(mu_key, {})
    aem = row.get("alpha_em@mu")
    as_ = row.get("alpha_s@mu")
    # 参考（文档）：α_em ≈ 1/128.0 ≈ 0.0078125, α_s ≈ 0.1181
    aem_ref = 0.007816
    as_ref = 0.1181
    rel_aem = (aem - aem_ref)/aem_ref if (aem is not None and aem_ref) else None
    rel_as = (as_ - as_ref)/as_ref if (as_ is not None and as_ref) else None
    payload = {
        "mu_GeV": float(mu_key),
        "alpha_em_pred": aem,
        "alpha_em_ref": aem_ref,
        "alpha_s_pred": as_,
        "alpha_s_ref": as_ref,
        "rel_err_alpha_em": rel_aem,
        "rel_err_alpha_s": rel_as,
        "notes": {"principle": "耦合随能标按统一 RG 关系演化，参照 Z 标尺对比（零参数）"}
    }
    os.makedirs(OUT_JSON.parent, exist_ok=True)
    json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    md = []
    md.append("# 讲义：电强耦合常数（随能标演化）\n\n")
    md.append("## 观察者视角\n- 观测到的是通道耦合的显化，值随能标 μ 关系演化。\n\n")
    md.append("## 实际对比（μ = M_Z）\n")
    md.append(f"- α_em(pred) = {aem}, ref ≈ {aem_ref}\n")
    md.append(f"- α_s(pred) = {as_}, ref ≈ {as_ref}\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print("wrote:", str(OUT_JSON), str(OUT_MD))

if __name__ == "__main__":
    run()
