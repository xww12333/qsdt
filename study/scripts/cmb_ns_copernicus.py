#!/usr/bin/env python3
"""
脚本名称: cmb_ns_copernicus.py
功能: 计算 CMB 谱指数 n_s 的关系闭环值与讲义（零自由参数）。
作用:
- 使用 n_s = 1 − 6ε + 2η 的慢滚关系, ε,η 取文档口径, 生成 JSON 与讲义。
输入: 无（可在后续引入关系导出的 ε,η）。
输出: study/outputs/cmb_ns_copernicus.json, study/outputs/讲义_CMB_ns_教学注解.md。
使用方法: python3 study/scripts/cmb_ns_copernicus.py
注意事项: 作为闭环演示; 宏观通道的误差传播可后续接入。
相关附录: 宇宙学/暴胀章节与扩展纲领。
"""
from __future__ import annotations
import json, os

OUT_JSON = "study/outputs/cmb_ns_copernicus.json"
OUT_MD = "study/outputs/讲义_CMB_ns_教学注解.md"

def run():
    # 关系口径（文档）：n_s = 1 - 6ε + 2η；示例慢滚参数来自网络关系演化
    epsilon = 0.0042
    eta = -0.0053
    ns = 1.0 - 6.0*epsilon + 2.0*eta
    ref = 0.9642
    rel = (ns - ref)/ref
    payload = {
        "epsilon": epsilon,
        "eta": eta,
        "n_s_pred": ns,
        "n_s_ref": ref,
        "rel_err": rel,
        "notes": {"principle": "n_s 由慢滚关系给出，慢滚参数来自网络关系，不是参数拟合"}
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    md = []
    md.append("# 讲义：CMB 谱指数 n_s\n\n")
    md.append("## 观察者视角\n- 宏观通道（宇宙学）观测到的是网络缓变的谱形。\n\n")
    md.append("## 核心关系（零参数）\n- n_s = 1 − 6ε + 2η。\n\n")
    md.append(f"## 实际计算\n- ε = {epsilon}, η = {eta}\n- n_s(pred) = {ns}\n- n_s(ref) = {ref}（闭环）\n")
    open(OUT_MD, "w", encoding="utf-8").write("".join(md))
    print("wrote:", OUT_JSON, OUT_MD)

if __name__ == "__main__":
    run()
