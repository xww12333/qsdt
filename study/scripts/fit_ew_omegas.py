#!/usr/bin/env python3
"""
脚本名称: fit_ew_omegas.py
功能: 从两通道层级谱数据 (N, m_GeV) 拟合 ω_ΓW 与 ω_ΓY, 生成 QSDT-omega(★) 路线所需的输入 JSON。
作用:
- 自动线性回归 ln m vs N 得斜率 b, 由 b≈(1/2)ln(1+ω) 反解 ω = exp(2b)−1。
输入:
- study/inputs/ew_spectrum_W.csv, ew_spectrum_Y.csv（CSV: N,m_GeV）
输出:
- study/inputs/electroweak_omega.json（含 omega_W, omega_Y, beta_ratio 占位）
使用方法: python3 study/scripts/fit_ew_omegas.py（run_all.sh 会自动尝试执行）
注意事项: 若 CSV 缺失或点数不足则跳过（不报错）；beta_ratio 默认 1.0, 可手工修订。
相关附录: 56（QSDT-omega ★）。
"""
from __future__ import annotations
import os, csv, math, json, sys

W_CSV = "study/inputs/ew_spectrum_W.csv"
Y_CSV = "study/inputs/ew_spectrum_Y.csv"
OUT_JSON = "study/inputs/electroweak_omega.json"

def fit_omega(csv_path: str) -> float|None:
    if not os.path.exists(csv_path):
        return None
    xs, ys = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            try:
                N = float(row[0].strip())
                m = float(row[1].strip())
            except Exception:
                continue
            if m <= 0:
                continue
            xs.append(N)
            ys.append(math.log(m))
    n = len(xs)
    if n < 2:
        return None
    # linear regression y = a + b x
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x*x for x in xs)
    sxy = sum(x*y for x,y in zip(xs,ys))
    denom = n*sxx - sx*sx
    if abs(denom) < 1e-12:
        return None
    b = (n*sxy - sx*sy) / denom
    # slope b ≈ (1/2) ln(1+omega) => omega = exp(2b) - 1
    omega = math.exp(2.0*b) - 1.0
    return omega

def main():
    w = fit_omega(W_CSV)
    y = fit_omega(Y_CSV)
    if w is None or y is None:
        print("skip: missing spectra CSVs; provide both to generate", file=sys.stderr)
        return 0
    payload = {
        "omega_W": w,
        "omega_Y": y,
        "beta_ratio": 1.0,
        "source": {
            "W_CSV": W_CSV,
            "Y_CSV": Y_CSV,
            "note": "beta_ratio=β_Y/β_W assumed 1.0; edit JSON as needed"
        }
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote:", OUT_JSON)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
