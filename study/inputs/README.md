# study/inputs 说明

本目录存放可选输入数据，供“零参数”的多路径关系路线调用（若缺失则跳过或采用演示回退）。

## 电弱两通道层级谱（驱动 QSDT-omega★）
- `ew_spectrum_W.csv`, `ew_spectrum_Y.csv`
- CSV 格式：`N,m_GeV`（示例）
```
# N,m_GeV
1,62.55
2,31.27
3,15.64
```
- 运行 `bash study/scripts/run_all.sh` 时，脚本会自动调用 `scripts/fit_ew_omegas.py` 拟合 ω，并生成 `electroweak_omega.json`。

## QSDT-omega★ 直接 JSON 输入
- `electroweak_omega.json`（示例见仓库 `electroweak_omega.example.json`）
```
{
  "omega_W": 0.00327,
  "omega_Y": 0.00160,
  "beta_ratio": 1.0,
  "note": "来自两通道层级谱拟合的 ω 与几何比"
}
```
- 字段说明：
  - `omega_W`：SU(2) 通道的层级常数 ω_ΓW
  - `omega_Y`：U(1)_Y 通道的层级常数 ω_ΓY
  - `beta_ratio`：β_ΓY/β_ΓW 的几何比

> 未提供时，`electroweak_copernicus.py` 会回退为 g/g′ 比值近似（仅演示用）；建议提供谱或 JSON 以驱动“★”路线。
