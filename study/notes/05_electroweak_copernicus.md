# 电弱量：哥白尼计划（反推→正推→闭环）

口径与关系
- 固定不变量：`m0, g, β_Γ → ω`。
- 模族家族定位（二进制瀑布）：`E_n = m0/2^n`。
- 层内增益（层级律）：`m = m_ref · (1+ω)^{N/2}`。
- 电弱结构关系（标准口径，与“同行评审快速理解.md”一致）：
  - `m_W = m_Z · cos θ`，因此 `sin²θ_W = 1 - (m_W/m_Z)²`。

反推→正推
1) 对 `W, Z` 质量分别进行：先选 `n` 使 `m0/2^n ≤ m_obs`；再求整数 `N` 使 `m_pred` 最接近 `m_obs`。
2) 用得到的 `m_W, m_Z` 计算 `sin²θ_W(pred) = 1 - (m_W/m_Z)²`，与文档参考值对比。

脚本
- `study/scripts/electroweak_copernicus.py`
- 输出：`study/outputs/electroweak_copernicus.{json,md}`

备注
- 全程零自由参数；若需修正（如几何/规范关系项），后续将按附录关系式以函数接入。

