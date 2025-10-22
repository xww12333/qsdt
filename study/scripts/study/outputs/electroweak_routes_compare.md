# 电弱 sin²θ_W 多路线对比

| 路线 | sin²θ_W | 相对误差 |
| --- | ---: | ---: |
| on-shell | 0.225061 | -2.571% |
| cosφ 修正 | 0.246347 | +6.644% |
| QSDT 修正RG | 0.220526 | -4.534% |
| 统一边界+RG | 0.220526 | -4.534% |
| 几何投影+RG | 0.241516 | +4.553% |
| QSDT-omega(★) | 0.225061 | -2.571% |

> on-shell 不确定度（附录55）：±0.000234，有效角参考：0.2315


```mermaid
pie showData
  title sin²θ_W 各路线误差占比
  "on-shell" : 0.025709
  "cosφ 修正" : 0.066437
  "QSDT 修正RG" : 0.045343
  "统一边界+RG" : 0.045343
  "几何投影+RG" : 0.045525
  "QSDT-omega(★)" : 0.025709
```

```mermaid
flowchart LR
  A[W/Z 质量闭环] --> B(on-shell)
  A --> C(cosφ 修正)
  A --> D(QSDT 修正RG)
  A --> E(统一边界+RG)
  A --> F(几何投影+RG)
  B:::n -->|0.225061| G[sin²θ_W]
  C:::n -->|0.246347| G
  D:::n -->|0.220526| G
  E:::n -->|0.220526| G
  F:::n -->|0.241516| G
  classDef n fill:#eef,stroke:#88f,stroke-width:1px;
```
