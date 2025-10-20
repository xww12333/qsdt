# QSDT理论验证脚本集

本目录包含用于QSDT（量子空间动力学理论）理论验证和数值计算的Python脚本。

## 脚本功能概览

### 1. appendix_a_qutip.py
**QSDT附录A数值验证脚本 - 使用QuTiP库**

- **功能**：基于时间箭头计算脚本v2.md的理论验证
- **核心功能**：
  - 模拟一维自旋链的退相干过程
  - 计算冯诺依曼熵S(t)的时间演化
  - 拟合峰值dS/dt与系统大小N的标度关系：dS/dt ~ N^α
- **理论依据**：
  - QSDT理论中时间箭头的微观起源
  - 量子退相干与熵产生的动力学过程
  - 系统大小对熵产生速率的影响
- **依赖**：numpy, qutip, matplotlib, scipy

### 2. arrow_time_numpy.py
**最小化Lindblad模拟器 - 仅使用numpy库**

- **功能**：用于QSDT附录A理论验证
- **核心功能**：
  - 模拟退相干XY自旋-1/2链的动力学演化
  - 计算冯诺依曼熵S(t)的时间演化
  - 验证QSDT理论中时间箭头的微观机制
- **设计目标**：
  - 仅依赖numpy库，无其他外部依赖
  - 小系统尺寸（N=3-6）以保持密度矩阵演化可处理
  - 使用RK4时间积分确保数值稳定性
- **依赖**：仅numpy

### 3. arrow_time_scan.py
**QSDT附录A标度扫描脚本**

- **功能**：扫描不同耗散机制下峰值dS/dt与系统大小N的标度关系
- **核心功能**：
  - 研究不同耗散类型对熵产生速率的影响
  - 验证QSDT理论中时间箭头的标度行为
  - 分析系统大小对动力学过程的影响
- **耗散类型**：
  - dephasing（退相干）：L_i = sqrt(gamma) * sigma_z^i
  - amplitude（振幅阻尼）：L_i = sqrt(gamma) * sigma_-^i  
  - combined（组合）：L_i = sqrt(gamma_z) * sigma_z^i + sqrt(gamma_a) * sigma_-^i
- **依赖**：numpy, arrow_time_numpy

### 4. qsdt_estimates.py
**QSDT理论信号快速数值估计脚本**

- **功能**：计算QSDT理论预测的可观测信号
- **核心功能**：
  - 提供理论验证的数值参考值
  - 支持实验设计和数据分析
- **理论信号**：
  1. **红移阶梯间隔**：Δz ≈ Lp / λ_em
     - 反映QSDT理论中空间量子化的可观测效应
  2. **洛伦兹不变性破坏诱导的时间延迟**：Δt ≈ (E/E_Pl)^n * (L/c)
     - 反映QSDT理论对相对论的修正
- **应用场景**：
  - 高能天体物理观测数据分析
  - 宇宙学距离测量精度评估
  - 量子引力效应实验设计
- **依赖**：仅math（标准库）

## 使用说明

### 快速开始
```bash
# 运行QSDT信号估计
python qsdt_estimates.py

# 运行时间箭头数值验证（需要安装qutip）
python appendix_a_qutip.py

# 运行最小化模拟器
python arrow_time_numpy.py

# 运行标度扫描
python arrow_time_scan.py
```

### 依赖安装
```bash
# 基础依赖（仅numpy）
pip install numpy

# 完整依赖（包含QuTiP）
pip install numpy qutip matplotlib scipy
```

## 理论背景

这些脚本基于QSDT（量子空间动力学理论）的核心概念：

1. **时间箭头的微观起源**：通过量子退相干过程解释时间的方向性
2. **空间量子化效应**：普朗克尺度下的空间结构对宏观观测的影响
3. **洛伦兹不变性修正**：高能极限下相对论的量子引力修正
4. **熵产生动力学**：系统大小对熵产生速率的标度关系

## 注意事项

- 所有脚本都已完善中文注释，便于理解和使用
- 数值计算使用双精度浮点数，确保计算精度
- 时间积分使用RK4方法，保证数值稳定性
- 小系统尺寸限制确保密度矩阵演化的可处理性

## 相关文档

- 时间箭头计算脚本.md
- 时间箭头计算脚本v2.md
- 量子空间动力学-附录A.md（理论背景）
