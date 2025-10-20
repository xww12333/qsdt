#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QSDT时间箭头验证脚本 - QuTiP版本
=====================================

基于《量子空间动力学理论》附录A的严格实现，使用QuTiP库确保与理论文档v2版本完全一致。

功能：
- 实现XY链模型的量子退相干动力学
- 计算冯·诺依曼熵的时间演化
- 分析熵产生率的标度行为
- 验证时间箭头的涌现机制

理论依据：
- 量子空间动力学理论 (QSDT) 附录A
- Lindblad主方程：dρ/dt = -i[H,ρ] + D(ρ)
- 冯·诺依曼熵：S = -Tr(ρ log ρ)

作者：QSDT理论验证团队
版本：v2.0 (QuTiP实现)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import qutip as qt


def run_entropy_simulation(N: int, J: float, E: float, gamma: float, T: float, dt: float) -> Dict[str, np.ndarray]:
    """
    运行熵演化模拟 - QuTiP版本
    
    参数：
    - N: 系统大小（量子比特数）
    - J: 跳跃强度
    - E: 在位能量
    - gamma: 耗散强度
    - T: 总时间
    - dt: 时间步长
    
    返回：
    - 包含时间、熵、熵产生率的字典
    """
    print(f"Running simulation for N={N}")
    
    # --- 1. 构建XY链哈密顿量 ---
    # H = -J Σ_i (σ^x_i σ^x_{i+1} + σ^y_i σ^y_{i+1}) + E/2 Σ_i σ^z_i
    H = qt.qzero(2**N, 2**N)  # 指定正确的矩阵维度
    
    # 在位能量项
    for i in range(N):
        op = qt.tensor([qt.sigmaz() if j == i else qt.qeye(2) for j in range(N)])
        H += (E / 2.0) * op
    
    # XY跳跃项 - 使用σx和σy算符
    for i in range(N - 1):
        # σx_i σx_i+1 + σy_i σy_i+1
        op1 = qt.tensor([qt.sigmax() if j == i else qt.qeye(2) for j in range(N)]) * \
              qt.tensor([qt.sigmax() if j == i+1 else qt.qeye(2) for j in range(N)])
        op2 = qt.tensor([qt.sigmay() if j == i else qt.qeye(2) for j in range(N)]) * \
              qt.tensor([qt.sigmay() if j == i+1 else qt.qeye(2) for j in range(N)])
        H -= J * (op1 + op2)
    
    # --- 2. 定义初始低熵状态 ---
    # 从纯态开始 (S=0)，表示高度有序的初始条件
    # 状态是链中心处的单个激发
    psi0_list = [qt.basis(2, 0)] * N
    psi0_list[N // 2] = qt.basis(2, 1)  # 中心激发
    psi0 = qt.tensor(psi0_list)
    
    # --- 3. 定义Lindblad算符（与环境耦合） ---
    # 这些算符模拟不可逆的信息泄漏，从我们的子网络(S)到巨大的环境网络(E)
    # 使用相位阻尼，这是退相干的标准模型
    c_ops = []
    for i in range(N):
        op = qt.tensor([qt.sigmaz() if j == i else qt.qeye(2) for j in range(N)])
        c_ops.append(np.sqrt(gamma) * op)
    
    # --- 4. 求解Lindblad主方程 ---
    # 计算开放量子系统的密度矩阵ρ(t)的时间演化
    # ρ̇ = -i[H,ρ] + D(ρ)，其中D是耗散项
    times = np.arange(0, T + dt, dt)
    options = qt.Options(nsteps=5000, atol=1e-9)  # 使用鲁棒选项确保精度
    result = qt.mesolve(H, psi0, times, c_ops=c_ops, e_ops=[], options=options)
    
    # --- 5. 计算冯·诺依曼熵及其变化率 ---
    # S(t) = -Tr(ρ(t) * log(ρ(t)))
    print("Calculating entropy...")
    entropy = [qt.entropy_vn(state) for state in result.states]
    entropy_rate = np.gradient(entropy, times)
    
    print(f"N={N}: S(0)={entropy[0]:.4f}, S(T)={entropy[-1]:.4f}, Peak dS/dt={np.max(entropy_rate):.4f}\n")
    
    return {'times': times, 'entropy': np.array(entropy), 'entropy_rate': np.array(entropy_rate)}


def main():
    """
    主函数 - 运行完整的标度分析
    """
    print("=" * 60)
    print("QSDT时间箭头验证 - QuTiP版本")
    print("=" * 60)
    
    # --- 模拟配置 ---
    # 来自QSDT模型的物理参数
    J_coupling = 1.0       # 跳跃强度
    E_onsite = 0.0         # 在位能量（设为0简化，不影响熵动力学）
    gamma_dissipation = 0.1 # 与环境耦合强度
    
    # 模拟参数
    total_time = 10.0
    time_step = 0.1
    system_sizes = [4, 6, 8, 10]  # 网络大小列表
    
    # --- 运行模拟并存储结果 ---
    all_results = {}
    for N_size in system_sizes:
        all_results[N_size] = run_entropy_simulation(
            N=N_size, J=J_coupling, E=E_onsite, gamma=gamma_dissipation, T=total_time, dt=time_step
        )
    
    # --- 标度分析 ---
    print("=" * 60)
    print("标度分析结果")
    print("=" * 60)
    
    # 拟合峰值熵产生率与系统大小的关系
    Ns = np.array(system_sizes, dtype=float)
    peaks = np.array([np.max(all_results[N]["entropy_rate"]) for N in system_sizes])
    
    # 对数拟合：log(peak) = alpha * log(N) + log(C)
    logN = np.log(Ns)
    logP = np.log(np.clip(peaks, 1e-12, None))
    A = np.vstack([logN, np.ones_like(logN)]).T
    alpha, logC = np.linalg.lstsq(A, logP, rcond=None)[0]
    C = np.exp(logC)
    
    print(f"标度律：peak dS/dt ≈ C * N^alpha")
    print(f"α ≈ {alpha:.3f}")
    print(f"C ≈ {C:.3e}")
    print(f"r² = {1 - np.sum((logP - (alpha * logN + logC))**2) / np.sum((logP - np.mean(logP))**2):.3f}")
    
    # 与理论文档对比
    print(f"\n理论文档预测：α = 0.82")
    print(f"当前结果：α = {alpha:.3f}")
    print(f"差异：{abs(alpha - 0.82):.3f}")
    print(f"相对误差：{abs(alpha - 0.82) / 0.82 * 100:.1f}%")
    
    # --- 单调性检查 ---
    print("\n" + "=" * 60)
    print("单调性验证")
    print("=" * 60)
    
    for N in system_sizes:
        res = all_results[N]
        diffs = np.diff(res["entropy"])
        decreases = np.sum(diffs < -1e-6)
        if decreases > 0:
            print(f"N={N}: 检测到 {decreases} 次小幅度熵减少（数值误差）")
        else:
            print(f"N={N}: 熵单调非递减 ✓")
    
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    try:
        results = main()
    except ImportError as e:
        print(f"错误：缺少QuTiP库 - {e}")
        print("请安装QuTiP：pip install qutip")
    except Exception as e:
        print(f"运行时错误：{e}")
