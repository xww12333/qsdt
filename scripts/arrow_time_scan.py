#!/usr/bin/env python3
"""
QSDT附录A标度扫描脚本
扫描不同耗散机制下峰值dS/dt与系统大小N的标度关系

功能：
- 研究不同耗散类型对熵产生速率的影响
- 验证QSDT理论中时间箭头的标度行为
- 分析系统大小对动力学过程的影响

耗散类型：
- dephasing（退相干）：L_i = sqrt(gamma) * sigma_z^i
- amplitude（振幅阻尼）：L_i = sqrt(gamma) * sigma_-^i  
- combined（组合）：L_i = sqrt(gamma_z) * sigma_z^i + sqrt(gamma_a) * sigma_-^i

技术实现：
- 使用arrow_time_numpy中的纯numpy Lindblad积分器
- 避免外部依赖，确保计算的可重现性
"""
import numpy as np
from typing import List, Dict, Tuple
from arrow_time_numpy import (
    build_paulis, embed, xy_chain_hamiltonian, lindblad_dissipator,
    drho_dt, rk4_step, von_neumann_entropy, initial_state_N_spin
)


def dephasing_ops(N: int, gamma: float) -> List[np.ndarray]:
    _, sz, _, _, id2 = build_paulis()
    return [np.sqrt(gamma) * embed(sz, i, N, id2) for i in range(N)]


def amplitude_ops(N: int, gamma: float) -> List[np.ndarray]:
    _, _, _, sm, id2 = build_paulis()
    return [np.sqrt(gamma) * embed(sm, i, N, id2) for i in range(N)]


def combined_ops(N: int, gamma_z: float, gamma_a: float) -> List[np.ndarray]:
    _, sz, _, sm, id2 = build_paulis()
    Ls = [np.sqrt(gamma_z) * embed(sz, i, N, id2) for i in range(N)]
    Ls += [np.sqrt(gamma_a) * embed(sm, i, N, id2) for i in range(N)]
    return Ls


def run(N: int, J: float, E: float, T: float, dt: float, Ls: List[np.ndarray]) -> Dict[str, np.ndarray]:
    H = xy_chain_hamiltonian(N, J, E)
    rho = initial_state_N_spin(N)
    nsteps = int(T / dt) + 1
    times = np.linspace(0.0, T, nsteps)
    S = np.zeros(nsteps)
    S[0] = von_neumann_entropy(rho)
    for k in range(1, nsteps):
        rho = rk4_step(H, rho, Ls, dt)
        S[k] = von_neumann_entropy(rho)
    dSdt = np.gradient(S, times)
    return {"times": times, "entropy": S, "entropy_rate": dSdt}


def fit_alpha(Ns: List[int], peaks: List[float]) -> Tuple[float, float]:
    x = np.log(np.array(Ns, dtype=float))
    y = np.log(np.clip(np.array(peaks, dtype=float), 1e-12, None))
    A = np.vstack([x, np.ones_like(x)]).T
    alpha, logC = np.linalg.lstsq(A, y, rcond=None)[0]
    return alpha, np.exp(logC)


def main():
    J, E = 1.0, 0.0
    T, dt = 5.0, 0.2   # 减少计算量
    Ns = [4, 6, 8]  # 限制在N=8以内，确保快速计算

    # keep N<=8 for tractability
    Ns = [n for n in Ns if n <= 8]

    configs = [
        ("dephasing", lambda N: dephasing_ops(N, gamma=0.1)),
        ("amplitude", lambda N: amplitude_ops(N, gamma=0.05)),
        ("combined",  lambda N: combined_ops(N, gamma_z=0.05, gamma_a=0.05)),
    ]

    for name, Lgen in configs:
        peaks = []
        print(f"\n=== {name} dissipator ===")
        for N in Ns:
            res = run(N, J, E, T, dt, Lgen(N))
            peak = float(np.max(res["entropy_rate"]))
            peaks.append(peak)
            print(f"N={N}: S(T)={res['entropy'][-1]:.3f}, peak dS/dt={peak:.4f}")
        alpha, C = fit_alpha(Ns, peaks)
        print(f"alpha≈{alpha:.3f}, C≈{C:.3e}")


if __name__ == "__main__":
    main()
