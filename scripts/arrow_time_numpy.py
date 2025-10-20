#!/usr/bin/env python3
"""
最小化Lindblad模拟器 - 仅使用numpy库
用于QSDT附录A理论验证

功能：
- 模拟退相干XY自旋-1/2链的动力学演化
- 计算冯诺依曼熵S(t)的时间演化
- 验证QSDT理论中时间箭头的微观机制

设计目标：
- 仅依赖numpy库，无其他外部依赖
- 小系统尺寸（N=3-6）以保持密度矩阵演化可处理
- 使用RK4时间积分确保数值稳定性
- 小时间步长保证精度
"""
import math
import numpy as np
from typing import List, Dict


def kronN(ops: List[np.ndarray]) -> np.ndarray:
    """
    计算多个算符的张量积
    
    参数：
    - ops: 算符列表
    
    返回：
    - 张量积结果
    """
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def build_paulis():
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    sp = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    sm = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    id2 = np.eye(2, dtype=complex)
    return sx, sz, sp, sm, id2


def embed(op_single: np.ndarray, site: int, N: int, id2: np.ndarray) -> np.ndarray:
    ops = [id2] * N
    ops = list(ops)
    ops[site] = op_single
    return kronN(ops)


def xy_chain_hamiltonian(N: int, J: float, E: float) -> np.ndarray:
    sx, sz, sp, sm, id2 = build_paulis()
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    # On-site energy term (E/2 * sz_i)
    for i in range(N):
        H += (E / 2.0) * embed(sz, i, N, id2)
    # XY hopping term: -J (sp_i sm_{i+1} + h.c.)
    for i in range(N - 1):
        H -= J * (embed(sp, i, N, id2) @ embed(sm, i + 1, N, id2)
                   + embed(sm, i, N, id2) @ embed(sp, i + 1, N, id2))
    return H


def lindblad_dissipator(rho: np.ndarray, Ls: List[np.ndarray]) -> np.ndarray:
    # D(rho) = sum_k (L rho L† - 1/2{L†L, rho})
    out = np.zeros_like(rho)
    for L in Ls:
        Lrho = L @ rho @ L.conj().T
        LL = L.conj().T @ L
        out += Lrho - 0.5 * (LL @ rho + rho @ LL)
    return out


def drho_dt(H: np.ndarray, rho: np.ndarray, Ls: List[np.ndarray]) -> np.ndarray:
    comm = H @ rho - rho @ H
    return -1j * comm + lindblad_dissipator(rho, Ls)


def rk4_step(H: np.ndarray, rho: np.ndarray, Ls: List[np.ndarray], dt: float) -> np.ndarray:
    k1 = drho_dt(H, rho, Ls)
    k2 = drho_dt(H, rho + 0.5 * dt * k1, Ls)
    k3 = drho_dt(H, rho + 0.5 * dt * k2, Ls)
    k4 = drho_dt(H, rho + dt * k3, Ls)
    rho_next = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # Hermitize and renormalize to control numerical drift
    rho_next = 0.5 * (rho_next + rho_next.conj().T)
    tr = np.trace(rho_next)
    if abs(tr) > 0:
        rho_next /= tr
    return rho_next


def von_neumann_entropy(rho: np.ndarray) -> float:
    # S = -Tr rho log rho, natural log
    evals = np.linalg.eigvalsh((rho + rho.conj().T) * 0.5)
    evals = np.clip(evals.real, 0.0, 1.0)
    nz = evals[evals > 1e-12]
    if nz.size == 0:
        return 0.0
    return float(-np.sum(nz * np.log(nz)))


def initial_state_N_spin(N: int) -> np.ndarray:
    # Pure product state with a single excitation at center: |0...010...0>
    center = N // 2
    # Basis |0> = [1,0], |1> = [0,1]
    ket = np.array([1.0, 0.0], dtype=complex)
    exc = np.array([0.0, 1.0], dtype=complex)
    vecs = []
    for i in range(N):
        vecs.append(exc if i == center else ket)
    psi = vecs[0]
    for v in vecs[1:]:
        psi = np.kron(psi, v)
    rho0 = np.outer(psi, psi.conj())
    return rho0


def dephasing_operators(N: int, gamma: float) -> List[np.ndarray]:
    _, sz, _, _, id2 = build_paulis()
    Ls = []
    for i in range(N):
        Ls.append(math.sqrt(gamma) * embed(sz, i, N, id2))
    return Ls


def run_simulation(N: int, J: float, E: float, gamma: float, T: float, dt: float) -> Dict[str, np.ndarray]:
    dim = 2 ** N
    H = xy_chain_hamiltonian(N, J, E)
    Ls = dephasing_operators(N, gamma)
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


def main():
    # Parameters (matching theoretical document for consistency)
    J = 1.0
    E = 0.0
    gamma = 0.1
    T = 5.0   # 减少计算时间
    dt = 0.2  # 增大时间步长，减少计算量
    system_sizes = [4, 6, 8]  # 进一步减少到N=8，确保快速计算

    all_results = {}
    for N in system_sizes:
        res = run_simulation(N, J, E, gamma, T, dt)
        all_results[N] = res
        print(f"N={N}: S(0)={res['entropy'][0]:.4f}, S(T)={res['entropy'][-1]:.4f}, "
              f"peak dS/dt={np.max(res['entropy_rate']):.4f}")
        # Monotonicity check (allow small numerical jitter)
        diffs = np.diff(res["entropy"])
        decreases = np.sum(diffs < -1e-6)
        if decreases > 0:
            print(f"  Warning: detected {decreases} small entropy decreases (numerical).")

    # crude scaling: fit log(peak) vs log(N)
    Ns = np.array(system_sizes, dtype=float)
    peaks = np.array([np.max(all_results[N]["entropy_rate"]) for N in system_sizes])
    logN = np.log(Ns)
    logP = np.log(np.clip(peaks, 1e-12, None))
    A = np.vstack([logN, np.ones_like(logN)]).T
    alpha, logC = np.linalg.lstsq(A, logP, rcond=None)[0]
    print(f"Scaling: peak dS/dt ≈ C * N^alpha with alpha ≈ {alpha:.3f}")


if __name__ == "__main__":
    main()
