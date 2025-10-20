#!/usr/bin/env python3
"""
QSDT附录A数值验证脚本 - 使用QuTiP库
基于时间箭头计算脚本v2.md的理论验证

功能：
- 模拟一维自旋链的退相干过程
- 计算冯诺依曼熵S(t)的时间演化
- 拟合峰值dS/dt与系统大小N的标度关系：dS/dt ~ N^α

理论依据：
- QSDT理论中时间箭头的微观起源
- 量子退相干与熵产生的动力学过程
- 系统大小对熵产生速率的影响
"""
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def run_entropy_simulation(N, J, E, gamma, T, dt, dissipator="dephasing", peak_frac=1.0):
    """
    运行熵演化模拟
    
    参数：
    - N: 自旋链长度
    - J: 交换耦合强度
    - E: 能量参数
    - gamma: 退相干强度
    - T: 总演化时间
    - dt: 时间步长
    - dissipator: 耗散类型（"dephasing"退相干）
    - peak_frac: 峰值检测比例
    
    返回：
    - times: 时间数组
    - entropy: 熵演化数组
    - peak_idx: 峰值位置
    - peak_dS_dt: 峰值熵产生速率
    """
    times = np.linspace(0, T, int(T / dt) + 1)

    sx, sz, sp, sm = qt.sigmax(), qt.sigmaz(), qt.sigmap(), qt.sigmam()
    id_op = qt.qeye(2)

    # Hamiltonian
    H = 0 * qt.tensor([id_op for _ in range(N)])
    for i in range(N):
        H += (E / 2) * qt.tensor([sz if j == i else id_op for j in range(N)])
    for i in range(N - 1):
        op1 = qt.tensor([sp if j == i else sm if j == i + 1 else id_op for j in range(N)])
        op2 = qt.tensor([sm if j == i else sp if j == i + 1 else id_op for j in range(N)])
        H -= J * (op1 + op2)

    # Initial state: center excitation
    psi0_list = [qt.basis(2, 0)] * N
    psi0_list[N // 2] = qt.basis(2, 1)
    psi0 = qt.tensor(psi0_list)

    # Lindblad operators
    c_ops = []
    if dissipator == "dephasing":
        for i in range(N):
            op = qt.tensor([sz if j == i else id_op for j in range(N)])
            c_ops.append(np.sqrt(gamma) * op)
    elif dissipator == "amplitude":
        for i in range(N):
            op = qt.tensor([sm if j == i else id_op for j in range(N)])
            c_ops.append(np.sqrt(gamma) * op)
    elif dissipator == "combined":
        for i in range(N):
            op_z = qt.tensor([sz if j == i else id_op for j in range(N)])
            op_m = qt.tensor([sm if j == i else id_op for j in range(N)])
            c_ops.append(np.sqrt(gamma/2) * op_z)
            c_ops.append(np.sqrt(gamma/2) * op_m)
    else:
        raise ValueError("unknown dissipator: " + str(dissipator))

    options = qt.Options(nsteps=5000, atol=1e-9)
    result = qt.mesolve(H, psi0, times, c_ops=c_ops, e_ops=[], options=options)

    entropy = [qt.entropy_vn(state) for state in result.states]
    entropy = np.array(entropy)
    entropy_rate = np.gradient(entropy, times)
    # restrict peak window if needed
    kmax = int(len(times) * max(0.0, min(1.0, peak_frac)))
    if kmax < 2:
        kmax = len(times)
    peak_local = np.max(entropy_rate[:kmax])
    return {"times": times, "entropy": entropy, "entropy_rate": entropy_rate, "peak_local": peak_local}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dissipator", default="dephasing", choices=["dephasing","amplitude","combined"])
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--T", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--Ns", type=str, default="4,6,8,10")
    ap.add_argument("--peak_frac", type=float, default=1.0, help="fraction of time to search peak dS/dt (0-1)")
    args = ap.parse_args()
    J_coupling = 1.0
    E_onsite = 0.0
    gamma_dissipation = args.gamma
    total_time = args.T
    time_step = args.dt
    system_sizes = [int(x) for x in args.Ns.split(',')]

    all_results = {}
    for N_size in system_sizes:
        res = run_entropy_simulation(
            N=N_size,
            J=J_coupling,
            E=E_onsite,
            gamma=gamma_dissipation,
            T=total_time,
            dt=time_step,
            peak_frac=args.peak_frac,
            dissipator=args.dissipator,
        )
        all_results[N_size] = res
        print(f"N={N_size}: S(0)={res['entropy'][0]:.4f}, S(T)={res['entropy'][-1]:.4f}, "
              f"peak dS/dt={res['peak_local']:.4f}")

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(system_sizes)))
    for i, N in enumerate(system_sizes):
        res = all_results[N]
        axs[0].plot(res['times'], res['entropy'], label=f'N={N}', color=colors[i], linewidth=2)
    axs[0].set_title('Entropy Growth vs. System Size (Arrow of Time)')
    axs[0].set_xlabel('Time (t)')
    axs[0].set_ylabel('Von Neumann Entropy S(t)')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    peak_rates = [all_results[N]['peak_local'] for N in system_sizes]

    def power_law(x, alpha, C):
        return C * x**alpha

    popt, _ = curve_fit(power_law, system_sizes, peak_rates, p0=[1.0, 0.1])
    alpha_fit = popt[0]
    fit_N = np.linspace(min(system_sizes), max(system_sizes), 100)
    axs[1].plot(system_sizes, peak_rates, 'o--', label='Peak dS/dt')
    axs[1].plot(fit_N, power_law(fit_N, *popt), 'r-', label=f'Fit: dS/dt ~ N^{alpha_fit:.2f}')
    axs[1].set_title('Scaling of Irreversibility')
    axs[1].set_xlabel('System Size (N)')
    axs[1].set_ylabel('Peak Entropy Rate')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('QSDT_Arrow_of_Time_Verification_QuTiP.png', dpi=300)
    print("Generated plot: QSDT_Arrow_of_Time_Verification_QuTiP.png")
    print(f"Fitted alpha ≈ {alpha_fit:.4f}")


if __name__ == '__main__':
    main()
