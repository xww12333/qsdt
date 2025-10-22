QSDT 学习工作区（关系先行）

范围
- `study/` 目录仅作为推导与校验的沙箱。
- 原则：关系先行，零自由参数；公式只是实体量子网络关系的翻译。
- 一切计算均应引用“实体量子网络”的内禀不变量。

方法（闭环）
- 实体网络 → 关系不变量 → 集体激发态（存在性） → 宏观显化 → 实验/观测对比。

当前聚焦
- 质量量子化链：推导/使用 ω 与层级谱 m_N = m0·(1+ω)^{N/2}。
- 修正项（如 QED/几何）需明确为“关系驱动”的显式项；不得拟合。

目录
- scripts/：计算器与命令行工具。
- notes/：推导笔记（关系与不变量）。
- outputs/：计算输出用于核验。
- inputs/：可选输入（电弱两通道谱 CSV、QSDT-omega JSON 等）。

用法
- 一键生成全部报告：`bash study/scripts/run_all.sh`
  - 产出集中在 `study/outputs/`，包括：统一多路径分组报告、各模块多路线对比页、讲义页等。
- 单脚本使用：
  - 质量谱计算：`python3 study/scripts/mass_spectrum.py --help`
  - 轻子/夸克/电弱/Δm_np 等模块脚本头部均有“脚本名称/功能/输入/输出/使用方法/注意事项”说明。

输入数据（可选）
- 电弱两通道层级谱（驱动 QSDT-omega★ 路线）
  - `study/inputs/ew_spectrum_W.csv`、`study/inputs/ew_spectrum_Y.csv`
  - CSV 格式：`N,m_GeV`（无表头/有表头均可，脚本会忽略第一行表头）
  - 运行 `run_all.sh` 时会自动调用 `scripts/fit_ew_omegas.py` 拟合 ω 并生成 `study/inputs/electroweak_omega.json`
- QSDT-omega★ 直接 JSON 输入（可替代拟合）
  - `study/inputs/electroweak_omega.json`（示例见 `study/inputs/electroweak_omega.example.json`）
  - 字段：`omega_W`、`omega_Y`、`beta_ratio=β_Y/β_W`

主要脚本（摘要）
- `scripts/run_all.sh`：一键生成全部产物（计算 + 多路径对比 + 统一报告）。
- `scripts/lepton_copernicus.py`：轻子质量（含附录52/54 路线与不确定度带）。
- `scripts/quark_copernicus.py`：六夸克质量离散映射闭环。
- `scripts/np_split_copernicus.py`：Δm_np 分解（ΔE_quark + ΔE_EM + 反推 ΔE_QCD）。
- `scripts/electroweak_copernicus.py`：电弱多路线（on-shell/cosφ/QSDT-RG/边界+RG/投影+RG/QSDT-omega★）。
- `scripts/ew_tree_solver.py`：树级/含 Δr 联立对照（教学用）。
- `scripts/fit_ew_omegas.py`：从两通道谱拟合 ω，生成 QSDT-omega★ 输入 JSON。
- `scripts/ae_copernicus.py`、`scripts/amu_copernicus.py`、`scripts/ckm_copernicus.py`、`scripts/cmb_ns_copernicus.py`、`scripts/strong_cp_copernicus.py`、`scripts/alphas_copernicus.py`：对应 a_e/a_μ/V_us/n_s/θ/α_em/α_s 的闭环计算与讲义输出。
- `scripts/*report*.py`：各模块多路径对比页或统一报告生成器，含 Mermaid 图像化。

多路径对比（教学）
- 对比页索引：`study/outputs/讲义_多路径对比索引.md`
- 电弱/PMNS/a_μ/电子/夸克/Δm_np 等均有对比页（表格+Mermaid 饼图/流程图）。

注意事项
- 零参数：所有脚本均不引入拟合参数；修正项需为显式关系函数。
- 不确定度：统一报告已展示关键项误差占比；电子（附录54）已输出统一修正不确定度带。
- 环境：不依赖网络；仅读取仓库内文件；输出在 `study/outputs/`。

扩展建议
- 若要把某附录的关系式接入，按脚本头部约定添加显式函数并在对应模块聚合。
- 若要接入不确定度传播，可在统一报告层添加参数区间与采样分析，不改变计算口径。
