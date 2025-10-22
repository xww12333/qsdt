#!/usr/bin/env bash
#
# 脚本名称: run_all.sh
# 功能: 生成 study/ 全部航海计划产物（计算 + 多路径对比 + 统一报告）。
# 用法: bash study/scripts/run_all.sh
# 注意: 无网络、零参数；若输入数据缺失（如电弱谱 CSV）, 会跳过对应拟合。
set -euo pipefail
cd "$(dirname "$0")"/..

echo "[1/10] Leptons (Copernicus)" && python3 scripts/lepton_copernicus.py
echo "[2/10] Quarks (Copernicus)" && python3 scripts/quark_copernicus.py --out outputs/quark_copernicus.json
echo "[3/10] Δm_np (Copernicus)" && python3 scripts/np_split_copernicus.py --out outputs/np_split_copernicus.json && python3 scripts/report_delta_mnp_compare.py
echo "[3.8/11] Fit EW omegas (if spectra provided)" && python3 scripts/fit_ew_omegas.py >/dev/null 2>&1 || true
echo "[4/10] Electroweak (Copernicus)" && python3 scripts/electroweak_copernicus.py --out outputs/electroweak_copernicus.json && python3 scripts/report_electroweak_compare.py && python3 scripts/ew_tree_solver.py
echo "[5/10] a_e (Copernicus)" && python3 scripts/ae_copernicus.py
echo "[6/10] CKM V_us (Copernicus)" && python3 scripts/ckm_copernicus.py && python3 scripts/report_quark_compare.py
echo "[7/10] CMB n_s (Copernicus)" && python3 scripts/cmb_ns_copernicus.py
echo "[8/10] Strong CP θ (Copernicus)" && python3 scripts/strong_cp_copernicus.py
echo "[9/11] α_em/α_s @ M_Z (Copernicus)" && python3 scripts/alphas_copernicus.py
echo "[10/11] a_μ (Copernicus multi-route)" && python3 scripts/amu_copernicus.py && python3 scripts/report_amu_compare.py
echo "[10.2/11] PMNS route compare" && python3 scripts/report_pmns_compare.py && python3 scripts/report_electron_compare.py && python3 scripts/report_index.py
echo "[11/12] Navigator summary" && python3 scripts/copernicus_navigator.py
echo "[12/12] Unified multi-route report" && python3 scripts/report_copernicus_unified.py
echo "All study reports generated under study/outputs/"
