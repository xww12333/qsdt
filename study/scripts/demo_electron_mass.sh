#!/usr/bin/env bash
#
# 脚本名称: demo_electron_mass.sh
# 功能: 演示质量谱计算器在电子路线的使用（关系先行, 零参数）。
# 用法: bash study/scripts/demo_electron_mass.sh
# 说明: 将结果输出到 study/outputs/electron_mass.json
set -euo pipefail
cd "$(dirname "$0")"

python3 mass_spectrum.py \
  --g 0.223 \
  --beta-gamma 3.75 \
  --m0 125.1 \
  --units GeV \
  --N 10 \
  --out-json ../outputs/electron_mass.json
