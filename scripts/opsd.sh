#!/usr/bin/env bash
# ShopSimulator 上的 OPSD：与 GRPO 共用同一 rollout 协议，只把 group 优势换成
# 教师特权契约下的逐 token k1 信号。见 docs/training.md#opsd。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# prepare_opsd 要读商品目录，只能用 ShopSimulator 的解释器（主 venv 没有 flask）。
SHOPSIM_PYTHON="${SHOPSIM_PYTHON:-$ROOT/environments/ShopSimulator/.venv-shopsim/bin/python}"

cd "$ROOT"
"$SHOPSIM_PYTHON" scripts/prepare_opsd.py

exec "$ROOT/.venv/bin/python" scripts/train_grpo.py \
  --config "$ROOT/configs/opsd.yaml" \
  --output "${OPSD_OUTPUT_DIR:-$ROOT/outputs/models/opsd}" \
  --experiment-name "${OPSD_EXPERIMENT_NAME:-shopping-agent-opsd}" \
  "$@"
