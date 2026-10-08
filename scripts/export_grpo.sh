#!/usr/bin/env bash
# 把 FSDP actor checkpoint 导出为可服务/可续训的 HF 目录。
#   export_grpo.sh <global_step_*/actor> [output_dir] [base_model]
# 给 base_model 时顺带把 LoRA 合并进基座，得到可作为下一轮起点的合并权重；
# 不给时保留 lora_adapter/，vLLM 直挂（评测用这个）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTOR_CHECKPOINT="${1:?usage: bash scripts/export_grpo.sh <global_step_*/actor> [output_dir] [base_model]}"
OUTPUT_DIR="${2:-$ROOT/outputs/models/grpo-merged}"
BASE_MODEL="${3:-}"

"$ROOT/.venv/bin/python" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_CHECKPOINT" \
  --target_dir "$OUTPUT_DIR" \
  --trust-remote-code

if [[ -z "$BASE_MODEL" ]]; then
  exit 0
fi

# LoRA 在校验通过后才合并；失败不留下半成品目录。
[[ -f "$OUTPUT_DIR/lora_adapter/adapter_config.json" ]] || {
  echo "导出目录没有 lora_adapter/，无需合并：$OUTPUT_DIR" >&2
  exit 1
}
TRAINED_DIR="${OUTPUT_DIR}-trained"
exec "$ROOT/.venv/bin/python" scripts/merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$OUTPUT_DIR/lora_adapter" \
  --output "$TRAINED_DIR" \
  --bf16
