#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:-model}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/$LABEL}"
SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
: "${OPENAI_BASE_URL:?设置 Flash API 的 OPENAI_BASE_URL；LLM_BASE_URL 仅用于 Actor}"
: "${OPENAI_API_KEY:?设置 Flash API 的 OPENAI_API_KEY}"
"$ROOT/.venv/bin/python" scripts/evaluate_shop_benchmark.py \
  --benchmark data/evaluation/tasks.jsonl \
  --output "$OUTPUT_DIR/trajectories.jsonl" \
  --summary "$OUTPUT_DIR/summary.json" \
  --base-url "$SHOPSIM_BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY"

"$ROOT/.venv/bin/python" scripts/build_eval_report.py --run-dir "$OUTPUT_DIR"

JUDGE_DIR="$OUTPUT_DIR/judge"
SHARED_DIR="${EVAL_SHARED_DIR:-$ROOT/outputs/evaluation/judge-shared}"
mkdir -p "$JUDGE_DIR" "$SHARED_DIR"
SHOPSIM_PYTHON="${SHOPSIM_PYTHON:-$ROOT/environments/ShopSimulator/.venv-shopsim/bin/python}"
remote_args=(--allow-blind-final)
rebuild_args=(--allow-blind-final --force)
if [[ "${EVAL_FORCE:-0}" == "1" ]]; then
  remote_args+=(--force)
elif [[ "${EVAL_RESUME:-0}" == "1" ]]; then
  remote_args+=(--resume)
fi

if [[ ! -f "$SHARED_DIR/task_facts.jsonl" ]]; then
  PYTHONPATH="$ROOT/src:$ROOT/environments/ShopSimulator/shop_env" \
    "$SHOPSIM_PYTHON" scripts/eval_rubric_judge.py facts \
    --tasks data/evaluation/tasks.jsonl \
    --output "$SHARED_DIR/task_facts.jsonl" --allow-blind-final
fi

if [[ ! -f "$SHARED_DIR/rubric_candidates.jsonl" ]]; then
  "$ROOT/.venv/bin/python" scripts/eval_rubric_judge.py rubric-candidates \
    --task-facts "$SHARED_DIR/task_facts.jsonl" \
    --output "$SHARED_DIR/rubric_candidates.jsonl" --allow-blind-final
fi

"$ROOT/.venv/bin/python" scripts/eval_rubric_judge.py rubric \
  --task-facts "$SHARED_DIR/task_facts.jsonl" \
  --candidates "$SHARED_DIR/rubric_candidates.jsonl" \
  --output "$SHARED_DIR/rubrics.jsonl" --allow-blind-final --resume

"$ROOT/.venv/bin/python" scripts/eval_rubric_judge.py preprocess \
  --raw "$OUTPUT_DIR/trajectories.jsonl" \
  --output "$JUDGE_DIR/preprocessed.jsonl" "${rebuild_args[@]}"

"$ROOT/.venv/bin/python" scripts/eval_rubric_judge.py judge-inputs \
  --preprocessed "$JUDGE_DIR/preprocessed.jsonl" \
  --rubrics "$SHARED_DIR/rubrics.jsonl" \
  --output "$JUDGE_DIR/judge_requests.jsonl" "${rebuild_args[@]}"

"$ROOT/.venv/bin/python" scripts/eval_rubric_judge.py judge \
  --requests "$JUDGE_DIR/judge_requests.jsonl" \
  --output "$JUDGE_DIR/judges.jsonl" "${remote_args[@]}"

"$ROOT/.venv/bin/python" scripts/eval_rubric_judge.py assemble \
  --preprocessed "$JUDGE_DIR/preprocessed.jsonl" \
  --rubrics "$SHARED_DIR/rubrics.jsonl" \
  --judges "$JUDGE_DIR/judges.jsonl" \
  --tasks data/evaluation/tasks.jsonl \
  --actor-label "$SERVED_MODEL_NAME" \
  --output "$JUDGE_DIR/evaluations.jsonl" \
  --summary "$JUDGE_DIR/evaluation_summary.json" "${rebuild_args[@]}"
