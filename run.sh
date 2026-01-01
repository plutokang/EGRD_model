#!/usr/bin/env bash
set -euo pipefail

release_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python_bin="${NOVELTY_PIPELINE_PYTHON_BIN:-python}"
data_dir="${NOVELTY_PIPELINE_DATA_DIR:-dataset}"
output_root="${NOVELTY_PIPELINE_OUTPUT_ROOT:-results/runs/original_v23_deberta_top5}"
model_name="${NOVELTY_PIPELINE_MODEL_NAME:-microsoft/deberta-v3-base}"
pair_epochs="${NOVELTY_PIPELINE_PAIR_EPOCHS:-4}"
joint_epochs="${NOVELTY_PIPELINE_JOINT_EPOCHS:-5}"
residual_epochs="${NOVELTY_PIPELINE_RESIDUAL_EPOCHS:-10}"

command=(
  "$python_bin" "$release_dir/novelty/run_pipeline.py"
  --start-stage A
  --stop-stage C
  --pipeline-profile original_v23
  --data-dir "$data_dir"
  --output-root "$output_root"
  --model-name "$model_name"
  --top-k 5
  --num-semantic-units 8
  --max-length 384
  --pair-split-source temporal
  --pair-max-length 768
  --pair-epochs "$pair_epochs"
  --pair-batch-size 4
  --pair-eval-batch-size 16
  --pair-lr 1e-5
  --joint-epochs "$joint_epochs"
  --joint-batch-size 2
  --joint-eval-batch-size 8
  --joint-gradient-accumulation-steps 8
  --residual-epochs "$residual_epochs"
  --residual-batch-size 2
  --residual-eval-batch-size 8
  --residual-ranking-source dataset
  --residual-amp-dtype bf16
  --ranking-source raw
  --ranking-label-bonus 0.02
  --amp
  --no-bf16
  --gradient-checkpointing
)

printf 'Running:'
printf ' %q' "${command[@]}" "$@"
printf '\n'

"${command[@]}" "$@"
