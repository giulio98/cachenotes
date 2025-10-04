#!/usr/bin/env bash
set -euo pipefail

dataset="synthetic_dataset"
data_dirs=("phase1" "phase2" "phase3" "phase4" "phase5" "phase6" "phase7" "phase8")
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
press_names=("no_press")

device="cuda:0"   # <- pick the GPU you want to use

for data_dir in "${data_dirs[@]}"; do
  echo "Evaluating dataset: $data_dir"
  for press in "${press_names[@]}"; do
    echo "Running press_name: $press on $device"
    python evaluate.py \
    --dataset "$dataset" \
    --data_dir "$data_dir" \
    --model "$model" \
    --press_name "$press" \
    --device "$device"
  done
done

echo "All evaluations completed."
