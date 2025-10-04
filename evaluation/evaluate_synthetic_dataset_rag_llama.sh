#!/usr/bin/env bash
set -euo pipefail

dataset="synthetic_dataset_rag"
data_dirs=("phase1" "phase2" "phase3" "phase4" "phase5" "phase6" "phase7" "phase8")
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
max_capacity_contexts=(256 512 1024 2048)
press_names=("no_press")

device="cuda:0"   # <- pick the GPU you want to use

for data_dir in "${data_dirs[@]}"; do
  echo "Evaluating dataset: $data_dir"
  for press in "${press_names[@]}"; do
    for max_capacity_context in "${max_capacity_contexts[@]}"; do
      echo "Running press_name: $press with max_capacity_context: $max_capacity_context on $device"
      python evaluate.py \
        --dataset "$dataset" \
        --data_dir "$data_dir" \
        --model "$model" \
        --press_name "$press" \
        --max_context_length "$max_capacity_context" \
        --device "$device"
    done
  done
done

echo "All evaluations completed."
