#!/usr/bin/env bash
set -euo pipefail

dataset="companynotes_rag"
data_dirs=("")
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
max_capacity_contexts=(512 1024)
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
