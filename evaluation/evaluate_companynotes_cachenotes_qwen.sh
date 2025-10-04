#!/usr/bin/env bash
set -euo pipefail

dataset="companynotes_qwen_cheatsheet"
data_dirs=("")
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
max_capacity_contexts=(512 1024)
press_names=("cachenotes")

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
        --max_capacity_context "$max_capacity_context" \
        --compress_task_cheatsheet True \
        --model_kwargs '{\"max_position_embeddings\": 131072, \"rope_scaling\": {\"factor\": 4.0, \"original_max_position_embeddings\": 32768, \"type\": \"yarn\"}}' \
        --device "$device" \
    done
  done
done

echo "All evaluations completed."
