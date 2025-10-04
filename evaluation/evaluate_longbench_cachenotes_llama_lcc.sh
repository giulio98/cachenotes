#!/usr/bin/env bash
set -euo pipefail

dataset="longbench_llama_cheatsheet_lcc"
data_dirs=("lcc")
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
max_capacity_contexts=(512 1024 2048)
press_names=("cachenotes")

device="cuda:0"

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
        --device "$device"
    done
  done
done

echo "All evaluations completed."
