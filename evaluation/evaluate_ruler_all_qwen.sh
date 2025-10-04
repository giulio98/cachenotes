#!/usr/bin/env bash
set -euo pipefail

dataset="ruler"
data_dirs=("4096")
model="Qwen/Qwen3-4B-Instruct-2507"
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
