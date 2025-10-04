#!/usr/bin/env bash
set -euo pipefail

dataset="ruler"
data_dirs=("4096")
model="Qwen/Qwen3-4B-Instruct-2507"
compression_ratios=(0.25 0.5 0.75)
press_names=("expected_attention")

device="cuda:0"   # <- pick the GPU you want to use

for data_dir in "${data_dirs[@]}"; do
  echo "Evaluating dataset: $data_dir"
  for press in "${press_names[@]}"; do
    for compression_ratio in "${compression_ratios[@]}"; do
      echo "Running press_name: $press with compression_ratio: $compression_ratio on $device"
      python evaluate.py \
        --dataset "$dataset" \
        --data_dir "$data_dir" \
        --model "$model" \
        --press_name "$press" \
        --compression_ratio "$compression_ratio" \
        --device "$device"
    done
  done
done

echo "All evaluations completed."
