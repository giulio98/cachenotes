#!/usr/bin/env bash
set -euo pipefail

dataset="longbench"
data_dirs=("narrativeqa" "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa"
  "musique" "gov_report" "qmsum" "multi_news" "trec" "triviaqa"
  "samsum" "passage_count" "passage_retrieval_en" "lcc" "repobench-p")
model="Qwen/Qwen3-4B-Instruct-2507"
max_capacity_contexts=(512 1024 2048)
press_names=("kvzip")

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
        --device "$device"
    done
  done
done

echo "All evaluations completed."
