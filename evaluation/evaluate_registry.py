# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by Giulio on 2026-01-08
from benchmarks.longbench.calculate_metrics import calculate_metrics as longbench_scorer
from benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer
from benchmarks.synthetic_dataset.calculate_metrics import calculate_metrics as synthetic_dataset_scorer
from benchmarks.company_notes.calculate_metrics import calculate_metrics as dummy_metric

from kvpress import (
    ExpectedAttentionPress,
    FinchPress,
    KVzipPress,
    SnapKVPress,
)

# These dictionaries define the available datasets, scorers, and KVPress methods for evaluation.
DATASET_REGISTRY = {
    "ruler": "giulio98/ruler",
    "ruler_llama_cheatsheet": "giulio98/ruler",
    "ruler_qwen_cheatsheet": "giulio98/ruler-Qwen",
    "longbench": "giulio98/LongBench",
    "longbench_llama_cheatsheet": "giulio98/LongBench",
    "longbench_llama_cheatsheet_lcc": "giulio98/LongBench-lcc",
    "longbench_llama_cheatsheet_s": "giulio98/LongBench-s",
    "longbench_qwen_cheatsheet": "giulio98/LongBench-Qwen",
    "longbench_rag": "giulio98/LongBench-{max_context_length}",
    "longbench_bm25": "giulio98/LongBench-BM25-{max_context_length}",
    "synthetic_dataset": "giulio98/synthetic-dataset",
    "synthetic_dataset_rag": "giulio98/synthetic_dataset-{max_context_length}",
    "companynotes_llama_cheatsheet": "giulio98/company-notes",
    "companynotes_qwen_cheatsheet": "giulio98/company-notes-Qwen",
    "companynotes_rag": "giulio98/company-notes-{max_context_length}",
}

SCORER_REGISTRY = {
    "ruler": ruler_scorer,
    "longbench": longbench_scorer,
    "longbench_llama_cheatsheet": longbench_scorer,
    "longbench_llama_cheatsheet_lcc": longbench_scorer,
    "longbench_llama_cheatsheet_s": longbench_scorer,
    "longbench_qwen_cheatsheet": longbench_scorer,
    "longbench_rag": longbench_scorer,
    "longbench_bm25": longbench_scorer,
    "ruler_llama_cheatsheet": ruler_scorer,
    "ruler_qwen_cheatsheet": ruler_scorer,
    "synthetic_dataset": synthetic_dataset_scorer,
    "synthetic_dataset_rag": synthetic_dataset_scorer,
    "companynotes_llama_cheatsheet": dummy_metric,
    "companynotes_qwen_cheatsheet": dummy_metric,
    "companynotes_rag": dummy_metric,
    
}


PRESS_REGISTRY = {
    "expected_attention": ExpectedAttentionPress(),
    "finch": FinchPress(),
    "cachenotes": FinchPress(use_vnorm=True),
    "kvzip": KVzipPress(layerwise=True, headwise=True),
    "snapkv": SnapKVPress(),
    "no_press": None,
}
