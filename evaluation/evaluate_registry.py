from benchmarks.longbench.calculate_metrics import calculate_metrics as longbench_scorer
from benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer
from benchmarks.synthetic_dataset.calculate_metrics import calculate_metrics as synthetic_dataset_scorer

from kvpress import (
    ExpectedAttentionPress,
    FinchPress,
    KVzipPress,
    SnapKVPress,
)

# These dictionaries define the available datasets, scorers, and KVPress methods for evaluation.
DATASET_REGISTRY = {
    "ruler": "anon-submission/ruler",
    "ruler_llama_cheatsheet": "anon-submission/ruler",
    "ruler_qwen_cheatsheet": "anon-submission/ruler-Qwen",
    "longbench": "anon-submission/LongBench",
    "longbench_llama_cheatsheet": "anon-submission/LongBench",
    "longbench_llama_cheatsheet_lcc": "anon-submission/LongBench-lcc",
    "longbench_llama_cheatsheet_s": "anon-submission/LongBench-s",
    "longbench_qwen_cheatsheet": "anon-submission/LongBench-Qwen",
    "longbench_rag": "anon-submission/LongBench-{max_context_length}",
    "longbench_bm25": "anon-submission/LongBench-BM25-{max_context_length}",
    "synthetic_dataset": "anon-submission/synthetic-dataset",
    "synthetic_dataset_rag": "anon-submission/synthetic_dataset-{max_context_length}",
    "companynotes_llama_cheatsheet": "anon-submission/company-notes",
    "companynotes_qwen_cheatsheet": "anon-submission/company-notes-Qwen",
    "companynotes_rag": "anon-submission/company-notes-{max_context_length}",
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
    
}


PRESS_REGISTRY = {
    "expected_attention": ExpectedAttentionPress(),
    "finch": FinchPress(),
    "cachenotes": FinchPress(use_vnorm=True),
    "kvzip": KVzipPress(),
    "snapkv": SnapKVPress(),
    "no_press": None,
}
