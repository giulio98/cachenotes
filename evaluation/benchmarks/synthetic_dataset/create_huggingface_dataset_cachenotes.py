 
import argparse, sys, math
from typing import List, Dict, Any, Tuple

import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, HfFolder
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams  # pip install vllm

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Create ruler cheatsheets with vLLM (no Ray) and push to HF."
    )
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--repo_id", type=str, default="giulio98/synthetic-dataset-Llama")
    p.add_argument("--max_new_tokens_cheatsheet", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--dtype", type=str, default="auto")
    return p.parse_args()

# ---------------- HF access check ----------------
def ensure_hf_write(repo_id: str):
    token = HfFolder.get_token()
    if not token:
        raise RuntimeError("No HF token found. Run `huggingface-cli login` or set HF_TOKEN.")
    api = HfApi()
    _ = api.whoami(token=token)  # auth sanity-check
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    return api

cheat_sheet_prompt = "You were given a task. Before I give you the question, imagine you are a student memorizing this material according to the task that you will have to perform. Repeat the context concisely yet comprehensively to aid memorization, ensuring you preserve all critical details and create a cheat sheet covering the entire context for solving the task. Once you have done that, I will give you the question.\n\n"
context_suffix = """Here are some Examples:
Question: "Which projects does [Person] belong to?"  
Answer: "[Project1], [Project2]"  
Question: "Which role does [Person] have?"  
Answer: "[Role1], [Role2]"  
Question: "Which departments is [Person] part of?"  
Answer: "[Department1], [Department2]"  
Question: "What are [Person]'s projects' domains?"  
Answer: "[Domain1], [Domain2]"  
Question: "What are [Person]'s projects' started years?"  
Answer: "[Year1], [Year2]"  
Question: "Who sponsors [Person]'s projects?"  
Answer: "[Sponsor1], [Sponsor2]\n\n"""
TASKS = [
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "phase8",
]

# ---------------- main ----------------
def main():
    args = parse_args()

    # HF write access
    ensure_hf_write(args.repo_id)

    # Tokenizer (for chat template + token counting)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    has_chat_template = hasattr(tok, "apply_chat_template") and tok.chat_template
    print(f"Using chat template: {has_chat_template}")

    # vLLM engine
    llm = LLM(
        model=args.model_name,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=True,
        trust_remote_code=True,
        seed=42,
    )

    # global sampling params for cheatsheets
    sp_cheat = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=42,
        max_tokens=args.max_new_tokens_cheatsheet,
    )

    for task in TASKS:
        print(f"\n=== Processing task: {task} ===")
        hf_ds = load_dataset("giulio98/synthetic_dataset", data_dir=task, split="test")

        # Add fields (+ stable row id)
        def add_fields(x, i):
            return {
                "_row_id": i,
                "prompt_cheatsheet": cheat_sheet_prompt,
            }
        hf_ds = hf_ds.map(add_fields, with_indices=True)

        # Prepare prompts via chat template (portable) or fallback to manual concat
        rows = hf_ds.to_dict()  # column-wise dict
        n = len(hf_ds)
        prompts = []
        for i in range(n):
            msgs = [{"role": "user", "content": rows["context"][i] + rows["prompt_cheatsheet"][i] + context_suffix}]
            if has_chat_template:
                prompt = tok.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                # Fallback: simple manual template
                prompt = f"{rows['context'][i]}{rows['prompt_cheatsheet'][i]}"
            prompts.append(prompt)

        generations = [""] * len(prompts)

        outs = llm.generate(prompts, sp_cheat)  # one shot

        for i, out in enumerate(outs):
            generations[i] = out.outputs[0].text if out.outputs else ""

        # Build final HF dataset and push
        pdf = hf_ds.to_pandas()
        pdf["cheatsheet"] = generations
        out_df = pdf.sort_values("_row_id").reset_index(drop=True)

        final_ds = Dataset.from_pandas(out_df, preserve_index=False)

        final_ds.push_to_hub(args.repo_id, config_name=task, split="test")
        print(f"Pushed {task} to {args.repo_id}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)