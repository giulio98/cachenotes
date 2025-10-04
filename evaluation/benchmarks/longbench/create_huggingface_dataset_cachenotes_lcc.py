 
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
        description="Create LongBench cheatsheets with vLLM (no Ray) and push to HF."
    )
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--repo_id", type=str, default="anon-submission/LongBench-lcc")
    p.add_argument("--max_new_tokens_cheatsheet", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
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

# yarn_mistral_templates from: https://github.com/THUDM/LongBench/blob/main/LongBench/pred.py
cheat_sheet_context_prefix = {
    "lcc": "Your task is code completion.\n{context}\n\n",
}
cheat_sheet_prompt = {
    "lcc": "Before solving the code completion task, imagine you are a student memorizing this material according to the task that you will have to perform. Ensure you preserve all critical details to solve the task and create a cheat sheet covering the entire context. Once you have done that, I will prompt you to solve the code completion task.\n\n",
    
}

code_completion_prompt = (
    "Since your task is code completion, write a cheat sheet that is tailored to help you write the correct next line(s) of code. "
    "Highlight or list the specific lines or variables in the context that are most relevant for this task.\n"
    "For example:\n"
    "Context:\n"
    "[Previous code... ~10,000 lines omitted for brevity]\n"
    "    public class Example {\n"
    "    private int count;\n"
    "    public void increment() {\n"
    "        count++;\n"
    "    }\n"
    "    public int getCount() {\n"
    "Cheatsheet (for next line):\n"
    "- We are in getCount(), which should return the current value of count.\n"
    "- count is a private integer field.\n"
    "- Convention: Getter methods return the corresponding field.\n"
    "- [Relevant lines: declaration of count, method header]\n"
    "Next line will likely be: return count;"
)
context_prefix = {
    "lcc": "Please complete the code given below. \n{context}",
}
TASKS = [
    "lcc"
]
question_template = {
    "lcc": "{input}",
}
answer_prefix = {
    "lcc": "Next line of code:\n",
}
DATA_NAME_TO_MAX_NEW_TOKENS = {
    "lcc": 64,
}

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
        hf_ds = load_dataset("zai-org/LongBench", task, split="test")


        # Add fields (+ stable row id)
        def add_fields(x, i):
            return {
                "_row_id": i,
                "context_cheatsheet": cheat_sheet_context_prefix[task].format(**x),
                "prompt_cheatsheet": cheat_sheet_prompt[task] + code_completion_prompt,
            }
        hf_ds = hf_ds.map(add_fields, with_indices=True)
        hf_ds = hf_ds.map(lambda x: {"context": context_prefix[task].format(**x)})

        # Prepare prompts via chat template (portable) or fallback to manual concat
        rows = hf_ds.to_dict()  # column-wise dict
        n = len(hf_ds)
        prompts = []
        for i in range(n):
            msgs = [{"role": "user", "content": rows["context_cheatsheet"][i] + rows["prompt_cheatsheet"][i]}]
            if has_chat_template:
                prompt = tok.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                # Fallback: simple manual template
                prompt = f"{rows['context_cheatsheet'][i]}{rows['prompt_cheatsheet'][i]}"
            prompts.append(prompt)

        generations = [""] * len(prompts)

        outs = llm.generate(prompts, sp_cheat)  # one shot

        for i, out in enumerate(outs):
            generations[i] = out.outputs[0].text if out.outputs else ""


        
        
        if task == "trec":
            hf_ds = hf_ds.map(
                lambda x: {"input": question_template[task].format(input=x["input"].removesuffix("Type:"))}
            )
        elif task == "triviaqa":
            hf_ds = hf_ds.map(
                lambda x: {"input": question_template[task].format(input=x["input"].removesuffix("Answer:"))}
            )
        elif task == "samsum":
            hf_ds = hf_ds.map(
                lambda x: {"input": question_template[task].format(input=x["input"].removesuffix("Summary:"))}
            )
        else:
            hf_ds = hf_ds.map(lambda x: {"input": question_template[task].format(**x)})

        df = hf_ds.to_pandas()
        df = df.rename(columns={"input": "question"})
        df["answer_prefix"] = answer_prefix.get(task, "")
        # df = df[["context", "question", "answer_prefix", "answers", "all_classes"]]
        df["task"] = task

        # be a bit more generous with token generation to avoid any cut-offs
        df["max_new_tokens"] = DATA_NAME_TO_MAX_NEW_TOKENS[task] + 20
        
        df["cheatsheet"] = generations
        dataset = Dataset.from_pandas(df)

        dataset.push_to_hub(args.repo_id, config_name=task, split="test")
        print(f"Pushed {task} to {args.repo_id}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)