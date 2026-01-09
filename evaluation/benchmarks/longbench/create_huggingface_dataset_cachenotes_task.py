
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
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--repo_id", type=str, default="giulio98/LongBench-Qwen3")
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
    "narrativeqa": "You are given a story, which can be either a novel or a movie script. Your task is to answer user question about the story.\n\nStory: {context}\n\n",
    "qasper": 'You are given a scientific article and a question. Your task is to answer user question about the article.\n\nArticle: {context}\n\n',
    "multifieldqa_en": "Read the following text. Your task is to answer user question about the text.\n\n{context}\n\n",
    "hotpotqa": "Your task is to answer user question about the passages.\n\nThe following are given passages.\n{context}\n\n",
    "2wikimqa": "Your task is to answer user question about the passages.\n\nThe following are given passages.\n{context}\n\n",
    "musique": "Your task is to answer user question about the passages.\n\nThe following are given passages.\n{context}\n\n",
    "gov_report": "You are given a report by a government agency. Your task is to summarize the report.\n\nReport:\n{context}\n\n",
    "qmsum": "You are given a meeting transcript. Your task is to answer following user queries based on this transcript.\n\nTranscript:\n{context}\n\n",
    "multi_news": "You are given several news passages. Your task is to summarize all news.\n\nNews:\n{context}\n\n",
    "trec": "Your task is to determine the type of the question. You are given some examples of the Task you have to perform.\n\n{context}\n\n",
    "triviaqa": "Your task is to answer the question based on the given passage. You are given some examples of the Task you have to perform.\n\n{context}\n\n",
    "samsum": "Your task is summarization of dialogues. The following are some examples.\n\n{context}\n\n",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Your task is to count non-repeating paragraphs in total.\n\n{context}\n\n",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract.  Your task is to determine which paragraph the abstract is from\n\n{context}\n\n",
    "lcc": "Your task is code completion.\n{context}\n\n",
    "repobench-p": "Your task is code completion.\n{context}\n\n",
}
cheat_sheet_prompt = {
    "narrativeqa": "Before I give you the question, imagine you are a student memorizing this material according to the task that you will have to perform. Repeat the context concisely yet comprehensively to aid memorization, ensuring you preserve all critical details and create a cheat sheet covering the entire context. Once you have done that, I will give you the question.\n\n",
    "qasper": 'Before I give you the question, imagine you are a student memorizing this material according to the task that you will have to perform. Repeat the context concisely yet comprehensively to aid memorization, ensuring you preserve all critical details and create a cheat sheet covering the entire context. Once you have done that, I will give you the question.\n\n',
    "multifieldqa_en": "Before I give you the question, imagine you are a student memorizing this material according to the task that you will have to perform. Repeat the context concisely yet comprehensively to aid memorization, ensuring you preserve all critical details and create a cheat sheet covering the entire context. Once you have done that, I will give you the question.\n\n",
    "hotpotqa": "Before I give you the question, imagine you are a student memorizing this material according to the task that you will have to perform. Repeat the context concisely yet comprehensively to aid memorization, ensuring you preserve all critical details and create a cheat sheet covering the entire context. Once you have done that, I will give you the question.\n\n",
    "2wikimqa": "Before I give you the question, imagine you are a student memorizing this material according to the task that you will have to perform. Repeat the context concisely yet comprehensively to aid memorization, ensuring you preserve all critical details and create a cheat sheet covering the entire context. Once you have done that, I will give you the question.\n\n",
    "musique": "Before I give you the question, imagine you are a student memorizing this material according to the task that you will have to perform. Repeat the context concisely yet comprehensively to aid memorization, ensuring you preserve all critical details and create a cheat sheet covering the entire context. Once you have done that, I will give you the question.\n\n",
    "gov_report": "Before solving the summarization task, imagine you are a student memorizing this material according to the task that you will have to perform. Repeat the context concisely yet comprehensively to aid memorization, ensuring you preserve all critical details and create a cheat sheet covering the entire context. Once you have done that, I will prompt you to solve the task.\n\n",
    "qmsum": "Before I give you the question, imagine you are a student memorizing this material according to the task that you will have to perform. Repeat the context concisely yet comprehensively to aid memorization, ensuring you preserve all critical details and create a cheat sheet covering the entire context. Once you have done that, I will give you the question.\n\n",
    "multi_news": "Before solving the summarization task, imagine you are a student memorizing this material according to the task that you will have to perform. Repeat the context concisely yet comprehensively to aid memorization, ensuring you preserve all critical details and create a cheat sheet covering the entire context. Once you have done that, I will prompt you to solve the task.\n\n",
    "trec": "Before I give you the new question, imagine you are a student memorizing this material according to the task that you will have to perform. Reason about the given examples and the expected output, ensuring you preserve all critical details and create a cheat sheet covering the entire context. Once you have done that, I give you the new question.\n\n",
    "triviaqa": "Before I give you the new passage and the question, imagine you are a student memorizing this material according to the task that you will have to perform. Reason about the given examples and the expected output, ensuring you preserve all critical details and create a cheat sheet covering the entire context. Once you have done that, I give you the new passage and question.\n\n",
    "samsum": "Before I give you the new dialogue to summarize, imagine you are a student memorizing this material according to the task that you will have to perform. Reason about the given examples and the expected output, ensuring you preserve all critical details and create a cheat sheet covering the entire context. Once you have done that, I give you the new dialogue.\n\n",
    "passage_count": "Before solving the counting task, imagine you are a student memorizing this material according to the task that you will have to perform. Ensure you preserve all critical details to solve the task and create a cheat sheet covering the entire context. Once you have done that, I will prompt you to solve the task.\n\n",
    "passage_retrieval_en": "Before solving the passage retrieval task and give you the abstract, imagine you are a student memorizing this material according to the task that you will have to perform. Ensure you preserve all critical details to solve the task and create a cheat sheet covering the entire context. Once you have done that, I will prompt you to solve the task giving you the abstract.\n\n",
    "lcc": "Before solving the code completion task, imagine you are a student memorizing this material according to the task that you will have to perform. Ensure you preserve all critical details to solve the task and create a cheat sheet covering the entire context. Once you have done that, I will prompt you to solve the code completion task.\n\n",
    "repobench-p": "Before giving you the last snippet of code and solving the code completion task, imagine you are a student memorizing this material according to the task that you will have to perform. Ensure you preserve all critical details to solve the task and create a cheat sheet covering the entire context. Once you have done that, I will prompt you to solve the code completion task giving you the last snippet of code.\n\n",
    
}

context_prefix = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\n",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\n",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\n",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\n",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\n",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\n",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\n",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\n",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\n",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n",
    "lcc": "Please complete the code given below. \n{context}",
    "repobench-p": "Please complete the code given below. \n{context}",
}
TASKS = [
    "narrativeqa", "qasper","multifieldqa_en","hotpotqa","2wikimqa","musique",
    "gov_report","qmsum","multi_news","trec","triviaqa","samsum",
    "passage_count","passage_retrieval_en","lcc","repobench-p",
]
question_template = {
    "narrativeqa": "Question: {input}\n\n",
    "qasper": "Question: {input}\n\n",
    "multifieldqa_en": "Question: {input}\n",
    "multifieldqa_zh": "问题：{input}\n",
    "hotpotqa": "Question: {input}\n",
    "2wikimqa": "Question: {input}\n",
    "musique": "Question: {input}\n",
    "dureader": "问题：{input}\n",
    "gov_report": "Now, write a one-page summary of the report.\n\n",
    "qmsum": "Query: {input}\n",
    "multi_news": "Now, write a one-page summary of all the news.\n\n",
    "vcsum": "",
    "trec": "{input}",
    "triviaqa": "{input}",
    "samsum": "{input}",
    "lsht": "{input}",
    "passage_count": "Please enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\n",
    "passage_retrieval_en": '{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\n',
    "passage_retrieval_zh": '{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n',
    "lcc": "{input}",
    "repobench-p": "{input}",
}
answer_prefix = {
    "narrativeqa": "Answer:",
    "qasper": "Answer:",
    "multifieldqa_en": "Answer:",
    "multifieldqa_zh": "回答：",
    "hotpotqa": "Answer:",
    "2wikimqa": "Answer:",
    "musique": "Answer:",
    "dureader": "回答：",
    "gov_report": "Summary:",
    "qmsum": "Answer:",
    "trec": "Type:",
    "multi_news": "Summary:",
    "samsum": "Summary:",
    "triviaqa": "Answer:",
    "vcsum": "会议总结：",
    "passage_count": "The final answer is: ",
    "passage_retrieval_en": "The answer is: ",
    "passage_retrieval_zh": "答案是：",
    "lcc": "Next line of code:\n",
    "repobench-p": "Next line of code:\n",
}
DATA_NAME_TO_MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
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
                "prompt_cheatsheet": cheat_sheet_prompt[task],
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