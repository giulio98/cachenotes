import json
import openai
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re

# === CONFIGURATION ===
RAG_CSV = (
    "/path/to/rag/results"
)
CACHENOTES_CSV = (
    "/path/to/cachenotes/results"
)
DATASET_PATH = "giulio98/company-notes"
# Where to write the evaluation results
OUTPUT_JSON = EXP_CSV.replace('.csv', '_eval_res.json')

# Prefix constants to strip
ASSISTANT_PREFIX = (
    "You are a helpful assistant who can answer the user query according ONLY the SAP Note provided. "
    "Provide detailed and accurate information based on the user's questions, ensuring that the responses "
    "are relevant and informative. Here is the SAP Note context (HTML):\n\n"
)
PREFIX_LEN = len(ASSISTANT_PREFIX)
QUESTION_PREFIX = "Here is the user question:\n"
QUESTION_SUFFIX = "\n\nPlease provide your answer."

# ───────────────────────── helpers ─────────────────────────

def extract_question(q_field: str) -> str:
    # remove question wrapper
    if q_field.startswith(QUESTION_PREFIX) and q_field.endswith(QUESTION_SUFFIX):
        return q_field[len(QUESTION_PREFIX):-len(QUESTION_SUFFIX)]
    return q_field.strip()

# === HELPER FUNCTIONS ===
def map_ranking(ranking: str):
    if "A" in ranking:
        return [1, 2]
    elif "B" in ranking:
        return [2, 1]
    else:
        return [1, 1]


def parse_res(evaluation: str):
    # Remove formatting and split into blocks
    blocks = [b.strip() for b in evaluation.replace('*', '').split("\n\n") if b.strip()]
    results = []
    for block in blocks:
        lines = [l for l in block.split("\n") if l.strip()]
        aspect, ranking = lines[0].split(": ", 1)
        reason = lines[1].split(": ", 1)[1]
        results.append({
            "aspect": aspect,
            "ranking": map_ranking(ranking),
            "reason": reason
        })
    return results


def process_request(messages):
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL")
    )
    model = os.getenv("OPENAI_API_MODEL")
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(model=model, messages=messages)
            return parse_res(resp.choices[0].message.content)
        except Exception as e:
            if attempt == 3:
                print("ERROR generating response after retries:", e)
                return None
SELECTED_NOTES = {
    1642148, 1999930, 2081473, 2100040, 2119087, 2154870, 2160391,
    2169283, 2186744, 2220627, 2222200, 2222250, 2570371,
}
if __name__ == "__main__":
    # Load both CSV files
    df_exp  = pd.read_csv(EXP_CSV)
    df_hypo = pd.read_csv(HYPO_CSV)

    # Load the HF test split once, unchanged
    ds_full = load_dataset(DATASET_PATH)["test"].to_pandas()

    # Get the *row numbers* whose sap_note_id is selected
    sel_idx = [
        i for i, nid in enumerate(ds_full["sap_note_id"])
        if nid in SELECTED_NOTES
    ]

    if not sel_idx:
        raise ValueError("None of the requested IDs are present in the dataset!")

    # Use those indices everywhere so the three objects stay aligned
    ds = ds_full.select(sel_idx)                   # HF Dataset
    df_ctx = ds.to_pandas()                        # Context/questions
    df_exp  = df_exp .iloc[sel_idx].reset_index(drop=True)
    df_hypo = df_hypo.iloc[sel_idx].reset_index(drop=True)

    # Optional sanity-check
    assert len(df_ctx) == len(df_exp) == len(df_hypo), "Row counts diverged!"
    questions = df_ctx["question"].apply(extract_question).tolist()
    contexts = [
        c[PREFIX_LEN:] if isinstance(c, str) and c.startswith(ASSISTANT_PREFIX) else str(c)
        for c in df_ctx["context"].tolist()
    ]

    # System + User prompts
    SYS = (
        "You are a specialized Support Engineer at SAP tasked with comparing two answers provided in response to a given question and the context. "
        "Evaluate them based on the relevance and correctness of the information, the richness and diversity of the content as well as practical utility"
        "then provide a ranking (A/B) and a brief reason for each dimension."
    )
    USR = (
        "Here is the question:\n{question}\n"
        "-- start of answer A --\n{ansA}\n-- end of answer A --\n"
        "-- start of answer B --\n{ansB}\n-- end of answer B --\n"
        "Here is the context:\n{context}\n"
        "Respond strictly in the format:"
        "\nWinner: A/B\nReason: ..."
    )

    # Build all messages
    all_pairs = []
    for idx, ((_, row_exp), (_, row_hypo)) in enumerate(zip(df_exp.iterrows(), df_hypo.iterrows())):
        q = questions[idx]
        aA = "A: " + str(row_exp['predicted_answer'])
        aB = "B: " + str(row_hypo['predicted_answer'])
        ctx = contexts[idx]

        messages = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": USR.format(
                question=q,
                ansA=aA,
                ansB=aB,
                context=ctx
            )}
        ]
        all_pairs.append((idx, messages))

    # Evaluate in parallel
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_request, msgs): idx for idx, msgs in all_pairs}
        for future in tqdm(futures):
            idx = futures[future]
            res = future.result()
            if res:
                results.append({
                    'index': idx,
                    'question': questions[idx],
                    'eval_result': res,
                    'answers': [
                        df_exp.loc[idx, 'predicted_answer'],
                        df_hypo.loc[idx, 'predicted_answer']
                    ]
                })
                print(f"Processed {idx} with result: {res}")

    # Save to JSON
    with open(OUTPUT_JSON, 'w') as out_f:
        json.dump(results, out_f, indent=2)
    print(f"Saved {len(results)} evaluations to {OUTPUT_JSON}")
