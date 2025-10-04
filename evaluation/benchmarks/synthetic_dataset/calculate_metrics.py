 


import re, string
import pandas as pd

# ---------- normalisation helpers ----------
_ARTICLE_RE = re.compile(r'\b(a|an|the)\b', flags=re.I)
_PUNC = set(string.punctuation)

def _normalise_token(tok: str) -> str:
    """Normalise a single token (already stripped)."""
    tok = tok.lower()
    tok = _ARTICLE_RE.sub(' ', tok)            # drop articles
    tok = ''.join(ch for ch in tok if ch not in _PUNC)  # drop punct.
    tok = ' '.join(tok.split())               # squeeze whitespace
    return tok


_SPLIT_RE = re.compile(r'[,\;/|\s]+')   # one or more of , ; / | OR any whitespace

def normalise_and_split(ans: str) -> list[str]:
    """
    "IT, Marketing / finance"  -->  ["it", "marketing", "finance"]
    Splits on comma, semicolon, slash, vertical-bar, or any whitespace.
    """
    parts = _SPLIT_RE.split(ans)          # <= HERE: whitespace also delimiters
    tokens = [_normalise_token(p) for p in parts]
    return [t for t in tokens if t]       # drop empties



def word_overlap(pred: list[str], ref: list[str]) -> tuple[float,float,float]:
    """Precision, recall, F1 for one row."""
    ps, rs = set(pred), set(ref)
    inter = len(ps & rs)
    if not ps and not rs:        # both empty → perfect match
        return 1.0, 1.0, 1.0
    if not ps or not rs:         # one empty → zero scores
        return 0.0, 0.0, 0.0
    precision = inter / len(ps)
    recall    = inter / len(rs)
    f1 = 2 * precision * recall / (precision + recall) if precision+recall else 0.0
    return precision, recall, f1




def calculate_metrics(df: pd.DataFrame) -> dict:
    out = {}
    processed_groups = []

    for raw_complexity, g in df.groupby("complexity", sort=False):
        complexity = "direct_retrieval" if raw_complexity == 0 else "join_like"

        g = g.copy()
        g["pred_list"] = g["predicted_answer"].apply(normalise_and_split)
        g["ref_list"]  = g["answer"].apply(normalise_and_split)

        g[["prec", "rec", "f1"]] = g.apply(
            lambda r: pd.Series(word_overlap(r.pred_list, r.ref_list)), axis=1
        )

        out[complexity] = {
            "precision": float(g["prec"].mean()),   # ← cast here
            "recall":    float(g["rec"].mean()),
            "f1":        float(g["f1"].mean()),
        }
        processed_groups.append(g)

    all_rows = pd.concat(processed_groups, ignore_index=True)
    out["overall"] = {
        "precision": float(all_rows["prec"].mean()),
        "recall":    float(all_rows["rec"].mean()),
        "f1":        float(all_rows["f1"].mean()),
    }
    return out

