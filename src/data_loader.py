"""
data_loader.py
Loads the SemEval-2016 Task 6 dataset (dual-labeled: Stance + Sentiment).
Falls back to a small built-in sample if the real CSVs aren't present,
so the pipeline is always runnable end-to-end.
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

TASK_A_TARGETS = [
    "Hillary Clinton",
    "Feminist Movement",
    "Legalization of Abortion",
    "Atheism",
    "Climate Change is a Real Concern",
]

# A small, schema-faithful sample used only if the real CSVs are missing.
# Sentiment labels follow the official scheme: pos / neg / other (neutral).
_SAMPLE_ROWS = [
    ("I hate how the government keeps attacking this policy #SemST", "Climate Change is a Real Concern", "FAVOR", "neg"),
    ("Great, another reason to reject it #SemST", "Legalization of Abortion", "AGAINST", "pos"),
    ("Climate change is real and we must act now #SemST", "Climate Change is a Real Concern", "FAVOR", "pos"),
    ("This is just fear-mongering nonsense #SemST", "Climate Change is a Real Concern", "AGAINST", "neg"),
    ("I don't really have an opinion on this #SemST", "Atheism", "NONE", "other"),
]


def load_raw(split="train"):
    """Load the real CSV if present, else fall back to the sample."""
    path = os.path.join(DATA_DIR, f"{split}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="latin1", engine="python", quoting=1)
        df = df[df["Target"].isin(TASK_A_TARGETS)].reset_index(drop=True)
        df = df.rename(columns={"Tweet": "text", "Target": "target",
                                 "Stance": "stance", "Sentiment": "sentiment"})
        return df[["text", "target", "stance", "sentiment"]]
    else:
        return pd.DataFrame(_SAMPLE_ROWS, columns=["text", "target", "stance", "sentiment"])


if __name__ == "__main__":
    for split in ["train", "test"]:
        df = load_raw(split)
        print(f"{split}: {df.shape[0]} rows, targets={df['target'].nunique()}")
        print(df['stance'].value_counts().to_dict())
        print(df['sentiment'].value_counts().to_dict())
