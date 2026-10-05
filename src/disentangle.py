"""
disentangle.py
The project's novel contribution: quantifying where stance and sentiment
diverge, rather than assuming a relationship between them.

Metrics:
  - divergence_rate: fraction of tweets where gold stance and gold sentiment
    disagree in the "surprising" direction (FAVOR+negative or AGAINST+positive).
    This is an EMPIRICAL property of the real data -- it is NOT guaranteed to
    be any particular value, unlike proxy accuracy on that same subset.
  - cramers_v: association strength between stance and sentiment labels
    (0 = independent, 1 = perfectly associated).
  - proxy_test: maps sentiment -> a predicted stance (pos->FAVOR, neg->AGAINST,
    other->NONE) and scores it against gold stance. NOTE: by construction,
    proxy accuracy on the divergent subset is mathematically guaranteed to be
    0% -- that is a property of the definition, not a discovery. The
    genuinely empirical question is how LARGE the divergent subset is
    (the divergence rate) and what the proxy's overall accuracy is once
    that subset is folded in.
  - target_dependence: divergence rate broken out per target, showing
    whether the sentiment/stance relationship depends on the target.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def divergence_rate(df, stance_col="stance", sentiment_col="sentiment"):
    divergent = (
        ((df[stance_col] == "FAVOR") & (df[sentiment_col] == "neg")) |
        ((df[stance_col] == "AGAINST") & (df[sentiment_col] == "pos"))
    )
    return divergent.mean(), divergent


def cramers_v(df, col1="stance", col2="sentiment"):
    ct = pd.crosstab(df[col1], df[col2])
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.sum().sum()
    r, k = ct.shape
    return np.sqrt((chi2 / n) / (min(r - 1, k - 1)))


def proxy_test(df, stance_col="stance", sentiment_col="sentiment"):
    mapping = {"pos": "FAVOR", "neg": "AGAINST", "other": "NONE"}
    proxy_pred = df[sentiment_col].map(mapping)
    overall_acc = (proxy_pred == df[stance_col]).mean()

    _, divergent_mask = divergence_rate(df, stance_col, sentiment_col)
    if divergent_mask.sum() > 0:
        divergent_acc = (proxy_pred[divergent_mask] == df[stance_col][divergent_mask]).mean()
    else:
        divergent_acc = float("nan")

    return {
        "overall_accuracy": overall_acc,
        "divergent_subset_accuracy": divergent_acc,  # guaranteed 0.0 by construction -- see note above
        "divergent_subset_size": int(divergent_mask.sum()),
        "divergent_subset_fraction": float(divergent_mask.mean()),
    }


def target_dependence(df, target_col="target", stance_col="stance", sentiment_col="sentiment"):
    rows = []
    for target, group in df.groupby(target_col):
        rate, _ = divergence_rate(group, stance_col, sentiment_col)
        rows.append({"target": target, "n": len(group), "divergence_rate": rate})
    return pd.DataFrame(rows).sort_values("divergence_rate", ascending=False)


def plot_stance_vs_sentiment(df, out_path, stance_col="stance", sentiment_col="sentiment"):
    ct = pd.crosstab(df[stance_col], df[sentiment_col])
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
    plt.title("Stance vs Sentiment (gold labels)")
    plt.ylabel("Stance")
    plt.xlabel("Sentiment")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_stance_within_sentiment(df, out_path, stance_col="stance", sentiment_col="sentiment"):
    ct = pd.crosstab(df[sentiment_col], df[stance_col])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    ct_norm.plot(kind="bar", stacked=True, figsize=(6, 4.5),
                 color=["#2a9d8f", "#e76f51", "#adb5bd"])
    plt.title("Stance distribution within each sentiment class")
    plt.ylabel("proportion")
    plt.xlabel("Sentiment")
    plt.legend(title="Stance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    from data_loader import load_raw

    df = load_raw("train")
    rate, _ = divergence_rate(df)
    v = cramers_v(df)
    proxy = proxy_test(df)
    td = target_dependence(df)

    print(f"Divergence rate (gold, real train data): {rate:.4f}")
    print(f"Cramer's V (stance vs sentiment association): {v:.4f}")
    print("Proxy test:", proxy)
    print("\nTarget dependence:")
    print(td.to_string(index=False))
