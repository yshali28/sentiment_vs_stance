"""
main.py
Wires the full pipeline: load -> preprocess -> train stance & sentiment
classifiers -> run the disentanglement analysis -> save figures + a
results summary.
"""
import argparse
import json
import os

from data_loader import load_raw
from preprocess import preprocess_dataframe
from baselines import train_and_eval
from disentangle import (
    divergence_rate, cramers_v, proxy_test, target_dependence,
    plot_stance_vs_sentiment, plot_stance_within_sentiment,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def run(model_name="logreg"):
    os.makedirs(OUT_DIR, exist_ok=True)

    train = preprocess_dataframe(load_raw("train"))
    test = preprocess_dataframe(load_raw("test"))
    print(f"Loaded train={len(train)} rows, test={len(test)} rows")

    # ---- Phase 1 baselines ----
    stance_res, _, _ = train_and_eval(
        train["clean_text"], train["stance"], test["clean_text"], test["stance"],
        model_name=model_name, label_name="stance",
    )
    sent_res, _, _ = train_and_eval(
        train["clean_text"], train["sentiment"], test["clean_text"], test["sentiment"],
        model_name=model_name, label_name="sentiment",
    )

    print("\n=== STANCE classification report ===")
    print(stance_res["report"])
    print("Official SemEval macro-F1 (FAVOR/AGAINST):", round(stance_res["official_macro_f1"], 4))

    print("\n=== SENTIMENT classification report ===")
    print(sent_res["report"])

    # ---- Disentanglement analysis (the novel contribution) ----
    # Run on gold TRAIN labels (the true relationship in the data),
    # and separately on TEST-set gold labels for a held-out check.
    rate_train, _ = divergence_rate(train)
    v_train = cramers_v(train)
    proxy_train = proxy_test(train)
    td_train = target_dependence(train)

    rate_test, _ = divergence_rate(test)
    v_test = cramers_v(test)
    proxy_test_ = proxy_test(test)

    plot_stance_vs_sentiment(train, os.path.join(OUT_DIR, "stance_vs_sentiment_heatmap.png"))
    plot_stance_within_sentiment(train, os.path.join(OUT_DIR, "stance_within_sentiment.png"))

    summary = {
        "n_train": len(train),
        "n_test": len(test),
        "model": model_name,
        "stance_official_macro_f1": round(stance_res["official_macro_f1"], 4),
        "divergence_rate_train": round(float(rate_train), 4),
        "divergence_rate_test": round(float(rate_test), 4),
        "cramers_v_train": round(float(v_train), 4),
        "cramers_v_test": round(float(v_test), 4),
        "proxy_overall_accuracy_train": round(float(proxy_train["overall_accuracy"]), 4),
        "proxy_divergent_subset_accuracy_train": proxy_train["divergent_subset_accuracy"],
        "proxy_divergent_subset_fraction_train": round(float(proxy_train["divergent_subset_fraction"]), 4),
        "target_dependence_train": td_train.to_dict(orient="records"),
    }

    with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DISENTANGLEMENT ANALYSIS (real SemEval data) ===")
    print(f"Divergence rate on TRAIN gold labels: {rate_train:.4f} ({rate_train*100:.2f}% of tweets)")
    print(f"Divergence rate on TEST gold labels:  {rate_test:.4f} ({rate_test*100:.2f}% of tweets)")
    print(f"Cramer's V (train): {v_train:.4f}  |  Cramer's V (test): {v_test:.4f}")
    print(f"Sentiment-as-proxy overall accuracy (train): {proxy_train['overall_accuracy']:.4f}")
    print(f"  -> On the divergent subset specifically: {proxy_train['divergent_subset_accuracy']:.4f} "
          f"(guaranteed 0.0 by definition -- see disentangle.py docstring)")
    print("\nDivergence rate by target:")
    print(td_train.to_string(index=False))
    print(f"\nSaved figures + summary to {OUT_DIR}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logreg", choices=["logreg", "svm"])
    args = parser.parse_args()
    run(model_name=args.model)
