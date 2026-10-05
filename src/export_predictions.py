"""
export_predictions.py
Uses the existing data_loader / preprocess / baselines modules to train the
Phase 1 stance classifier and save its row-level test-set predictions to
outputs/phase1_predictions.csv — the one thing main.py never saved on its
own (it only saved the aggregate summary + figures).

Run this from inside src/, same way you run main.py:
    cd src
    python export_predictions.py
"""
import os
import pandas as pd

from data_loader import load_raw
from preprocess import preprocess_dataframe
from baselines import train_and_eval

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def run(model_name="logreg"):
    os.makedirs(OUT_DIR, exist_ok=True)

    train = preprocess_dataframe(load_raw("train"))
    test = preprocess_dataframe(load_raw("test"))
    print(f"Loaded train={len(train)} rows, test={len(test)} rows")

    # train_and_eval already predicts on the test set internally and hands
    # those predictions back in result["predictions"] — no need to retrain
    # or predict separately here
    stance_res, _, _ = train_and_eval(
        train["clean_text"], train["stance"], test["clean_text"], test["stance"],
        model_name=model_name, label_name="stance",
    )
    print("Phase 1 official stance macro-F1:", round(stance_res["official_macro_f1"], 4))

    out = test[["text", "target", "stance", "sentiment"]].copy()
    out["pred_stance"] = stance_res["predictions"]

    out_path = os.path.join(OUT_DIR, "phase1_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} — {len(out)} rows")


if __name__ == "__main__":
    run()
