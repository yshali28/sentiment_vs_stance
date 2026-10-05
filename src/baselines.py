"""
baselines.py
Phase 1 baselines: TF-IDF (uni+bigrams) + Logistic Regression / Linear SVM,
trained separately for stance (FAVOR/AGAINST/NONE) and sentiment
(pos/neg/other), on the identical train/test split so predictions are
directly comparable.

Reports the official SemEval macro-F1 (over FAVOR and AGAINST only,
NONE excluded) for stance, plus standard per-class metrics for both tasks.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score


def build_model(model_name="logreg"):
    if model_name == "svm":
        clf = LinearSVC(class_weight="balanced", max_iter=5000)
    else:
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    return vec, clf


def train_and_eval(train_texts, train_labels, test_texts, test_labels,
                    model_name="logreg", label_name="stance"):
    vec, clf = build_model(model_name)
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    clf.fit(Xtr, train_labels)
    preds = clf.predict(Xte)

    report = classification_report(test_labels, preds, zero_division=0)
    result = {
        "label_name": label_name,
        "model": model_name,
        "report": report,
        "predictions": preds,
    }

    if label_name == "stance":
        # official SemEval metric: macro-F1 over FAVOR & AGAINST only
        mask_labels = ["FAVOR", "AGAINST"]
        result["official_macro_f1"] = f1_score(
            test_labels, preds, labels=mask_labels, average="macro", zero_division=0
        )
    return result, clf, vec


if __name__ == "__main__":
    from data_loader import load_raw
    from preprocess import preprocess_dataframe

    train = preprocess_dataframe(load_raw("train"))
    test = preprocess_dataframe(load_raw("test"))

    stance_res, _, _ = train_and_eval(
        train["clean_text"], train["stance"], test["clean_text"], test["stance"],
        model_name="logreg", label_name="stance",
    )
    print("=== STANCE (LogReg) ===")
    print(stance_res["report"])
    print("Official SemEval macro-F1 (FAVOR/AGAINST):", round(stance_res["official_macro_f1"], 4))

    sent_res, _, _ = train_and_eval(
        train["clean_text"], train["sentiment"], test["clean_text"], test["sentiment"],
        model_name="logreg", label_name="sentiment",
    )
    print("\n=== SENTIMENT (LogReg) ===")
    print(sent_res["report"])
