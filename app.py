"""
app.py
Live demo for the stance vs sentiment project.
Run from the project root:   streamlit run app.py

Tab 1: type a tweet + pick a target, see what the baseline and BERTweet predict.
Tab 2: browse the saved test-set predictions (no models needed).
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, "src")
from data_loader import load_raw, TASK_A_TARGETS
from preprocess import preprocess_dataframe, clean_tweet
from baselines import train_and_eval

MERGED_PATH = "outputs/phase1_vs_phase2_merged.csv"
STANCE_MODEL_DIR = "models/stance_model"
SENTIMENT_MODEL_DIR = "models/sentiment_model"

# label order used when the BERTweet models were trained (see the Kaggle notebook)
STANCE_LABELS = ["AGAINST", "FAVOR", "NONE"]
SENTIMENT_LABELS = ["neg", "pos", "other"]
SENT_NAMES = {"neg": "negative", "pos": "positive", "other": "other"}

st.set_page_config(page_title="Stance vs Sentiment", layout="wide")


def is_divergent(stance, sentiment):
    # same definition as the report: FAVOR + negative, or AGAINST + positive
    return (stance == "FAVOR" and sentiment == "neg") or (stance == "AGAINST" and sentiment == "pos")


# ---------------------------------------------------------------------
# Baseline: TF-IDF + logistic regression, trained when the app starts
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the baseline (a few seconds)...")
def load_baseline():
    train = preprocess_dataframe(load_raw("train"))
    test = preprocess_dataframe(load_raw("test"))
    out = {}
    for label in ["stance", "sentiment"]:
        res, clf, vec = train_and_eval(train["clean_text"], train[label],
                                       test["clean_text"], test[label],
                                       model_name="logreg", label_name=label)
        out[label] = (clf, vec)
    return out


def baseline_predict(models, text):
    clean = clean_tweet(text)
    result = {}
    for label in ["stance", "sentiment"]:
        clf, vec = models[label]
        probs = clf.predict_proba(vec.transform([clean]))[0]
        result[label] = dict(zip(clf.classes_, probs))
    return result


# ---------------------------------------------------------------------
# BERTweet: loads the fine-tuned models from models/ if they are there
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading BERTweet models...")
def load_bertweet():
    if not (os.path.isdir(STANCE_MODEL_DIR) and os.path.isdir(SENTIMENT_MODEL_DIR)):
        return None, "BERTweet models not found. Put the saved folders in models/ (see README)."
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        return None, "torch / transformers are not installed."
    tok = AutoTokenizer.from_pretrained(STANCE_MODEL_DIR)
    stance_model = AutoModelForSequenceClassification.from_pretrained(STANCE_MODEL_DIR).eval()
    sent_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR).eval()
    return (tok, stance_model, sent_model), None


def bertweet_predict(bundle, text, target):
    import torch
    tok, stance_model, sent_model = bundle
    # same input as training: the target and the cleaned tweet as a sentence pair
    enc = tok(target, clean_tweet(text), truncation=True, max_length=96, return_tensors="pt")
    result = {}
    with torch.no_grad():
        for label, model, names in [("stance", stance_model, STANCE_LABELS),
                                    ("sentiment", sent_model, SENTIMENT_LABELS)]:
            probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
            result[label] = dict(zip(names, probs))
    return result


def show_prediction(title, preds):
    stance = max(preds["stance"], key=preds["stance"].get)
    sentiment = max(preds["sentiment"], key=preds["sentiment"].get)
    st.subheader(title)
    st.markdown(f"**Stance:** {stance}")
    st.bar_chart(pd.Series(preds["stance"]).reindex(["FAVOR", "AGAINST", "NONE"]), height=150)
    st.markdown(f"**Sentiment:** {SENT_NAMES[sentiment]}")
    st.bar_chart(pd.Series(preds["sentiment"]).reindex(["pos", "neg", "other"]), height=150)
    if is_divergent(stance, sentiment):
        st.warning("Divergent: the tone and the stance point in opposite directions.")
    else:
        st.info("Not divergent: the tone and the stance agree (or one is neutral).")


# ---------------------------------------------------------------------
# page
# ---------------------------------------------------------------------
st.title("Stance vs Sentiment in Tweets")
st.caption("SemEval-2016 Task 6 | Baseline (TF-IDF + logistic regression) vs target-aware BERTweet")

tab_live, tab_browse = st.tabs(["Try a tweet", "Browse test results"])

with tab_live:
    merged = pd.read_csv(MERGED_PATH) if os.path.exists(MERGED_PATH) else None

    # a few ready-made divergent test tweets to click through
    examples = {}
    if merged is not None:
        div = merged[((merged["stance"] == "FAVOR") & (merged["sentiment"] == "neg")) |
                     ((merged["stance"] == "AGAINST") & (merged["sentiment"] == "pos"))]
        recovered = div[(div["pred_stance_phase1"] != div["stance"]) & (div["pred_stance_phase2"] == div["stance"])]
        for _, r in recovered.sample(min(12, len(recovered)), random_state=3).iterrows():
            examples[f"{r['target']}: {r['text'][:60]}"] = (r["text"], r["target"], r["stance"], r["sentiment"])

    choice = st.selectbox("Pick an example (divergent test tweets) or type your own below",
                          ["(type my own)"] + list(examples))
    if choice != "(type my own)":
        default_text, default_target, gold_stance, gold_sent = examples[choice]
        st.caption(f"Gold labels in the dataset: stance = {gold_stance}, sentiment = {SENT_NAMES[gold_sent]}")
    else:
        default_text, default_target = "", TASK_A_TARGETS[0]

    left, right = st.columns([3, 1])
    # the key changes with the example so the boxes refresh when a new one is picked
    text = left.text_area("Tweet", value=default_text, height=100, key=f"text_{choice}")
    target = right.selectbox("Target", TASK_A_TARGETS, index=TASK_A_TARGETS.index(default_target),
                             key=f"target_{choice}")

    if st.button("Predict", type="primary") and text.strip():
        col1, col2 = st.columns(2)
        with col1:
            show_prediction("Baseline (TF-IDF + LogReg)", baseline_predict(load_baseline(), text))
        with col2:
            bundle, err = load_bertweet()
            if bundle is None:
                st.subheader("Target-aware BERTweet")
                st.error(err)
            else:
                show_prediction("Target-aware BERTweet", bertweet_predict(bundle, text, target))

with tab_browse:
    if merged is None:
        st.error(f"{MERGED_PATH} not found. Run compare_phase1_phase2.py first.")
    else:
        df = merged.copy()
        df["divergent"] = ((df["stance"] == "FAVOR") & (df["sentiment"] == "neg")) | \
                          ((df["stance"] == "AGAINST") & (df["sentiment"] == "pos"))

        def outcome(r):
            b, m = r["pred_stance_phase1"] == r["stance"], r["pred_stance_phase2"] == r["stance"]
            if m and not b:
                return "BERTweet fixed it"
            if b and not m:
                return "BERTweet broke it"
            return "both right" if b else "both wrong"
        df["outcome"] = df.apply(outcome, axis=1)

        c1, c2, c3 = st.columns(3)
        t = c1.selectbox("Target", ["All"] + TASK_A_TARGETS)
        only_div = c2.checkbox("Only divergent tweets", value=True)
        o = c3.selectbox("Outcome", ["All", "BERTweet fixed it", "BERTweet broke it", "both wrong", "both right"])
        if t != "All":
            df = df[df["target"] == t]
        if only_div:
            df = df[df["divergent"]]
        if o != "All":
            df = df[df["outcome"] == o]

        st.write(f"{len(df)} tweets")
        show = df[["text", "target", "stance", "sentiment", "pred_stance_phase1", "pred_stance_phase2", "outcome"]]
        show = show.rename(columns={"pred_stance_phase1": "baseline", "pred_stance_phase2": "BERTweet"})
        st.dataframe(show, hide_index=True)
