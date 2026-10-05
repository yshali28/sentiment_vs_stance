"""
preprocess.py
Stance-aware tweet cleaning: strips URLs/mentions/the #SemST tag, splits
hashtags into words, lowercases, but KEEPS negation cues (not, n't, no,
never) since they can flip stance/sentiment meaning.
"""
import re

NEGATION_WORDS = {"not", "no", "never", "n't", "cannot", "nothing", "none"}

_url_re = re.compile(r"http\S+|www\.\S+")
_mention_re = re.compile(r"@\w+")
_semst_re = re.compile(r"#semst", re.IGNORECASE)
_hashtag_re = re.compile(r"#(\w+)")
_camel_split_re = re.compile(r"(?<=[a-z])(?=[A-Z])")
_non_alpha_re = re.compile(r"[^a-zA-Z'\s]")
_multi_space_re = re.compile(r"\s+")


def _split_hashtag(word):
    # split CamelCase hashtags into separate words, e.g. #ClimateChange -> Climate Change
    return _camel_split_re.sub(" ", word)


def clean_tweet(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _url_re.sub(" ", text)
    text = _mention_re.sub(" ", text)
    text = _semst_re.sub(" ", text)

    def _hashtag_repl(m):
        return " " + _split_hashtag(m.group(1)) + " "
    text = _hashtag_re.sub(_hashtag_repl, text)

    text = _non_alpha_re.sub(" ", text)
    text = _multi_space_re.sub(" ", text).strip().lower()
    return text


def preprocess_dataframe(df, text_col="text"):
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_tweet)
    return df


if __name__ == "__main__":
    samples = [
        "I hate how the government keeps attacking this policy #SemST",
        "@user Great, another reason to reject it! http://t.co/xyz #SemST",
        "This is NOT okay #ClimateChangeIsReal #SemST",
    ]
    for s in samples:
        print(repr(s), "->", repr(clean_tweet(s)))
