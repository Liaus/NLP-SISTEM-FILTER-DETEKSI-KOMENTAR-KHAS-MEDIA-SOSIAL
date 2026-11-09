from __future__ import annotations

import re
from typing import Any

import emoji


SLANG_DICT: dict[str, str] = {
    "yg": "yang",
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "enggak": "tidak",
    "tdk": "tidak",
    "bgt": "banget",
    "banget": "banget",
    "aja": "saja",
    "jd": "jadi",
    "dgn": "dengan",
    "utk": "untuk",
    "krn": "karena",
    "karna": "karena",
    "gw": "saya",
    "gue": "saya",
    "gua": "saya",
    "lu": "kamu",
    "lo": "kamu",
    "km": "kamu",
    "kmu": "kamu",
    "udh": "sudah",
    "udah": "sudah",
    "blm": "belum",
    "bkn": "bukan",
    "emg": "memang",
    "sm": "sama",
    "dr": "dari",
    "dpt": "dapat",
    "org": "orang",
    "kalo": "kalau",
    "kl": "kalau",
    "tp": "tetapi",
    "tpi": "tetapi",
    "trs": "terus",
    "mantul": "mantap",
    "wkwk": "",
    "wkwkwk": "",
    "lol": "",
}

STOPWORDS: set[str] = {
    "yang", "dan", "di", "ke", "dari",
    "ini", "itu", "atau", "untuk", "pada",
    "dengan", "adalah", "akan", "dalam",
    "juga", "karena",
    "ia", "dia", "mereka", "kami", "kita",
    "ada", "sudah", "belum", "saja",
    "agar", "oleh", "sebagai", "dapat",
}


def normalize_slang(text: str) -> str:
    """Normalize slang terms using the same dictionary from the EDA notebook."""
    if text is None:
        return ""
    words = str(text).lower().split()
    normalized = [SLANG_DICT.get(word, word) for word in words]
    return " ".join(part for part in normalized if part)


def cleaning_text(text: Any) -> str:
    """Apply the same cleaning steps used in the notebook: lowercasing, URL removal, mention/hashtag removal, emoji removal, digit removal, punctuation stripping."""
    if text is None:
        return ""

    cleaned = str(text).strip()
    if cleaned == "":
        return ""

    cleaned = cleaned.lower()
    cleaned = re.sub(r"http\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"@\w+|#\w+", " ", cleaned)
    cleaned = emoji.replace_emoji(cleaned, replace=" ")
    cleaned = re.sub(r"\d+", " ", cleaned)
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def preprocess_text(text: Any) -> str:
    """Clean and normalize text while preserving negation and sentiment-bearing words."""
    if text is None:
        return ""

    value = str(text).strip()
    if value == "" or value.lower() in {"nan", "none"}:
        return ""

    cleaned = cleaning_text(value)
    if cleaned == "":
        return ""

    normalized = normalize_slang(cleaned)
    tokens = [token for token in normalized.split() if token and token not in STOPWORDS]
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "Lu gak tau yg gw maksud, @user #bad!"
    print(preprocess_text(sample))
