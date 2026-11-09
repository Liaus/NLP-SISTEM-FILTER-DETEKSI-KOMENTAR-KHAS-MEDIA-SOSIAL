from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.config import load_config, project_path
from src.preprocessing import cleaning_text, preprocess_text

RAW_DATASET_URL = "https://raw.githubusercontent.com/cahkanor/id-cyberbullying-instagram/master/id-cyberbullying-instagram.tsv"


def download_raw_dataset(force: bool = False) -> Path:
    """Download the raw dataset referenced by the previous notebook when it is not present in the workspace."""
    config = load_config()
    raw_path = project_path(*config["data"]["raw_path"].split("/"))
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists() and not force:
        return raw_path

    urllib.request.urlretrieve(RAW_DATASET_URL, raw_path)
    return raw_path


def choose_text_column(df: pd.DataFrame, preferred: str = "komentar_preprocessed") -> str:
    """Prefer original text so it can be transformed once by the active preprocessing function."""
    candidates = ["komentar", "text", preferred, "komentar_clean"]
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Dataset tidak memiliki kolom teks yang valid. Kolom yang tersedia: {df.columns.tolist()}")


def normalize_labels(series: pd.Series) -> pd.Series:
    """Standardize labels to aman and toksik."""
    labels = series.astype(str).str.strip().str.lower()
    label_map = {
        "non_cb": "aman",
        "noncb": "aman",
        "non-cb": "aman",
        "cb": "toksik",
        "aman": "aman",
        "toksik": "toksik",
    }
    labels = labels.replace(label_map)
    return labels


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, and prepare the dataset for training."""
    if df.empty:
        raise ValueError("Dataset kosong setelah dibaca.")

    if "label" not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'label'.")

    text_col = choose_text_column(df)
    cleaned = df[[text_col, "label"]].copy()
    cleaned.columns = ["text_raw", "label_raw"]
    cleaned = cleaned.dropna(subset=["text_raw", "label_raw"]).copy()
    cleaned["text_raw"] = cleaned["text_raw"].astype(str)
    cleaned["label_raw"] = cleaned["label_raw"].astype(str)
    cleaned = cleaned[cleaned["text_raw"].str.strip() != ""].copy()
    cleaned = cleaned[cleaned["label_raw"].str.strip() != ""].copy()

    cleaned["label"] = normalize_labels(cleaned["label_raw"])

    if text_col in {"komentar", "text"}:
        cleaned["text"] = cleaned["text_raw"].apply(preprocess_text)
    else:
        cleaned["text"] = cleaned["text_raw"].str.strip()

    valid_labels = {"aman", "toksik"}
    unknown_labels = sorted(set(cleaned["label"].dropna()) - valid_labels)
    if unknown_labels:
        raise ValueError(
            "Dataset memiliki label yang tidak dikenal setelah normalisasi: "
            f"{unknown_labels}. Label yang diizinkan: {sorted(valid_labels)}."
        )

    if cleaned.empty:
        raise ValueError("Dataset tidak memiliki label yang valid setelah normalisasi.")

    cleaned["text"] = cleaned["text"].replace({"": pd.NA})
    cleaned = cleaned.dropna(subset=["text"]).copy()
    cleaned = cleaned.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("Dataset kosong setelah pembersihan data dan duplikasi dihapus.")

    return cleaned[["text", "label"]].rename(columns={"text": "komentar_preprocessed", "label": "label"})


def _write_processed_files(
    source_df: pd.DataFrame,
    text_col: str,
    processed: pd.DataFrame,
    config: dict,
) -> None:
    """Refresh processed CSV outputs from original comments using the active preprocessing function."""
    processed_path = project_path(*config["data"]["processed_path"].split("/"))
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_path, index=False, encoding="utf-8-sig")

    clean_path = project_path(*config["data"]["clean_path"].split("/"))
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean = source_df[[text_col, "label"]].copy()
    clean = clean.dropna(subset=[text_col, "label"]).copy()
    clean = clean.rename(columns={text_col: "komentar"})
    clean["komentar"] = clean["komentar"].astype(str)
    clean["label"] = normalize_labels(clean["label"])
    clean = clean[clean["label"].isin({"aman", "toksik"})]
    clean = clean[clean["komentar"].str.strip() != ""]
    clean = clean.drop_duplicates(subset=["komentar", "label"]).reset_index(drop=True)
    clean["komentar_clean"] = clean["komentar"].apply(cleaning_text)
    clean["komentar_preprocessed"] = clean["komentar"].apply(preprocess_text)
    clean = clean[clean["komentar_preprocessed"].str.strip() != ""]
    clean.to_csv(clean_path, index=False, encoding="utf-8-sig")


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Load raw comments when available and preprocess them once; use preprocessed-only CSVs as-is."""
    config = load_config()

    source_is_raw = False
    if dataset_path is not None:
        source_path = Path(dataset_path)
        if not source_path.is_absolute():
            source_path = project_path(source_path.as_posix())
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset tidak ditemukan: {source_path}")
    else:
        raw_path = project_path(*config["data"]["raw_path"].split("/"))
        clean_path = project_path(*config["data"]["clean_path"].split("/"))
        processed_path = project_path(*config["data"]["processed_path"].split("/"))

        if raw_path.exists():
            source_path = raw_path
            source_is_raw = True
        elif clean_path.exists():
            source_path = clean_path
            source_is_raw = True
        elif processed_path.exists():
            source_path = processed_path
        else:
            source_path = download_raw_dataset()
            source_is_raw = True

    if source_path.suffix.lower() == ".tsv":
        df = pd.read_csv(source_path, sep="\t", encoding="latin-1")
    else:
        df = pd.read_csv(source_path, encoding="utf-8-sig")

    text_col = choose_text_column(df)
    prepared = validate_dataset(df)
    if source_is_raw and text_col in {"komentar", "text"}:
        _write_processed_files(df, text_col, prepared, config)

    if prepared["label"].isnull().any():
        raise ValueError("Terdapat label kosong pada dataset final.")

    if prepared["komentar_preprocessed"].isnull().any() or (prepared["komentar_preprocessed"].astype(str).str.strip() == "").any():
        raise ValueError("Terdapat komentar kosong pada dataset final.")

    print(f"Jumlah data final: {len(prepared)}")
    print("Distribusi label:")
    print(prepared["label"].value_counts())
    return prepared


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
