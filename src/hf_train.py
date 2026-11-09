from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import project_path
from src.hf_dataset import (
    BINARY_ELIGIBLE_COLUMN,
    EXPERIMENT_DATA_DIR,
    EXPERIMENT_LABEL_COLUMN,
    GROUP_CONFLICT_COLUMN,
    GROUP_ID_COLUMN,
    MAPPED_DATASET_PATH,
    SPLIT_STATUS_COLUMN,
    CONFLICT_REPORT_PATH,
    prepare_experiment_data,
)
from src.preprocessing import preprocess_text


TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
TFIDF_PARAMETERS: dict[str, Any] = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 1,
    "strip_accents": "unicode",
}
NAIVE_BAYES_PARAMETERS: dict[str, Any] = {"alpha": 2.0}
EXPERIMENT_MODEL_DIR = project_path("models", "huggingface_experiment")
EXPERIMENT_REPORT_DIR = project_path("reports", "huggingface_experiment")
MODEL_PATH = EXPERIMENT_MODEL_DIR / "tfidf_multinomial_nb.joblib"
SPLIT_PATH = EXPERIMENT_DATA_DIR / "train_test_split.csv"
TRAINING_SUMMARY_PATH = EXPERIMENT_REPORT_DIR / "training_summary.json"


def build_experiment_pipeline() -> Pipeline:
    """Build a separate TF-IDF + Multinomial Naive Bayes experiment pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMETERS)),
            ("classifier", MultinomialNB(**NAIVE_BAYES_PARAMETERS)),
        ]
    )


def make_grouped_train_test_split(
    mapped_dataset: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    n_splits: int = CV_FOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create an approximately stratified holdout while keeping each normalized Tweet group intact."""
    required = {
        "text_processed",
        EXPERIMENT_LABEL_COLUMN,
        GROUP_ID_COLUMN,
        BINARY_ELIGIBLE_COLUMN,
        GROUP_CONFLICT_COLUMN,
    }
    missing = sorted(required - set(mapped_dataset.columns))
    if missing:
        raise ValueError(f"Dataset mapped kehilangan kolom wajib: {missing}")

    eligible = mapped_dataset.loc[mapped_dataset[BINARY_ELIGIBLE_COLUMN]].copy()
    if eligible.empty:
        raise ValueError("Tidak ada baris eligible untuk split binary.")

    class_counts = eligible[EXPERIMENT_LABEL_COLUMN].value_counts()
    if set(class_counts.index) != {"aman", "toksik"}:
        raise ValueError(
            "Eksperimen memerlukan kedua kelas aman dan toksik. "
            f"Distribusi eligible: {class_counts.to_dict()}"
        )

    groups_per_class = eligible.groupby(EXPERIMENT_LABEL_COLUMN)[GROUP_ID_COLUMN].nunique()
    if int(groups_per_class.min()) < n_splits:
        raise ValueError(
            f"Tidak cukup kelompok Tweet untuk {n_splits}-fold group stratification: "
            f"{groups_per_class.to_dict()}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    X = eligible["text_processed"].astype(str)
    y = eligible[EXPERIMENT_LABEL_COLUMN].astype(str)
    groups = eligible[GROUP_ID_COLUMN].astype(str)
    overall_distribution = y.value_counts(normalize=True)

    best_candidate: tuple[float, np.ndarray, np.ndarray] | None = None
    for train_indices, test_indices in splitter.split(X, y, groups):
        test_labels = y.iloc[test_indices]
        if set(y.iloc[train_indices]) != {"aman", "toksik"}:
            continue
        if set(test_labels) != {"aman", "toksik"}:
            continue

        test_distribution = test_labels.value_counts(normalize=True)
        distribution_error = sum(
            abs(float(test_distribution[label]) - float(overall_distribution[label]))
            for label in ("aman", "toksik")
        )
        size_error = abs(len(test_indices) / len(eligible) - test_size)
        score = size_error + distribution_error
        if best_candidate is None or score < best_candidate[0]:
            best_candidate = (score, train_indices, test_indices)

    if best_candidate is None:
        raise ValueError(
            "StratifiedGroupKFold tidak menghasilkan train/test dengan kedua kelas "
            "di kedua partisi."
        )

    _, train_indices, test_indices = best_candidate
    train = eligible.iloc[train_indices].copy()
    test = eligible.iloc[test_indices].copy()

    train_groups = set(train[GROUP_ID_COLUMN])
    test_groups = set(test[GROUP_ID_COLUMN])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(
            f"Group leakage terdeteksi: {len(overlap)} grup Tweet muncul pada train dan test."
        )

    split_dataset = mapped_dataset.copy()
    split_dataset[SPLIT_STATUS_COLUMN] = "excluded_ambiguous"
    split_dataset.loc[
        split_dataset[GROUP_CONFLICT_COLUMN],
        SPLIT_STATUS_COLUMN,
    ] = "excluded_conflict"
    split_dataset.loc[
        split_dataset["text_processed"].astype(str).str.strip().eq(""),
        SPLIT_STATUS_COLUMN,
    ] = "excluded_empty_after_preprocessing"
    split_dataset.loc[train.index, SPLIT_STATUS_COLUMN] = "train"
    split_dataset.loc[test.index, SPLIT_STATUS_COLUMN] = "test"

    train_indices_set = set(train.index)
    test_indices_set = set(test.index)
    if train_indices_set.intersection(test_indices_set):
        raise RuntimeError("Ada baris yang masuk ke train dan test sekaligus.")

    return train, test, split_dataset


def load_experiment_model(model_path: str | Path = MODEL_PATH) -> Pipeline:
    """Load the independent Hugging Face experiment model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model eksperimen belum tersedia: {path}. Jalankan python src/hf_train.py."
        )
    return joblib.load(path)


def predict_experiment_comment(text: str) -> tuple[str, float | None]:
    """Preprocess one comment with the shared project function and predict using the experiment model."""
    if text is None or not str(text).strip():
        raise ValueError("Komentar tidak boleh kosong.")
    processed = preprocess_text(text)
    if not processed:
        raise ValueError("Komentar kosong setelah preprocessing.")

    model = load_experiment_model()
    prediction = str(model.predict([processed])[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([processed])[0]
        class_index = list(model.classes_).index(prediction)
        confidence = float(probabilities[class_index])
    return prediction, confidence


def train_huggingface_experiment() -> dict[str, Any]:
    """Prepare Strategy C data, create group-safe split, train, and save only experiment artifacts."""
    mapped, conflict_rows = prepare_experiment_data()
    train, test, split_dataset = make_grouped_train_test_split(mapped)

    pipeline = build_experiment_pipeline()
    pipeline.fit(
        train["text_processed"].astype(str),
        train[EXPERIMENT_LABEL_COLUMN].astype(str),
    )

    group_sizes = mapped.groupby(GROUP_ID_COLUMN).size()
    repeated_groups = group_sizes[group_sizes > 1]
    model_metadata = {
        "dataset": "nahiar/hate_speech_detection",
        "label_mapping": {
            "HS=1": "toksik",
            "HS=0 and Abusive=0": "aman",
            "HS=0 and Abusive=1": "ambigu",
        },
        "tfidf_parameters": {
            **TFIDF_PARAMETERS,
            "ngram_range": list(TFIDF_PARAMETERS["ngram_range"]),
        },
        "naive_bayes_parameters": NAIVE_BAYES_PARAMETERS,
        "test_size_target": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "split_method": "StratifiedGroupKFold; one of five folds selected by class-proportion and size proximity",
        "group_split_folds": CV_FOLDS,
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "training_group_count": int(train[GROUP_ID_COLUMN].nunique()),
        "test_group_count": int(test[GROUP_ID_COLUMN].nunique()),
        "repeated_tweet_group_count": int(len(repeated_groups)),
        "rows_in_repeated_tweet_groups": int(repeated_groups.sum()),
        "extra_duplicate_rows": int((repeated_groups - 1).sum()),
        "group_conflict_count": int(conflict_rows[GROUP_ID_COLUMN].nunique()),
        "group_conflict_rows": int(len(conflict_rows)),
        "ambiguous_rows": int(mapped[EXPERIMENT_LABEL_COLUMN].eq("ambigu").sum()),
        "binary_eligible_rows": int(mapped[BINARY_ELIGIBLE_COLUMN].sum()),
        "class_distribution": {
            "train": train[EXPERIMENT_LABEL_COLUMN].value_counts().to_dict(),
            "test": test[EXPERIMENT_LABEL_COLUMN].value_counts().to_dict(),
        },
    }
    pipeline.nlp_huggingface_experiment_metadata_ = model_metadata

    EXPERIMENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    split_dataset.to_csv(SPLIT_PATH, index=False, encoding="utf-8-sig")
    with TRAINING_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(model_metadata, file, indent=2, ensure_ascii=False)

    print(f"Mapped dataset: {MAPPED_DATASET_PATH}")
    print(f"Conflict rows report: {CONFLICT_REPORT_PATH}")
    print(f"Train rows: {len(train)}; test rows: {len(test)}")
    print(f"Train labels: {model_metadata['class_distribution']['train']}")
    print(f"Test labels: {model_metadata['class_distribution']['test']}")
    print(
        f"Excluded ambiguous rows: {model_metadata['ambiguous_rows']}; "
        f"conflicting groups: {model_metadata['group_conflict_count']} "
        f"({model_metadata['group_conflict_rows']} rows)"
    )
    print(f"Experiment model saved to: {MODEL_PATH}")
    return model_metadata


if __name__ == "__main__":
    train_huggingface_experiment()
