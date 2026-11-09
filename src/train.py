from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import ensure_dir, load_config, project_path
from src.data_loader import load_dataset


def get_train_test_split(df: pd.DataFrame):
    """Return the same train/test split used throughout the project to keep evaluation consistent."""
    config = load_config()
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]

    X = df[text_col].astype(str)
    y = df[label_col].astype(str).str.strip().str.lower()

    if len(np.unique(y)) < 2:
        raise ValueError("Dataset hanya memiliki 1 kelas, sehingga train/test split stratified tidak dapat dijalankan.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(
            "Dataset terlalu kecil untuk stratified split yang valid. Minimal dibutuhkan representasi kedua kelas di data training dan testing."
        )

    return X_train, X_test, y_train, y_test


def build_pipeline() -> Pipeline:
    """Create the TF-IDF + Multinomial Naive Bayes pipeline described in the project specification."""
    config = load_config()
    tfidf_cfg = config["model"]["tfidf"]
    nb_cfg = config["model"]["naive_bayes"]

    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features", 5000),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        min_df=tfidf_cfg.get("min_df", 1),
        strip_accents=tfidf_cfg.get("strip_accents", "unicode"),
    )
    classifier = MultinomialNB(alpha=nb_cfg.get("alpha", 1.0))

    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


def train_model() -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Tune on training folds only, refit on the training partition, and save the selected pipeline."""
    config = load_config()
    df = load_dataset()
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    search_config = config["model"]["cross_validation"]
    if int(y_train.value_counts().min()) < search_config["folds"]:
        raise ValueError(
            "Jumlah sampel pada kelas terkecil di training tidak cukup untuk "
            f"{search_config['folds']}-fold stratified cross-validation."
        )

    param_grid = {
        "tfidf__ngram_range": [tuple(value) for value in search_config["ngram_range"]],
        "tfidf__min_df": search_config["min_df"],
        "tfidf__max_features": search_config["max_features"],
        "classifier__alpha": search_config["alpha"],
    }
    cv = StratifiedKFold(
        n_splits=search_config["folds"],
        shuffle=True,
        random_state=config["model"]["random_state"],
    )
    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    pipeline = search.best_estimator_
    pipeline.nlp_training_metadata_ = {
        "cv_folds": search_config["folds"],
        "cv_best_macro_f1": float(search.best_score_),
        "best_params": search.best_params_,
        "training_samples": int(len(X_train)),
    }

    model_path = project_path(*config["model"]["output_path"].split("/"))
    ensure_dir(model_path.parent)
    joblib.dump(pipeline, model_path)

    print(f"Best CV macro-F1: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Model berhasil disimpan di: {model_path}")
    return pipeline, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_model()
