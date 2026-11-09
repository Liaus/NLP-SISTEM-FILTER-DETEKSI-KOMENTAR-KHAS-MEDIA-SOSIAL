from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import project_path
from src.hf_dataset import (
    EXPERIMENT_LABEL_COLUMN,
    GROUP_ID_COLUMN,
    SPLIT_STATUS_COLUMN,
)
from src.hf_train import (
    EXPERIMENT_REPORT_DIR,
    MODEL_PATH,
    SPLIT_PATH,
    TRAINING_SUMMARY_PATH,
)


METRICS_PATH = EXPERIMENT_REPORT_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = EXPERIMENT_REPORT_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = EXPERIMENT_REPORT_DIR / "confusion_matrix.png"
CONFUSION_MATRIX_CSV_PATH = EXPERIMENT_REPORT_DIR / "confusion_matrix.csv"


def evaluate_huggingface_experiment() -> dict[str, Any]:
    """Evaluate the saved experiment pipeline using only the persisted group-safe test rows."""
    for path in (MODEL_PATH, SPLIT_PATH, TRAINING_SUMMARY_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Artefak eksperimen belum lengkap: {path}. "
                "Jalankan python src/hf_train.py terlebih dahulu."
            )

    split_dataset = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")
    test = split_dataset.loc[split_dataset[SPLIT_STATUS_COLUMN].eq("test")].copy()
    train = split_dataset.loc[split_dataset[SPLIT_STATUS_COLUMN].eq("train")].copy()
    if test.empty or train.empty:
        raise ValueError("File split eksperimen tidak memiliki train/test yang valid.")

    group_overlap = set(train[GROUP_ID_COLUMN]).intersection(test[GROUP_ID_COLUMN])
    if group_overlap:
        raise RuntimeError(
            f"Group leakage terdeteksi pada evaluasi: {len(group_overlap)} grup."
        )

    model = joblib.load(MODEL_PATH)
    y_true = test[EXPERIMENT_LABEL_COLUMN].astype(str)
    y_pred = model.predict(test["text_processed"].astype(str))
    labels = ["aman", "toksik"]

    report_details = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics: dict[str, Any] = {
        "dataset": "nahiar/hate_speech_detection",
        "test_rows": int(len(test)),
        "test_groups": int(test[GROUP_ID_COLUMN].nunique()),
        "split_method": "StratifiedGroupKFold; persisted test partition",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_toksik": float(
            precision_score(
                y_true, y_pred, pos_label="toksik", average="binary", zero_division=0
            )
        ),
        "recall_toksik": float(
            recall_score(
                y_true, y_pred, pos_label="toksik", average="binary", zero_division=0
            )
        ),
        "f1_toksik": float(
            f1_score(
                y_true, y_pred, pos_label="toksik", average="binary", zero_division=0
            )
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_by_class": {
            label: float(report_details[label]["precision"]) for label in labels
        },
        "recall_by_class": {
            label: float(report_details[label]["recall"]) for label in labels
        },
        "f1_by_class": {
            label: float(report_details[label]["f1-score"]) for label in labels
        },
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
        "label_mapping": {
            "HS=1": "toksik",
            "HS=0 and Abusive=0": "aman",
            "HS=0 and Abusive=1": "ambigu (excluded from binary training)",
        },
    }

    legacy_metrics_path = project_path("reports", "metrics.json")
    if legacy_metrics_path.exists():
        with legacy_metrics_path.open("r", encoding="utf-8") as file:
            metrics["legacy_primary_reference"] = json.load(file)
        metrics["legacy_comparison_note"] = (
            "Not directly comparable: the legacy project evaluates a separate, "
            "small cyberbullying dataset with different source and label definitions."
        )

    EXPERIMENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    CLASSIFICATION_REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    pd.DataFrame(
        cm,
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    ).to_csv(CONFUSION_MATRIX_CSV_PATH, encoding="utf-8-sig")

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Hugging Face Experiment — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (toksik): {metrics['precision_toksik']:.4f}")
    print(f"Recall (toksik): {metrics['recall_toksik']:.4f}")
    print(f"F1-score (toksik): {metrics['f1_toksik']:.4f}")
    print(f"Macro-F1: {metrics['f1_macro']:.4f}")
    print(f"Confusion matrix: {cm.tolist()}")
    print(report)
    print(f"Metrics saved to: {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    evaluate_huggingface_experiment()
