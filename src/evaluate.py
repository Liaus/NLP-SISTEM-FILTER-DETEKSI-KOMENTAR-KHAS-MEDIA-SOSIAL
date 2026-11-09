from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import matplotlib
import pandas as pd
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import ensure_dir, load_config, project_path
from src.data_loader import load_dataset
from src.preprocessing import preprocess_text
from src.train import get_train_test_split

MANUAL_CASES = [
    ("aman", "kamu ganteng banget"),
    ("aman", "kamu cantik banget"),
    ("aman", "kamu baik sekali"),
    ("aman", "terima kasih banyak"),
    ("aman", "semangat terus, kamu hebat"),
    ("aman", "hasil karyamu sangat bagus"),
    ("aman", "saya suka dengan desain ini"),
    ("aman", "kamu luar biasa"),
    ("aman", "selamat atas pencapaiannya"),
    ("aman", "konten ini sangat membantu"),
    ("toksik", "dasar bodoh, tidak punya otak"),
    ("toksik", "muka kamu jelek sekali"),
    ("toksik", "kamu tidak berguna"),
    ("toksik", "dasar manusia sampah"),
    ("toksik", "lebih baik kamu diam saja"),
    ("toksik", "orang seperti kamu memang menyebalkan"),
    ("toksik", "kamu sangat memalukan"),
    ("toksik", "dasar tidak tahu diri"),
    ("toksik", "pergi saja dari sini"),
    ("toksik", "tidak ada yang suka denganmu"),
]


def _preserve_baseline(config: dict) -> dict | None:
    """Keep the original evaluation artifacts before replacing them with current results."""
    metrics_path = project_path(*config["reports"]["metrics_path"].split("/"))
    baseline_path = project_path(*config["reports"]["baseline_metrics_path"].split("/"))
    baseline_report_path = project_path(
        *config["reports"]["baseline_classification_report_path"].split("/")
    )

    if baseline_path.exists():
        with baseline_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    if not metrics_path.exists():
        return None

    with metrics_path.open("r", encoding="utf-8") as fp:
        baseline = json.load(fp)
    ensure_dir(baseline_path.parent)
    with baseline_path.open("w", encoding="utf-8") as fp:
        json.dump(baseline, fp, indent=2, ensure_ascii=False)

    current_report_path = project_path(
        *config["reports"]["classification_report_path"].split("/")
    )
    if current_report_path.exists() and not baseline_report_path.exists():
        shutil.copyfile(current_report_path, baseline_report_path)
    return baseline


def _manual_prediction_results(model) -> pd.DataFrame:
    """Run the specified manual cases as a separate evaluation set, never as training data."""
    rows = []
    for expected_label, comment in MANUAL_CASES:
        processed = preprocess_text(comment)
        predicted_label = str(model.predict([processed])[0])
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([processed])[0]
            class_index = list(model.classes_).index(predicted_label)
            confidence = float(probabilities[class_index])
        rows.append(
            {
                "komentar": comment,
                "expected_label": expected_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "is_correct": predicted_label == expected_label,
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(model_path: str | Path | None = None):
    """Evaluate the trained model on the fixed test split and save metrics plus visualizations."""
    config = load_config()
    df = load_dataset()
    _, X_test, _, y_test = get_train_test_split(df)

    if model_path is None:
        model_path = project_path(*config["model"]["output_path"].split("/"))
    model = joblib.load(model_path)
    baseline = _preserve_baseline(config)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    labels = sorted(df[config["data"]["label_column"]].unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    report_details = classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "precision_by_class": {
            label: float(report_details[label]["precision"]) for label in labels
        },
        "recall_by_class": {
            label: float(report_details[label]["recall"]) for label in labels
        },
        "f1_by_class": {
            label: float(report_details[label]["f1-score"]) for label in labels
        },
    }
    if baseline:
        comparable_keys = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        metrics["baseline"] = {
            key: float(baseline[key]) for key in comparable_keys if key in baseline
        }
        metrics["delta_vs_baseline"] = {
            key: metrics[key] - metrics["baseline"][key]
            for key in comparable_keys
            if key in metrics and key in metrics["baseline"]
        }
    if hasattr(model, "nlp_training_metadata_"):
        metrics["cross_validation"] = model.nlp_training_metadata_

    reports_dir = project_path(*config["reports"]["output_dir"].split("/"))
    ensure_dir(reports_dir)

    report_path = project_path(
        *config["reports"]["classification_report_path"].split("/")
    )
    if baseline:
        report += (
            "\n\nBaseline-to-current comparison (same fixed stratified holdout):\n"
            f"Accuracy: {baseline.get('accuracy', 'unavailable')} -> {accuracy:.4f}\n"
            f"Macro precision: {baseline.get('precision_macro', 'unavailable')} -> {precision:.4f}\n"
            f"Macro recall: {baseline.get('recall_macro', 'unavailable')} -> {recall:.4f}\n"
            f"Macro F1: {baseline.get('f1_macro', 'unavailable')} -> {f1:.4f}\n"
        )

    with report_path.open("w", encoding="utf-8") as fp:
        fp.write(report)

    with (project_path(*config["reports"]["metrics_path"].split("/"))).open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)

    manual_results = _manual_prediction_results(model)
    manual_path = project_path(*config["reports"]["manual_prediction_path"].split("/"))
    manual_results.to_csv(manual_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()
    plt.savefig(project_path(*config["reports"]["confusion_matrix_path"].split("/")), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=config["data"]["label_column"], order=["aman", "toksik"])
    plt.title("Distribusi Label")
    plt.xlabel("Label")
    plt.ylabel("Jumlah")
    plt.tight_layout()
    plt.savefig(project_path(*config["reports"]["label_distribution_path"].split("/")), dpi=300, bbox_inches="tight")
    plt.close()

    print("Accuracy:", accuracy)
    print("Precision (macro):", precision)
    print("Recall (macro):", recall)
    print("F1-score (macro):", f1)
    print("\nClassification Report:\n", report)
    print(
        f"\nManual prediction accuracy: "
        f"{manual_results['is_correct'].sum()}/{len(manual_results)}"
    )
    print(f"Manual prediction report saved to: {manual_path}")

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "manual_predictions": manual_results,
    }


if __name__ == "__main__":
    evaluate_model()
