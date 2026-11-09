from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib

from src.config import load_config, project_path
from src.preprocessing import preprocess_text


def load_model(model_path: str | Path | None = None):
    """Load the persisted TF-IDF + Naive Bayes pipeline."""
    config = load_config()
    if model_path is None:
        model_path = project_path(*config["model"]["output_path"].split("/"))

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model belum tersedia di {model_path}. Jalankan training terlebih dahulu dengan: python src/train.py"
        )

    return joblib.load(model_path)


def predict_comment(text: str):
    """Predict whether a comment is aman or toksik and return the label plus probability confidence when available."""
    if text is None or not str(text).strip():
        raise ValueError("Input komentar tidak boleh kosong.")

    model = load_model()
    cleaned = preprocess_text(text)
    if cleaned == "":
        raise ValueError("Komentar setelah preprocessing masih kosong. Mohon masukkan teks yang valid.")

    prediction = model.predict([cleaned])[0]
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]
        class_index = list(model.classes_).index(prediction)
        confidence = float(probabilities[class_index])

    return prediction, confidence


if __name__ == "__main__":
    label, confidence = predict_comment("Kamu keren banget, terima kasih")
    print(label, confidence)
