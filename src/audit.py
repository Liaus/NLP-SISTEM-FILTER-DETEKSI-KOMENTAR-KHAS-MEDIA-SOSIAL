from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.config import load_config, project_path
from src.data_loader import normalize_labels, validate_dataset
from src.preprocessing import preprocess_text


POSITIVE_REVIEW_CUES = ("ganteng", "cantik", "hebat", "baik", "bagus", "terima kasih")
NEGATIVE_REVIEW_CUES = ("bodoh", "goblok", "jelek", "sampah", "diam", "menyebalkan")


def _read_source() -> tuple[pd.DataFrame, str, Path]:
    config = load_config()
    raw_path = project_path(*config["data"]["raw_path"].split("/"))
    clean_path = project_path(*config["data"]["clean_path"].split("/"))
    if raw_path.exists():
        return pd.read_csv(raw_path, sep="\t", encoding="latin-1"), "text", raw_path
    if clean_path.exists():
        return pd.read_csv(clean_path, encoding="utf-8-sig"), "komentar", clean_path
    raise FileNotFoundError(
        "Dataset sumber tidak ditemukan. Audit memerlukan data/raw atau data/processed/dataset_clean.csv."
    )


def _format_examples(
    source: pd.DataFrame,
    text_column: str,
    source_labels: set[str],
    cues: tuple[str, ...],
    limit: int = 8,
) -> list[str]:
    labels = normalize_labels(source["label"])
    text = source[text_column].fillna("").astype(str)
    matches = source.loc[
        labels.isin(source_labels)
        & text.str.contains("|".join(re.escape(cue) for cue in cues), case=False, na=False)
    ]
    return [
        f"- [{normalize_labels(pd.Series([row['label']])).iloc[0]}] {str(row[text_column])[:350]}"
        for _, row in matches.head(limit).iterrows()
    ]


def create_data_quality_audit() -> Path:
    """Write observed data quality, label review candidates, and preprocessing examples."""
    config = load_config()
    source, text_column, source_path = _read_source()
    prepared = validate_dataset(source)
    raw_text = source[text_column].fillna("").astype(str)
    normalized_labels = normalize_labels(source["label"])

    duplicate_rows = int(source.duplicated(subset=[text_column], keep=False).sum())
    duplicate_texts = int(source.loc[source.duplicated(subset=[text_column], keep=False), text_column].nunique())
    full_duplicate_rows = int(source.duplicated().sum())
    conflicts = int(
        (
            pd.DataFrame({"text": source[text_column], "label": normalized_labels})
            .groupby("text", dropna=False)["label"]
            .nunique()
            .gt(1)
        ).sum()
    )
    valid_source = pd.DataFrame(
        {"text": raw_text, "label": normalized_labels}
    ).dropna(subset=["text", "label"])

    lines = [
        "NLP COMMENT DETECTION — DATA QUALITY AUDIT",
        "=" * 48,
        f"Source file: {source_path}",
        f"Source shape: {source.shape[0]} rows x {source.shape[1]} columns",
        f"Source columns: {', '.join(map(str, source.columns))}",
        f"Source text column: {text_column}",
        f"Missing values by column: {source.isna().sum().to_dict()}",
        f"Whitespace-only text rows: {int(raw_text.str.strip().eq('').sum())}",
        f"Full-row duplicates: {full_duplicate_rows}",
        f"Raw source label counts: {normalized_labels.value_counts(dropna=False).to_dict()}",
        f"Usable rows after validation, preprocessing, and same-label duplicate removal: {len(prepared)}",
        f"Prepared label counts: {prepared['label'].value_counts().to_dict()}",
        f"Rows participating in exact-text duplicates: {duplicate_rows}",
        f"Unique duplicated texts: {duplicate_texts}",
        f"Exact text values with conflicting labels: {conflicts}",
        "Duplicate text examples (same-label repeats, shown for audit):",
        "",
        "Representative examples by original normalized label (not independently re-labeled):",
    ]
    duplicate_samples = source.loc[
        source.duplicated(subset=[text_column], keep=False), [text_column, "label"]
    ].drop_duplicates()
    duplicate_report = [
        f"- [{normalize_labels(pd.Series([row['label']])).iloc[0]}] {str(row[text_column])[:350]}"
        for _, row in duplicate_samples.iterrows()
    ]
    lines[lines.index(""):lines.index("")] = duplicate_report or ["- None."]
    for label in ("aman", "toksik"):
        lines.append(f"\n{label.upper()}:")
        examples = valid_source.loc[valid_source["label"] == label, "text"].head(8)
        lines.extend(f"- {str(example)[:350]}" for example in examples)

    lines.extend(
        [
            "",
            "Potential label-review examples (keyword search only; NOT confirmed errors):",
            "Positive-sounding cues found in toksik-labeled source comments:",
        ]
    )
    lines.extend(
        _format_examples(source, text_column, {"toksik"}, POSITIVE_REVIEW_CUES)
        or ["- No candidates found."]
    )
    lines.append("Negative-sounding cues found in aman-labeled source comments:")
    lines.extend(
        _format_examples(source, text_column, {"aman"}, NEGATIVE_REVIEW_CUES)
        or ["- No candidates found."]
    )
    lines.extend(
        [
            "These are context-dependent candidates for human inspection only. No source label was changed.",
            "",
            "Preprocessing examples (raw input -> current output):",
        ]
    )
    for example in (
        "kamu ganteng banget",
        "kamu cantik banget",
        "kamu tidak berguna",
        "lebih baik kamu diam saja",
        "dasar tidak tahu diri",
        "terima kasih banyak",
    ):
        lines.append(f"- {example} -> {preprocess_text(example)}")
    lines.extend(
        [
            "",
            "Preprocessing audit findings:",
            "- Negation words (tidak, bukan, jangan), degree/qualification words (kurang, sangat, lebih), and addressees (kamu, saya) are preserved.",
            "- Sastrawi stemming is not applied; it previously reduced interpretable forms such as 'berguna' and 'memalukan'.",
            "- Raw comments are preprocessed once by the same function for training and inference. Preprocessed-only input files are read as-is.",
            "- The dataset is small and topic-specific; sentiment-positive words can occur sarcastically in toksik comments.",
        ]
    )

    output_path = project_path(*config["reports"]["data_quality_audit_path"].split("/"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Data quality audit saved to: {output_path}")
    print(f"Prepared rows: {len(prepared)}")
    print(f"Prepared label counts: {prepared['label'].value_counts().to_dict()}")
    return output_path


if __name__ == "__main__":
    create_data_quality_audit()
