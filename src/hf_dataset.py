from __future__ import annotations

import sys
import tempfile
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from huggingface_hub import hf_hub_download

from src.config import project_path
from src.preprocessing import preprocess_text


REPOSITORY_ID = "nahiar/hate_speech_detection"
REPOSITORY_FILE = "data.csv"
DATASET_DIR = project_path("data", "raw", "huggingface_hate_speech_detection")
DATASET_PATH = DATASET_DIR / REPOSITORY_FILE
EXPERIMENT_DATA_DIR = project_path(
    "data", "processed", "huggingface_experiment"
)
MAPPED_DATASET_PATH = EXPERIMENT_DATA_DIR / "mapped_dataset.csv"
CONFLICT_REPORT_PATH = EXPERIMENT_DATA_DIR / "conflicting_tweet_groups.csv"
EXPECTED_LABEL_COLUMNS = (
    "HS",
    "Abusive",
    "HS_Individual",
    "HS_Group",
    "HS_Religion",
    "HS_Race",
    "HS_Physical",
    "HS_Gender",
    "HS_Other",
    "HS_Weak",
    "HS_Moderate",
    "HS_Strong",
)
ORIGINAL_LABEL_COLUMNS = EXPECTED_LABEL_COLUMNS
EXPERIMENT_LABEL_COLUMN = "experiment_label"
GROUP_ID_COLUMN = "tweet_group_id"
GROUP_CONFLICT_COLUMN = "group_target_conflict"
BINARY_ELIGIBLE_COLUMN = "binary_eligible"
SPLIT_STATUS_COLUMN = "split_status"


def _canonicalize_tweet(text: str) -> str:
    """Normalize Unicode, case, and whitespace only for duplicate grouping."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _tweet_group_id(text: str) -> str:
    canonical = _canonicalize_tweet(text)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def map_experiment_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    """Apply the approved HS/Abusive mapping without altering original columns."""
    required = {"Tweet", "HS", "Abusive"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Kolom wajib dataset Hugging Face tidak tersedia: {missing}")

    mapped = dataset.copy()
    if mapped["Tweet"].isna().any():
        raise ValueError("Kolom Tweet memiliki nilai kosong; tidak ada baris yang diubah.")

    mapped["Tweet"] = mapped["Tweet"].astype(str)
    if mapped["Tweet"].str.strip().eq("").any():
        raise ValueError("Kolom Tweet memiliki teks kosong; tidak ada baris yang diubah.")

    for column in ("HS", "Abusive"):
        numeric_values = pd.to_numeric(mapped[column], errors="coerce")
        if numeric_values.isna().any() or not numeric_values.isin([0, 1]).all():
            invalid = sorted(
                mapped.loc[
                    numeric_values.isna() | ~numeric_values.isin([0, 1]), column
                ].astype(str).unique()
            )
            raise ValueError(
                f"Kolom {column} harus berisi nilai biner 0/1. Nilai invalid: {invalid}"
            )
        mapped[column] = numeric_values.astype("int64")

    mapped[EXPERIMENT_LABEL_COLUMN] = "ambigu"
    mapped.loc[mapped["HS"].eq(1), EXPERIMENT_LABEL_COLUMN] = "toksik"
    mapped.loc[
        mapped["HS"].eq(0) & mapped["Abusive"].eq(0),
        EXPERIMENT_LABEL_COLUMN,
    ] = "aman"

    canonical_tweets = mapped["Tweet"].map(_canonicalize_tweet)
    mapped[GROUP_ID_COLUMN] = canonical_tweets.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    mapped["text_processed"] = mapped["Tweet"].map(preprocess_text)

    target_counts = mapped.groupby(GROUP_ID_COLUMN)[EXPERIMENT_LABEL_COLUMN].nunique()
    conflicting_groups = set(target_counts[target_counts > 1].index)
    mapped[GROUP_CONFLICT_COLUMN] = mapped[GROUP_ID_COLUMN].isin(
        conflicting_groups
    )
    mapped[BINARY_ELIGIBLE_COLUMN] = (
        mapped[EXPERIMENT_LABEL_COLUMN].isin({"aman", "toksik"})
        & ~mapped[GROUP_CONFLICT_COLUMN]
        & mapped["text_processed"].str.strip().ne("")
    )

    mapped[SPLIT_STATUS_COLUMN] = "excluded_ambiguous"
    mapped.loc[
        mapped[EXPERIMENT_LABEL_COLUMN].isin({"aman", "toksik"}),
        SPLIT_STATUS_COLUMN,
    ] = "excluded_conflict"
    mapped.loc[
        mapped[BINARY_ELIGIBLE_COLUMN],
        SPLIT_STATUS_COLUMN,
    ] = "not_split"
    mapped.loc[
        mapped["text_processed"].str.strip().eq(""),
        SPLIT_STATUS_COLUMN,
    ] = "excluded_empty_after_preprocessing"
    return mapped


def prepare_experiment_data(
    raw_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw HF rows, add Strategy C fields, and save mapped data plus conflict rows."""
    dataset_path = Path(raw_path) if raw_path is not None else DATASET_PATH
    if not dataset_path.is_absolute():
        dataset_path = project_path(dataset_path.as_posix())
    if not dataset_path.exists():
        if raw_path is not None:
            raise FileNotFoundError(f"Dataset Hugging Face tidak ditemukan: {dataset_path}")
        dataset_path = download_dataset()

    original = load_dataset(dataset_path)
    mapped = map_experiment_labels(original)
    conflict_rows = mapped.loc[mapped[GROUP_CONFLICT_COLUMN]].copy()

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(MAPPED_DATASET_PATH, index=False, encoding="utf-8-sig")
    conflict_rows.to_csv(CONFLICT_REPORT_PATH, index=False, encoding="utf-8-sig")
    return mapped, conflict_rows


def _read_csv(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read the raw CSV without transforming rows or values."""
    csv_path = Path(path)
    try:
        return pd.read_csv(csv_path, encoding="utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1"), "latin-1"


def inspect_dataset(path: str | Path = DATASET_PATH) -> dict[str, Any]:
    """Read-only inspection of CSV structure, missing values, duplicates, and label distributions."""
    dataset, encoding = _read_csv(path)
    label_columns = [column for column in EXPECTED_LABEL_COLUMNS if column in dataset.columns]

    return {
        "path": str(Path(path).resolve()),
        "encoding_used_for_reading": encoding,
        "rows": int(dataset.shape[0]),
        "columns_count": int(dataset.shape[1]),
        "columns": list(dataset.columns),
        "dtypes": {column: str(dtype) for column, dtype in dataset.dtypes.items()},
        "missing_values": {
            column: int(count) for column, count in dataset.isna().sum().items()
        },
        "duplicate_rows": int(dataset.duplicated().sum()),
        "text_column": "Tweet" if "Tweet" in dataset.columns else None,
        "label_columns_found": label_columns,
        "label_distributions": {
            column: {
                str(value): int(count)
                for value, count in dataset[column].value_counts(dropna=False).items()
            }
            for column in label_columns
        },
        "head": dataset.head(5),
    }


def load_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Load the untouched raw dataset. This function performs no preprocessing or label mapping."""
    dataset, _ = _read_csv(path)
    return dataset


def _validate_download(path: str | Path) -> dict[str, Any]:
    report = inspect_dataset(path)
    if report["rows"] <= 0 or report["columns_count"] <= 0:
        raise ValueError(f"Dataset CSV kosong atau tidak memiliki kolom: {path}")
    if report["text_column"] is None:
        raise ValueError(
            f"Kolom teks 'Tweet' tidak ditemukan. Kolom aktual: {report['columns']}"
        )
    if not report["label_columns_found"]:
        raise ValueError(
            "Tidak ada kolom label yang dikenal ditemukan. "
            f"Kolom aktual: {report['columns']}"
        )
    return report


def download_dataset() -> Path:
    """Download only the repository's data.csv and preserve an existing local copy."""
    if DATASET_PATH.exists():
        _validate_download(DATASET_PATH)
        return DATASET_PATH

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    try:
        cached_path = hf_hub_download(
            repo_id=REPOSITORY_ID,
            filename=REPOSITORY_FILE,
            repo_type="dataset",
            token=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Gagal mengunduh {REPOSITORY_FILE} dari dataset publik "
            f"{REPOSITORY_ID}: {exc}"
        ) from exc

    temporary_path: Path | None = None
    try:
        _validate_download(cached_path)
        with tempfile.NamedTemporaryFile(
            dir=DATASET_DIR,
            prefix=".data.csv.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        temporary_path.write_bytes(Path(cached_path).read_bytes())
        _validate_download(temporary_path)
        temporary_path.replace(DATASET_PATH)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return DATASET_PATH


def print_dataset_report(path: str | Path = DATASET_PATH) -> None:
    """Print the requested dataset summary and a five-row sample."""
    report = inspect_dataset(path)
    dataset = report["head"]

    print(f"Dataset: {REPOSITORY_ID}")
    print(f"File: {report['path']}")
    print(f"Rows: {report['rows']}")
    print(f"Columns: {report['columns_count']}")
    print(f"Column names: {report['columns']}")
    print(f"CSV encoding used for reading: {report['encoding_used_for_reading']}")
    print(f"Data types: {report['dtypes']}")
    print(f"Missing values by column: {report['missing_values']}")
    print(f"Exact duplicate rows: {report['duplicate_rows']}")
    print(f"Text column: {report['text_column'] or 'not found'}")
    print(f"Expected label columns found: {report['label_columns_found']}")
    print("Label value counts:")
    for column, distribution in report["label_distributions"].items():
        print(f"  {column}: {distribution}")
    print("First five rows:")
    print(dataset.to_string(index=False))


def main() -> None:
    dataset_path = download_dataset()
    print_dataset_report(dataset_path)


if __name__ == "__main__":
    main()
