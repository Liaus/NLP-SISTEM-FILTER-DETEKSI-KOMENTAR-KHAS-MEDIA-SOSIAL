import pandas as pd

from src.data_loader import validate_dataset
from src.preprocessing import preprocess_text


def test_preprocess_text_normal_comment():
    text = "Lu gak tau yg gw maksud, @user #bad!"
    output = preprocess_text(text)
    assert isinstance(output, str)
    assert output.strip() != ""
    assert output == "kamu tidak tau saya maksud"


def test_preprocess_preserves_meaning_bearing_words():
    assert preprocess_text("kamu tidak berguna") == "kamu tidak berguna"
    assert preprocess_text("lebih baik kamu diam saja") == "lebih baik kamu diam"
    assert preprocess_text("jangan lakukan itu, hasilnya sangat kurang baik") == (
        "jangan lakukan hasilnya sangat kurang baik"
    )


def test_preprocess_text_empty_string():
    assert preprocess_text("   ") == ""


def test_preprocess_text_none():
    assert preprocess_text(None) == ""


def test_preprocessed_column_is_not_processed_again():
    dataset = pd.DataFrame(
        {"komentar_preprocessed": ["Bukan token baru"], "label": ["aman"]}
    )
    prepared = validate_dataset(dataset)
    assert prepared.loc[0, "komentar_preprocessed"] == "Bukan token baru"


def test_raw_comment_uses_current_preprocessing_once():
    dataset = pd.DataFrame({"komentar": ["Kamu tidak berguna"], "label": ["toksik"]})
    prepared = validate_dataset(dataset)
    assert prepared.loc[0, "komentar_preprocessed"] == "kamu tidak berguna"
