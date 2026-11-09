from unittest.mock import patch

from src.predict import predict_comment


class StubModel:
    classes_ = ["aman", "toksik"]

    def predict(self, _texts):
        return ["aman"]

    def predict_proba(self, _texts):
        return [[0.8, 0.2]]


def test_prediction_returns_label_and_confidence():
    with patch("src.predict.load_model", return_value=StubModel()):
        label, confidence = predict_comment("Kamu tidak berguna")
    assert label in {"aman", "toksik"}
    assert label == "aman"
    assert confidence == 0.8


def test_prediction_rejects_empty_preprocessed_input():
    with patch("src.predict.load_model", return_value=StubModel()):
        try:
            predict_comment("http://example.com")
        except ValueError as exc:
            assert "preprocessing" in str(exc)
        else:
            raise AssertionError("Expected ValueError for input that preprocesses to empty text")
