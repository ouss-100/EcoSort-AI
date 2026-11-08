from model.model import is_model_available, reset_model, get_model
from model import model as m


def test_is_model_available_false_when_missing(tmp_path):
    fake_path = str(tmp_path / "nonexistent.pt")
    assert is_model_available(fake_path) is False


def test_is_model_available_true_when_file_exists(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake weights")
    assert is_model_available(str(weights)) is True


def test_reset_model_clears_singleton():
    reset_model()

    assert m._model_instance is None
    assert m._model_path_loaded is None


def test_get_model_returns_none_for_missing_weights(tmp_path):
    reset_model()
    
    result = get_model(str(tmp_path / "missing.pt"))
    assert result is None

