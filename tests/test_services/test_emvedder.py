import pytest
import numpy as np
from unittest.mock import patch, MagicMock, ANY
from app.services.embedder import Embedder


@pytest.fixture
def mock_model():
    """Fixture that returns a mocked SentenceTransformer model."""
    mock = MagicMock()
    mock.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    return mock


@patch("app.services.embedder.SentenceTransformer")
def test_ensure_model_loads(mock_sentence_transformer):
    mock_sentence_transformer.return_value = MagicMock(
        encode=MagicMock(return_value=np.array([[0.1,0.2,0.3]], dtype=np.float32))
    )
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    embedder._ensure_model()
    assert embedder._model is not None
    mock_sentence_transformer.assert_called_once_with(
        "all-MiniLM-L6-v2", cache_folder="./models", device=ANY
    )


@patch("app.services.embedder.SentenceTransformer")
def test_embed_returns_numpy_array(mock_sentence_transformer, mock_model):
    mock_sentence_transformer.return_value = mock_model
    embedder = Embedder()
    result = embedder.embed(["Test text 1", "Test text 2"])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (1, 3)

@patch("app.services.embedder.SentenceTransformer")
def test_embed_propagates_errors(mock_sentence_transformer):
    failing_model = MagicMock()
    failing_model.encode.side_effect = Exception("Embedding failed")
    mock_sentence_transformer.return_value = failing_model
    embedder = Embedder()
    with pytest.raises(Exception) as excinfo:
        embedder.embed(["Some text"])
    assert "Embedding failed" in str(excinfo.value)
