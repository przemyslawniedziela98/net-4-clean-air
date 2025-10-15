import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.qdrant_wrapper import QdrantWrapper, QdrantConfig
from app.models import AppConfig


@pytest.fixture
def sample_df():
    """Return a sample DataFrame for testing upsert."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "TITLE OF THE PAPER": ["Title1", "Title2", "Title3"],
        "AIM OF THE PAPER": ["Aim1", "Aim2", "Aim3"],
        "MAIN FINDINGS OF THE PAPER": ["Finding1", "Finding2", "Finding3"],
        "document": ["doc1", "doc2", "doc3"]
    })


@pytest.fixture
def embeddings():
    """Return dummy embeddings matching the sample DataFrame."""
    return np.array([[0.1,0.2,0.3]] * 3)


@patch("app.services.qdrant_wrapper.QdrantClient")
def test_ensure_collection_creates(mock_client):
    """Test that ensure_collection calls recreate_collection if collection does not exist."""
    mock_instance = mock_client.return_value
    mock_instance.collection_exists.return_value = False
    wrapper = QdrantWrapper(AppConfig(), collection_name="test_collection")
    wrapper.client = mock_instance
    wrapper.ensure_collection(vector_size=3)
    mock_instance.recreate_collection.assert_called_once()


@patch("app.services.qdrant_wrapper.QdrantClient")
def test_upsert_dataframe_calls_upsert(mock_client, sample_df, embeddings):
    """Test that upsert_dataframe calls Qdrant upsert in batches."""
    mock_instance = mock_client.return_value
    wrapper = QdrantWrapper(AppConfig(), collection_name="test_collection")
    wrapper.client = mock_instance
    wrapper.upsert_dataframe(sample_df, embeddings, id_column="id")
    assert mock_instance.upsert.call_count > 0


@patch("app.services.qdrant_wrapper.QdrantClient")
def test_search_returns_expected_format(mock_client):
    """Test that search returns a list of dicts with id, score, and payload."""
    mock_instance = mock_client.return_value
    mock_hit = MagicMock()
    mock_hit.id = 1
    mock_hit.score = 0.9
    mock_hit.payload = {"TITLE OF THE PAPER": "Title"}
    mock_instance.search.return_value = [mock_hit]
    wrapper = QdrantWrapper(AppConfig(), collection_name="test_collection")
    wrapper.client = mock_instance
    results = wrapper.search(np.array([0.1,0.2,0.3]), top_k=1)
    assert isinstance(results, list)
    assert all(k in results[0] for k in ["id", "score", "payload"])
