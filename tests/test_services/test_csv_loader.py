import io
import pandas as pd
import pytest
from app.services.csv_loader import CSVLoader


@pytest.fixture
def simple_csv_bytes():
    """Fixture providing a simple CSV with an 'Id' column."""
    content = """Id,Title of the paper,Main findings of the paper
1,Paper A,Interesting results
2,Paper B,More findings
"""
    return content.encode("utf-8")


@pytest.fixture
def csv_without_id():
    """Fixture providing a CSV without an 'Id' column to test auto-indexing."""
    content = """Title of the paper,Main findings of the paper
Paper C,Data is cool
Paper D,Even more data
"""
    return content.encode("utf-8")


def test_load_with_id_column(simple_csv_bytes):
    """Test that CSVs with an 'Id' column are correctly loaded and normalized."""
    loader = CSVLoader(simple_csv_bytes)
    df = loader.load()

    assert isinstance(df, pd.DataFrame)
    assert "id" in df.columns
    assert "document" in df.columns
    assert list(df["id"]) == [1, 2]
    assert "Paper A" in df.loc[0, "document"]
    assert "Interesting results" in df.loc[0, "document"]
    assert "\n" in df.loc[0, "document"]


def test_load_without_id_column(csv_without_id):
    """Test that CSVs without an 'Id' column receive auto-generated integer IDs."""
    loader = CSVLoader(csv_without_id)
    df = loader.load()

    assert list(df["id"]) == [0, 1]
    assert all(isinstance(i, (int, float)) for i in df["id"])
    assert "Paper C" in df.loc[0, "document"]
    assert "Data is cool" in df.loc[0, "document"]


def test_normalize_id_handles_strings():
    """Test that the internal _normalize_id method handles various input types correctly."""
    assert CSVLoader._normalize_id("10") == 10
    assert CSVLoader._normalize_id("10.0") == 10
    assert CSVLoader._normalize_id("abc") == "abc"
    assert CSVLoader._normalize_id(None) is None


def test_load_with_unexpected_columns():
    """Test that CSVs with unknown column names are still processed into valid documents."""
    content = """foo,bar,baz
x,y,z
a,b,c
"""
    df = CSVLoader(content.encode()).load()

    assert "id" in df.columns
    assert "document" in df.columns
    assert all(isinstance(x, str) for x in df["document"])
    assert len(df) == 2
