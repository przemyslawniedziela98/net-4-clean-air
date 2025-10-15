import pytest
from unittest.mock import MagicMock, patch
from app.services.chat import ChatService


@pytest.fixture
def mock_embedder():
    """Fixture returning a mocked Embedder that produces deterministic embeddings."""
    mock = MagicMock()
    mock.embed.return_value = [[0.1, 0.2, 0.3]]
    return mock


@pytest.fixture
def mock_qdrant():
    """Fixture returning a mocked QdrantWrapper that simulates a vector search."""
    mock = MagicMock()
    mock.search.return_value = [
        {"id": 1, "score": 0.9, "payload": {
            "TITLE OF THE PAPER": "Paper A",
            "AIM OF THE PAPER": "Study indoor air quality",
            "MAIN FINDINGS OF THE PAPER": "Ventilation improves air quality",
        }},
        {"id": 2, "score": 0.8, "payload": {
            "TITLE OF THE PAPER": "Paper B",
            "AIM OF THE PAPER": "Analyze CO2 levels",
            "MAIN FINDINGS OF THE PAPER": "High CO2 reduces performance",
        }},
    ]
    return mock


@pytest.fixture
def chat_service(mock_embedder, mock_qdrant):
    """Fixture creating a ChatService instance with mocked dependencies."""
    return ChatService(qdrant=mock_qdrant, embedder=mock_embedder, model="gpt-3.5-turbo")


def test_embed_query(chat_service, mock_embedder):
    """Test that _embed_query calls the embedder and returns the correct vector."""
    result = chat_service._embed_query("What is air quality?")
    assert result == [0.1, 0.2, 0.3]
    mock_embedder.embed.assert_called_once_with(["What is air quality?"])


def test_search_qdrant(chat_service, mock_qdrant):
    """Test that _search_qdrant returns valid Qdrant results."""
    results = chat_service._search_qdrant([0.1, 0.2, 0.3], top_k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]["payload"]["TITLE OF THE PAPER"] == "Paper A"


def test_build_context(chat_service, mock_qdrant):
    """Test that _build_context produces readable text from Qdrant payloads."""
    context = chat_service._build_context(mock_qdrant.search.return_value)
    assert "Paper A" in context
    assert "Ventilation improves air quality" in context
    assert "CO2" in context


def test_build_prompt(chat_service, mock_qdrant):
    """Test that _build_prompt correctly embeds context and the question."""
    results = mock_qdrant.search.return_value
    prompt = chat_service._build_prompt("What are findings?", results)
    assert "Answer the following question" in prompt
    assert "Question: What are findings?" in prompt
    assert "Paper A" in prompt


@patch("app.services.chat.openai.ChatCompletion.create")
def test_generate_answer(mock_openai, chat_service):
    """Test that _generate_answer calls OpenAI and returns its response text."""
    mock_openai.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="The answer is here."))]
    )

    result = chat_service._generate_answer("Prompt text", max_tokens=50)
    assert result == "The answer is here."
    mock_openai.assert_called_once()


@patch("app.services.chat.openai.ChatCompletion.create", side_effect=Exception("API failure"))
def test_generate_answer_handles_error(mock_openai, chat_service):
    """Test that _generate_answer handles OpenAI API errors gracefully."""
    result = chat_service._generate_answer("Prompt text", max_tokens=50)
    assert "Error:" in result

@patch.object(ChatService, "_generate_answer", return_value="Indoor air quality improves with ventilation.")
def test_answer_question_full_flow(mock_gen_answer, chat_service):
    """Test that answer_question orchestrates embedding, search, prompt building, and answer generation."""
    result = chat_service.answer_question("How to improve indoor air?")
    assert "answer" in result
    assert "context_docs" in result
    assert isinstance(result["context_docs"], list)
    assert result["answer"].startswith("Indoor air quality")
