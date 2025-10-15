import io
import pytest
from unittest.mock import patch, MagicMock
from app.routes import routes

@pytest.fixture
def client():
    from flask import Flask
    app = Flask(__name__)
    app.secret_key = "test_secret"
    app.register_blueprint(routes)
    app.testing = True
    return app.test_client()

@patch('app.routes.render_template', return_value="INDEX_HTML")
def test_index_get(mock_render, client):
    """GET request returns index page"""
    response = client.get('/')
    assert response.data == b"INDEX_HTML"
    mock_render.assert_called_with('index.html')

@patch('app.routes.render_template', return_value="INDEX_HTML")
@patch('app.routes.CSVLoader')
@patch('app.routes.Embedder.embed')
@patch('app.routes.QdrantWrapper')
def test_index_post_success(mock_qdrant, mock_embed, mock_csvloader, mock_render, client):
    """POST request with valid CSV file should flash success"""
    mock_df = MagicMock()
    mock_df.__getitem__.return_value.tolist.return_value = ["doc1", "doc2"]
    mock_csvloader.return_value.load.return_value = mock_df
    mock_embed.return_value.shape = (2, 768)

    file_data = io.BytesIO(b"col1,col2\ndoc1,text1\ndoc2,text2")
    data = {'csv_file': (file_data, 'test.csv')}

    response = client.post('/', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert b"INDEX_HTML" in response.data
    mock_qdrant.return_value.ensure_collection.assert_called()
    mock_qdrant.return_value.upsert_dataframe.assert_called()


@patch('app.routes.render_template', return_value="INDEX_HTML")
def test_index_post_no_file(mock_render, client):
    """POST request without file should flash an error"""
    response = client.post('/', data={}, follow_redirects=True)
    assert b"INDEX_HTML" in response.data

@patch('app.routes.render_template', return_value="INDEX_HTML")
def test_index_post_empty_filename(mock_render, client):
    """POST request with empty filename should flash warning"""
    file_data = io.BytesIO(b"some,data")
    data = {'csv_file': (file_data, '')}
    response = client.post('/', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert b"INDEX_HTML" in response.data

@patch('app.routes.render_template', return_value="SEARCH_HTML")
@patch('app.routes.qwrap', new_callable=MagicMock)
@patch('app.routes.Embedder.embed')
def test_search_post_success(mock_embed, mock_qwrap, mock_render, client):
    """POST search with valid query returns results"""
    mock_qwrap.search.return_value = [{'document': 'doc1'}]
    mock_embed.return_value = [[0.1]*768]

    data = {'query': 'test query', 'top_k': '3'}
    response = client.post('/search', data=data)
    assert b"SEARCH_HTML" in response.data
    mock_qwrap.search.assert_called()

@patch('app.routes.render_template', return_value="SEARCH_HTML")
@patch('app.routes.qwrap', None)
def test_search_post_no_qwrap(mock_render, client):
    """POST search with no indexed collection should flash danger"""
    data = {'query': 'test query', 'top_k': '3'}
    response = client.post('/search', data=data)
    assert b"SEARCH_HTML" in response.data

@patch('app.routes.render_template', return_value="SEARCH_HTML")
def test_search_post_empty_query(mock_render, client):
    """Empty search query should flash warning"""
    response = client.post('/search', data={'query': ''}, follow_redirects=True)
    assert b"SEARCH_HTML" in response.data

@patch('app.routes.render_template', return_value="CHAT_HTML")
@patch('app.routes.qwrap', new_callable=MagicMock)
@patch('app.routes.ChatService')
def test_chat_post_success(mock_chat_service, mock_qwrap, mock_render, client):
    """POST chat with valid question returns answer"""
    mock_qwrap.search.return_value = []
    mock_chat_service.return_value.answer_question.return_value = {
        "answer": "This is the answer",
        "context_docs": ["doc1"]
    }

    data = {'question': 'What is this?', 'top_k': '3'}
    response = client.post('/chat', data=data)
    assert b"CHAT_HTML" in response.data
    mock_chat_service.return_value.answer_question.assert_called()


@patch('app.routes.render_template', return_value="CHAT_HTML")
def test_chat_post_empty_question(mock_render, client):
    """POST chat with empty question should flash warning"""
    response = client.post('/chat', data={'question': ''}, follow_redirects=True)
    assert b"CHAT_HTML" in response.data

@patch('app.routes.render_template', return_value="CHAT_HTML")
@patch('app.routes.qwrap', None)
def test_chat_post_no_qwrap(mock_render, client):
    """POST chat without indexed collection should flash danger"""
    response = client.post('/chat', data={'question': 'Any question'})
    assert b"CHAT_HTML" in response.data

def test_metrics_route(client):
    """GET /metrics returns Prometheus metrics"""
    response = client.get('/metrics')
    assert response.status_code == 200
    assert b"# HELP" in response.data  

@patch('app.routes.render_template', return_value="DASHBOARD_HTML")
def test_metrics_dashboard(mock_render, client):
    """GET /metrics_dashboard renders dashboard page"""
    response = client.get('/metrics_dashboard')
    assert b"DASHBOARD_HTML" in response.data
