from flask import Blueprint, render_template, request, redirect, flash
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import os
from typing import List, Dict, Any

from app.services.csv_loader import CSVLoader
from app.services.embedder import Embedder
from app.services.qdrant_wrapper import QdrantWrapper
from app.models import AppConfig
from app.logger import logger  

routes = Blueprint('routes', __name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

config = AppConfig()
embedder = Embedder(model_name=config.embed_model)
qwrap: QdrantWrapper = None


@routes.route('/', methods=['GET', 'POST'])
def index() -> str:
    """Render the home page and handle CSV file upload for indexing.

    Returns:
        str: Rendered HTML page.
    """
    if request.method == 'POST':
        logger.info("Received file upload request")

        if 'csv_file' not in request.files:
            logger.warning("No file part in the request")
            flash('No file part', 'danger')
            return redirect(request.url)

        file: FileStorage = request.files['csv_file']
        if file.filename == '':
            logger.warning("Empty filename submitted")
            flash('No selected file', 'warning')
            return redirect(request.url)

        if file:
            filename: str = secure_filename(file.filename)
            filepath: str = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            logger.info(f"File saved to {filepath}")

            try:
                with open(filepath, 'rb') as f:
                    loader = CSVLoader(f.read())
                    df = loader.load()
                logger.info(f"Loaded CSV with {len(df)} rows")

                embs: Any = embedder.embed(df['document'].tolist())
                logger.debug(f"Generated embeddings shape: {embs.shape}")

                global qwrap
                qwrap = QdrantWrapper(collection_name=config.default_collection)
                qwrap.ensure_collection(vector_size=embs.shape[1])
                qwrap.upsert_dataframe(df, embs)

                flash(f'Successfully indexed {len(df)} rows', 'success')
                logger.info(f"Successfully indexed {len(df)} rows into Qdrant")

            except Exception as e:
                logger.exception(f"Error processing uploaded file {filename}: {e}")
                flash(f'Error processing file: {e}', 'danger')

    return render_template('index.html')


@routes.route('/search', methods=['GET', 'POST'])
def search() -> str:
    """Render the search page and handle query submissions.

    Returns:
        str: Rendered HTML page with search results.
    """
    results: List[Dict[str, Any]] = []
    top_k: int = 5

    if request.method == 'POST':
        query_text: str = request.form.get('query', '')
        top_k_str: str = request.form.get('top_k', '5')

        try:
            top_k = max(1, int(top_k_str))
        except ValueError:
            logger.warning(f"Invalid top_k value received: {top_k_str}. Falling back to 5.")
            top_k = 5

        if not query_text:
            logger.warning("Empty query submitted")
            flash('Please enter a query', 'warning')
        elif qwrap is None:
            logger.warning("Search attempted before collection was indexed")
            flash('No collection indexed yet', 'danger')
        else:
            logger.info(f"Search initiated: query='{query_text}', top_k={top_k}")
            try:
                q_emb: Any = embedder.embed([query_text])[0]
                results = qwrap.search(q_emb, top_k=top_k)
                logger.info(f"Search returned {len(results)} results")
                logger.debug(f"Search results: {results}")
            except Exception as e:
                logger.exception(f"Search error for query '{query_text}': {e}")
                flash(f'Error during search: {e}', 'danger')

    return render_template('search.html', results=results, top_k=top_k)
