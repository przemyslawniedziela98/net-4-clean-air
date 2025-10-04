from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from app.logger import logger

class Embedder:
    """Wrapper for computing text embeddings using SentenceTransformer.

    Attributes:
        model_name: Name of the pre-trained embedding model.
        _model: Internal SentenceTransformer model instance.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2') -> None:
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def _ensure_model(self) -> None:
        """Load the model if it hasn't been loaded yet."""
        if self._model is None:
            try:
                logger.info(f"Loading embedding model '{self.model_name}'...")
                self._model = SentenceTransformer(self.model_name)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.exception(f"Failed to load model '{self.model_name}': {e}")
                raise

    def embed(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        """Compute embeddings for a list of texts.

        Args:
            texts: Iterable of strings to embed.
            batch_size: Number of texts per batch.

        Returns:
            np.ndarray: Array of shape (n_texts, embedding_dim) with float32 values.
        """
        self._ensure_model()
        texts_list = list(texts)
        logger.info(f"Embedding {len(texts_list)} texts (batch_size={batch_size})")
        try:
            embeddings = self._model.encode(
                texts_list,
                batch_size=batch_size,
                show_progress_bar=False
            )
            logger.info(f"Generated embeddings with shape {embeddings.shape}")
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as e:
            logger.exception(f"Error during embedding: {e}")
            raise
