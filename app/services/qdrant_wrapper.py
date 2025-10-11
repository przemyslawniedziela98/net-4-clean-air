from __future__ import annotations
from typing import List, Optional
import json
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from app.models import AppConfig
from dataclasses import dataclass
from app.logger import logger  
from app.services.prometheus import metrics


DEFAULT_DISTANCE = qmodels.Distance.COSINE

@dataclass
class QdrantConfig:
    host: str
    port: int
    api_key: Optional[str] = None
    prefer_grpc: bool = False

    @classmethod
    def from_app_config(cls, config: AppConfig) -> 'QdrantConfig':
        return cls(host=config.qdrant_host, port=config.qdrant_port, api_key=config.qdrant_api_key)


class QdrantWrapper:
    def __init__(self, app_config: AppConfig = AppConfig(), collection_name: Optional[str] = None, distance: qmodels.Distance = DEFAULT_DISTANCE) -> None:
        """Initialize Qdrant wrapper using AppConfig.

        Args:
            app_config (AppConfig): Centralized application configuration.
            collection_name (Optional[str]): Name of the Qdrant collection. Defaults to app_config.default_collection.
            distance (qmodels.Distance): Distance metric for vectors.
        """
        self.config = QdrantConfig.from_app_config(app_config)
        self.collection_name = collection_name or app_config.default_collection
        self.distance = distance
        self.client = QdrantClient(url=f'http://{self.config.host}:{self.config.port}', api_key=self.config.api_key, prefer_grpc=self.config.prefer_grpc)

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if it does not exist.

        Args:
            vector_size (int): Dimensionality of the vectors to be stored.
        """
        """Create the collection if it does not exist."""
        try:
            if not self.client.collection_exists(self.collection_name):
                logger.info(f"Creating collection '{self.collection_name}' with vector size {vector_size}")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(size=vector_size, distance=self.distance)
                )
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception as e:
            logger.exception(f"Error ensuring collection '{self.collection_name}': {e}")
            raise

    def _normalize_ids(self, df: pd.DataFrame, id_column: str = 'id') -> pd.Series:
        """Normalize IDs to integer or string suitable for Qdrant. """
        ids_fixed = []
        for idx, pid in enumerate(df[id_column]):
            if isinstance(pid, float):
                ids_fixed.append(int(pid))
            elif isinstance(pid, int):
                ids_fixed.append(pid)
            elif isinstance(pid, str):
                try:
                    ids_fixed.append(int(float(pid)))
                except:
                    ids_fixed.append(pid)
            else:
                ids_fixed.append(idx)
        return pd.Series(ids_fixed, name='id_fixed')

    def _prepare_payload(self, row: pd.Series) -> dict:
        """Prepare the payload dictionary for a single DataFrame row.  """
        payload = row.drop(labels=['document', 'id_fixed']).to_dict()
        for k, v in payload.items():
            if isinstance(v, (np.generic,)):
                payload[k] = v.item()
            try:
                json.dumps(payload[k])
            except Exception:
                payload[k] = str(payload[k])
        return payload

    def _create_points(self, df: pd.DataFrame, embeddings: np.ndarray) -> List[qmodels.PointStruct]:
        """Convert a DataFrame and embeddings into Qdrant PointStruct objects.  """
        points = []
        for idx, row in df.iterrows():
            point_id = row['id_fixed']
            payload = self._prepare_payload(row)
            points.append(qmodels.PointStruct(id=point_id, vector=embeddings[idx].tolist(), payload=payload))
        return points

    def upsert_dataframe(self, df: pd.DataFrame, embeddings: np.ndarray, id_column: str = 'id') -> None:
        """Upsert a DataFrame into the Qdrant collection in batches.

        Args:
            df (pd.DataFrame): DataFrame with data and documents.
            embeddings (np.ndarray): Embeddings corresponding to the document column.
            id_column (str): Column name to use as unique IDs.
        """
        if df.shape[0] != embeddings.shape[0]:
            raise ValueError('Number of embeddings must match number of rows in df')
        
        logger.info(f"Upserting {df.shape[0]} rows into collection '{self.collection_name}'")
        df['id_fixed'] = self._normalize_ids(df, id_column=id_column)
        points = self._create_points(df, embeddings)
        batch_size = 128

        with metrics.QDRANT_UPSERT_LATENCY.time():
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                    logger.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} points)")
                    metrics.QDRANT_UPSERT_COUNTER.inc(len(batch))
                except Exception as e:
                    logger.exception(f"Error during upsert of batch starting at index {i}: {e}")
                    raise

    def search(self, query_embedding: np.ndarray, top_k: int = 5, with_payload: bool = True) -> List[dict]:
        """Search the Qdrant collection using a query embedding.

        Args:
            query_embedding (np.ndarray): Embedding vector for the query.
            top_k (int): Maximum number of results to return.
            with_payload (bool): Include payload data in search results.

        Returns:
            List[dict]: List of search results with 'id', 'score', and 'payload'.
        """
        try:
            logger.info(f"Searching collection '{self.collection_name}' with top_k={top_k}")
            metrics.QDRANT_SEARCH_COUNTER.inc()  
            with metrics.QDRANT_SEARCH_LATENCY.time(): 
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=top_k,
                    with_payload=with_payload
                )
                results = [{'id': h.id, 'score': h.score, 'payload': h.payload} for h in hits]
                logger.info(f"Search returned {len(results)} results")
                return results
        except Exception as e:
            logger.exception(f"Search failed: {e}")
            raise
