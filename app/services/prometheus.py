from prometheus_client import REGISTRY
from prometheus_client import Counter, Histogram, Gauge


class metrics:
    """Central Prometheus metrics registry for the application."""
    _registry = REGISTRY

    QDRANT_UPSERT_COUNTER = Counter(
        'qdrant_upsert_total', 
        'Total number of points upserted into Qdrant', 
        registry=_registry
    )
    QDRANT_SEARCH_COUNTER = Counter(
        'qdrant_search_total', 
        'Total number of search queries to Qdrant', 
        registry=_registry
    )
    QDRANT_SEARCH_LATENCY = Histogram(
        'qdrant_search_latency_seconds', 
        'Latency of Qdrant search queries in seconds', 
        registry=_registry
    )
    QDRANT_UPSERT_LATENCY = Histogram(
        'qdrant_upsert_latency_seconds', 
        'Latency of upsert operations in seconds', 
        registry=_registry
    )
    QDRANT_COLLECTION_SIZE = Gauge(
        'qdrant_collection_size', 
        'Number of points in Qdrant collection', 
        ['collection'], 
        registry=_registry
    )

    EMBEDDING_REQUESTS = Counter(
        'embedder_requests_total', 
        'Total number of embedding requests', 
        ['model_name'], 
        registry=_registry
    )

    EMBEDDING_ERRORS = Counter(
        'embedder_errors_total', 
        'Total number of embedding errors', 
        ['model_name'], 
        registry=_registry
    )

    EMBEDDING_DURATION = Histogram(
        'embedder_duration_seconds', 
        'Time spent generating embeddings', 
        ['model_name'], 
        registry=_registry
    )

    CURRENT_EMBEDDING_LOAD = Gauge(
        'embedder_in_progress', 
        'Number of embedding operations currently running', 
        ['model_name'], 
        registry=_registry
    )

    CHAT_REQUESTS = Counter(
        "chatservice_requests_total", 
        "Total number of chat questions received", 
        ["model"], 
        registry=_registry
    )

    CHAT_ERRORS = Counter(
        "chatservice_errors_total", 
        "Total number of errors during question answering", 
        ["model", "stage"], 
        registry=_registry
    )

    CHAT_LATENCY = Histogram(
        "chatservice_request_duration_seconds", 
        "Time taken to process a chat question end-to-end", 
        ["model"], 
        registry=_registry
    )

    OPENAI_LATENCY = Histogram(
        "chatservice_openai_latency_seconds", 
        "Latency of OpenAI API call", 
        ["model"], 
        registry=_registry
    )