from dataclasses import dataclass
from typing import Optional
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

@dataclass
class AppConfig:
    qdrant_host: str = config.get('QDRANT', 'host', fallback='localhost')
    qdrant_port: int = config.getint('QDRANT', 'port', fallback=6333)
    qdrant_api_key: Optional[str] = config.get('QDRANT', 'api_key', fallback=None)
    embed_model: str = config.get('EMBEDDING', 'model_name', fallback='all-MiniLM-L6-v2')
    default_collection: str = config.get('EMBEDDING', 'default_collection', fallback='papers_poc')
    flask_secret_key: str = config.get('FLASK', 'flask_secret_key', fallback='test')
    grafana_url: str = config.get('FLASK', 'grafana_url', fallback='http://localhost:3000/dashboards')

