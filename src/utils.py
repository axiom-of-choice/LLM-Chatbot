import pinecone
import logging
from config.config import *

logger = logging.getLogger(__name__)


def connect_index(index_name: str, API_KEY: str = PINECONE_API_KEY, ENV: str = PINECONE_ENV) -> pinecone.Index:
    pinecone.init(api_key=API_KEY, environment=ENV)
    index = pinecone.Index(index_name)
    logger.info(f"Connected to Pinecone index {index_name}")
    return index


def init_pinecone_index(api_key: str = PINECONE_API_KEY, environment: str = PINECONE_ENV) -> None:
    pinecone.init(api_key=api_key, environment=environment)
    logger.info("Pinecone initialized")
    return None


if __name__ == "__main__":
    connect_index("stab-test")
