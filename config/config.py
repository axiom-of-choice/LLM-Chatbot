from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = Path(BASE_DIR, 'data/')
EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT", "us-west4-gcp-free")
OPENAI_API_KEY = os.environ.get("OPEN_AI_KEY")
GLOB = os.environ.get("GLOB", None)

if __name__ == '__main__':
    print(BASE_DIR)
    print(DATA_DIR)
    