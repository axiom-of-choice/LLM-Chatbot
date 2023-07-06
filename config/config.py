from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = Path(BASE_DIR, 'data')
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_INDEX_NAME = "test-docs"

if __name__ == '__main__':
    print(BASE_DIR)
    print(DATA_DIR)
    