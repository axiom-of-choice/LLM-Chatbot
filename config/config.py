from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
import logging
import sys
from langchain.chat_models import ChatOpenAI
from langchain.llms import Cohere, OpenAIChat
from langchain.embeddings import CohereEmbeddings, HuggingFaceEmbeddings

import toml

## TO-DO replace from langchain.chat_models import ChatOpenAI

# Dirs
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = Path(BASE_DIR, "data/")
STATIC_DIR = Path(BASE_DIR, "static/")
# BACKGROUNDS_DIR = Path(STATIC_DIR, "background.png")
# LOGO_DIR = Path(STATIC_DIR, "logo.png")
LOGS_DIR = Path(BASE_DIR, "logs")
TOML_DIR = os.path.join(BASE_DIR, "client_config.toml")

# MODELS AND STORAGES
HF_EMBEDDING_MODEL_NAME = os.environ.get("HF_EMBEDDING_MODEL_NAME")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT", "us-west4-gcp-free")
OPENAI_API_KEY = os.environ.get("OPEN_AI_KEY")
GLOB = os.environ.get("GLOB", None)
BUCKET_NAME = os.environ.get("BUCKET_NAME")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
COHERE_MODEL_NAME = os.environ.get("COHERE_EMBEDDING_MODEL_NAME")
COHERE_EMBEDDING_MODEL_NAME = os.environ.get("COHERE_EMBEDDING_MODEL_NAME")

# MODEL CATALOG
AVAILABLE_LLMS = {
    "GPT 3.5 turbo": ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.0),
    "Cohere LLM": Cohere(cohere_api_key=COHERE_API_KEY, temperature=0.0, truncate="START"),
}

AVAILABLE_EMBEDDINGS = {
    "Cohere": CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model=COHERE_EMBEDDING_MODEL_NAME),
    "stsb-xlm-r-multilingual": HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL_NAME),
}

client_config = toml.load(TOML_DIR)
TITLE = client_config["branding"]["title"]
BACKGROUNDS_DIR = client_config["branding"]["background_image_url"]
LOGO_DIR = client_config["branding"]["logo_url"]
CLIENT_DATASOURCE = client_config["available_datasources"]["client_datasource"]
CLIENT_DATASOURCE_URI = client_config["available_datasources"]["client_datasource_uri"]

if __name__ == "__main__":
    print(client_config)
