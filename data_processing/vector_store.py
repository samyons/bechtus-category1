# aaoifi_cove_streamlit_app/data_processing/vector_store.py
import logging
import os
import chromadb
from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever

logger = logging.getLogger(__name__)

# Define ChromaDB path relative to this file's parent directory
# This makes it work correctly when app.py (in parent) imports it.
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIRECTORY = os.path.join(APP_ROOT, "chroma_db_store")

CHROMA_COLLECTION_NAME = "aaoifi_standards_collection"
EMBEDDING_MODEL_NAME = "models/embedding-001"

def get_embedding_function() -> Optional[GoogleGenerativeAIEmbeddings]:
    """Initializes and returns the Google Generative AI embedding function."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, task_type="retrieval_document")
        logger.debug(f"Using embedding model: {EMBEDDING_MODEL_NAME}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embedding function: {e}", exc_info=True)
        return None

def get_chroma_vector_store(
    doc_chunks: Optional[List[Document]] = None,
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
    collection_name: str = CHROMA_COLLECTION_NAME
) -> Optional[Chroma]:
    """
    Creates a new Chroma vector store from document chunks or loads an existing one.
    """
    embedding_function = get_embedding_function()
    if not embedding_function:
        logger.error("Embedding function not available for vector store.")
        return None

    logger.info(f"ChromaDB persist directory: {os.path.abspath(persist_directory)}")
    os.makedirs(persist_directory, exist_ok=True) # Ensure directory exists

    # Initialize Chroma client for persistent storage
    client = chromadb.PersistentClient(path=persist_directory)

    if doc_chunks:
        logger.info(f"Attempting to create/update Chroma vector store collection '{collection_name}'.")
        try:
            vector_store = Chroma.from_documents(
                documents=doc_chunks,
                embedding=embedding_function,
                collection_name=collection_name,
                client=client,
                # persist_directory=persist_directory # Not needed if client is passed correctly
            )
            logger.info(f"Successfully created/updated Chroma vector store for collection '{collection_name}'.")
            return vector_store
        except Exception as e:
            logger.error(f"Error creating Chroma vector store: {e}", exc_info=True)
            return None
    else:
        logger.info(f"Attempting to load existing Chroma vector store collection '{collection_name}'.")
        try:
            # Check if collection exists first
            client.get_collection(name=collection_name) # Throws exception if not found
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_function,
                client=client,
                # persist_directory=persist_directory
            )
            if vector_store._collection.count() == 0:
                 logger.warning(f"Loaded collection '{collection_name}' is empty. It might need to be populated.")
            else:
                logger.info(f"Successfully loaded existing Chroma vector store with {vector_store._collection.count()} items for collection '{collection_name}'.")
            return vector_store
        except Exception: # Catches collection not found and other errors
            logger.warning(f"Collection '{collection_name}' not found or error loading. A new one will be created if chunks are provided.")
            return None


def get_retriever(
    vector_store: Chroma,
    k_results: int = 5,
    search_type: str = "similarity"
) -> Optional[VectorStoreRetriever]:
    """Creates a retriever from the Chroma vector store."""
    if not vector_store:
        logger.error("Cannot create retriever: Vector store is not available.")
        return None
    try:
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k_results}
        )
        logger.info(f"Retriever created: k={k_results}, type='{search_type}'.")
        return retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {e}", exc_info=True)
        return None