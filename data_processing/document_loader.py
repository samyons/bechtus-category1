# aaoifi_cove_streamlit_app/data_processing/document_loader.py
import os
import logging
from typing import List, Optional
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

def load_pdfs_from_directory(pdf_directory: str = "aaoifi_pdfs") -> Optional[List[Document]]:
    """Loads all PDF documents from a specified directory."""
    if not os.path.exists(pdf_directory) or not os.listdir(pdf_directory):
        # Try relative path if absolute fails (e.g., when run from Streamlit context)
        pdf_directory_rel = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", pdf_directory)
        if not os.path.exists(pdf_directory_rel) or not os.listdir(pdf_directory_rel):
             logger.error(f"Directory '{pdf_directory}' (and relative '{pdf_directory_rel}') not found or is empty.")
             return None
        else:
            pdf_directory = pdf_directory_rel


    logger.info(f"Attempting to load PDFs from: {os.path.abspath(pdf_directory)}")
    try:
        loader = PyPDFDirectoryLoader(pdf_directory)
        documents = loader.load()
        if not documents:
            logger.warning(f"No documents loaded from '{pdf_directory}'. Check PDF content and permissions.")
            return None
        logger.info(f"Loaded {len(documents)} PDF document(s) from '{pdf_directory}'.")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDFs from '{pdf_directory}': {e}", exc_info=True)
        return None

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 40
) -> Optional[List[Document]]:
    """Splits a list of documents into smaller chunks."""
    if not documents:
        logger.warning("No documents provided for chunking.")
        return None
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        doc_chunks = text_splitter.split_documents(documents)
        if not doc_chunks:
            logger.warning("No text chunks were created from the documents.")
            return None
        logger.info(f"Split {len(documents)} document(s) into {len(doc_chunks)} chunks.")
        return doc_chunks
    except Exception as e:
        logger.error(f"Error chunking documents: {e}", exc_info=True)
        return None