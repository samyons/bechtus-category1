import os
import logging
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Adjust imports based on the new structure
from data_processing.document_loader import load_pdfs_from_directory, chunk_documents
from data_processing.vector_store import get_chroma_vector_store, get_retriever, CHROMA_COLLECTION_NAME
from core.cove_pipeline import CoVeChainOrchestrator

# --- Page and Logging Configuration ---
st.set_page_config(page_title="AAOIFI CoVe Assistant", layout="wide", initial_sidebar_state="expanded")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- API Key and Resource Initialization ---
@st.cache_resource
def configure_google_api_cached():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("🔴 GOOGLE_API_KEY not found. Please set it in your .env file in the project root.")
        return False
    try:
        genai.configure(api_key=google_api_key)
        logger.info("Google Generative AI API configured successfully.")
        return True
    except Exception as e:
        st.error(f"🔴 Error configuring Google API: {e}")
        logger.critical(f"CRITICAL: Error configuring Google API: {e}", exc_info=True)
        return False

@st.cache_resource
def initialize_all_resources(force_recreate_vs: bool = False):
    """Loads docs, chunks, creates/loads vector store, and returns retriever."""
    logger.info("Attempting to initialize all resources...")
    with st.spinner("Loading AAOIFI standards and preparing knowledge base... (this may take a moment on first run)"):
        documents = load_pdfs_from_directory()
        if not documents:
            st.error("🔴 Failed to load PDF documents. Check `aaoifi_pdfs` directory and file permissions.")
            return None

        doc_chunks = chunk_documents(documents)
        if not doc_chunks:
            st.error("🔴 Failed to chunk documents.")
            return None
        
        # Determine if vector store needs creation or can be loaded
        vs_instance = None
        if force_recreate_vs:
            logger.info("Forcing vector store recreation as per request.")
            vs_instance = get_chroma_vector_store(doc_chunks=doc_chunks)
        else:
            vs_instance = get_chroma_vector_store() # Try loading
            if not vs_instance or vs_instance._collection.count() == 0 : # Check if collection is empty or not found
                logger.info("Vector store not found or empty. Recreating with current documents.")
                vs_instance = get_chroma_vector_store(doc_chunks=doc_chunks)
        
        if not vs_instance:
            st.error("🔴 Failed to initialize ChromaDB vector store.")
            return None

        retriever_instance = get_retriever(vs_instance, k_results=5) # k_results for CoVe initial context
        if not retriever_instance:
            st.error("🔴 Failed to initialize retriever from vector store.")
            return None
    
    logger.info("All data resources initialized successfully for CoVe pipeline.")
    st.success("📚 Knowledge base ready!")
    return retriever_instance

# --- Main Application Logic ---
def main_app():
    st.title("🕋 AAOIFI Standards CoVe RAG Assistant")
    st.markdown("""
        Ask questions about AAOIFI standards. This system uses a Chain-of-Verification (CoVe)
        process to enhance factual accuracy. Powered by Google Gemini and LangChain.
    """)

    if not configure_google_api_cached():
        st.stop()

    # Initialize retriever (cached)
    # Allow force recreation of vector store via a button for updates
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    
    if st.sidebar.button("🔄 Reload Knowledge Base (Force Recreate Vector Store)"):
        st.session_state.retriever = initialize_all_resources(force_recreate_vs=True)
    
    if st.session_state.retriever is None:
         st.session_state.retriever = initialize_all_resources(force_recreate_vs=False)


    if not st.session_state.retriever:
        st.error("🔴 Retriever could not be initialized. The application cannot proceed.")
        st.stop()

    # Instantiate CoVeChainOrchestrator (not cached as it contains LLM instances that might not be serializable for caching with Streamlit)
    try:
        cove_pipeline = CoVeChainOrchestrator(retriever=st.session_state.retriever)
    except ValueError as e:
        st.error(f"🔴 Error initializing CoVe Pipeline: {e}. Check API key and model availability.")
        logger.critical(f"CRITICAL: Failed to initialize CoVeChainOrchestrator: {e}", exc_info=True)
        st.stop()
    except Exception as e:
        st.error(f"🔴 An unexpected error occurred during CoVe Pipeline setup: {e}")
        logger.critical(f"CRITICAL: Unexpected error initializing CoVeChainOrchestrator: {e}", exc_info=True)
        st.stop()


    st.sidebar.markdown("---")
    st.sidebar.subheader("Example Scenario & Question:")
    example_scenario = st.sidebar.text_area(
        "Scenario Example:",
        "Alpha Islamic Bank (Lessee) has an Ijarah MBT for a generator. Term 2 yrs, ownership transfer likely. Cost $450k, tax $12k, freight $30k. Option $3k. Annual rental $300k.",
        height=150
    )
    example_question = st.sidebar.text_area(
        "Question Example:",
        "How should Alpha Bank initially recognize the ROU asset and Ijarah Liability per AAOIFI FAS? Provide calculations for ROU, Deferred Ijarah Cost, and Liability, citing FAS principles.",
        height=150
    )

    st.markdown("---")
    st.header("✍️ Your Query")
    
    scenario_input = st.text_area("Enter the Use Case Scenario:", value=example_scenario, height=150, key="scenario_user_input")
    question_input = st.text_input("Enter your specific question about the scenario:", value=example_question, key="question_user_input")

    if st.button("🔍 Process with CoVe Verification", type="primary", use_container_width=True):
        if not scenario_input or not question_input:
            st.warning("⚠️ Please provide both a scenario and a question.")
        else:
            with st.spinner("🧠 Thinking with Chain-of-Verification... This may take some time."):
                try:
                    result = cove_pipeline.invoke({"scenario": scenario_input, "question": question_input})
                    
                    st.markdown("---")
                    st.subheader("📝 CoVe Process Output")

                    with st.expander("1. Draft Answer (Initial Response)", expanded=False):
                        st.markdown(result.get('draft_answer', "N/A"))

                    st.markdown("---")
                    st.subheader("2. Verification Steps")
                    verification_log = result.get('verification_log', [])
                    if verification_log:
                        for i, item in enumerate(verification_log):
                            with st.expander(f"🔎 Verification Question {i+1}: {item.get('question', 'N/A')}", expanded=False):
                                st.caption(f"Context Source for VQ: {item.get('context_source', 'N/A')}")
                                st.markdown(f"**Answer:** {item.get('answer', 'N/A')}")
                    else:
                        st.info("No verification questions were planned or executed for this query.")
                    
                    st.markdown("---")
                    st.subheader("✅ Final Verified Answer")
                    st.markdown(result.get('final_answer', "N/A"))

                except Exception as e:
                    st.error(f"🔴 An error occurred during CoVe processing: {e}")
                    logger.error(f"Error during Streamlit CoVe run: {e}", exc_info=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("This app demonstrates CoVe for enhanced RAG. Processing takes longer due to multiple LLM calls.")

if __name__ == "__main__":
    main_app()
