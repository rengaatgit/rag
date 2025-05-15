import os
import logging
# Import specific readers from llama_index.readers.file
from llama_index.readers.file import PyMuPDFReader # Use PyMuPDFReader for PDFs

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.databricks import DatabricksEmbedding
from llama_index.llms.databricks import DatabricksLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# New input file for PDF paths
PDF_PATHS_FILE = "pdf_paths.txt"
INDEX_STORAGE_DIR = "./vector_index_pdfs" # Use a different directory to avoid conflicts with web index
# New file to track processed PDF paths
PROCESSED_PDF_PATHS_FILE = "./processed_pdf_paths.txt"

# Databricks Model Names (Same as before)
EMBEDDING_MODEL_NAME = "txt-embeddings-3-large"
LLM_MODEL_NAME = "gpt-4o"

# Llama-Index Settings (Tune these based on your content and desired chunking)
# These are crucial parameters you can adjust.
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200 # Overlap helps maintain context between chunks

# --- Helper Functions ---

def load_processed_paths(file_path: str) -> set[str]:
    """Loads the set of file paths that have already been processed."""
    processed_paths = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                processed_paths.add(line.strip())
    return processed_paths

def save_processed_paths(file_path: str, paths: set[str]):
    """Saves the set of processed file paths to a file."""
    with open(file_path, 'w') as f:
            for path in paths:
                f.write(path + '\n')

# --- Main Processing Logic ---

def build_or_update_index_from_pdfs():
    """
    Reads PDF paths, loads/creates index, processes new PDFs, and saves the index.
    """
    processed_paths = load_processed_paths(PROCESSED_PDF_PATHS_FILE)
    all_paths = set()
    if os.path.exists(PDF_PATHS_FILE):
        with open(PDF_PATHS_FILE, 'r') as f:
            for line in f:
                pdf_path = line.strip()
                if pdf_path and os.path.exists(pdf_path): # Check if path is valid and file exists
                    all_paths.add(pdf_path)
                elif pdf_path:
                     logger.warning(f"Warning: PDF file not found at path: {pdf_path}")
    else:
        logger.error(f"Error: {PDF_PATHS_FILE} not found. Create this file with PDF paths.")
        return

    new_paths_to_process = list(all_paths - processed_paths)

    if not new_paths_to_process:
        logger.info("No new PDF paths to process.")
        # Load index even if no new paths, so it's ready for querying
        if os.path.exists(INDEX_STORAGE_DIR):
             logger.info(f"Loading existing index from {INDEX_STORAGE_DIR}.")
             return load_index_from_storage(StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR))
        else:
             logger.warning("No new PDF paths and no existing index found.")
             return None


    logger.info(f"Found {len(new_paths_to_process)} new PDF paths to process.")

    # --- Initialize Databricks Models and Llama-Index Settings ---
    # Ensure DATABRICKS_HOST and DATABRICKS_TOKEN environment variables are set
    try:
        db_embedding = DatabricksEmbedding(model=EMBEDDING_MODEL_NAME)
        db_llm = DatabricksLLM(model=LLM_MODEL_NAME)

        # Configure global settings for Llama-Index
        Settings.embed_model = db_embedding
        Settings.llm = db_llm
        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP

    except Exception as e:
        logger.error(f"Failed to initialize Databricks models. Ensure DATABRICKS_HOST and DATABRICKS_TOKEN are set. Error: {e}")
        return


    # --- Load Documents from PDFs ---
    logger.info("Loading documents from new PDF paths...")
    documents = []
    reader = PyMuPDFReader() # Initialize the PDF reader

    # Note: PyMuPDFReader attempts OCR if pytesseract is installed and configured.
    # For best results with scanned PDFs, ensure Tesseract is properly set up on your system.
    # You might need to set the TESSDATA_PREFIX environment variable if Tesseract data files
    # are not in a standard location.

    for pdf_path in new_paths_to_process:
        try:
            logger.info(f"Loading {pdf_path}...")
            # PyMuPDFReader.load_data expects a file path
            docs = reader.load_data(file_path=pdf_path)
            documents.extend(docs)
            logger.info(f"Successfully loaded {pdf_path}.")
        except Exception as e:
            logger.error(f"Failed to load {pdf_path}. Error: {e}")

    if not documents:
        logger.warning("No documents were successfully loaded from new PDF paths.")
        # Load existing index if it exists, even if no new docs were loaded
        if os.path.exists(INDEX_STORAGE_DIR):
             logger.info("Loading existing index as no new documents were loaded.")
             return load_index_from_storage(StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR))
        else:
             logger.warning("No documents loaded and no existing index found.")
             return None


    # --- Build or Update Index ---
    index = None
    if os.path.exists(INDEX_STORAGE_DIR):
        logger.info(f"Loading index from {INDEX_STORAGE_DIR}...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
            index = load_index_from_storage(storage_context)
            logger.info("Index loaded successfully. Adding new documents...")
            # Add new documents to the existing index
            for doc in documents:
                index.insert(doc)
            logger.info("New documents added to the index.")
        except Exception as e:
            logger.error(f"Failed to load existing index or add documents. Creating a new index. Error: {e}")
            index = VectorStoreIndex.from_documents(documents)
            logger.info("Created a new index from new documents.")
    else:
        logger.info("No existing index found. Creating a new index...")
        index = VectorStoreIndex.from_documents(documents)
        logger.info("New index created.")

    # --- Save Index ---
    if index:
        logger.info(f"Saving index to {INDEX_STORAGE_DIR}...")
        index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
        logger.info("Index saved successfully.")

        # --- Update Processed Paths List ---
        updated_processed_paths = processed_paths.union(set(new_paths_to_process))
        save_processed_paths(PROCESSED_PDF_PATHS_FILE, updated_processed_paths)
        logger.info(f"Updated processed paths list with {len(new_paths_to_process)} new paths.")

    return index

# --- Querying Logic ---

def query_index(index):
    """Sets up the query engine and runs sample queries."""
    if index is None:
        logger.warning("Index is not available for querying.")
        return

    logger.info("Setting up query engine...")
    # The LLM set in Settings will be used by the query engine
    query_engine = index.as_query_engine()
    logger.info("Query engine ready.")

    # --- Sample Queries ---
    print("\n--- Running Sample Queries ---")

    queries = [
        "What is the main topic of the documents?",
        "Summarize the key information from the PDFs.",
        "Extract any dates mentioned in the documents."
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        try:
            response = query_engine.query(query)
            print(f"Response: {response}")
        except Exception as e:
            logger.error(f"Error during query '{query}': {e}")

# --- Main Execution ---

if __name__ == "__main__":
    logger.info("Starting RAG pipeline setup for PDFs...")
    vector_index = build_or_update_index_from_pdfs()
    if vector_index:
        query_index(vector_index)
    else:
        logger.error("Index could not be built or loaded. Cannot proceed with querying.")
    logger.info("Pipeline execution finished.")
