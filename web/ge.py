https://www.example.com/page1
https://www.anothersite.org/article-about-rag
https://docs.llamaindex.ai/en/stable/


import os
import logging
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.readers.web import WebPageReader # In Llama-Index 0.12.x, WebPageReader is often in llama_index.readers.web
from llama_index.embeddings.databricks import DatabricksEmbedding
from llama_index.llms.databricks import DatabricksLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
URLS_FILE = "urls.txt"
INDEX_STORAGE_DIR = "./vector_index"
PROCESSED_URLS_FILE = "./processed_urls.txt"

# Databricks Model Names
EMBEDDING_MODEL_NAME = "txt-embeddings-3-large"
LLM_MODEL_NAME = "gpt-4o"

# Llama-Index Settings (Tune these based on your content and desired chunking)
# While Llama-Index doesn't automatically 'derive' the absolute best,
# these are crucial parameters you can adjust.
# Reasonable starting points:
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200 # Overlap helps maintain context between chunks

# --- Helper Functions ---

def load_processed_urls(file_path: str) -> set[str]:
    """Loads the set of URLs that have already been processed."""
    processed_urls = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                processed_urls.add(line.strip())
    return processed_urls

def save_processed_urls(file_path: str, urls: set[str]):
    """Saves the set of processed URLs to a file."""
    with open(file_path, 'w') as f:
        for url in urls:
            f.write(url + '\n')

# --- Main Processing Logic ---

def build_or_update_index():
    """
    Reads URLs, loads/creates index, processes new URLs, and saves the index.
    """
    processed_urls = load_processed_urls(PROCESSED_URLS_FILE)
    all_urls = set()
    if os.path.exists(URLS_FILE):
        with open(URLS_FILE, 'r') as f:
            for line in f:
                url = line.strip()
                if url:
                    all_urls.add(url)
    else:
        logger.error(f"Error: {URLS_FILE} not found. Create this file with URLs.")
        return

    new_urls_to_process = list(all_urls - processed_urls)

    if not new_urls_to_process:
        logger.info("No new URLs to process.")
        # Load index even if no new URLs, so it's ready for querying
        if os.path.exists(INDEX_STORAGE_DIR):
             logger.info("Loading existing index.")
             return load_index_from_storage(StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR))
        else:
             logger.warning("No new URLs and no existing index found.")
             return None


    logger.info(f"Found {len(new_urls_to_process)} new URLs to process.")

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


    # --- Load Documents from URLs ---
    logger.info("Loading documents from new URLs...")
    documents = []
    reader = WebPageReader()
    # WebPageReader in 0.12.x might not have explicit async batching parameters
    # like newer versions, but it handles loading.
    # For large numbers of URLs, consider implementing your own async loading wrapper
    # if performance is critical and WebPageReader is a bottleneck.
    for url in new_urls_to_process:
        try:
            logger.info(f"Loading {url}...")
            docs = reader.load_data(urls=[url]) # load_data expects a list
            documents.extend(docs)
            logger.info(f"Successfully loaded {url}.")
        except Exception as e:
            logger.error(f"Failed to load {url}. Error: {e}")

    if not documents:
        logger.warning("No documents were successfully loaded from new URLs.")
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

        # --- Update Processed URLs List ---
        updated_processed_urls = processed_urls.union(set(new_urls_to_process))
        save_processed_urls(PROCESSED_URLS_FILE, updated_processed_urls)
        logger.info(f"Updated processed URLs list with {len(new_urls_to_process)} new URLs.")

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
        "What is Llama-Index?",
        "Explain the concept of RAG.",
        "Summarize the key points from the loaded web pages."
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
    logger.info("Starting RAG pipeline setup...")
    vector_index = build_or_update_index()
    if vector_index:
        query_index(vector_index)
    else:
        logger.error("Index could not be built or loaded. Cannot proceed with querying.")
    logger.info("Pipeline execution finished.")

Here's a breakdown of the Python script and how to run it:Configuration:URLS_FILE: Specifies the name of the text file containing your website URLs.INDEX_STORAGE_DIR: This is the directory where the Llama-Index will be saved and loaded from. It will store the processed documents and embeddings.PROCESSED_URLS_FILE: A simple text file to keep track of which URLs have already been successfully processed to avoid duplicates.EMBEDDING_MODEL_NAME and LLM_MODEL_NAME: Set to the Databricks model names you specified (txt-embeddings-3-large and gpt-4o).CHUNK_SIZE and CHUNK_OVERLAP: These are parameters for the text splitter. Llama-Index breaks down large documents into smaller chunks before embedding. CHUNK_SIZE is the size of these chunks, and CHUNK_OVERLAP specifies how many tokens overlap between consecutive chunks to maintain context. You might need to experiment with these values based on the nature of your content for optimal results.Helper Functions:load_processed_urls: Reads the list of previously processed URLs from the PROCESSED_URLS_FILE.save_processed_urls: Writes the current list of processed URLs to the same file.build_or_update_index() Function:This is the core function for data loading and indexing.It first loads the list of URLs that were processed in previous runs.It reads all URLs from the urls.txt file.It identifies the new_urls_to_process by subtracting the already processed URLs from the list in urls.txt.If there are no new URLs, it attempts to load the existing index so you can still query it.It initializes the DatabricksEmbedding and DatabricksLLM models. Crucially, ensure your Databricks access is configured via environment variables (DATABRICKS_HOST, DATABRICKS_TOKEN) before running.It sets the global Settings for Llama-Index, assigning the Databricks models and the chosen chunking parameters. Llama-Index uses these global settings during indexing and querying unless overridden.It uses WebPageReader to load the content from the new_urls_to_process. It iterates through the URLs and loads them one by one. While WebPageReader handles the fetching, Llama-Index's internal processes (like the embedding step) often handle batching and parallelization automatically. For very large numbers of URLs, you might explore custom async loading if the default reader is a bottleneck.It checks if the INDEX_STORAGE_DIR exists.If it exists, it loads the existing index and then inserts the newly loaded documents into it.If it doesn't exist, it creates a brand new index from the loaded documents.It saves the updated (or newly created) index to the INDEX_STORAGE_DIR.Finally, it updates the PROCESSED_URLS_FILE with the URLs that were just processed.query_index() Function:This function takes the index object as input.It creates a query_engine from the index. Since we set the Settings.llm globally, the query engine will automatically use the configured Databricks gpt-4o model.It provides a list of sample queries.It iterates through the sample queries, sends them to the query engine, and prints the responses.if __name__ == "__main__": Block:This is the entry point of the script.It calls build_or_update_index() to prepare the index.If the index is successfully built or loaded, it calls query_index() to demonstrate querying.How to Run:Save the code: Save the Python script as a .py file (e.g., rag_pipeline.py).Create urls.txt: Create a file named urls.txt in the same directory and add the URLs you want to process, one URL per line.Set Databricks Environment Variables: Make sure your DATABRICKS_HOST and DATABRICKS_TOKEN environment variables are set correctly in your terminal or system.Run the script: Open your terminal or command prompt, navigate to the directory where you saved the files, and run:python rag_pipeline.py
The script will first process any new URLs, build/update the index, save it, and then run the sample queries. On subsequent runs with the same urls.txt (unless you add new URLs), it will detect that the URLs are already processed and load the existing index directly before proceeding to querying.

You now have a basic RAG pipeline set up using Llama-Index and Databricks models.This script provides:Loading and processing web content.Using Databricks for embeddings and LLM.Storing and loading the index locally.Handling already processed URLs.A simple query interface.Next Steps:Tune Chunking Parameters: Experiment with CHUNK_SIZE and CHUNK_OVERLAP in the script to see how they affect the quality of your query responses. Optimal values depend on the nature of your data.Error Handling: Enhance error handling for web page loading and Databricks API calls.Advanced Querying: Explore more advanced querying techniques provided by Llama-Index, such as different retriever types, response synthesis methods, or adding metadata filtering.Scalability: For a production environment with a very large number of documents, consider using a dedicated vector database instead of local storage, and look into more robust data ingestion pipelines.Monitoring and Logging: Add more detailed logging to monitor the processing and querying steps.