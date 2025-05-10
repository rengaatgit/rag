import os
import json
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.core.response.schema import ResponseMode # For specifying response synthesis mode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as OpenAI_LLM # Renamed to avoid confusion
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- Configuration ---
# Embedding Model Configuration (Databricks)
DATABRICKS_EMBEDDING_ENDPOINT_URL = "YOUR_DATABRICKS_EMBEDDING_ENDPOINT_URL/invocations"
DATABRICKS_EMBEDDING_TOKEN = "YOUR_DATABRICKS_EMBEDDING_PAT_TOKEN" # Token for embedding model
EMBEDDING_MODEL_NAME = "text-embedding-ada-002" # Or your specific embedding model name

# Language Model (LLM) for Response Synthesis Configuration (Databricks)
# This endpoint should serve a generative LLM (e.g., Llama, MPT, GPT-like)
# and be OpenAI API compatible (e.g., for /v1/chat/completions or /v1/completions)
DATABRICKS_LLM_ENDPOINT_URL = "YOUR_DATABRICKS_LLM_ENDPOINT_URL/invocations" # Or /v1/chat/completions etc.
DATABRICKS_LLM_TOKEN = "YOUR_DATABRICKS_LLM_PAT_TOKEN" # Token for LLM, can be same as embedding token
# Example: "databricks-llama-2-70b-chat" or "gpt-3.5-turbo" if your endpoint serves that
LLM_MODEL_NAME = "databricks-llm-model" # Name your Databricks LLM endpoint expects

# Path to your JSON file generated from "git log --stat"
GIT_LOG_JSON_PATH = "git_log.json"

# ChromaDB configuration
CHROMA_DB_PATH = "./chroma_db" # Directory to store ChromaDB data
CHROMA_COLLECTION_NAME = "git_log_embeddings"

def setup_embedding_model():
    """
    Sets up the OpenAIEmbedding object to use Databricks model serving endpoint for embeddings.
    """
    print(f"Configuring embedding model to use Databricks endpoint: {DATABRICKS_EMBEDDING_ENDPOINT_URL}")
    
    embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL_NAME,
        api_base=DATABRICKS_EMBEDDING_ENDPOINT_URL,
        api_key=DATABRICKS_EMBEDDING_TOKEN,
    )
    
    Settings.embed_model = embed_model
    print("Embedding model configured.")
    return embed_model

def setup_llm():
    """
    Sets up the OpenAI LLM object to use Databricks model serving endpoint for response synthesis.
    Assumes the Databricks endpoint is compatible with OpenAI's API (e.g., for chat completions).
    """
    print(f"Configuring LLM to use Databricks endpoint: {DATABRICKS_LLM_ENDPOINT_URL}")
    
    # If your Databricks endpoint is for chat models (recommended for LlamaIndex)
    llm = OpenAI_LLM(
        model=LLM_MODEL_NAME,
        api_base=DATABRICKS_LLM_ENDPOINT_URL,
        api_key=DATABRICKS_LLM_TOKEN,
        # temperature=0.7, # Optional: control creativity
        # max_tokens=500,  # Optional: control response length
    )
    # If your endpoint is a legacy completions endpoint, you might need a different class
    # or specific parameters. However, LlamaIndex generally prefers chat model interfaces.

    Settings.llm = llm
    print("LLM for response synthesis configured.")
    return llm

def load_and_process_git_log(file_path):
    """
    Loads git log data from a JSON file and converts it into LlamaIndex Documents.
    """
    print(f"Loading git log data from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            git_commits = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {file_path}. Please create it or check the path.")
        # (Sample JSON printout omitted for brevity, it's the same as before)
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Please ensure it's valid JSON.")
        return []

    documents = []
    for commit in git_commits:
        content = (
            f"Commit: {commit.get('commit_hash', '')}\n"
            f"Author: {commit.get('author_name', '')} <{commit.get('author_email', '')}>\n"
            f"Date: {commit.get('date', '')}\n"
            f"Subject: {commit.get('subject', '')}\n"
            f"Body: {commit.get('body', '')}\n"
            f"Stats: {commit.get('stats', '')}"
        )
        metadata = {
            "commit_hash": commit.get('commit_hash'),
            "author_name": commit.get('author_name'),
            "date": commit.get('date'),
            "file_path": file_path
        }
        doc = Document(text=content, metadata=metadata)
        documents.append(doc)
    
    print(f"Loaded and processed {len(documents)} commits into LlamaIndex Documents.")
    return documents

def store_in_chromadb(documents, db_path, collection_name):
    """
    Stores LlamaIndex Documents into ChromaDB.
    """
    if not documents:
        print("No documents to store. Skipping ChromaDB storage.")
        return None

    print(f"Initializing ChromaDB client at: {db_path}")
    db = chromadb.PersistentClient(path=db_path)
    print(f"Getting or creating Chroma collection: {collection_name}")
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Building VectorStoreIndex and ingesting documents...")
    # Embeddings are generated here using Settings.embed_model
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print(f"Documents ingested into ChromaDB collection '{collection_name}'.")
    return index

def load_index_from_chromadb(db_path, collection_name):
    """
    Loads an existing index from ChromaDB.
    Requires Settings.embed_model (and Settings.llm if query engine needs it implicitly) to be set.
    """
    print(f"Loading index from ChromaDB. Path: {db_path}, Collection: {collection_name}")
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Settings.embed_model is used here to configure the index for query-time embedding generation
    # Settings.llm will be used by the query engine later
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )
    print("Index loaded from ChromaDB.")
    return index

def query_index(index, user_query):
    """
    Queries the VectorStoreIndex with a user prompt, using the configured LLM for response synthesis.
    """
    if not index:
        print("Index not available. Cannot query.")
        return None
    
    if not Settings.llm:
        print("LLM not configured in Settings. Response synthesis might be basic.")
        # Fallback or error, depending on desired behavior
    
    print(f"\nQuerying index with: '{user_query}'")
    # Create a query engine.
    # ResponseMode.COMPACT (or REFINE, TREE_SUMMARIZE) will use the LLM from Settings.llm
    # to synthesize a response based on retrieved nodes.
    query_engine = index.as_query_engine(
        similarity_top_k=3, # Retrieve top 3 similar documents
        response_mode=ResponseMode.COMPACT # Use LLM to synthesize a compact answer
    )
    
    response = query_engine.query(user_query)
    
    print("\nSynthesized Query Response (from LLM):")
    print(response) # This is the synthesized response string from the LLM
    
    print("\nSource Nodes (used for synthesis):")
    for node in response.source_nodes:
        print(f"  Score: {node.score:.4f}")
        print(f"  Commit Hash: {node.metadata.get('commit_hash', 'N/A')}")
        print(f"  Author: {node.metadata.get('author_name', 'N/A')}")
        print(f"  Text Snippet: {node.text[:200]}...")
        print("-" * 20)
    return response

def main():
    """
    Main function to run the RAG pipeline.
    """
    # --- 1. Setup Models (Embedding and LLM using Databricks) ---
    # Check placeholder values for embedding model
    if "YOUR_DATABRICKS" in DATABRICKS_EMBEDDING_ENDPOINT_URL or \
       "YOUR_DATABRICKS" in DATABRICKS_EMBEDDING_TOKEN:
        print("Error: Please replace placeholder values for DATABRICKS_EMBEDDING_ENDPOINT_URL and DATABRICKS_EMBEDDING_TOKEN.")
        # (Dummy git_log.json creation logic omitted for brevity)
        return

    # Check placeholder values for LLM
    if "YOUR_DATABRICKS" in DATABRICKS_LLM_ENDPOINT_URL or \
       "YOUR_DATABRICKS" in DATABRICKS_LLM_TOKEN:
        print("Error: Please replace placeholder values for DATABRICKS_LLM_ENDPOINT_URL and DATABRICKS_LLM_TOKEN.")
        return

    setup_embedding_model()
    setup_llm() # Setup the generative LLM

    # --- 2. Process JSON and Store/Load from ChromaDB ---
    index = None
    should_ingest = True 
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Found existing ChromaDB path at '{CHROMA_DB_PATH}'. Attempting to load index.")
        try:
            # Both embed_model and llm should be set in Settings before loading/querying
            index = load_index_from_chromadb(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)
            print("Successfully loaded index from ChromaDB.")
            should_ingest = False 
        except Exception as e:
            print(f"Could not load index from ChromaDB: {e}. Will attempt to ingest new data.")
            should_ingest = True
            # if os.path.exists(CHROMA_DB_PATH): # Clean up if loading failed and re-ingest
            #     import shutil
            #     shutil.rmtree(CHROMA_DB_PATH)
            # os.makedirs(CHROMA_DB_PATH, exist_ok=True)


    if should_ingest:
        documents = load_and_process_git_log(GIT_LOG_JSON_PATH)
        if not documents:
            print("No documents loaded. Exiting.")
            return
        index = store_in_chromadb(documents, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)
    
    if not index:
        print("Failed to create or load the index. Exiting.")
        return

    # --- 3. Query Index with LLM Synthesis ---
    user_query = input("\nEnter your query about the git history (e.g., 'summarize bug fixes by John Doe related to payments'): ")
    if not user_query:
        print("No query provided. Exiting.")
        return
        
    query_index(index, user_query)

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    # (Initial dummy git_log.json creation logic omitted for brevity, it's the same as before)
    if not os.path.exists(GIT_LOG_JSON_PATH):
        print(f"'{GIT_LOG_JSON_PATH}' not found.")
        # ... (sample file creation code) ...
        print(f"A sample '{GIT_LOG_JSON_PATH}' has been created. Populate it and configure Databricks details.")

    main()
