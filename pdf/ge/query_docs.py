# query_docs.py

import os
import chromadb
import traceback # Import traceback module
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.core.embeddings import resolve_embed_model # Used for global settings
from llama_index.embeddings.openai import OpenAIEmbedding # For direct instantiation
from llama_index.llms.lmstudio import LMStudio # For direct instantiation
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.config import Settings as ChromaSettings

# --- Configuration (Should match process_pdfs.py for consistency) ---
# Model Configurations
EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_API_BASE = "http://localhost:1236/v1" # CHANGE IF YOUR LM STUDIO IS ON A DIFFERENT PORT/URL
EMBEDDING_API_KEY = "lm-studio"

# ChromaDB Configuration (Must point to the same store used by process_pdfs.py)
CHROMA_PERSIST_DIR = "./data/chroma_db_store_cli"
CHROMA_COLLECTION_NAME = "pdf_document_embeddings_cli"

# Global query engine
query_engine = None

# --- Checkpoint 1: Initialize Models and LlamaIndex Settings ---
def initialize_settings_and_models():
    """
    Initializes LlamaIndex global settings with embedding and LLM models.
    """
    print("\n--- Checkpoint 1: Initializing Models and LlamaIndex Settings ---")
    try:
        embed_model = OpenAIEmbedding(
            model_name=EMBEDDING_MODEL_ID,
            api_base=EMBEDDING_API_BASE,
            api_key=EMBEDDING_API_KEY
        )
        Settings.embed_model = embed_model
        print(f"Embedding Model: Configured for LlamaIndex using '{EMBEDDING_MODEL_ID}' at '{EMBEDDING_API_BASE}'.")

        
        Settings.llm = LMStudio(
                    model_name="hermes-3-llama-3.2-3b",  # your Llama model name in LM Studio
                    base_url="http://localhost:1236/v1",  # LM Studio local server URL
                    temperature=0.7,
                    request_timeout=120)
        print("LLM Model: Configured for LlamaIndex using 'hermes-3-llama-3.2-3b'.")
        print("--- Models and Settings Initialized Successfully ---")
        return True
    except Exception as e:
        print(f"ERROR: Failed to initialize models or LlamaIndex settings: {e}")
        # Print full traceback for debugging initialization errors
        traceback.print_exc()
        return False

# --- Checkpoint 2: Setup ChromaDB Vector Store and Load Index ---
def load_index_and_setup_query_engine():
    """
    Connects to ChromaDB, loads the existing vector index, and sets up the query engine.
    """
    global query_engine
    print("\n--- Checkpoint 2: Loading Index and Setting up Query Engine ---")
    try:
        if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
            print(f"ERROR: ChromaDB persist directory '{CHROMA_PERSIST_DIR}' is empty or does not exist.")
            print("Please run 'process_pdfs.py' first to create and populate the vector store.")
            return False
            
        db_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Verify collection exists
        try:
            chroma_collection = db_client.get_collection(name=CHROMA_COLLECTION_NAME)
        except Exception as ce: # Catches exceptions if collection doesn't exist
            print(f"ERROR: Could not get ChromaDB collection '{CHROMA_COLLECTION_NAME}'. It might not exist.")
            print(f"Underlying error: {ce}")
            print("Please ensure 'process_pdfs.py' ran successfully and created this collection.")
            return False

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        print("Attempting to load index from vector store...")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context # Though often not strictly needed if vector_store is primary
        )
        print(f"Successfully loaded index from ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")

        query_engine = index.as_query_engine(
            similarity_top_k=3,
            # You might want to enable streaming for more immediate feedback if the LLM supports it well with Ollama
            # streaming=True, 
        )
        print("Query Engine is ready.")
        print("--- Index Loaded and Query Engine Setup Successfully ---")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load index or set up query engine: {e}")
        print(f"Ensure '{CHROMA_PERSIST_DIR}' contains a valid index created by 'process_pdfs.py'.")
        print(f"Also check if the collection '{CHROMA_COLLECTION_NAME}' exists and is populated.")
        # Print full traceback for debugging index loading errors
        traceback.print_exc()
        return False

# --- Checkpoint 3: Interactive Query Loop ---
def interactive_query_loop():
    """
    Allows the user to interactively ask questions using the query engine.
    """
    global query_engine
    if not query_engine:
        print("\nQuery engine is not available. Cannot start interactive query loop.")
        return

    print("\n--- Checkpoint 3: Starting Interactive Query Mode ---")
    print("Type your questions below. Type 'exit' or 'quit' to end.")
    
    while True:
        try:
            user_query = input("\nAsk a question: ")
            if user_query.lower() in ['exit', 'quit']:
                print("Exiting query mode.")
                break
            if not user_query.strip():
                print("Please enter a question.")
                continue

            print(f"Querying with: \"{user_query}\"")
            print("Please wait for the response...")
            
            # This is the critical call
            response = query_engine.query(user_query)
            
            print("\nLLM Response:")
            print("--------------------------------------------------")
            print(str(response)) # Or response.response for just the text
            print("--------------------------------------------------")

            if hasattr(response, 'source_nodes') and response.source_nodes:
                print("\nSource(s) for the response (snippets from documents):")
                for i, source_node in enumerate(response.source_nodes):
                    print(f"  Source {i+1} (Similarity: {source_node.score:.4f}):")
                    if source_node.node.metadata:
                         print(f"    Metadata: {source_node.node.metadata}")
                    else:
                        print("    Metadata: Not available")
                    print(f"    Text: \"{source_node.node.get_text()[:200].strip()}...\"")
                print("--------------------------------------------------")

        except Exception as e:
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR during query execution: {e}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Print the full traceback to understand where the error occurred
            traceback.print_exc()
            print("--------------------------------------------------")
            # Optionally, you could try to re-initialize the query engine or offer other recovery steps
            # For now, we'll just report and continue the loop or break
            # break # Uncomment to exit loop on error
        except KeyboardInterrupt:
            print("\nExiting query mode due to KeyboardInterrupt.")
            break
            
    print("--- Interactive Query Mode Ended ---")


# --- Main Execution Flow ---
def main():
    """
    Main function to initialize and run the query application.
    """
    print("======================================================")
    print("====== PDF Document Querying Application (CLI) =======")
    print("======================================================")

    if not initialize_settings_and_models():
        print("\nApplication halted due to initialization errors.")
        return

    if not load_index_and_setup_query_engine():
        print("\nApplication halted due to errors loading the index or setting up query engine.")
        return
        
    interactive_query_loop()

    print("\n======================================================")
    print("====== Querying Application Finished              ======")
    print("======================================================")

if __name__ == "__main__":
    main()
