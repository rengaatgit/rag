# process_pdfs.py

import os
import hashlib
import shutil
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.core.embeddings import resolve_embed_model # Used for global settings
from llama_index.embeddings.openai import OpenAIEmbedding # For direct instantiation
from llama_index.llms.lmstudio import LMStudio
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.config import Settings as ChromaSettings

# --- Configuration ---
# IMPORTANT: Adjust these paths and model details if necessary.

# PDF Source Directory (Update this to your local path)
# Use raw string (r"...") or double backslashes ("\\") for Windows paths.
PDF_SOURCE_DIR = r"C:\Users\areng\Desktop\files\gs"

# Model Configurations
# For LM Studio (Embedding Model)
# Ensure LM Studio is running and serving the model at the specified API base.
# The model_name should be what LM Studio expects in API calls for this model.
EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5" # Model name for LM Studio API
EMBEDDING_API_BASE = "http://localhost:1236/v1" # CHANGE IF YOUR LM STUDIO IS ON A DIFFERENT PORT/URL
EMBEDDING_API_KEY = "lm-studio" # Placeholder, often not needed or can be any string for local LM Studio


# ChromaDB Configuration
CHROMA_PERSIST_DIR = "./data/chroma_db_store_cli" # Persistent storage for ChromaDB
CHROMA_COLLECTION_NAME = "pdf_document_embeddings_cli"
PROCESSED_FILES_LOG = "processed_files_cli.log" # Log for processed file hashes

# Create necessary directories
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
# PDF_SOURCE_DIR should already exist, but good to check or inform user.
if not os.path.isdir(PDF_SOURCE_DIR):
    print(f"ERROR: PDF Source Directory not found: {PDF_SOURCE_DIR}")
    print("Please create this directory and place your PDF files in it.")
    exit() # Exit if source directory doesn't exist

processed_file_hashes = set()

# --- Checkpoint 1: Initialize Models and LlamaIndex Settings ---
def initialize_settings_and_models():
    """
    Initializes LlamaIndex global settings with embedding and LLM models.
    This step is crucial for LlamaIndex operations.
    """
    print("\n--- Checkpoint 1: Initializing Models and LlamaIndex Settings ---")
    try:
        # Initialize Embedding Model (from LM Studio via OpenAI-compatible API)
        embed_model = OpenAIEmbedding(
            model_name=EMBEDDING_MODEL_ID,
            api_base=EMBEDDING_API_BASE,
            api_key=EMBEDDING_API_KEY
        )
        Settings.embed_model = embed_model
        print(f"Embedding Model: Configured to use '{EMBEDDING_MODEL_ID}' via LM Studio at '{EMBEDDING_API_BASE}'")

        Settings.llm = LMStudio(
                    model_name="hermes-3-llama-3.2-3b",  # your Llama model name in LM Studio
                    base_url="http://localhost:1236/v1",  # LM Studio local server URL
                    temperature=0.7,
                    request_timeout=120)

        # LlamaIndex default chunk size and overlap are generally good.
        # To customize: Settings.chunk_size = 512; Settings.chunk_overlap = 20
        print(f"LlamaIndex Settings: Chunk Size = {Settings.chunk_size}, Chunk Overlap = {Settings.chunk_overlap}")
        print("--- Models and Settings Initialized Successfully ---")
        return True
    except Exception as e:
        print(f"ERROR: Failed to initialize models or LlamaIndex settings: {e}")
        return False

# --- Checkpoint 2: Load Processed File Hashes ---
def load_processed_files_log():
    """Loads hashes of previously processed files to avoid re-processing."""
    global processed_file_hashes
    print("\n--- Checkpoint 2: Loading Processed Files Log ---")
    try:
        if os.path.exists(PROCESSED_FILES_LOG):
            with open(PROCESSED_FILES_LOG, "r") as f:
                processed_file_hashes = set(line.strip() for line in f)
            print(f"Loaded {len(processed_file_hashes)} entries from '{PROCESSED_FILES_LOG}'.")
        else:
            print(f"No processed files log ('{PROCESSED_FILES_LOG}') found. Will create a new one.")
        print("--- Processed Files Log Loaded Successfully (or initialized if new) ---")
        return True
    except Exception as e:
        print(f"WARNING: Error loading processed files log: {e}. Proceeding, but might re-process files.")
        return True # Non-critical for startup, but good to warn

def add_to_processed_files_log(file_hash, filename):
    """Adds a file hash to the log and the in-memory set."""
    global processed_file_hashes
    try:
        with open(PROCESSED_FILES_LOG, "a") as f:
            f.write(file_hash + "\n")
        processed_file_hashes.add(file_hash)
        print(f"Added hash for '{filename}' ({file_hash}) to processed files log.")
    except Exception as e:
        print(f"WARNING: Error adding to processed files log for '{filename}': {e}")

def get_file_hash(filepath):
    """Computes SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read(65536) # Read in chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        print(f"ERROR: Could not calculate hash for {filepath}: {e}")
        return None

# --- Checkpoint 3: Setup ChromaDB Vector Store ---
def setup_vector_store():
    """
    Sets up the ChromaDB client and collection.
    Embeddings will be stored here. Loads existing collection if present.
    """
    print("\n--- Checkpoint 3: Setting up ChromaDB Vector Store ---")
    try:
        db_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False) # Disables telemetry
        )
        # get_or_create_collection is idempotent
        chroma_collection = db_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
            # The embedding function is handled by LlamaIndex when it writes,
            # so not explicitly set here for ChromaDB collection itself.
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        print(f"ChromaDB: Collection '{CHROMA_COLLECTION_NAME}' is ready at '{CHROMA_PERSIST_DIR}'.")
        print("--- ChromaDB Vector Store Setup Successfully ---")
        return vector_store
    except Exception as e:
        print(f"ERROR: Failed to set up ChromaDB: {e}")
        print("Check permissions for the persist directory and ChromaDB installation.")
        return None

# --- Checkpoint 4: Process PDF Documents from Folder ---
def process_documents_in_folder(vector_store):
    """
    Scans the PDF_SOURCE_DIR, processes new PDF files, embeds them,
    and stores them in the vector_store.
    """
    print("\n--- Checkpoint 4: Processing PDF Documents from Folder ---")
    if not vector_store:
        print("ERROR: Vector store not available. Cannot process documents.")
        return False

    new_files_processed_count = 0
    skipped_files_count = 0
    files_to_process_paths = []
    filenames_for_processing = []

    print(f"Scanning for PDF files in: {PDF_SOURCE_DIR}")
    for filename in os.listdir(PDF_SOURCE_DIR):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(PDF_SOURCE_DIR, filename)
            file_hash = get_file_hash(filepath)

            if not file_hash: # Error in hashing
                print(f"Skipping '{filename}' due to hashing error.")
                continue

            if file_hash in processed_file_hashes:
                print(f"Ignoring '{filename}': Already processed (Hash: {file_hash[:8]}...).")
                skipped_files_count += 1
                continue
            
            files_to_process_paths.append(filepath)
            filenames_for_processing.append(filename)

    if not files_to_process_paths:
        if skipped_files_count > 0:
            print(f"No new PDF files to process. {skipped_files_count} file(s) were already processed.")
        else:
            print("No PDF files found in the source directory to process.")
        print("--- Document Processing Check Complete (No new files) ---")
        return True # No new files is not an error in itself

    print(f"Found {len(files_to_process_paths)} new PDF file(s) to process: {filenames_for_processing}")

    # Use a temporary directory for SimpleDirectoryReader if many files or complex paths,
    # but for direct paths, it can also work. Here, we pass file paths directly.
    # However, SimpleDirectoryReader is often more robust with a directory.
    # For simplicity with a fixed source dir, we can try loading them individually or use input_files.
    
    # Create a temporary directory to copy files for SimpleDirectoryReader
    # This ensures SimpleDirectoryReader has a clean space and handles various file path issues.
    temp_processing_dir = "./data/temp_pdf_processing_staging"
    os.makedirs(temp_processing_dir, exist_ok=True)
    
    staged_file_paths_for_reader = []
    original_paths_for_logging = []

    for original_path, filename in zip(files_to_process_paths, filenames_for_processing):
        staged_path = os.path.join(temp_processing_dir, filename)
        try:
            shutil.copy2(original_path, staged_path) # copy2 preserves metadata
            staged_file_paths_for_reader.append(staged_path)
            original_paths_for_logging.append(original_path) # Keep track of original for hashing
        except Exception as e:
            print(f"Error copying '{filename}' to staging area: {e}. Skipping this file.")

    if not staged_file_paths_for_reader:
        print("No files were successfully staged for processing.")
        shutil.rmtree(temp_processing_dir)
        return False

    try:
        print(f"Loading content from {len(staged_file_paths_for_reader)} staged PDF files...")
        # SimpleDirectoryReader loads all files from input_dir or specific files from input_files
        # It uses LlamaIndex global Settings.embed_model for embeddings later.
        # Chunking (size, overlap) is also governed by global Settings.
        reader = SimpleDirectoryReader(input_dir=temp_processing_dir, required_exts=[".pdf"])
        documents = reader.load_data(show_progress=True)
        
        if not documents:
            print("No content could be extracted from the new PDF files after loading.")
            shutil.rmtree(temp_processing_dir)
            return False # Indicate that processing didn't yield documents

        print(f"Loaded {len(documents)} document chunks from {len(staged_file_paths_for_reader)} PDF file(s).")

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Try to load an existing index from the vector store to insert into it.
        # If it doesn't exist, a new one will be created.
        try:
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
            print("Existing index loaded from ChromaDB. Inserting new documents...")
            index.insert_nodes(documents) # Insert new document chunks
        except ValueError as ve: # Often indicates an empty or incompatible store for from_vector_store
            print(f"Could not load existing index (may be first run or empty store: {ve}). Creating new index.")
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
                # embed_batch_size can be set here if needed, e.g., embed_batch_size=10
            )
            print("New index created and initial documents embedded.")
        
        # Log successfully processed files
        for original_filepath, staged_filename in zip(original_paths_for_logging, os.listdir(temp_processing_dir)):
            # It's more robust to re-calculate hash from original path or use stored one
            file_hash = get_file_hash(original_filepath) # Ensure we log the correct hash
            if file_hash:
                 add_to_processed_files_log(file_hash, os.path.basename(original_filepath))
        
        new_files_processed_count = len(original_paths_for_logging)
        print(f"Successfully processed and embedded {new_files_processed_count} new PDF file(s).")
        print("Embeddings stored in ChromaDB.")
        print("--- Document Processing and Embedding Successful ---")
        return True

    except Exception as e:
        print(f"ERROR: An error occurred during PDF document loading or embedding: {e}")
        return False
    finally:
        # Clean up the temporary staging directory
        if os.path.exists(temp_processing_dir):
            shutil.rmtree(temp_processing_dir)
            print(f"Cleaned up temporary staging directory: {temp_processing_dir}")


# --- Main Execution Flow ---
def main():
    """
    Main function to orchestrate the PDF processing pipeline.
    """
    print("======================================================")
    print("=== Starting PDF Document Processing Pipeline (CLI) ===")
    print("======================================================")

    # Step 1: Initialize models and LlamaIndex settings
    if not initialize_settings_and_models():
        print("\nPipeline halted due to initialization errors.")
        return

    # Step 2: Load history of processed files
    if not load_processed_files_log():
        # This function currently always returns True but might change
        print("\nContinuing despite potential issues with processed files log.")

    # Step 3: Setup ChromaDB vector store
    vector_store = setup_vector_store()
    if not vector_store:
        print("\nPipeline halted due to vector store setup errors.")
        return

    # Step 4: Process documents in the specified folder
    success = process_documents_in_folder(vector_store)
    if success:
        print("\nPDF processing completed.")
    else:
        print("\nPDF processing encountered errors or no new files were processed.")

    print("\n======================================================")
    print("=== PDF Document Processing Pipeline Finished       ===")
    print("======================================================")
    print(f"Persistent data stored in: {os.path.abspath(CHROMA_PERSIST_DIR)}")
    print(f"Processed files log: {os.path.abspath(PROCESSED_FILES_LOG)}")
    print("You can now use 'query_docs.py' to ask questions about the processed documents.")

if __name__ == "__main__":
    main()
