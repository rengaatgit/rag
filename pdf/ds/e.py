import os
import hashlib
import json
from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Configuration
PDF_DIR = r"C:\Users\areng\Desktop\files\gs"
CHROMA_DIR = "./data/chroma_db"
METADATA_FILE = "./data/processed_files.json"
MODEL_ENDPOINT = "http://localhost:1236/v1"

def load_processed_files():
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        return {}

def save_processed_files(processed):
    try:
        with open(METADATA_FILE, "w") as f:
            json.dump(processed, f)
        return True
    except Exception as e:
        print(f"Error saving metadata: {str(e)}")
        return False

def file_hash(file_path):
    try:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error calculating hash: {str(e)}")
        return None

def initialize_models():
    try:
        # LLM configuration
        Settings.llm = OpenAI(
            api_key="no-key-needed",
            api_base=MODEL_ENDPOINT,
            model="hermes-3-llama-3.2-3b",
            temperature=0.1
        )
        
        # Embedding configuration
        Settings.embed_model = OpenAIEmbedding(
            api_key="no-key-needed",
            api_base=MODEL_ENDPOINT,
            model_name="nomic-embed-text-v1.5"
        )
        
        # Configure chunking parameters
        Settings.chunk_size = 1024  # Auto-tuned based on model context
        Settings.chunk_overlap = 200
        return True
    except Exception as e:
        print(f"Model initialization failed: {str(e)}")
        return False

def process_documents():
    try:
        # Check input directory
        if not os.path.exists(PDF_DIR):
            print(f"PDF directory not found: {PDF_DIR}")
            return False
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        chroma_collection = chroma_client.get_or_create_collection("pdf_docs")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Load processed files metadata
        processed_files = load_processed_files()
        new_files = []
        
        # Process PDF files
        reader = SimpleDirectoryReader(
            input_dir=PDF_DIR,
            recursive=True,
            required_exts=[".pdf"]
        )
        
        for file_path in Path(PDF_DIR).rglob("*.pdf"):
            file_id = file_hash(file_path)
            if not file_id:
                continue
                
            if file_id in processed_files:
                print(f"Skipping already processed file: {file_path}")
                continue
                
            try:
                documents = reader.load_file(file_path)
                VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    show_progress=True
                )
                processed_files[file_id] = str(file_path)
                new_files.append(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
                
        if new_files:
            if save_processed_files(processed_files):
                print(f"Processed {len(new_files)} new files")
            else:
                print("Failed to save processing metadata")
        else:
            print("No new files to process")
            
        return True
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Checkpoint-based execution
    if not initialize_models():
        exit(1)
        
    if not process_documents():
        exit(1)
        
    print("Embedding pipeline completed successfully")