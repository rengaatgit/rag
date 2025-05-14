import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

# Configuration
PDF_SOURCE_DIR = Path(r"C:\Users\areng\Desktop\files\gs")
EMBEDDINGS_DIR = Path("data/embeddings")
PROCESSED_HASHES_FILE = Path("data/processed_hashes.txt")

# Model Configuration
LM_STUDIO_ENDPOINT = "http://localhost:1236/v1"

def checkpoint(name: str, func, *args, **kwargs):
    """Checkpoint execution with error handling"""
    print(f"\n{'='*40}\nCheckpoint: {name}\n{'='*40}")
    try:
        result = func(*args, **kwargs)
        print(f"✓ Success: {name}")
        return result
    except Exception as e:
        print(f"✗ Failed: {name}\nError: {str(e)}\nType: {type(e).__name__}")
        sys.exit(1)

def setup_directories():
    """Create required directories"""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_HASHES_FILE.parent.mkdir(parents=True, exist_ok=True)

def get_file_hash(file_path: Path) -> str:
    """Generate SHA-256 hash of file contents"""
    hasher = hashlib.sha256()
    hasher.update(file_path.read_bytes())
    return hasher.hexdigest()

def load_processed_hashes() -> set:
    """Load existing file hashes"""
    if PROCESSED_HASHES_FILE.exists():
        return set(PROCESSED_HASHES_FILE.read_text().splitlines())
    return set()

def process_pdfs():
    """Main PDF processing pipeline"""
    # Checkpoint 1: Directory setup
    checkpoint("Directory Setup", setup_directories)
    
    # Checkpoint 2: Model initialization
    Settings.llm = checkpoint(
            "LLM Model Initialization",
            OpenAI,
            api_key="no-key-needed",
            api_base=LM_STUDIO_ENDPOINT,
            model="hermes-3-llama-3.2-3b",
            temperature=0.1
        )
    
    Settings.embed_model = checkpoint(
        "Embedding Model Initialization",
        OpenAIEmbedding,
        model_name="nomic-ai/nomic-embed-text-v1.5",
        api_base=LM_STUDIO_ENDPOINT,
        api_key="no-key-required"
    )
    
    # Checkpoint 3: PDF file discovery
    pdf_files = checkpoint(
        "PDF File Discovery",
        lambda: list(PDF_SOURCE_DIR.glob("*.pdf")))
    
    # Checkpoint 4: File filtering
    processed_hashes = checkpoint(
        "Processed Hashes Load",
        load_processed_hashes
    )
    
    new_files = []
    for pdf_file in pdf_files:
        file_hash = checkpoint(
            f"Hash Generation: {pdf_file.name}",
            get_file_hash, pdf_file
        )
        if file_hash not in processed_hashes:
            new_files.append(pdf_file)
    
    if not new_files:
        print("No new files to process")
        return

    # Checkpoint 5: Document processing
    documents = checkpoint(
        "Document Loading",
        SimpleDirectoryReader(
            input_dir=str(PDF_SOURCE_DIR),
            file_extractor={"pdf": "PDFReader"},
            recursive=True
        ).load_data
    )
    
    # Checkpoint 6: Text splitting
    node_parser = checkpoint(
        "Node Parser Configuration",
        SemanticSplitterNodeParser.from_defaults,
        buffer_size=512,  # Context window size
        breakpoint_percentile_threshold=95,  # Auto chunk sizing
        embed_model=Settings.embed_model
    )
    
    nodes = checkpoint(
        "Node Creation",
        node_parser.get_nodes_from_documents,
        documents
    )
    
    # Checkpoint 7: Vector store setup
    chroma_client = checkpoint(
        "ChromaDB Connection",
        chromadb.PersistentClient,
        path=str(EMBEDDINGS_DIR),
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    vector_store = checkpoint(
        "Vector Store Initialization",
        ChromaVectorStore,
        chroma_collection=chroma_client.get_or_create_collection("pdf_docs")
    )
    
    # Checkpoint 8: Index creation
    index = checkpoint(
        "Index Creation",
        VectorStoreIndex,
        nodes=nodes,
        vector_store=vector_store,
        show_progress=True
    )
    
    # Checkpoint 9: Persist data
    checkpoint(
        "Index Persistence",
        index.storage_context.persist,
        persist_dir=str(EMBEDDINGS_DIR))
    
    # Update processed hashes
    with open(PROCESSED_HASHES_FILE, "a") as f:
        for pdf_file in new_files:
            f.write(f"{get_file_hash(pdf_file)}\n")
    
    print(f"Processed {len(new_files)} new files")

if __name__ == "__main__":
    process_pdfs()