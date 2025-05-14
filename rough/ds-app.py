import os
import sys
import time
import hashlib
import gradio as gr
from pathlib import Path
from typing import List, Dict
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

# Configuration
PDF_UPLOAD_DIR = "data/pdfs"
EMBEDDINGS_DIR = "data/embeddings"
PROCESSED_HASHES = "data/processed_hashes.txt"
Path(PDF_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)

# Local Model Configuration
LOCAL_OLLAMA_ENDPOINT = "http://localhost:11434"
LOCAL_LM_STUDIO_ENDPOINT = "http://localhost:1234/v1"

# Initialize Models
Settings.llm = Ollama(
    model="llama3:8b",  # Updated to correct model name format
    base_url=LOCAL_OLLAMA_ENDPOINT,
    temperature=0.1,
    request_timeout=120,
)

class LMStudioEmbedding(OpenAIEmbedding):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            api_base=LOCAL_LM_STUDIO_ENDPOINT,
            api_key="lm-studio",  # Placeholder, not actually used
            **kwargs
        )

Settings.embed_model = LMStudioEmbedding()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(
    path=EMBEDDINGS_DIR,
    settings=ChromaSettings(
        allow_reset=False,
        anonymized_telemetry=False,
        is_persistent=True
    ),
)

def get_file_hash(file_path: str) -> str:
    """Generate unique file hash to prevent duplicates"""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_processed_hashes() -> set:
    """Load processed file hashes"""
    if os.path.exists(PROCESSED_HASHES):
        with open(PROCESSED_HASHES, "r") as f:
            return set(f.read().splitlines())
    return set()

def save_processed_hash(file_hash: str):
    """Store processed file hash"""
    with open(PROCESSED_HASHES, "a") as f:
        f.write(f"{file_hash}\n")

def process_pdfs(files: List[str]) -> Dict:
    """Process PDFs with local models"""
    processed_hashes = load_processed_hashes()
    new_files = []
    
    for file_path in files:
        file_hash = get_file_hash(file_path)
        if file_hash in processed_hashes:
            continue
        
        dest_path = Path(PDF_UPLOAD_DIR) / Path(file_path).name
        dest_path.write_bytes(Path(file_path).read_bytes())
        save_processed_hash(file_hash)
        new_files.append(dest_path)
    
    if not new_files:
        return {"status": "skipped", "message": "All files already processed"}
    
    try:
        # Configure document processing
        node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separator="\n"
        )
        
        reader = SimpleDirectoryReader(
            input_dir=PDF_UPLOAD_DIR,
            file_extractor={"pdf": "PDFReader"},
            recursive=True,
           #num_workers=os.cpu_count(),
        )
        
        documents = reader.load_data(show_progress=True)
        nodes = node_parser(documents)
        
        vector_store = ChromaVectorStore(
            chroma_client.get_or_create_collection("pdf_docs")
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
            show_progress=True,
        )
        
        index.storage_context.persist(persist_dir=EMBEDDINGS_DIR)
        return {
            "status": "success",
            "message": f"Processed {len(new_files)} PDFs. Total chunks: {len(nodes)}",
        }
    except Exception as e:
        return {"status": "error", "message": f"Processing failed: {str(e)}"}

def query_engine():
    """Initialize query engine with local models"""
    vector_store = ChromaVectorStore(chroma_client.get_collection("pdf_docs"))
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=EMBEDDINGS_DIR,
    )
    index = load_index_from_storage(storage_context)
    return index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
        streaming=True,
    )

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Local PDF Analyzer") as demo:
    chatbot = gr.Chatbot(height=700, label="Document Analysis Console", type="messages")
    
    with gr.Row():
        upload_btn = gr.UploadButton(
            "📤 Upload PDFs",
            file_types=[".pdf"],
            file_count="multiple",
            scale=1,
        )
        query_input = gr.Textbox(
            placeholder="Ask about your documents...",
            show_label=False,
            container=False,
            scale=6,
            autofocus=True,
        )
        submit_btn = gr.Button("Analyze", variant="primary", scale=1)
    
    def handle_upload(files, history):
        try:
            result = process_pdfs([file.name for file in files])
            status_msg = (
                f"✅ {result['message']}" if result["status"] == "success" else
                f"⏭️ {result['message']}" if result["status"] == "skipped" else
                f"❌ {result['message']}"
            )
            return history + [[None, status_msg]]
        except Exception as e:
            return history + [[None, f"❌ Critical Error: {str(e)}"]]
    
    def handle_query(query, history):
        try:
            engine = query_engine()
            response = engine.query(query)
            return history + [[query, response.response]]
        except Exception as e:
            return history + [[query, f"❌ Query Failed: {str(e)}"]]
    
    upload_btn.upload(
        handle_upload,
        [upload_btn, chatbot],
        chatbot,
    )
    
    submit_btn.click(
        handle_query,
        [query_input, chatbot],
        chatbot,
    ).then(lambda: "", outputs=query_input)
    
    query_input.submit(
        handle_query,
        [query_input, chatbot],
        chatbot,
    ).then(lambda: "", outputs=query_input)

def execute_with_checkpoint(step_name: str, func, *args, **kwargs):
    """Helper function for checkpoint execution"""
    print(f"\n{'='*40}\nStarting checkpoint: {step_name}\n{'='*40}")
    try:
        result = func(*args, **kwargs)
        print(f"\n{'✓'*5} Checkpoint passed: {step_name}")
        return result
    except Exception as e:
        print(f"\n{'✗'*5} Checkpoint failed: {step_name}")
        print(f"Error details: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("Skipping subsequent steps...")
        return None

if __name__ == "__main__":
    # Checkpoint 1: Directory validation
    test_dir = execute_with_checkpoint(
        "Directory Validation",
        lambda: Path(r"C:\Users\areng\Desktop\files")
    )
    if not test_dir:
        sys.exit(1)
        
    if not execute_with_checkpoint("Directory Existence Check", lambda: test_dir.exists()):
        sys.exit(1)

    # Checkpoint 2: File discovery
    test_files = execute_with_checkpoint(
        "PDF File Discovery",
        lambda: list(test_dir.glob("*.pdf"))
    )
    if not test_files:
        print("No PDF files found. Launching UI with empty knowledge base.")
    else:
        # Checkpoint 3: File processing
        file_paths = [str(f.absolute()) for f in test_files]
        process_result = execute_with_checkpoint(
            "PDF Processing Pipeline",
            process_pdfs,
            file_paths
        )
        
        if process_result and process_result.get("status") == "success":
            # Checkpoint 4: Query engine setup
            engine = execute_with_checkpoint(
                "Query Engine Initialization",
                query_engine
            )
            
            if engine:
                # Checkpoint 5: Test queries
                test_queries = [
                    "What is the main topic of these documents?",
                    "List 3 key points from the documents",
                ]
                for idx, query in enumerate(test_queries, 1):
                    execute_with_checkpoint(
                        f"Test Query {idx} - '{query[:15]}...'",
                        lambda q: print(f"\nQuery: {q}\nResponse: {engine.query(q)}"),
                        query
                    )

    # Final Checkpoint: UI Launch
    print("\n\nLaunching Gradio Interface...")
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            prevent_thread_lock=False
        )
        print("UI successfully launched at http://localhost:7860")
        print("Press Ctrl+C to shutdown the server")
        
        # Keep alive indefinitely
        while True:
            time.sleep(3600)
            
    except KeyboardInterrupt:
        print("\nGraceful shutdown initiated...")
    except Exception as e:
        print(f"UI launch failed: {str(e)}")
        sys.exit(1)