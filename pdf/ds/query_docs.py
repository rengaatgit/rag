import sys
from pathlib import Path
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.lmstudio import LMStudio
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

# Configuration
EMBEDDINGS_DIR = Path("data/embeddings")
LM_STUDIO_ENDPOINT = "http://localhost:1236/v1"

def initialize_models():
# Instantiate the LMStudio LLM client
    Settings.llm = LMStudio(
    model_name="hermes-3-llama-3.2-3b",  # your Llama model name in LM Studio
    base_url="http://localhost:1236/v1",  # LM Studio local server URL
    temperature=0.7,
    request_timeout=120)
    
    Settings.embed_model = OpenAIEmbedding(
        model_name="nomic-embed-text-v1.5",
        api_base=LM_STUDIO_ENDPOINT,
        api_key="no-key-needed"
    )

def query_engine():
    """Initialize query engine with proper model settings"""
    # Initialize models first
    initialize_models()
    
    # ChromaDB setup
    chroma_client = chromadb.PersistentClient(
        path=str(EMBEDDINGS_DIR),
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_client.get_collection("pdf_docs")
    )
    
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(EMBEDDINGS_DIR)
    )
    
    index = load_index_from_storage(storage_context)
    
    return index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
        streaming=True
    )

if __name__ == "__main__":
    engine = query_engine()
    print("Query engine ready. Type 'exit' to quit.")
    
    while True:
        try:
            query = input("\nQuestion: ")
            if query.lower() == 'exit':
                break
                
            response = engine.query(query)
            print(f"\nAnswer: {response}")
            
        except Exception as e:
            print(f"Error: {str(e)}")