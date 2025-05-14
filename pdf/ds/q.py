from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

MODEL_ENDPOINT = "http://localhost:1236/v1"
CHROMA_DIR = "./data/chroma_db"

def initialize_query_engine():
    try:
        # Initialize models with proper embedding configuration
        Settings.llm = OpenAI(
            api_key="no-key-needed",
            api_base=MODEL_ENDPOINT,
            model="hermes-3-llama-3.2-3b",
            temperature=0.1
        )
        
        Settings.embed_model = OpenAIEmbedding(
            api_key="no-key-needed",
            api_base=MODEL_ENDPOINT,
            model_name="nomic-embed-text-v1.5"
        )

        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        chroma_collection = chroma_client.get_collection("pdf_docs")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create index with proper configuration
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )
        
        return index.as_query_engine(
            similarity_top_k=5,
            verbose=True
        )
    except Exception as e:
        print(f"Query engine initialization failed: {str(e)}")
        return None

if __name__ == "__main__":
    query_engine = initialize_query_engine()
    if not query_engine:
        exit(1)
        
    while True:
        try:
            query = input("\nEnter your question (type 'exit' to quit): ")
            if query.lower() == "exit":
                break
                
            response = query_engine.query(query)
            print(f"\nAnswer: {response.response}")
            print("\nSources:")
            for node in response.source_nodes:
                print(f"- {node.metadata.get('file_name', 'Unknown')} (Score: {node.score:.2f})")
                
        except Exception as e:
            print(f"Query failed: {str(e)}")