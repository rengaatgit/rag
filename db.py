import os
import requests
import chromadb
import json
from typing import List, Dict, Any, Optional, Tuple


class DatabricksEmbedder:
    """Class to handle embeddings generation using a Databricks-hosted model."""
    
    def __init__(self, model_endpoint: str, api_token: str):
        """
        Initialize the embedder with Databricks model endpoint and API token.
        
        Args:
            model_endpoint: URL of the Databricks model serving endpoint
            api_token: Databricks API token for authentication
        """
        self.model_endpoint = model_endpoint
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            List of embedding vectors (as lists of floats)
        """
        # Prepare the payload for Databricks model serving
        payload = {
            "inputs": texts
        }
        
        try:
            response = requests.post(
                self.model_endpoint,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            # The exact structure of the response depends on your model's output format
            # Adjust the parsing below based on your model's response structure
            embeddings = result.get("predictions", [])
            
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise


class DatabricksLLM:
    """Class to handle LLM inference using a Databricks-hosted model."""
    
    def __init__(self, model_endpoint: str, api_token: str):
        """
        Initialize the LLM with Databricks model endpoint and API token.
        
        Args:
            model_endpoint: URL of the Databricks LLM serving endpoint
            api_token: Databricks API token for authentication
        """
        self.model_endpoint = model_endpoint
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, prompt: str, context: Optional[str] = None, 
                         max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate a response from the LLM using provided prompt and context.
        
        Args:
            prompt: The query or instruction for the LLM
            context: Optional context to include with the prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            Generated text response
        """
        # Construct a prompt that includes context if provided
        full_prompt = prompt
        if context:
            full_prompt = f"Context information:\n{context}\n\nQuery: {prompt}\n\nResponse:"
        
        # Prepare the payload for Databricks model serving
        payload = {
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                self.model_endpoint,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            # The exact structure of the response depends on your LLM's output format
            # Adjust the parsing below based on your model's response structure
            generated_text = result.get("predictions", [""])[0]
            
            return generated_text
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            raise


class ChromaDBManager:
    """Class to handle interactions with ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
    
    def create_collection(self, collection_name: str):
        """
        Create a new collection or get existing one.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            ChromaDB collection object
        """
        return self.client.get_or_create_collection(name=collection_name)
    
    def add_documents(self, collection, documents: List[str], 
                      embeddings: List[List[float]], 
                      metadatas: List[Dict[str, Any]] = None,
                      ids: List[str] = None):
        """
        Add documents and their embeddings to a collection.
        
        Args:
            collection: ChromaDB collection
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of IDs for each document
        """
        if ids is None:
            # Generate simple IDs based on position
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadatas is None:
            # Create empty metadata for each document
            metadatas = [{} for _ in documents]
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def query_collection(self, collection, query_embedding: List[float], 
                        n_results: int = 5) -> Dict:
        """
        Query the collection for similar documents.
        
        Args:
            collection: ChromaDB collection
            query_embedding: Embedding vector of the query
            n_results: Number of results to return
            
        Returns:
            Dictionary with query results
        """
        return collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline using Databricks and ChromaDB."""
    
    def __init__(self, 
                embedding_endpoint: str, 
                llm_endpoint: str, 
                api_token: str,
                collection_name: str = "documents",
                persist_dir: str = "./chroma_db"):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_endpoint: URL of the Databricks embedding model endpoint
            llm_endpoint: URL of the Databricks LLM endpoint
            api_token: Databricks API token for authentication
            collection_name: Name of the ChromaDB collection
            persist_dir: Directory to persist ChromaDB data
        """
        self.embedder = DatabricksEmbedder(embedding_endpoint, api_token)
        self.llm = DatabricksLLM(llm_endpoint, api_token)
        self.chroma_manager = ChromaDBManager(persist_dir)
        self.collection_name = collection_name
        self.collection = self.chroma_manager.create_collection(collection_name)
    
    def ingest_documents(self, documents: List[str], 
                        metadatas: List[Dict[str, Any]] = None,
                        ids: List[str] = None):
        """
        Ingest documents into the RAG pipeline.
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of IDs for each document
        """
        # Generate embeddings
        embeddings = self.embedder.get_embeddings(documents)
        
        # Store in ChromaDB
        self.chroma_manager.add_documents(
            self.collection, 
            documents, 
            embeddings,
            metadatas,
            ids
        )
        
        return len(documents)
    
    def query(self, query_text: str, n_results: int = 3) -> str:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query_text: Query text
            n_results: Number of documents to retrieve
            
        Returns:
            Generated response from the LLM
        """
        # Generate embedding for the query
        query_embedding = self.embedder.get_embeddings([query_text])[0]
        
        # Query ChromaDB for relevant documents
        results = self.chroma_manager.query_collection(
            self.collection,
            query_embedding,
            n_results
        )
        
        # Extract retrieved documents
        retrieved_docs = results.get("documents", [[]])[0]
        
        # Combine retrieved documents into context
        context = "\n\n".join(retrieved_docs)
        
        # Generate response using the LLM
        response = self.llm.generate_response(query_text, context)
        
        return response


def main():
    # Configuration - replace with your actual values
    DATABRICKS_EMBEDDING_ENDPOINT = os.environ.get(
        "DATABRICKS_EMBEDDING_ENDPOINT", 
        "https://your-databricks-instance.cloud.databricks.com/model-endpoint/embedding-model/invocations"
    )
    DATABRICKS_LLM_ENDPOINT = os.environ.get(
        "DATABRICKS_LLM_ENDPOINT",
        "https://your-databricks-instance.cloud.databricks.com/model-endpoint/llm-model/invocations"
    )
    DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "your-databricks-token")
    COLLECTION_NAME = "my_documents"
    PERSIST_DIR = "./chroma_db"
    
    # Sample documents
    documents = [
        "Databricks is a data intelligence platform that offers various tools for data engineering, machine learning, and analytics.",
        "ChromaDB is an open-source vector database designed to store and search embeddings for use in AI applications.",
        "Embeddings are vector representations of data that capture semantic meaning, allowing for similarity search.",
        "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation to produce more accurate responses.",
        "Large Language Models (LLMs) are transformer-based models trained on vast amounts of text data to generate human-like text."
    ]
    
    # Create metadata for documents
    metadatas = [
        {"source": "documentation", "topic": "databricks"},
        {"source": "documentation", "topic": "vector_db"},
        {"source": "glossary", "topic": "machine_learning"},
        {"source": "research", "topic": "llm_architecture"},
        {"source": "glossary", "topic": "llm"}
    ]
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        DATABRICKS_EMBEDDING_ENDPOINT,
        DATABRICKS_LLM_ENDPOINT,
        DATABRICKS_TOKEN,
        COLLECTION_NAME,
        PERSIST_DIR
    )
    
    try:
        # Ingest documents
        print("Ingesting documents...")
        count = rag.ingest_documents(documents, metadatas)
        print(f"Ingested {count} documents into ChromaDB collection '{COLLECTION_NAME}'")
        
        # Test query
        test_query = "How do embeddings relate to RAG systems?"
        print(f"\nTest query: '{test_query}'")
        response = rag.query(test_query)
        print("\nLLM Response:")
        print(response)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
