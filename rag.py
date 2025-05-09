import os
import requests
import json
from typing import List, Optional, Any, Dict

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    Document,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.custom import CustomLLM
from llama_index.embeddings.custom import CustomEmbedding
import chromadb

class DatabricksEmbedding(CustomEmbedding):
    """
    Custom embedding class for Databricks embedding model serving
    """
    def __init__(
        self,
        databricks_host: str,
        model_endpoint: str,
        token: str,
        embed_batch_size: int = 10,
    ):
        """
        Initialize the Databricks embedding model
        
        Args:
            databricks_host: Databricks workspace host URL
            model_endpoint: Name of the model serving endpoint
            token: Databricks access token
            embed_batch_size: Batch size for embedding requests
        """
        self.databricks_host = databricks_host
        self.model_endpoint = model_endpoint
        self.token = token
        self.embed_batch_size = embed_batch_size
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.api_url = f"{self.databricks_host}/serving-endpoints/{self.model_endpoint}/invocations"
    
    def _get_embeddings_from_databricks(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from Databricks model serving endpoint
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            payload = {"inputs": texts}
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Assuming the response format is {"embeddings": [[0.1, 0.2, ...], ...]}
            result = response.json()
            if "embeddings" in result:
                return result["embeddings"]
            else:
                # Adapt to different response formats if needed
                return result
                
        except Exception as e:
            print(f"Error getting embeddings from Databricks: {e}")
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_embeddings_from_databricks([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_embeddings_from_databricks([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._get_embeddings_from_databricks(texts)


class DatabricksLLM(CustomLLM):
    """
    Custom LLM class for Databricks LLM model serving
    """
    def __init__(
        self,
        databricks_host: str,
        model_endpoint: str,
        token: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        """
        Initialize the Databricks LLM
        
        Args:
            databricks_host: Databricks workspace host URL
            model_endpoint: Name of the model serving endpoint
            token: Databricks access token
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.databricks_host = databricks_host
        self.model_endpoint = model_endpoint
        self.token = token
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.api_url = f"{self.databricks_host}/serving-endpoints/{self.model_endpoint}/invocations"
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get LLM metadata."""
        return {
            "model_name": f"databricks-{self.model_endpoint}",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
    
    def complete(self, prompt: str, **kwargs) -> Any:
        """
        Complete a prompt using the Databricks LLM
        """
        try:
            # Adapt this payload structure based on your specific model's requirements
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                }
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            # Adjust the response parsing based on your model's output format
            # This is an example and might need to be modified
            if isinstance(result, dict) and "predictions" in result:
                return result["predictions"][0]
            elif isinstance(result, list):
                return result[0]
            else:
                return str(result)
                
        except Exception as e:
            print(f"Error completing prompt with Databricks LLM: {e}")
            raise

    def stream_complete(self, prompt: str, **kwargs) -> Any:
        """
        Stream complete a prompt (if your endpoint supports streaming)
        """
        # This is a simplified implementation that doesn't actually stream
        # Implement real streaming if your Databricks endpoint supports it
        return self.complete(prompt, **kwargs)


def create_chroma_index(
    documents_dir: str,
    databricks_host: str,
    embedding_endpoint: str,
    llm_endpoint: str,
    access_token: str,
    collection_name: str = "llamaindex_collection",
    persist_dir: str = "./chroma_db"
):
    """
    Create a ChromaDB vector store index from documents using Databricks models
    
    Args:
        documents_dir: Directory containing documents to index
        databricks_host: Databricks workspace host URL
        embedding_endpoint: Name of the embedding model serving endpoint
        llm_endpoint: Name of the LLM model serving endpoint
        access_token: Databricks access token
        collection_name: Name of the ChromaDB collection
        persist_dir: Directory to persist the ChromaDB data
    
    Returns:
        The LlamaIndex VectorStoreIndex object
    """
    # Initialize Databricks models
    embed_model = DatabricksEmbedding(
        databricks_host=databricks_host,
        model_endpoint=embedding_endpoint,
        token=access_token
    )
    
    llm_model = DatabricksLLM(
        databricks_host=databricks_host,
        model_endpoint=llm_endpoint,
        token=access_token
    )
    
    # Configure global settings
    Settings.embed_model = embed_model
    Settings.llm = llm_model
    Settings.node_parser = SentenceSplitter(chunk_size=1024)
    
    # Create ChromaDB client and collection
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(collection_name)
    
    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Load documents
    documents = SimpleDirectoryReader(documents_dir).load_data()
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    
    return index


def query_index(index, query_text, similarity_top_k=3):
    """
    Query the index
    
    Args:
        index: LlamaIndex VectorStoreIndex
        query_text: Query string
        similarity_top_k: Number of top results to retrieve
        
    Returns:
        Response from the query engine
    """
    # Create query engine
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
    
    # Get response
    response = query_engine.query(query_text)
    
    return response


def main():
    # Databricks configuration
    databricks_host = "https://your-databricks-workspace.cloud.databricks.com"  # Replace with your Databricks host
    access_token = "your-databricks-access-token"  # Replace with your access token
    embedding_endpoint = "embedding-model-endpoint"  # Replace with your embedding model endpoint name
    llm_endpoint = "llm-model-endpoint"  # Replace with your LLM endpoint name
    
    # Path to your documents
    documents_dir = "./documents"
    
    # Create index
    index = create_chroma_index(
        documents_dir=documents_dir,
        databricks_host=databricks_host,
        embedding_endpoint=embedding_endpoint,
        llm_endpoint=llm_endpoint,
        access_token=access_token
    )
    
    # Sample query
    query_text = "What are the main points in these documents?"
    response = query_index(index, query_text)
    
    print(f"Query: {query_text}")
    print(f"Response: {response}")
    
    # You can also persist the index for later use
    # index.storage_context.persist()


if __name__ == "__main__":
    main()
