"""
LlamaIndex 0.12.34 with ChromaDB 1.0.8 and Databricks integration for embeddings and LLM
- Uses Databricks serving endpoints for both embeddings and LLM models
- Authenticates with Databricks access token
- Sets up a simple document indexing and query system
"""

import os
import requests
import json
from typing import List, Optional, Callable

# Install required packages
# pip install llama-index==0.12.34 chromadb==1.0.8 requests

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


class DatabricksEmbedding(BaseEmbedding):
    """Custom embedding class for Databricks serving endpoint."""
    
    def __init__(
        self,
        endpoint_url: str,
        access_token: str,
        model_name: str = "databricks-embedding",
        embed_batch_size: int = 10,
    ):
        """Initialize DatabricksEmbedding.
        
        Args:
            endpoint_url: URL of the Databricks serving endpoint for embeddings
            access_token: Databricks access token for authentication
            model_name: Name of the embedding model (for reference only)
            embed_batch_size: Batch size for embedding requests
        """
        super().__init__()
        self.endpoint_url = endpoint_url
        self.access_token = access_token
        self.model_name = model_name
        self.embed_batch_size = embed_batch_size
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    def _get_embeddings_from_databricks(self, texts: List[str]) -> List[List[float]]:
        """Send request to Databricks endpoint to get embeddings."""
        payload = {
            "dataframe_records": [{"text": text} for text in texts]
        }
        
        response = requests.post(
            self.endpoint_url,
            headers=self.headers,
            data=json.dumps(payload)
        )
        
        if response.status_code != 200:
            raise Exception(f"Error from Databricks API: {response.status_code}, {response.text}")
        
        embeddings = response.json()["predictions"]
        return embeddings
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query string."""
        return self._get_embeddings_from_databricks([query])[0]
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string."""
        return self._get_embeddings_from_databricks([text])[0]
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        results = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i+self.embed_batch_size]
            results.extend(self._get_embeddings_from_databricks(batch))
        return results
        
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of get_query_embedding."""
        # For simplicity, we're using the sync version since Databricks endpoints
        # don't natively support async calls without additional complexity
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of get_text_embedding."""
        return self._get_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async version of get_text_embeddings."""
        return self._get_text_embeddings(texts)


class DatabricksLLM(LLM):
    """Custom LLM class for Databricks serving endpoint."""
    
    def __init__(
        self,
        endpoint_url: str,
        access_token: str,
        model_name: str = "databricks-llm",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        """Initialize DatabricksLLM.
        
        Args:
            endpoint_url: URL of the Databricks serving endpoint for LLM
            access_token: Databricks access token for authentication
            model_name: Name of the LLM model (for reference only)
            temperature: Temperature parameter for LLM generation
            max_tokens: Maximum tokens to generate
        """
        self.endpoint_url = endpoint_url
        self.access_token = access_token
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.model_name,
            max_input_size=4096,  # Adjust based on your model's capabilities
            num_output=self.max_tokens,
            context_window=4096,  # Adjust based on your model's capabilities
            is_chat_model=True,
        )
    
    def _completion_with_retry(self, prompt: str) -> str:
        """Send completion request to Databricks endpoint with retry logic."""
        payload = {
            "dataframe_records": [{
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }]
        }
        
        response = requests.post(
            self.endpoint_url,
            headers=self.headers,
            data=json.dumps(payload)
        )
        
        if response.status_code != 200:
            raise Exception(f"Error from Databricks API: {response.status_code}, {response.text}")
        
        result = response.json()["predictions"][0]
        
        # The actual response format may vary based on your model endpoint
        # Adjust this extraction logic as needed
        if isinstance(result, dict) and "text" in result:
            return result["text"]
        elif isinstance(result, str):
            return result
        else:
            return str(result)
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Complete a prompt."""
        response_text = self._completion_with_retry(prompt)
        return CompletionResponse(text=response_text)
    
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        """Stream complete a prompt (not implemented in this basic example).
        For a real implementation, you would need to check if the Databricks
        endpoint supports streaming and implement accordingly.
        """
        # This is a simplified version that doesn't actually stream
        response_text = self._completion_with_retry(prompt)
        
        def gen():
            yield CompletionResponse(text=response_text)
            
        return gen()
        
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Async version of complete."""
        # For simplicity, we're using the sync version
        return self.complete(prompt, **kwargs)
    
    async def astream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        """Async version of stream_complete."""
        return self.stream_complete(prompt, **kwargs)


def setup_llama_index_with_databricks(
    documents_dir: str,
    databricks_embedding_endpoint: str,
    databricks_llm_endpoint: str,
    databricks_token: str,
    chroma_persist_dir: str = "./chroma_db",
    collection_name: str = "document_collection"
):
    """Set up LlamaIndex with Databricks and ChromaDB.
    
    Args:
        documents_dir: Directory containing documents to index
        databricks_embedding_endpoint: Databricks endpoint URL for embeddings
        databricks_llm_endpoint: Databricks endpoint URL for LLM
        databricks_token: Databricks access token
        chroma_persist_dir: Directory to persist ChromaDB
        collection_name: ChromaDB collection name
    
    Returns:
        index: LlamaIndex vector store index
    """
    # Create custom embedding and LLM models
    embedding_model = DatabricksEmbedding(
        endpoint_url=databricks_embedding_endpoint,
        access_token=databricks_token
    )
    
    llm_model = DatabricksLLM(
        endpoint_url=databricks_llm_endpoint,
        access_token=databricks_token
    )
    
    # Set as default for llama-index
    Settings.embed_model = embedding_model
    Settings.llm = llm_model
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
    
    # Create or get collection
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Load documents
    if documents_dir:
        documents = SimpleDirectoryReader(documents_dir).load_data()
    else:
        # Example document if no directory is provided
        documents = [Document(text="This is a sample document for testing LlamaIndex with Databricks integration.")]
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents=documents,
        vector_store=vector_store
    )
    
    return index


def query_index(index, query_text: str):
    """Query the index with the given text."""
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return response


if __name__ == "__main__":
    # Example usage
    DATABRICKS_EMBEDDING_ENDPOINT = "https://your-databricks-instance.cloud.databricks.com/serving-endpoints/embedding-model/invocations"
    DATABRICKS_LLM_ENDPOINT = "https://your-databricks-instance.cloud.databricks.com/serving-endpoints/llm-model/invocations"
    DATABRICKS_TOKEN = "dapi_your_access_token_here"
    
    # Directory with documents to index
    DOCUMENTS_DIR = "./documents"
    
    # Create index
    index = setup_llama_index_with_databricks(
        documents_dir=DOCUMENTS_DIR,
        databricks_embedding_endpoint=DATABRICKS_EMBEDDING_ENDPOINT,
        databricks_llm_endpoint=DATABRICKS_LLM_ENDPOINT,
        databricks_token=DATABRICKS_TOKEN
    )
    
    # Example query
    query = "What are the main topics in the documents?"
    response = query_index(index, query)
    print(f"Query: {query}")
    print(f"Response: {response}")
    # Note: Ensure to replace the endpoint URLs and token with your actual Databricks credentials.
    # The above code is a simplified example and may require adjustments based on your specific use case.