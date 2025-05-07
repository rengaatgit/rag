import os
import requests
import json
from typing import List, Dict, Any

import chromadb
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.vector_stores import ChromaVectorStore
from llama_index.embeddings import BaseEmbedding
from llama_index.llms import BaseLLM, CompletionResponse, LLMMetadata
from llama_index.callbacks import CallbackManager


# 1. Custom Embedding Model using Databricks Endpoint
class DatabricksEmbeddingModel(BaseEmbedding):
    """Custom embedding model that uses a Databricks model serving endpoint."""
    
    def __init__(
        self,
        endpoint_url: str,
        api_token: str,
        model_name: str = "embedding-model",
        embed_batch_size: int = 10
    ):
        self.endpoint_url = endpoint_url
        self.api_token = api_token
        self.model_name = model_name
        self.embed_batch_size = embed_batch_size
        super().__init__()
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query."""
        return self._get_text_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text."""
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": text
        }
        
        response = requests.post(self.endpoint_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Error from Databricks API: {response.text}")
        
        embedding = response.json()["embeddings"]
        return embedding
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i+self.embed_batch_size]
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": batch
            }
            
            response = requests.post(self.endpoint_url, headers=headers, json=payload)
            
            if response.status_code != 200:
                raise Exception(f"Error from Databricks API: {response.text}")
            
            batch_embeddings = response.json()["embeddings"]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings


# 2. Custom LLM using Databricks Model Serving Endpoint
class DatabricksLLM(BaseLLM):
    """Custom LLM that uses a Databricks model serving endpoint."""
    
    def __init__(
        self,
        endpoint_url: str,
        api_token: str,
        model_name: str = "llm-model",
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        self.endpoint_url = endpoint_url
        self.api_token = api_token
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        super().__init__()
    
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.model_name,
            max_tokens=self.max_tokens
        )
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Complete a prompt."""
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        response = requests.post(self.endpoint_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Error from Databricks API: {response.text}")
        
        completion = response.json()["completion"]
        return CompletionResponse(text=completion)
    
    def stream_complete(self, prompt: str, **kwargs):
        """Stream completion not implemented for this example."""
        raise NotImplementedError("Streaming completion not implemented")


# 3. Main RAG Pipeline
class RAGPipeline:
    """Main RAG pipeline using LlamaIndex, ChromaDB, and Databricks endpoints."""
    
    def __init__(
        self,
        embedding_endpoint_url: str,
        embedding_api_token: str,
        llm_endpoint_url: str,
        llm_api_token: str,
        chroma_persist_dir: str = "./chroma_db",
        index_persist_dir: str = "./storage",
        collection_name: str = "document_collection"
    ):
        # Initialize embedding model
        self.embedding_model = DatabricksEmbeddingModel(
            endpoint_url=embedding_endpoint_url,
            api_token=embedding_api_token
        )
        
        # Initialize LLM
        self.llm = DatabricksLLM(
            endpoint_url=llm_endpoint_url,
            api_token=llm_api_token
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
        try:
            self.chroma_collection = self.chroma_client.get_collection(collection_name)
            print(f"Using existing collection: {collection_name}")
        except:
            self.chroma_collection = self.chroma_client.create_collection(collection_name)
            print(f"Created new collection: {collection_name}")
        
        # Initialize vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        # Initialize storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            persist_dir=index_persist_dir
        )
        
        # Initialize service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embedding_model,
            callback_manager=CallbackManager([])
        )
        
        # Initialize index
        self.index = None
        self.index_persist_dir = index_persist_dir
        self.collection_name = collection_name
    
    def ingest_documents(self, document_dir: str):
        """Ingest documents from a directory."""
        try:
            # Load existing index if available
            self.index = load_index_from_storage(
                storage_context=self.storage_context,
                service_context=self.service_context
            )
            print("Loaded existing index")
        except:
            # Create new index if not available
            documents = SimpleDirectoryReader(document_dir).load_data()
            print(f"Loaded {len(documents)} documents")
            
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                service_context=self.service_context
            )
            
            # Persist index
            self.index.storage_context.persist(persist_dir=self.index_persist_dir)
            print(f"Created and persisted new index")
    
    def query(self, question: str, num_results: int = 3) -> Dict[str, Any]:
        """Query the RAG pipeline."""
        if self.index is None:
            raise ValueError("Index not initialized. Call ingest_documents first.")
        
        # Create query engine
        query_engine = self.index.as_query_engine(
            similarity_top_k=num_results
        )
        
        # Get response
        response = query_engine.query(question)
        
        # Format result
        result = {
            "answer": response.response,
            "source_documents": []
        }
        
        # Add source documents if available
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                result["source_documents"].append({
                    "text": node.node.text,
                    "score": node.score,
                    "metadata": node.node.metadata
                })
        
        return result


# 4. Example usage
def main():
    # Environment variables (for production, use proper secret management)
    embedding_endpoint_url = os.environ.get("DATABRICKS_EMBEDDING_ENDPOINT")
    embedding_api_token = os.environ.get("DATABRICKS_EMBEDDING_TOKEN")
    llm_endpoint_url = os.environ.get("DATABRICKS_LLM_ENDPOINT")
    llm_api_token = os.environ.get("DATABRICKS_LLM_TOKEN")
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        embedding_endpoint_url=embedding_endpoint_url,
        embedding_api_token=embedding_api_token,
        llm_endpoint_url=llm_endpoint_url,
        llm_api_token=llm_api_token
    )
    
    # Ingest documents
    rag.ingest_documents("./documents")
    
    # Query
    question = "What is the capital of France?"
    result = rag.query(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    
    if result["source_documents"]:
        print("Sources:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"  {i+1}. {doc['text'][:100]}... (Score: {doc['score']})")


if __name__ == "__main__":
    main()