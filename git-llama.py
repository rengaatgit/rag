"""
Git Log Processor using LlamaIndex 0.12.34, Databricks Model Serving, and ChromaDB

This script:
1. Parses Git log data from a JSON file
2. Creates embeddings using OpenAI models via Databricks Model Serving endpoints
3. Indexes the data in ChromaDB
4. Provides a query interface for the user

Requirements:
- pip install llama-index==0.12.34 chromadb openai requests
"""

import json
import os
from typing import List, Dict, Any

# LlamaIndex imports
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

class DatabricksOpenAIEmbedding(OpenAIEmbedding):
    """Custom OpenAI Embedding class that uses Databricks model serving endpoints."""
    
    def __init__(
        self,
        databricks_host: str,
        endpoint_name: str,
        api_token: str,
        model_name: str = "text-embedding-ada-002",
        embed_batch_size: int = 10,
    ):
        """Initialize with Databricks model serving endpoint details."""
        import openai
        
        # Initialize the base OpenAI embedding class
        super().__init__(
            model=model_name,
            embed_batch_size=embed_batch_size,
        )
        
        # Configure OpenAI client to use Databricks endpoint
        openai.api_base = f"https://{databricks_host}/serving-endpoints/{endpoint_name}/invocations"
        openai.api_key = api_token
        
        # Set custom headers for Databricks
        self.client.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    async def _get_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Override to handle Databricks specific API if needed."""
        return await super()._get_embeddings_async(texts)

def parse_git_log_json(file_path: str) -> List[Document]:
    """
    Parse the JSON file generated from git log --stat and convert to LlamaIndex Documents.
    """
    with open(file_path, 'r') as f:
        git_log_data = json.load(f)
    
    documents = []
    for commit in git_log_data:
        # Create a structured representation of the commit
        content = f"""
        Commit: {commit.get('commit', '')}
        Author: {commit.get('author', '')}
        Date: {commit.get('date', '')}
        
        Message:
        {commit.get('message', '')}
        
        Changes:
        {commit.get('changes', '')}
        
        Stats:
        {commit.get('stats', {})}
        """
        
        # Create metadata
        metadata = {
            "commit_hash": commit.get('commit', ''),
            "author": commit.get('author', ''),
            "date": commit.get('date', ''),
            "type": "git_commit"
        }
        
        # Create Document object
        doc = Document(text=content, metadata=metadata)
        documents.append(doc)
    
    return documents

def setup_chroma_db(collection_name: str = "git_log_collection") -> chromadb.Collection:
    """
    Set up and return a ChromaDB collection.
    """
    # Initialize ChromaDB client (persistent directory)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or get collection
    try:
        collection = chroma_client.get_collection(collection_name)
        print(f"Retrieved existing collection: {collection_name}")
    except:
        collection = chroma_client.create_collection(collection_name)
        print(f"Created new collection: {collection_name}")
    
    return collection

def index_documents(
    documents: List[Document],
    embedding_model: DatabricksOpenAIEmbedding,
    collection_name: str = "git_log_collection"
) -> VectorStoreIndex:
    """
    Process the documents, generate embeddings, and store in ChromaDB.
    """
    # Set up node parser
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    
    # Set the embedding model in Settings
    Settings.embed_model = embedding_model
    
    # Set up ChromaDB
    chroma_collection = setup_chroma_db(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create index from the documents
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        transformations=[node_parser]
    )
    
    return index

def query_git_log(
    query: str,
    index: VectorStoreIndex,
    similarity_top_k: int = 5
) -> Dict[str, Any]:
    """
    Query the git log using the provided index.
    """
    # Create a query engine
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
    
    # Execute query
    response = query_engine.query(query)
    
    return {
        "response": str(response),
        "source_nodes": [
            {
                "text": node.node.text,
                "metadata": node.node.metadata,
                "score": node.score
            }
            for node in response.source_nodes
        ]
    }

def main():
    # Configuration
    DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "your-workspace.cloud.databricks.com")
    ENDPOINT_NAME = os.environ.get("DATABRICKS_ENDPOINT_NAME", "openai-embedding-endpoint")
    API_TOKEN = os.environ.get("DATABRICKS_API_TOKEN", "your-token")
    GIT_LOG_FILE = os.environ.get("GIT_LOG_FILE", "git_log.json")
    
    # Initialize the Databricks OpenAI Embedding
    embedding_model = DatabricksOpenAIEmbedding(
        databricks_host=DATABRICKS_HOST,
        endpoint_name=ENDPOINT_NAME,
        api_token=API_TOKEN,
        model_name="text-embedding-ada-002",
    )
    
    # Parse git log data
    print(f"Parsing git log data from {GIT_LOG_FILE}...")
    documents = parse_git_log_json(GIT_LOG_FILE)
    print(f"Parsed {len(documents)} commit records.")
    
    # Index the documents
    print("Indexing documents in ChromaDB...")
    index = index_documents(documents, embedding_model)
    print("Indexing complete!")
    
    # Interactive query loop
    print("\nGit Log Query Interface")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("Enter your query: ")
        if query.lower() == 'exit':
            break
        
        print("\nSearching...")
        try:
            results = query_git_log(query, index)
            
            print("\n" + "="*50)
            print("RESPONSE:")
            print(results["response"])
            print("\n" + "="*50)
            print(f"Found {len(results['source_nodes'])} related commits")
            
            # Display source details if requested
            show_sources = input("\nShow source details? (y/n): ")
            if show_sources.lower() == 'y':
                for i, node in enumerate(results['source_nodes']):
                    print(f"\nSource {i+1} (Score: {node['score']:.4f}):")
                    print(f"Commit: {node['metadata'].get('commit_hash', 'N/A')}")
                    print(f"Author: {node['metadata'].get('author', 'N/A')}")
                    print(f"Date: {node['metadata'].get('date', 'N/A')}")
                    print("-" * 40)
                    print(node['text'][:300] + "..." if len(node['text']) > 300 else node['text'])
        
        except Exception as e:
            print(f"Error processing query: {str(e)}")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()