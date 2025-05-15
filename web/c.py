"""
RAG Pipeline with LlamaIndex and Databricks Models

This script implements a RAG pipeline that:
1. Processes website URLs
2. Creates embeddings using Databricks text-embeddings-3-large
3. Stores embeddings locally
4. Provides a query interface using Databricks gpt-4o
"""

import os
import json
import asyncio
import nest_asyncio
from typing import List, Set, Dict, Any, Optional
import logging
from pathlib import Path

# LlamaIndex imports
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.databricks import DatabricksLLM
from llama_index.embeddings.databricks import DatabricksEmbedding
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow for nested event loops (needed for async operations)
nest_asyncio.apply()

class RAGPipeline:
    """RAG Pipeline using LlamaIndex and Databricks models."""
    
    def __init__(self, 
                 storage_dir: str = "storage",
                 databricks_host: Optional[str] = None,
                 databricks_token: Optional[str] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            storage_dir: Directory to store processed data and indices
            databricks_host: Databricks host URL (defaults to env var DATABRICKS_HOST)
            databricks_token: Databricks API token (defaults to env var DATABRICKS_TOKEN)
        """
        # Storage settings
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        # Track processed URLs
        self.processed_urls_file = self.storage_dir / "processed_urls.json"
        self.processed_urls = self._load_processed_urls()
        
        # Initialize Databricks credentials
        self.databricks_host = databricks_host or os.environ.get("DATABRICKS_HOST")
        self.databricks_token = databricks_token or os.environ.get("DATABRICKS_TOKEN")
        
        if not self.databricks_host or not self.databricks_token:
            raise ValueError("Databricks host and token must be provided either as parameters or environment variables")
        
        # Initialize LlamaIndex components
        self._initialize_llama_index()
        
    def _initialize_llama_index(self):
        """Initialize LlamaIndex components with Databricks models."""
        # Set up Databricks embeddings (text-embeddings-3-large)
        embed_model = DatabricksEmbedding(
            host=self.databricks_host,
            token=self.databricks_token,
            endpoint_name="databricks-bge-large-en", # Using BGE as an example, adjust as needed
        )
        
        # Set up Databricks LLM (gpt-4o)
        llm = DatabricksLLM(
            host=self.databricks_host,
            token=self.databricks_token,
            endpoint_name="databricks-gpt-4o", # Adjust as needed for your endpoint
            model_kwargs={"temperature": 0.1, "max_tokens": 1024}
        )
        
        # Configure optimal chunking parameters
        # These can be tuned based on your specific use case
        node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200,
        )
        
        # Update global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser
        
        # Initialize storage context
        self.storage_context = StorageContext.from_defaults(
            persist_dir=str(self.storage_dir / "index_storage")
        )
    
    def _load_processed_urls(self) -> Set[str]:
        """Load set of previously processed URLs."""
        if self.processed_urls_file.exists():
            with open(self.processed_urls_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def _save_processed_urls(self):
        """Save processed URLs to disk."""
        with open(self.processed_urls_file, 'w') as f:
            json.dump(list(self.processed_urls), f)
    
    async def process_urls_async(self, urls_file: str) -> VectorStoreIndex:
        """
        Process URLs from a file asynchronously, create embeddings, and build index.
        
        Args:
            urls_file: Path to text file containing URLs (one per line)
            
        Returns:
            VectorStoreIndex: The created vector index
        """
        # Load URLs from file
        with open(urls_file, 'r') as f:
            all_urls = [url.strip() for url in f.readlines() if url.strip()]
        
        # Filter out already processed URLs
        new_urls = [url for url in all_urls if url not in self.processed_urls]
        
        if not new_urls:
            logger.info("No new URLs to process")
            # Load existing index if available
            try:
                index = VectorStoreIndex.from_storage(self.storage_context)
                logger.info("Loaded existing index")
                return index
            except Exception as e:
                logger.error(f"Error loading existing index: {e}")
                raise
        
        logger.info(f"Processing {len(new_urls)} new URLs")
        
        # Process URLs in batches for better performance
        batch_size = 5  # Adjust based on your needs
        batches = [new_urls[i:i+batch_size] for i in range(0, len(new_urls), batch_size)]
        
        all_documents = []
        
        for batch_idx, url_batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
            
            try:
                # Use SimpleWebPageReader to load pages in batch
                documents = SimpleWebPageReader().load_data(urls=url_batch)
                all_documents.extend(documents)
                
                # Mark these URLs as processed
                self.processed_urls.update(url_batch)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx+1}: {e}")
                # Continue with next batch even if this one fails
        
        # Save updated processed URLs list
        self._save_processed_urls()
        
        if not all_documents:
            logger.warning("No documents were successfully processed")
            # Try to load existing index
            try:
                index = VectorStoreIndex.from_storage(self.storage_context)
                return index
            except Exception:
                raise ValueError("No documents processed and no existing index found")
        
        # Create ingestion pipeline with optimal settings
        ingestion_pipeline = IngestionPipeline(
            transformations=[Settings.node_parser],
            vector_storage=self.storage_context.vector_store,
        )
        
        # Process documents through the pipeline
        nodes = await ingestion_pipeline.arun(all_documents)
        
        # Create and persist the index
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
        )
        
        # Persist index to disk
        self.storage_context.persist()
        
        logger.info(f"Successfully processed {len(all_documents)} documents")
        return index
    
    def process_urls(self, urls_file: str) -> VectorStoreIndex:
        """
        Synchronous wrapper for process_urls_async.
        
        Args:
            urls_file: Path to text file containing URLs (one per line)
            
        Returns:
            VectorStoreIndex: The created vector index
        """
        return asyncio.run(self.process_urls_async(urls_file))
    
    def load_index(self) -> VectorStoreIndex:
        """
        Load index from disk if available.
        
        Returns:
            VectorStoreIndex: The loaded vector index
        """
        try:
            return VectorStoreIndex.from_storage(self.storage_context)
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise ValueError("No index found. Please process URLs first.")
    
    def query(self, query_text: str, index: Optional[VectorStoreIndex] = None) -> str:
        """
        Query the RAG system.
        
        Args:
            query_text: The query string
            index: Optional vector index (will load from disk if not provided)
            
        Returns:
            str: The response from the LLM
        """
        if index is None:
            index = self.load_index()
        
        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            # Adjust these parameters based on your needs
            response_mode="compact",
        )
        
        # Execute query
        response = query_engine.query(query_text)
        
        return str(response)


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline(
        storage_dir="./rag_storage",
        # Add your Databricks credentials or use environment variables
    )
    
    # Process URLs
    urls_file = "sample_urls.txt"
    index = pipeline.process_urls(urls_file)
    
    # Query example
    query = "What is the main topic of these websites?"
    response = pipeline.query(query, index)
    print(f"Query: {query}")
    print(f"Response: {response}")
    
    
    """
Test script for the RAG Pipeline
"""

import os
from rag_pipeline import RAGPipeline

# Create a sample URLs file
def create_sample_urls_file(filename="sample_urls.txt"):
    """Create a sample URLs file with some example websites."""
    urls = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://www.databricks.com/blog/llm-inference-retrieval-augmented-generation-databricks",
        "https://www.llamaindex.ai/blog/using-llama-index-for-rag",
        "https://python.langchain.com/docs/use_cases/question_answering/",
        "https://www.pinecone.io/learn/retrieval-augmented-generation/"
    ]
    
    with open(filename, "w") as f:
        for url in urls:
            f.write(f"{url}\n")
    
    print(f"Created sample URLs file: {filename}")
    return filename

def main():
    # Set up environment variables for Databricks
    # In a real scenario, you would set these securely or pass them as parameters
    os.environ["DATABRICKS_HOST"] = "https://your-databricks-instance.cloud.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "your-databricks-token"
    
    # Create sample URLs file
    urls_file = create_sample_urls_file()
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(storage_dir="./rag_storage")
    
    # Process URLs and create index
    print("Processing URLs and creating index...")
    index = pipeline.process_urls(urls_file)
    print("Index created successfully!")
    
    # Run some sample queries
    sample_queries = [
        "What is retrieval-augmented generation?",
        "How does LlamaIndex help with RAG applications?",
        "What are the benefits of RAG compared to traditional LLM approaches?",
        "How does Databricks integrate with RAG systems?"
    ]
    
    print("\n--- Testing Queries ---")
    for query in sample_queries:
        print(f"\nQuery: {query}")
        response = pipeline.query(query, index)
        print(f"Response: {response}")
    
    # Test loading existing index
    print("\n--- Testing Loading Existing Index ---")
    loaded_index = pipeline.load_index()
    query = "What are the main components of a RAG system?"
    print(f"\nQuery: {query}")
    response = pipeline.query(query, loaded_index)
    print(f"Response: {response}")
    
    # Add a new URL to test handling of already processed URLs
    print("\n--- Testing Adding New URLs ---")
    with open(urls_file, "a") as f:
        f.write("https://huggingface.co/blog/retrieval-augmented-generation\n")
    
    # Process the updated URLs file
    print("Processing updated URLs file...")
    updated_index = pipeline.process_urls(urls_file)
    
    # Test query with the updated index
    query = "What does Hugging Face say about RAG?"
    print(f"\nQuery: {query}")
    response = pipeline.query(query, updated_index)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
    
    
    """
URL processing utility for the RAG Pipeline
This utility provides enhanced URL handling capabilities:
- URL validation
- Content type detection
- Error handling and retry logic
- Progress tracking
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from urllib.parse import urlparse

import aiohttp
import tqdm
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class URLProcessor:
    """Advanced URL processing utility for the RAG pipeline."""
    
    def __init__(self, storage_dir: str = "storage"):
        """
        Initialize the URL processor.
        
        Args:
            storage_dir: Directory to store URL processing metadata
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        self.url_metadata_file = self.storage_dir / "url_metadata.json"
        self.url_metadata = self._load_url_metadata()
    
    def _load_url_metadata(self) -> Dict:
        """Load URL metadata from disk."""
        if self.url_metadata_file.exists():
            with open(self.url_metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_url_metadata(self):
        """Save URL metadata to disk."""
        with open(self.url_metadata_file, 'w') as f:
            json.dump(self.url_metadata, f, indent=2)
    
    def is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid.
        
        Args:
            url: The URL to validate
            
        Returns:
            bool: True if URL is valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme in ('http', 'https'), result.netloc])
        except Exception:
            return False
    
    async def check_url_accessibility(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a URL is accessible.
        
        Args:
            url: The URL to check
            
        Returns:
            Tuple[bool, Optional[str]]: (is_accessible, content_type)
        """
        timeout = aiohttp.ClientTimeout(total=30)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.head(url, allow_redirects=True) as response:
                    is_accessible = response.status == 200
                    content_type = response.headers.get('Content-Type', '').lower()
                    return is_accessible, content_type
        except Exception as e:
            logger.warning(f"Error checking URL {url}: {e}")
            return False, None
    
    async def get_url_title(self, url: str) -> Optional[str]:
        """
        Get the title of a webpage.
        
        Args:
            url: The URL to get the title for
            
        Returns:
            Optional[str]: The page title if available
        """
        timeout = aiohttp.ClientTimeout(total=30)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        title = soup.title.string if soup.title else None
                        return title
        except Exception as e:
            logger.warning(f"Error getting title for URL {url}: {e}")
        
        return None
    
    async def analyze_url(self, url: str) -> Dict:
        """
        Analyze a URL and get metadata.
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dict: URL metadata
        """
        # Check if we already have metadata for this URL
        if url in self.url_metadata:
            return self.url_metadata[url]
        
        metadata = {
            "url": url,
            "valid": self.is_valid_url(url),
            "last_checked": time.time(),
            "accessible": False,
            "content_type": None,
            "title": None,
            "processed": False,
            "processing_attempts": 0,
            "last_processing_time": None,
        }
        
        if metadata["valid"]:
            is_accessible, content_type = await self.check_url_accessibility(url)
            metadata["accessible"] = is_accessible
            metadata["content_type"] = content_type
            
            if is_accessible and content_type and 'text/html' in content_type:
                title = await self.get_url_title(url)
                metadata["title"] = title
        
        # Store metadata
        self.url_metadata[url] = metadata
        self._save_url_metadata()
        
        return metadata
    
    async def process_urls_from_file(self, file_path: str) -> Tuple[List[str], List[str]]:
        """
        Process URLs from a file, validating and gathering metadata.
        
        Args:
            file_path: Path to the text file containing URLs
            
        Returns:
            Tuple[List[str], List[str]]: (valid_urls, invalid_urls)
        """
        # Read URLs from file
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        valid_urls = []
        invalid_urls = []
        
        # Process URLs with progress bar
        for url in tqdm.tqdm(urls, desc="Analyzing URLs"):
            metadata = await self.analyze_url(url)
            
            if metadata["valid"] and metadata["accessible"]:
                valid_urls.append(url)
            else:
                invalid_urls.append(url)
        
        # Log results
        logger.info(f"URL analysis complete: {len(valid_urls)} valid, {len(invalid_urls)} invalid")
        
        # Save metadata
        self._save_url_metadata()
        
        return valid_urls, invalid_urls
    
    def mark_url_as_processed(self, url: str, success: bool = True):
        """
        Mark a URL as processed.
        
        Args:
            url: The URL to mark
            success: Whether processing was successful
        """
        if url in self.url_metadata:
            self.url_metadata[url]["processed"] = success
            self.url_metadata[url]["processing_attempts"] += 1
            self.url_metadata[url]["last_processing_time"] = time.time()
            self._save_url_metadata()
    
    def get_unprocessed_urls(self, urls: List[str]) -> List[str]:
        """
        Filter list to only include unprocessed URLs.
        
        Args:
            urls: List of URLs to filter
            
        Returns:
            List[str]: Unprocessed URLs
        """
        return [url for url in urls if url not in self.url_metadata or not self.url_metadata[url]["processed"]]
    
    def generate_processing_report(self) -> Dict:
        """
        Generate a report of URL processing status.
        
        Returns:
            Dict: Processing report
        """
        total_urls = len(self.url_metadata)
        valid_urls = sum(1 for meta in self.url_metadata.values() if meta["valid"])
        accessible_urls = sum(1 for meta in self.url_metadata.values() if meta["accessible"])
        processed_urls = sum(1 for meta in self.url_metadata.values() if meta["processed"])
        
        content_types = {}
        for meta in self.url_metadata.values():
            if meta["content_type"]:
                content_type = meta["content_type"].split(';')[0]  # Remove charset part
                content_types[content_type] = content_types.get(content_type, 0) + 1
        
        return {
            "total_urls": total_urls,
            "valid_urls": valid_urls,
            "accessible_urls": accessible_urls,
            "processed_urls": processed_urls,
            "content_types": content_types,
        }


# Example usage
if __name__ == "__main__":
    async def main():
        processor = URLProcessor(storage_dir="./url_storage")
        
        # Create a sample file
        with open("test_urls.txt", "w") as f:
            f.write("https://www.example.com\n")
            f.write("https://www.python.org\n")
            f.write("invalid-url\n")
            f.write("https://nonexistentwebsite123456789.org\n")
        
        # Process URLs
        valid_urls, invalid_urls = await processor.process_urls_from_file("test_urls.txt")
        
        print(f"Valid URLs: {valid_urls}")
        print(f"Invalid URLs: {invalid_urls}")
        
        # Generate report
        report = processor.generate_processing_report()
        print(f"Processing report: {json.dumps(report, indent=2)}")
    
    # Run the async main function
    asyncio.run(main())