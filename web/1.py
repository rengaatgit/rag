pip install llama-index==0.12.0
pip install llama-index-readers-web
pip install llama-index-embeddings-databricks
pip install llama-index-llms-databricks
pip install html2text
pip install aiohttp


import os
import json
import asyncio
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# LlamaIndex imports
from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.web import SimpleWebPageReader
from llama_index.embeddings.databricks import DatabricksEmbedding
from llama_index.llms.databricks import Databricks

import aiohttp
import html2text

class AsyncWebPageReader:
    """Asynchronous web page reader for efficient URL processing."""
    
    @staticmethod
    async def load_pages(urls: List[str]) -> List[Tuple[str, str]]:
        """Load multiple web pages asynchronously."""
        async def fetch_url(session, url):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        # Convert HTML to text
                        h = html2text.HTML2Text()
                        h.ignore_links = False
                        text = h.handle(html)
                        return url, text
                    else:
                        print(f"Error fetching {url}: HTTP {response.status}")
                        return url, None
            except Exception as e:
                print(f"Exception fetching {url}: {str(e)}")
                return url, None
        
        results = []
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_url(session, url) for url in urls]
            responses = await asyncio.gather(*tasks)
            
            for url, content in responses:
                if content:
                    results.append((url, content))
        
        return results

class RAGPipeline:
    def __init__(
        self,
        databricks_token: str,
        databricks_endpoint: str,
        embeddings_model: str = "databricks-txt-embeddings-3-large",
        llm_model: str = "databricks-gpt-4o", 
        storage_dir: str = "./storage",
        processed_urls_file: str = "./processed_urls.json",
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
    ):
        self.databricks_token = databricks_token
        self.databricks_endpoint = databricks_endpoint
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model
        self.storage_dir = Path(storage_dir)
        self.processed_urls_file = Path(processed_urls_file)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processed URLs tracker
        self.processed_urls = self._load_processed_urls()
        
        # Initialize Databricks components and LlamaIndex settings
        self._setup_llama_index()
    
    def _load_processed_urls(self) -> Dict[str, Dict[str, Any]]:
        """Load record of previously processed URLs."""
        if self.processed_urls_file.exists():
            with open(self.processed_urls_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_processed_urls(self) -> None:
        """Save record of processed URLs."""
        with open(self.processed_urls_file, "w") as f:
            json.dump(self.processed_urls, f, indent=2)
    
    def _get_url_hash(self, url: str) -> str:
        """Generate a unique hash for a URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _setup_llama_index(self) -> None:
        """Configure LlamaIndex with Databricks embeddings and LLM."""
        # Set environmental variables for Databricks
        os.environ["DATABRICKS_TOKEN"] = self.databricks_token
        os.environ["DATABRICKS_SERVING_ENDPOINT"] = self.databricks_endpoint
        
        # Set up embedding model
        embed_model = DatabricksEmbedding(model=self.embeddings_model)
        
        # Set up LLM
        llm = Databricks(
            model=self.llm_model,
            api_key=self.databricks_token,
            api_base=self.databricks_endpoint
        )
        
        # Configure LlamaIndex settings
        Settings.embed_model = embed_model
        Settings.llm = llm
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        
        # Initialize node parser with optimal chunk parameters
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )


    # Synchronous convenience methods
    def process_urls_from_file_sync(self, file_path: str, batch_size: int = 5) -> None:
        """Synchronous wrapper for processing URLs from a file."""
        asyncio.run(self.process_urls_from_file(file_path, batch_size))
    
    def optimize_parameters_sync(self, sample_urls: List[str]) -> None:
        """Synchronous wrapper for parameter optimization."""
        asyncio.run(self.optimize_parameters(sample_urls))
    
    # Asynchronous methods for efficient processing
    async def process_urls_from_file(self, file_path: str, batch_size: int = 5) -> None:
        """Process URLs from a text file in batches."""
        # Read URLs from file
        with open(file_path, "r") as f:
            urls = [line.strip() for line in f.readlines() if line.strip()]
        
        # Filter out already processed URLs
        new_urls = []
        for url in urls:
            url_hash = self._get_url_hash(url)
            if url_hash not in self.processed_urls:
                new_urls.append(url)
        
        if not new_urls:
            print("No new URLs to process.")
            return
        
        print(f"Processing {len(new_urls)} new URLs...")
        
        # Process URLs in batches
        for i in range(0, len(new_urls), batch_size):
            batch_urls = new_urls[i:i+batch_size]
            await self._process_url_batch_async(batch_urls)
        
        # Save updated processed URLs record
        self._save_processed_urls()
    
    async def _process_url_batch_async(self, urls: List[str]) -> None:
        """Process a batch of URLs asynchronously."""
        print(f"Processing batch of {len(urls)} URLs...")
        
        try:
            # Load pages asynchronously
            url_contents = await AsyncWebPageReader.load_pages(urls)
            
            # Process each URL and its content
            for url, content in url_contents:
                if not content:
                    print(f"Skipping {url}: no content retrieved")
                    continue
                
                url_hash = self._get_url_hash(url)
                
                # Create a document from the content
                document = Document(
                    text=content,
                    metadata={"source": url}
                )
                
                # Use the node parser to split the document into nodes
                nodes = self.node_parser.get_nodes_from_documents([document])
                
                # Create a unique storage path for this URL
                url_storage_dir = self.storage_dir / url_hash
                
                # Create a vector index from the nodes
                storage_context = StorageContext.from_defaults()
                index = VectorStoreIndex(nodes, storage_context=storage_context)
                
                # Persist the index to disk
                storage_context.persist(persist_dir=str(url_storage_dir))
                
                # Record this URL as processed
                self.processed_urls[url_hash] = {
                    "url": url,
                    "processed_at": datetime.now().isoformat(),
                    "storage_path": str(url_storage_dir),
                    "document_id": document.id_,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                }
                
                print(f"Processed and indexed: {url}")
        
        except Exception as e:
            print(f"Error processing URLs: {str(e)}")


    async def optimize_parameters(self, sample_urls: List[str]) -> None:
        """Determine optimal parameters based on sample URLs content."""
        print("Optimizing indexing parameters...")
        
        try:
            # Load sample pages asynchronously
            url_contents = await AsyncWebPageReader.load_pages(sample_urls)
            
            if not url_contents:
                print("Could not retrieve sample content for parameter optimization")
                return
            
            # Analyze content characteristics
            total_words = 0
            longest_paragraph = 0
            avg_paragraph_length = 0
            paragraph_count = 0
            
            for _, content in url_contents:
                paragraphs = [p for p in content.split('\n\n') if p.strip()]
                paragraph_count += len(paragraphs)
                
                for p in paragraphs:
                    words = len(p.split())
                    total_words += words
                    longest_paragraph = max(longest_paragraph, words)
            
            avg_paragraph_length = total_words / max(paragraph_count, 1)
            avg_content_length = total_words / len(url_contents)
            
            # Advanced heuristic for chunk size and overlap
            if avg_content_length > 10000:
                # Very long content
                self.chunk_size = 2048
                self.chunk_overlap = 100
            elif avg_content_length > 5000:
                # Long content
                self.chunk_size = 1536
                self.chunk_overlap = 75
            elif avg_content_length > 2000:
                # Medium content
                self.chunk_size = 1024
                self.chunk_overlap = 50
            else:
                # Short content
                self.chunk_size = 512
                self.chunk_overlap = 20
            
            # Adjust based on average paragraph length
            if avg_paragraph_length > 200:
                # Increase chunk size to capture complete paragraphs
                self.chunk_size = max(self.chunk_size, int(avg_paragraph_length * 2.5))
                self.chunk_overlap = max(self.chunk_overlap, int(avg_paragraph_length * 0.2))
            
            # Cap at reasonable limits
            self.chunk_size = min(self.chunk_size, 4096)
            self.chunk_overlap = min(self.chunk_overlap, 200)
            
            print(f"Optimized parameters based on content analysis:")
            print(f"  - Average content length: {avg_content_length:.1f} words")
            print(f"  - Average paragraph length: {avg_paragraph_length:.1f} words")
            print(f"  - Selected chunk_size: {self.chunk_size}")
            print(f"  - Selected chunk_overlap: {self.chunk_overlap}")
            
            # Update settings with new parameters
            Settings.chunk_size = self.chunk_size
            Settings.chunk_overlap = self.chunk_overlap
            
            # Update the node parser with new parameters
            self.node_parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
        except Exception as e:
            print(f"Error optimizing parameters: {str(e)}")
            print("Using default parameters.")


    def query(self, query_text: str, url: Optional[str] = None) -> str:
        """Query indexed URLs and return results.
        
        Args:
            query_text: The query text
            url: Optional specific URL to query. If None, will query all indexed URLs.
        """
        all_responses = []
        
        # Determine which URL indices to query
        url_infos = []
        if url:
            url_hash = self._get_url_hash(url)
            if url_hash in self.processed_urls:
                url_infos = [self.processed_urls[url_hash]]
            else:
                return f"URL {url} has not been indexed."
        else:
            url_infos = list(self.processed_urls.values())
        
        if not url_infos:
            return "No URLs have been indexed yet."
        
        # Query each selected index
        for url_info in url_infos:
            storage_path = url_info["storage_path"]
            
            try:
                # Load the index for this URL
                storage_context = StorageContext.from_defaults(persist_dir=storage_path)
                index = load_index_from_storage(storage_context)
                
                # Create a query engine
                query_engine = index.as_query_engine()
                
                # Execute query
                response = query_engine.query(query_text)
                
                all_responses.append({
                    "url": url_info["url"],
                    "response": str(response)
                })
            
            except Exception as e:
                print(f"Error querying index for {url_info['url']}: {str(e)}")
        
        # Combine responses
        if not all_responses:
            return "No relevant information found."
        
        # For a single URL query, just return that response
        if url:
            return all_responses[0]["response"]
        
        # For multiple URLs, combine responses
        combined_response = "\n\n".join([
            f"From {r['url']}:\n{r['response']}" for r in all_responses
        ])
        
        return combined_response


def run_rag_example():
    """Complete example demonstrating the RAG pipeline."""
    # Set Databricks credentials
    DATABRICKS_TOKEN = "your-databricks-token"
    DATABRICKS_ENDPOINT = "https://your-workspace.cloud.databricks.com/serving-endpoints"
    
    # Initialize the RAG pipeline
    rag = RAGPipeline(
        databricks_token=DATABRICKS_TOKEN,
        databricks_endpoint=DATABRICKS_ENDPOINT,
        embeddings_model="databricks-txt-embeddings-3-large",  # Using as requested
        llm_model="databricks-gpt-4o",  # Using as requested
        storage_dir="./rag_storage",
    )
    
    # Create a sample URLs file
    sample_urls = [
        "http://paulgraham.com/worked.html",
        "https://docs.llamaindex.ai/en/stable/",
        "https://docs.databricks.com/aws/en/generative-ai/agent-framework/llamaindex-uc-integration",
    ]
    
    with open("sample_urls.txt", "w") as f:
        for url in sample_urls:
            f.write(f"{url}\n")
    
    # Optimize parameters based on sample URLs
    rag.optimize_parameters_sync(sample_urls[:1])
    
    # Process URLs from the file
    rag.process_urls_from_file_sync("sample_urls.txt", batch_size=2)
    
    # Example queries
    queries = [
        "What is LlamaIndex and how does it integrate with Databricks?",
        "What did Paul Graham write about in his essay?",
    ]
    
    for query in queries:
        print(f"\n\nQuery: {query}")
        response = rag.query(query)
        print(f"Response: {response}")

# Run the example
if __name__ == "__main__":
    run_rag_example()


