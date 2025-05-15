# pip installs required (run these first)
"""
pip install llama-index-core==0.12.0
pip install llama-index-embeddings-databricks==0.12.0
pip install llama-index-llms-databricks==0.12.0
pip install beautifulsoup4 requests trafilatura
"""

import os
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.databricks import DatabricksEmbedding
from llama_index.llms.databricks import Databricks
from llama_index.readers.web import SimpleWebPageReader

# Configuration
PERSIST_DIR = "./storage"
PROCESSED_URLS_FILE = "processed_urls.txt"
DATABRICKS_ENDPOINT = "databricks"  # Your Databricks endpoint
DATABRICKS_EMBED_MODEL = "databricks-bge-large-en"
DATABRICKS_LLM_MODEL = "databricks-dbrx-instruct"

class RAGPipeline:
    def __init__(self):
        # Initialize models
        self.embed_model = DatabricksEmbedding(
            model_name=DATABRICKS_EMBED_MODEL,
            databricks_token=os.getenv("DATABRICKS_TOKEN"),
            databricks_host=os.getenv("DATABRICKS_HOST"),
        )
        
        self.llm = Databricks(
            model=DATABRICKS_LLM_MODEL,
            databricks_token=os.getenv("DATABRICKS_TOKEN"),
            databricks_host=os.getenv("DATABRICKS_HOST"),
        )
        
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        Settings.chunk_size = 1024  # Auto-tuned based on model context
        Settings.chunk_overlap = 200
        
        self.processed_urls = self._load_processed_urls()

    def _load_processed_urls(self):
        try:
            with open(PROCESSED_URLS_FILE, "r") as f:
                return set(f.read().splitlines())
        except FileNotFoundError:
            return set()

    def _save_processed_urls(self):
        with open(PROCESSED_URLS_FILE, "w") as f:
            f.write("\n".join(self.processed_urls))

    def process_urls(self, url_file: str):
        # Read URLs and filter new ones
        with open(url_file, "r") as f:
            urls = [url.strip() for url in f.read().splitlines() if url.strip()]
        
        new_urls = [url for url in urls if url not in self.processed_urls]
        if not new_urls:
            print("No new URLs to process")
            return

        # Load documents with async processing
        print(f"Processing {len(new_urls)} new URLs...")
        reader = SimpleWebPageReader(html_to_text=True)
        documents = reader.load_data(new_urls)

        # Auto-configure node parser
        node_parser = SentenceSplitter.from_defaults(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap,
        )
        nodes = node_parser.get_nodes_from_documents(documents)

        # Create or update index
        if not os.path.exists(PERSIST_DIR):
            storage_context = StorageContext.from_defaults()
            index = VectorStoreIndex(nodes, storage_context=storage_context)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            index.insert_nodes(nodes)

        # Persist index and update processed URLs
        storage_context.persist(persist_dir=PERSIST_DIR)
        self.processed_urls.update(new_urls)
        self._save_processed_urls()

    def query_engine(self):
        # Load existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        return index.as_query_engine()

# Test case
if __name__ == "__main__":
    # Set these environment variables first
    os.environ["DATABRICKS_TOKEN"] = "your_api_token"
    os.environ["DATABRICKS_HOST"] = "your_databricks_host"

    # Initialize pipeline
    pipeline = RAGPipeline()

    # Sample URLs file (urls.txt)
    """
    https://llamaindex.ai/blog
    https://docs.databricks.com/en/machine-learning/foundation-models/index.html
    """

    # Process URLs
    pipeline.process_urls("urls.txt")

    # Query example
    query_engine = pipeline.query_engine()
    response = query_engine.query("What are the key features of Databricks Foundation Models?")
    print(response)