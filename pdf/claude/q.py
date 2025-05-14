import os
import sys
import logging
from typing import Tuple, Optional
import chromadb
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("query_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_documents"

class QueryPipeline:
    """
    Pipeline for querying PDF document embeddings stored in ChromaDB.
    """
    
    def __init__(self):
        """Initialize the Query Pipeline with necessary configurations."""
        try:
            logger.info("Initializing Query Pipeline...")
            
            # Check if ChromaDB directory exists
            if not os.path.exists(CHROMA_DB_DIR):
                logger.error(f"ChromaDB directory does not exist: {CHROMA_DB_DIR}")
                raise Exception(f"ChromaDB directory not found: {CHROMA_DB_DIR}")
            
            # Initialize ChromaDB client
            try:
                self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
                logger.info("ChromaDB client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
                raise Exception(f"ChromaDB initialization error: {str(e)}")
            
            # Check if collection exists
            try:
                self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
                collection_count = len(self.collection.get(limit=1)['ids'])
                logger.info(f"Using ChromaDB collection: {COLLECTION_NAME} (contains data: {collection_count > 0})")
                
                if collection_count == 0:
                    logger.warning("The ChromaDB collection exists but appears to be empty.")
            except Exception as e:
                logger.error(f"Failed to get ChromaDB collection: {str(e)}")
                raise Exception(f"ChromaDB collection error: {str(e)}")
            
            # Initialize LLM from Ollama
            try:
                self.llm = Ollama(
                    model="llama3.2:1b",
                    base_url="http://localhost:11434"  # Default Ollama port
                )
                logger.info("LLM initialized successfully from Ollama.")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {str(e)}")
                raise Exception(f"LLM initialization error: {str(e)}")
            
            # Set global settings
            Settings.llm = self.llm
            
            # Initialize vector store with ChromaDB
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Load the index from the vector store
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store
                )
                logger.info("Vector index loaded successfully from ChromaDB.")
            except Exception as e:
                logger.error(f"Failed to load vector index: {str(e)}")
                raise Exception(f"Vector index loading error: {str(e)}")
            
            # Create query engine
            try:
                self.query_engine = self.index.as_query_engine(
                    similarity_top_k=3,  # Get top 3 most similar chunks
                    streaming=False  # Set to True if you want streaming responses
                )
                logger.info("Query engine created successfully.")
            except Exception as e:
                logger.error(f"Failed to create query engine: {str(e)}")
                raise Exception(f"Query engine creation error: {str(e)}")
            
            logger.info("Query Pipeline initialization completed successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Query Pipeline: {str(e)}")
            raise

    def query(self, query_text: str) -> Tuple[bool, str]:
        """
        Query the indexed documents.
        
        Args:
            query_text: The query string
            
        Returns:
            Tuple of (success status, response)
        """
        try:
            if not query_text or query_text.strip() == "":
                return False, "Query text cannot be empty."
                
            logger.info(f"Processing query: {query_text}")
            
            # Execute query
            response = self.query_engine.query(query_text)
            
            # Log source nodes used
            if hasattr(response, 'source_nodes') and response.source_nodes:
                sources = [
                    f"Node {i+1}: {node.node.get_text()[:50]}..."
                    for i, node in enumerate(response.source_nodes)
                ]
                logger.info(f"Query used {len(sources)} source nodes: {sources}")
            
            logger.info("Query processed successfully")
            return True, str(response)
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


def main():
    """Main function to run the query pipeline interactive session."""
    print("Starting PDF Document Query Pipeline...")
    
    try:
        # Initialize the pipeline
        pipeline = QueryPipeline()
        print("Pipeline initialized successfully.")
        print("You can now ask questions about the documents that were previously embedded.")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        # Interactive query loop
        while True:
            query = input("\nEnter your question: ")
            
            if query.lower() in ['exit', 'quit']:
                print("Exiting query session.")
                break
                
            if not query.strip():
                print("Please enter a valid question.")
                continue
                
            # Process query
            success, response = pipeline.query(query)
            
            if success:
                print("\nAnswer:")
                print(response)
            else:
                print(f"Error: {response}")
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        logger.critical(f"Fatal error in main execution: {str(e)}")


if __name__ == "__main__":
    main()