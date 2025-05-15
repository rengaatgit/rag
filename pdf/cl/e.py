import os
import sys
import hashlib
import logging
from typing import List, Dict, Tuple, Optional
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    VectorStoreIndex,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("embedding_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = r"C:\Users\areng\Desktop\files"  # Source PDF directory
PROCESSED_FILES_RECORD = "processed_files.txt"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_documents"

class EmbeddingPipeline:
    """
    Pipeline for processing PDF documents and creating embeddings stored in ChromaDB.
    """
    
    def __init__(self):
        """Initialize the Embedding Pipeline with necessary configurations."""
        try:
            logger.info("Initializing Embedding Pipeline...")
            
            # Create ChromaDB directory if it doesn't exist
            os.makedirs(CHROMA_DB_DIR, exist_ok=True)
            
            # Initialize ChromaDB client
            try:
                self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
                logger.info("ChromaDB client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
                raise Exception(f"ChromaDB initialization error: {str(e)}")
            
            # Check if collection exists, if not create it
            try:
                self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)
                logger.info(f"Using ChromaDB collection: {COLLECTION_NAME}")
            except Exception as e:
                logger.error(f"Failed to get or create ChromaDB collection: {str(e)}")
                raise Exception(f"ChromaDB collection error: {str(e)}")
            
            # Initialize embeddings model (from Ollama)
            try:
                self.embed_model = OllamaEmbedding(
                    model_name="nomic-embed-text",
                    base_url="http://localhost:11434"  # Default Ollama port
                )
                # Test the embedding model
                test_embedding = self.embed_model.get_text_embedding("Test embedding model")
                if not test_embedding or len(test_embedding) == 0:
                    raise Exception("Embedding model returned empty embedding")
                logger.info(f"Embedding model initialized successfully. Embedding dimensions: {len(test_embedding)}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {str(e)}")
                raise Exception(f"Embedding model error: {str(e)}")
            
            # Set global settings
            Settings.embed_model = self.embed_model
            
            # Initialize processed files tracker
            self.processed_files = self._load_processed_files()
            
            # Initialize vector store with ChromaDB
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Initialize chunking parameters with defaults
            # These will be adjusted based on document analysis
            self.chunk_size = 1024
            self.chunk_overlap = 200
            
            logger.info("Embedding Pipeline initialization completed successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Embedding Pipeline: {str(e)}")
            raise

    def _load_processed_files(self) -> Dict[str, str]:
        """Load the list of already processed files and their hashes."""
        processed_files = {}
        try:
            if os.path.exists(PROCESSED_FILES_RECORD):
                with open(PROCESSED_FILES_RECORD, "r") as f:
                    for line in f:
                        parts = line.strip().split("||")
                        if len(parts) == 2:
                            file_path, file_hash = parts
                            processed_files[file_path] = file_hash
                logger.info(f"Loaded {len(processed_files)} processed files from record.")
            else:
                logger.info("No processed files record found. Starting fresh.")
        except Exception as e:
            logger.error(f"Error loading processed files record: {str(e)}")
        
        return processed_files

    def _save_processed_files(self):
        """Save the list of processed files and their hashes."""
        try:
            with open(PROCESSED_FILES_RECORD, "w") as f:
                for file_path, file_hash in self.processed_files.items():
                    f.write(f"{file_path}||{file_hash}\n")
            logger.info(f"Saved {len(self.processed_files)} processed files to record.")
        except Exception as e:
            logger.error(f"Error saving processed files record: {str(e)}")

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file to check for changes."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            raise

    def _optimize_chunk_parameters(self, file_paths: List[str]) -> None:
        """
        Analyze documents to determine optimal chunking parameters.
        
        This optimizes chunk size and overlap based on the total size of documents.
        """
        try:
            # Get total size of all documents
            total_size = sum(os.path.getsize(path) for path in file_paths)
            logger.info(f"Total document size: {total_size / (1024*1024):.2f} MB")
            
            # Adjust chunk size based on total document size
            if total_size > 50 * 1024 * 1024:  # > 50 MB
                self.chunk_size = 384
                self.chunk_overlap = 40
                logger.info("Large document set detected. Using smaller chunks for efficiency.")
            elif total_size > 20 * 1024 * 1024:  # > 20 MB
                self.chunk_size = 512
                self.chunk_overlap = 50
                logger.info("Medium-large document set detected. Adjusting chunk parameters.")
            elif total_size > 5 * 1024 * 1024:  # > 5 MB
                self.chunk_size = 768
                self.chunk_overlap = 100
                logger.info("Medium document set detected. Using moderate chunks.")
            else:
                self.chunk_size = 1024
                self.chunk_overlap = 200
                logger.info("Smaller document set detected. Using larger chunks for better context.")
                
            logger.info(f"Optimized chunk parameters: size={self.chunk_size}, overlap={self.chunk_overlap}")
        except Exception as e:
            logger.error(f"Error optimizing chunk parameters: {str(e)}")
            # Fall back to default parameters
            self.chunk_size = 1024
            self.chunk_overlap = 200
            logger.info("Using default chunk parameters due to error.")

    def _get_pdf_files_from_directory(self) -> List[str]:
        """Get all PDF files from the data directory."""
        pdf_files = []
        try:
            if not os.path.exists(DATA_DIR):
                logger.error(f"Data directory does not exist: {DATA_DIR}")
                return pdf_files
                
            for file in os.listdir(DATA_DIR):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(DATA_DIR, file))
            
            logger.info(f"Found {len(pdf_files)} PDF files in {DATA_DIR}")
        except Exception as e:
            logger.error(f"Error scanning directory for PDF files: {str(e)}")
        
        return pdf_files

    def process_documents(self) -> Tuple[bool, str]:
        """
        Process PDF documents, create embeddings, and store in ChromaDB.
        
        Returns:
            Tuple of (success status, message)
        """
        try:
            # Get PDF files from the directory
            file_paths = self._get_pdf_files_from_directory()
            
            if not file_paths:
                return False, f"No PDF files found in directory: {DATA_DIR}"
            
            logger.info(f"Starting to process {len(file_paths)} PDF documents...")
            
            # Filter out already processed files
            new_files = []
            ignored_files = []
            
            for file_path in file_paths:
                try:
                    # Calculate hash of the file
                    file_hash = self._calculate_file_hash(file_path)
                    
                    # Check if this file was processed before
                    if file_path in self.processed_files and self.processed_files[file_path] == file_hash:
                        ignored_files.append(os.path.basename(file_path))
                        logger.info(f"Ignoring already processed file: {file_path}")
                    else:
                        new_files.append(file_path)
                        self.processed_files[file_path] = file_hash
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    return False, f"Error checking file {os.path.basename(file_path)}: {str(e)}"
            
            if not new_files:
                ignored_msg = ", ".join(ignored_files)
                return True, f"All files have been processed before. Ignored: {ignored_msg}"
            
            # Optimize chunking parameters based on the documents
            self._optimize_chunk_parameters(new_files)
            
            # Load documents
            documents = []
            for file_path in new_files:
                try:
                    logger.info(f"Loading document: {file_path}")
                    loader = SimpleDirectoryReader(input_files=[file_path])
                    docs = loader.load_data()
                    documents.extend(docs)
                    logger.info(f"Successfully loaded document: {file_path} ({len(docs)} document(s))")
                except Exception as e:
                    logger.error(f"Error loading document {file_path}: {str(e)}")
                    return False, f"Error processing {os.path.basename(file_path)}: {str(e)}"
            
            if not documents:
                return False, "Failed to load any documents."
            
            # Create node parser with optimized parameters
            node_parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Process documents into nodes
            logger.info("Parsing documents into nodes...")
            try:
                nodes = node_parser.get_nodes_from_documents(documents)
                logger.info(f"Created {len(nodes)} nodes from documents.")
                if len(nodes) == 0:
                    raise Exception("No nodes were created from documents")
            except Exception as e:
                logger.error(f"Error parsing documents into nodes: {str(e)}")
                return False, f"Failed to parse documents: {str(e)}"
            
            # Create vector index
            logger.info("Creating vector index...")
            try:
                index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=self.storage_context,
                )
                
                # Verify indexing worked by checking collection count
                doc_count = len(self.collection.get(limit=1)['ids'])
                if doc_count <= 0:
                    raise Exception("No documents were added to the collection")
                    
                logger.info(f"Vector index created successfully with {doc_count} entries.")
            except Exception as e:
                logger.error(f"Error creating vector index: {str(e)}")
                return False, f"Failed to create vector index: {str(e)}"
            
            # Save processed files record
            self._save_processed_files()
            
            processed_files_msg = ", ".join([os.path.basename(f) for f in new_files])
            ignored_files_msg = ", ".join(ignored_files)
            
            success_msg = f"Successfully processed {len(new_files)} documents: {processed_files_msg}"
            if ignored_files:
                success_msg += f"\nIgnored {len(ignored_files)} already processed documents: {ignored_files_msg}"
                
            logger.info(success_msg)
            return True, success_msg
            
        except Exception as e:
            error_msg = f"Error processing documents: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


def main():
    """Main function to run the embedding pipeline."""
    print("Starting PDF Document Embedding Pipeline...")
    
    try:
        # Initialize the pipeline
        pipeline = EmbeddingPipeline()
        print("Pipeline initialized successfully.")
        
        # Process documents
        success, message = pipeline.process_documents()
        
        if success:
            print(f"Success: {message}")
            print(f"\nEmbeddings have been stored in ChromaDB at: {os.path.abspath(CHROMA_DB_DIR)}")
            print("You can now use the query script to ask questions about these documents.")
        else:
            print(f"Error: {message}")
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        logger.critical(f"Fatal error in main execution: {str(e)}")


if __name__ == "__main__":
    main()