import os
import sys
import hashlib
import logging
import gradio as gr
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rag_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
PROCESSED_FILES_RECORD = "processed_files.txt"
CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "pdf_documents"

class RAGPipeline:
    """
    RAG Pipeline for processing PDF documents using llama-index and ChromaDB.
    """
    
    def __init__(self):
        """Initialize the RAG Pipeline with necessary configurations."""
        try:
            logger.info("Initializing RAG Pipeline...")
            
            # Create necessary directories
            os.makedirs(DATA_DIR, exist_ok=True)
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            
            # Check if collection exists, if not create it
            try:
                self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
                logger.info(f"Collection '{COLLECTION_NAME}' already exists. Using existing collection.")
            except Exception:
                logger.info(f"Creating new collection '{COLLECTION_NAME}'.")
                self.collection = self.chroma_client.create_collection(COLLECTION_NAME)
            
            # Initialize embeddings model (LM Studio model)
            self.embed_model = OllamaEmbedding(
                model_name="nomic-embed-text",
                base_url="http://localhost:11434"  # LM Studio default port
            )
            
            # Initialize LLM (Ollama)
            self.llm = Ollama(
                model="llama3.2:1b",
                base_url="http://localhost:11434"
            )
            
            # Set global settings
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm
            
            # Initialize processed files tracker
            self.processed_files = self._load_processed_files()
            
            # Initialize vector store with ChromaDB
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Initialize the index (will be populated later)
            self.index = None
            
            # Determine optimal chunking parameters
            # These will be adjusted based on document analysis
            self.chunk_size = 1024
            self.chunk_overlap = 200
            
            logger.info("RAG Pipeline initialization completed successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
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
        """Calculate MD5 hash of a file."""
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
        
        This is a simplified version. In a production environment, you might want
        to implement more sophisticated analysis.
        """
        try:
            # Get total size of all documents
            total_size = sum(os.path.getsize(path) for path in file_paths)
            
            # Adjust chunk size based on total document size
            if total_size > 10 * 1024 * 1024:  # Greater than 10MB
                self.chunk_size = 512
                self.chunk_overlap = 50
            elif total_size > 5 * 1024 * 1024:  # Greater than 5MB
                self.chunk_size = 768
                self.chunk_overlap = 100
            else:
                self.chunk_size = 1024
                self.chunk_overlap = 200
                
            logger.info(f"Optimized chunk parameters: size={self.chunk_size}, overlap={self.chunk_overlap}")
        except Exception as e:
            logger.error(f"Error optimizing chunk parameters: {str(e)}")
            # Fall back to default parameters
            self.chunk_size = 1024
            self.chunk_overlap = 200

    def process_documents(self, file_paths: List[str]) -> Tuple[bool, str]:
        """
        Process PDF documents, create embeddings, and store in ChromaDB.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Tuple of (success status, message)
        """
        if not file_paths:
            return False, "No files provided for processing."
        
        try:
            logger.info(f"Starting to process {len(file_paths)} documents...")
            
            # Filter out already processed files
            new_files = []
            ignored_files = []
            
            for file_path in file_paths:
                # Calculate hash of the file
                file_hash = self._calculate_file_hash(file_path)
                
                # Check if this file was processed before
                if file_path in self.processed_files and self.processed_files[file_path] == file_hash:
                    ignored_files.append(os.path.basename(file_path))
                    logger.info(f"Ignoring already processed file: {file_path}")
                else:
                    new_files.append(file_path)
                    self.processed_files[file_path] = file_hash
            
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
                    logger.info(f"Successfully loaded document: {file_path}")
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
            nodes = node_parser.get_nodes_from_documents(documents)
            logger.info(f"Created {len(nodes)} nodes from documents.")
            
            # Create or update index
            logger.info("Creating vector index...")
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
            )
            
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

    def query(self, query_text: str) -> Tuple[bool, str]:
        """
        Query the indexed documents.
        
        Args:
            query_text: The query string
            
        Returns:
            Tuple of (success status, response)
        """
        try:
            logger.info(f"Processing query: {query_text}")
            
            # Check if index exists
            if self.index is None:
                return False, "No documents have been indexed yet. Please upload and process documents first."
            
            # Create query engine
            query_engine = self.index.as_query_engine()
            
            # Execute query
            logger.info("Executing query...")
            response = query_engine.query(query_text)
            
            logger.info("Query processed successfully")
            return True, str(response)
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


# Create and initialize the RAG pipeline
def initialize_pipeline() -> Tuple[bool, str, Optional[RAGPipeline]]:
    """Initialize the RAG pipeline and return status."""
    try:
        pipeline = RAGPipeline()
        return True, "RAG Pipeline initialized successfully.", pipeline
    except Exception as e:
        error_msg = f"Failed to initialize RAG Pipeline: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, None


# Gradio UI functions
def upload_and_process(files, pipeline_state):
    """Handle file upload and processing in Gradio."""
    if pipeline_state is None:
        return "Pipeline not initialized. Please check the logs for errors."
    
    # Save uploaded files to data directory
    file_paths = []
    for file in files:
        file_name = os.path.basename(file.name)
        save_path = os.path.join(DATA_DIR, file_name)
        
        # If the file already exists in the data directory, we'll use the existing file
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(open(file.name, "rb").read())
        
        file_paths.append(save_path)
    
    # Process the documents
    success, message = pipeline_state.process_documents(file_paths)
    
    if success:
        return message
    else:
        return f"Error: {message}"


def query_documents(query, history, pipeline_state):
    """Handle queries in Gradio chat interface."""
    if pipeline_state is None:
        return "Pipeline not initialized. Please check the logs for errors."
    
    if not query.strip():
        return "Please enter a query."
    
    success, response = pipeline_state.query(query)
    
    if success:
        return response
    else:
        return f"Error: {response}"


# Main application
def main():
    """Main function to run the RAG pipeline with Gradio UI."""
    # Initialize the pipeline
    success, message, pipeline_state = initialize_pipeline()
    
    if not success:
        logger.error(f"Failed to initialize pipeline: {message}")
        print(f"Error: {message}")
        return
    
    logger.info("Setting up Gradio interface...")
    
    # Define Gradio interface
    with gr.Blocks(title="PDF Document RAG System") as demo:
        gr.Markdown("# PDF Document RAG System")
        gr.Markdown("Upload PDF documents to index them, then ask questions about their content.")
        
        with gr.Tab("Upload Documents"):
            upload_files = gr.File(
                file_count="multiple",
                label="Upload PDF Documents",
                file_types=[".pdf"]
            )
            upload_button = gr.Button("Process Documents")
            upload_output = gr.Textbox(label="Processing Status")
            
            upload_button.click(
                upload_and_process,
                inputs=[upload_files, gr.State(pipeline_state)],
                outputs=upload_output
            )
        
        with gr.Tab("Chat Interface"):
            chatbot = gr.Chatbot(label="Conversation")
            msg = gr.Textbox(label="Ask a question about your uploaded documents")
            clear = gr.Button("Clear")
            
            def respond(query, history):
                response = query_documents(query, history, pipeline_state)
                history.append((query, response))
                return "", history
            
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
    
    logger.info("Launching Gradio interface...")
    demo.launch()
    logger.info("Gradio interface closed.")


if __name__ == "__main__":
    main()