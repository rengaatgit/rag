"""
RAG Pipeline with Model Context Protocol (MCP) using LlamaIndex
"""
 
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseSynthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI

# For MCP implementation
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore

# Constants
DATA_DIR = "data/file_uploads"
STORAGE_DIR = "storage"
EMBED_MODEL = "local:BAAI/bge-small-en-v1.5"  # You can change this to any model you prefer

class MCPFormatter:
    """
    Model Context Protocol (MCP) formatter for RAG pipeline.
    
    MCP provides a structured way to format context for LLMs, improving
    the quality and relevance of responses.
    """
    
    @staticmethod
    def format_context(nodes_with_scores: List[NodeWithScore]) -> str:
        """
        Format retrieved context according to MCP guidelines.
        
        Args:
            nodes_with_scores: List of retrieved nodes with relevance scores
            
        Returns:
            Formatted context string following MCP
        """
        # Sort nodes by score in descending order
        sorted_nodes = sorted(nodes_with_scores, key=lambda x: x.score if x.score is not None else 0, reverse=True)
        
        # Format according to MCP
        formatted_context = "# CONTEXT\n\n"
        
        for i, node in enumerate(sorted_nodes):
            # Extract metadata
            source = node.node.metadata.get("file_name", "Unknown source")
            page_num = node.node.metadata.get("page_label", "")
            page_info = f" (page {page_num})" if page_num else ""
            
            # Format each context chunk with source attribution and relevance indicator
            relevance = "★★★" if i < 3 else "★★" if i < 6 else "★"
            formatted_context += f"## Source {i+1}: {source}{page_info} | Relevance: {relevance}\n\n"
            formatted_context += f"{node.node.text}\n\n"
        
        return formatted_context

class RAGWithMCP:
    """RAG pipeline with Model Context Protocol integration."""
    
    def __init__(self, data_dir: str = DATA_DIR, storage_dir: str = STORAGE_DIR):
        """
        Initialize the RAG pipeline with MCP.
        
        Args:
            data_dir: Directory containing documents for indexing
            storage_dir: Directory to store the vector index
        """
        self.data_dir = data_dir
        self.storage_dir = storage_dir
        self.index = None
        self.mcp_formatter = MCPFormatter()
        
        # Create directories if they don't exist
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        Path(storage_dir).mkdir(exist_ok=True, parents=True)
        
        # Initialize LLM
        self.llm = OpenAI(model="gpt-3.5-turbo")
        
        # Configure LlamaIndex settings
        Settings.llm = self.llm
        Settings.embed_model = EMBED_MODEL
        
        # Load or create index
        self._load_or_create_index()
        
        # Create MCP-aware prompt template
        self.mcp_prompt_template = PromptTemplate(
            """
            <context>
            {context_str}
            </context>
            
            Given the context information provided above, please answer the following question:
            Question: {query_str}
            
            Answer:
            """
        )
    
    def _load_or_create_index(self):
        """Load existing index or create a new one if it doesn't exist."""
        try:
            # Try to load existing index
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            self.index = load_index_from_storage(storage_context)
            print(f"Loaded existing index from {self.storage_dir}")
        except:
            # Create new index if loading fails
            print(f"Creating new index from documents in {self.data_dir}")
            self._create_index()
    
    def _create_index(self):
        """Create a new vector index from documents."""
        # Check if there are documents to index
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            print(f"No documents found in {self.data_dir}")
            # Create an empty index
            self.index = VectorStoreIndex([])
            return
        
        # Load documents
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        
        # Parse documents into nodes
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = parser.get_nodes_from_documents(documents)
        
        # Create index
        self.index = VectorStoreIndex(nodes)
        
        # Persist index
        self.index.storage_context.persist(persist_dir=self.storage_dir)
        print(f"Created and saved new index to {self.storage_dir}")
    
    def add_documents(self, file_paths: List[str] = None):
        """
        Add new documents to the index.
        
        Args:
            file_paths: List of file paths to add. If None, all files in data_dir are indexed.
        """
        if file_paths:
            documents = []
            for file_path in file_paths:
                if os.path.exists(file_path):
                    file_documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
                    documents.extend(file_documents)
        else:
            documents = SimpleDirectoryReader(self.data_dir).load_data()
        
        if not documents:
            print("No new documents to add")
            return
        
        # Parse documents into nodes
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = parser.get_nodes_from_documents(documents)
        
        # Update index with new nodes
        if self.index is None:
            self.index = VectorStoreIndex(nodes)
        else:
            self.index.insert_nodes(nodes)
        
        # Persist updated index
        self.index.storage_context.persist(persist_dir=self.storage_dir)
        print(f"Added {len(documents)} documents to the index")
    
    def query(self, query_text: str, top_k: int = 5) -> str:
        """
        Query the RAG pipeline with MCP formatting.
        
        Args:
            query_text: The query text
            top_k: Number of top documents to retrieve
            
        Returns:
            Response from the LLM
        """
        if self.index is None:
            return "No index available. Please add documents first."
        
        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )
        
        # Get nodes
        nodes_with_scores = retriever.retrieve(query_text)
        
        # Format context using MCP
        mcp_context = self.mcp_formatter.format_context(nodes_with_scores)
        
        # Create response synthesizer with MCP prompt
        response_synthesizer = ResponseSynthesizer.from_args(
            response_mode="compact",
            text_qa_template=self.mcp_prompt_template
        )
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        )
        
        # Execute query
        response = query_engine.query(query_text)
        
        return response.response

# Example usage
if __name__ == "__main__":
    # Initialize RAG with MCP
    rag = RAGWithMCP()
    
    # Add documents (if any exist in the data directory)
    rag.add_documents()
    
    # Example query
    query = "What are the key concepts in the documents?"
    response = rag.query(query)
    
    print(f"Query: {query}")
    print(f"Response: {response}")