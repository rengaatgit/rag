import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from databricks_langchain import DatabricksEmbeddings
import chromadb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Initialize Databricks embedding model
# Explanation: Uses
# Uses databricks-bge-large-en, a 1024-dimension embedding model optimized for RAG.
logger.info("Initializing Databricks embedding model...")
try:
    embed_model = DatabricksEmbeddings(
        endpoint="databricks-bge-large-en",
        host=os.getenv("DATABRICKS_HOST", "https://your-workspace.cloud.databricks.com"),
        token=os.getenv("DATABRICKS_TOKEN", "your-access-token")
    )
    # Test embedding model with a sample text
    test_text = "This is a test sentence."
    test_embedding = embed_model.embed_query(test_text)
    logger.info(f"Embedding model test successful. Embedding length: {len(test_embedding)}")
except Exception as e:
    logger.error(f"Failed to initialize or test embedding model: {str(e)}")
    exit(1)

# Step 2: Configure global settings for embedding model
# Explanation: Sets the embedding model globally for LlamaIndex 0.12.x.
logger.info("Configuring global settings...")
try:
    Settings.embed_model = embed_model
except Exception as e:
    logger.error(f"Failed to configure settings: {str(e)}")
    exit(1)

# Step 3: Set up node parser with embedding parameters
# Explanation: chunk_size=512 matches BGE's token window; overlap=50 ensures continuity.
logger.info("Setting up node parser...")
try:
    node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
except Exception as e:
    logger.error(f"Failed to set up node parser: {str(e)}")
    exit(1)

# Step 4: Set up ChromaDB client and collection
# Explanation: Persistent storage in ./chroma_db for scalability; reuses existing collection.
logger.info("Setting up ChromaDB...")
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "pdf_embeddings"
    try:
        collection = client.get_collection(collection_name)
        logger.info(f"Loaded existing ChromaDB collection: {collection_name}")
    except:
        collection = client.create_collection(collection_name)
        logger.info(f"Created new ChromaDB collection: {collection_name}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
except Exception as e:
    logger.error(f"Failed to set up ChromaDB: {str(e)}")
    exit(1)

# Step 5: Load existing document IDs from ChromaDB
# Explanation: Prevents reprocessing by checking existing IDs; uses file paths as unique IDs.
logger.info("Loading existing document IDs...")
try:
    collection_data = collection.get()
    existing_ids = set(collection_data['ids'])
    logger.info(f"Found {len(existing_ids)} existing document IDs in ChromaDB")
except Exception as e:
    logger.error(f"Failed to load existing IDs: {str(e)}")
    exit(1)

# Step 6: Initialize VectorStoreIndex with ChromaDB
# Explanation: Links LlamaIndex with ChromaDB for embedding storage and retrieval.
logger.info("Initializing VectorStoreIndex...")
try:
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
except Exception as e:
    logger.error(f"Failed to initialize index: {str(e)}")
    exit(1)

# Step 7: Process PDFs from directory
# Explanation: Reads PDFs one-by-one to manage memory; skips processed files.
pdf_dir = r"C:\Users\areng\Desktop\files"
logger.info(f"Scanning directory: {pdf_dir}")
try:
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        logger.warning("No PDF files found in directory.")
        exit(0)
    logger.info(f"Found {len(pdf_files)} PDF files")
except Exception as e:
    logger.error(f"Failed to scan directory: {str(e)}")
    exit(1)

for pdf_path in pdf_files:
    doc_id = pdf_path  # Using full file path as unique document ID
    logger.info(f"Processing {pdf_path}...")
    
    # Checkpoint: Skip if already processed
    if doc_id in existing_ids:
        logger.info(f"Skipping {pdf_path} - already processed.")
        continue
    
    # Checkpoint: Load PDF
    try:
        reader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = reader.load_data()
        if not documents:
            logger.warning(f"No content extracted from {pdf_path}. Possibly empty or non-text PDF.")
            continue
        logger.info(f"Loaded {len(documents)} document(s) from {pdf_path}")
        # Log sample content
        sample_content = documents[0].text[:100].replace('\n', ' ')
        logger.info(f"Sample content: {sample_content}...")
    except Exception as e:
        logger.error(f"Failed to load {pdf_path}: {str(e)}")
        continue
    
    # Checkpoint: Parse document into nodes
    try:
        document = documents[0]  # Single PDF typically yields one document
        nodes = node_parser.get_nodes_from_documents([document])
        if not nodes:
            logger.warning(f"No nodes generated from {pdf_path}. Check document content or parser settings.")
            continue
        logger.info(f"Generated {len(nodes)} nodes from {pdf_path}")
        # Log sample node
        sample_node = nodes[0].text[:100].replace('\n', ' ')
        logger.info(f"Sample node: {sample_node}...")
    except Exception as e:
        logger.error(f"Failed to parse {pdf_path}: {str(e)}")
        continue
    
    # Checkpoint: Insert nodes into index and verify storage
    try:
        for node in nodes:
            node.id_ = doc_id  # Assign document ID to node
            index.insert_nodes([node])
        # Verify storage in ChromaDB
        collection_data = collection.get(where={"id": doc_id})
        if not collection_data['ids']:
            logger.error(f"Failed to store embeddings for {pdf_path} in ChromaDB")
            continue
        logger.info(f"Successfully embedded {pdf_path}. Stored {len(collection_data['ids'])} embeddings.")
    except Exception as e:
        logger.error(f"Failed to embed {pdf_path}: {str(e)}")
        continue

# Step 8: Final verification of ChromaDB
logger.info("Verifying ChromaDB contents...")
try:
    final_data = collection.get()
    logger.info(f"Total embeddings stored: {len(final_data['ids'])}")
    if len(final_data['ids']) == 0:
        logger.warning("No embeddings stored in ChromaDB. Check PDF content or embedding process.")
except Exception as e:
    logger.error(f"Failed to verify ChromaDB contents: {str(e)}")

logger.info("Embedding process completed.")