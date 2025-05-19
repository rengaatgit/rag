from config import Config
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from .db_operations import get_storage_context

# Remove local embed_model definition and use the config version
def process_files(folder_path, collection_name):
    documents = SimpleDirectoryReader(folder_path).load_data()
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(documents)
    
    storage_context = get_storage_context(collection_name)
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=Config.get_embedding_model()  # Use from config
    )
    return True