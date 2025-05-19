from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from config import Config

def get_chroma_client():
    return chromadb.PersistentClient(path=Config.CHROMA_DIR)

def get_vector_store(collection_name):
    client = get_chroma_client()
    chroma_collection = client.get_or_create_collection(collection_name)
    return ChromaVectorStore(chroma_collection=chroma_collection)

def get_storage_context(collection_name, embed_model=None):
    return StorageContext.from_defaults(
        vector_store=get_vector_store(collection_name),
        embed_model=embed_model
    )