import os
from llama_index.embeddings.databricks import DatabricksEmbedding

class Config:
    REF_FOLDER = "ref"
    USER_FOLDER = "user-files"
    CHROMA_DIR = "chroma_db"
    DB_API_TOKEN = "DATABRICKS_TOKEN"
    EMBEDDING_ENDPOINT = "databricks-embedding-endpoint"
    LLM_ENDPOINT = "databricks-llm-endpoint"
    
    @classmethod
    def get_embedding_model(cls):
        return DatabricksEmbedding(
            endpoint=cls.EMBEDDING_ENDPOINT,
            databricks_token=cls.DB_API_TOKEN,
            model="gpt-3.5-turbo"
        )