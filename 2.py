import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.databricks import Databricks
from llama_index.embeddings.databricks import DatabricksEmbedding

import chromadb

# Optionally, set credentials in code if not using environment variables
# os.environ["DATABRICKS_TOKEN"] = "your-databricks-access-token"
# os.environ["DATABRICKS_SERVING_ENDPOINT"] = "https://<your-workspace>.cloud.databricks.com/serving-endpoints/<your-llm-endpoint>"
# os.environ["DATABRICKS_EMBEDDING_ENDPOINT"] = "https://<your-workspace>.cloud.databricks.com/serving-endpoints/<your-embedding-endpoint>"

# Instantiate Databricks LLM
llm = Databricks(
    model="databricks-dbrx-instruct",  # or your LLM model name
    api_key=os.environ["DATABRICKS_TOKEN"],
    api_base=os.environ["DATABRICKS_SERVING_ENDPOINT"],
)

# Instantiate Databricks Embedding Model
embed_model = DatabricksEmbedding(
    model="databricks-bge-large-en",   # or your embedding model name
    api_key=os.environ["DATABRICKS_TOKEN"],
    endpoint=os.environ["DATABRICKS_EMBEDDING_ENDPOINT"],
)

# Set LLM and embedding model for LlamaIndex
Settings.llm = llm
Settings.embed_model = embed_model

# Set up ChromaDB as vector store
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# You may need to configure LlamaIndex's vector store integration to use this client

# Load documents from a directory (replace "data" with your folder)
documents = SimpleDirectoryReader("data").load_data()

# Build the index
index = VectorStoreIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

print("===========================================")
print("Question: What did the author do growing up?")
print("DBRX Response:", response)

