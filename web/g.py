import os
import json
from llama_index import SimpleWebPageReader, VectorStoreIndex, StorageContext, QueryEngine
from llama_index.embeddings.databricks import DatabricksEmbedding
from llama_index.llms.databricks import Databricks
from llama_index.vector_stores.databricks import DatabricksVectorSearch
from databricks.vector_search.client import VectorSearchClient
from llama_index.node_parser import SimpleNodeParser

# Set up Databricks credentials
os.environ["DATABRICKS_TOKEN"] = "<your_databricks_token>"
os.environ["DATABRICKS_SERVING_ENDPOINT"] = "<your_databricks_endpoint>"

# Configure embedding model and LLM
embed_model = DatabricksEmbedding(model="databricks-bge-large-en")
llm = Databricks(model="databricks-dolly-v2-12b")

# Load URLs from file
with open("urls.txt", "r") as f:
    urls = [line.strip() for line in f.readlines()]

# Load processed URLs
processed_urls_file = "processed_urls.json"
try:
    with open(processed_urls_file, "r") as f:
        processed_urls = set(json.load(f))
except FileNotFoundError:
    processed_urls = set()

# Filter new URLs
new_urls = [url for url in urls if url not in processed_urls]

# Load new documents
if new_urls:
    loader = SimpleWebPageReader()
    documents = loader.load_data(urls=new_urls)
else:
    documents = []

# Update processed URLs
processed_urls.update(new_urls)
with open(processed_urls_file, "w") as f:
    json.dump(list(processed_urls), f)

# Process new documents
if documents:
    # Split documents into nodes
    nodes = SimpleNodeParser().get_nodes_from_documents(documents)

    # Set up Vector Search client
    client = VectorSearchClient()
    endpoint_name = "llamaindex_dbx_vector_store_test_endpoint"

    # Create vector search index
    databricks_index = client.create_direct_access_index(
        endpoint_name=endpoint_name,
        index_name="my_catalog.my_schema.my_test_table",
        primary_key="my_primary_key_name",
        embedding_dimension=1536,
        embedding_vector_column="my_embedding_vector_column_name",
        schema={
            "my_primary_key_name": "string",
            "my_embedding_vector_column_name": "array<double>",
            "text": "string",
            "doc_id": "string",
        },
    )

    # Create vector store
    vector_store = DatabricksVectorSearch(index=databricks_index, text_column="text")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    # Optional: Persist index locally
    # index.storage_context.persist(persist_dir="./storage")
else:
    # Load existing index (assumes index is set up)
    pass

# Create query engine
query_engine = index.as_query_engine(llm=llm)

# Test query
response = query_engine.query("What is the main topic of the web pages?")
print(response)