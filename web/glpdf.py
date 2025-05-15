import os
import json
import requests
from llama_index import VectorStoreIndex, StorageContext, QueryEngine
from llama_index.embeddings.databricks import DatabricksEmbedding
from llama_index.llms.databricks import Databricks
from llama_index.vector_stores.databricks import DatabricksVectorSearch
from databricks.vector_search.client import VectorSearchClient
from llama_index.node_parser import SimpleNodeParser
from llama_parse import LlamaParse

# Set up Databricks credentials and models
os.environ["DATABRICKS_TOKEN"] = "<your_databricks_token>"
os.environ["DATABRICKS_SERVING_ENDPOINT"] = "<your_databricks_endpoint>"
embed_model = DatabricksEmbedding(model="databricks-bge-large-en")
llm = Databricks(model="databricks-dolly-v2-12b")

# Set up LlamaParse with API key
os.environ["LLAMA_CLOUD_API_KEY"] = "<your_llama_cloud_api_key>"
parser = LlamaParse(result_type="markdown")

# Function to download PDF from URL
def download_pdf(url, local_dir="pdfs"):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    filename = os.path.join(local_dir, os.path.basename(url))
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
    return filename

# Load PDF URLs from text file
with open("pdf_urls.txt", "r") as f:
    pdf_urls = [line.strip() for line in f.readlines()]

# Load processed PDFs
processed_pdfs_file = "processed_pdfs.json"
try:
    with open(processed_pdfs_file, "r") as f:
        processed_pdfs = set(json.load(f))
except FileNotFoundError:
    processed_pdfs = set()

# Filter new PDFs
new_pdf_urls = [url for url in pdf_urls if url not in processed_pdfs]

# Download new PDFs and get local paths
local_pdf_paths = []
for url in new_pdf_urls:
    local_path = download_pdf(url)
    local_pdf_paths.append(local_path)

# Update processed PDFs
processed_pdfs.update(new_pdf_urls)
with open(processed_pdfs_file, "w") as f:
    json.dump(list(processed_pdfs), f)

# Load documents using LlamaParse
documents = []
for path in local_pdf_paths:
    docs = parser.load_data(path)
    documents.extend(docs)

# Split documents into nodes
nodes = SimpleNodeParser().get_nodes_from_documents(documents)

# Set up Vector Search client
client = VectorSearchClient()
endpoint_name = "llamaindex_dbx_vector_store_test_endpoint"
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

# Create vector store and storage context
vector_store = DatabricksVectorSearch(index=databricks_index, text_column="text")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build index
index = VectorStoreIndex(nodes, storage_context=storage_context)

# Create query engine
query_engine = index.as_query_engine(llm=llm)

# Test query
response = query_engine.query("What is the main topic of the PDF documents?")
print(response)