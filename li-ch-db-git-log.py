import os
import re
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.databricks import DatabricksEmbedding
from llama_index.llms.databricks import Databricks
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import glob

# Set up Databricks credentials
os.environ["DATABRICKS_TOKEN"] = "<your_token>"
os.environ["DATABRICKS_SERVING_ENDPOINT"] = "<your_endpoint>"

# Configure embedding model
embed_model = DatabricksEmbedding(model="databricks-bge-large-en")
Settings.embed_model = embed_model

# Configure LLM
llm = Databricks(model="databricks-dbrx-instruct")
Settings.llm = llm

# Initialize ChromaDB vector store
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("git_logs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Function to parse git log file
def parse_git_log(file_path, repo_name):
    with open(file_path, 'r') as f:
        content = f.read()
    # Split on 'commit ' lines
    commit_blocks = re.split(r'(?m)^commit ', content)[1:]
    documents = []
    for block in commit_blocks:
        lines = block.split('\n')
        hash_line = lines[0].strip()
        author_line = lines[1].strip()
        date_line = lines[2].strip()
        message_start = 4
        while message_start < len(lines) and lines[message_start].strip() == '':
            message_start += 1
        message_lines = []
        stat_lines = []
        in_message = True
        for line in lines[message_start:]:
            if line.strip() == '':
                in_message = False
            elif in_message:
                message_lines.append(line)
            else:
                stat_lines.append(line)
        message = '\n'.join(message_lines).strip()
        stat = '\n'.join(stat_lines).strip()
        text = f"Commit message:\n{message}\n\nStat:\n{stat}"
        metadata = {
            'repository': repo_name,
            'hash': hash_line,
            'author': author_line.split(': ')[1],
            'date': date_line.split(': ')[1],
        }
        doc = Document(text=text, metadata=metadata)
        documents.append(doc)
    return documents

# Load all log files from the logs directory
log_files = glob.glob('logs/*.log')
all_documents = []
for file_path in log_files:
    repo_name = os.path.basename(file_path).replace('.log', '')
    documents = parse_git_log(file_path, repo_name)
    all_documents.extend(documents)

# Create nodes (one per commit to avoid splitting)
nodes = [TextNode(text=doc.text, metadata=doc.metadata) for doc in all_documents]

# Build the vector store index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

# Create a query engine for natural language queries
query_engine = index.as_query_engine()

# Example natural language query
response = query_engine.query("What are the recent commits by Alice?")
print("Recent commits by Alice:", response)

# Example aggregate query (count commits by author)
from collections import Counter
all_nodes = index.vector_store.get_nodes()
author_counts = Counter(node.metadata['author'] for node in all_nodes)
print("Commit counts by author:", author_counts)
