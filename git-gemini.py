# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
# 1. Install necessary libraries:
#    pip install llama-index>=0.10.0 chromadb>=1.0.0 llama-index-llms-databricks llama-index-embeddings-databricks python-dotenv
#
# 2. Databricks Credentials:
#    - Set up Databricks token authentication. Create a `.env` file in the
#      same directory as your script with:
#      DATABRICKS_HOST=<Your Databricks Host, e.g., https://your-workspace.cloud.databricks.com>
#      DATABRICKS_TOKEN=<Your Databricks Personal Access Token>
#    - Ensure the token has permissions to invoke the specified model endpoints.
#
# 3. Git Log Data:
#    - Generate your raw git log: `git log --stat > raw_git_log.txt`
#    - Place the `raw_git_log.txt` file in the same directory or specify the path.
#    - The script includes a function to format this raw log.
#    - Alternatively, use the provided `sample_formatted_git_log.txt`.
# ---------------------------------------------------------------------------

import os
import re
import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.databricks import DatabricksEmbedding
from llama_index.llms.databricks import Databricks
import logging
import sys

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Databricks Model Configuration (Replace with your desired models)
# Find available models in your Databricks workspace (e.g., via UI or API)
# Common embedding model: databricks-bge-large-en
# Common LLM: databricks-dbrx-instruct or databricks-meta-llama-3-70b-instruct
DATABRICKS_EMBEDDING_MODEL = "databricks-bge-large-en"
DATABRICKS_LLM = "databricks-dbrx-instruct"

# Data and ChromaDB Configuration
DATA_DIR = "./git_log_data" # Directory to store formatted logs
RAW_LOG_FILE = "raw_git_log.txt" # Input raw git log file
FORMATTED_LOG_FILE = "formatted_git_log.txt" # Output formatted file
CHROMA_DB_PATH = "./chroma_db_git" # Path to store ChromaDB data
CHROMA_COLLECTION_NAME = "git_log_commits" # ChromaDB collection name

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# --- Helper Function: Format Git Log ---
def format_git_log(raw_log_path, formatted_log_path):
    """
    Parses a raw 'git log --stat' output and formats it into a structured
    text file, with each commit as a separate entry.

    Args:
        raw_log_path (str): Path to the raw git log file.
        formatted_log_path (str): Path to save the formatted log file.
    """
    try:
        with open(raw_log_path, 'r', encoding='utf-8') as f_in:
            raw_log_content = f_in.read()
    except FileNotFoundError:
        logging.error(f"Error: Raw log file not found at {raw_log_path}")
        print(f"Error: Raw log file not found at {raw_log_path}")
        print("Please generate it using 'git log --stat > raw_git_log.txt'")
        return False
    except Exception as e:
        logging.error(f"Error reading raw log file: {e}")
        print(f"Error reading raw log file: {e}")
        return False

    commits = []
    # Split log into individual commits based on the "commit " line
    raw_commits = re.split(r'\ncommit ', raw_log_content)
    # The first element might be empty or preamble, handle the first commit separately
    if not raw_log_content.startswith('commit '):
         if len(raw_commits) > 0:
             raw_commits.pop(0) # remove potential preamble if log doesn't start with commit
    if raw_log_content.startswith('commit '):
         raw_commits[0] = 'commit ' + raw_commits[0] # Add back the 'commit ' prefix for the first one

    for i, raw_commit in enumerate(raw_commits):
        if not raw_commit.strip():
            continue

        # Add back the 'commit ' prefix if it was removed by split (except for the first one handled above)
        if not raw_commit.startswith('commit ') and i > 0:
             commit_text = 'commit ' + raw_commit.strip()
        else:
             commit_text = raw_commit.strip()


        # Extract key information using regex (adjust regex if your log format differs)
        commit_hash_match = re.search(r'^commit\s+([0-9a-f]+)', commit_text, re.MULTILINE)
        author_match = re.search(r'^Author:\s+(.+)', commit_text, re.MULTILINE)
        date_match = re.search(r'^Date:\s+(.+)', commit_text, re.MULTILINE)
        message_match = re.search(r'^\s{4}(.+?)\n\n', commit_text, re.DOTALL) # Capture message block

        commit_hash = commit_hash_match.group(1) if commit_hash_match else "N/A"
        author = author_match.group(1).strip() if author_match else "N/A"
        date = date_match.group(1).strip() if date_match else "N/A"
        message = message_match.group(1).strip() if message_match else "N/A"

        # Extract stats (files changed, insertions, deletions)
        stats_summary_match = re.search(r'(\d+)\s+file[s]?\s+changed(?:,\s*(\d+)\s+insertion[s]?\(\+\))?(?:,\s*(\d+)\s+deletion[s]?\(-\))?', commit_text)
        files_changed = int(stats_summary_match.group(1)) if stats_summary_match else 0
        insertions = int(stats_summary_match.group(2)) if stats_summary_match and stats_summary_match.group(2) else 0
        deletions = int(stats_summary_match.group(3)) if stats_summary_match and stats_summary_match.group(3) else 0

        # Extract detailed file stats (optional, but useful for some queries)
        detailed_stats_match = re.search(r'\n\s(.*?)\s+\|\s+\d+\s+[+-]+$', commit_text, re.MULTILINE | re.DOTALL)
        detailed_stats = ""
        if detailed_stats_match:
             # Find the start of the detailed stats section more reliably
             stat_lines_start = commit_text.find(detailed_stats_match.group(1).split('\n')[0].strip())
             if stat_lines_start != -1:
                 stat_lines_end = commit_text.find(' file', stat_lines_start) # Find the summary line after stats
                 if stat_lines_end != -1:
                      detailed_stats_block = commit_text[stat_lines_start:stat_lines_end].strip()
                      # Clean up potential leading/trailing whitespace per line
                      detailed_stats = "\n".join(line.strip() for line in detailed_stats_block.split('\n') if line.strip())


        # Format the commit entry
        formatted_entry = (
            f"Commit: {commit_hash}\n"
            f"Author: {author}\n"
            f"Date: {date}\n"
            f"Message: {message}\n"
            f"Files Changed: {files_changed}\n"
            f"Insertions: {insertions}\n"
            f"Deletions: {deletions}\n"
            # f"Stats:\n{detailed_stats}\n" # Optionally include detailed stats
            f"---\n" # Separator
        )
        commits.append(formatted_entry)

    # Write formatted data to file
    try:
        with open(formatted_log_path, 'w', encoding='utf-8') as f_out:
            f_out.write("\n".join(commits))
        logging.info(f"Successfully formatted {len(commits)} commits to {formatted_log_path}")
        print(f"Successfully formatted {len(commits)} commits to {formatted_log_path}")
        return True
    except Exception as e:
        logging.error(f"Error writing formatted log file: {e}")
        print(f"Error writing formatted log file: {e}")
        return False

# --- Main Pipeline ---
def run_git_log_rag_pipeline():
    """
    Executes the RAG pipeline: format data, setup models, index, and query.
    """
    formatted_log_path = os.path.join(DATA_DIR, FORMATTED_LOG_FILE)

    # Step 1: Format the Git Log (if raw file exists and formatted doesn't, or if forced)
    # You can comment this out if you provide the formatted file directly
    print(f"Checking for raw log file at: {RAW_LOG_FILE}")
    if os.path.exists(RAW_LOG_FILE):
         print("Raw log file found. Formatting...")
         if not format_git_log(RAW_LOG_FILE, formatted_log_path):
              print("Failed to format git log. Exiting.")
              return
    elif not os.path.exists(formatted_log_path):
         print(f"Error: Neither raw log file ('{RAW_LOG_FILE}') nor formatted log file ('{formatted_log_path}') found.")
         print("Please provide one of them.")
         # Try using the sample file if it exists
         sample_file_path = "sample_formatted_git_log.txt"
         if os.path.exists(sample_file_path):
             print(f"Attempting to use '{sample_file_path}'...")
             import shutil
             shutil.copy(sample_file_path, formatted_log_path)
             print(f"Copied '{sample_file_path}' to '{formatted_log_path}'.")
         else:
             print(f"Sample file '{sample_file_path}' also not found. Exiting.")
             return # Exit if no data source is available


    # Step 2: Configure LlamaIndex Settings (Embeddings and LLM)
    print("Configuring Databricks models...")
    try:
        # It's crucial that DATABRICKS_HOST and DATABRICKS_TOKEN are set in your environment
        if not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"):
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN environment variables must be set.")

        Settings.embed_model = DatabricksEmbedding(model=DATABRICKS_EMBEDDING_MODEL)
        Settings.llm = Databricks(model=DATABRICKS_LLM)
        # Optional: Adjust chunk size and overlap if needed. Defaults are often reasonable.
        # Settings.chunk_size = 512
        # Settings.chunk_overlap = 20
        print("Databricks models configured successfully.")
    except Exception as e:
        print(f"Error configuring Databricks models: {e}")
        print("Please ensure DATABRICKS_HOST and DATABRICKS_TOKEN are set correctly in your .env file or environment.")
        return

    # Step 3: Setup ChromaDB Vector Store
    print(f"Setting up ChromaDB client at path: {CHROMA_DB_PATH}")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    print(f"Getting or creating ChromaDB collection: {CHROMA_COLLECTION_NAME}")
    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("ChromaDB setup complete.")

    # Step 4: Load Data and Build Index
    # Check if the index already exists in the collection to avoid re-indexing
    # Note: This is a basic check. For production, you might need more robust versioning or checking.
    existing_docs_count = chroma_collection.count()
    print(f"Found {existing_docs_count} existing documents in Chroma collection.")

    index = None
    if existing_docs_count > 0:
        print("Loading existing index from ChromaDB...")
        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
            )
            print("Existing index loaded successfully.")
        except Exception as e:
            print(f"Error loading existing index: {e}. Will attempt to build a new one.")
            # Fallback to building index if loading fails

    if index is None:
        print(f"Building new index from data in: {DATA_DIR}")
        try:
            # Load documents from the formatted log file(s) in the directory
            # SimpleDirectoryReader is good for text files. It reads each file as a Document.
            # LlamaIndex automatically handles chunking based on Settings.chunk_size.
            # It reads files sequentially by default. For very large datasets or many files,
            # explore `num_workers` in SimpleDirectoryReader or custom loading logic.
            reader = SimpleDirectoryReader(input_dir=DATA_DIR)
            documents = reader.load_data(show_progress=True)

            if not documents:
                 print(f"No documents found in {DATA_DIR}. Make sure '{FORMATTED_LOG_FILE}' exists and has content.")
                 return

            print(f"Loaded {len(documents)} document(s). Indexing...")
            # Embeddings and storage happen here. LlamaIndex uses Settings.embed_model.
            # Embeddings are generated in batches (default batch size 10).
            # You can adjust Settings.embed_batch_size if needed, depending on the model limits and memory.
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True,
            )
            print("New index built and stored in ChromaDB successfully.")
        except Exception as e:
            print(f"Error building index: {e}")
            return

    # Step 5: Query the Index
    if index:
        print("\n--- Querying Engine Ready ---")
        query_engine = index.as_query_engine(
            # Optional: Adjust similarity_top_k to retrieve more/fewer relevant chunks
            # similarity_top_k=3
        )

        queries = [
            "Which author has the highest number of commits?",
            "List the authors and their total commit counts.",
            "How many commits did 'John Doe <john.doe@example.com>' make?", # Replace with an actual author from your log
            "Summarize the main changes made by 'Jane Smith <jane.s@example.com>'.", # Replace with an actual author
            "Which files are most frequently changed across all commits?",
            "What was the total number of insertions and deletions in the log?"
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            try:
                response = query_engine.query(query)
                print(f"Response: {response}")
                # You can also inspect the source nodes (chunks) the LLM used:
                # print("Source Nodes:")
                # for node in response.source_nodes:
                #     print(f" - Score: {node.score:.4f}, Text: {node.text[:100]}...") # Print snippet
            except Exception as e:
                print(f"Error querying index: {e}")
    else:
        print("Index could not be loaded or built. Querying is not possible.")

# --- Run the pipeline ---
if __name__ == "__main__":
    run_git_log_rag_pipeline()
```text
Commit: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
Author: John Doe <john.doe@example.com>
Date: Mon May 12 10:30:00 2025 +0800
Message: Feat: Add user authentication module

Implemented basic login and registration functionality.
Includes password hashing and session management.
Files Changed: 3
Insertions: 150
Deletions: 10
---
Commit: f0e9d8c7b6a5f4e3d2c1b0a9f8e7d6c5b4a3f2e1
Author: Jane Smith <jane.s@example.com>
Date: Sun May 11 15:45:12 2025 +0800
Message: Fix: Correct calculation error in reporting

The monthly summary report had an off-by-one error in totals.
Added unit tests to cover this case.
Files Changed: 2
Insertions: 25
Deletions: 5
---
Commit: 9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b
Author: John Doe <john.doe@example.com>
Date: Sat May 10 09:00:05 2025 +0800
Message: Refactor: Improve database query performance

Optimized SQL queries for the dashboard loading.
Reduced query time by ~30%.
Files Changed: 5
Insertions: 40
Deletions: 60
---
Commit: 1f2e3d4c5b6a7f8e9d0c1b2a3f4e5d6c7b8a9f0e
Author: Alice Brown <alice.b@example.com>
Date: Fri May 9 18:12:34 2025 +0800
Message: Docs: Update README with setup instructions

Added detailed steps for local development environment setup.
Included prerequisites and common troubleshooting tips.
Files Changed: 1
Insertions: 55
Deletions: 0
---
Commit: e5d6c7b8a9f0e1f2a3b4c5d6e7f8a9b0a1b2c3d4
Author: John Doe <john.doe@example.com>
Date: Fri May 9 11:05:00 2025 +0800
Message: Chore: Update dependencies

Updated Python packages to latest stable versions.
Resolved minor compatibility issues.
Files Changed: 2
Insertions: 15
Deletions: 15
---