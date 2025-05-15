import os
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from databricks_langchain import ChatDatabricks
import chromadb
from llama_index.core.retrievers import VectorIndexRetriever

# Step 1: Initialize Databricks LLM
# Explanation: Uses Meta Llama 3.3 70B Instruct for conversational RAG queries.
print("Initializing Databricks LLM...")
try:
    llm = ChatDatabricks(
        endpoint="databricks-meta-llama-3-3-70b-instruct",
        host=os.getenv("DATABRICKS_HOST", "https://your-workspace.cloud.databricks.com"),
        token=os.getenv("DATABRICKS_TOKEN", "your-access-token"),
        temperature=0.1,
        max_tokens=250
    )
except Exception as e:
    print(f"Failed to initialize LLM: {str(e)}")
    exit(1)

# Step 2: Configure global settings for LLM
# Explanation: Sets the LLM globally for LlamaIndex 0.12.x query engine.
print("Configuring global settings...")
try:
    Settings.llm = llm
except Exception as e:
    print(f"Failed to configure settings: {str(e)}")
    exit(1)

# Step 3: Set up ChromaDB client and collection
# Explanation: Reuses the same collection created by the embedding program.
print("Setting up ChromaDB...")
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "pdf_embeddings"
    collection = client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
except Exception as e:
    print(f"Failed to set up ChromaDB: {str(e)}")
    exit(1)

# Step 4: Verify ChromaDB contents
# Explanation: Checks if the collection has any embeddings.
print("Verifying ChromaDB contents...")
try:
    collection_data = collection.get()
    num_embeddings = len(collection_data['ids'])
    print(f"Found {num_embeddings} embeddings in collection.")
    if num_embeddings == 0:
        print("Error: ChromaDB collection is empty. Run embed_pdfs.py to process PDFs.")
        exit(1)
except Exception as e:
    print(f"Failed to verify ChromaDB contents: {str(e)}")
    exit(1)

# Step 5: Load VectorStoreIndex
# Explanation: Loads precomputed embeddings; no embedding model needed here.
print("Loading VectorStoreIndex...")
try:
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
except Exception as e:
    print(f"Failed to load index: {str(e)}")
    exit(1)

# Step 6: Create query engine with retriever
# Explanation: Adds a retriever to inspect retrieved nodes before LLM processing.
print("Creating query engine...")
try:
    retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
    query_engine = index.as_query_engine(retriever=retriever)
except Exception as e:
    print(f"Failed to create query engine: {str(e)}")
    exit(1)

# Step 7: Interactive query loop with debugging
print("\nReady to process queries.")
while True:
    query = input("Enter your query (or 'quit' to exit): ")
    if query.lower() == 'quit':
        break
    print(f"\nProcessing query: {query}")
    try:
        # Check retrieved nodes
        retrieved_nodes = retriever.retrieve(query)
        print(f"Retrieved {len(retrieved_nodes)} nodes.")
        if not retrieved_nodes:
            print("Warning: No relevant nodes retrieved. Try a different query or check embeddings.")
            continue
        
        # Log retrieved content
        for i, node in enumerate(retrieved_nodes):
            print(f"Node {i+1} (score: {node.score:.3f}): {node.get_text()[:100]}...")

        # Execute query
        response = query_engine.query(query)
        if not response or str(response).strip() == "":
            print("Error: Empty response from LLM. Check LLM endpoint or query relevance.")
            continue
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        continue

print("Query process completed.")