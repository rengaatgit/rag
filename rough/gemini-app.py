# main_rag_app.py
from pathlib import Path
import os
import hashlib
import shutil
import chromadb
import gradio as gr
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document
)
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.config import Settings as ChromaSettings

# --- Configuration ---
# Model Configurations
# For LM Studio, ensure it's running and the server is active.
# The base_url should point to your LM Studio server's endpoint.
# If your model in LM Studio doesn't use 'nomic-ai/nomic-embed-text-v1.5' as its API model name,
# you might need to adjust 'local:nomic-ai/nomic-embed-text-v1.5'.
# Sometimes, LM Studio uses a generic 'local-model' or the model's filename.
# Check LM Studio's server logs for the correct model identifier if issues arise.
# The API key "lm-studio" is a placeholder and might not be needed if your LM Studio server doesn't require one.
EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5" # Model name as known by LM Studio or its API
# Note: LlamaIndex's resolve_embed_model format for OpenAI-compatible APIs is "local:<model_name>"
# or "local:<base_url>/<model_name>" if the base_url is not the default http://localhost:1234/v1
# For LM Studio, if it's running on http://localhost:5000, you might use:
# EMBEDDING_MODEL_API_BASE = "http://localhost:5000/v1" # Example LM Studio API base
# EMBEDDING_MODEL_SPEC = f"local:{EMBEDDING_MODEL_API_BASE}/{EMBEDDING_MODEL_ID}"
# However, resolve_embed_model can often infer if it's a known local model type.
# We will try with a simpler specification first.
# For Ollama, ensure 'llama3.2:1b' is pulled and Ollama server is running.
LLM_MODEL_ID = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama API base

# ChromaDB Configuration
CHROMA_PERSIST_DIR = "./data/chroma_db_store"
CHROMA_COLLECTION_NAME = "pdf_document_embeddings"
PROCESSED_FILES_LOG = "processed_files.log" # Simple log for processed files

# Data Directory for PDFs
PDF_UPLOAD_DIR = "./data/uploaded_pdfs"
Path(PDF_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

# Global variable for query engine
query_engine = None
processed_file_hashes = set()

# --- Checkpoint 1: Initialize Models and Settings ---
def initialize_settings_and_models():
    """
    Initializes LlamaIndex settings with embedding and LLM models.
    This is a critical first step. If models are not accessible,
    the pipeline cannot proceed.
    """
    global llm, embed_model
    print("--- Checkpoint 1: Initializing Models and Settings ---")
    try:
        # Initialize Embedding Model from LM Studio (OpenAI compatible endpoint)
        # Note: `resolve_embed_model` with "local:..." assumes an OpenAI-compatible API.
        # Ensure your LM Studio server is configured to provide this for the embedding model.
        # The format "local:<model_name>" tells LlamaIndex to use an OpenAI client
        # pointed at http://localhost:1234/v1 by default.
        # If your LM Studio runs on a different port, you need to specify it:
        # e.g., Settings.embed_model = resolve_embed_model("local:http://localhost:5000/v1/nomic-ai/nomic-embed-text-v1.5")
        # For simplicity, we assume LM Studio is on default OpenAI-compatible port or LlamaIndex handles it.
        # A common way is to set OPENAI_API_BASE env var or pass it to OpenAIEmbedding directly.
        # Here, we rely on resolve_embed_model's capabilities.
        # If LM Studio uses a specific API key, set OPENAI_API_KEY env var.
        # For "text-embedding-nomic-embed-text-v1.5-embedding", the model name in the API call
        # might just be "nomic-embed-text-v1.5" or similar. Check LM Studio logs.
        # We will use a generic placeholder that often works with LM Studio if it's exposing an OpenAI-like endpoint.
        # The key "lm-studio" is often a dummy key for local servers.
        # LlamaIndex uses `embed_model = OpenAIEmbedding(model=model_name, api_base=api_base, api_key=api_key)` under the hood.
        
        # Correct way to specify local OpenAI-compatible embedding model:
        # Option 1: Set environment variables (OPENAI_API_BASE, OPENAI_API_KEY)
        # os.environ["OPENAI_API_BASE"] = "http://localhost:5001/v1" # Adjust to your LM Studio port
        # os.environ["OPENAI_API_KEY"] = "your-lm-studio-api-key-if-any" # Often not needed for local
        # Settings.embed_model = resolve_embed_model(f"local:{EMBEDDING_MODEL_ID}")
        
        # Option 2: Directly instantiate (more explicit control)
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(
            model_name=EMBEDDING_MODEL_ID, # This is the model identifier LM Studio uses in its API
            api_base="http://localhost:1234/v1", # CHANGE THIS TO YOUR LM STUDIO SERVER URL + /v1
            api_key="lm-studio" # Placeholder, change if LM Studio requires a key
        )
        Settings.embed_model = embed_model
        print(f"Embedding Model: Configured to use {EMBEDDING_MODEL_ID} via LM Studio at http://localhost:1234/v1")

        # Initialize LLM from Ollama
        llm = Ollama(model=LLM_MODEL_ID, base_url=OLLAMA_BASE_URL, request_timeout=120.0) # Increased timeout for potentially slow local models
        Settings.llm = llm
        print(f"LLM: Configured to use {LLM_MODEL_ID} via Ollama at {OLLAMA_BASE_URL}")

        # LlamaIndex default chunk size and overlap are generally good starting points.
        # These can be customized via Settings.chunk_size and Settings.chunk_overlap if needed.
        # Settings.chunk_size = 512 # Example: if you want to override default
        # Settings.chunk_overlap = 20  # Example: if you want to override default
        print(f"LlamaIndex Settings: Chunk Size = {Settings.chunk_size}, Chunk Overlap = {Settings.chunk_overlap}")
        print("--- Models and Settings Initialized Successfully ---")
        return True
    except Exception as e:
        print(f"Error initializing models/settings: {e}")
        gr.Error(f"Failed to initialize models. Check server endpoints and model names. Error: {e}")
        return False

# --- Checkpoint 2: Load Processed File Hashes ---
def load_processed_files_log():
    """Loads the hashes of previously processed files to avoid re-processing."""
    global processed_file_hashes
    print("--- Checkpoint 2: Loading Processed Files Log ---")
    try:
        if os.path.exists(PROCESSED_FILES_LOG):
            with open(PROCESSED_FILES_LOG, "r") as f:
                processed_file_hashes = set(line.strip() for line in f)
            print(f"Loaded {len(processed_file_hashes)} entries from processed files log.")
        else:
            print("No processed files log found. Starting fresh.")
        print("--- Processed Files Log Loaded Successfully ---")
        return True
    except Exception as e:
        print(f"Error loading processed files log: {e}")
        # Non-critical, can proceed but might re-process files.
        return True # Still return true as it's not a fatal error for startup

def add_to_processed_files_log(file_hash):
    """Adds a file hash to the log."""
    global processed_file_hashes
    try:
        with open(PROCESSED_FILES_LOG, "a") as f:
            f.write(file_hash + "\n")
        processed_file_hashes.add(file_hash)
        print(f"Added hash {file_hash} to processed files log.")
    except Exception as e:
        print(f"Error adding to processed files log: {e}")


def get_file_hash(filepath):
    """Computes SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# --- Checkpoint 3: Setup ChromaDB Vector Store ---
def setup_vector_store():
    """
    Sets up the ChromaDB client and collection. This is where embeddings will be stored.
    If the collection already exists, it will be loaded.
    """
    print("--- Checkpoint 3: Setting up ChromaDB Vector Store ---")
    try:
        db_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=ChromaSettings(anonymized_telemetry=False))
        # ChromaDB's get_or_create_collection is idempotent
        chroma_collection = db_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            # If using a specific embedding function with ChromaDB directly (not via LlamaIndex's store abstraction)
            # you might need to specify it here. However, LlamaIndex's ChromaVectorStore handles this.
            # embedding_function= ... # Not needed when LlamaIndex manages embeddings
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        print(f"ChromaDB Collection '{CHROMA_COLLECTION_NAME}' ready at {CHROMA_PERSIST_DIR}")
        print("--- ChromaDB Vector Store Setup Successfully ---")
        return vector_store
    except Exception as e:
        print(f"Error setting up ChromaDB: {e}")
        gr.Error(f"Failed to setup ChromaDB. Check permissions and path. Error: {e}")
        return None

# --- Checkpoint 4: Load Documents and Create/Update Index ---
def load_and_embed_documents(file_paths, vector_store):
    """
    Loads PDF documents from the given file paths, checks for duplicates,
    embeds new documents, and stores them in the vector store.
    LlamaIndex handles chunking and other parameters internally based on Settings.
    """
    global query_engine # To update the query engine after new docs are added
    print("--- Checkpoint 4: Loading and Embedding Documents ---")
    if not file_paths:
        print("No files provided for embedding.")
        return "No files uploaded or all files were already processed.", []

    new_files_processed_count = 0
    skipped_files_count = 0
    processed_file_names = []
    messages = []

    # Temporary directory for SimpleDirectoryReader
    temp_docs_dir = os.path.join(PDF_UPLOAD_DIR, "temp_processing")
    os.makedirs(temp_docs_dir, exist_ok=True)
    
    # Copy uploaded files to a temporary directory for SimpleDirectoryReader
    # This is because SimpleDirectoryReader works best with a directory.
    valid_files_to_process_paths = []
    files_to_load_for_llama_index = []

    for uploaded_file_obj in file_paths:
        original_filepath = uploaded_file_obj.name # Gradio file object
        filename = os.path.basename(original_filepath)
        
        # Copy to a persistent location first to calculate hash and store
        persistent_filepath = os.path.join(PDF_UPLOAD_DIR, filename)
        shutil.copyfile(original_filepath, persistent_filepath)
        print(f"Copied '{filename}' to '{persistent_filepath}'")

        file_hash = get_file_hash(persistent_filepath)

        if file_hash in processed_file_hashes:
            messages.append(f"Ignoring '{filename}': Already processed.")
            print(f"Skipping '{filename}' (hash: {file_hash}), already processed.")
            skipped_files_count += 1
            continue
        
        # If not skipped, prepare for processing
        temp_processing_path = os.path.join(temp_docs_dir, filename)
        shutil.copyfile(persistent_filepath, temp_processing_path) # Copy to temp dir for SimpleDirectoryReader
        files_to_load_for_llama_index.append(temp_processing_path)
        valid_files_to_process_paths.append(persistent_filepath) # Keep track of original path for logging hash
        processed_file_names.append(filename)


    if not files_to_load_for_llama_index:
        shutil.rmtree(temp_docs_dir) # Clean up temp directory
        if skipped_files_count > 0:
            return "\n".join(messages) + "\nNo new files to process.", processed_file_names
        else:
            return "No valid new files found to process.", []


    try:
        print(f"Processing {len(files_to_load_for_llama_index)} new file(s) from: {temp_docs_dir}")
        # SimpleDirectoryReader will load all files from the specified directory
        # LlamaIndex will use Settings.embed_model for embeddings.
        # LlamaIndex handles chunk_size and chunk_overlap based on global Settings.
        reader = SimpleDirectoryReader(input_dir=temp_docs_dir, required_exts=[".pdf"])
        documents = reader.load_data(show_progress=True)
        
        if not documents:
            messages.append("No content could be extracted from the new PDF files.")
            shutil.rmtree(temp_docs_dir) # Clean up
            return "\n".join(messages), processed_file_names

        print(f"Loaded {len(documents)} document chunks from new PDFs.")

        # Add documents to the index (and thus to ChromaDB via the vector_store)
        # If an index already exists and is connected to the vector_store,
        # new documents can be inserted.
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Check if an index already exists. If so, load it and insert.
        # Otherwise, create a new one.
        try:
            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            print("Existing index loaded from vector store. Inserting new documents.")
            index.insert_nodes(documents) # Insert new documents
        except Exception as e: # A bit broad, but Chroma/LlamaIndex might raise various things if store is empty/new
            print(f"Could not load existing index (may be first run or empty store): {e}. Creating new index.")
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            print("New index created and documents embedded.")

        # Persist hashes of successfully processed files
        for original_filepath, temp_path in zip(valid_files_to_process_paths, files_to_load_for_llama_index):
            # We use the hash from the persistent_filepath
            file_hash = get_file_hash(original_filepath) # Re-calc or retrieve from earlier
            add_to_processed_files_log(file_hash)
        
        new_files_processed_count = len(files_to_load_for_llama_index)
        messages.append(f"Successfully processed and embedded {new_files_processed_count} new PDF file(s).")
        
        # Update the query engine to use the latest index
        query_engine = index.as_query_engine(
            similarity_top_k=3, # Configure retrieval: number of top similar chunks to retrieve
            # response_mode="compact" # Configure response synthesis
        )
        print("Query engine updated with new documents.")
        print("--- Document Loading and Embedding Successful ---")

    except Exception as e:
        error_msg = f"Error during document loading/embedding: {e}"
        print(error_msg)
        messages.append(f"Failed to process some files. Error: {error_msg}")
        gr.Error(error_msg) # Show error in Gradio UI
    finally:
        # Clean up the temporary processing directory
        shutil.rmtree(temp_docs_dir)
        print(f"Cleaned up temporary directory: {temp_docs_dir}")

    final_message = "\n".join(messages)
    return final_message, processed_file_names


# --- Checkpoint 5: Setup Query Engine ---
def setup_query_engine(vector_store):
    """
    Sets up the query engine from the existing vector store.
    This is called on startup and after new documents are indexed.
    """
    global query_engine
    print("--- Checkpoint 5: Setting up Query Engine ---")
    try:
        # Try to load an existing index from the vector store
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        query_engine = index.as_query_engine(
            similarity_top_k=3, # How many context chunks to retrieve
            # Example of other parameters:
            # response_mode="refine" # How the LLM generates the response based on context
        )
        print("Query Engine ready. Loaded from existing ChromaDB index.")
        print("--- Query Engine Setup Successfully ---")
        return True
    except Exception as e:
        # This can happen if the vector store is empty (no documents indexed yet)
        print(f"Could not load index from vector store (likely empty): {e}. Query engine will be set up after first document processing.")
        # query_engine remains None, will be initialized after first successful embedding
        return True # Not a fatal error for startup, but querying won't work yet.


# --- Gradio Interface Functions ---
def process_uploaded_files(files_list, chat_history):
    """
    Gradio action for file uploads. Invokes the embedding pipeline.
    """
    if not files_list:
        return chat_history + [("System", "No files were uploaded.")]

    # Ensure models are initialized (in case of restart or issue)
    if not (Settings.llm and Settings.embed_model):
        init_success = initialize_settings_and_models()
        if not init_success:
             return chat_history + [("System", "Model initialization failed. Cannot process files.")]
    
    vector_store = setup_vector_store() # Ensure DB is ready
    if not vector_store:
        return chat_history + [("System", "ChromaDB setup failed. Cannot process files.")]

    status_message, processed_names = load_and_embed_documents(files_list, vector_store)
    
    if processed_names:
        user_message = f"Uploaded and initiated processing for: {', '.join(processed_names)}."
    else:
        user_message = "File upload attempt made." # Generic if no new files processed

    # Add user's implicit action (uploading) and system's response to chat
    updated_chat_history = chat_history + [(user_message, status_message)]
    return updated_chat_history, None # Return None to clear the file upload component


def chat_with_bot(user_input, chat_history):
    """
    Gradio action for chat input. Queries the LLM.
    """
    global query_engine
    if query_engine is None:
        response_text = "The document query engine is not ready. Please upload and process PDF documents first."
    elif not user_input.strip():
        response_text = "Please enter a query."
    else:
        print(f"User query: {user_input}")
        try:
            response = query_engine.query(user_input)
            response_text = str(response)
            # You can also access source nodes for citation:
            # for s_node in response.source_nodes:
            #    print(f"Source Node ID: {s_node.node_id}, Score: {s_node.score}")
            #    print(f"Source Node Text: {s_node.text[:150]}...") # Print snippet
        except Exception as e:
            response_text = f"Error querying the LLM: {e}"
            print(response_text)
            gr.Error(response_text)

    chat_history.append((user_input, response_text))
    return chat_history, "" # Return "" to clear the textbox


# --- Main Application Setup and Launch ---
def main():
    """
    Main function to initialize components and launch the Gradio app.
    """
    print("Starting RAG PDF Analysis Application...")

    # --- Initial Setup Sequence ---
    # 1. Initialize LlamaIndex Settings (LLM and Embed Model)
    if not initialize_settings_and_models():
        print("CRITICAL ERROR: Model initialization failed. Application cannot start.")
        # Optionally, could launch Gradio with a message, but better to fail early.
        return

    # 2. Load processed file log
    load_processed_files_log()

    # 3. Setup Vector Store (ChromaDB)
    vector_store = setup_vector_store()
    if not vector_store:
        print("CRITICAL ERROR: Vector store setup failed. Application cannot start.")
        return

    # 4. Setup Query Engine (load existing index if available)
    # This will allow querying existing documents on startup.
    # If no docs, query_engine remains None until first upload.
    setup_query_engine(vector_store)

    print("\n--- Application Ready ---")
    print(f"PDF Upload Directory: {os.path.abspath(PDF_UPLOAD_DIR)}")
    print(f"ChromaDB Persist Directory: {os.path.abspath(CHROMA_PERSIST_DIR)}")
    print(f"Processed Files Log: {os.path.abspath(PROCESSED_FILES_LOG)}")
    if query_engine:
        print("Query engine is active with existing data.")
    else:
        print("Query engine will be initialized after document processing.")
    print("LM Studio for embeddings should be running at: http://localhost:5001/v1 (or as configured)") # Remind user
    print("Ollama for LLM should be running at: http://localhost:11434 (or as configured)") # Remind user
    print("--- Launching Gradio Interface ---")


    # --- Gradio Interface Definition ---
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # RAG Pipeline: PDF Document Analysis 📄🔍🤖
            Upload PDF documents, and then ask questions about their content.
            **Models:**
            * LLM: `llama3.2:1b` (via Ollama)
            * Embedding: `text-embedding-nomic-embed-text-v1.5-embedding` (via LM Studio)
            **Setup Notes:**
            1. Ensure Ollama server is running with `llama3.2:1b` model pulled.
            2. Ensure LM Studio is running, serving the `text-embedding-nomic-embed-text-v1.5-embedding` model (or equivalent)
               at an OpenAI-compatible endpoint (default assumed: `http://localhost:5001/v1`).
               **Adjust `EMBEDDING_MODEL_API_BASE` in the script if your LM Studio port is different.**
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_uploader = gr.File(
                    label="Upload PDF Documents",
                    file_count="multiple",
                    file_types=[".pdf"],
                    # type="filepath" # 'filepath' type for gr.File is often simpler
                )
                # process_button = gr.Button("Process Uploaded PDFs") # Implicit processing on upload for now

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat with your Documents",
                    bubble_full_width=False,
                    height=600
                )
                query_textbox = gr.Textbox(
                    label="Enter your query:",
                    placeholder="e.g., What are the main conclusions of document X?",
                    show_label=False,
                    lines=2,
                )
                submit_button = gr.Button("Ask", variant="primary")

        # --- Event Handlers ---
        # When files are uploaded, process them.
        # The `upload` event of gr.File directly passes the list of file objects (TempFilePath)
        file_uploader.upload(
            fn=process_uploaded_files,
            inputs=[file_uploader, chatbot], # Pass current chat history
            outputs=[chatbot, file_uploader] # Update chat, clear uploader
        )

        # Handle chat submission (Enter key or button click)
        query_textbox.submit(
            fn=chat_with_bot,
            inputs=[query_textbox, chatbot],
            outputs=[chatbot, query_textbox] # Update chat, clear textbox
        )
        submit_button.click(
            fn=chat_with_bot,
            inputs=[query_textbox, chatbot],
            outputs=[chatbot, query_textbox]
        )

    # Launch the Gradio app
    demo.queue().launch(share=False) # share=True for public link (use with caution)

if __name__ == "__main__":
    main()

