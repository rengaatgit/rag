"""
Integration of RAG with MCP into the chat UI
"""

import gradio as gr
import time
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Import the RAG with MCP implementation
from rag_with_mcp import RAGWithMCP

# Create uploads directory if it doesn't exist
FILE_UPLOAD_FOLDER = 'data/file_uploads'  # Define your upload folder
Path(FILE_UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)

URL_UPLOAD_FOLDER = 'data/url_uploads'  # Define your URL upload folder
Path(URL_UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)

# Initialize RAG with MCP
rag = RAGWithMCP(data_dir=FILE_UPLOAD_FOLDER)

def save_uploaded_file(file):
    filename = os.path.basename(file.name)
    save_path = os.path.join(FILE_UPLOAD_FOLDER, filename)
    
    if os.path.exists(save_path):
        abs_save_path = os.path.abspath(save_path)
        raise FileExistsError(f"File '{filename}' already exists at '{abs_save_path}'.")

    # For temporary files created by Gradio
    # This part can raise IOError if reading source or writing destination fails
    with open(file.name, "rb") as source_file:
        with open(save_path, "wb") as dest_file:
            dest_file.write(source_file.read())
    
    return save_path

def upload_files(files):
    saved_file_names = []
    skipped_file_names = []
    error_file_names = []
    saved_file_paths = []

    for uploaded_file_obj in files: # Iterate through Gradio's temporary file objects
        # Determine the filename that would be used for saving, based on current logic
        # (basename of the temporary file)
        # This name is used for messages and for checking existence.
        target_filename = os.path.basename(uploaded_file_obj.name)
        try:
            saved_path = save_uploaded_file(uploaded_file_obj) # This might raise FileExistsError or IOError
            saved_file_names.append(target_filename)
            saved_file_paths.append(saved_path)
        except FileExistsError as e:
            print(e) # Log the specific error
            skipped_file_names.append(target_filename)
        except IOError as e:
            print(f"IOError saving {target_filename}: {e}")
            error_file_names.append(f"{target_filename} (IO error)")
        except Exception as e:
            print(f"An unexpected error occurred while saving {target_filename}: {e}")
            error_file_names.append(f"{target_filename} (unknown error)")
    
    # Update the RAG index with new files
    if saved_file_paths:
        try:
            rag.add_documents(saved_file_paths)
            print(f"Added {len(saved_file_paths)} documents to the RAG index")
        except Exception as e:
            print(f"Error adding documents to RAG index: {e}")
            error_file_names.extend([f"{os.path.basename(path)} (indexing error)" for path in saved_file_paths])
    
    message_parts = []
    if saved_file_names:
        message_parts.append(f"📁 Uploaded {len(saved_file_names)} files: {', '.join(saved_file_names)}")
    if skipped_file_names:
        message_parts.append(f"⚠️ Skipped {len(skipped_file_names)} files (already exist): {', '.join(skipped_file_names)}")
    if error_file_names:
        message_parts.append(f"❌ Error saving {len(error_file_names)} files: {', '.join(error_file_names)}")

    return ". ".join(message_parts) if message_parts else "No new files were uploaded or processed."
 
def add_url(history, url):
    if url.strip() != "":
        now = datetime.now()   
        url_dir = URL_UPLOAD_FOLDER
        base_filename = "url_file"
        extension = ".txt"
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        milliseconds = now.strftime("%f")[:3]  # Get the first three digits of microseconds
        file_name = f"{base_filename}_{timestamp_str}_{milliseconds}{extension}"
        file_path = f"{url_dir}/{file_name}"

        print(f"Generated filename: {file_name}")

        try:
            with open(file_path, 'w') as file_object:
                file_object.write(url)
            print(f"Successfully created and wrote to '{file_name}'")
            
            # Add the URL file to the RAG index
            try:
                rag.add_documents([file_path])
                print(f"Added URL content to the RAG index")
            except Exception as e:
                print(f"Error adding URL to RAG index: {e}")
                return history + [[None, f"🔗 URL added: {url} (but failed to index)"]]
                
            return history + [[None, f"🔗 URL added and indexed: {url}"]]
        except IOError as e:
            print(f"An error occurred: {e}")
            return history + [[None, f"❌ Error adding URL: {url} - {str(e)}"]]
    return history

def add_message(history, message):
    if message.strip() != "":
        return history + [[message, None]]
    return history

def bot_response(history):
    # Get the last user message
    user_message = history[-1][0]
    
    # Use RAG with MCP to generate a response
    try:
        response = rag.query(user_message)
    except Exception as e:
        print(f"Error generating response: {e}")
        response = f"I encountered an error while processing your request: {str(e)}"
    
    # Stream the response character by character
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.01)  # Slightly faster than the original for better UX
        yield history

def handle_upload(history, files):
    if files:
        status = upload_files(files)
        history = history + [[None, status]]
    return history

with gr.Blocks(title="RAG Chat with Model Context Protocol") as demo:
    gr.Markdown("# RAG Chat with Model Context Protocol (MCP)")
    gr.Markdown("""
    This chat interface uses a RAG (Retrieval-Augmented Generation) pipeline with Model Context Protocol (MCP) 
    to provide more relevant and accurate responses based on your documents.
    
    - Upload files to include them in the knowledge base
    - Add URLs to include web content
    - Ask questions about your documents
    """)
    
    with gr.Row():
        chatbot = gr.Chatbot(height=500)
    
    with gr.Row():
        # Left side buttons (vertical stack)
        with gr.Column(scale=1, min_width=60):
            with gr.Group():
                upload_btn = gr.UploadButton(
                    "➕ File",
                    file_types=["file", "text", "image", "pdf", "docx"],
                    file_count="multiple",
                    size="sm"
                )
                url_btn = gr.Button("🌐 URL", size="sm")
                url_input = gr.Textbox(
                    placeholder="Paste URL...",
                    visible=False,
                    label="URL Input",
                    scale=0
                )
        
        # Middle chat input
        with gr.Column(scale=8):
            msg = gr.Textbox(show_label=False, placeholder="Type your question about the documents...")
        
        # Right send button
        with gr.Column(scale=1, min_width=80):
            send_btn = gr.Button("Send", variant="primary")

    # Toggle URL input visibility
    url_btn.click(
        lambda: gr.update(visible=True),
        outputs=url_input
    )

    # URL submission handling
    url_input.submit(
        add_url,
        [chatbot, url_input],
        chatbot
    ).then(
        lambda: gr.update(value="", visible=False),
        outputs=url_input
    )

    # File upload handling
    upload_btn.upload(
        handle_upload,
        [chatbot, upload_btn],
        chatbot,
        queue=False
    )

    # Message handling
    send_btn.click(
        add_message,
        [chatbot, msg],
        chatbot,
        queue=False
    ).then(
        lambda: "",  # Clear the message box
        None,
        msg
    ).then(
        bot_response,
        chatbot,
        chatbot
    )
    
    msg.submit(
        add_message,
        [chatbot, msg],
        chatbot,
        queue=False
    ).then(
        lambda: "",  # Clear the message box
        None,
        msg
    ).then(
        bot_response,
        chatbot,
        chatbot
    )

if __name__ == "__main__":
    demo.launch()