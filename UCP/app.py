import gradio as gr
import shutil
import os
from typing import Optional, Tuple
from config import Config
from data_processor.file_processor import process_files
from llama_index.core import VectorStoreIndex
from llama_index.llms.databricks import Databricks
from data_processor.db_operations import get_storage_context

os.environ["DATABRICKS_TOKEN"] = Config.DB_API_TOKEN

llm = Databricks(
    endpoint=Config.LLM_ENDPOINT,
    databricks_token=Config.DB_API_TOKEN,
    model="gpt-3.5-turbo"
)

embed_model = Config.get_embedding_model()  # Get from config

def upload_file(files):
    os.makedirs(Config.USER_FOLDER, exist_ok=True)
    saved_files = []
    for file in files:
        dest_path = os.path.join(Config.USER_FOLDER, os.path.basename(file.name))
        shutil.copy(file.name, dest_path)
        saved_files.append(dest_path)
    process_files(Config.USER_FOLDER, "user_documents")
    return saved_files

def document_compliance_check(query: str, document_content: str) -> str:
    # Retrieve relevant rules
    rules_index = VectorStoreIndex.from_vector_store(
        get_storage_context("ucp600_rules").vector_store,
        embed_model=embed_model
    )
    retriever = rules_index.as_retriever(similarity_top_k=3)
    relevant_rules = retriever.retrieve(query + "\n" + document_content)
    context = "\n".join([node.text for node in relevant_rules])
    
    # Generate compliance analysis
    prompt = f"""Analyze the document content in relation to the user's query and UCP600 rules:
    
    User Query: {query}
    
    Relevant UCP600 Rules:
    {context}
    
    Document Content:
    {document_content}
    
    Provide a detailed response that:
    1. Directly addresses the user's question
    2. Highlights compliance issues if any
    3. Cites specific UCP600 rules
    4. Suggests corrective actions if needed
    
    Analysis:"""
    
    response = llm.complete(prompt)
    return response.text

def chat_response(message: str, history: list, uploaded_files: list) -> Tuple[str, list]:
    if not uploaded_files:
        # Handle general queries
        rules_index = VectorStoreIndex.from_vector_store(
            get_storage_context("ucp600_rules").vector_store,
            embed_model=embed_model
        )
        query_engine = rules_index.as_query_engine(llm=llm)
        return query_engine.query(message).response
    
    # Process uploaded documents for compliance queries
    responses = []
    for file_path in uploaded_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                response = document_compliance_check(message, content)
                responses.append(
                    f"Analysis for {os.path.basename(file_path)}:\n{response}"
                )
    
    return "\n\n".join(responses), uploaded_files

with gr.Blocks() as demo:
    uploaded_files = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload Documents for Compliance Check",
                file_types=[".pdf", ".docx"],
                file_count="multiple"
            )
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
            
            file_upload.upload(
                fn=lambda files: (files, files),
                inputs=[file_upload],
                outputs=[upload_status, uploaded_files]
            )
        
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=lambda msg, hist, files: chat_response(msg, hist, files),
                additional_inputs=[uploaded_files],
                title="UCP600 Compliance Assistant",
                description="Upload documents and ask compliance questions or general UCP600 queries"
            )

if __name__ == "__main__":
    demo.launch()