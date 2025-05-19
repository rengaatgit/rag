import gradio as gr
import shutil
import os
from typing import List, Tuple
from config import Config
from llama_index.core import VectorStoreIndex
from llama_index.llms.databricks import Databricks
from data_processor.db_operations import get_storage_context
from data_processor.file_processor import process_files

# Initialize models
llm = Databricks(
    endpoint=Config.LLM_ENDPOINT,
    databricks_token=Config.DB_API_TOKEN
)
embed_model = Config.get_embedding_model()

def upload_files(files: List[str]) -> List[str]:
    """Process uploaded files and return their paths"""
    os.makedirs(Config.USER_FOLDER, exist_ok=True)
    saved_paths = []
    for file in files:
        dest_path = os.path.join(Config.USER_FOLDER, os.path.basename(file))
        shutil.copy(file, dest_path)
        saved_paths.append(dest_path)
    process_files(Config.USER_FOLDER, "user_documents")
    return saved_paths

def document_compliance_check(query: str, document_path: str) -> str:
    """Enhanced compliance check using both rules and document embeddings"""
    # Load the document content
    with open(document_path, 'r') as f:
        content = f.read()
    
    # Get both vector stores
    rules_index = VectorStoreIndex.from_vector_store(
        get_storage_context("ucp600_rules").vector_store,
        embed_model=embed_model
    )
    user_docs_index = VectorStoreIndex.from_vector_store(
        get_storage_context("user_documents").vector_store,
        embed_model=embed_model
    )
    
    # Retrieve most relevant rules AND document sections
    rules_retriever = rules_index.as_retriever(similarity_top_k=3)
    docs_retriever = user_docs_index.as_retriever(similarity_top_k=2)
    
    relevant_rules = rules_retriever.retrieve(query)
    relevant_doc_sections = docs_retriever.retrieve(content)
    
    # Prepare context
    rules_context = "\n".join([f"UCP600 Rule {i+1}: {node.text}" 
                             for i, node in enumerate(relevant_rules)])
    doc_context = "\n".join([f"Document Section {i+1}: {node.text}" 
                           for i, node in enumerate(relevant_doc_sections)])
    
    prompt = f"""Perform comprehensive UCP600 compliance analysis:
    
    User Query: {query}
    
    Relevant UCP600 Rules:
    {rules_context}
    
    Relevant Document Sections:
    {doc_context}
    
    Analysis Requirements:
    1. Cross-reference specific document sections with relevant rules
    2. Highlight exact matches and discrepancies
    3. Cite rule numbers and document locations
    4. Provide actionable recommendations
    
    Compliance Analysis:"""
    
    return llm.complete(prompt).text

def chat_response(message: str, history: List[Tuple[str, str]], uploaded_files: List[str]) -> Tuple[str, List[str]]:
    """Handle both general queries and document-specific compliance checks"""
    if not uploaded_files:
        # General UCP600 questions
        rules_index = VectorStoreIndex.from_vector_store(
            get_storage_context("ucp600_rules").vector_store,
            embed_model=embed_model
        )
        return rules_index.as_query_engine(llm=llm).query(message).response, uploaded_files
    
    # Document-specific compliance checks
    responses = []
    for doc_path in uploaded_files:
        if os.path.exists(doc_path):
            response = document_compliance_check(message, doc_path)
            doc_name = os.path.basename(doc_path)
            responses.append(f"=== {doc_name} ===\n{response}")
    
    return "\n\n".join(responses), uploaded_files

with gr.Blocks() as demo:
    uploaded_files = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload Trade Documents",
                file_types=[".pdf", ".docx", ".txt"],
                file_count="multiple",
                elem_id="file-upload"
            )
            
            upload_button = gr.Button("Process Documents")
            
            def handle_upload(files):
                saved_paths = upload_files([f.name for f in files])
                return saved_paths, "Documents processed successfully!"
            
            upload_button.click(
                fn=handle_upload,
                inputs=[file_upload],
                outputs=[uploaded_files, gr.Textbox(label="Upload Status")]
            )
        
        with gr.Column(scale=3):
            chat = gr.ChatInterface(
                fn=lambda msg, hist: chat_response(msg, hist, uploaded_files.value),
                additional_inputs=[uploaded_files],
                title="UCP600 Compliance Assistant",
                examples=[
                    "Does my document comply with UCP600 Article 14?",
                    "What are the requirements for a bill of lading?",
                    "Check this LC for discrepancies"
                ]
            )

if __name__ == "__main__":
    demo.launch()