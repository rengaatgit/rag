# Updated pip requirements
"""
pip install llama-index-core==0.12.0
pip install llama-index-embeddings-databricks==0.12.0
pip install llama-index-llms-databricks==0.12.0
pip install pymupdf pytesseract pillow python-magic pdf2image
pip install pdfplumber unstructured[pdf]
"""

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import pdfplumber
import os
from pathlib import Path

class PDFRAGPipeline(RAGPipeline):
    def _pdf_to_images(self, pdf_path):
        """Convert PDF pages to images"""
        return convert_from_path(pdf_path, dpi=300)

    def _ocr_image(self, image):
        """Perform OCR on single image"""
        return pytesseract.image_to_string(image, lang='eng')

    def _extract_pdf_content(self, pdf_path):
        """Hybrid text extraction: Direct text + OCR fallback"""
        text_content = []
        
        # First try text extraction with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and len(text) > 50:  # Simple validity check
                    text_content.append(text)
                else:
                    # Fallback to OCR
                    images = self._pdf_to_images(pdf_path)
                    for img in images:
                        text_content.append(self._ocr_image(img))
        
        return "\n".join(text_content)

    def process_pdfs(self, pdf_list_file: str):
        """Process PDFs listed in text file"""
        with open(pdf_list_file, "r") as f:
            pdf_paths = [p.strip() for p in f.read().splitlines() if p.strip()]
        
        new_pdfs = [p for p in pdf_paths if p not in self.processed_urls]
        if not new_pdfs:
            print("No new PDFs to process")
            return

        documents = []
        for pdf_path in new_pdfs:
            try:
                content = self._extract_pdf_content(pdf_path)
                documents.append(Document(text=content, metadata={"source": pdf_path}))
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
                continue

        # Rest of the processing remains same as parent class
        node_parser = SentenceSplitter.from_defaults(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap,
        )
        nodes = node_parser.get_nodes_from_documents(documents)

        if not os.path.exists(PERSIST_DIR):
            storage_context = StorageContext.from_defaults()
            index = VectorStoreIndex(nodes, storage_context=storage_context)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            index.insert_nodes(nodes)

        storage_context.persist(persist_dir=PERSIST_DIR)
        self.processed_urls.update(new_pdfs)
        self._save_processed_urls()

# Test case
if __name__ == "__main__":
    os.environ["DATABRICKS_TOKEN"] = "your_api_token"
    os.environ["DATABRICKS_HOST"] = "your_databricks_host"

    pipeline = PDFRAGPipeline()

    # Sample PDF list (pdfs.txt)
    """
    /path/to/document1.pdf
    /path/to/scanned_doc.pdf
    """

    # Process PDFs
    pipeline.process_pdfs("pdfs.txt")

    # Query example
    query_engine = pipeline.query_engine()
    response = query_engine.query("What's the main topic of the scanned document?")
    print(response)