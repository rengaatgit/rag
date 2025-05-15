"""
PDF Processor module for the RAG Pipeline

This module provides capabilities to:
1. Extract text from regular PDFs
2. Perform OCR on scanned PDFs with images
3. Extract and process images from PDFs
4. Handle different PDF quality levels
"""

import os
import io
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import pytesseract
import requests
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDFs including scanned documents with OCR capabilities."""
    
    def __init__(self, 
                 storage_dir: str = "pdf_storage",
                 tesseract_path: Optional[str] = None,
                 language: str = "eng",
                 ocr_threshold: float = 10.0,
                 dpi: int = 300):
        """
        Initialize the PDF processor.
        
        Args:
            storage_dir: Directory to store processed PDF data
            tesseract_path: Path to Tesseract executable (if not in PATH)
            language: OCR language(s) to use (e.g., "eng" or "eng+fra")
            ocr_threshold: Threshold for determining if a page needs OCR (text chars per page)
            dpi: Resolution for image extraction (higher = better OCR but slower)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        # Store extracted images here
        self.images_dir = self.storage_dir / "images"
        self.images_dir.mkdir(exist_ok=True, parents=True)
        
        # Configure Tesseract
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.language = language
        self.ocr_threshold = ocr_threshold
        self.dpi = dpi
    
    def download_pdf(self, url: str) -> Optional[str]:
        """
        Download a PDF from a URL.
        
        Args:
            url: URL of the PDF to download
            
        Returns:
            Optional[str]: Path to the downloaded PDF, or None if download failed
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Check if the content is actually a PDF
            if not response.headers.get('content-type', '').lower().startswith('application/pdf'):
                logger.warning(f"URL {url} does not point to a PDF file")
                return None
            
            # Generate a filename based on the URL
            filename = url.split('/')[-1]
            if not filename.lower().endswith('.pdf'):
                filename = f"{filename}.pdf"
            
            # Ensure filename is valid
            filename = ''.join(c for c in filename if c.isalnum() or c in '._-')
            filepath = self.storage_dir / filename
            
            # Download the file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded PDF from {url} to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            return None
    
    def _extract_page_text(self, page) -> str:
        """Extract text from a PyMuPDF page object."""
        return page.get_text()
    
    def _needs_ocr(self, page) -> bool:
        """Determine if a page needs OCR based on text density."""
        text = self._extract_page_text(page)
        # If page has very little text, it likely needs OCR
        return len(text) < self.ocr_threshold
    
    def _perform_ocr(self, image: Image.Image) -> str:
        """Perform OCR on an image using Tesseract."""
        try:
            return pytesseract.image_to_string(image, lang=self.language)
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def _extract_page_image(self, page) -> Optional[Image.Image]:
        """Extract an image from a PDF page for OCR processing."""
        try:
            # Render page to an image at the specified DPI
            pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img
        except Exception as e:
            logger.error(f"Error extracting page image: {e}")
            return None
    
    def _process_page(self, page, page_num: int, pdf_path: str) -> Dict:
        """
        Process a single PDF page, extracting text and/or performing OCR.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            pdf_path: Path to the PDF file
            
        Returns:
            Dict: Page metadata and content
        """
        # Try to extract text directly first
        text = self._extract_page_text(page)
        needs_ocr = self._needs_ocr(page)
        images = []
        
        # If the page needs OCR, extract image and perform OCR
        if needs_ocr:
            img = self._extract_page_image(page)
            if img:
                # Save image for later reference
                pdf_name = Path(pdf_path).stem
                img_path = self.images_dir / f"{pdf_name}_page_{page_num+1}.png"
                img.save(img_path)
                
                # Perform OCR on the image
                ocr_text = self._perform_ocr(img)
                if ocr_text:
                    text = ocr_text
                
                images.append(str(img_path))
        
        # Extract embedded images if they exist
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = fitz.Pixmap(page.parent, xref)
                if base_image.n - base_image.alpha < 4:  # No CMYK
                    pil_img = Image.frombytes(
                        "RGB" if base_image.n == 3 else "RGBA", 
                        [base_image.width, base_image.height], 
                        base_image.samples
                    )
                    
                    # Save image
                    pdf_name = Path(pdf_path).stem
                    img_path = self.images_dir / f"{pdf_name}_page_{page_num+1}_img_{img_index}.png"
                    pil_img.save(img_path)
                    images.append(str(img_path))
            except Exception as e:
                logger.warning(f"Error extracting embedded image: {e}")
        
        return {
            "page_num": page_num + 1,
            "text": text,
            "needs_ocr": needs_ocr,
            "images": images,
        }
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Process a PDF file, extracting text and performing OCR where needed.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict: Processed PDF content and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        result = {
            "path": pdf_path,
            "filename": Path(pdf_path).name,
            "pages": [],
            "total_pages": 0,
            "ocr_pages": 0,
            "extracted_images": 0,
            "total_text": "",
        }
        
        try:
            # Open the PDF
            pdf = fitz.open(pdf_path)
            result["total_pages"] = len(pdf)
            
            # Process each page
            all_text = []
            all_images = []
            
            for page_num, page in tqdm(enumerate(pdf), desc=f"Processing PDF: {Path(pdf_path).name}", total=len(pdf)):
                page_data = self._process_page(page, page_num, pdf_path)
                result["pages"].append(page_data)
                
                all_text.append(page_data["text"])
                all_images.extend(page_data["images"])
                
                if page_data["needs_ocr"]:
                    result["ocr_pages"] += 1
            
            # Aggregate results
            result["total_text"] = "\n\n".join(all_text)
            result["extracted_images"] = len(all_images)
            
            logger.info(f"PDF processing complete: {Path(pdf_path).name}, "
                        f"{result['total_pages']} pages, "
                        f"{result['ocr_pages']} OCR pages, "
                        f"{result['extracted_images']} images extracted")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def get_document_from_pdf(self, pdf_path: str) -> Dict:
        """
        Convert a processed PDF into a format suitable for RAG processing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict: Document data for RAG
        """
        # Process the PDF
        pdf_data = self.process_pdf(pdf_path)
        
        # Create document metadata
        metadata = {
            "source": pdf_path,
            "filename": Path(pdf_path).name,
            "total_pages": pdf_data["total_pages"],
            "ocr_pages": pdf_data["ocr_pages"],
            "extracted_images": pdf_data["extracted_images"],
        }
        
        return {
            "text": pdf_data["total_text"],
            "metadata": metadata,
            "images": [img for page in pdf_data["pages"] for img in page["images"]],
            "pages": pdf_data["pages"],
        }


# Example usage
if __name__ == "__main__":
    processor = PDFProcessor(storage_dir="./pdf_storage")
    
    # Example: Download and process a PDF from a URL
    # url = "https://arxiv.org/pdf/2307.09288.pdf"  # Example PDF URL
    # pdf_path = processor.download_pdf(url)
    
    # Example: Process a local PDF
    pdf_path = "example.pdf"  # Replace with path to a local PDF
    
    if pdf_path and os.path.exists(pdf_path):
        document = processor.get_document_from_pdf(pdf_path)
        print(f"Processed document: {document['metadata']['filename']}")
        print(f"Total pages: {document['metadata']['total_pages']}")
        print(f"OCR pages: {document['metadata']['ocr_pages']}")
        print(f"Extracted images: {document['metadata']['extracted_images']}")
        print(f"Text sample: {document['text'][:200]}...")
    else:
        print("PDF not found or download failed")