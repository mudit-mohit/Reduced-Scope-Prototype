import fitz
from typing import List
import logging
import re

def extract_text_from_pdf(file) -> str:
    """
    Extract text from a PDF file uploaded via Streamlit.
    
    Args:
        file: Streamlit UploadedFile object containing PDF data.
    
    Returns:
        str: Extracted text from all pages, joined with newlines.
    
    Raises:
        ValueError: If the PDF is invalid or no text is extracted.
    """
    try:
        # Reset file pointer to start
        file.seek(0)
        data = file.read()
        
        # Open PDF from binary stream
        doc = fitz.open(stream=data, filetype="pdf")
        text: List[str] = []
        
        # Extract text from each page
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text("text").strip()
            if page_text:
                # Normalize whitespace and remove non-standard characters
                page_text = re.sub(r"\s+", " ", page_text.replace("\xA0", " "))
                text.append(page_text)
                logging.info(f"Extracted text from page {page_num}: {page_text[:100]}...")
            else:
                logging.warning(f"No text extracted from page {page_num}")
        
        # Close the document
        doc.close()
        
        # Join pages with double newlines for paragraph separation
        combined_text = "\n\n".join(text)
        
        if not combined_text.strip():
            logging.error("No text extracted from PDF")
            raise ValueError("No text could be extracted from the PDF file.")
        
        logging.info(f"Total extracted text length: {len(combined_text)} characters")
        return combined_text
    
    except Exception as e:
        logging.error(f"PDF extraction failed: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
