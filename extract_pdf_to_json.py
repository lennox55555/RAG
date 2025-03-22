# extract_pdf_to_json.py
import os
import json
from pathlib import Path
import logging
import PyPDF2
from pdf2image import convert_from_path
import pytesseract

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure Tesseract is installed and available
# On macOS, install via: `brew install tesseract`
# Set the path to tesseract executable if needed
# pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'  # Uncomment and set path if needed

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2 (for text-based PDFs) or OCR (for scanned PDFs).
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text.
    """
    text = ""
    try:
        # First, try to extract text directly using PyPDF2 (for text-based PDFs)
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # If no text is extracted (likely a scanned PDF), use OCR
        if not text.strip():
            logger.info(f"No text extracted directly from {pdf_path}, using OCR...")
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            for image in images:
                # Use pytesseract to perform OCR on the image
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n"
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def process_pdfs_to_json(pdf_folder: str, output_folder: str):
    """
    Process all PDFs in the given folder, extract text, and save each to a separate JSON file.
    
    Args:
        pdf_folder (str): Path to the folder containing PDFs.
        output_folder (str): Path to the folder where JSON files will be saved.
    """
    pdf_folder_path = Path(pdf_folder)
    output_folder_path = Path(output_folder)
    
    if not pdf_folder_path.exists() or not pdf_folder_path.is_dir():
        raise ValueError(f"PDF folder {pdf_folder} does not exist or is not a directory")
    
    # Create output folder if it doesn’t exist
    output_folder_path.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    # Process each PDF in the folder
    for pdf_file in pdf_folder_path.glob("*.pdf"):
        pdf_path = str(pdf_file)
        # Use the filename (without .pdf) as the title, key, source, and JSON filename
        filename = pdf_file.stem  # e.g., "report1" from "report1.pdf"
        logger.info(f"Processing PDF: {pdf_path}")

        # Extract text
        content = extract_text_from_pdf(pdf_path)
        if not content:
            logger.warning(f"No content extracted from {pdf_path}, skipping...")
            continue

        # Create document entry
        doc = {
            "title": filename,
            "text": content,
            "key": filename,
            "source": filename
        }

        # Save to individual JSON file
        output_json_path = output_folder_path / f"{filename}.json"
        with open(output_json_path, 'w') as f:
            json.dump(doc, f, indent=2)
        
        processed_count += 1
        logger.info(f"Saved document to {output_json_path}")

    logger.info(f"Processed {processed_count} PDFs and saved to {output_folder}")

if __name__ == "__main__":
    pdf_folder = "data/report_pdfs"
    output_folder = "data/pdf_extracted" 
    process_pdfs_to_json(pdf_folder, output_folder)