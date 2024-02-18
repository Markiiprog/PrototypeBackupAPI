import docx
import PyPDF4
import textract
import os 

from pathlib import Path
from pypdf import PdfReader




def extract_text_from_file(file_path):
    file_extension = Path(file_path).suffix.lower()
    text=""

    if file_extension == ".doc"  :
        # Extract text from DOCX file
        doc = docx.Document(file_path)

        for paragraph in doc.paragraphs:
          text += paragraph.text + "\n"  # Add a newline after each paragraph

    elif file_extension == ".pdf":
        # Extract text from PDF file
        reader = PdfReader(file_path)
        
        for page in reader.pages:
            text += page.extract_text()
            
    elif file_extension == ".txt":
        # Extract text from TXT file
        with open(file_path, "r", encoding="utf-8") as txt_file:
            text = txt_file.read()

    else:
        raise ValueError("Unsupported file format. Only DOCX, PDF, and TXT are supported.")

    return text