import os
import pytesseract
import logging
# PyPDF2
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, TextStringObject
import platform
from pdf2image import convert_from_path
from langdetect import detect, LangDetectException
import pycountry
import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance

POPPLER_PATH = None
program_files = os.environ.get('ProgramFiles')
tessdata_dir = os.path.join("installer_files", "tessdata")
tessdata_dir_config = f'--tessdata-dir "{tessdata_dir}"'

if platform.system() == 'Windows':
    program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
    PYTESSERACT_CMD = os.path.join(program_files, 'Tesseract-OCR', 'tesseract.exe')
    pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_CMD
    POPPLER_PATH = r'.\installer_files\poppler-24.07.0\Library\bin'

def adaptive_image_processing(image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    
    # Apply a slight Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(clahe_image, (5, 5), 0)
    
    # Enhance contrast
    pil_image = Image.fromarray(blurred_image)
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_image = enhancer.enhance(1.5)
    
    return enhanced_image

def check_pdf_has_readable_text(file_path: str) -> bool:
    """Check if the PDF contains any readable text."""
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    return True
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
    return False

def check_pdf_metadata_keys(file_path: str, required_keys: list) -> bool:
    """Check if the PDF metadata contains all the required keys."""
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            metadata = reader.metadata
            return all(key in metadata for key in required_keys)
    except Exception as e:
        logging.error(f"Error reading metadata from {file_path}: {e}")
        return False

def convert_image_to_pdf(image_path):
    """Convert an image file to a PDF and return the new PDF path."""
    try:
        image = Image.open(image_path)
        pdf_path = image_path.lower().replace('.png', '.pdf').replace('.jpg', '.pdf').replace('.jpeg', '.pdf')
        rgb_image = image.convert('RGB')
        rgb_image.save(pdf_path, 'PDF', resolution=100.0)
        os.remove(image_path)
        print(f"Converted {image_path} to {pdf_path}")
        return pdf_path
    except Exception as e:
        logging.error(f"Error converting image to PDF: {e}")
        return None

def ocr_file(input_file_path):
    """Perform OCR on a file and return the path to the processed file."""
    try:
        # Determine file type and handle accordingly
        file_extension = os.path.splitext(input_file_path)[1].lower()
        if file_extension in ['.txt']:
            print(f"The file {input_file_path} is a text file. Returning the file path.")
            return input_file_path
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            print(f"The file {input_file_path} is an image. Converting to PDF...")
            input_file_path = convert_image_to_pdf(input_file_path)
            if not input_file_path:
                return None

        # Check if PDF already contains readable text and required metadata keys
        required_metadata_keys = ["/document_id", "/original_file_name", "/given_document_name"]
        if check_pdf_has_readable_text(input_file_path) and check_pdf_metadata_keys(input_file_path, required_metadata_keys):
            print(f"The PDF {input_file_path} already contains readable text and the required metadata keys. Skipping OCR.")
            return input_file_path
        
        images = convert_from_path(input_file_path, poppler_path=POPPLER_PATH)
        pdf_writer = PdfWriter()
        
        # OCR the first page and detect its language
        first_page_text = pytesseract.image_to_string(images[0])
        try:
            detected_lang = detect(first_page_text)
            detected_lang_iso3 = pycountry.languages.get(alpha_2=detected_lang).alpha_3
            print(f"Detected language: {detected_lang_iso3}")
        except LangDetectException:
            detected_lang_iso3 = 'eng'  # Default to English if language detection fails
        
        # OCR the entire PDF with the detected language
        for image in images:
            processed_image = adaptive_image_processing(image)
            if processed_image is None:
                print("Error: processed_image is None")
                continue

            pdf_bytes = pytesseract.image_to_pdf_or_hocr(processed_image, extension='pdf', lang=detected_lang_iso3, config=tessdata_dir_config) 
            if pdf_bytes is None:
                print("Error: pdf_bytes is None")
                continue

            pdf_stream = io.BytesIO(pdf_bytes)
            pdf = PdfReader(pdf_stream)
            pdf_writer.add_page(pdf.pages[0])

        output_pdf_path = input_file_path  # Keep the file path consistent
        with open(output_pdf_path, "wb") as f_out:
            pdf_writer.write(f_out)

        print(f"OCR processed and replaced {output_pdf_path}")
        return output_pdf_path
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None

def ocr_directory(directory_path: str, only_pdf: bool = False) -> None:
    """Process all files in a directory, optionally only processing PDF files."""
    try:
        for root, _, files in os.walk(directory_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_extension = os.path.splitext(file_name)[1].lower()

                if only_pdf and file_extension != '.pdf':
                    print(f"Skipping non-PDF file: {file_path}")
                    continue

                print(f"Processing file: {file_path}")
                ocr_file(file_path)
    except Exception as e:
        print(f"Error processing files in directory {directory_path}: {e}")
