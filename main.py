from components.param_manager import ParamManager
import pytesseract
import platform
import os
from pdf2image import convert_from_path
from PyPDF2 import PdfWriter, PdfReader
import io
import logging
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
tessdata_dir = os.path.join("installer_files", "tessdata")
tessdata_dir_config = f'--tessdata-dir "{tessdata_dir}"'

if platform.system() == 'Windows':
    POPPLER_PATH = r'.\installer_files\poppler-24.07.0\Library\bin'
    program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
    PYTESSERACT_CMD = os.path.join(program_files, 'Tesseract-OCR', 'tesseract.exe')
    pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_CMD
else:
    POPPLER_PATH = None  # Poppler is installed system-wide on Linux

param_manager = ParamManager(defaults={'param1': 42, 'param2': 'Hello World'})
value = param_manager.get_param('param1')
print(value)
param_manager.set_param('new_param', 3.14)

global_params = param_manager.get_all_params()
print(global_params)

def ocr_pdf(input_pdf_path):
    try:
        images = convert_from_path(input_pdf_path, poppler_path=POPPLER_PATH)
        pdf_writer = PdfWriter()

        # OCR the first page and detect its language
        first_page_text = pytesseract.image_to_string(images[0], config=tessdata_dir_config)
        print(f"First page text: {first_page_text}")

        # OCR the entire PDF
        for image in images:
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(image, extension='pdf', config=tessdata_dir_config)
            pdf_stream = io.BytesIO(pdf_bytes)
            pdf = PdfReader(pdf_stream)
            pdf_writer.add_page(pdf.pages[0])

        output_pdf_path = input_pdf_path.replace('.pdf', '_ocr.pdf')
        with open(output_pdf_path, "wb") as f_out:
            pdf_writer.write(f_out)

        print(f"OCR processed and saved to {output_pdf_path}")
        return output_pdf_path
    except Exception as e:
        logging.error(f"Error during OCR: {e}")
        return None

# Example usage
if __name__ == "__main__":
    input_pdf_path = "example.pdf"  # Replace with your PDF file path
    ocr_pdf(input_pdf_path)