import os
import pytesseract
# PyPDF2
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, TextStringObject
import platform

POPPLER_PATH = r'.\installer_files\poppler-24.07.0\Library\bin'
program_files = os.environ.get('ProgramFiles')
tessdata_dir = os.path.join("installer_files", "tessdata")
tessdata_dir_config = f'--tessdata-dir "{tessdata_dir}"'

if platform.system() == 'Windows':
    program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
    PYTESSERACT_CMD = os.path.join(program_files, 'Tesseract-OCR', 'tesseract.exe')
    pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_CMD

