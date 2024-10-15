from components.param_manager import ParamManager
import pytesseract
import platform
import os

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


param_manager = ParamManager(defaults={'param1': 42, 'param2': 'Hello World'})
value = param_manager.get_param('param1')
print(value)
param_manager.set_param('new_param', 3.14)

global_params = param_manager.get_all_params()
print(global_params)