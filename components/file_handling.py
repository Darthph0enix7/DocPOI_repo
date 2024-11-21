import os
import re
import shutil
from components.param_manager import ParamManager
from components.ocr import ocr_file, ocr_directory
from components.metadata_creation import generate_metadata_and_name

# Initialize ParamManager
param_manager = ParamManager()
params = param_manager.get_all_params()

def sanitize_filename(filename):
    # Remove invalid characters for filenames
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def process_file(file, file_loader, metadata_llm, naming_llm):
    copy_docs = param_manager.get_param('copy_docs', default=True)
    
    if copy_docs:
        documents_folder = os.path.join(os.path.dirname(__file__), 'documents')
        if not os.path.exists(documents_folder):
            os.makedirs(documents_folder)
    else:
        documents_folder = os.path.dirname(file)
    
    destination = shutil.copy(file, documents_folder)
    
    # Process the file using OCR
    ocr_file(destination)
    
    # Generate metadata and new document name
    new_document_name, formatted_metadata = generate_metadata_and_name(destination, metadata_llm, naming_llm, default_folder=copy_docs)
    
    # Ensure new_document_name is a string
    if not isinstance(new_document_name, str):
        raise TypeError(f"The generated document name is not a string. {new_document_name}")
    
    # Sanitize the new document name
    new_document_name = sanitize_filename(new_document_name)
    
    file_directory = os.path.dirname(destination)
    file_extension = os.path.splitext(destination)[1]
    new_destination = os.path.join(file_directory, f"{new_document_name}{file_extension}")
    
    if os.path.exists(destination):
        os.rename(destination, new_destination)
    else:
        possible_new_path = os.path.join(file_directory, f"{new_document_name}{file_extension}")
        if os.path.exists(possible_new_path):
            new_destination = possible_new_path
        else:
            return "File not found during renaming."
    
    documents = file_loader.load(file_path=new_destination)
    return documents


def process_folder(folder_path, folder_loader, metadata_llm, naming_llm):
    copy_docs = param_manager.get_param('copy_docs', default=True)
    
    if copy_docs:
        documents_folder = os.path.join(os.path.dirname(__file__), 'documents')
        if not os.path.exists(documents_folder):
            os.makedirs(documents_folder)
    else:
        documents_folder = folder_path
    
    # Process the entire folder using OCR
    ocr_directory(folder_path)

    # Generate metadata and new document names for each file
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file_path)[1].lower()

            # Only process .txt and .pdf files for metadata creation
            if file_extension not in ['.txt', '.pdf']:
                print(f"Skipping metadata creation for unsupported file: {file_path}")
                continue

            new_document_name = generate_metadata_and_name(file_path, metadata_llm, naming_llm, default_folder=copy_docs)
            
            file_directory = os.path.dirname(file_path)
            new_destination = os.path.join(file_directory, f"{new_document_name}{file_extension}")
            
            if os.path.exists(file_path):
                os.rename(file_path, new_destination)
            else:
                possible_new_path = os.path.join(file_directory, f"{new_document_name}{file_extension}")
                if os.path.exists(possible_new_path):
                    new_destination = possible_new_path
                else:
                    return "File not found during renaming."

    # Load all files in the folder
    documents = folder_loader.load(folder_path=documents_folder)

    return documents
