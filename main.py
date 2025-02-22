import os
import shutil
import torch
from components.metadata_generation import generate_metadata_and_name
from components.record_manager import (
    initialize_vectorstore, 
    add_folder_to_vectorstore, 
    add_file_to_vectorstore, 
    reset_vectorstore
)
from components.parsing import parse_document
from langchain_ollama import ChatOllama

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def copy_files_to_original(input_folder, original_docs_folder):
    """
    Copies files from input_folder to original_documents and returns a list of copied file paths.
    """
    os.makedirs(original_docs_folder, exist_ok=True)
    new_files = []

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):
            dest_path = os.path.join(original_docs_folder, file_name)
            shutil.copy(file_path, dest_path)
            new_files.append(dest_path)  # Store full paths of copied files

    return new_files  # Return only newly copied files


def parse_files(file_paths, documents_folder):
    """Parses all files and returns a list of text file paths."""
    text_file_paths = []
    os.makedirs(documents_folder, exist_ok=True)
    for file_path in file_paths:
        try:
            text_file_path = parse_document(file_path)
            dest_path = os.path.join(documents_folder, os.path.basename(text_file_path))
            shutil.move(text_file_path, dest_path)
            text_file_paths.append(dest_path)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
    return text_file_paths


def generate_metadata_and_rename_files(text_file_paths, metadata_llm, naming_llm, max_tokens, base_dir, original_docs_folder):
    """Generates metadata and renames files based on parsed text files."""
    processed_files = []
    for text_file_path in text_file_paths:
        try:
            document_name, metadata = generate_metadata_and_name(
                text_file_path, metadata_llm, naming_llm, max_tokens, base_dir
            )
            original_file_path = os.path.join(original_docs_folder, os.path.basename(text_file_path))
            new_original_path = os.path.join(original_docs_folder, f"{document_name}{os.path.splitext(original_file_path)[1]}")
            os.rename(original_file_path, new_original_path)
            processed_files.append(new_original_path)
        except Exception as e:
            print(f"Error generating metadata for {text_file_path}: {e}")
    return processed_files


def process_documents(input_folder, metadata_llm, naming_llm, max_tokens, vector_store, record_manager):
    """
    Processes only the files copied from input_folder, ignoring pre-existing documents in original_documents.
    """

    # Define base working directory (where main.py is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define original documents directory
    original_docs_folder = os.path.join(base_dir, "original_documents")

    # Define documents directory for parsed text files
    documents_folder = os.path.join(base_dir, "documents")

    # Step 1: Copy files and get a list of newly copied files
    new_file_paths = copy_files_to_original(input_folder, original_docs_folder)

    if not new_file_paths:
        print("No new files to process.")
        return

    # Step 2: Parse all newly copied files
    text_file_paths = parse_files(new_file_paths, documents_folder)

    # Free up GPU memory
    torch.cuda.empty_cache()

    # Step 3: Generate metadata and rename files
    processed_files = generate_metadata_and_rename_files(
        text_file_paths, metadata_llm, naming_llm, max_tokens, base_dir, original_docs_folder
    )

    # Step 4: Add only processed files to the vector store
    print("Adding newly processed documents to the vector store...")
    for processed_file in processed_files:
        add_file_to_vectorstore(processed_file, base_dir, vector_store, record_manager)

    print("Processing complete.")

if __name__ == "__main__":
    # Define input folder
    INPUT_FOLDER = "test_docs"

    # Define LLM parameters (replace with actual model objects)
    METADATA_LLM = ChatOllama(model="qwen2.5:7b", temperature=0.5, num_ctx=8000, num_predict=500)
    NAMING_LLM = ChatOllama(model="qwen2.5:7b", temperature=0.7, num_ctx=8000, num_predict=40)
    MAX_TOKENS = 1024

    # Initialize vector store
    vector_store, record_manager = initialize_vectorstore()

    # Process only new documents
    process_documents(INPUT_FOLDER, METADATA_LLM, NAMING_LLM, MAX_TOKENS, vector_store, record_manager)