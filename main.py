import os
import shutil
import torch
from components.metadata_generation import generate_metadata_and_name
from components.record_manager import (
    initialize_vectorstore, 
    add_file_to_vectorstore
)
from components.parsing import parse_document
from langchain_ollama import ChatOllama

# Set CUDA visibility for GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def copy_files_to_tmp(input_folder, tmp_folder):
    """Copies files from input folder to a temporary directory and returns a list of copied file paths."""
    os.makedirs(tmp_folder, exist_ok=True)
    copied_files = []

    for file_name in os.listdir(input_folder):
        src_path = os.path.join(input_folder, file_name)
        dest_path = os.path.join(tmp_folder, file_name)

        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)
            copied_files.append(dest_path)

    return copied_files


def parse_documents(files, output_folder):
    """Parses documents, moves them to output_folder, and returns a list of parsed file paths."""
    os.makedirs(output_folder, exist_ok=True)
    parsed_files = []

    for file in files:
        try:
            parsed_file = parse_document(file)
            dest_path = os.path.join(output_folder, os.path.basename(parsed_file))
            shutil.move(parsed_file, dest_path)
            parsed_files.append(dest_path)
        except Exception as e:
            print(f"Error parsing {file}: {e}")

    return parsed_files


def generate_metadata_and_rename(parsed_files, metadata_llm, naming_llm, max_tokens, base_dir, tmp_folder):
    """Generates metadata, renames files accordingly, and returns a list of processed file paths."""
    processed_files = []

    for parsed_file in parsed_files:
        try:
            new_name, metadata, metadata_file, renamed_txt_file = generate_metadata_and_name(
                parsed_file, metadata_llm, naming_llm, max_tokens, base_dir
            )

            original_file = os.path.join(tmp_folder, os.path.splitext(os.path.basename(parsed_file))[0])
            new_original_file = os.path.join(tmp_folder, f"{new_name}{os.path.splitext(original_file)[1]}")

            if os.path.exists(original_file):
                os.rename(original_file, new_original_file)

            processed_files.append(new_original_file)
        except Exception as e:
            print(f"Error generating metadata for {parsed_file}: {e}")

    return processed_files


def move_processed_files(processed_files, parsed_files, metadata_folder, documents_folder, originals_folder):
    """Moves processed files to their respective directories."""
    os.makedirs(metadata_folder, exist_ok=True)
    os.makedirs(documents_folder, exist_ok=True)
    os.makedirs(originals_folder, exist_ok=True)

    for processed_file, parsed_file in zip(processed_files, parsed_files):
        try:
            shutil.move(processed_file, originals_folder)
            shutil.move(parsed_file, documents_folder)
        except Exception as e:
            print(f"Error moving files: {e}")


def process_documents(input_folder, metadata_llm, naming_llm, max_tokens, vector_store, record_manager):
    """Main document processing pipeline."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_folder = os.path.join(base_dir, "tmp")
    documents_folder = os.path.join(base_dir, "documents")
    metadata_folder = os.path.join(base_dir, "metadata")
    originals_folder = os.path.join(base_dir, "original_documents")

    # Step 1: Copy files to temp folder
    copied_files = copy_files_to_tmp(input_folder, tmp_folder)
    if not copied_files:
        print("No new files found.")
        return

    # Step 2: Parse copied files
    parsed_files = parse_documents(copied_files, documents_folder)

    # Free up GPU memory
    torch.cuda.empty_cache()

    # Step 3: Generate metadata and rename files
    processed_files = generate_metadata_and_rename(parsed_files, metadata_llm, naming_llm, max_tokens, base_dir, tmp_folder)

    # Step 4: Move processed files to their respective directories
    move_processed_files(processed_files, parsed_files, metadata_folder, documents_folder, originals_folder)

    # Step 5: Add processed files to vector store
    print("Adding processed files to vector store...")
    for file in processed_files:
        add_file_to_vectorstore(file, base_dir, vector_store, record_manager)

    print("Processing complete.")


if __name__ == "__main__":
    INPUT_FOLDER = "test_docs"

    # Initialize LLM models
    METADATA_LLM = ChatOllama(model="qwen2.5:7b", temperature=0.5, num_ctx=8000, num_predict=500)
    NAMING_LLM = ChatOllama(model="qwen2.5:7b", temperature=0.7, num_ctx=8000, num_predict=40)
    MAX_TOKENS = 1024

    # Initialize vector store
    vector_store, record_manager = initialize_vectorstore()

    # Start processing
    process_documents(INPUT_FOLDER, METADATA_LLM, NAMING_LLM, MAX_TOKENS, vector_store, record_manager)
