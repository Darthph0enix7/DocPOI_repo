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
    """Copies files from input_folder to tmp_folder and returns a list of copied file paths."""
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
            if not parsed_file or not os.path.exists(parsed_file):
                print(f"Error: Parsing failed for {file}. Skipping.")
                continue

            dest_path = os.path.join(output_folder, os.path.basename(parsed_file))
            shutil.move(parsed_file, dest_path)
            parsed_files.append(dest_path)
        except Exception as e:
            print(f"Error parsing {file}: {e}")

    return parsed_files


def generate_metadata_and_rename(parsed_files, metadata_llm, naming_llm, max_tokens, base_dir):
    """Generates metadata, renames parsed text files accordingly, and returns updated paths."""
    updated_text_files = []
    metadata_files = []

    for parsed_file in parsed_files:
        try:
            new_name, metadata, metadata_file, renamed_txt_file = generate_metadata_and_name(
                parsed_file, metadata_llm, naming_llm, max_tokens, base_dir
            )

            # Ensure parsed text file gets renamed
            new_text_file_path = os.path.join(os.path.dirname(parsed_file), f"{new_name}.txt")
            os.rename(parsed_file, new_text_file_path)

            # Store updated file paths
            updated_text_files.append(new_text_file_path)
            metadata_files.append(metadata_file)

        except Exception as e:
            print(f"Error generating metadata for {parsed_file}: {e}")

    return updated_text_files, metadata_files


def move_processed_files(updated_text_files, metadata_files, metadata_folder, documents_folder):
    """Moves processed text and metadata files to their respective directories."""
    os.makedirs(metadata_folder, exist_ok=True)
    os.makedirs(documents_folder, exist_ok=True)

    for text_file, metadata_file in zip(updated_text_files, metadata_files):
        try:
            if os.path.exists(text_file):
                shutil.move(text_file, documents_folder)  # Move parsed text document
            else:
                print(f"Warning: {text_file} does not exist, skipping move.")

            if os.path.exists(metadata_file):
                shutil.move(metadata_file, metadata_folder)  # Move metadata file
            else:
                print(f"Warning: {metadata_file} does not exist, skipping move.")

        except Exception as e:
            print(f"Error moving files: {e}")


def clean_tmp_folder(tmp_folder):
    """Deletes the temporary folder after processing is complete."""
    try:
        shutil.rmtree(tmp_folder)
    except Exception as e:
        print(f"Error deleting tmp folder: {e}")


def process_documents(input_folder, metadata_llm, naming_llm, max_tokens, vector_store, record_manager):
    """Main document processing pipeline."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_folder = os.path.join(base_dir, "tmp")
    documents_folder = os.path.join(base_dir, "documents")
    metadata_folder = os.path.join(base_dir, "metadata")

    # Step 1: Copy files to tmp folder
    copied_files = copy_files_to_tmp(input_folder, tmp_folder)
    if not copied_files:
        print("No new files found.")
        return

    # Step 2: Parse copied files
    parsed_files = parse_documents(copied_files, documents_folder)

    # Free up GPU memory
    torch.cuda.empty_cache()

    # Step 3: Generate metadata and rename files
    updated_text_files, metadata_files = generate_metadata_and_rename(parsed_files, metadata_llm, naming_llm, max_tokens, base_dir)

    # Step 4: Move parsed text files and metadata to their respective directories
    move_processed_files(updated_text_files, metadata_files, metadata_folder, documents_folder)

    # Step 5: Add processed files and metadata to vector store (ONLY from `documents/` and `metadata/`)
    print("Adding processed files to vector store...")
    for text_file in updated_text_files:
        metadata_file = os.path.join(metadata_folder, os.path.basename(text_file).replace('.txt', '.json'))
        add_file_to_vectorstore(text_file, metadata_file, vector_store, record_manager)

    # Step 6: Clean up temporary folder
    clean_tmp_folder(tmp_folder)

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
