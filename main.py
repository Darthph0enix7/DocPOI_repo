import os
import shutil
import concurrent.futures
from components.metadata_generation import generate_metadata_and_name
from components.record_manager import (
    initialize_vectorstore, 
    add_folder_to_vectorstore, 
    add_file_to_vectorstore, 
    reset_vectorstore
)
from components.parsing import parse_document
from langchain_ollama import ChatOllama

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


def process_file(file_path, metadata_llm, naming_llm, max_tokens, base_dir, original_docs_folder):
    """Processes a single file: Parses it, generates metadata, and renames it."""
    
    print(f"Processing {os.path.basename(file_path)}...")

    # Step 1: Parse document into a text file
    text_file_path = parse_document(file_path)

    # Step 2: Generate metadata and rename file
    document_name, metadata = generate_metadata_and_name(
        text_file_path, metadata_llm, naming_llm, max_tokens, base_dir
    )

    # Step 3: Rename original file in 'original_documents'
    new_original_path = os.path.join(original_docs_folder, f"{document_name}{os.path.splitext(file_path)[1]}")
    os.rename(file_path, new_original_path)

    return new_original_path  # Return for potential vector store processing


def process_documents(input_folder, metadata_llm, naming_llm, max_tokens, vector_store, record_manager):
    """
    Processes only the files copied from input_folder, ignoring pre-existing documents in original_documents.
    """

    # Define base working directory (where main.py is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define original documents directory
    original_docs_folder = os.path.join(base_dir, "original_documents")

    # Step 1: Copy files and get a list of newly copied files
    new_file_paths = copy_files_to_original(input_folder, original_docs_folder)

    if not new_file_paths:
        print("No new files to process.")
        return

    # Step 2: Process only the newly copied files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_file, file_path, metadata_llm, naming_llm, max_tokens, base_dir, original_docs_folder): file_path
            for file_path in new_file_paths
        }

        processed_files = []
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                result = future.result()
                if result:
                    processed_files.append(result)
            except Exception as e:
                print(f"Error processing {future_to_file[future]}: {e}")

    # Step 3: Add only processed files to the vector store
    print("Adding newly processed documents to the vector store...")
    for processed_file in processed_files:
        add_file_to_vectorstore(processed_file, base_dir, vector_store, record_manager)

    print("Processing complete.")

if __name__ == "__main__":
    # Define input folder
    INPUT_FOLDER = "test_docs"

    # Define LLM parameters (replace with actual model objects)
    METADATA_LLM = ChatOllama(model="qwen2.5:7b", temperature=0.5, num_ctx=12000, num_predict=500)
    NAMING_LLM = ChatOllama(model="qwen2.5:7b", temperature=0.7, num_ctx=12000, num_predict=20)
    MAX_TOKENS = 1024

    # Initialize vector store
    vector_store, record_manager = initialize_vectorstore()

    # Process only new documents
    process_documents(INPUT_FOLDER, METADATA_LLM, NAMING_LLM, MAX_TOKENS, vector_store, record_manager)
