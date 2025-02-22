from components.record_manager import (
    initialize_vectorstore, 
    add_file_to_vectorstore, 
    reset_vectorstore
)
import os

# Initialize vector store
vector_store, record_manager = initialize_vectorstore()

# Define file paths
file_path = "documents\Bescheinigung_Kalinsazlioglu_Schwalm-Eder_2023.txt"
metadata_path = "metadata\Bescheinigung_Kalinsazlioglu_Schwalm-Eder_2023.json"

# Add file to vector store
add_file_to_vectorstore(file_path, metadata_path, vector_store, record_manager)

print("File and metadata have been added to the vector store.")