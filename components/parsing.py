import os
import torch
from docling.document_converter import DocumentConverter

def parse_document(file_path: str) -> str:
    # Initialize the converter
    converter = DocumentConverter()
    
    # Convert the document
    result = converter.convert(file_path)
    
    # Generate output file path with the same name but .txt extension
    output_path = os.path.splitext(file_path)[0] + ".txt"
    
    # Save the result to the text file with UTF-8 encoding
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(result.document.export_to_markdown())

    return output_path
