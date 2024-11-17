import json
import os
import uuid
import shutil
from datetime import datetime
from collections import OrderedDict
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, TextStringObject
from components.prompts import metadata_template, naming_template

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def update_pdfmetadata(file_path: str, new_metadata: dict) -> None:
    """Updates the metadata of the given PDF file with new keys.

    Args:
        file_path: The path to the PDF file.
        new_metadata: A dictionary of new metadata to add.
    """    
    # Open the existing PDF
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        writer = PdfWriter()
        writer.append_pages_from_reader(reader)
        
        # Get existing metadata
        existing_metadata = reader.metadata
        
        # Update existing metadata with new keys
        updated_metadata = {NameObject(key): TextStringObject(value) for key, value in existing_metadata.items()}
        
        for key, value in new_metadata.items():
            updated_metadata[NameObject(key)] = TextStringObject(value)
        
        # Add updated metadata
        writer.add_metadata(updated_metadata)
        
        # Save the PDF with the updated metadata back to the same file
        with open(file_path, "wb") as updated_file:
            writer.write(updated_file)

def generate_metadata_and_name(file_path, metadata_llm, naming_llm, default_folder=True):
    # Load the document content
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path, extract_images=False)
    elif file_extension == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    docs = loader.load()
    metadata_prompt = ChatPromptTemplate.from_template(metadata_template)

    # Create the chain for metadata extraction
    metadata_chain = metadata_prompt | metadata_llm

    # Invoke the chain with the document content
    metadata_result = metadata_chain.invoke({
        "context": docs
    })

    # Extract the content from the result
    metadata_content = metadata_result.content

    # Parse the JSON part of the content
    json_start = metadata_content.find('{')
    json_end = metadata_content.rfind('}') + 1
    json_content = metadata_content[json_start:json_end]
    metadata = json.loads(json_content)

    # Ensure all keys have values that are basic lists or primitive types
    def collapse_dicts(value):
        if isinstance(value, list):
            return [item['name'] if isinstance(item, dict) and 'name' in item else item for item in value]
        return value

    metadata = {key: collapse_dicts(value) for key, value in metadata.items()}

    # Format the metadata in a readable format
    formatted_metadata = json.dumps(metadata, indent=4, ensure_ascii=False)
    naming_prompt = ChatPromptTemplate.from_template(naming_template)

    # Create the chain for document naming
    naming_chain = naming_prompt | naming_llm

    # Invoke the chain with the context and metadata
    naming_result = naming_chain.invoke({
        "question": "What is the most suitable name for this document based on its content?",
        "context": docs,
        "metadata": formatted_metadata
    })

    # Extract the document name from the result
    naming_content = naming_result.content
    temp_document_name = naming_content.split('\n')[0].strip()

    # Replace spaces in document name with underscores
    document_name = temp_document_name.replace(" ", "_")

    # Load the metadata JSON
    metadata = json.loads(formatted_metadata)

    # Ensure given_document_name is the first key
    ordered_metadata = OrderedDict([("given_document_name", document_name)])
    ordered_metadata.update(metadata)

    # Generate a unique ID for the document
    document_id = str(uuid.uuid4())

    # Get file details
    file_directory = os.path.dirname(file_path)
    original_file_name = os.path.basename(file_path)

    # Get file creation and modification dates
    file_creation_date = datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
    file_modification_date = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
    metadata_creation_date = datetime.now().isoformat()

    # Append additional metadata
    ordered_metadata['document_id'] = document_id
    ordered_metadata['file_directory'] = file_directory
    ordered_metadata['original_file_name'] = original_file_name
    ordered_metadata['file_creation_date'] = file_creation_date
    ordered_metadata['file_modification_date'] = file_modification_date
    ordered_metadata['metadata_creation_date'] = metadata_creation_date

    # Format the metadata in a readable format
    formatted_metadata = json.dumps(ordered_metadata, indent=4, ensure_ascii=False)

    # Save the metadata to a JSON file with the same name as the document
    metadata_file_path = os.path.join(file_directory, f"{document_name}.json")
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_metadata)

    # Set up the documents folder
    documents_folder = os.path.join(os.path.dirname(__file__), 'documents')
    if default_folder:
        os.makedirs(documents_folder, exist_ok=True)

    if file_extension == ".pdf":
        # Update PDF metadata with relevant keys (only for PDFs)
        pdf_metadata = {
            "/document_id": document_id,
            "/original_file_name": original_file_name,
            "/given_document_name": document_name
        }
        update_pdfmetadata(file_path, pdf_metadata)
        new_file_path = os.path.join(file_directory, f"{document_name}.pdf")
    elif file_extension == ".txt":
        new_file_path = os.path.join(file_directory, f"{document_name}.txt")
    
    # Ensure no overwriting issues occur during renaming
    if not os.path.exists(new_file_path):
        os.rename(file_path, new_file_path)

    # Copy to default folder if required
    if default_folder:
        processed_file_destination = os.path.join(documents_folder, os.path.basename(new_file_path))
        metadata_file_destination = os.path.join(documents_folder, os.path.basename(metadata_file_path))

        # Check if the file already exists in the destination folder
        if not os.path.exists(processed_file_destination):
            shutil.copy(new_file_path, processed_file_destination)
        if not os.path.exists(metadata_file_destination):
            shutil.copy(metadata_file_path, metadata_file_destination)

    # Return the document name and metadata
    return document_name, formatted_metadata
