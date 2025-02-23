import os
import re
import json
import hashlib
from datetime import datetime
from collections import OrderedDict
from components.prompts import metadata_template, naming_template
from langchain_core.prompts import ChatPromptTemplate


def ensure_directories_exist(base_folder):
    """Ensure that metadata and documents directories exist within the given base folder."""
    metadata_folder = os.path.join(base_folder, "metadata")
    documents_folder = os.path.join(base_folder, "documents")
    
    os.makedirs(metadata_folder, exist_ok=True)
    os.makedirs(documents_folder, exist_ok=True)
    
    return metadata_folder, documents_folder


def compute_document_hash(file_path):
    """Compute a SHA-256 hash for the content of a document."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def split_and_sample_text(file_path, max_tokens):
    """Splits and samples text from a file to fit within max_tokens limit."""
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    delimiters = r"(?<=[\.\!\?])\s|(?<=,)\s|(?<=\n)"
    sentences = re.split(delimiters, full_text)

    # Allocate tokens
    beginning_tokens = int(max_tokens * 0.6)
    chunk_tokens = int(max_tokens * 0.1)

    # Collect the first 60% tokens
    current_chunk = ""
    first_chunk = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= beginning_tokens:
            current_chunk += sentence + " "
        else:
            first_chunk.append(current_chunk.strip())
            current_chunk = ""
            break
    first_chunk_text = " ".join(first_chunk)

    # Randomly sample remaining 10% chunks
    remaining_chunks = []
    total_tokens_used = len(first_chunk_text)
    while total_tokens_used < max_tokens:
        if not sentences:
            break
        random_chunk = sentences.pop(len(sentences) // 2)
        if len(random_chunk) <= chunk_tokens:
            remaining_chunks.append(random_chunk)
            total_tokens_used += len(random_chunk)

    sampled_text = first_chunk_text + " " + " ".join(remaining_chunks)
    return sampled_text[:max_tokens], remaining_chunks[:len(remaining_chunks) - 2]


def generate_metadata_and_name(file_path, metadata_llm, naming_llm, max_tokens, save_folder):
    """
    Generate metadata and rename the document based on its content.
    The metadata and renamed document will be saved in the specified folder.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension != ".txt":
        raise ValueError("Unsupported file type. Only .txt files are allowed.")

    document_id = compute_document_hash(file_path)
    sampled_text, naming_chunks = split_and_sample_text(file_path, max_tokens)

    # Generate metadata
    metadata_prompt = ChatPromptTemplate.from_template(metadata_template)
    metadata_chain = metadata_prompt | metadata_llm
    metadata_result = metadata_chain.invoke({"context": sampled_text})
    metadata_content = metadata_result.content

    json_start = metadata_content.find('{')
    json_end = metadata_content.rfind('}') + 1
    json_content = metadata_content[json_start:json_end]
    metadata = json.loads(json_content, strict=False)

    def collapse_dicts(value):
        if isinstance(value, list):
            return [item['name'] if isinstance(item, dict) and 'name' in item else item for item in value]
        return value

    metadata = {key: collapse_dicts(value) for key, value in metadata.items()}
    formatted_metadata = json.dumps(metadata, indent=4, ensure_ascii=False)

    # Generate document name
    naming_prompt = ChatPromptTemplate.from_template(naming_template)
    naming_chain = naming_prompt | naming_llm
    naming_sampled_text = " ".join(naming_chunks)
    naming_result = naming_chain.invoke({
        "question": "What is the most suitable name for this document based on its content, in its own language?",
        "context": naming_sampled_text,
        "metadata": formatted_metadata
    })
    naming_content = naming_result.content
    temp_document_name = naming_content.split('\n')[0].strip()
    document_name = temp_document_name.replace(" ", "_")

    # Prepare metadata with additional file details
    file_creation_date = datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
    file_modification_date = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
    metadata_creation_date = datetime.now().isoformat()

    ordered_metadata = OrderedDict([
        ("given_document_name", document_name),
        ("document_id", document_id),
        ("original_file_name", os.path.basename(file_path)),
        ("file_creation_date", file_creation_date),
        ("file_modification_date", file_modification_date),
        ("metadata_creation_date", metadata_creation_date),
    ])
    ordered_metadata.update(metadata)

    formatted_metadata = json.dumps(ordered_metadata, indent=4, ensure_ascii=False)

    # Ensure directories exist
    metadata_folder, documents_folder = ensure_directories_exist(save_folder)

    # Save metadata
    metadata_file_path = os.path.join(metadata_folder, f"{document_name}.json")
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_metadata)

    # Move and rename the document
    new_file_path = os.path.join(documents_folder, f"{document_name}.txt")
    os.rename(file_path, new_file_path)

    return document_name, formatted_metadata, metadata_file_path, new_file_path
