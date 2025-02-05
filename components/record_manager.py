import os
import json
from datetime import datetime
from typing import List, Optional
from collections import OrderedDict
from os.path import splitext, exists
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings, ChatOllama

embed_model = OllamaEmbeddings(model="qwen2.5:7b")
# Assuming you have an embedding model instance

# --------------------------
# Document Loaders
# --------------------------

class DocPOIDirectoryLoader:
    """Loads and processes text documents from a directory along with metadata."""

    def __init__(self, directory_path: str, metadata_path: Optional[str] = None) -> None:
        self.directory_path = directory_path
        self.metadata_path = metadata_path or directory_path

    def load(self) -> List[Document]:
        """Loads all text documents in the directory and returns a list of Document objects."""
        documents = []
        for filename in os.listdir(self.directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.directory_path, filename)
                metadata_file = os.path.join(self.metadata_path, f"{os.path.splitext(filename)[0]}.json")

                metadata = self.load_metadata(metadata_file)
                metadata['document_id'] = metadata.get('document_id', os.path.splitext(filename)[0])

                documents.extend(self.load_text(file_path, metadata))
        return documents

    def load_metadata(self, metadata_file: str) -> dict:
        """Loads metadata from a JSON file if available."""
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def sanitize_metadata(self, metadata: dict) -> dict:
        """Ensures all metadata values are of type str, int, float, or bool."""
        sanitized_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                sanitized_metadata[key] = ", ".join(map(str, value))  # Convert list to comma-separated string
            elif isinstance(value, (str, int, float, bool)):
                sanitized_metadata[key] = value  # Keep valid types
            else:
                sanitized_metadata[key] = str(value)  # Convert other types to string
        return sanitized_metadata

    def load_text(self, file_path: str, metadata: dict) -> List[Document]:
        """Reads a text file, chunks it, and creates Document objects."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        embedding = embed_model
        text_splitter = SemanticChunker(embedding, breakpoint_threshold_type="gradient", min_chunk_size=1000)
        chunks = text_splitter.create_documents([text_content])

        sanitized_metadata = self.sanitize_metadata(metadata)  # Sanitize metadata before passing

        return [
            Document(
                page_content=chunk.page_content,
                metadata=OrderedDict(sanitized_metadata, page_number=page_number + 1, source=file_path)
            ) for page_number, chunk in enumerate(chunks)
        ]

class DocPOI:
    """Loads and processes a single text document along with its metadata."""

    def __init__(self, file_path: str, metadata_path: Optional[str] = None) -> None:
        self.file_path = file_path
        self.metadata_path = metadata_path or (splitext(file_path)[0] + '.json' if exists(splitext(file_path)[0] + '.json') else None)

    def load(self) -> List[Document]:
        """Loads a single text document and returns a list of Document objects."""
        metadata = self.load_metadata()
        sanitized_metadata = self.sanitize_metadata(metadata)  # Ensure metadata is valid

        embedding = embed_model
        text_splitter = SemanticChunker(embedding, breakpoint_threshold_type="gradient", min_chunk_size=1000)

        with open(self.file_path, 'r', encoding='utf-8') as file:
            full_text = file.read()

        documents = text_splitter.create_documents([full_text])

        return [
            Document(
                page_content=chunk.page_content,
                metadata=OrderedDict(sanitized_metadata, page_number=page_number + 1)
            ) for page_number, chunk in enumerate(documents)
        ]

    def load_metadata(self) -> dict:
        """Loads metadata from a JSON file if available."""
        if self.metadata_path and exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'source': self.file_path, 'processed_date': datetime.now().isoformat(), 'document_id': os.path.splitext(os.path.basename(self.file_path))[0]}

    def sanitize_metadata(self, metadata: dict) -> dict:
        """Ensures all metadata values are of type str, int, float, or bool."""
        sanitized_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                sanitized_metadata[key] = ", ".join(map(str, value))  # Convert list to comma-separated string
            elif isinstance(value, (str, int, float, bool)):
                sanitized_metadata[key] = value  # Keep valid types
            else:
                sanitized_metadata[key] = str(value)  # Convert other types to string
        return sanitized_metadata

# --------------------------
# Vectorstore Initialization
# --------------------------

def initialize_vectorstore(collection_name="docpoi"):
    """Initializes the vector store and record manager."""
    embedding = embed_model  # Replace with actual embedding model

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory="./DocPOI_db"
    )

    namespace = f"chroma/{collection_name}"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()
    
    return vector_store, record_manager


# --------------------------
# Indexing and Document Insertion
# --------------------------

def add_folder_to_vectorstore(directory_path: str, metadata_path: Optional[str] = None, vectorstore=None, record_manager=None):
    """Loads and adds all documents from a directory to the vectorstore."""
    vectorstore, record_manager = initialize_vectorstore()

    loader = DocPOIDirectoryLoader(directory_path, metadata_path)
    documents = loader.load()

    index(
        documents,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="document_id",
    )


def add_file_to_vectorstore(file_path: str, metadata_path: Optional[str] = None, vectorstore=None, record_manager=None):

    loader = DocPOI(file_path, metadata_path)
    documents = loader.load()

    index(
        documents,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="document_id",
    )


def reset_vectorstore():
    """Resets the vectorstore by clearing all stored documents."""
    vectorstore, record_manager = initialize_vectorstore()

    index(
        [],  # Empty list to clear the vectorstore
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="document_id",
    )
    return "Vectorstore has been reset."
