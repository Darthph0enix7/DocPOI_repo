from langchain_core.document_loaders import BaseLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from collections import OrderedDict
from typing import List, Optional
import os
from os.path import splitext, exists
import json
import fitz
import datetime

def init_loaders(embeddings):
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

    class DocPOIDirectoryLoader(BaseLoader):
        def __init__(self, directory_path: str, metadata_path: Optional[str] = None) -> None:
            self.directory_path = directory_path
            self.metadata_path = metadata_path or directory_path

        def load(self) -> List[Document]:
            documents = []
            for filename in os.listdir(self.directory_path):
                file_path = os.path.join(self.directory_path, filename)
                metadata_file = os.path.join(self.metadata_path, f"{os.path.splitext(filename)[0]}.json")
                
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                else:
                    metadata = {}

                # Ensure document_id is included in metadata
                if 'document_id' not in metadata:
                    metadata['document_id'] = os.path.splitext(filename)[0]

                if filename.endswith('.pdf'):
                    documents.extend(self.load_pdf(file_path, metadata))
                elif filename.endswith('.txt'):
                    documents.extend(self.load_text(file_path, metadata))
            return documents

        def load_pdf(self, file_path: str, metadata: dict) -> List[Document]:
            with fitz.open(file_path) as pdf_document:
                full_text = ''.join([pdf_document.load_page(page_number).get_text() for page_number in range(len(pdf_document))])

            chunks = text_splitter.create_documents([full_text])

            return [
                Document(
                    page_content=chunk.page_content,
                    metadata=OrderedDict(metadata, page_number=page_number + 1, source=file_path)
                ) for page_number, chunk in enumerate(chunks)
            ]

        def load_text(self, file_path: str, metadata: dict) -> List[Document]:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()

            chunks = text_splitter.create_documents([text_content])

            return [
                Document(
                    page_content=chunk.page_content,
                    metadata=OrderedDict(metadata, page_number=page_number + 1, source=file_path)
                ) for page_number, chunk in enumerate(chunks)
            ]

    class DocPOI(BaseLoader):
        """A custom document loader that reads and processes PDF or TXT files."""

        def __init__(self, file_path: str, metadata_path: str = None) -> None:
            """
            Initialize the loader with a file path and an optional metadata path.
            Args:
                file_path: Path to the PDF or TXT file.
                metadata_path: Path to the metadata file (optional, defaults to None).
            """
            self.file_path = file_path
            # Set metadata path based on file path if not provided
            if not metadata_path:
                assumed_metadata_path = splitext(file_path)[0] + '.json'
                if exists(assumed_metadata_path):
                    metadata_path = assumed_metadata_path
                else:
                    print("No metadata file found, proceeding without external metadata.")
            self.metadata_path = metadata_path

        def load(self) -> list:
            """
            Load and process the file, returning a list of Document objects.
            """
            # Load metadata from a JSON file if provided
            if self.metadata_path and exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {'source': self.file_path, 'processed_date': datetime.datetime.now().isoformat()}

            # Ensure document_id is included in metadata
            if 'document_id' not in metadata:
                metadata['document_id'] = os.path.splitext(os.path.basename(self.file_path))[0]

            ordered_metadata = OrderedDict(metadata)

            # Read and process the file
            if self.file_path.endswith('.pdf'):
                with fitz.open(self.file_path) as pdf:
                    full_text = ''.join([page.get_text() for page in pdf])
            elif self.file_path.endswith('.txt'):
                with open(self.file_path, 'r', encoding='utf-8') as file:
                    full_text = file.read()
            else:
                raise ValueError("Unsupported file type. Please provide a PDF or TXT file.")

            # Use the SemanticChunker to split the text
            documents = text_splitter.create_documents([full_text])

            # Generate Document objects
            return [
                Document(
                    page_content=chunk.page_content,
                    metadata=OrderedDict(ordered_metadata, page_number=page_number + 1)
                ) for page_number, chunk in enumerate(documents)
            ]

    return DocPOIDirectoryLoader, DocPOI