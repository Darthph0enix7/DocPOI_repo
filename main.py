import os
import io
import gc
import time
import json
from typing import List, Optional
import wave
import uuid
import shutil
import random
import logging
import webbrowser
import threading
import soundfile as sf
import pytesseract
import torch
import pycountry
from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect
import cv2
import whisper
import numpy as np
from datetime import datetime
from os.path import splitext, exists
from collections import OrderedDict
from pynput import keyboard
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
from scipy.io.wavfile import write
import gradio as gr
import tempfile

# PyMuPDF
import fitz  

# PyPDF2
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, TextStringObject

# Langchain and related imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain_elasticsearch import ElasticsearchStore
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import tkinter as tk
from tkinter import filedialog
import subprocess
import sys
import re
# Constants
POPPLER_PATH = r'.\installer_files\poppler-24.07.0\Library\bin'
program_files = os.environ.get('ProgramFiles')
PYTESSERACT_CMD = os.path.join(program_files, 'Tesseract-OCR', 'tesseract.exe')
PARAM_FILE = "params.json"
LOG_FILE = "process.log"
is_recording = False  # To track if we are currently recording
recording = None  # Global variable to hold the recording data
stream = None  # To handle the audio stream
filename = "output_combined.wav"  # File to save the recording



system_prompt = """You are a helpful assistant with the name Jarvis created by Eren Kalinsazlioglu at Enpoi co. that has access to users' documents. Your primary goal is to be as helpful and precise as possible in answering the users' questions. If the user asks a specific or personalized question that you do not have knowledge of, you can retrieve the relevant documents like this:

retriever_tool: [here describe what kind of document the user wants to retrieve, this will be used for the similarity search so write some queries that are likely to be in the document]

only use the retriever when you need the document. Do not include any filter text or explanation, only the retriever calling.
Try to answer general questions without using the retriever. If you need to provide information about a specific document."""
metadata_template = """
You are tasked with extracting detailed metadata information from the content of a document. Follow these detailed guidelines to ensure the metadata is comprehensive and accurately reflects the document's content.

**Guidelines**:

1. **Document Type**:
    - Identify the type of document (e.g., research paper, article, report).
    - Examples: "Research Paper", "Article", "Report", "Forschungsbericht", "Artikel", "Bericht"

2. **Mentions**:
    - Extract the main names like persons and companies mentioned in the document.
    - Examples: "John Doe", "Acme Corporation", "United Nations", "Johann Schmidt", "Siemens AG", "Vereinte Nationen"

3. **Keywords**:
    - Identify relevant keywords central to the document's topic.
    - Examples: "Machine Learning", "Climate Change", "Economic Policy", "Maschinelles Lernen", "Klimawandel", "Wirtschaftspolitik"

4. **About**:
    - Provide a brief description of the document's purpose and main arguments/findings.
    - Examples: "This research paper explores the impact of AI on healthcare, focusing on predictive analytics and patient outcomes.", "Dieses Forschungspapier untersucht die Auswirkungen von KI auf das Gesundheitswesen, mit einem Fokus auf prädiktive Analysen und Patientenergebnisse."

5. **Questions**:
    - List questions the document can answer.
    - Examples: "What are the benefits of renewable energy?", "How does blockchain technology work?", "Welche Vorteile bietet erneuerbare Energie?", "Wie funktioniert die Blockchain-Technologie?"

6. **Entities**:
    - Identify the main entities (people, places, organizations) mentioned.
    - Examples: "Albert Einstein", "New York City", "World Health Organization", "Albert Einstein", "New York City", "Weltgesundheitsorganisation"

7. **Summaries**:
    - Provide summaries of different sections or key points.
    - Examples: "Introduction: Overview of AI in healthcare", "Methodology: Data collection and analysis techniques", "Conclusion: Implications of findings for future research", "Einleitung: Überblick über KI im Gesundheitswesen", "Methodik: Datenerfassungs- und Analysetechniken", "Fazit: Auswirkungen der Ergebnisse auf zukünftige Forschung"

8. **Authors**:
    - List the document's authors.
    - Examples: "Jane Smith", "John Doe", "Alice Johnson", "Hans Müller", "Peter Schmid", "Anna Meier"

9. **Source**:
    - Specify the source or location where the document can be found.
    - Examples: "https://example.com/research-paper", "Library of Congress", "Journal of Medical Research", "https://beispiel.de/forschungspapier", "Bibliothek des Kongresses", "Zeitschrift für medizinische Forschung"

10. **Language**:
    - Indicate the language(s) the document is written in.
    - Examples: "English", "German", "Spanish", "Englisch", "Deutsch", "Spanisch"

11. **Audience**:
    - Describe the intended audience for the document.
    - Examples: "Healthcare professionals", "University students", "Policy makers", "Gesundheitsfachkräfte", "Universitätsstudenten", "Politische Entscheidungsträger"

**Context**:
{context}

**Task**:
Extract and provide the following metadata from the document's content based on the above guidelines. Ensure that extracted information is in the original language of the document.

**Output Format**:
Return the metadata in the following structured format with no filter text or extra explanation, only give the extracted metadata:
```json
{{
"document_type": "Type of document",
"mentions": ["Main names mentioned"],
"keywords": ["Relevant keywords"],
"about": "Brief description",
"questions": ["Questions the document can answer"],
"entities": ["Main entities mentioned"],
"summaries": ["Summaries of key sections"],
"authors": ["List of authors"],
"source": "Source or location",
"language": "Document language",
"audience": "Intended audience"
}}
```
"""
naming_template = """
You are tasked with generating appropriate and consistent names for documents based on their content. Follow these detailed guidelines to ensure the names are informative, unique, and easy to manage:

1. **Think about your files**:
    - Identify the group of files your naming convention will cover.
    - Check for established file naming conventions in your discipline or group.

2. **Identify metadata**:
    - Include important information to easily locate a specific file.
    - Consider including a combination of the following:
        - Experiment conditions
        - Type of data
        - Researcher name/initials
        - Lab name/location
        - Project or experiment name or acronym
        - Experiment number or sample ID (use leading zeros for clarity)

3. **Abbreviate or encode metadata**:
    - Standardize categories and/or replace them with 2- or 3-letter codes.
    - Document any codes used.

4. **Think about how you will search for your files**:
    - Decide what metadata should appear at the beginning.
    - Use default ordering: alphabetically, numerically, or chronologically.

5. **Deliberately separate metadata elements**:
    - Avoid spaces or special characters in file names.
    - Use dashes (-), underscores (_), or capitalize the first letter of each word.

**Example Naming Convention**:
    - Format: [Type]_[Project]_[SampleID].[ext]
    - Example: FinancialReport_ProjectX_001.pdf

**Context**:
{context}

**Extracted Metadata**:
The extracted metadata contains important information such as keywords, entities, mentions, summaries, and other details that are useful for naming the document. This metadata helps in creating a name that is both descriptive and unique.

{metadata}

**Task**:
Generate a new, unique name for this document based on its content and the provided metadata. The new name should be formal, detailed, and distinctive to avoid confusion with other documents. Ensure the name is concise yet informative, highlighting significant details like names, firms, companies, etc. that capture the essence and purpose of the document. Be specific.

**Output Format**:
Provide only the new name in the following format with no filter or extra explanation, give only the new name: [Type]_[Name]_[YearRange]

**Question**: {question}
"""
# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Initialize libraries
pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_CMD

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

            if filename.endswith('.pdf'):
                documents.extend(self.load_pdf(file_path, metadata))
            elif filename.endswith('.txt'):
                documents.extend(self.load_text(file_path, metadata))
        return documents

    def load_pdf(self, file_path: str, metadata: dict) -> List[Document]:
        with fitz.open(file_path) as pdf_document:
            full_text = ''.join([pdf_document.load_page(page_number).get_text() for page_number in range(len(pdf_document))])

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2", model_kwargs={'device': 'cuda'})
        text_splitter = SemanticChunker(embedding, breakpoint_threshold_type="percentile")
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

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2", model_kwargs={'device': 'cuda'})
        text_splitter = SemanticChunker(embedding, breakpoint_threshold_type="percentile")
        chunks = text_splitter.create_documents([text_content])

        return [
            Document(
                page_content=chunk.page_content,  # Corrected from chunk.content to chunk.page_content
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
            metadata = {'source': self.file_path, 'processed_date': datetime.now().isoformat()}

        ordered_metadata = OrderedDict(metadata)

        # Set up the text chunker
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
        text_splitter = SemanticChunker(embedding, breakpoint_threshold_type="percentile")

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

class TTSStreamer:
    def __init__(self, model_path, config_path, vocab_path, speaker_wav="thunder"):
        self.model_path = model_path
        self.config_path = config_path
        self.vocab_path = vocab_path
        self.speaker_wav = f"audio_samples\\{speaker_wav}.wav"
        self.model = self.load_model()
        self.stop_flag = threading.Event()  # To control stopping
        self.playback_thread = None
        self.text_chunks = []  # Store text chunks

    def load_model(self):
        config = XttsConfig()
        config.load_json(self.config_path)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=self.model_path, eval=True, vocab_path=self.vocab_path)
        model.cuda()
        return model

    def unload_model(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        print("Model unloaded and GPU memory cleared successfully.")

    def estimate_times(self, text_chunk, avg_gen_time_per_char, avg_audio_time_per_char):
        gen_time = len(text_chunk) * avg_gen_time_per_char
        audio_duration = len(text_chunk) * avg_audio_time_per_char
        return gen_time, audio_duration

    def split_text_into_sentences(self, text):
        # Split text into sentences using regular expressions
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        
        final_sentences = []
        
        for sentence in sentences:
            if len(sentence) > 200:
                # Further split long sentences using commas and other punctuation marks
                parts = re.split(r'(?<=[,;:]) +', sentence)
                final_sentences.extend(parts)
            else:
                final_sentences.append(sentence)
        
        return final_sentences

    def generate_audio_chunk(self, chunk, chunk_index, audio_buffer, playback_event, avg_gen_time_per_char, avg_audio_time_per_char, total_gen_time, language, speed):
        if self.stop_flag.is_set():
            return
        est_gen_time, est_audio_duration = self.estimate_times(chunk, avg_gen_time_per_char, avg_audio_time_per_char)
        print(f"Chunk {chunk_index + 1} estimated generation time: {est_gen_time:.2f} seconds, estimated audio duration: {est_audio_duration:.2f} seconds")

        print(f"Generating audio for chunk {chunk_index + 1}...")
        start_gen_time = time.time()
        outputs = self.model.synthesize(
            text=chunk,
            config=self.model.config,
            speaker_wav=self.speaker_wav,
            gpt_cond_len=10,
            language=language,
            speed=speed
        )
        end_gen_time = time.time()
        generation_time = end_gen_time - start_gen_time
        total_gen_time[0] += generation_time
        print(f"Chunk {chunk_index + 1} generated in {generation_time:.2f} seconds (estimated: {est_gen_time:.2f} seconds)")

        wav_data = outputs['wav']
        temp_output_file = f'temp_output_{chunk_index}.wav'
        sf.write(temp_output_file, wav_data, 22050)
        line_audio = AudioSegment.from_wav(temp_output_file)

        actual_audio_duration = len(line_audio) / 1000.0
        print(f"Chunk {chunk_index + 1} actual audio duration: {actual_audio_duration:.2f} seconds (estimated: {est_audio_duration:.2f} seconds)")

        audio_buffer[chunk_index] = line_audio
        print(f"Chunk {chunk_index + 1} audio saved and buffered")

        playback_event.set()

    def stream_audio_with_buffering(self, text, language="en", speed=1.2, speaker=None, fireup_delay=1.0, avg_gen_time_per_char=0.08058659382140704, avg_audio_time_per_char=0.1064346054068992):
        self.stop_flag.clear()  # Clear the stop flag at the start
        if speaker:
            self.speaker_wav = f"audio_samples\\{speaker}.wav"

        print("Starting the audio streaming process...")
        start_time = time.time()

        self.text_chunks = self.split_text_into_sentences(text)  # Store text chunks
        audio_buffer = [None] * len(self.text_chunks)
        playback_events = [threading.Event() for _ in self.text_chunks]
        total_gen_time = [0]

        def start_playback_after_delay():
            print(f"Waiting {fireup_delay:.2f} seconds before starting playback...")
            time.sleep(fireup_delay)
            print("Fireup delay is over, starting playback...")
            for chunk_index in range(len(self.text_chunks)):
                if self.stop_flag.is_set():
                    break
                playback_events[chunk_index].wait()
                if audio_buffer[chunk_index] is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        temp_output_file = temp_file.name
                    self.play_audio_segment(audio_buffer[chunk_index], temp_output_file)
                    if os.path.exists(temp_output_file):
                        os.remove(temp_output_file)

        self.playback_thread = threading.Thread(target=start_playback_after_delay)
        self.playback_thread.start()

        for chunk_index, chunk in enumerate(self.text_chunks):
            if self.stop_flag.is_set():
                break
        
            print(f"Processing chunk {chunk_index + 1}/{len(self.text_chunks)}: '{chunk}'")
            self.generate_audio_chunk(chunk, chunk_index, audio_buffer, playback_events[chunk_index], avg_gen_time_per_char, avg_audio_time_per_char, total_gen_time, language, speed)

        self.playback_thread.join()
        print("Audio streaming process completed.")
        print(f"Total generation time: {total_gen_time[0]:.2f} seconds")

    def stop_streaming(self):
        """Stops the audio streaming process."""
        self.stop_flag.set()
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join()
        # Remove all temporary files
        for chunk_index in range(len(self.text_chunks)):
            temp_output_file = f'temp_output_{chunk_index}.wav'
            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)

    def play_audio_segment(self, audio_segment, temp_output_file):
        audio_segment.export(temp_output_file, format="wav")
        play(AudioSegment.from_wav(temp_output_file))

class DocumentAssistant:
    def __init__(self, model_name, temperature=0.9):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Initialize the default agent prompt
        self.agent_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ]
        )

        # Initialize the LLM chain with the agent prompt
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.agent_prompt,
            verbose=True,
            memory=self.memory
        )

    def reset_memory(self):
        """Resets the memory of the assistant."""
        self.memory.clear()

    def create_filter_criteria(self, filter_criteria):
        if filter_criteria:
            filter_key = filter_criteria.get("key")
            filter_value = filter_criteria.get("value")
            if filter_key and filter_value:
                return {"term": {f"metadata.{filter_key}.keyword": filter_value}}
        return None

    def document_retriever(self, user_query, top_k, score_threshold, filter_criteria):
        """
        Retrieves relevant documents based on the user's query and returns them with their scores and metadata.
        """
        # Perform the similarity search with scores
        results = vectorstore.similarity_search_with_score(
            query=user_query, 
            k=top_k,
        )
        
        # Filter documents based on the score threshold
        filtered_documents = [
            (doc, score) for doc, score in results if score >= score_threshold
        ]
        
        # Extract document names, content, metadata, and score for returning
        retrieved_documents = []
        for doc, score in filtered_documents:
            document_info = {
                "document_name": doc.metadata.get('given_document_name', 'Unnamed'),
                "document_content": doc.page_content,
                "metadata": doc.metadata,  # Include all metadata
                "score": score
            }
            print(f"Retrieved document with score {score}: {document_info['document_name']}")
            retrieved_documents.append(document_info)

        return retrieved_documents

    def formulate_final_prompt(self, user_query, context):
        """
        Formulates the final input for the LLM considering the retrieved documents.
        """
        combined_input = f"Here is the context from retrieved documents. Please use this information to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {user_query}"
        return combined_input

    def query_llm(self, user_query, top_k, score_threshold, filter_criteria):
        filter_criteria = self.create_filter_criteria(filter_criteria)

        # First, use the LLM chain to determine whether document retrieval is necessary
        response = self.chain.invoke({"input": user_query})
        print(f"Initial response: {response['text']}")
        if "retriever_tool:" in response['text'].lower():
            retrieval_instruction = response['text'].split("retriever_tool:")[1].strip()
            combined_query = f"{user_query} {retrieval_instruction}"
            retrieved_documents = self.document_retriever(combined_query, top_k, score_threshold, filter_criteria)
            
            context = "\n\n".join(
                [f"Document Name: {doc['document_name']}\nMetadata: {doc['metadata']}\nContent:\n{doc['document_content']}" 
                 for doc in retrieved_documents]
            )
            combined_input = self.formulate_final_prompt(user_query, context)
            
            # Use the LLM chain again to answer the user's query based on the retrieved documents
            final_response = self.chain.invoke({"input": combined_input})
            print(f"Final response: {final_response['text']}")  
            parsed_response = final_response['text']
        else:
            parsed_response = response['text']
            retrieved_documents = []

        return parsed_response, retrieved_documents

def convert_image_to_pdf(image_path):
    """Convert an image file to a PDF and return the new PDF path."""
    try:
        image = Image.open(image_path)
        pdf_path = image_path.lower().replace('.png', '.pdf').replace('.jpg', '.pdf').replace('.jpeg', '.pdf')
        rgb_image = image.convert('RGB')
        rgb_image.save(pdf_path, 'PDF', resolution=100.0)
        os.remove(image_path)
        print(f"Converted {image_path} to {pdf_path}")
        return pdf_path
    except Exception as e:
        logging.error(f"Error converting image to PDF: {e}")
        return None
    
def adaptive_image_processing(image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    
    # Apply a slight Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(clahe_image, (5, 5), 0)
    
    # Enhance contrast
    pil_image = Image.fromarray(blurred_image)
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_image = enhancer.enhance(1.5)
    
    return enhanced_image

def ocr_pdf(input_pdf_path):
    try:
        images = convert_from_path(input_pdf_path, poppler_path=POPPLER_PATH)
        pdf_writer = PdfWriter()
        full_text = ""
        
        # OCR the first page and detect its language
        first_page_text = pytesseract.image_to_string(images[0])
        try:
            detected_lang = detect(first_page_text)
            detected_lang_iso3 = pycountry.languages.get(alpha_2=detected_lang).alpha_3
            print(f"Detected language: {detected_lang_iso3}")
        except LangDetectException:
            logging.warning("Language detection failed, defaulting to English.")
            detected_lang_iso3 = 'eng'  # Default to English if language detection fails
        
        # OCR the entire PDF with the detected language
        for image in images:
            processed_image = adaptive_image_processing(image)
            page_text = pytesseract.image_to_string(processed_image, lang=detected_lang_iso3)
            full_text += page_text
            
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(processed_image, extension='pdf', lang=detected_lang_iso3)
            pdf_stream = io.BytesIO(pdf_bytes)
            pdf = PdfReader(pdf_stream)
            pdf_writer.add_page(pdf.pages[0])

        output_pdf_path = input_pdf_path  # Keep the file path consistent
        with open(output_pdf_path, "wb") as f_out:
            pdf_writer.write(f_out)

        print(f"OCR processed and replaced {output_pdf_path}")
        return output_pdf_path, full_text
    except Exception as e:
        logging.error(f"Error during OCR: {e}")
        return None, ""

def check_pdf_has_readable_text(file_path: str) -> bool:
    """Check if the PDF contains any readable text."""
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    return True
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
    return False

def check_pdf_metadata_keys(file_path: str, required_keys: list) -> bool:
    """Check if the PDF metadata contains all the required keys."""
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            metadata = reader.metadata
            return all(key in metadata for key in required_keys)
    except Exception as e:
        logging.error(f"Error reading metadata from {file_path}: {e}")
        return False

def check_for_metadata_json(file_path: str) -> bool:
    """Check if the corresponding JSON metadata file exists."""
    json_file_path = f"{os.path.splitext(file_path)[0]}.json"
    return os.path.exists(json_file_path)

def process_pdf_file(file_path: str) -> None:
    """Process a single PDF file and perform all checks."""
    print(f"Processing {file_path}...")

    has_text = check_pdf_has_readable_text(file_path)
    metadata_keys = ["/document_id", "/original_file_name", "/given_document_name"]

    # Perform OCR if no readable text is found
    if not has_text:
        print(f"The PDF {file_path} does not contain readable text. Performing OCR...")
        file_path, _ = ocr_pdf(file_path)  # Update file path if the file was replaced

    # Generate metadata if necessary
    if not check_for_metadata_json(file_path):
        print(f"The corresponding metadata JSON file does not exist for {file_path}. Performing metadata extraction...")
        document_name, metadata = generate_metadata_and_name(file_path)
        
    # Check for required metadata keys
    if check_pdf_metadata_keys(file_path, metadata_keys):
        print(f"The PDF {file_path} contains the required metadata keys.")
    else:
        print(f"The PDF {file_path} is missing some required metadata keys.")
    return document_name

def process_txt_file(file_path: str) -> None:
    """Process a TXT file by generating metadata if necessary."""
    print(f"Processing TXT file {file_path}...")

    if check_for_metadata_json(file_path):
        print(f"The corresponding metadata JSON file exists for {file_path}.")
    else:
        print(f"The corresponding metadata JSON file does not exist for {file_path}. Generating metadata...")
        document_name, metadata = generate_metadata_and_name(file_path)
    return document_name

def process_image_file(file_path: str) -> None:
    """Convert an image file to a PDF, perform OCR, and generate metadata."""
    print(f"Processing image file {file_path}...")

    pdf_path = convert_image_to_pdf(file_path)
    if pdf_path:
        print(f"Converted image to PDF: {pdf_path}. Performing OCR and generating metadata...")
        pdf_path, _ = ocr_pdf(pdf_path)  # Update file path after OCR
        document_name, metadata = generate_metadata_and_name(file_path)
    return document_name

def process_files_in_directory(directory_path: str, only_pdf: bool = False) -> None:
    def process():
        print(f"Processing files in directory {directory_path}...")
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".pdf"):
                    process_pdf_file(file_path)
                elif not only_pdf:
                    if file.endswith(".txt"):
                        process_txt_file(file_path)
                    elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        process_image_file(file_path)

    try:
        process()
    except Exception as e:
        print(f"Error encountered: {e}. Retrying...")
        try:
            process()
        except Exception as e:
            print(f"Retry failed: {e}")

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
def generate_metadata_and_name(file_path):
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

    # Initialize the LLM for metadata extraction
    metadata_llm = ChatOllama(model="llama3.1:8b", temperature=0.9)

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

    # Initialize the LLM for naming
    naming_llm = ChatOllama(model="llama3.1:8b", temperature=0.9, num_predict=30)

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
    document_name = naming_content.split('\n')[0].strip()

    # Replace spaces in document name with underscores
    document_name = document_name.replace(" ", "_")
    # Determine the file extension

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
       
    if file_extension == ".pdf":
        # Update PDF metadata with relevant keys (only for PDFs)
        pdf_metadata = {
            "/document_id": document_id,
            "/original_file_name": original_file_name,
            "/given_document_name": document_name
        }
        update_pdfmetadata(file_path, pdf_metadata)
        # Rename the original PDF file to the new document name
        new_file_path = os.path.join(file_directory, f"{document_name}.pdf")
    elif file_extension == ".txt":
        # Rename the original TXT file to the new document name
        new_file_path = os.path.join(file_directory, f"{document_name}.txt")
    
    os.rename(file_path, new_file_path)

    
    return document_name, formatted_metadata

# Function to save parameters to a file
def save_params(params):
    with open(PARAM_FILE, "w") as f:
        json.dump(params, f)

def load_params():
    # Define default parameters
    default_params = {
        "collection_name": "docpoi",
        "score_threshold": 0.7,
        "top_k": 5,
        "voiceover_speed": 1.3,
        "fireup_speed": 5.0,
        "language": "en",
        "speaker": "thunder",
        "use_voiceover": False  # Default to unchecked
    }
    
    params_file = PARAM_FILE
    
    # Attempt to load parameters from the file
    try:
        with open(params_file, "r") as file:
            params = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        params = {}
    
    # Update missing parameters with default values
    for key, value in default_params.items():
        if key not in params:
            params[key] = value
    
    return params

def initialize_vectorstore(collection_name="docpoi"):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2", model_kwargs={'device': 'cuda'})

    vectorstore = ElasticsearchStore(
        es_url="http://localhost:9200", index_name=collection_name, embedding=embedding, strategy=ElasticsearchStore.ExactRetrievalStrategy() 
    )

    namespace = f"elasticsearch/{collection_name}"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )

    record_manager.create_schema()
    
    return vectorstore, record_manager
# Function to reload the vectorstore
def reload_vectorstore():
    global vectorstore, record_manager
    vectorstore, record_manager = initialize_vectorstore(params["collection_name"])
    return "Vectorstore reloaded with collection: " + params["collection_name"]

def add_to_vectorstore():
    loader = DocPOIDirectoryLoader(directory_path=DIRECTORY_PATH)
    documents = loader.load()
    index(
        documents,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="document_id",
    )

def reset_vectorstore():
    index(
        [],  # Empty list to clear the vectorstore
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
    )
    return "Vectorstore has been reset."

def upload_file(file):
    params = load_params()
    UPLOAD_FOLDER = params.get("directory", "ocr/data")
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    destination = shutil.copy(file, UPLOAD_FOLDER)
    
    if destination.endswith(".pdf"):
        new_document_name = process_pdf_file(destination)
    elif destination.endswith(".txt"):
        new_document_name = process_txt_file(destination)
    elif destination.lower().endswith(('.png', '.jpg', '.jpeg')):
        new_document_name = process_image_file(destination)

    file_directory = os.path.dirname(destination)
    file_extension = os.path.splitext(destination)[1]
    new_destination = os.path.join(file_directory, f"{new_document_name}{file_extension}")

    if os.path.exists(destination):
        os.rename(destination, new_destination)
    else:
        possible_new_path = os.path.join(file_directory, f"{new_document_name}{file_extension}")
        if os.path.exists(possible_new_path):
            new_destination = possible_new_path
        else:
            return "File not found during renaming."

    loader = DocPOI(file_path=new_destination)
    documents = loader.load()

    index(
        documents,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="document_id",
    )

    return "File Uploaded and Processed!!!"

def process_files():
    # Load parameters
    params = load_params()
    
    # Check if mode key is set to default or only_pdf
    mode = params.get("mode", "default")
    only_pdf = mode == "only_pdf"
    
    # Process files in directory with the only_pdf parameter
    process_files_in_directory(DIRECTORY_PATH, only_pdf)
    
    add_to_vectorstore()
    return "Files are being processed..."

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def on_param_change(top_k, score_threshold, collection_name):
    params["top_k"] = top_k
    params["score_threshold"] = score_threshold
    params["collection_name"] = collection_name
    save_params(params)
    return gr.update()

def on_advanced_param_change(voiceover_speed, fireup_speed, language, speaker, use_voiceover):
    params["voiceover_speed"] = voiceover_speed
    params["fireup_speed"] = fireup_speed
    params["language"] = language
    params["speaker"] = speaker
    params["use_voiceover"] = use_voiceover
    save_params(params)
    return gr.update()

def list_microphones():
    """Lists available audio input devices (microphones)."""
    devices = sd.query_devices()
    input_devices = [device for device in devices if device['max_input_channels'] > 0]
    print("Available Microphones:")
    for i, device in enumerate(input_devices):
        print(f"{i}: {device['name']}")

def start_recording():
    """Starts recording audio."""
    global is_recording, recording, stream
    if not is_recording:
        print("Recording started...")
        play(AudioSegment.from_file("audio_samples\start_sound.mp3"))  # Play start sound
        recording = []
        is_recording = True
        stream = sd.InputStream(callback=callback, samplerate=44100, channels=1)
        stream.start()

def stop_recording_and_save():
    """Stops recording and saves the audio to a file."""
    global is_recording, recording, stream
    if is_recording:
        print("Recording stopped. Saving to file...")
        stream.stop()
        stream.close()
        play(AudioSegment.from_file("audio_samples\stop_sound.mp3"))  # Play stop sound
        # Convert the list of recordings to a NumPy array
        recording = np.concatenate(recording, axis=0)
        # Save to a WAV file using scipy.io.wavfile.write
        write(filename, 44100, recording)
        print(f"Recording saved to {filename}")
        recording = None  # Reset the recording
        is_recording = False

def callback(indata, frames, time, status):
    """This function is called for each audio block."""
    global recording
    if recording is not None:
        recording.append(indata.copy())

def on_key_press(key, history):
    """Triggered when the key is pressed."""
    global is_recording
    if is_recording:
        stop_recording_and_save()
        transcribed_text = transcribe_audio_to_text(filename)  # Transcribe the audio
        print("Transcribed Text:", transcribed_text)
        # Simulate the transcription as user input and pass it to the chatbot
        history.append([transcribed_text, ""])  # Add the transcribed text as user input
        for response in bot_response(history):
            pass  # Generate bot response based on transcribed text
    else:
        start_recording()
def transcribe_audio_to_text(filename):
    """Transcribes the recorded audio to text using Whisper."""
    whisper_model = whisper.load_model("small")
    result = whisper_model.transcribe(filename)
    return result["text"]

def setup_keyboard_shortcuts(history):
    """Sets up keyboard shortcuts for Ctrl + Alt + F13, F14, and F15."""
    def on_press(key):
        try:
            if key == keyboard.Key.f13 and {keyboard.Key.ctrl_l, keyboard.Key.alt_l}.issubset(pressed_keys):
                on_key_press(key, history)
            elif key == keyboard.Key.f14 and {keyboard.Key.ctrl_l, keyboard.Key.alt_l}.issubset(pressed_keys):
                on_key_press(key, history)
            elif key == keyboard.Key.f15 and {keyboard.Key.ctrl_l, keyboard.Key.alt_l}.issubset(pressed_keys):
                on_key_press(key, history)
        except AttributeError:
            pass

    pressed_keys = set()

    def on_press_wrapper(key):
        if key in {keyboard.Key.ctrl_l, keyboard.Key.alt_l, keyboard.Key.f13, keyboard.Key.f14, keyboard.Key.f15}:
            pressed_keys.add(key)
            on_press(key)

    def on_release_wrapper(key):
        if key in {keyboard.Key.ctrl_l, keyboard.Key.alt_l, keyboard.Key.f13, keyboard.Key.f14, keyboard.Key.f15}:
            pressed_keys.discard(key)

    listener = keyboard.Listener(on_press=on_press_wrapper, on_release=on_release_wrapper)
    listener.start()

# Function to get the full path with extension
def get_full_path(doc_name):
    base_path = f"{DIRECTORY_PATH}/"
    pdf_path = os.path.join(base_path, f"{doc_name}.pdf")
    txt_path = os.path.join(base_path, f"{doc_name}.txt")
    
    if os.path.exists(pdf_path):
        return pdf_path
    elif os.path.exists(txt_path):
        return txt_path
    else:
        raise FileNotFoundError(f"Neither {pdf_path} nor {txt_path} exists.")

def bot_response(history):

    # Get the last message from the user
    user_message = history[-1][0]
    
    # Load parameters
    params = load_params()

    # Extract the key and value for filtering
    filter_key = key_box.value
    filter_value = value_box.value
    filter_criteria = {filter_key: filter_value} if filter_key and filter_value else None

    # Simulate a query to the LLM
    llm_response, retrieved_documents = assistant.query_llm(
        user_query=user_message,
        top_k=params["top_k"],
        score_threshold=params["score_threshold"],
        filter_criteria=filter_criteria
    )

    # Prepare the full response text
    full_response = llm_response

    # Make the retrieved documents available for download
    downloadable_files = [get_full_path(doc['document_name']) for doc in retrieved_documents]
    
    # If use_voiceover is True, run the TTS streamer in a separate thread
    if params["use_voiceover"]:
        tts_thread = threading.Thread(
            target=tts_streamer.stream_audio_with_buffering,
            args=(full_response,),
            kwargs={
                "language": params["language"],
                "speed": params["voiceover_speed"],
                "speaker": params["speaker"],
                "fireup_delay": params["fireup_speed"]
            }
        )
        tts_thread.start()

    # Initialize the bot's response in the history
    history[-1][1] = ""

    # Stream the response character by character
    for character in full_response:
        if tts_streamer.stop_flag.is_set():  # Check if stop was requested
            break
        history[-1][1] += character
        time.sleep(0.01)  # Adjust the speed of streaming if needed
        yield history, None

    # After streaming the text, yield the downloadable files
    yield history, gr.File(value=downloadable_files)

    # Wait for the TTS thread to complete if it was started
    if params["use_voiceover"]:
        tts_thread.join()

def stop_all_streaming():
    """Stops all ongoing text and voiceover streaming."""
    tts_streamer.stop_streaming()

def reset_conversation():
    """Resets the conversation by clearing the memory and chat history."""
    assistant.reset_memory()  # Reset the assistant's memory
    return [], []  # Return empty lists for the chatbot and its state

def check_setup():    
    # Check if the params file exists
    if not os.path.exists(PARAM_FILE):
        return False
    
    # Load the parameters from the file
    with open(PARAM_FILE, "r") as file:
        params = json.load(file)
    
    # Check if the required parameters are set
    if "use_voiceover" in params and "directory" in params:
        return True
    
    return False

user_responses = {}
params = load_params()

# Function to save parameters to a JSON file
def save_params(params):
    with open(PARAM_FILE, "w") as f:
        json.dump(params, f)

# Function to list available speakers (dummy list for now)
def list_speakers():
    return [
        {"name": "Thunder"},
        {"name": "Serenity"},
        {"name": "Blaze"},
    ]

# Function to handle the welcome message and ask for the user's name
def handle_welcome(history):
    bot_message = "Welcome to the setup process. What is your name? (Type 'skip' to use 'User')"
    history[-1][1] = bot_message
    return history

# Function to handle the user's name input
def handle_ask_name(history):
    user_message = history[-1][0].strip()
    if not user_message or user_message.lower() == "skip":
        user_message = "User"
    user_responses['name'] = user_message
    save_params({"name": user_responses['name']})
    bot_message = f"""Great to meet you, {user_responses['name']}! I'd love to hear your voice, too. Would you like to enable my speech function? (yes/no)\n
Note: This feature is recommended for systems with at least 10GB of GPU VRAM for optimal performance. You can disable it later if needed."""
    history[-1][1] = bot_message
    return history

# Function to handle voice recognition choice
def handle_voice_recognition_choice(history):
    user_message = history[-1][0].strip().lower()
    if user_message == "no" or user_message == "skip":
        user_responses['use_voiceover'] = False
        return handle_general_setup(history)
    elif user_message == "yes":
        user_responses['use_voiceover'] = True
        save_params({"name": user_responses['name'], "use_voiceover": True})
        return handle_microphone_selection(history)
    else:
        bot_message = "Invalid response. Please type 'yes' or 'no'."
        history[-1][1] = bot_message
        return history

# Function to handle microphone selection
def handle_microphone_selection(history):
    input_devices = list_microphones()
    bot_message = "Please choose your microphone from the list by entering the corresponding number:\n"
    for i, device in enumerate(input_devices):
        bot_message += f"{i + 1}. {device['name']}\n"
    history[-1][1] = bot_message
    return history

def handle_microphone_response(history):
    user_message = history[-1][0].strip()
    input_devices = list_microphones()
    if user_message.isdigit() and 1 <= int(user_message) <= len(input_devices):
        selected_device = input_devices[int(user_message) - 1]
        user_responses['microphone'] = selected_device['name']
        save_params(user_responses)
        return handle_speaker_selection(history)
    else:
        bot_message = "Invalid selection. Please choose a microphone by number."
        history[-1][1] = bot_message
        return history

# Function to handle speaker selection
def handle_speaker_selection(history):
    speakers = list_speakers()
    bot_message = "I have a range of voices available. Please select your preferred voice by number. You can listen to samples if you'd like:\n"
    for i, speaker in enumerate(speakers):
        bot_message += f"{i + 1}. {speaker['name']}\n"
    history[-1][1] = bot_message
    
    return history

def handle_speaker_response(history):
    user_message = history[-1][0].strip()
    speakers = list_speakers()
    if user_message.isdigit() and 1 <= int(user_message) <= len(speakers):
        selected_speaker = speakers[int(user_message) - 1]
        user_responses['speaker'] = selected_speaker['name']
        save_params(user_responses)
        return handle_key_combination_selection(history)
    else:
        bot_message = "Invalid selection. Please choose a speaker by number."
        history[-1][1] = bot_message
        return history

# Function to handle key combination selection
def handle_key_combination_selection(history):
    bot_message = """Let's set up voice recognition. Please enter the key combination you'd like to use to activate it.
The default is Ctrl+Alt+F13 for custom keyboards. 
Type 'skip' or press Enter to keep the default. Use standard key names like Ctrl, Alt, Win, Tab, Shift, etc."""
    history[-1][1] = bot_message
    return history

def handle_key_combination_response(history):
    user_message = history[-1][0].strip()
    if not user_message or user_message.lower() == "skip":
        user_message = "Ctrl+Alt+F13"
    user_responses['key_combination'] = user_message
    save_params(user_responses)
    return handle_general_setup(history)

current_state = None

# Function to handle the general setup (directory selection and language)
def handle_general_setup(history):
    global current_state
    bot_message = "Now, let's select the main directory where your files are stored. The file explorer will open shortly."
    history[-1][1] = bot_message

    # Display the bot message before opening the file explorer
    history.append([None, bot_message])

    # Open the file explorer to select a directory
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    selected_directory = filedialog.askdirectory()  # Open the file explorer for directory selection
    root.destroy()  # Destroy the root window after selection
    
    if selected_directory:
        user_responses['directory'] = selected_directory  # Save the full path of the selected directory
    else:
        user_responses['directory'] = "Default Directory"
    
    save_params(user_responses)
    bot_message = f"""Directory chosen: {user_responses['directory']}.
Important: All files in this directory will be processed, including images and documents. Files will be converted, renamed, or removed. If you have files with extensions like (.pdf, .png, .jpeg, .jpg, .txt) that aren’t documents (such as pictures), consider isolating those or choose to limit processing to PDFs only.
Type 'only_pdf' to limit to PDF files, type 'reselect' to choose a different directory, or press Enter to proceed."""
    history.append([None, bot_message])
    
    current_state = "waiting_for_directory_response"  # Set the state for the next expected input
    return history

# Function to handle the directory response
def handle_directory_response(history):
    global current_state
    user_message = history[-1][0].strip().lower()
    
    if user_message == "reselect":
        return handle_general_setup(history)  # Reopen the file explorer for directory selection
    elif not user_message or user_message == "skip":
        user_responses['mode'] = "default"

    elif user_message == "only_pdf":
        user_responses['mode'] = "only_pdf"
    else:
        user_responses['mode'] = "default"
    
    save_params(user_responses)
    current_state = "language_selection"  # Move to the next state
    return handle_language_selection(history)

def handle_language_selection(history):
    global current_state
    bot_message = "Lastly, please enter your primary language. Type 'skip' to default to English."
    history[-1][1] = bot_message
    current_state = "waiting_for_language_response"
    return history

# Function to restart the script
def restart_script():
    subprocess.Popen(["restart.bat"])
    sys.exit()

def handle_language_response(history):
    global current_state
    user_message = history[-1][0].strip().lower()
    
    if not user_message or user_message == "skip":
        user_message = "English"
    user_responses['main_language'] = user_message
    save_params(user_responses)
    bot_message = "Language preference saved. The setup is complete. refresh the page after some time to start using the assistant."
    history[-1][1] = bot_message
    current_state = None  # Reset the state as the process is complete

    restart_script()
    
    return history

# Updated bot_response function
def setup_bot_response(history):
    global current_state
    step = len(history)
    
    if step == 1:
        return handle_welcome(history)
    elif step == 2:
        return handle_ask_name(history)
    elif step == 3:
        return handle_voice_recognition_choice(history)
    elif step == 4 and user_responses.get('use_voiceover'):
        return handle_microphone_response(history)
    elif step == 5 and user_responses.get('use_voiceover'):
        return handle_speaker_response(history)
    elif step == 6 and user_responses.get('use_voiceover'):
        return handle_key_combination_response(history)
    elif step == 4 and not user_responses.get('use_voiceover'):
        return handle_general_setup(history)
    elif current_state == "waiting_for_directory_response":
        return handle_directory_response(history)
    elif current_state == "language_selection":
        return handle_language_selection(history)
    elif current_state == "waiting_for_language_response":
        return handle_language_response(history)
    else:
        bot_message = "I'm not sure how to respond to that. Could you please provide more details?"
        history[-1][1] = bot_message
        return history

# Function to add a message to the history and reset the input box
def add_message_setup(history, message):
    if message is not None:
        history.append([message, None])
    return history, ""

history = []

# Gradio app for setup interface
with gr.Blocks(theme=gr.themes.Soft(), css="footer{display:none !important} #chatbot { height: 100%; flex-grow: 1;  }") as setup_demo:
    # Display audio samples when speaker selection is reached
    def display_audio_samples():
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    # Hide audio samples after speaker selection
    def hide_audio_samples():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(elem_id="chatbot")
            chat_input = gr.Textbox(interactive=True, placeholder="Enter message...", show_label=False, autoscroll=True)
            chat_msg = chat_input.submit(add_message_setup, [chatbot, chat_input], [chatbot, chat_input])
            bot_msg = chat_msg.then(setup_bot_response, chatbot, chatbot, api_name="setup_bot_response")

    with gr.Row():
        # Audio components for speaker samples, initially hidden
        audio1 = gr.Audio("audio_samples\Thunder_sample.wav", autoplay=False, format="wav", visible=False, label="Thunder")
        audio2 = gr.Audio("audio_samples\Serenity_sample.wav", autoplay=False, format="wav", visible=False, label="Serenity")
        audio3 = gr.Audio("audio_samples\Blaze_sample.wav", autoplay=False, format="wav", visible=False, label="Blaze")

    # When the bot reaches the speaker selection step, show audio samples
    bot_msg.then(display_audio_samples, [], [audio1, audio2, audio3])

    # After selecting a speaker, hide the audio samples
    chat_msg.then(hide_audio_samples, [], [audio1, audio2, audio3])

    # Start the conversation with an initial message
    setup_demo.load(lambda: [[None, "Welcome to your personal assistant setup! Before we dive into our conversations, how would you like me to address you?"]], outputs=chatbot)
    # Call the restart function at the end of the script

# Gradio app starts here
with gr.Blocks(theme=gr.themes.Soft(), css="footer{display:none !important} #chatbot { height: 100%; flex-grow: 1;  }") as demo:
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot([], elem_id="chatbot")
            with gr.Row():
                chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False, autoscroll=True, scale=6)
                stop_button = gr.Button("Stop", size="sm", scale=1)
                reset_button = gr.Button("Reset Conversation", size="sm", scale=1)
            chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
            bot_msg = chat_msg.then(bot_response, chatbot, [chatbot, gr.File(label="Related Documents")], api_name="bot_response")
            bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
            chatbot.like(print_like_dislike, None, None)
            stop_button.click(stop_all_streaming)
            reset_button.click(reset_conversation, [], [chatbot, chatbot])
        
        with gr.Column():
            upload_button = gr.UploadButton("Click to Upload a File")    
            upload_button.upload(upload_file, upload_button)
            process_button = gr.Button("Process Files")
            process_button.click(process_files, [], [])
            gr.Markdown("### Filters")
            with gr.Row():
                key_box = gr.Dropdown(choices=["source", "mentions", "date"], label="Key")
                value_box = gr.Dropdown(choices=["Name1", "Name2", "Name3"], label="Value")
            gr.Markdown("### Parameters")
            score_threshold_slider = gr.Slider(0.01, 0.99, label="Score Threshold", value=params["score_threshold"], interactive=True)
            top_k_input = gr.Number(label="Top K", value=params["top_k"], interactive=True)
            with gr.Row():
                collection_name_input = gr.Textbox(placeholder="Collection Name", label="Collection Name", value=params["collection_name"], interactive=True)
                reload_button = gr.Button("Reload Vectorstore", scale=0)
                reload_button.click(reload_vectorstore, [], None)
            score_threshold_slider.change(on_param_change, [top_k_input, score_threshold_slider, collection_name_input])
            top_k_input.change(on_param_change, [top_k_input, score_threshold_slider, collection_name_input])
            collection_name_input.change(on_param_change, [top_k_input, score_threshold_slider, collection_name_input])

            with gr.Accordion("Advanced Settings", open=False):
                use_voiceover_checkbox = gr.Checkbox(label="Use Voice Over", value=params["use_voiceover"], interactive=True)
                voiceover_speed_slider = gr.Slider(0.5, 1.5, value=params["voiceover_speed"], step=0.1, interactive=True, label="Voiceover Speed")
                fireup_speed_box = gr.Number(label="Fireup Speed (seconds)", value=params["fireup_speed"], interactive=True)
                language_input = gr.Textbox(placeholder="Language", label="Language", value=params["language"], interactive=True)
                speaker_dropdown = gr.Dropdown(["Thunder", "Serenity", "Blaze"], label="Speaker", value=params["speaker"], interactive=True)
            voiceover_speed_slider.change(on_advanced_param_change, [voiceover_speed_slider, fireup_speed_box, language_input, speaker_dropdown, use_voiceover_checkbox])
            fireup_speed_box.change(on_advanced_param_change, [voiceover_speed_slider, fireup_speed_box, language_input, speaker_dropdown, use_voiceover_checkbox])
            language_input.change(on_advanced_param_change, [voiceover_speed_slider, fireup_speed_box, language_input, speaker_dropdown, use_voiceover_checkbox])
            speaker_dropdown.change(on_advanced_param_change, [voiceover_speed_slider, fireup_speed_box, language_input, speaker_dropdown, use_voiceover_checkbox])
            use_voiceover_checkbox.change(on_advanced_param_change, [voiceover_speed_slider, fireup_speed_box, language_input, speaker_dropdown, use_voiceover_checkbox])

# Main execution
if check_setup():
    print("Setup is already complete. Launching the main interface...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    params = load_params()

    # Initialize vectorstore and record manager
    vectorstore, record_manager = initialize_vectorstore(params["collection_name"])

    # Get the value of the key 'directory' and set it as DIRECTORY_PATH
    DIRECTORY_PATH = params.get("directory", "data")
    tts_streamer = TTSStreamer(model_path="XTTS-v2", config_path="XTTS-v2\\config.json", vocab_path="XTTS-v2\\vocab.json")
    # Check if use_voiceover is True
    if params.get("use_voiceover", False):
        whisper_model = whisper.load_model("small")
        tts_streamer = TTSStreamer(model_path="XTTS-v2", config_path="XTTS-v2\\config.json", vocab_path="XTTS-v2\\vocab.json")
        list_microphones()

    assistant = DocumentAssistant(model_name="llama3.1:8b", temperature=0.9)
    history = []  # Initialize empty history
    setup_keyboard_shortcuts(history)

    # Define the system prompt for the assistant
    demo.launch(inbrowser=True, show_error=True)
else:
    print("Starting setup process...")
    setup_demo.launch(inbrowser=True, show_error=False)