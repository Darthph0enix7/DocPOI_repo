import gradio as gr
import os
import fitz  # PyMuPDF
import threading
import logging
import time
from components.settings import create_settings_interface
from components.param_manager import ParamManager
from components.tools_interface import create_tools_interface
from components.models import setup_models
from main import is_setup_needed

from langchain_chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config = {}



# Check if setup is needed and create setup.flag if necessary
if is_setup_needed():
    # Wait until setup.flag is created
    while not os.path.exists("setup.flag"):
        logger.info("Waiting for setup to complete...")
        time.sleep(5)  # Wait for 5 seconds before checking again
else:
    llm, embeddings = setup_models()
    

namespace = f"chroma/collection"
record_manager = SQLRecordManager(
    namespace, db_url="sqlite:///record_manager_cache.sql"
)
record_manager.create_schema()
vectorstore = Chroma(
    collection_name="collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.8, 'k': 4, 'filter': None},
    )

tool = create_retriever_tool(
    retriever,
    "Document_retriever",
    "Searches and returns relevant documents based on the query, use when you need to get information or context from users documents.",
)
tools = [tool]
memory = MemorySaver()
system_prompt = "you are a helpful assistant that can provide information and context from documents, you can also help with summarization, translation, and more."
agent_executor = create_react_agent(
    llm, tools, checkpointer=memory, state_modifier=system_prompt
)
config = {"configurable": {"thread_id": "default"}}

def send_message(message):
    logger.info(f"Received message: {message}")
    human_message = HumanMessage(content=message)
    
    response = agent_executor.invoke(
        {"messages": [human_message]},
        config=config,
    )
    output = get_most_recent_ai_message_content_and_tool_calls(response)
    logger.info(f"Received response: {output}")
    return output

def get_most_recent_ai_message_content_and_tool_calls(response):
    messages = response.get('messages', [])
    most_recent_content = None
    tool_calls = []

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage):
            if message.content:
                most_recent_content = message.content
            if 'tool_calls' in message.additional_kwargs:
                tool_calls.extend(message.additional_kwargs.get('tool_calls', []))

    return most_recent_content, tool_calls

# Initialize ParamManager
param_manager = ParamManager()
params = param_manager.get_all_params()

# Function to update parameter
def update_param(param_name, value):
    param_manager.set_param(param_name, value)

# HTML Template for embedding a PDF with page control, removing browser UI
pdf_viewer_template = """
<div style="width: 100%; height: 90vh; margin: 0; padding: 0;">
  <iframe id="pdf_viewer" src="{pdf_url}#toolbar=0&navpanes=0&scrollbar=0&page={page_number}" width="100%" height="100%" style="border: none; margin: 0; padding: 0;"></iframe>
</div>
"""

def highlight_text_in_pdf(pdf_path, page_number=None, search_text=None):
    # Make sure the PDF path is absolute and exists
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.abspath(pdf_path)
    if not os.path.exists(pdf_path):
        return "PDF not found"
    
    # Open the PDF and highlight text using PyMuPDF
    doc = fitz.open(pdf_path)
    if page_number is None or page_number < 1:
        page_number = 1
    page = doc.load_page(page_number - 1)  # PyMuPDF pages are zero-indexed
    
    if search_text:
        text_instances = page.search_for(search_text)
        
        # Remove existing highlights to ensure fresh highlighting
        for annot in page.annots():
            if annot.type[0] == 8:  # Check if it's a highlight annotation
                annot.delete()
        
        # Highlight all found instances
        for inst in text_instances:
            page.add_highlight_annot(inst)
    
    # Save to a temporary PDF
    highlighted_pdf_path = "temp.pdf"
    doc.save(highlighted_pdf_path)
    doc.close()

    # Serve the highlighted PDF using a dedicated URL
    pdf_url = f"http://127.0.0.1:7860/pdf?path={os.path.abspath(highlighted_pdf_path)}"
    
    # Embed the PDF URL with the specific page
    return pdf_viewer_template.format(pdf_url=pdf_url, page_number=page_number)

def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot_response(history):
    # Check if history is empty
    if not history:
        history.append(["Bot", "This is a placeholder response."])
        yield history
        return
    
    # Get the last message from the user
    user_message = history[-1][0] if history[-1][1] is None else history[-1][1]
    
    # Ensure the last message is not None
    if user_message is None:
        user_message = ""
    
    # Get the bot response using send_message
    bot_reply, tool_calls = send_message(user_message)
    
    # Add tool usage metadata if any tools were called
    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call['function']['name']
            tool_arguments = tool_call['function']['arguments']
            tool_metadata = f"🛠️ Used tool {tool_name} with arguments: {tool_arguments}"
            history.append(["bot", tool_metadata])
            yield history

    # Stream the response character by character
    history.append(["Bot", ""])
    for character in bot_reply:
        history[-1][1] += character
        time.sleep(0.01)  # Adjust the speed of streaming if needed
        yield history

    # Final yield to complete the response
    yield history

def print_like_dislike():
    print("Like/Dislike button clicked")

def reset_conversation():
    return [], []

# Function to toggle visibility of the PDF viewer column
def toggle_visibility(state):
    state = not state
    return state

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(text_size="sm"), css="footer{display:none !important} #chatbot { height: 100%; flex-grow: 1;  }") as main_interface_blocks:
    with gr.Tab("Chat Interface"):
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot([], elem_id="chatbot", height=470, label="DocPOI V2.0")
                with gr.Row():
                    chat_input = gr.MultimodalTextbox(label="DocPOI V2.0", interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False, autoscroll=True, scale=6)
                    stop_button = gr.Button("Stop", size="sm", scale=1, min_width=1)
                reset_button = gr.Button("Reset Conversation", size="sm", scale=1, min_width=1)
                chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
                bot_msg = chat_msg.then(bot_response, chatbot, [chatbot], api_name="bot_response")
                bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
                chatbot.like(print_like_dislike, None, None)
                #stop_button.click(stop_all_streaming)
                reset_button.click(reset_conversation, [], [chatbot, chatbot])
            
            # State variable to track visibility
            pdf_visible = gr.State(False)
            
            with gr.Column(scale=1, visible=False) as pdf_column:
                # Hardcoded parameters
                pdf_path = "C:\\Users\\kalin\\Downloads\\formblatt_03_Seda.pdf"
                page_number = 1
                search_text = "E"
                
                # HTML output to display the PDF
                pdf_display = gr.HTML()
                
                # Function to highlight text in PDF with hardcoded parameters
                def highlight_text_in_pdf_hardcoded():
                    return highlight_text_in_pdf(pdf_path, page_number, search_text)
                
                # Display the PDF with highlighted text
                pdf_display.value = highlight_text_in_pdf_hardcoded()
            
            # Use the stop button to toggle visibility
            stop_button.click(toggle_visibility, pdf_visible, pdf_visible).then(
                lambda visible: gr.update(visible=visible), pdf_visible, pdf_column
            )
    
    # Create settings interface
    create_settings_interface(params)

    # Create tools interface tab
    if params.get("agent_type") in ["ReAct agent", "OpenAI API"]:
        with gr.Tab("Tools Interface") as tools_tab:
            create_tools_interface()
