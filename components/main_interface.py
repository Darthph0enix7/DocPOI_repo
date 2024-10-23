import gradio as gr
import os
import fitz  # PyMuPDF
import threading
import time
from components.settings import create_settings_interface
from components.param_manager import ParamManager

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
    user_message = history[-1][1]
    
    # Placeholder response
    placeholder_response = "This is a placeholder response."

    # Ensure the last message is not None
    if history[-1][1] is None:
        history[-1][1] = ""

    # Stream the response character by character
    for character in placeholder_response:
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
    with gr.Tab("chat Interface"):
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
                
    create_settings_interface()   

        
    # Set up the interaction
    #submit_button.click(highlight_text_in_pdf, inputs=[pdf_path_input, page_number_input, search_text_input], outputs=pdf_display)