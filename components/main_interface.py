import gradio as gr
import os
import fitz  # PyMuPDF

# HTML Template for embedding a PDF with page control, removing browser UI
pdf_viewer_template = """
<div style="width: 100%; height: 100vh; margin: 0; padding: 0;">
  <iframe id="pdf_viewer" src="{pdf_url}#toolbar=0&navpanes=0&scrollbar=0&page={page_number}" width="100%" height="100%" style="border: none; margin: 0; padding: 0;"></iframe>
</div>
"""

def highlight_text_in_pdf(pdf_path, page_number, search_text):
    # Make sure the PDF path is absolute and exists
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.abspath(pdf_path)
    if not os.path.exists(pdf_path):
        return "PDF not found"
    
    # Open the PDF and highlight text using PyMuPDF
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)  # PyMuPDF pages are zero-indexed
    text_instances = page.search_for(search_text)
    
    # Remove existing highlights to ensure fresh highlighting
    for annot in page.annots():
        if annot.type[0] == 8:  # Check if it's a highlight annotation
            annot.delete()
    
    # Highlight all found instances
    for inst in text_instances:
        page.add_highlight_annot(inst)
    
    # Save to a new PDF
    highlighted_pdf_path = "highlighted_output.pdf"
    doc.save(highlighted_pdf_path)
    doc.close()

    # Serve the highlighted PDF using a dedicated URL
    pdf_url = f"http://127.0.0.1:7860/pdf?path={os.path.abspath(highlighted_pdf_path)}"
    
    # Embed the PDF URL with the specific page
    return pdf_viewer_template.format(pdf_url=pdf_url, page_number=page_number)

# Gradio interface
with gr.Blocks() as main_interface_blocks:
    with gr.Row():
        with gr.Column(scale=1):
            # Input for the PDF file path, page number, and text to highlight
            pdf_path_input = gr.Textbox(label="PDF File Path", placeholder="Enter the full path to the PDF file here...")
            page_number_input = gr.Number(label="Page Number", value=1)
            search_text_input = gr.Textbox(label="Text to Highlight", placeholder="Enter the text to highlight...")

            # Button to trigger the PDF display with highlights
            submit_button = gr.Button("Highlight and Display PDF")
        
        with gr.Column(scale=3):
            # HTML output to display the PDF
            pdf_display = gr.HTML()
    
    # Set up the interaction
    submit_button.click(highlight_text_in_pdf, inputs=[pdf_path_input, page_number_input, search_text_input], outputs=pdf_display)
