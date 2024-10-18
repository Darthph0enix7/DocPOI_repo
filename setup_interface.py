import gradio as gr

# Define Setup Screen Function
def setup_user(name):
    # After the setup, create the setup flag file and return a completion message
    with open("setup.flag", "w") as f:
        f.write("setup complete")
    setup_complete_message = f"Setup complete! Welcome, {name}. Please refresh the page to proceed to the main screen."
    return setup_complete_message

# Create Gradio Interface for Setup Screen
with gr.Blocks() as setup_interface:
    gr.Markdown("## Setup Screen")
    name_input = gr.Textbox(label="Enter your name")
    setup_output = gr.HTML(label="Output")
    submit_btn = gr.Button("Complete Setup")
    submit_btn.click(fn=setup_user, inputs=name_input, outputs=setup_output)
