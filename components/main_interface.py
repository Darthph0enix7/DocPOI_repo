import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)
# Create Gradio Interface for Main Screen
with gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
) as main_interface_blocks:
    gr.Markdown("## Main Screen")
    gr.Label("Welcome to the main screen!")
    gr.Label("You have successfully completed the setup process.")